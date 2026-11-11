"""What the manager environment reader refuses, and how it says so: every refusal names the
variable and never its value, the reason code stays the stable contract, and both shapes of
the same block decode to the same mapping."""
import json
import re
import subprocess

import pytest

from hermes_cli.gateway_runtime_service_identity import (
    Unverified,
    manager_environment_pairs,
    manager_environment_pairs_json,
)


def _reason(value, reader=manager_environment_pairs):
    """The reason the reader gives for refusing *value*."""
    with pytest.raises(Unverified) as refused:
        reader(value)
    return refused.value.check


# ── a refusal names the variable and never carries its value ──

@pytest.mark.parametrize("dump, variable", [
    ("1AB=x", "1AB"),
    ("A-B=x", "A-B"),
    ("A.B=x", "A.B"),
    ("A B=x", "A B"),
    ("{A}=x", "{A}"),
])
def test_a_refusal_names_the_variable_it_refused(dump, variable):
    reason = _reason(dump)
    assert variable in reason
    assert "not a valid variable name" in reason


@pytest.mark.parametrize("dump, expected", [
    ("NOVALUE\n", "not an assignment"),
    ("Z=$'abc", "never closes"),
    ("Z=$'a'b'", "closes its ANSI-C quote"),
    ("Z=$'a\\qc'", "escape systemd does not write"),
    ("Z=$'a\\12'", "incomplete octal escape"),
])
def test_a_refusal_says_what_went_wrong(dump, expected):
    assert expected in _reason(dump)


@pytest.mark.parametrize("dump, reader, expected", [
    ('{"A":"b"="c"}', manager_environment_pairs_json, "JSON"),
    ('["A","b"]', manager_environment_pairs_json, "not an object"),
    ('{"A":1}', manager_environment_pairs_json, "non-string"),
    ('{"1AB":"x"}', manager_environment_pairs_json, "'1AB'"),
])
def test_a_json_refusal_says_what_went_wrong(dump, reader, expected):
    assert expected in _reason(dump, reader)


@pytest.mark.parametrize("dump, value", [
    # The name is refused, so the reason names it and drops the value.
    ("1AB=hunter2-not-a-real-value\n", "hunter2-not-a-real-value"),
    # The quote is refused, so the reason names the variable and drops the value.
    ("TOKEN=$'sk-live-1234567890\n", "sk-live-1234567890"),
    # The line is refused, so the reason drops the whole line.
    ("NOEQUALS\nSECRET=hunter2\n", "hunter2"),
])
def test_a_refusal_never_carries_the_value_of_the_variable(dump, value):
    assert value not in _reason(dump)


def test_a_json_refusal_never_carries_the_value():
    secret = "s3cr3t-token-value"
    assert secret not in _reason(f'{{"1AB":"{secret}"}}', manager_environment_pairs_json)


def test_a_valid_value_is_read():
    assert manager_environment_pairs("TOKEN=sk-live-1234567890\n") == {"TOKEN": "sk-live-1234567890"}


@pytest.mark.parametrize("dump", [
    "Z=$'a\\qc'",      # an escape systemd does not write: it doubles the backslash
    "Z=$'a\\12'",      # an incomplete octal escape: systemd always writes three digits
    "Z=$'abc",         # a quote that is never closed
    "Z=$'a\\'",        # an escaped quote with nothing after it
])
def test_an_escape_systemd_cannot_write_is_refused(dump):
    with pytest.raises(Unverified):
        manager_environment_pairs(dump)


# ── the reason code stays the stable contract ──

def test_the_reason_code_is_unchanged():
    with pytest.raises(Unverified) as refused:
        manager_environment_pairs("1AB=x")
    assert str(refused.value) == "service_identity_unverified"


# ── the reason reaches the user ──

def test_the_check_reaches_the_runtime_start_error(monkeypatch, tmp_path):
    """The reason has to survive the hop into the error Desktop prints."""
    import os
    import pwd
    from types import SimpleNamespace

    from hermes_cli import gateway_runtime_service as service

    # The unit runs as the user who owns it; the check on the running account compares that
    # account with the one asking, so an empty User= reads as root and never matches.
    user = pwd.getpwuid(os.getuid()).pw_name

    def run(argv, *_args, **_kwargs):
        if "show-environment" in argv:
            # A manager that answered with a name systemd would never print.
            stdout = "1AB=x\n" if "--output=json" not in argv else '{"1AB":"x"}'
            # The manager environment is read as bytes: a value can hold a raw CR.
            return SimpleNamespace(returncode=0, stdout=stdout.encode(), stderr=b"")
        if "show" in argv:
            if "--system" in argv:
                # No unit in the system scope: it must not take the finding away.
                return SimpleNamespace(returncode=1, stdout="", stderr="not found")
            # Every identity property verify_systemd requires, so the refusal that reaches the
            # user is the one about the manager environment and not a missing property.
            stdout = "\n".join([
                "LoadState=loaded", "ActiveState=inactive", f"User={user}", "DynamicUser=no",
                "EnvironmentFiles=", "PAMName=", "RootDirectory=", "RootImage=",
                "ExecStartPre=", "ExecCondition=", "PassEnvironment=", "UnsetEnvironment=",
                "Environment=", "DropInPaths=", "ExecStart=",
            ]) + "\n"
            return SimpleNamespace(returncode=0, stdout=stdout, stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setattr(service, "_run", run)
    # The unit is installed in the user scope only.
    monkeypatch.setattr(service, "_exists", lambda path: "systemd/system" not in str(path))
    # ``_s6_running`` is imported inside the function from the service manager.
    from hermes_cli import service_manager
    monkeypatch.setattr(service_manager, "_s6_running", lambda: False)

    with pytest.raises(service.RuntimeStartError) as refused:
        service._systemd(home, 0)
    assert refused.value.reason == "service_identity_unverified"
    assert refused.value.detail is not None
    assert "1AB" in refused.value.detail


def test_a_readable_manager_environment_is_not_refused(monkeypatch):
    """The happy path stays clean: a readable environment raises nothing."""
    from types import SimpleNamespace

    from hermes_cli import gateway_runtime_service as service

    def run(argv, *_args, **_kwargs):
        stdout = '{"HOME":"/home/u","LANG":"C"}' if "--output=json" in argv else "HOME=/home/u\nLANG=C\n"
        # The manager environment is read as bytes: a value can hold a raw CR.
        return SimpleNamespace(returncode=0, stdout=stdout.encode(), stderr=b"")

    monkeypatch.setattr(service, "_run", run)
    assert service._manager_environment(("systemctl", "--user"), 0) == {"HOME": "/home/u", "LANG": "C"}


# ── the two forms of the same block decode to the same mapping ──

def _text_form(mapping):
    """The block as systemd prints it in text, quoting the values that need it."""
    lines = []
    for key, val in mapping.items():
        if val and re.search(r"[\\'\"$*?\[\]()<>|&;!#\s]", val):
            escaped = val.replace("\\", "\\\\").replace("'", "\\'")
            lines.append(f"{key}=$'{escaped}'")
        else:
            lines.append(f"{key}={val}")
    return "\n".join(lines)


@pytest.mark.parametrize("mapping", [
    {"HOME": "/home/u"},
    {"LANG": "C"},
    {"MANPAGER": "sh -c 'col -bx | bat -l man -p'"},
    {"PATH": "/usr/bin:/bin"},
    {"EMPTY": ""},
])
def test_the_json_and_the_text_form_decode_to_the_same_mapping(mapping):
    """The reader is picked from the shape of the output, so both shapes have to
    give the same answer for the same block."""
    assert manager_environment_pairs(_text_form(mapping)) == mapping
    assert manager_environment_pairs_json(json.dumps(mapping)) == mapping


def test_the_two_readers_agree_on_the_manager_block_of_this_host():
    """The real block, read both ways."""
    def dump(*args):
        return subprocess.run(["systemctl", "--user", "show-environment", *args],
                              capture_output=True, timeout=30, check=False).stdout.decode(
            "utf-8", "surrogateescape")

    text, as_json = dump(), dump("--output=json")
    if not text.strip():
        pytest.skip("this host has no user systemd manager to ask")
    if not as_json.startswith("{"):
        pytest.skip("this systemd is older than v250 and prints no JSON form")
    assert manager_environment_pairs(text) == manager_environment_pairs_json(as_json)
