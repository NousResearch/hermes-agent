"""Gateway-identity matching must recognise NAMED-profile invocations.

Regression coverage for the "dashboard shows named-profile gateway as offline"
bug. A named-profile gateway runs as::

    python -m hermes_cli.main --profile apollo gateway run --replace

so the ``--profile <name>`` token sits between the entrypoint and the
``gateway`` subcommand. The old contiguous-substring match on
``"hermes_cli.main gateway"`` only recognised the DEFAULT profile, so every
named profile (zeus/apollo/athena/librarian/...) was wrongly classified as
"not a gateway" by both:

  * ``_looks_like_gateway_process`` (live cmdline oracle), and
  * ``_record_looks_like_gateway`` (PID-file / runtime-status argv oracle).

That made ``get_running_pid`` / ``get_runtime_status_running_pid`` fall through
to "stopped" for named profiles and tricked ``acquire_scoped_lock`` into
evicting a live gateway's lock as stale on platforms with no ``/proc``
start-time signal (macOS / Windows).

These tests assert the INVARIANT (named profiles match exactly like the default
profile; sibling subcommands like ``dashboard`` do NOT match), not a frozen
pattern list, so they survive a future refactor of the matcher internals.
"""

import pytest

from gateway import status as status_mod
from gateway.status import (
    _cmdline_looks_like_gateway,
    _record_looks_like_gateway,
)


# A representative entrypoint path; the matcher must be path-prefix agnostic.
_PY = "/opt/hermes/venv/bin/python"
_MOD = "hermes_cli.main"
_SCRIPT = "/opt/hermes/hermes_cli/main.py"

_NAMED_PROFILES = ["zeus", "apollo", "athena", "librarian"]


# --------------------------------------------------------------------------- #
# _cmdline_looks_like_gateway — the shared oracle
# --------------------------------------------------------------------------- #


def test_default_profile_module_invocation_matches():
    assert _cmdline_looks_like_gateway(
        f"{_PY} -m {_MOD} gateway run --replace"
    )


def test_default_profile_script_invocation_matches():
    assert _cmdline_looks_like_gateway(
        f"{_PY} {_SCRIPT} gateway run --replace"
    )


@pytest.mark.parametrize("profile", _NAMED_PROFILES)
def test_named_profile_long_flag_module_invocation_matches(profile):
    assert _cmdline_looks_like_gateway(
        f"{_PY} -m {_MOD} --profile {profile} gateway run --replace"
    )


@pytest.mark.parametrize("profile", _NAMED_PROFILES)
def test_named_profile_short_flag_module_invocation_matches(profile):
    assert _cmdline_looks_like_gateway(
        f"{_PY} -m {_MOD} -p {profile} gateway run --replace"
    )


@pytest.mark.parametrize("profile", _NAMED_PROFILES)
def test_named_profile_script_invocation_matches(profile):
    assert _cmdline_looks_like_gateway(
        f"{_PY} {_SCRIPT} --profile {profile} gateway run --replace"
    )


def test_named_and_default_invocations_match_identically():
    """The whole bug: named profiles must match exactly like the default."""
    default = _cmdline_looks_like_gateway(f"{_PY} -m {_MOD} gateway run")
    named = _cmdline_looks_like_gateway(
        f"{_PY} -m {_MOD} --profile apollo gateway run"
    )
    assert default == named == True  # noqa: E712 — explicit invariant


def test_gateway_console_shim_and_runpy_match_standalone():
    assert _cmdline_looks_like_gateway("/usr/local/bin/hermes-gateway")
    assert _cmdline_looks_like_gateway(f"{_PY} /opt/hermes/gateway/run.py")


# --------------------------------------------------------------------------- #
# False-positive guard: sibling subcommands sharing the entrypoint must NOT
# be mistaken for a gateway (the reason we require the ``gateway`` token rather
# than a bare ``--profile`` substring).
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("profile", _NAMED_PROFILES)
def test_named_profile_dashboard_is_not_a_gateway(profile):
    assert not _cmdline_looks_like_gateway(
        f"{_PY} -m {_MOD} --profile {profile} dashboard --no-open --port 0"
    )


@pytest.mark.parametrize("profile", _NAMED_PROFILES)
def test_named_profile_chat_is_not_a_gateway(profile):
    assert not _cmdline_looks_like_gateway(
        f"{_PY} -m {_MOD} --profile {profile} chat"
    )


def test_empty_and_unrelated_cmdlines_do_not_match():
    assert not _cmdline_looks_like_gateway(None)
    assert not _cmdline_looks_like_gateway("")
    assert not _cmdline_looks_like_gateway(f"{_PY} -m some_other_thing")


def test_substring_gateway_in_path_without_subcommand_does_not_match():
    """A path component containing 'gateway' must not, by itself, qualify.

    Only the ``gateway`` *subcommand* token (or the unambiguous standalone
    signals) counts. A working directory like ``/srv/gateway-tools`` that
    happens to contain the substring should not flip an unrelated process.
    """
    assert not _cmdline_looks_like_gateway(
        f"{_PY} -m some_tool --cwd /srv/gateway-tools/run"
    )


# --------------------------------------------------------------------------- #
# _record_looks_like_gateway — the PID-file / runtime-status argv oracle.
# This is the path the dashboard relies on when /proc start-time is absent
# (macOS / Windows), where the live process is real but start_time is None.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("profile", _NAMED_PROFILES)
def test_record_oracle_matches_named_profile_argv(profile):
    record = {
        "kind": "hermes-gateway",
        "argv": [_SCRIPT, "--profile", profile, "gateway", "run", "--replace"],
        "start_time": None,
    }
    assert _record_looks_like_gateway(record)


def test_record_oracle_matches_default_profile_argv():
    record = {
        "kind": "hermes-gateway",
        "argv": [_SCRIPT, "gateway", "run", "--replace"],
        "start_time": None,
    }
    assert _record_looks_like_gateway(record)


def test_record_oracle_rejects_wrong_kind():
    record = {
        "kind": "not-a-gateway",
        "argv": [_SCRIPT, "--profile", "apollo", "gateway", "run"],
    }
    assert not _record_looks_like_gateway(record)


def test_record_oracle_rejects_dashboard_argv():
    record = {
        "kind": "hermes-gateway",  # even with the right kind, dashboard argv != gateway
        "argv": [_SCRIPT, "--profile", "apollo", "dashboard", "--no-open"],
    }
    assert not _record_looks_like_gateway(record)


def test_record_oracle_normalises_windows_backslashes():
    record = {
        "kind": "hermes-gateway",
        "argv": [
            "C:\\hermes\\hermes_cli\\main.py",
            "--profile",
            "apollo",
            "gateway",
            "run",
        ],
    }
    assert _record_looks_like_gateway(record)


# --------------------------------------------------------------------------- #
# End-to-end through get_runtime_status_running_pid: a live named-profile
# gateway with a null start_time (the macOS reality) must resolve to its PID.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("profile", _NAMED_PROFILES)
def test_runtime_status_pid_resolves_for_named_profile_on_macos(profile, monkeypatch):
    runtime = {
        "pid": 54321,
        "kind": "hermes-gateway",
        "argv": [_SCRIPT, "--profile", profile, "gateway", "run", "--replace"],
        "start_time": None,  # macOS has no /proc, so this is null in practice
        "gateway_state": "running",
    }
    monkeypatch.setattr(status_mod, "_pid_exists", lambda pid: pid == 54321)
    monkeypatch.setattr(status_mod, "_get_process_start_time", lambda pid: None)
    # Force the live cmdline oracle unavailable (e.g. process inspection blocked)
    # so resolution must succeed via the argv record oracle alone.
    monkeypatch.setattr(status_mod, "_read_process_cmdline", lambda pid: None)

    assert status_mod.get_runtime_status_running_pid(runtime) == 54321
