"""The one-time import runs only when every kept-local ``.env`` line keeps its value.

Before anything is sent or rewritten, the import computes each kept-local name's value now (the
local file, as every reader reads it) and afterwards (the stored entries, then the file the import
leaves, a kept line frozen when that round-trips). When any differs on any reader the import is
refused: nothing is sent, ``.env`` and ``auth.json`` stay byte for byte, the process serves the
home from them, and ``hermes doctor`` and the boot log name the lines (never a value). The next
boot checks again. Expected values come from running the local store on the same text.
"""
from __future__ import annotations

import json
import logging
import random

import pytest

import hermes_secret_routing as routing
from tests.plugins.secret_store.test_remote_load_at_start import _boot, _fresh_start, _readers

pytest_plugins = ["tests.plugins.secret_store.test_remote_secret_store"]

NAME = "GATEWAY_RELAY_IDP_CLIENT_SECRET"   # kept local by the remote store (its IdP credential)
REF = "TERMINAL_RV_REF"                    # kept local: process-global
_NAMES = ("RV_DIR", "RV_X", "RV_SHELL", NAME, REF)


@pytest.fixture(autouse=True)
def _plane_token(monkeypatch):
    """The plane credential comes from the stub directly: these files reassign the IdP secret name."""
    from plugins.secret_store.remote import client
    from tests.plugins.secret_store.stub_secrets_plane import TOKEN
    monkeypatch.setattr(client, "bearer", lambda home: TOKEN)


def _local_values(plane, monkeypatch, text, shell, names):
    """What the local store serves for *text*: the base every remote outcome must equal. The file
    the local load leaves (its sanitizer may rewrite it) is ``plane.local_bytes``."""
    _fresh(monkeypatch, shell)
    monkeypatch.setenv("HERMES_SECRET_STORE", "local")
    routing.reset_for_tests()
    (plane.home / ".env").write_text(text, encoding="utf-8", newline="")
    _boot(plane.home)
    values = {name: _readers(plane.home, name) for name in names}
    plane.local_bytes = (plane.home / ".env").read_bytes()
    monkeypatch.setenv("HERMES_SECRET_STORE", "remote")
    return values


def _fresh(monkeypatch, shell):
    _fresh_start(monkeypatch, {})
    for name in _NAMES:
        monkeypatch.delenv(name, raising=False)
    for name, value in shell.items():
        monkeypatch.setenv(name, value)


def _remote_boot(plane, monkeypatch, text, shell, names):
    """A remote first boot on *text*; then, as every later start, a fresh one on what it left."""
    _fresh(monkeypatch, shell)
    (plane.home / ".env").write_text(text, encoding="utf-8", newline="")
    _boot(plane.home)
    first = {name: _readers(plane.home, name) for name in names}
    _fresh(monkeypatch, shell)
    _boot(plane.home)
    later = {name: _readers(plane.home, name) for name in names}
    return first, later


SPANNING = {
    "escaped_lf": 'RV_DIR="first\\npart"\nGATEWAY_RELAY_IDP_CLIENT_SECRET=${RV_DIR}|\nRV_DIR=last\n',
    "escaped_cr": 'RV_DIR="first\\rpart"\nGATEWAY_RELAY_IDP_CLIENT_SECRET=${RV_DIR}|\nRV_DIR=last\n',
    "physical_lf": 'RV_DIR="first\npart"\nGATEWAY_RELAY_IDP_CLIENT_SECRET=${RV_DIR}|\nRV_DIR=last\n',
}


@pytest.mark.parametrize("case", SPANNING)
def test_an_import_that_would_change_a_kept_line_is_refused_and_writes_nothing(plane, monkeypatch, caplog, case):
    """The reviewer's file: the kept line's value has a line break (it cannot be frozen) and depends on
    a name the import would collapse to its last assignment. Refused: the files are untouched and every
    reader serves the local value, on this boot and the next (which checks again)."""
    text = SPANNING[case]
    base = _local_values(plane, monkeypatch, text, {}, [NAME])
    assert set(base[NAME].values()) == {text.split('"')[1].replace("\\n", "\n").replace("\\r", "\r") + "|"}
    auth = json.dumps({"providers": {"openai-codex": {"t": "local-record"}}})
    (plane.home / "auth.json").write_text(auth, encoding="utf-8")
    caplog.set_level(logging.WARNING)
    first, later = _remote_boot(plane, monkeypatch, text, {}, [NAME])
    assert plane.patches() == []
    assert (plane.home / ".env").read_bytes() == plane.local_bytes == text.encode("utf-8")
    assert (plane.home / "auth.json").read_text(encoding="utf-8") == auth
    assert not (plane.home / ".secrets-import.json").exists()
    assert first == base and later == base
    from hermes_cli.auth import _load_auth_store
    assert _load_auth_store(plane.home / "auth.json")["providers"]["openai-codex"] == {"t": "local-record"}
    warnings = routing.routed_provider().doctor_warnings(plane.home)
    assert len(warnings) == 1 and NAME in warnings[0][0] and NAME in warnings[0][1]
    assert NAME in caplog.text and "not imported" in caplog.text
    for shown in (json.dumps(warnings), caplog.text):
        assert "first" not in shown and "part|" not in shown and "local-record" not in shown
    assert "import refused" in routing.routed_provider().describe(plane.home)


def test_a_refused_home_writes_to_its_local_files(plane, monkeypatch):
    from hermes_cli.config import save_env_value
    text = SPANNING["escaped_lf"]
    _remote_boot(plane, monkeypatch, text, {}, [NAME])
    save_env_value("OPENAI_API_KEY", "written-locally")
    assert plane.patches() == []
    assert "OPENAI_API_KEY=written-locally" in (plane.home / ".env").read_text(encoding="utf-8")


def test_a_simplified_file_imports_on_the_next_boot(plane, monkeypatch):
    _remote_boot(plane, monkeypatch, SPANNING["escaped_lf"], {}, [NAME])
    assert plane.patches() == []
    simplified = 'RV_DIR="first\\npart"\nGATEWAY_RELAY_IDP_CLIENT_SECRET=${RV_DIR}|\n'
    first, later = _remote_boot(plane, monkeypatch, simplified, {}, [NAME])
    assert len(plane.patches()) == 1
    assert set(first[NAME].values()) == set(later[NAME].values()) == {"first\npart|"}
    assert routing.routed_provider().doctor_warnings(plane.home) == ()


FROZEN_OK = {
    "spanning_one_line": ("RV_DIR=first\nGATEWAY_RELAY_IDP_CLIENT_SECRET=${RV_DIR}|\nRV_DIR=last\n", {}),
    "spanning_shell": ("GATEWAY_RELAY_IDP_CLIENT_SECRET=${RV_SHELL}|\nRV_DIR=x\n", {"RV_SHELL": "from-shell"}),
    "multiline_single_assignment": ('RV_DIR="l1\\nl2"\nGATEWAY_RELAY_IDP_CLIENT_SECRET=${RV_DIR}|\n', {}),
}


@pytest.mark.parametrize("case", FROZEN_OK)
def test_kept_lines_that_keep_their_value_still_import(plane, monkeypatch, case):
    text, shell = FROZEN_OK[case]
    base = _local_values(plane, monkeypatch, text, shell, [NAME])
    first, later = _remote_boot(plane, monkeypatch, text, shell, [NAME])
    assert len(plane.patches()) == 1
    assert first == base and later == base


def _generated():
    """``.env`` files whose kept line references an imported name, a shell name, or one assigned twice."""
    from hermes_cli.config import _quote_env_value
    rng = random.Random(0x1F0E)
    alphabet = ["a", "Z", "0", " ", "\t", "\n", "\r", '"', "'", "\\", "$", "${", "${RV_X}", "${RV_X:-d}", "#"]
    values = ["line\nsecond", "line\rsecond", "a\r\nb", "x\ty", '"quote"', "single'", "back\\slash", "$", "${",
              "${RV_X}", "#tag", "a # b", "\\n", "a"]
    values += ["".join(rng.choice(alphabet) for _ in range(rng.randint(1, 6))) for _ in range(20)]
    for i, value in enumerate(values):
        quoted = _quote_env_value(value)
        yield f"shell-{i}", f"{REF}=${{RV_SHELL}}|\nRV_DIR=x\n", {"RV_SHELL": value}
        yield f"file-{i}", f"RV_DIR={quoted}\n{REF}=${{RV_DIR}}|\n", {}
        yield f"spanning-{i}", f"RV_DIR={quoted}\n{REF}=${{RV_DIR}}|\nRV_DIR=last\n", {}
        yield f"spanning-raw-{i}", f'RV_DIR="{value}"\n{REF}=${{RV_DIR}}|\nRV_DIR=last\n', {}


_CASES = list(_generated())


@pytest.mark.parametrize("label,text,shell", _CASES, ids=[c[0] for c in _CASES])
def test_generated_files_either_import_with_base_values_or_are_refused_untouched(plane, monkeypatch, label, text, shell):
    """For every generated ``.env``: the import ran and every reader serves the local value at the
    import's boot and every later one, or it was refused with nothing sent and the file untouched
    (and the readers still serve the local value)."""
    base = _local_values(plane, monkeypatch, text, shell, [REF])
    first, later = _remote_boot(plane, monkeypatch, text, shell, [REF])
    if plane.patches():
        assert len(plane.patches()) == 1, label
    else:
        assert (plane.home / ".env").read_bytes() == plane.local_bytes, label
        assert routing.routed_provider().doctor_warnings(plane.home), label
    assert first == base, (label, base, first)
    assert later == base, (label, base, later)


def test_the_generated_files_include_both_outcomes(plane, monkeypatch):
    """Guard against a generator that never reaches the refusal (or never imports)."""
    from plugins.secret_store.remote import importer
    provider = routing.routed_provider()
    outcomes = set()
    for _label, text, shell in _CASES:
        _fresh(monkeypatch, shell)
        (plane.home / ".env").write_text(text, encoding="utf-8", newline="")
        env = {k: v for k, v in importer._dotenv_values(plane.home / ".env").items()
               if not routing.stays_local(provider, k)}
        outcomes.add(bool(importer.unmatched_kept_names(plane.home, provider, env)))
    assert outcomes == {True, False}
