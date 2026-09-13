"""ByteRover's ``brv`` child carries the SERVED profile's cloud identity, never the launch profile's.

Regression for #108993: ``_run_brv`` built the child env from raw ``os.environ``, which under
``gateway.multiplex_profiles`` is the default profile's ``.env`` — a secondary's turn curated into
the default's ByteRover cloud account while its local context tree was already profile-scoped."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent import secret_scope
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from plugins.memory import byterover


@pytest.fixture
def two_profiles(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    prof_b = root / "profiles" / "b"
    prof_b.mkdir(parents=True)
    (root / ".env").write_text("BRV_API_KEY=DEFAULT-PROFILE-KEY\nHERMES_MODEL=default-model\n")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("BRV_API_KEY", "DEFAULT-PROFILE-KEY")  # the gateway loaded default's .env at boot
    monkeypatch.setenv("HERMES_MODEL", "default-model")
    monkeypatch.setattr(byterover, "_resolve_brv_path", lambda: "/opt/brv/bin/brv")
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["env"] = kwargs["env"]

        class _R:
            returncode, stdout, stderr = 0, "", ""

        return _R()

    monkeypatch.setattr(byterover.subprocess, "run", fake_run)
    return root, prof_b, captured


def _served_turn(prof_home: Path, scope: dict):
    secret_scope.set_multiplex_active(True)
    home_tok = set_hermes_home_override(str(prof_home))
    scope_tok = secret_scope.set_secret_scope(scope)
    return home_tok, scope_tok


def _end_turn(tokens):
    home_tok, scope_tok = tokens
    secret_scope.reset_secret_scope(scope_tok)
    reset_hermes_home_override(home_tok)
    secret_scope.set_multiplex_active(False)


def test_secondary_profile_child_uses_its_own_key_not_defaults(two_profiles):
    _root, prof_b, captured = two_profiles
    tokens = _served_turn(prof_b, {"BRV_API_KEY": "PROFILE-B-KEY"})
    try:
        byterover._run_brv(["query", "--", "hello"], cwd=str(prof_b / "byterover"))
    finally:
        _end_turn(tokens)
    env = captured["env"]
    assert env["BRV_API_KEY"] == "PROFILE-B-KEY"
    assert env["HERMES_HOME"] == str(prof_b)
    assert "HERMES_MODEL" not in env  # launch profile's .env residue is stripped too
    assert env["PATH"].startswith("/opt/brv/bin")


def test_secondary_without_key_gets_no_key_never_defaults(two_profiles):
    _root, prof_b, captured = two_profiles
    tokens = _served_turn(prof_b, {"OTHER": "x"})
    try:
        byterover._run_brv(["curate", "--", "note"], cwd=str(prof_b / "byterover"))
    finally:
        _end_turn(tokens)
    assert "BRV_API_KEY" not in captured["env"]
