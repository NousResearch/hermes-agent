"""Copied process snapshots at real managed-worker and web-action consumers."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def unpinned_temp_environment(monkeypatch):
    # The canonical runner may own TMPDIR without a Hermes marker. These HOME
    # cases exercise Hermes-default scratch, not the explicit-user-temp override.
    for key in ("TMPDIR", "TMP", "TEMP", "HERMES_SCRATCH_DIR"):
        monkeypatch.delenv(key, raising=False)



@pytest.mark.parametrize("routed", [False, True], ids=["launch-owner-snapshot", "routed-target"])
@pytest.mark.parametrize("mode", ["real", "profile"])
def test_managed_worker_snapshot_home_and_secret_boundary(tmp_path, monkeypatch, routed, mode):
    import hermes_constants as hc
    from agent import secret_scope
    from gateway.session_managed_worker import _worker_env
    from tools.terminal_scope import set_terminal_scope, reset_terminal_scope, get_terminal_scope

    real = tmp_path / "real"
    launch = tmp_path / ".hermes"
    target = launch / "profiles" / "worker" if routed else launch
    real.mkdir()
    (target / "home").mkdir(parents=True)
    (launch / ".env").write_text("LAUNCH_PRIVATE=launch-only\nOPENAI_API_KEY=launch-provider\n")
    if routed:
        (target / ".env").write_text("OPENAI_API_KEY=target-provider\nTARGET_PRIVATE=target-only\n")
    (target / "config.yaml").write_text(f"terminal: {{home_mode: {mode}}}\n")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(hc, "_PINNED_PROCESS_HERMES_HOME", str(launch))
    for k, v in {"HOME": str(real), "HERMES_REAL_HOME": str(real), "HERMES_HOME": str(launch),
                 "TERMINAL_HOME_MODE": "profile" if mode == "real" else "real",
                 "LAUNCH_PRIVATE": "launch-only", "OPENAI_API_KEY": "launch-provider",
                 "AUXILIARY_VISION_API_KEY": "internal-never-forward"}.items():
        monkeypatch.setenv(k, v)
    before = dict(os.environ)
    secret_scope.set_multiplex_active(True)
    home_token = hc.set_hermes_home_override(str(target))
    scope = {"TERMINAL_HOME_MODE": mode}
    scope_token = set_terminal_scope(scope)
    secret_token = secret_scope.set_secret_scope(secret_scope.build_profile_secret_scope(target), profile_home=str(target))
    try:
        child = _worker_env(SimpleNamespace(profile_id=str(target)))
        assert child["HOME"] == str(real if mode == "real" else target / "home")
        assert child["HERMES_HOME"] == str(target)
        assert child["HERMES_REAL_HOME"] == str(real)
        for key in ("TMPDIR", "TMP", "TEMP"):
            assert child[key] == str(target / "cache" / "scratch")
        assert "AUXILIARY_VISION_API_KEY" not in child
        assert child["OPENAI_API_KEY"] == ("target-provider" if routed else "launch-provider")
        if routed:
            assert "LAUNCH_PRIVATE" not in child
            assert child["TARGET_PRIVATE"] == "target-only"
        assert get_terminal_scope() is scope
    finally:
        secret_scope.reset_secret_scope(secret_token)
        reset_terminal_scope(scope_token)
        hc.reset_hermes_home_override(home_token)
        secret_scope.set_multiplex_active(False)
    assert dict(os.environ) == before
