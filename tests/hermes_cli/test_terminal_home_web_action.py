"""Named web actions recompute target HOME without changing the caller's scope."""
import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def unpinned_temp_environment(monkeypatch):
    # The canonical runner may own TMPDIR without a Hermes marker. These HOME
    # cases exercise Hermes-default scratch, not the explicit-user-temp override.
    for key in ("TMPDIR", "TMP", "TEMP", "HERMES_SCRATCH_DIR"):
        monkeypatch.delenv(key, raising=False)



@pytest.mark.parametrize("mode", ["real", "profile", None])
def test_web_action_snapshot_targets_its_profile_not_bound_launcher(tmp_path, monkeypatch, mode):
    import hermes_constants as hc
    from hermes_cli.web_server_gateway import _profile_action_environment
    from tools.terminal_scope import set_terminal_scope, reset_terminal_scope, get_terminal_scope

    root = tmp_path / ".hermes"
    target = root / "profiles" / "target"
    (root / "home").mkdir(parents=True)
    (target / "home").mkdir(parents=True)
    (root / ".env").write_text("LAUNCH_PRIVATE=launch-only\nOPENAI_API_KEY=launch-provider\n")
    (target / "config.yaml").write_text("terminal:\n  backend: local\n" + (f"  home_mode: {mode}\n" if mode else ""))
    real = tmp_path / "real"
    real.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(hc, "_PINNED_PROCESS_HERMES_HOME", str(root))
    for k, v in {"HOME": str(real), "HERMES_REAL_HOME": str(real), "HERMES_HOME": str(root),
                 "TERMINAL_HOME_MODE": "real" if mode == "profile" else "profile",
                 "OPENAI_API_KEY": "launch-provider", "LAUNCH_PRIVATE": "launch-only"}.items():
        monkeypatch.setenv(k, v)
    before = dict(os.environ)
    home_token = hc.set_hermes_home_override(str(root))
    scope = {"TERMINAL_HOME_MODE": "real" if mode == "profile" else "profile"}
    scope_token = set_terminal_scope(scope)
    try:
        # `config path` is not a host-gateway action, so it uses the named-profile
        # snapshot path without starting a process or touching any service.
        child = _profile_action_environment(["-p", "target", "config", "path"])
        expected_profile = mode == "profile" or (mode is None and hc.is_container())
        assert child["HOME"] == str(target / "home" if expected_profile else real)
        assert child["HERMES_HOME"] == str(target)
        assert child["HERMES_REAL_HOME"] == str(real)
        for key in ("TMPDIR", "TMP", "TEMP"):
            assert child[key] == str(target / "cache" / "scratch")
        assert "OPENAI_API_KEY" not in child and "LAUNCH_PRIVATE" not in child
        assert child["HERMES_NONINTERACTIVE"] == "1"
        assert get_terminal_scope() is scope
    finally:
        reset_terminal_scope(scope_token)
        hc.reset_hermes_home_override(home_token)
    assert dict(os.environ) == before
