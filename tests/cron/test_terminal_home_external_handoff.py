"""Restart-safe cron's actual Popen handoff has destination HOME in either direction."""
import json
import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def unpinned_temp_environment(monkeypatch):
    # The canonical runner may own TMPDIR without a Hermes marker. These HOME
    # cases exercise Hermes-default scratch, not the explicit-user-temp override.
    for key in ("TMPDIR", "TMP", "TEMP", "HERMES_SCRATCH_DIR"):
        monkeypatch.delenv(key, raising=False)



class HandoffObserved(RuntimeError):
    """Stop at the transport seam, without starting a cron worker."""


@pytest.mark.parametrize("reverse", [False, True], ids=["default-to-secondary", "secondary-to-default"])
@pytest.mark.parametrize("mode", ["real", "profile", None])
def test_external_cron_worker_target_home_handoff(tmp_path, monkeypatch, reverse, mode):
    import hermes_constants as hc
    from cron import scheduler
    from tools import process_registry
    from tools.process_registry import GatewayChildDispatch
    from tools.terminal_scope import get_terminal_scope, set_terminal_scope, reset_terminal_scope
    from agent import secret_scope

    root = tmp_path / ".hermes"
    secondary = root / "profiles" / "secondary"
    (root / "home").mkdir(parents=True)
    (secondary / "home").mkdir(parents=True)
    launch, target = (secondary, root) if reverse else (root, secondary)
    real = tmp_path / "real"
    real.mkdir()
    (launch / ".env").write_text("LAUNCH_PRIVATE=launch-secret\nOPENAI_API_KEY=launch-key\n")
    (target / ".env").write_text("TARGET_MARKER=target-dotenv\n")
    (target / "config.yaml").write_text("terminal:\n  backend: local\n" + (f"  home_mode: {mode}\n" if mode else ""))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(hc, "_PINNED_PROCESS_HERMES_HOME", str(launch))
    for key, value in {"HOME": str(real), "HERMES_REAL_HOME": str(real), "HERMES_HOME": str(launch),
                       "TERMINAL_HOME_MODE": "real" if mode == "profile" else "profile",
                       "TERMINAL_ENV": "docker", "TERMINAL_DOCKER_IMAGE": "launch-image",
                       "TERMINAL_DOCKER_FORWARD_ENV": '["LAUNCH_PRIVATE"]',
                       "LAUNCH_PRIVATE": "launch-secret", "OPENAI_API_KEY": "launch-key"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(process_registry, "restart_safe_gateway_child_argv", lambda argv, **kw: GatewayChildDispatch("degraded", argv))
    monkeypatch.setattr(process_registry, "systemd_user_bus_env", lambda env: env)
    monkeypatch.setattr(scheduler, "mark_execution_handoff_pending", lambda eid: {"id": eid})
    captured = {}
    def popen(argv, **kwargs):
        captured.update(env=dict(kwargs["env"]), argv=list(argv))
        payload = Path(argv[argv.index("--external-worker-file") + 1])
        captured["payload"] = json.loads(payload.read_text())
        raise HandoffObserved("captured external cron boundary")
    monkeypatch.setattr(scheduler.subprocess, "Popen", popen)
    before = dict(os.environ)
    home_token = hc.set_hermes_home_override(str(target))
    scope_token = set_terminal_scope(None)  # this entrypoint can run before in-process fire scope
    multiplex_before = secret_scope.is_multiplex_active()
    try:
        with pytest.raises(HandoffObserved, match="external cron boundary"):
            scheduler._launch_external_cron_worker({"id": "home-handoff", "execution_id": "owned-execution", "prompt": "unused"})
        env = captured["env"]
        expected_profile = mode == "profile" or (mode is None and hc.is_container())
        assert env["HOME"] == str(target / "home" if expected_profile else real)
        assert env["HERMES_HOME"] == str(target)
        assert env["HERMES_REAL_HOME"] == str(real)
        for key in ("TMPDIR", "TMP", "TEMP"):
            assert env[key] == str(target / "cache" / "scratch")
        assert not {"TERMINAL_ENV", "TERMINAL_HOME_MODE", "TERMINAL_DOCKER_IMAGE", "TERMINAL_DOCKER_FORWARD_ENV"} & env.keys()
        assert "LAUNCH_PRIVATE" not in env and "OPENAI_API_KEY" not in env
        assert captured["payload"]["profile_home"] == str(target)
        assert captured["payload"]["multiplex_active"] is True
        assert get_terminal_scope() is None
        assert secret_scope.is_multiplex_active() == multiplex_before
        assert not list((target / "cron" / "external-workers").glob("owned-execution.*"))
    finally:
        reset_terminal_scope(scope_token)
        hc.reset_hermes_home_override(home_token)
    assert dict(os.environ) == before
