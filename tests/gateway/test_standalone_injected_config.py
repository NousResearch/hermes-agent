"""The runner boundary honours a profile opt-out even for explicit --config inputs."""

import asyncio
from pathlib import Path

import pytest


@pytest.mark.parametrize("flag", [None, False, True])
def test_injected_config_cannot_multiplex_a_standalone_home(tmp_path, monkeypatch, flag):
    from agent.secret_scope import is_multiplex_active, set_multiplex_active
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner, _profile_runtime_scope

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    solo = root / "profiles" / "solo"
    solo.mkdir(parents=True)
    (root / "config.yaml").write_text("{}\n")
    (solo / "config.yaml").write_text("gateway:\n  standalone: true\n")
    monkeypatch.setenv("HERMES_HOME", str(root))
    previous = is_multiplex_active()
    try:
        for home in (solo, root, solo):
            with _profile_runtime_scope(home):
                config = GatewayConfig(multiplex_profiles=flag, sessions_dir=home / "sessions")
                runner = GatewayRunner(config)
                try:
                    assert runner.config.multiplex_profiles is (False if home == solo else flag)
                    assert is_multiplex_active() is (False if home == solo else bool(flag))
                finally:
                    if runner._session_db is not None:
                        asyncio.run(runner._session_db.close())
    finally:
        set_multiplex_active(previous)
