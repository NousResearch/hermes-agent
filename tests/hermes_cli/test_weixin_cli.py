import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def test_top_level_weixin_command_dispatches(monkeypatch):
    import hermes_cli.main as main_mod

    captured = {}

    def fake_cmd_weixin(args):
        captured["command"] = args.command

    monkeypatch.setattr(main_mod, "cmd_weixin", fake_cmd_weixin)
    monkeypatch.setattr(sys, "argv", ["hermes", "weixin"])

    main_mod.main()

    assert captured == {"command": "weixin"}


def test_gateway_setup_weixin_delegates_to_main_flow():
    import hermes_cli.gateway as gateway_mod

    with patch("hermes_cli.main.cmd_weixin") as mock_cmd:
        gateway_mod._setup_weixin()

    mock_cmd.assert_called_once()


def test_setup_wizard_registers_and_delegates_weixin():
    import hermes_cli.setup as setup_mod

    assert ("Weixin", "WEIXIN_ENABLED", setup_mod._setup_weixin) in setup_mod._GATEWAY_PLATFORMS

    with patch("hermes_cli.main.cmd_weixin") as mock_cmd:
        setup_mod._setup_weixin()

    mock_cmd.assert_called_once()


def test_cmd_weixin_repair_kills_existing_bridge_before_clearing_session(monkeypatch, tmp_path):
    import hermes_cli.main as main_mod

    bridge_dir = tmp_path / "bridge"
    bridge_dir.mkdir()
    (bridge_dir / "node_modules").mkdir()
    bridge_script = bridge_dir / "bridge.js"
    bridge_script.write_text("// test bridge\n", encoding="utf-8")

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "credentials.json").write_text(
        '{"accountId":"acct-1","userId":"wxid_123"}',
        encoding="utf-8",
    )

    env_values = {
        "WEIXIN_ENABLED": "true",
        "WEIXIN_ALLOWED_USERS": "wxid_123",
        "WEIXIN_HOME_CHANNEL": "wxid_123",
        "WEIXIN_BRIDGE_PORT": "3010",
        "WEIXIN_SESSION_PATH": str(session_dir),
        "WEIXIN_BRIDGE_SCRIPT": str(bridge_script),
    }
    events = []

    def fake_get_env_value(name):
        return env_values.get(name, "")

    def fake_kill_port_process(port):
        assert port == 3010
        events.append("kill")

    def fake_rmtree(path, ignore_errors=False):
        assert Path(path) == session_dir
        assert ignore_errors is True
        events.append("rmtree")

    mock_proc = SimpleNamespace(poll=lambda: None)

    monkeypatch.setattr(main_mod, "_require_tty", lambda _: None)
    monkeypatch.setattr("hermes_cli.config.get_env_value", fake_get_env_value)
    monkeypatch.setattr("hermes_cli.config.save_env_value", lambda *args, **kwargs: None)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: tmp_path / "home")
    monkeypatch.setattr("gateway.platforms.weixin.check_weixin_requirements", lambda: True)
    monkeypatch.setattr("gateway.platforms.weixin._kill_port_process", fake_kill_port_process)
    monkeypatch.setattr(main_mod, "_weixin_bridge_health", lambda _: {"status": "connected", "accountId": "acct-2"})
    monkeypatch.setattr(main_mod, "_terminate_child_process", lambda proc: None)
    monkeypatch.setattr("shutil.rmtree", fake_rmtree)
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: mock_proc)
    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)

    with patch("builtins.input", side_effect=["y"]):
        main_mod.cmd_weixin(SimpleNamespace(command="weixin"))

    assert events[:2] == ["kill", "rmtree"]


def test_cmd_weixin_honors_env_overrides_and_expands_paths(monkeypatch, tmp_path):
    import hermes_cli.main as main_mod

    monkeypatch.setenv("HOME", str(tmp_path))

    bridge_script = tmp_path / "custom-bridge" / "bridge.js"
    bridge_script.parent.mkdir(parents=True)
    (bridge_script.parent / "node_modules").mkdir()
    bridge_script.write_text("// test bridge\n", encoding="utf-8")

    env_values = {
        "WEIXIN_ENABLED": "true",
        "WEIXIN_ALLOW_ALL_USERS": "true",
        "WEIXIN_HOME_CHANNEL": "wxid_home",
        "WEIXIN_BRIDGE_PORT": "3015",
        "WEIXIN_SESSION_PATH": "~/custom-session",
        "WEIXIN_BRIDGE_SCRIPT": "~/custom-bridge/bridge.js",
    }

    def fake_get_env_value(name):
        return env_values.get(name, "")

    popen_calls = {}

    class _Proc:
        def poll(self):
            return None

    def fake_popen(cmd, cwd=None):
        popen_calls["cmd"] = cmd
        popen_calls["cwd"] = cwd
        return _Proc()

    health_states = iter([None, {"status": "connected", "accountId": "acct-3"}])

    monkeypatch.setattr(main_mod, "_require_tty", lambda _: None)
    monkeypatch.setattr("hermes_cli.config.get_env_value", fake_get_env_value)
    monkeypatch.setattr("hermes_cli.config.save_env_value", lambda *args, **kwargs: None)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: tmp_path / "unused-home")
    monkeypatch.setattr("gateway.platforms.weixin.check_weixin_requirements", lambda: True)
    monkeypatch.setattr("gateway.platforms.weixin._kill_port_process", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_mod, "_weixin_bridge_health", lambda _: next(health_states))
    monkeypatch.setattr(main_mod, "_terminate_child_process", lambda proc: None)
    monkeypatch.setattr("subprocess.Popen", fake_popen)
    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)
    main_mod.cmd_weixin(SimpleNamespace(command="weixin"))

    assert popen_calls["cmd"] == [
        "node",
        str(bridge_script),
        "--port",
        "3015",
        "--session",
        str(tmp_path / "custom-session"),
    ]
    assert popen_calls["cwd"] == str(bridge_script.parent)


def test_cmd_weixin_defaults_to_open_dm_access(monkeypatch, tmp_path):
    import hermes_cli.main as main_mod

    bridge_dir = tmp_path / "bridge"
    bridge_dir.mkdir()
    (bridge_dir / "node_modules").mkdir()
    bridge_script = bridge_dir / "bridge.js"
    bridge_script.write_text("// test bridge\n", encoding="utf-8")

    env_values = {
        "WEIXIN_ENABLED": "true",
        "WEIXIN_HOME_CHANNEL": "wxid_home",
        "WEIXIN_BRIDGE_SCRIPT": str(bridge_script),
    }
    saved = []

    def fake_get_env_value(name):
        return env_values.get(name, "")

    def fake_save_env_value(key, value):
        saved.append((key, value))
        env_values[key] = value

    health_states = iter([None, {"status": "connected", "accountId": "acct-3", "userId": "wxid_test"}])

    class _Proc:
        def poll(self):
            return None

    monkeypatch.setattr(main_mod, "_require_tty", lambda _: None)
    monkeypatch.setattr("hermes_cli.config.get_env_value", fake_get_env_value)
    monkeypatch.setattr("hermes_cli.config.save_env_value", fake_save_env_value)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: tmp_path / "unused-home")
    monkeypatch.setattr("gateway.platforms.weixin.check_weixin_requirements", lambda: True)
    monkeypatch.setattr("gateway.platforms.weixin._kill_port_process", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_mod, "_weixin_bridge_health", lambda _: next(health_states))
    monkeypatch.setattr(main_mod, "_terminate_child_process", lambda proc: None)
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: _Proc())
    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)

    main_mod.cmd_weixin(SimpleNamespace(command="weixin"))

    assert ("WEIXIN_ALLOW_ALL_USERS", "true") in saved
