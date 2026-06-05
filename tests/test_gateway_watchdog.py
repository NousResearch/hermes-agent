import shlex

import hermes_cli.gateway as gateway


def test_launchd_watchdog_script_is_profile_scoped(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    project_root = tmp_path / "hermes-agent"
    monkeypatch.setattr(gateway, "get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr(gateway, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(gateway, "get_python_path", lambda: "/opt/hermes/.venv/bin/python")
    monkeypatch.setattr(gateway, "_profile_arg", lambda hermes_home=None: "")

    script = gateway.generate_launchd_watchdog_script(cpu_threshold=75)

    assert f"HERMES_HOME={shlex.quote(str(hermes_home))}" in script
    assert "PID_FILE=\"$HERMES_HOME/gateway.pid\"" in script
    assert "tail -n 50 \"$HEARTBEAT_FILE\"" in script
    assert "CPU_THRESHOLD=\"${HERMES_GATEWAY_WATCHDOG_CPU_THRESHOLD:-75}\"" in script
    assert "gateway start" in script
    assert "/Users/linyipeng" not in script
    assert "/.local/bin/hermes" not in script


def test_launchd_watchdog_plist_uses_interval_and_threshold(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    user_home = tmp_path / "user"
    monkeypatch.setattr(gateway, "get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr(gateway, "_launchd_user_home", lambda: user_home)
    monkeypatch.setattr(gateway, "get_launchd_label", lambda: "ai.hermes.gateway-test")

    plist = gateway.generate_launchd_watchdog_plist(interval=600, cpu_threshold=90)

    assert "<string>ai.hermes.gateway-test.watchdog</string>" in plist
    assert "<key>StartInterval</key>" in plist
    assert "<integer>600</integer>" in plist
    assert "<key>RunAtLoad</key>" in plist
    assert "<string>90</string>" in plist
    assert str(hermes_home / "bin" / "hermes-gateway-watchdog.sh") in plist
