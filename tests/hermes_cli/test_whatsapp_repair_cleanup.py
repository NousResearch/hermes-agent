"""Re-pair must not launch against credentials it failed to clear (#138627)."""

from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def pairing_wizard(tmp_path, monkeypatch):
    from hermes_cli import config, main, main_platform_setup
    from gateway.platforms import whatsapp_common
    import hermes_constants

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(main, "get_hermes_home", lambda: home)
    monkeypatch.setattr(main, "_require_tty", lambda *_: None)
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    (bridge / "bridge.js").write_text("// fixture", encoding="utf-8")
    (bridge / "node_modules").mkdir()
    monkeypatch.setattr(whatsapp_common, "resolve_whatsapp_bridge_dir", lambda: bridge)
    monkeypatch.setattr(hermes_constants, "find_node_executable", lambda _: "node")
    monkeypatch.setattr(hermes_constants, "with_hermes_node_path", dict)
    values = {"WHATSAPP_MODE": "bot", "WHATSAPP_ALLOWED_USERS": "15551234567"}
    monkeypatch.setattr(config, "get_env_value", values.get)
    monkeypatch.setattr(config, "save_env_value", values.__setitem__)
    monkeypatch.setattr("builtins.input", lambda prompt: "y" if "Re-pair?" in prompt else "n")
    session = home / "whatsapp" / "session"
    session.mkdir(parents=True)
    creds = session / "creds.json"
    creds.write_text('{"old": true}', encoding="utf-8")
    launches = []

    def pair(argv, **kwargs):
        assert argv[-2:] == ["--session", str(session)]
        assert kwargs["cwd"] == str(bridge)
        assert not creds.exists(), "old credentials reached the pairing process"
        launches.append(argv)
        creds.write_text('{"paired": true}', encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(main_platform_setup.subprocess, "run", pair)
    return SimpleNamespace(run=lambda: main_platform_setup.cmd_whatsapp(SimpleNamespace()),
                           creds=creds, launches=launches, values=values)


@pytest.mark.platforms("windows")
def test_locked_credentials_refuse_pairing_then_allow_retry(pairing_wizard, capsys):
    wizard = pairing_wizard
    # Native Windows open files deny deletion; no mocked rmtree or fake OS.
    with wizard.creds.open("rb"):
        with pytest.raises(SystemExit) as stopped:
            wizard.run()
        assert stopped.value.code == 1
        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert wizard.launches == [], "pairing started after failed session cleanup"
        assert "Session cleared" not in output
        assert "stop" in output.lower() and "gateway" in output.lower()
        assert wizard.creds.read_text(encoding="utf-8") == '{"old": true}'
        assert "WHATSAPP_ENABLED" not in wizard.values
    wizard.run()
    assert len(wizard.launches) == 1
    assert wizard.values["WHATSAPP_ENABLED"] == "true"


def test_unlocked_repair_clears_old_state_and_pairs(pairing_wizard, capsys):
    pairing_wizard.run()
    assert len(pairing_wizard.launches) == 1
    assert pairing_wizard.values["WHATSAPP_ENABLED"] == "true"
    assert "Session cleared" in capsys.readouterr().out


def test_recreation_failure_stops_pairing(pairing_wizard, monkeypatch, capsys):
    session = pairing_wizard.creds.parent
    original_mkdir = Path.mkdir

    def refuse_recreation(path, *args, **kwargs):
        if path == session and not path.exists():
            raise PermissionError("session recreation denied")
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", refuse_recreation)
    with pytest.raises(SystemExit) as stopped:
        pairing_wizard.run()
    assert stopped.value.code == 1
    captured = capsys.readouterr()
    assert "Session cleared" not in captured.out
    assert "session recreation denied" in captured.err
    assert pairing_wizard.launches == []
    assert "WHATSAPP_ENABLED" not in pairing_wizard.values
