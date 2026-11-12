"""Interactive passkey setup must not enable an unfinished pairing."""

import argparse
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("returncode", [0, 1, None])
def test_pairing_exit_controls_enablement_and_forwards_browser(tmp_path, monkeypatch, capsys, returncode):
    from hermes_agent.gateway.platforms import whatsapp_common
    from hermes_agent.hermes_cli import config, main, main_platform_setup as setup
    from hermes_agent.hermes_cli.subcommands.whatsapp import build_whatsapp_parser
    import hermes_constants

    parser = argparse.ArgumentParser()
    build_whatsapp_parser(parser.add_subparsers(), cmd_whatsapp=setup.cmd_whatsapp)
    browser = str(tmp_path / "browser with spaces")
    args = parser.parse_args(["whatsapp", "--passkey-browser", browser])
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    (bridge / "bridge.js").write_text("", encoding="utf-8")
    saved = {}
    monkeypatch.setattr(main, "_require_tty", lambda command: None)
    monkeypatch.setattr(main, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(setup, "_whatsapp_choose_mode", lambda *args: "bot")
    monkeypatch.setattr(setup, "_whatsapp_allowed_users", lambda *args: None)
    monkeypatch.setattr(setup, "_whatsapp_install_bridge", lambda path: True)
    monkeypatch.setattr(config, "get_env_value", lambda key: saved.get(key))
    monkeypatch.setattr(config, "save_env_value", lambda key, value: saved.update({key: value}))
    monkeypatch.setattr(whatsapp_common, "resolve_whatsapp_bridge_dir", lambda: bridge)
    monkeypatch.setattr(hermes_constants, "find_node_executable", lambda name: "isolated-node")
    monkeypatch.setattr(hermes_constants, "with_hermes_node_path", dict)

    def pair(command, **kwargs):
        assert command[-2:] == ["--passkey-browser", browser]
        assert command[:3] == ["isolated-node", str(bridge / "bridge.js"), "--pair-only"]
        session = Path(command[command.index("--session") + 1])
        assert session == tmp_path / "whatsapp" / "session"
        assert "timeout" not in kwargs  # QR scan + 600 s WebAuthn prompt + confirmation have no fixed bound
        # A paired identity is on disk in every case, so only the exit status decides enablement.
        (session / "creds.json").write_text('{"me":{"id":"15551234567:1@s.whatsapp.net"}}', encoding="utf-8")
        if returncode is None:
            raise KeyboardInterrupt
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(subprocess, "run", pair)
    args.func(args)
    assert (saved.get("WHATSAPP_ENABLED") == "true") == (returncode == 0)
    output = capsys.readouterr().out
    assert ("paired successfully" in output) == (returncode == 0)
