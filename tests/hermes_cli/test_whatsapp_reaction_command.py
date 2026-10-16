"""The public WhatsApp reaction command stays bound to its profile's live bridge."""

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from hermes_cli.subcommands.whatsapp import build_whatsapp_parser


def _parser():
    parser = argparse.ArgumentParser()
    build_whatsapp_parser(parser.add_subparsers(dest="command"), cmd_whatsapp=lambda _args: None)
    return parser


def _write_profile(home: Path, port: int, session: Path):
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(
        "platforms:\n  whatsapp:\n    enabled: true\n    extra:\n"
        f"      bridge_port: {port}\n      session_path: {session}\n", encoding="utf-8",
    )


def test_reaction_command_routes_only_to_own_bridge(tmp_path, monkeypatch, capsys):
    received = []
    state = {"session": str(tmp_path / "a-session")}

    class Bridge(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"session": state["session"]}).encode())

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(body)
            self.send_response(404 if body["messageId"] == "other-chat" else 200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Message not found in this chat"}
                                    if body["messageId"] == "other-chat" else {"success": True}).encode())

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Bridge)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_port
        a_home, b_home = tmp_path / "a", tmp_path / "b"
        _write_profile(a_home, port, tmp_path / "a-session")
        _write_profile(b_home, port, tmp_path / "b-session")
        parser = _parser()
        react = parser.parse_args(["whatsapp", "react", "--to", "15551234567", "--message-id", "incoming", "👍"])
        unreact = parser.parse_args(["whatsapp", "unreact", "--to", "15551234567", "--message-id", "incoming"])
        foreign = parser.parse_args(["whatsapp", "react", "--to", "15551234567", "--message-id", "other-chat", "👍"])

        monkeypatch.setenv("HERMES_HOME", str(a_home))
        assert react.func(react) == 0
        assert unreact.func(unreact) == 0
        assert received == [
            {"chatId": "15551234567@s.whatsapp.net", "messageId": "incoming", "emoji": "👍"},
            {"chatId": "15551234567@s.whatsapp.net", "messageId": "incoming", "emoji": ""},
        ]
        assert foreign.func(foreign) == 1
        assert "Message not found in this chat" in capsys.readouterr().err

        before = len(received)
        monkeypatch.setenv("HERMES_HOME", str(b_home))
        assert react.func(react) == 1
        assert len(received) == before
        assert "does not match this profile" in capsys.readouterr().err

        state["session"] = None
        monkeypatch.setenv("HERMES_HOME", str(a_home))
        assert react.func(react) == 1
        assert len(received) == before
        state["session"] = str(tmp_path / "a-session")

        assert react.func(react) == 0
        assert len(received) == before + 1

        with patch("tools.send_message_tool._authorize_relay_target", return_value="relay target denied"):
            assert react.func(react) == 1
        assert len(received) == before + 1
        assert "relay target denied" in capsys.readouterr().err
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
