"""CLI contract for Baileys WhatsApp native outbound mentions."""

from __future__ import annotations

import argparse
import json
import sys
import types

import pytest

from hermes_cli import send_cmd


def _parse(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    send_cmd.register_send_subparser(subparsers)
    return parser.parse_args(["send", *argv])


def _install_tool(monkeypatch, calls: list[dict]) -> None:
    module = types.ModuleType("tools.send_message_tool")

    def fake_send_message_tool(args, **_kwargs):
        calls.append(dict(args))
        return json.dumps({"success": True})

    module.send_message_tool = fake_send_message_tool
    monkeypatch.setitem(sys.modules, "tools.send_message_tool", module)
    monkeypatch.setattr(send_cmd, "_load_hermes_env", lambda: None)


def test_repeatable_comma_separated_mentions_are_normalized_and_forwarded(monkeypatch):
    calls: list[dict] = []
    _install_tool(monkeypatch, calls)
    args = _parse(
        "--to", "whatsapp:120363408391911677@g.us",
        "--mention", "919223223556, 919424513431@s.whatsapp.net",
        "--mention", "149606612619433@lid",
        "@919223223556 Test",
    )

    with pytest.raises(SystemExit) as exc:
        send_cmd.cmd_send(args)

    assert exc.value.code == 0
    assert calls == [{
        "action": "send",
        "target": "whatsapp:120363408391911677@g.us",
        "message": "@919223223556 Test",
        "mentions": [
            "919223223556@s.whatsapp.net",
            "919424513431@s.whatsapp.net",
            "149606612619433@lid",
        ],
    }]


def test_no_mentions_preserves_send_tool_payload_shape(monkeypatch):
    calls: list[dict] = []
    _install_tool(monkeypatch, calls)

    with pytest.raises(SystemExit) as exc:
        send_cmd.cmd_send(_parse("--to", "whatsapp:120363408391911677@g.us", "hello"))

    assert exc.value.code == 0
    assert calls == [{
        "action": "send",
        "target": "whatsapp:120363408391911677@g.us",
        "message": "hello",
    }]


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (("--to", "whatsapp:120363408391911677@g.us", "--mention", "alice", "hello"),
         "invalid whatsapp mention"),
        (("--to", "whatsapp:120363408391911677@g.us", "--mention", "123,,456", "hello"),
         "empty whatsapp mention"),
        (("--to", "telegram:-100123", "--mention", "919223223556", "hello"),
         "only supported for whatsapp group targets"),
        (("--to", "whatsapp:919223223556", "--mention", "919424513431", "hello"),
         "requires a whatsapp group target"),
    ],
)
def test_invalid_mention_usage_fails_before_send(monkeypatch, capsys, argv, message):
    calls: list[dict] = []
    _install_tool(monkeypatch, calls)

    with pytest.raises(SystemExit) as exc:
        send_cmd.cmd_send(_parse(*argv))

    assert exc.value.code == 2
    assert message in capsys.readouterr().err.lower()
    assert calls == []
