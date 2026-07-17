"""Gateway coverage for the read-only /journey slash command."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )
    return runner


def test_journey_is_available_on_gateway():
    from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS, resolve_command

    cmd = resolve_command("journey")
    assert cmd is not None
    assert not cmd.cli_only
    assert "journey" in GATEWAY_KNOWN_COMMANDS


def test_journey_uses_slack_hermes_subcommand_at_native_cap():
    from hermes_cli.commands import slack_native_slashes, slack_subcommand_map

    assert "journey" in slack_subcommand_map()
    assert "journey" not in {name for name, _desc, _hint in slack_native_slashes()}


def test_handle_journey_list_renders_plain_text(monkeypatch):
    import hermes_cli.journey as journey

    monkeypatch.setattr(
        journey,
        "_build_payload",
        lambda: {
            "nodes": [
                {
                    "id": "skill-alpha",
                    "kind": "skill",
                    "label": "Alpha Skill",
                    "timestamp": 1_700_000_000,
                }
            ]
        },
    )

    runner = _make_runner()
    result = asyncio.run(runner._handle_journey_command(_make_event("/journey list")))

    assert "skill-alpha" in result
    assert "Alpha Skill" in result
    assert "\x1b[" not in result


@pytest.mark.parametrize(
    "arguments",
    [
        "delete skill-alpha",
        "edit skill-alpha",
        "--play",
        "--json",
    ],
)
def test_handle_journey_non_read_only_forms_are_cli_only(arguments):
    runner = _make_runner()

    result = asyncio.run(
        runner._handle_journey_command(_make_event(f"/journey {arguments}"))
    )

    assert "read-only" in result
    assert "CLI" in result
