"""Standalone sends must not become assistant speech in the target session."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

# Load before the per-test home I/O guard: this module's bootstrap legitimately
# probes the install manifest beside a live source checkout.
import agent.conversation_loop
from gateway.config import Platform
from hermes_cli import send_cmd
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


def _parse(argv):
    import argparse

    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    send_cmd.register_send_subparser(subparsers)
    return parser.parse_args(["send", *argv])


def _run_async_immediately(coro):
    return asyncio.run(coro)


def test_standalone_send_mirror_replays_as_a_labelled_user_event(tmp_path, monkeypatch):
    """A report delivered between two DM turns must not merge into either assistant reply."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    db = storage_handle(tmp_path / "home")
    seed_session(db, "dm-session", "telegram")
    seed_messages(
        db, "dm-session",
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Anytime."},
    )

    config = SimpleNamespace(
        platforms={Platform.TELEGRAM: SimpleNamespace(enabled=True, token="test-token", extra={})},
        get_home_channel=lambda _platform: None,
    )
    session_env = lambda name, default="": "bot_chat" if name == "HERMES_SESSION_PLATFORM" else default

    with patch("hermes_cli.send_cmd._load_hermes_env"), \
         patch("gateway.config.load_gateway_config", return_value=config), \
         patch("gateway.outbound.send.prepare_send_message_platforms"), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("gateway.outbound.send._send_to_platform", new=AsyncMock(return_value={"success": True})), \
         patch("gateway.session_context.get_session_env", side_effect=session_env), \
         patch("gateway.mirror._find_session_id", return_value="dm-session"):
        with pytest.raises(SystemExit) as exc:
            send_cmd.cmd_send(_parse(["--to", "telegram:12345", "Wed Nov 18 evening report"]))

    assert exc.value.code == 0
    seed_messages(db, "dm-session", {"role": "user", "content": "thanks!"})
    resume = db.messages.read_resume_conversations("dm-session")
    model_history = [message.as_openai_message() for message in resume.model]

    assert [message["role"] for message in model_history] == ["user", "assistant", "user"]
    assert model_history[1]["content"] == "Anytime."
    assert model_history[2]["content"].startswith("[Standalone send: bot_chat]\n")
    assert "Wed Nov 18 evening report" in model_history[2]["content"]
    assert model_history[2]["content"].endswith("[Standalone send end: bot_chat]\n\nthanks!")
