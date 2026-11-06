"""Tests for the gateway /refresh slash command (prompt-refresh request surface).

Mirrors tests/agent/test_session_prompt_refresh.py's store semantics at the gateway seam:
the handler only records the epoch request in the serving profile's store; the rebuild
itself is the per-turn probe in agent/prompt_refresh.py.
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from hermes_cli.commands import resolve_command


def _make_event(text: str, chat_id: str = "555000111") -> MessageEvent:
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id=f"user-{chat_id}",
        chat_id=chat_id,
        user_name="tester",
        chat_type="dm",
    )
    return MessageEvent(text=text, source=source)


class _FakeDB:
    """Records request_prompt_refresh calls; resolves a canned prefix."""

    def __init__(self, stored_prompt: bool = True) -> None:
        self.requests: list[str | None] = []
        self.stored_prompt = stored_prompt

    async def resolve_session_id(self, prefix: str):
        if prefix.startswith("a"):
            return "sess-a1"  # a session already at its tip (no compression child)
        if prefix.startswith("r"):
            return "sess-root"  # an ended root with a compression child
        return None

    async def get_compression_tip(self, session_id: str):
        if session_id == "sess-a1":
            return None
        if session_id == "sess-root":
            return "sess-tip"
        return None

    async def get_session(self, session_id: str):
        return {"system_prompt": "p"} if self.stored_prompt else {}

    async def request_prompt_refresh(self, session_id):
        self.requests.append(session_id)
        return len(self.requests)

    async def count_sessions_with_stored_prompt(self):
        return 7 if self.stored_prompt else 0


class TestRegistry:
    @pytest.mark.parametrize(
        ("field", "expected"),
        [("gateway_only", True), ("busy_policy", "dispatch")],
    )
    def test_refresh_registered_gateway_only_and_dispatch(self, field, expected):
        cmd = resolve_command("refresh")
        assert cmd is not None, "/refresh must be a known command"
        assert getattr(cmd, field) == expected

    def test_refresh_composed_onto_runner(self):
        assert hasattr(gateway_run.GatewayRunner, "_handle_refresh_command")


def _make_runner(fake_db: _FakeDB, stored_session: str = "sess-conv-1") -> Any:
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.session_store = None
    runner.config = None

    async def get_or_create_session(source):
        return SimpleNamespace(session_id=stored_session)

    runner._async_session_store = SimpleNamespace(_store=None, get_or_create_session=get_or_create_session)
    runner._open_session_db_for_active_scope = lambda: fake_db
    return runner


class TestTargetedScopes:
    @pytest.mark.asyncio
    async def test_bare_refresh_names_the_commanding_session(self):
        db = _FakeDB()
        runner = _make_runner(db)
        reply = await runner._handle_refresh_command(_make_event("/refresh"))
        assert db.requests == ["sess-conv-1"]
        assert "sess-conv-1" in reply

    @pytest.mark.asyncio
    async def test_prefix_redirects_compression_tip_and_respects_the_tip(self):
        """A request keyed on an ended root must land on the tip, like the CLI verb;
        a session already at its tip is requested as-is."""
        db = _FakeDB()
        runner = _make_runner(db)
        reply = await runner._handle_refresh_command(_make_event("/refresh root-2026"))
        assert db.requests == ["sess-tip"]
        assert "sess-tip" in reply

        db = _FakeDB()
        runner = _make_runner(db)
        await runner._handle_refresh_command(_make_event("/refresh atip-1"))
        assert db.requests == ["sess-a1"]

    @pytest.mark.asyncio
    async def test_unknown_prefix_is_rejected(self):
        db = _FakeDB()
        runner = _make_runner(db)
        reply = await runner._handle_refresh_command(_make_event("/refresh nope-99"))
        assert db.requests == []
        assert "nope-99" in reply

    @pytest.mark.asyncio
    async def test_reply_states_the_next_turn_contract(self):
        db = _FakeDB()
        runner = _make_runner(db)
        reply = await runner._handle_refresh_command(_make_event("/refresh"))
        low = reply.lower()
        assert "next turn" in low
        assert "history is kept" in low


class TestProfileWideScope:
    @pytest.mark.asyncio
    async def test_all_goes_through_destructive_confirmation(self):
        """/refresh all must never fire without the destructive-slash confirm pass."""
        db = _FakeDB()
        runner = _make_runner(db)
        seen = {}

        async def fake_confirm(*, event, command, title, detail, execute):
            seen.update(command=command, title=title, count_in_detail="7" in detail)
            return await execute()

        runner._maybe_confirm_destructive_slash = fake_confirm
        reply = await runner._handle_refresh_command(_make_event("/refresh all"))
        assert seen == {"command": "refresh", "title": "/refresh all", "count_in_detail": True}
        assert db.requests == [None]  # None = profile-wide (same as the CLI verb)
        assert reply  # confirm path returned the queued reply
