"""Tests for the ``/skills`` gateway handler's review-only scope.

The gateway/chat surface answers only the write-approval review subcommands
(pending / approve / reject / diff / approval); the skills-hub mutations
(search / browse / inspect / install / audit) stay CLI-only — an unsupported
subcommand must never fall through to the hub. This is the backend guard the
desktop ``desktop_subcommands`` scope mirrors client-side (#98330 review).
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


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
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)

    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._background_tasks = set()
    runner._session_db = None
    runner._reasoning_config = None
    runner._evict_cached_agent = MagicMock()
    from gateway.run import GatewayRunner as _GR
    runner._session_key_for_source = _GR._session_key_for_source.__get__(runner, _GR)
    return runner


@pytest.fixture
def hermes_home(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("skills:\n  write_approval: true\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("gateway.run._hermes_home", home)
    return home


@pytest.mark.asyncio
async def test_hub_mutations_are_refused_not_routed_to_the_hub(hermes_home):
    runner = _make_runner()

    for sub in ("search", "browse", "inspect", "install my-skill", "audit"):
        out = await runner._handle_skills_command(_make_event(f"/skills {sub}"))
        assert out is not None
        assert "Unknown /skills subcommand on this platform" in out, sub
        assert "Search/install are CLI-only" in out, sub


@pytest.mark.asyncio
async def test_review_subcommands_answer_with_pending_state(hermes_home):
    runner = _make_runner()

    out = await runner._handle_skills_command(_make_event("/skills pending"))
    assert out is not None
    assert "CLI-only" not in out
