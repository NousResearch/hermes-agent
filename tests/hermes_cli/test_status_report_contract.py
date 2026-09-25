"""Contract: the CLI, gateway and TUI /status renderers report the SAME common facts.

Every surface renders through ``hermes_cli.status_report.build_status_fields``; each keeps
its own header, labels (the gateway is i18n) and extras. This test feeds the three real
renderers the same session facts and asserts every common value appears in each output —
a contract between one builder and three renderers, not a snapshot of any layout.
"""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionEntry, SessionSource
from hermes_cli.cli_session_mixin import CLISessionMixin

SESSION_ID = "status-contract-7f3a"
MODEL = "vendor/distinctive-model-9000"
PROVIDER = "distinctive-provider"
TITLE = "Distinctive status title"
CREATED = datetime(2031, 3, 14, 15, 9)
UPDATED = datetime(2031, 3, 15, 1, 5)
TOKENS = 1_234_567
HOME = "~/.hermes-status-contract"

COMMON_VALUES = (
    SESSION_ID, MODEL, PROVIDER, TITLE,
    CREATED.strftime("%Y-%m-%d %H:%M"), UPDATED.strftime("%Y-%m-%d %H:%M"), f"{TOKENS:,}",
)


def _meta() -> dict:
    return {"title": TITLE, "started_at": CREATED.timestamp(), "updated_at": UPDATED.timestamp()}


def _agent():
    return SimpleNamespace(model=MODEL, provider=PROVIDER, session_total_tokens=TOKENS, reasoning_config=None)


def _render_cli() -> str:
    db = MagicMock()
    db.get_session.return_value = _meta()
    rendered: list[str] = []
    cli = SimpleNamespace(
        _session_db=db, session_id=SESSION_ID, session_start=datetime(2000, 1, 1), agent=_agent(),
        provider=PROVIDER, model=MODEL, _agent_running=True, reasoning_config=None, show_reasoning=None,
        session_key="", _get_status_bar_snapshot=lambda: {},
        _console_print=lambda text, **_kw: rendered.append(text),
    )
    CLISessionMixin._show_session_status(cli)
    return "\n".join(rendered)


def _render_gateway() -> str:
    from gateway.run import GatewayRunner

    source = SessionSource(platform=Platform.TELEGRAM, user_id="u1", chat_id="c1", user_name="t", chat_type="dm")
    entry = SessionEntry(session_key="telegram:c1", session_id=SESSION_ID, created_at=CREATED, updated_at=UPDATED,
                         platform=Platform.TELEGRAM, chat_type="dm", total_tokens=0)
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._running_agents = {"telegram:c1": _agent()}
    runner._queue_depth = lambda *_a, **_k: 0
    runner._run_in_executor_with_context = AsyncMock(return_value=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = entry
    runner._session_db = None
    # Token totals come from the SQLite row (never from SessionEntry); model route from the live agent.
    runner._status_session_db_facts = AsyncMock(return_value=(TITLE, {}, TOKENS, {}))
    event = MessageEvent(text="/status", source=source, message_id="m1")
    return asyncio.run(runner._handle_status_command(event))


def _render_tui() -> str:
    from tui_gateway import server

    class _DB:
        def get_session(self, key):
            return _meta() if key == SESSION_ID else {}

    session = {
        "agent": _agent(), "session_key": SESSION_ID, "history": [], "history_lock": threading.Lock(),
        "history_version": 0, "running": True, "attached_images": [], "image_counter": 0, "cols": 80,
        "slash_worker": None, "show_reasoning": False, "tool_progress_mode": "all",
    }
    server._sessions["status-contract-sid"] = session
    try:
        with patch.object(server, "_get_db", lambda: _DB()):
            resp = server.handle_request(
                {"id": "1", "method": "session.status", "params": {"session_id": "status-contract-sid"}})
    finally:
        server._sessions.pop("status-contract-sid", None)
    return resp["result"]["output"]


def test_three_status_surfaces_report_the_same_common_fields():
    with patch("hermes_constants.display_hermes_home", return_value=HOME), \
         patch("tools.approval_context._get_approval_mode", side_effect=RuntimeError("n/a")):
        outputs = {"cli": _render_cli(), "gateway": _render_gateway(), "tui": _render_tui()}

    for surface, text in outputs.items():
        for value in COMMON_VALUES:
            assert value in text, f"{surface} /status lost {value!r}:\n{text}"
    # Path is English-only (the gateway catalog has no path line); running flag is per-surface wording.
    assert HOME in outputs["cli"] and HOME in outputs["tui"]
    assert "Agent Running: Yes" in outputs["cli"] and "Agent Running: Yes" in outputs["tui"]
    assert "Yes" in outputs["gateway"]
