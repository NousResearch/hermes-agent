"""Telegram Kanban blocker-ask buttons: the ``kb:`` callback namespace.

A press on an ask's inline button must be equivalent to the operator typing the
decision in chat — same authorization gate as the approval buttons, the same verb
the ``hermes kanban`` CLI applies, and the pressed message shows the outcome. The
chosen option travels only as ``kb:<task_id>:<index>`` (Telegram caps
``callback_data`` at 64 bytes), so it is resolved from the card's own stored
options; a press that does not resolve changes nothing.
"""

import contextlib
import os
import re
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from gateway.config import PlatformConfig  # noqa: E402
from hermes_cli import kanban_db as kb  # noqa: E402
from hermes_cli import kanban_db_connect as kbc  # noqa: E402
from plugins.platforms.telegram import kanban_buttons as kb_buttons  # noqa: E402
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402

DO_IT = kb_buttons.STANDARD_BASE
PARK_IT = kb_buttons.STANDARD_BASE + 1
DROP_IT = kb_buttons.STANDARD_BASE + 2

OPTIONS = [
    {"label": "Use the TVDB id", "value": "tvdb"},
    {"label": "Skip the rename", "value": "skip"},
]


# ===========================================================================
# helpers
# ===========================================================================

def _make_adapter():
    """TelegramAdapter with mocked internals (mirrors the approval-button tests)."""
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra={}))
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


@contextlib.contextmanager
def _board_conn():
    conn = kbc.connect(board="homelab")
    try:
        yield conn
    finally:
        conn.close()


def _new_card(*, options=None, reason="the worker needs your decision", blocked=True):
    """A freshly created card in ``homelab``, blocked with ``reason``/``options``."""
    with _board_conn() as conn:
        tid = kb.create_task(conn, title="Button test card", assignee="partner",
                             initial_status="running")
        if blocked:
            assert kb.block_task(conn, tid, reason=reason, kind="needs_input",
                                 options=options)
        return tid


def _status(tid):
    with _board_conn() as conn:
        task = kb.get_task(conn, tid)
        return task.status if task else None


def _result(tid):
    with _board_conn() as conn:
        task = kb.get_task(conn, tid)
        return task.result if task else None


def _comments(tid):
    with _board_conn() as conn:
        return [(c.author, c.body) for c in kb.list_comments(conn, tid)]


def _event(kind, tid):
    with _board_conn() as conn:
        matches = [e for e in kb.list_events(conn, tid) if e.kind == kind]
        return matches[-1].payload if matches else None


def _query(data, *, user_id="12345", chat_id=12345, name="Operator"):
    query = AsyncMock()
    query.data = data
    query.message = MagicMock()
    query.message.chat_id = chat_id
    query.message.chat.type = "private"
    query.from_user = MagicMock()
    query.from_user.id = user_id
    query.from_user.first_name = name
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    return query


async def _press(adapter, data, *, user_id="12345", allowed="*", name="Operator"):
    """Drive a callback all the way through the adapter's prefix dispatch table."""
    query = _query(data, user_id=user_id, name=name)
    update = MagicMock()
    update.callback_query = query
    with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": allowed}, clear=False):
        await adapter._handle_callback_query(update, MagicMock())
    return query


def _answer_text(query):
    return query.answer.call_args[1]["text"]


def _edit_text(query):
    return query.edit_message_text.call_args[1]["text"]


@pytest.fixture
def homelab_board(tmp_path, monkeypatch):
    """An isolated kanban root with ``homelab`` as the current board."""
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    kb.create_board("homelab", name="Homelab")
    kb.set_current_board("homelab")
    assert kb.get_current_board() == "homelab"
    return "homelab"


# ===========================================================================
# index space / callback parsing (pure)
# ===========================================================================

class TestCallbackEncoding:
    def test_parses_task_id_and_index(self):
        assert kb_buttons.parse_callback_data("kb:t_1a2b:3") == ("t_1a2b", 3)

    @pytest.mark.parametrize("data", [
        "", "kb:", "kb:t_1a2b", "kb::3", "kb:t_1a2b:x", "ea:once:5", "update_prompt:y",
    ])
    def test_rejects_anything_else(self, data):
        assert kb_buttons.parse_callback_data(data) is None

    def test_standard_row_is_a_fixed_index_range(self):
        for offset, label in enumerate(("Do it", "Park it", "Drop it")):
            choice = kb_buttons.resolve_choice(OPTIONS, kb_buttons.STANDARD_BASE + offset)
            assert choice["label"] == label and choice["declared"] is False

    def test_declared_options_resolve_in_order(self):
        choice = kb_buttons.resolve_choice(OPTIONS, 1)
        assert choice == {"label": "Skip the rename", "value": "skip",
                          "verb": kb_buttons.VERB_ANSWER, "declared": True}

    def test_standard_row_appended_after_declared_options_also_resolves(self):
        # A renderer with no fixed base: 2 declared options -> standard row at 2,3,4.
        assert kb_buttons.resolve_choice(OPTIONS, 2)["label"] == "Do it"
        assert kb_buttons.resolve_choice(OPTIONS, 3)["label"] == "Park it"
        assert kb_buttons.resolve_choice(OPTIONS, 4)["label"] == "Drop it"

    @pytest.mark.parametrize("index", [5, 7, 99, -1])
    def test_unknown_index_resolves_to_nothing(self, index):
        assert kb_buttons.resolve_choice(OPTIONS, index) is None


# ===========================================================================
# authorized presses: the three standard verbs + a task-declared option
# ===========================================================================

class TestAuthorizedPresses:
    @pytest.mark.asyncio
    async def test_do_it_unblocks_and_logs_the_decision(self, homelab_board):
        tid = _new_card(options=OPTIONS)
        adapter = _make_adapter()

        query = await _press(adapter, f"kb:{tid}:{DO_IT}")

        assert _status(tid) == "ready"
        (author, body), = _comments(tid)
        assert author == "Operator"
        assert re.fullmatch(
            r"USER CONFIRMED \(\d{4}-\d{2}-\d{2} \d{2}:\d{2}( \S+)?, "
            r"telegram button\): Do it", body), body
        assert "Do it" in _answer_text(query)
        assert _edit_text(query).startswith("✅ Do it")
        assert "released" in _edit_text(query)
        assert query.edit_message_text.call_args[1]["reply_markup"] is None

    @pytest.mark.asyncio
    async def test_park_it_schedules_with_the_reason(self, homelab_board):
        tid = _new_card(options=OPTIONS)
        adapter = _make_adapter()

        query = await _press(adapter, f"kb:{tid}:{PARK_IT}")

        assert _status(tid) == "scheduled"
        (_, body), = _comments(tid)
        assert body.endswith("telegram button): Park it")
        assert "Park it" in (_event("scheduled", tid) or {}).get("reason", "")
        assert "parked" in _edit_text(query)

    @pytest.mark.asyncio
    async def test_drop_it_completes_as_abandoned(self, homelab_board):
        tid = _new_card(options=OPTIONS)
        adapter = _make_adapter()

        query = await _press(adapter, f"kb:{tid}:{DROP_IT}")

        assert _status(tid) == "done"
        assert _result(tid).startswith("abandoned:")
        (_, body), = _comments(tid)
        assert body.endswith("telegram button): Drop it")
        assert "abandoned" in _edit_text(query)

    @pytest.mark.asyncio
    async def test_declared_option_is_the_operators_answer(self, homelab_board):
        tid = _new_card(options=OPTIONS)
        adapter = _make_adapter()

        query = await _press(adapter, f"kb:{tid}:1")

        assert _status(tid) == "ready"
        (_, body), = _comments(tid)
        assert body.startswith("USER CONFIRMED (")
        assert "telegram button): Skip the rename" in body
        assert "ANSWER: skip" in body          # the machine-readable half
        assert "Skip the rename" in _answer_text(query)
        assert "released" in _edit_text(query)

    @pytest.mark.asyncio
    async def test_prose_options_line_fallback(self, homelab_board):
        """A card blocked by hand (no declared options) still gets a working press."""
        tid = _new_card(
            options=None,
            reason="Which id should the rename use?\nOPTIONS: a=Use the TVDB id | b=Skip")
        adapter = _make_adapter()

        await _press(adapter, f"kb:{tid}:0")

        assert _status(tid) == "ready"
        (_, body), = _comments(tid)
        assert "telegram button): Use the TVDB id" in body
        assert "ANSWER: a" in body

    @pytest.mark.asyncio
    async def test_do_it_on_a_review_staged_card_completes_it(self, homelab_board):
        """Nothing to unblock — the card is staged for the human decision."""
        with _board_conn() as conn:
            tid = kb.create_task(conn, title="staged", assignee="partner",
                                 initial_status="running")
            ok, why = kb.request_review(conn, tid, summary="ready for you",
                                        with_reason=True)
            assert ok, why
        assert _status(tid) == "review"
        adapter = _make_adapter()

        await _press(adapter, f"kb:{tid}:{DO_IT}")

        assert _status(tid) == "done"
        assert "approved" in (_result(tid) or "")

    @pytest.mark.asyncio
    async def test_card_on_a_non_current_board_is_found(self, homelab_board):
        kb.create_board("side", name="Side")
        with contextlib.closing(kbc.connect(board="side")) as conn:
            tid = kb.create_task(conn, title="off-board card", assignee="partner",
                                 initial_status="running")
            assert kb.block_task(conn, tid, reason="decision?", kind="needs_input",
                                 options=OPTIONS)
        assert kb.get_current_board() == "homelab"
        adapter = _make_adapter()

        await _press(adapter, f"kb:{tid}:{DO_IT}")

        with contextlib.closing(kbc.connect(board="side")) as conn:
            assert kb.get_task(conn, tid).status == "ready"


# ===========================================================================
# refusals and staleness
# ===========================================================================

class TestRefusals:
    @pytest.mark.asyncio
    async def test_unauthorized_press_has_no_board_effect(self, homelab_board):
        tid = _new_card(options=OPTIONS)
        adapter = _make_adapter()

        query = await _press(adapter, f"kb:{tid}:{DO_IT}", user_id="222", allowed="111")

        assert "not authorized" in _answer_text(query).lower()
        assert _status(tid) == "blocked"
        assert _comments(tid) == []
        query.edit_message_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_second_press_is_answered_and_not_re_applied(self, homelab_board):
        tid = _new_card(options=OPTIONS)
        adapter = _make_adapter()

        await _press(adapter, f"kb:{tid}:{DO_IT}")
        assert _status(tid) == "ready"
        before = _comments(tid)

        query = await _press(adapter, f"kb:{tid}:{DO_IT}")

        assert _status(tid) == "ready"          # unchanged
        assert _comments(tid) == before         # no duplicate comment
        assert "already" in _answer_text(query)

    @pytest.mark.asyncio
    async def test_unknown_index_changes_nothing(self, homelab_board):
        tid = _new_card(options=OPTIONS)
        adapter = _make_adapter()

        query = await _press(adapter, f"kb:{tid}:7")

        assert "no longer matches" in _answer_text(query)
        assert _status(tid) == "blocked"
        assert _comments(tid) == []

    @pytest.mark.asyncio
    async def test_press_on_a_card_that_is_not_waiting_changes_nothing(self, homelab_board):
        """A card that is not blocked/scheduled/review has nothing to apply."""
        tid = _new_card(blocked=False)
        adapter = _make_adapter()

        query = await _press(adapter, f"kb:{tid}:{DO_IT}")

        assert "nothing applied" in _answer_text(query)
        assert _status(tid) == "ready"      # create_task normalises an unclaimed run
        assert _comments(tid) == []

    @pytest.mark.asyncio
    async def test_unknown_task_is_answered(self, homelab_board):
        adapter = _make_adapter()

        query = await _press(adapter, "kb:t_dead:100")

        assert "not on a board" in _answer_text(query)
        query.edit_message_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_malformed_callback_data_is_answered(self, homelab_board):
        adapter = _make_adapter()

        query = await _press(adapter, "kb:t_1a2b")

        assert "Unrecognised" in _answer_text(query)
        query.edit_message_text.assert_not_called()


# ===========================================================================
# the adapter's dispatch table reaches the resolver
# ===========================================================================

class TestDispatchWiring:
    @pytest.mark.asyncio
    async def test_kb_prefix_reaches_the_kanban_handler(self, homelab_board, monkeypatch):
        calls = []

        def _spy(callback_data, **kwargs):
            calls.append((callback_data, kwargs))
            return kb_buttons.PressOutcome(True, "Do it", kb_buttons.VERB_DO_IT, "t_1", "b", "ready",
                                           "ok", None)

        monkeypatch.setattr(kb_buttons, "apply_press", _spy)
        adapter = _make_adapter()

        await _press(adapter, "kb:t_1:100")

        assert calls == [("kb:t_1:100", {"pressed_by": "Operator", "author": "Operator"})]

    @pytest.mark.asyncio
    async def test_every_other_prefix_is_untouched(self, homelab_board, monkeypatch):
        adapter = _make_adapter()
        monkeypatch.setattr(kb_buttons, "apply_press",
                            lambda *a, **k: pytest.fail("kb handler must not fire"))

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as resolve:
            adapter._approval_state[7] = "agent:main:telegram:private:12345"
            await _press(adapter, "ea:once:7")

        resolve.assert_called_once()
