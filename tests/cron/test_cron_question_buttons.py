"""Cron reports deliver ``<question>`` blocks as inline buttons (#107138).

A scheduled run cannot call ``clarify`` (it blocks awaiting input), so decisions
at the end of a report used to arrive as dead text. The delivery layer now parses
the documented markup, strips it from the body, and hands the questions to the
platform adapter — which renders buttons wherever the platform supports them. The
run never blocks, and the tap re-enters the conversation as a user turn.

Two invariants these tests pin, both of which have bitten real delivery code:

* the parser is LOSSLESS — a block that does not match the contract stays in the
  text, so a malformed block can never swallow part of a report;
* a lane that cannot render buttons keeps the questions INLINE instead of
  stripping them into nothing.
"""

from __future__ import annotations

import asyncio
import sys
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cron import questions as questions_mod  # noqa: E402
from cron.questions import (  # noqa: E402
    MAX_OPTIONS,
    Question,
    QuestionOption,
    build_callback_data,
    claim_answer,
    forget_questions,
    get_question,
    parse_callback_data,
    parse_questions,
    record_questions,
    to_payload,
)
from cron.scheduler import _deliver_result  # noqa: E402
from gateway.config import Platform, PlatformConfig  # noqa: E402

REPORT = "Nightly report\n- 2 PRs merged\n"
BLOCK = (
    "<question>\n"
    "Merge PR #1827?\n"
    "- \u2705 Yes (Recommended)\n"
    "- \u23f8\ufe0f Not yet\n"
    "</question>\n"
)


@pytest.fixture
def store(monkeypatch, tmp_path):
    """Pending-question store in a temp profile, never the developer's own."""
    monkeypatch.setattr(questions_mod, "QUESTIONS_FILE", tmp_path / "cron" / "questions.db")
    return questions_mod


# ---------------------------------------------------------------------------
# The markup contract
# ---------------------------------------------------------------------------

class TestParseQuestions:
    def test_block_becomes_a_question_and_leaves_the_body(self):
        body, questions = parse_questions(REPORT + BLOCK)

        assert questions == [Question(
            text="Merge PR #1827?",
            options=[
                QuestionOption(label="\u2705 Yes", recommended=True),
                QuestionOption(label="\u23f8\ufe0f Not yet", recommended=False),
            ],
        )]
        assert questions[0].options[0].recommended is True
        assert "<question>" not in body
        assert "Merge PR #1827?" not in body
        # The report itself survives untouched.
        assert "2 PRs merged" in body

    @pytest.mark.parametrize("text", [
        "no markup at all",
        "<question>\nOnly a question, no options\n</question>",
        "<question>\nPick one\n- a\nand some prose\n- b\n</question>",
        "<question>\nUnterminated\n- a\n- b\n",
        "<question>\nToo many\n" + "".join(f"- option {i}\n" for i in range(MAX_OPTIONS + 1)) + "</question>",
    ])
    def test_unparseable_input_is_returned_verbatim(self, text):
        """A block that is not the documented shape must never disappear."""
        body, questions = parse_questions(text)

        assert questions == []
        assert body == text

    def test_multiple_blocks_are_all_extracted(self):
        body, questions = parse_questions(
            REPORT + BLOCK + "\n<question>\nShip it?\n- yes\n- no\n</question>\n")

        assert [q.text for q in questions] == ["Merge PR #1827?", "Ship it?"]
        assert "<question>" not in body

    def test_button_rows_carry_a_token_per_question_not_per_report(self):
        _, questions = parse_questions(BLOCK + "\n<question>\nShip it?\n- yes\n- no\n</question>\n")
        payload = to_payload(questions, ["aaaa", "bbbb"])

        rows = questions_mod.button_rows(payload)

        assert rows[0][0]["callback_data"] == build_callback_data("aaaa", 0)
        # Second question's options are numbered so a row is never ambiguous.
        assert rows[2][0]["label"].startswith("2. ")
        assert rows[2][0]["callback_data"] == build_callback_data("bbbb", 0)

    def test_callback_data_round_trips_within_telegram_budget(self):
        token = "0123456789ab"
        data = build_callback_data(token, 3)

        assert parse_callback_data(data) == (token, 3)
        assert len(data.encode("utf-8")) <= questions_mod.MAX_CALLBACK_DATA_BYTES
        assert parse_callback_data("cl:" + token + ":0") is None
        assert parse_callback_data("cq:" + token) is None


# ---------------------------------------------------------------------------
# The pending-question store
# ---------------------------------------------------------------------------

class TestPendingQuestionStore:
    def _question(self):
        return [Question(text="Merge?", options=[QuestionOption("yes"), QuestionOption("no")])]

    def test_answer_is_recorded_once_and_replays_are_reported(self, store):
        (token,) = record_questions("job1", "telegram", "-100", self._question())

        first = claim_answer(token, 0)
        replay = claim_answer(token, 1)

        assert first["status"] == "answered"
        assert first["answer_text"] == "yes"
        assert first["job_id"] == "job1"
        assert replay["status"] == "already_answered"
        # First write wins: the replay must not overwrite the recorded answer.
        assert replay["answer_text"] == "yes"
        assert get_question(token)["answer_text"] == "yes"

    def test_unknown_token_and_out_of_range_option_are_reported_not_raised(self, store):
        (token,) = record_questions("job1", "telegram", "-100", self._question())

        assert claim_answer("deadbeefdead", 0)["status"] == "unknown"
        assert claim_answer(token, 7)["status"] == "invalid_option"
        assert get_question(token)["answered_at"] is None

    def test_forget_questions_drops_buttons_that_never_shipped(self, store):
        tokens = record_questions("job1", "telegram", "-100", self._question())

        assert forget_questions(tokens) == 1
        assert get_question(tokens[0]) is None

    def test_stale_unanswered_rows_are_pruned_but_fresh_ones_survive(self, store, monkeypatch):
        """A job whose questions are never answered must not grow a row per run forever."""
        from datetime import timedelta

        from hermes_time import now as real_now

        monkeypatch.setattr(
            store, "_hermes_now", lambda: real_now() - timedelta(days=store.MAX_PENDING_AGE_DAYS + 1))
        (stale,) = record_questions("job1", "telegram", "-100", self._question())
        monkeypatch.setattr(store, "_hermes_now", real_now)
        (fresh,) = record_questions("job1", "telegram", "-100", self._question())

        assert get_question(stale) is None
        assert get_question(fresh) is not None


# ---------------------------------------------------------------------------
# The delivery lane
# ---------------------------------------------------------------------------

CHAT_ID = "-1001234567890"


def _job():
    return {
        "id": "qbjob1",
        "name": "Nightly",
        "deliver": "origin",
        "origin": {"platform": "telegram", "chat_id": CHAT_ID},
    }


def _run_delivery(content, *, live=True, cron_cfg=None):
    """Drive ``_deliver_result`` over the live lane with a stubbed router.

    Returns ``(body_sent, question_calls, standalone_calls)``.
    """
    adapter = MagicMock()
    question_calls = []

    async def _send_cron_questions(chat_id, payload, metadata=None):
        question_calls.append({"chat_id": chat_id, "payload": payload, "metadata": metadata})
        return {"success": True, "message_id": "42"}

    adapter.send_cron_questions = _send_cron_questions

    loop = MagicMock()
    loop.is_running.return_value = True
    # live=False: no running gateway loop, so the standalone sender lane is the only one.
    loop = loop if live else None

    def fake_schedule(coro, _loop):
        future = Future()
        try:
            future.set_result(asyncio.run(coro))
        except BaseException as exc:  # noqa: BLE001
            future.set_exception(exc)
        return future

    bodies = []
    router = MagicMock()

    async def _deliver_to_platform(target, text, metadata):
        bodies.append(text)
        return {"success": True, "message_id": "41"}

    router._deliver_to_platform = _deliver_to_platform

    standalone = []

    async def _fake_send_to_platform(platform, pconfig, chat_id, text, **kwargs):
        standalone.append(text)
        return {}

    config = MagicMock()
    config.platforms = {Platform.TELEGRAM: PlatformConfig(enabled=True)}
    config.get_home_channel = lambda platform: None

    with patch("gateway.config.load_gateway_config", return_value=config), \
            patch("cron.scheduler.load_config",
                  return_value={"cron": {"wrap_response": False, **(cron_cfg or {})}}), \
            patch("gateway.delivery.DeliveryRouter", return_value=router), \
            patch("tools.send_message_tool._send_to_platform", _fake_send_to_platform), \
            patch("asyncio.run_coroutine_threadsafe", side_effect=fake_schedule), \
            patch("cron.scheduler_delivery._record_delivery_verification"):
        error = _deliver_result(
            _job(), content, adapters={Platform.TELEGRAM: adapter}, loop=loop)
    assert error is None
    return (bodies[0] if bodies else None), question_calls, standalone


class TestDeliveryLane:
    def test_questions_become_buttons_and_leave_the_body(self, store):
        body, question_calls, _ = _run_delivery(REPORT + BLOCK)

        assert "<question>" not in body
        assert "Merge PR #1827?" not in body
        assert len(question_calls) == 1
        payload = question_calls[0]["payload"]
        assert [item.question for item in payload] == ["Merge PR #1827?"]
        assert payload[0].options == ["\u2705 Yes", "\u23f8\ufe0f Not yet"]
        assert question_calls[0]["chat_id"] == CHAT_ID
        # The delivered button is answerable: its token is in the durable store.
        assert get_question(payload[0].token)["job_id"] == "qbjob1"

    def test_reports_without_markup_are_delivered_byte_identically(self, store):
        body, question_calls, _ = _run_delivery(REPORT)

        assert body == REPORT.rstrip("\n")
        assert question_calls == []

    def test_lane_without_buttons_keeps_the_questions_in_the_body(self, store):
        """Standalone senders have no reply_markup path: keep the text, lose nothing."""
        body, question_calls, standalone = _run_delivery(REPORT + BLOCK, live=False)

        assert body is None  # the live lane never ran
        assert question_calls == []
        assert standalone and "<question>" in standalone[0]
        assert "Merge PR #1827?" in standalone[0]

    def test_question_buttons_can_be_switched_off(self, store):
        body, question_calls, _ = _run_delivery(
            REPORT + BLOCK, cron_cfg={"question_buttons": False})

        assert "<question>" in body
        assert question_calls == []
