"""``gateway.failure_reply``: one plain message in place of Hermes's own failure copy in chats.

The 23 Nov incident: a front-desk bot in a teammate's Discord thread posted "ChatGPT or Codex
Subscription rate-limited every one of 3 attempts ... Send /retry, or switch models with /model."
The model never ran, so no plugin or prompt rule could reword it. Every test here drives a real
gateway seam (the shaping, error-reply, status-callback and notice paths) with the option set in the
profile's config, and asserts what the chat would receive.
"""

import asyncio
import logging
from types import SimpleNamespace

import pytest

from agent.turn_failure_copy import FAILED_TURN_NOTICE, PARTIAL_FAILED_TURN_NOTICE
from gateway.config import Platform
from gateway.failure_reply import ReplacedFailureText
from gateway.run_turn import GatewayTurnMixin
from gateway.session import SessionSource

PLAIN = "Sorry, something went wrong on my side and I couldn't finish that. I'll get back to you here."
# Verbatim from the teammate's Discord thread (23 Nov 2026).
INCIDENT = (
    "ChatGPT or Codex Subscription rate-limited every one of 3 attempts — its usage limit resets in "
    "~43h. Send /retry, or switch models with /model."
)


def _configure(monkeypatch, **failure_reply):
    config = {"gateway": {"failure_reply": failure_reply}} if failure_reply else {}
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda *a, **k: config)


def _source(chat_type="thread", platform=Platform.DISCORD):
    return SessionSource(platform=platform, chat_id="1234567890", chat_type=chat_type, user_id="teammate")


class _Runner(GatewayTurnMixin):
    def __init__(self):
        self.async_session_store = SimpleNamespace(clear_resume_pending=self._noop)

    async def _noop(self, *_a, **_k):
        return None

    async def _clear_restart_failure_count(self, *_a, **_k):
        return None

    def _silence_allowed_for_source(self, source, display_kind, reply_expected, agent_result):
        from gateway.response_filters import silence_allowed
        return silence_allowed(display_kind, reply_expected)


async def _shape(agent_result, source):
    response, _silent, _messages = await _Runner()._hmwa_shape_agent_response(
        agent_result, source, history=[], session_entry=SimpleNamespace(session_id="s"), session_key=None,
        _quick_key=None, run_generation=0, _run_start_session_id="s", _platform_name=source.platform.value,
        _msg_start_time=0.0)
    return response


def _failed_turn(text=INCIDENT):
    return {"final_response": text, "failed": True, "completed": False, "failure_reason": "rate_limit",
            "api_calls": 3, "messages": []}


# ── 1. the incident message is replaced where the option applies ──────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("chat_type", ["thread", "group", "channel"])
async def test_incident_message_is_replaced_in_a_discord_chat_when_set(monkeypatch, chat_type):
    _configure(monkeypatch, text=PLAIN)
    response = await _shape(_failed_turn(), _source(chat_type))
    assert response == PLAIN
    assert isinstance(response, ReplacedFailureText)


@pytest.mark.asyncio
async def test_the_original_failure_text_is_logged_at_warning_without_the_chat_id(monkeypatch, caplog):
    _configure(monkeypatch, text=PLAIN)
    with caplog.at_level(logging.WARNING, logger="gateway.failure_reply"):
        await _shape(_failed_turn(), _source("thread"))
    logged = "\n".join(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
    assert "rate-limited every one of 3 attempts" in logged
    assert "1234567890" not in logged


@pytest.mark.asyncio
async def test_provider_error_envelope_in_a_final_reply_is_replaced(monkeypatch):
    """The category reply (``_gateway_provider_error_reply``) is not what a team chat sees either."""
    from gateway.run import _sanitize_gateway_final_response

    _configure(monkeypatch, text=PLAIN)
    envelope = "API call failed: HTTP 429 rate limit exceeded, retry after 4hr"
    assert _sanitize_gateway_final_response(Platform.DISCORD, envelope, source=_source("group")) == PLAIN
    # a caller that cannot name the chat keeps today's category reply
    assert _sanitize_gateway_final_response(Platform.DISCORD, envelope) != PLAIN


# ── 2. unset: nothing changes ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("config", [{}, {"text": ""}, {"text": "   "}, {"scope": "all"}],
                         ids=["no section", "empty text", "blank text", "scope without text"])
async def test_unset_option_leaves_the_failure_copy_alone(monkeypatch, config):
    _configure(monkeypatch, **config)
    response = await _shape(_failed_turn(), _source("thread"))
    assert response == INCIDENT
    assert not isinstance(response, ReplacedFailureText)


# ── 3. scope: DMs ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [None, "non_dm"])
async def test_a_dm_keeps_the_failure_copy_under_the_default_scope(monkeypatch, scope):
    _configure(monkeypatch, text=PLAIN, **({"scope": scope} if scope else {}))
    assert await _shape(_failed_turn(), _source("dm")) == INCIDENT


@pytest.mark.asyncio
async def test_a_dm_is_replaced_under_scope_all(monkeypatch):
    _configure(monkeypatch, text=PLAIN, scope="all")
    assert await _shape(_failed_turn(), _source("dm")) == PLAIN


@pytest.mark.asyncio
async def test_a_chat_nothing_classified_is_treated_as_shared(monkeypatch):
    _configure(monkeypatch, text=PLAIN)
    assert await _shape(_failed_turn(), _source("")) == PLAIN


@pytest.mark.asyncio
async def test_a_group_dm_every_participant_shares_is_not_a_dm(monkeypatch):
    _configure(monkeypatch, text=PLAIN)
    source = _source("dm")
    source.shared_session = True
    assert await _shape(_failed_turn(), source) == PLAIN


@pytest.mark.asyncio
async def test_an_unknown_scope_value_falls_back_to_non_dm(monkeypatch):
    _configure(monkeypatch, text=PLAIN, scope="everywhere")
    assert await _shape(_failed_turn(), _source("dm")) == INCIDENT
    assert await _shape(_failed_turn(), _source("thread")) == PLAIN


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", [Platform.LOCAL, Platform.API_SERVER, Platform.WEBHOOK])
async def test_local_and_programmatic_surfaces_keep_the_raw_copy(monkeypatch, platform):
    _configure(monkeypatch, text=PLAIN, scope="all")
    assert await _shape(_failed_turn(), _source("group", platform)) == INCIDENT


# ── 4. a model's own reply is never replaced ──────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("reply", [
    "I made 3 attempts at the export and each was rate-limited by the vendor, so I'll retry at noon.",
    INCIDENT,  # the model quoting the very text back (e.g. summarising an incident) is still its answer
    "Send /retry to the vendor's API and switch models with /model if you like.",
])
async def test_a_normal_model_reply_is_not_replaced(monkeypatch, reply):
    _configure(monkeypatch, text=PLAIN, scope="all")
    response = await _shape({"final_response": reply, "api_calls": 2, "messages": []}, _source("thread"))
    assert response == reply
    assert not isinstance(response, ReplacedFailureText)


@pytest.mark.asyncio
async def test_a_partial_model_answer_that_was_truncated_is_not_replaced(monkeypatch):
    _configure(monkeypatch, text=PLAIN)
    result = {"final_response": "Here are the first three rows: ...", "partial": True, "api_calls": 2, "messages": []}
    assert await _shape(result, _source("thread")) == "Here are the first three rows: ..."


# ── 5a. failed turns that carry no text of their own, and the notice appended after them ──────

@pytest.mark.asyncio
@pytest.mark.parametrize("agent_result", [
    {"final_response": "", "failed": True, "error": "boom", "api_calls": 1, "messages": []},
    {"final_response": "(empty)", "api_calls": 2, "model": "m", "messages": []},
    {"final_response": "", "api_calls": 2, "messages": []},
    {"final_response": "", "interrupted": True, "api_calls": 0, "messages": []},
    {"final_response": "⚠️ Agent turn ended early", "gateway_failure_copy": True, "api_calls": 1, "messages": []},
], ids=["failed, no text", "empty after retries", "work done, no text", "never ran", "runner-written copy"])
async def test_runtime_written_copy_for_a_turn_with_no_answer_is_replaced(monkeypatch, agent_result):
    _configure(monkeypatch, text=PLAIN)
    assert await _shape(dict(agent_result), _source("thread")) == PLAIN


def test_the_failed_turn_notice_is_not_appended_to_the_plain_message(monkeypatch):
    _configure(monkeypatch, text=PLAIN)
    runner = _Runner()
    plain = ReplacedFailureText(PLAIN)
    assert runner._hmwa_add_failed_turn_notice(plain, FAILED_TURN_NOTICE) == PLAIN
    assert runner._hmwa_add_failed_turn_notice(plain, PARTIAL_FAILED_TURN_NOTICE) == PLAIN
    # unconfigured text keeps its notice, as before
    assert runner._hmwa_add_failed_turn_notice(INCIDENT, FAILED_TURN_NOTICE) == f"{INCIDENT}\n\n{FAILED_TURN_NOTICE}"


def test_the_plain_message_carries_no_reasoning_usage_wording_or_compression_text():
    runner = _Runner()
    plain = ReplacedFailureText(PLAIN)
    result = {"last_reasoning": "the provider said 429", "usage_window_message": True, "failure_resets_at": 1}
    assert runner._hmwa_prepend_reasoning(result, plain, _source(), False) == PLAIN
    assert runner._usage_window_chat_wording(result, plain) == PLAIN
    reset = asyncio.run(runner._hmwa_compression_exhaustion_reset(
        {"compression_exhausted": True}, plain, None, None, SimpleNamespace(platform=Platform.API_SERVER)))
    assert reset[0] == PLAIN


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", [True, False])
async def test_an_exception_in_the_turn_gets_the_plain_message_only_when_configured(monkeypatch, configured):
    _configure(monkeypatch, **({"text": PLAIN} if configured else {}))
    runner = _Runner()
    runner._delivery_adapter_for = lambda source: None
    runner._session_state = lambda key: SimpleNamespace(turn=None)
    prepared = SimpleNamespace(history=[], message_text=None)

    class RateLimited(Exception):
        status_code = 429

    reply = await runner._hmwa_agent_error_reply(
        RateLimited("429 Too Many Requests"), SimpleNamespace(), _source("thread"), None, "key", prepared)
    if configured:
        assert reply == PLAIN
    else:
        assert PARTIAL_FAILED_TURN_NOTICE in reply


# ── 5b. status / progress callbacks ───────────────────────────────────────────────────────────

def _deliver_status(source, event_type, message):
    """Drive the production ``_status_callback_sync`` and return what the adapter was handed."""
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext

    delivered = []

    class Transport:
        async def send(self, chat_id, text, metadata=None):
            delivered.append(text)

    async def exercise():
        loop = asyncio.get_running_loop()
        ctx = TurnContext(source=source, user_config={}, _run_still_current=lambda: True, _loop_for_step=loop)
        ctx._status_adapter = Transport()
        ctx._status_chat_id = source.chat_id
        ctx._status_thread_metadata = None
        runner = TurnRunner(None, ctx)

        def emit():
            runner._status_callback_sync(event_type, message)
            asyncio.run_coroutine_threadsafe(asyncio.sleep(0), loop).result(timeout=5)

        await asyncio.to_thread(emit)

    asyncio.run(exercise())
    return delivered


RETRY_CHATTER = "⚠️ Switching to fallback model after repeated provider errors"


def test_terminal_failure_status_never_reaches_a_chat():
    """Already dropped today because the failed turn's final reply carries the message; with the option
    set that final reply is the plain message, so the chat still sees exactly one line."""
    from gateway.warning_notifications import TerminalFailureText

    assert _deliver_status(_source("thread"), "lifecycle",
                           TerminalFailureText("❌ API failed after 3 retries — rate limited")) == []


def test_diagnostic_warnings_are_held_back_from_a_team_chat_when_set(monkeypatch):
    _configure(monkeypatch, text=PLAIN)
    assert _deliver_status(_source("thread"), "warn", RETRY_CHATTER) == []


def test_diagnostic_warnings_are_unchanged_when_unset_or_in_a_dm(monkeypatch):
    _configure(monkeypatch)
    assert _deliver_status(_source("thread"), "warn", RETRY_CHATTER) == [RETRY_CHATTER]
    _configure(monkeypatch, text=PLAIN)
    assert _deliver_status(_source("dm"), "warn", RETRY_CHATTER) == [RETRY_CHATTER]


def test_diagnostic_warnings_are_held_back_in_a_dm_under_scope_all(monkeypatch):
    _configure(monkeypatch, text=PLAIN, scope="all")
    assert _deliver_status(_source("dm"), "warn", RETRY_CHATTER) == []


def test_ordinary_progress_status_is_not_touched(monkeypatch):
    _configure(monkeypatch, text=PLAIN, scope="all")
    assert _deliver_status(_source("thread"), "lifecycle", "🔧 Searching the web") == ["🔧 Searching the web"]


def test_the_status_filter_logs_what_it_held_back(monkeypatch, caplog):
    _configure(monkeypatch, text=PLAIN)
    with caplog.at_level(logging.WARNING, logger="gateway.failure_reply"):
        _deliver_status(_source("thread"), "warn", RETRY_CHATTER)
    assert any("fallback model" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)


# ── 5c. other doors a failure can leave through ───────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("configured", [True, False])
async def test_the_last_resort_turn_error_notice_follows_the_option(monkeypatch, configured):
    from contextlib import nullcontext

    from gateway.platforms.base import BasePlatformAdapter, MessageEvent

    _configure(monkeypatch, **({"text": PLAIN} if configured else {}))
    sent = []

    class Adapter:
        name = "discord"

        def _media_delivery_scope(self, source):
            return nullcontext()

        def warning_text(self, text, *_a, **_k):
            return text

        async def send(self, chat_id, content, metadata=None):
            sent.append(content)

    event = MessageEvent(text="hi", message_id="m", source=_source("thread"))
    await BasePlatformAdapter._notify_turn_error(Adapter(), event, RuntimeError("boom"))
    assert len(sent) == 1
    assert (sent[0] == PLAIN) is configured
