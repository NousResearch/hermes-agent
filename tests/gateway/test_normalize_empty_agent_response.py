"""Unit tests for persistence-failure-aware messaging in
``_normalize_empty_agent_response``.

When a turn is stopped because session persistence failed (SQLite lock
contention, disk exhaustion, ...), the user must NOT be told to /reset —
that destroys their conversation context and does nothing to fix storage.
They must also never see 'The request failed: None' when the gateway result
dict carries an explicit ``error: None``.
"""

import pytest

from gateway.run import _normalize_empty_agent_response


class TestPersistenceFailureRecoveryMessage:
    """Failed turns whose failure_reason marks a session-persistence
    failure get a dedicated recovery message: reassure the user their
    history is protected, tell them to resend — never suggest /reset."""

    def test_locked_persistence_failure_gets_recovery_message(self):
        agent_result = {
            "final_response": "",
            "failed": True,
            "failure_reason": "session_persistence_failed:locked",
            "error": "session storage was locked by another writer",
            "api_calls": 2,
        }

        response = _normalize_empty_agent_response(agent_result, "", history_len=10)

        assert "send it again" in response.lower()
        assert "/reset" not in response
        assert "unknown error" not in response.lower()

    def test_disk_persistence_failure_mentions_disk(self):
        agent_result = {
            "final_response": "",
            "failed": True,
            "failure_reason": "session_persistence_failed:disk",
            "error": "session storage write failed: disk full",
            "api_calls": 1,
        }

        response = _normalize_empty_agent_response(agent_result, "", history_len=10)

        assert "disk" in response.lower()
        assert "/reset" not in response
        assert "unknown error" not in response.lower()

    def test_unknown_cause_persistence_failure_still_avoids_reset(self):
        agent_result = {
            "final_response": "",
            "failed": True,
            "failure_reason": "session_persistence_failed:unknown",
            "error": "session storage failure",
            "api_calls": 1,
        }

        response = _normalize_empty_agent_response(agent_result, "", history_len=10)

        assert "/reset" not in response
        assert "send it again" in response.lower()

    def test_legacy_shape_error_text_mentioning_session_storage(self):
        """Legacy failed results carry no failure_reason but an error text
        naming session storage — they must get the same recovery message."""
        agent_result = {
            "final_response": "",
            "failed": True,
            "error": "turn stopped: session storage unavailable",
            "api_calls": 1,
        }

        response = _normalize_empty_agent_response(agent_result, "", history_len=10)

        assert "/reset" not in response
        assert "send it again" in response.lower()


class TestExplicitNoneErrorIsNoneSafe:
    """The gateway result dict is built with ``'error': holder.get('error')``
    and can carry an EXPLICIT None, which bypasses dict.get defaults."""

    def test_explicit_none_error_never_renders_none(self):
        agent_result = {
            "final_response": "",
            "failed": True,
            "error": None,
            "api_calls": 1,
        }

        response = _normalize_empty_agent_response(agent_result, "", history_len=10)

        assert "None" not in response
        # The generic failure copy names the recovery commands; the detail stays in the log.
        assert "/retry" in response and "/new" in response


class TestGenericFailureRegression:
    """Non-persistence failures get a plain reply that names /retry and /new; the raw exception
    text never reaches the chat (it goes to the gateway log)."""

    def test_provider_error_hides_raw_detail_and_names_retry(self):
        agent_result = {
            "final_response": "",
            "failed": True,
            "error": "ValueError: provider exploded {\"code\": 502}",
            "api_calls": 1,
        }

        response = _normalize_empty_agent_response(agent_result, "", history_len=10)

        assert "provider exploded" not in response and "ValueError" not in response
        assert "/retry" in response and "/new" in response
        assert "hermes logs" in response

    def test_partial_turn_hides_provider_envelope_and_names_retry(self):
        raw = "API call failed after 3 retries: HTTP 500 internal server error"
        agent_result = {"final_response": "", "partial": True, "error": raw, "api_calls": 2}

        response = _normalize_empty_agent_response(agent_result, "", history_len=10)

        assert "HTTP 500" not in response
        assert "/retry" in response and "/compress" in response

    def test_partial_turn_keeps_curated_loop_text(self):
        curated = "Response truncated due to output length limit"
        agent_result = {"final_response": "", "partial": True, "error": curated, "api_calls": 2}

        response = _normalize_empty_agent_response(agent_result, "", history_len=10)

        assert curated in response and "/retry" in response

    def test_context_failure_branch_unchanged(self):
        agent_result = {
            "final_response": "",
            "failed": True,
            "error": "prompt exceeds context window",
            "api_calls": 1,
        }

        response = _normalize_empty_agent_response(agent_result, "", history_len=60)

        assert "too long" in response
        assert "/compress" in response and "/new" in response


class TestNonempty400EnvelopeOverflowReply:
    """Failed turns that already carry the HTTP 400 envelope as ``final_response`` must still get
    the session-too-large rewrite (chat sanitizers would otherwise turn the envelope into a generic
    'provider failed' reply); every other failure with text keeps that text."""

    _ENVELOPE = 'HTTP 400: {"object":"error","model":"deepseek-v4-flash"}'

    def _failed(self, text):
        return {"final_response": text, "failed": True, "error": text, "api_calls": 1}

    def test_long_history_rewrites_envelope_to_session_too_large(self):
        response = _normalize_empty_agent_response(
            self._failed(self._ENVELOPE), self._ENVELOPE, history_len=138,
        )
        assert "too long" in response.lower()
        assert "/compress" in response
        assert self._ENVELOPE not in response

    @pytest.mark.parametrize("text", [
        "Billing or credits exhausted: HTTP 429: You exceeded your current quota",
        "API call failed after 3 retries: HTTP 429 rate limit exceeded",
        "HTTP 401: invalid authentication token",
    ])
    def test_non_overflow_failure_keeps_its_own_text(self, text):
        assert _normalize_empty_agent_response(self._failed(text), text, history_len=138) == text

    def test_curated_overflow_text_from_the_agent_survives(self):
        text = "Context compression timed out without reducing this conversation. No messages were dropped."
        result = {**self._failed(text), "compression_exhausted": True}
        assert _normalize_empty_agent_response(result, text, history_len=138) == text


class TestSystemInterruptNotice:
    """A turn the SYSTEM stopped (lease loss, watchdog) with no visible reply must tell the user who
    stopped it; a human stop and the gateway's own shutdown stay silent here (#84207, #144503)."""

    @staticmethod
    def _interrupted(reason, response=""):
        return {"final_response": response, "interrupted": True, "api_calls": 1, "turn_exit_reason": reason}

    def test_lease_lost_with_no_reply_gets_a_notice_naming_the_issuer(self):
        result = self._interrupted("interrupted_by_system(session_turn_lease_lost)")
        response = _normalize_empty_agent_response(result, "", history_len=10)
        assert "session turn lease lost" in response
        assert "send a message" in response.lower()

    def test_bare_operation_interrupted_placeholder_gets_the_notice(self):
        result = self._interrupted("interrupted_by_system(turn_liveness_watchdog)", "Operation interrupted.")
        response = _normalize_empty_agent_response(result, "Operation interrupted.", history_len=10)
        assert "turn liveness watchdog" in response

    def test_user_stop_stays_silent(self):
        assert _normalize_empty_agent_response(self._interrupted("interrupted_by_user"), "", history_len=10) == ""

    def test_lease_lost_during_model_call_gets_the_notice(self):
        # The waiting-for-model sentinel is stripped before send, so without the notice this is silence.
        sentinel = "Operation interrupted: waiting for model response (12.0s elapsed)."
        result = self._interrupted("interrupted_during_api_call(session_turn_lease_lost)", sentinel)
        response = _normalize_empty_agent_response(result, sentinel, history_len=10)
        assert "session turn lease lost" in response

    def test_user_stop_during_model_call_stays_unchanged(self):
        sentinel = "Operation interrupted: waiting for model response (3.0s elapsed)."
        result = self._interrupted("interrupted_during_api_call", sentinel)
        assert _normalize_empty_agent_response(result, sentinel, history_len=10) == sentinel

    def test_gateway_shutdown_stays_silent(self):
        result = self._interrupted("interrupted_by_system(gateway_shutdown)")
        assert _normalize_empty_agent_response(result, "", history_len=10) == ""

    def test_real_reply_under_system_interrupt_is_unchanged(self):
        text = "Here is what I found before the stop."
        result = self._interrupted("interrupted_by_system(session_turn_lease_lost)", text)
        assert _normalize_empty_agent_response(result, text, history_len=10) == text


@pytest.mark.asyncio
async def test_lease_lost_turn_with_no_queued_message_delivers_the_notice(monkeypatch):
    """End to end through the gateway: the system-interrupted turn's notice reaches ``adapter.send``."""
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.run_turn_runner import TurnRunner
    from gateway.session_authority import initialize_session_authority
    from plugins.platforms.discord.adapter import DiscordAdapter
    from tests.waiting import async_wait_until

    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "42")
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fixture-token", typing_indicator=False))
    sent = []

    async def send(chat_id, content, reply_to=None, metadata=None):
        sent.append(content)
        return SendResult(success=True, message_id="s1")

    monkeypatch.setattr(adapter, "send", send)
    adapter.gateway_runner = runner
    runner.adapters = {Platform.DISCORD: adapter}
    await initialize_session_authority(runner, profile_id="default", instance_id="owner")
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")

    def run_sync(self):
        result = {
            "final_response": "", "messages": [], "completed": False, "interrupted": True, "api_calls": 1,
            "turn_exit_reason": "interrupted_by_system(session_turn_lease_lost)",
            "interrupt_message": "Session turn lease lost; stopping to protect the transcript.",
        }
        self._ctx.result_holder[0] = result
        return result

    monkeypatch.setattr(TurnRunner, "run_sync", run_sync)
    # run_sync is faked, so the real agent module is never used; importing it runs the interrupted-pull
    # repair, which reads the checkout's git dir (outside the isolated home in a linked worktree).
    import sys
    import types
    monkeypatch.setitem(sys.modules, "run_agent", sys.modules.get("run_agent") or types.SimpleNamespace(AIAgent=object))
    await adapter.handle_message(MessageEvent(text="check the CI run", source=source, message_id="A"))
    key = runner.session_store._generate_session_key(source)
    await async_wait_until(lambda: not runner._is_session_running(key) and not adapter._background_tasks,
                           what="the turn to finish")
    assert any("session turn lease lost" in str(text) for text in sent), sent
