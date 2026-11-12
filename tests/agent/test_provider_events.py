"""Structured provider-health events for API clients (Otto's retry / usage / fallback line) and
the fast-fail for model-locked turns."""
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import provider_events
from agent.error_classifier import classify_api_error
from agent.turn_api_error import settle_unrecovered_error
from agent.turn_retry_state import TurnRetryState


class _Err(Exception):
    def __init__(self, status_code, message, body=None, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.response = SimpleNamespace(headers=headers or {})
        self.body = body


class _Agent:
    log_prefix = ""
    verbose = False
    platform = "api_server"
    _credential_pool = None
    _current_streamed_assistant_text = ""
    _interrupt_requested = False
    _auto_recovery_cycles = 5

    def __init__(self, *, locked=False, base_url="http://127.0.0.1:11434/v1", provider="custom", model="qwen"):
        self._model_locked = locked
        self.base_url = base_url
        self.provider = provider
        self.model = model
        self.events, self.statuses, self.primary_recoveries = [], [], 0
        self.provider_event_callback = self.events.append

    def _has_pending_fallback(self):
        return False

    def _try_activate_fallback(self, **kwargs):
        return False

    def _try_recover_primary_transport(self, *a, **k):
        self.primary_recoveries += 1
        return True

    def _has_content_after_think_block(self, text):
        return bool(text.strip())

    def _emit_diagnostic_status(self, text):
        self.statuses.append(str(text))

    def _extract_api_error_context(self, error):
        from agent.agent_runtime_helpers import extract_api_error_context
        return extract_api_error_context(error)

    def clear_interrupt(self, **kwargs):
        return False

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _settle(agent, err, *, retry_count=3, retry=None):
    retry = retry or TurnRetryState()
    classified = classify_api_error(err, provider=agent.provider)
    with patch("agent.conversation_loop._credential_refresh_profile", lambda a: None, create=True), \
            patch("agent.conversation_loop._is_copilot_provider", lambda a: False, create=True), \
            patch("agent.conversation_loop._arm_fallback_restart", lambda a, m, s, r: s), \
            patch("agent.turn_recovery.interruptible_backoff_sleep", lambda *a, **k: None), \
            patch("agent.turn_api_error.max_retries_exhausted_result", lambda *a, **k: {"failed": True}), \
            patch("agent.turn_api_error.nonretryable_client_error_result", lambda *a, **k: {"failed": True}):
        return settle_unrecovered_error(
            agent, api_error=err, classified=classified, _retry=retry, status_code=err.status_code,
            error_msg=str(err).lower(), is_context_length_error=False,
            is_rate_limited=classified.reason.value in ("rate_limit", "billing"), _is_zai_coding_overload=False,
            _provider=agent.provider, _base=agent.base_url, _model=agent.model, messages=[], api_messages=[],
            api_kwargs=None, active_system_prompt=None, conversation_history=None, approx_tokens=0,
            retry_count=retry_count, max_retries=3, compression_attempts=0, api_call_count=1,
        ), retry


@pytest.fixture(autouse=True)
def _health_file(tmp_path, monkeypatch):
    path = tmp_path / "model_health.json"
    monkeypatch.setattr(provider_events, "_health_path", lambda: str(path))
    return path


def test_emit_shapes_payload_drops_empty_fields_and_never_raises():
    agent = _Agent(locked=True)
    provider_events.emit(agent, "retry", attempt=2, max=3, wait_seconds=4.0, reset_at=None, reason="connection")
    assert agent.events == [{"kind": "retry", "provider": "custom", "model": "qwen", "locked": True,
                             "attempt": 2, "max": 3, "wait_seconds": 4.0, "reason": "connection"}]

    def boom(event):
        raise RuntimeError("closed socket")
    agent.provider_event_callback = boom
    provider_events.emit(agent, "retry")  # swallowed


def test_reset_at_normalises_ms_and_iso():
    assert provider_events.parse_reset_at(1_900_000_000_000) == pytest.approx(1_900_000_000.0)
    assert provider_events.parse_reset_at("2030-01-01T00:00:00Z") == pytest.approx(1893456000.0)
    assert provider_events.parse_reset_at(None) is None


@pytest.mark.parametrize("err,expected", [
    (_Err(429, "usage_limit_reached", body={"error": {"type": "usage_limit_reached", "message": "The usage limit has been reached",
                                                      "resets_in_seconds": 3600}}), True),
    (_Err(429, "rate_limit_error: This request would exceed your account's rate limit",
          headers={"retry-after": "20"}), False),
    (_Err(402, "Your credit balance is too low"), True),
    (_Err(503, "Service unavailable"), False),
])
def test_usage_exhausted_tells_allowance_from_throttle(err, expected):
    classified = classify_api_error(err, provider="openai-codex")
    from agent.agent_runtime_helpers import extract_api_error_context
    assert provider_events.is_usage_exhausted(classified, extract_api_error_context(err)) is expected


def test_error_class_words():
    assert provider_events.error_class(_Err(429, "x")) == "rate_limit"
    assert provider_events.error_class(_Err(502, "bad gateway")) == "server_error"
    assert provider_events.error_class(_Err(None, "Connection error.")) == "connection"
    assert provider_events.error_class(_Err(404, "model not found")) == "not_found"


def test_locked_local_model_down_fails_fast_without_ladder_or_transport_rebuild():
    """Locked + local endpoint refusing: max retries end the turn now (no primary-transport rebuild,
    no 15/30/60 s recovery ladder) and the failure is announced and remembered for the picker."""
    agent = _Agent(locked=True)
    verdict, retry = _settle(agent, _Err(None, "Connection error."))
    assert verdict.action == "return" and verdict.result == {"failed": True}
    assert agent.primary_recoveries == 0 and retry.auto_recovery_cycles_used == 0
    assert agent.events[-1]["kind"] == "failed" and agent.events[-1]["reason"] == "connection"
    assert "custom::qwen" in provider_events.recent_model_failures()


def test_unlocked_turn_keeps_the_recovery_path():
    agent = _Agent(locked=False)
    verdict, _ = _settle(agent, _Err(None, "Connection error."))
    assert verdict.action == "continue" and agent.primary_recoveries == 1


def test_locked_turn_stops_retrying_on_usage_limit():
    agent = _Agent(locked=True, base_url="https://chatgpt.com/backend-api/codex", provider="openai-codex",
                   model="gpt-6.1-sol")
    err = _Err(429, "usage_limit_reached", body={"error": {"type": "usage_limit_reached",
                                                           "message": "The usage limit has been reached",
                                                           "resets_in_seconds": 3600}})
    verdict, _ = _settle(agent, err, retry_count=1)
    assert verdict.action == "return"
    # Unlocked: the same error keeps the normal backoff (retry event, no terminal verdict yet).
    free = _Agent(locked=False, base_url="https://chatgpt.com/backend-api/codex", provider="openai-codex")
    with patch("agent.retry_utils.adaptive_rate_limit_backoff", lambda *a, **k: (1.0, None)):
        verdict, _ = _settle(free, err, retry_count=1)
    assert verdict.action == "fallthrough"
    assert free.events and free.events[-1]["kind"] == "rate_limited" and free.events[-1]["attempt"] == 1
    assert free.events[-1]["reset_at"] > time.time() + 3000


def test_model_health_file_records_and_clears(tmp_path):
    path = str(tmp_path / "h.json")
    provider_events.record_model_failure({"kind": "failed", "provider": "custom", "model": "qwen",
                                          "reason": "connection"}, path=path)
    assert provider_events.recent_model_failures(path=path)["custom::qwen"]["reason"] == "connection"
    provider_events.record_model_ok("custom", "qwen", path=path)
    assert provider_events.recent_model_failures(path=path) == {}


def test_run_event_callback_maps_to_provider_status_event():

    from gateway.platforms import api_server_runs
    sent = []
    run = SimpleNamespace(put_event=sent.append)

    class _Loop:
        def call_soon_threadsafe(self, fn, *args):
            fn(*args)
    cb = api_server_runs._make_provider_event_callback(run, "run_1", _Loop())
    cb({"kind": "fallback", "provider": "ollama-launch", "model": "qwen", "to_provider": "openai-codex",
        "to_model": "gpt-6.1-sol", "secret": "nope", "attempt": None})
    cb({"no": "kind"})
    assert len(sent) == 1
    event = sent[0]
    assert event["event"] == "provider.status" and event["run_id"] == "run_1"
    assert {k: event[k] for k in ("kind", "to_model")} == {"kind": "fallback", "to_model": "gpt-6.1-sol"}
    assert "secret" not in event and "attempt" not in event



def test_admission_observers_accept_the_provider_sink():
    from gateway import session_api_turn
    assert "provider_event_callback" in session_api_turn._OBSERVER_KEYS
