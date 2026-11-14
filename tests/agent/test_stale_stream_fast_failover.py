from agent.chat_completion_helpers import _StreamingCall, _should_skip_fallback_candidate


class _DummyAgent:
    def __init__(self):
        self.provider = "vertex"
        self.model = "google/gemini-4-argon"
        self.base_url = "https://aiplatform.googleapis.com/v1beta1/projects/uop-atlas/locations/global/endpoints/openapi"
        self._last_routed_base_url = "https://europe-west2-aiplatform.googleapis.com/v1beta1/projects/uop-atlas/locations/europe-west2/endpoints/openapi"
        self.api_mode = "chat_completions"
        self._interrupt_requested = False
        self._stream_options_unsupported = False
        self._stream_5xx_probe_ts = None
        self.logged_retries = []
        self.buffered_statuses = []

    def _warning_presentation_enabled(self) -> bool:
        return False

    def _is_provider_stream_parse_error(self, _err: Exception) -> bool:
        return False

    def _emit_stream_drop(self, **_kwargs) -> None:
        pass

    def _log_stream_retry(self, **kwargs) -> None:
        self.logged_retries.append(kwargs)

    def _buffer_diagnostic_status(self, msg: str) -> None:
        self.buffered_statuses.append(msg)

    def _maybe_disable_streaming(self, _err: Exception) -> None:
        pass


def test_stale_killed_stream_skips_inner_retries_and_marks_error():
    """When _kill_stale_stream already killed the current attempt after 600s,
    _handle_stream_error must NOT loop through max_retries more 600s inner attempts,
    and must mark the propagated exception with _stale_stream_killed = True so
    turn_recovery treats it as a timeout and flips Vertex regional -> global immediately.
    """
    agent = _DummyAgent()
    call = _StreamingCall(agent, {"model": agent.model}, None)
    call.stream_attempt_state["current"] = 1
    call._stale_counted_attempts.add(1)  # Attempt 1 was killed by _kill_stale_stream

    err = ConnectionError("[Errno 32] Broken pipe")
    should_retry = call._handle_stream_error(err, attempt=0, max_retries=2)

    assert should_retry is False
    assert call.result["error"] is err
    assert getattr(err, "_stale_stream_killed", False) is True


def test_should_skip_fallback_candidate_uses_last_routed_base_url():
    """When acquire_vertex_hybrid_region routed the failed request to europe-west2
    (_last_routed_base_url) while agent.base_url remained locations/global, the
    fallback entry for locations/global must NOT be skipped as 'same backend'.
    """
    agent = _DummyAgent()
    fb = {
        "provider": "vertex",
        "model": "google/gemini-4-argon",
        "base_url": "https://aiplatform.googleapis.com/v1beta1/projects/uop-atlas/locations/global/endpoints/openapi",
    }
    unavailable = set()
    assert _should_skip_fallback_candidate(
        agent,
        fb,
        ("vertex", "google/gemini-4-argon", fb["base_url"]),
        "vertex",
        "google/gemini-4-argon",
        unavailable,
    ) is False
