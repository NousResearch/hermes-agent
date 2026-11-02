"""Main-loop recovery for a backend that rejects Ollama's ``options`` request field (#142239).

``options`` is Ollama's own request extension (``options.num_ctx``). A backend that validates
its schema strictly - Gemini relayed through a local gateway - rejects the whole request, so
the field has to go: the recovery drops the session's num_ctx and retries once instead of
failing every request with the same 400.
"""

from agent.error_classifier import classify_api_error
from agent.turn_retry_state import TurnRetryState

_GEMINI_400 = (
    "Invalid JSON payload received. Unknown name \"options\": Cannot find field. "
    "status: INVALID_ARGUMENT"
)
_UNRELATED_400 = "Invalid JSON payload received. Unknown name \"safetySettings\": Cannot find field."


class _FakeApiError(Exception):
    def __init__(self, status_code, message):
        super().__init__(f"Error code: {status_code} - {message}")
        self.status_code = status_code
        self.message = message
        self.body = {"error": {"message": message, "status": "INVALID_ARGUMENT"}}


class _Agent:
    log_prefix = ""
    verbose = False
    provider = "custom"
    model = "gemini-3-pro"
    _fallback_chain = [object()]
    _fallback_index = 0
    _credential_pool = None

    def __init__(self, num_ctx: int | None = 131072):
        self._ollama_num_ctx = num_ctx
        self.notices = []

    def _recover_with_credential_pool(self, **kwargs):
        return False, False

    def _has_pending_fallback(self):
        return True

    def _try_activate_fallback(self, **kwargs):
        return True

    def _summarize_api_error(self, error):
        return str(error)

    def _vprint(self, line, **kwargs):
        self.notices.append(line)

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _recover(agent, message, retry=None):
    """Drive the production recovery chain with a 400 that names the field."""
    from agent.turn_recovery import recover_after_classification

    err = _FakeApiError(400, message)
    classified = classify_api_error(err, provider=agent.provider, model=agent.model)
    retry = retry if retry is not None else TurnRetryState()
    retry_now, _ = recover_after_classification(
        agent, err, classified, retry, status_code=400, error_context={}, messages=[], api_messages=[],
    )
    return retry_now, retry


def test_the_options_field_is_dropped_and_the_request_retried():
    agent = _Agent()
    retry_now, _ = _recover(agent, _GEMINI_400)
    assert retry_now is True
    assert agent._ollama_num_ctx is None, "the Ollama-only request extension must be dropped"
    assert any("options" in n for n in agent.notices)


def test_the_strip_happens_once_per_turn():
    """A second rejection must not loop on the same request."""
    agent = _Agent()
    retry = TurnRetryState()
    assert _recover(agent, _GEMINI_400, retry)[0] is True
    assert _recover(agent, _GEMINI_400, retry)[0] is False


def test_an_unrelated_unknown_field_leaves_num_ctx_alone():
    """Control: a 400 about another field must not drop the session's num_ctx."""
    agent = _Agent()
    retry_now, _ = _recover(agent, _UNRELATED_400)
    assert retry_now is False
    assert agent._ollama_num_ctx == 131072
