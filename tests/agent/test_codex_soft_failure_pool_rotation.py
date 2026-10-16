"""Codex Responses HTTP-200 soft failures (``response.status == "failed"``) must reach the
same-provider credential pool before cross-provider fallback (#24159).

The SDK never raises on these, so the exception path's ``_recover_with_credential_pool`` never
sees them; ``retry_invalid_response`` has to classify ``response.error`` itself.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.agent_runtime_helpers import recover_with_credential_pool
from agent.credential_pool import STATUS_EXHAUSTED, CredentialPool, PooledCredential
from agent.turn_response_check import retry_invalid_response
from agent.turn_retry_state import TurnRetryState

_BASE_URL = "https://chatgpt.com/backend-api/codex"


def _entry(i: int) -> PooledCredential:
    return PooledCredential(
        provider="openai-codex", id=f"cred-{i}", label=f"acct-{i}", auth_type="api_key", priority=i,
        source="manual", access_token=f"tok-{i}-1234567890", base_url=_BASE_URL,
    )


class _Agent:
    log_prefix = ""
    quiet_mode = True
    api_mode = "codex_responses"
    provider = "openai-codex"
    model = "gpt-5.1-codex"
    base_url = _BASE_URL
    _fallback_chain = ()
    _fallback_index = 0
    _credential_pool_revert_id = None

    def __init__(self, pool: CredentialPool) -> None:
        self._credential_pool = pool
        self.api_key = pool.select().access_token
        self.swapped_to: list = []
        self._try_activate_fallback = MagicMock(return_value=False)

    def _recover_with_credential_pool(self, **kwargs):
        return recover_with_credential_pool(self, **kwargs)

    def _extract_api_error_context(self, error):
        from agent.agent_runtime_helpers import extract_api_error_context

        return extract_api_error_context(error)

    def _swap_credential(self, entry):
        self.swapped_to.append(entry.id)
        self.api_key = entry.access_token
        return True

    def _has_pending_fallback(self):
        return False

    def _clean_error_message(self, msg):
        return msg

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _soft_failure(code: str, message: str) -> SimpleNamespace:
    # The SDK types ``response.error`` as ``ResponseError(code=..., message=...)``, not a dict.
    return SimpleNamespace(status="failed", output=[], output_text="", error=SimpleNamespace(code=code, message=message))


def _run(agent: _Agent, response: SimpleNamespace):
    return retry_invalid_response(
        agent, response=response, error_details=["response.status=failed"], _retry=TurnRetryState(),
        thinking_spinner=None, messages=[], api_messages=[], api_kwargs=None, active_system_prompt=None,
        conversation_history=None, retry_count=0, max_retries=3, compression_attempts=0, api_call_count=1,
        api_request_id="r", api_start_time=0.0, api_duration=0.4, effective_task_id="t", turn_id="turn",
    )


def test_quota_soft_failure_rotates_pool_before_provider_fallback():
    pool = CredentialPool("openai-codex", [_entry(0), _entry(1)])
    agent = _Agent(pool)

    verdict = _run(agent, _soft_failure("usage_limit_reached", "You've hit your usage limit. Try again at 3:00 PM."))

    assert verdict.action == "continue"
    assert agent.swapped_to == ["cred-1"] and agent.api_key == "tok-1-1234567890"
    benched = next(e for e in pool.entries() if e.id == "cred-0")
    assert benched.last_status == STATUS_EXHAUSTED and benched.last_error_reason == "usage_limit_reached"
    agent._try_activate_fallback.assert_not_called()


def test_content_policy_soft_failure_leaves_pool_alone():
    pool = CredentialPool("openai-codex", [_entry(0), _entry(1)])
    agent = _Agent(pool)

    verdict = _run(agent, _soft_failure("content_policy_violation", "Your request was rejected by our safety system."))

    assert verdict.action == "continue"  # ordinary invalid-response retry path
    assert agent.swapped_to == [] and agent.api_key == "tok-0-1234567890"
    assert all(e.last_status is None for e in pool.entries())
    agent._try_activate_fallback.assert_called()
