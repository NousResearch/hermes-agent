"""Terminal replies describe the backup chain that was actually tried."""

from types import SimpleNamespace

import pytest

from agent.error_classifier import classify_api_error
from agent.turn_api_call import nous_rate_limit_guard
from agent.turn_failure_copy import exhausted_copy
from agent.turn_recovery import max_retries_exhausted_result


ADD_BACKUP = "To avoid this in future, add a backup provider with `hermes fallback add`."
NO_BACKUP = "None of your backup providers could take over. Check them with `hermes fallback list`."
SWITCHED_BACKUP = (
    "Hermes had already switched from claude-x (Anthropic) to a backup, and no other backup "
    "could take over. Check your backups with `hermes fallback list`."
)
NOUS_COPY = (
    "Wait for the reset and send /retry, or switch models with /model. "
    "To avoid waits, add a backup provider with `hermes fallback add`."
)


class _Agent:
    log_prefix = ""
    verbose = False
    verbose_logging = False
    provider = "openrouter"
    model = "b"
    base_url = "https://openrouter.ai/api/v1"
    suppress_status_output = True
    _fallback_chain = ()
    _fallback_index = 0
    _provider_fallback_active = False
    _primary_runtime = {"model": "claude-x", "provider": "anthropic"}

    def _summarize_api_error(self, error):
        return str(error)

    def _try_activate_fallback(self):
        return False

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _agent(**state):
    agent = _Agent()
    agent.__dict__.update(state)
    return agent


def _exhausted(agent, *, reset=False):
    error = Exception("upstream unavailable")
    classified = classify_api_error(error, provider=agent.provider, model=agent.model)
    if reset:
        from agent.error_classifier import FailoverReason
        classified = SimpleNamespace(
            reason=FailoverReason.rate_limit, retryable=True, error_context={"reset_at": 3700},
        )
    return max_retries_exhausted_result(
        agent, error, classified, attempts=3, is_rate_limited=reset,
        error_msg=str(error), api_kwargs=None, api_messages=[], messages=[],
        conversation_history=None, api_call_count=3, approx_tokens=10,
        provider=agent.provider, base_url=agent.base_url, model=agent.model,
    )["final_response"]


def test_active_backup_names_abandoned_primary():
    text = _exhausted(_agent(
        _fallback_chain=[{"provider": "openrouter", "model": "b"}],
        _fallback_index=1, _provider_fallback_active=True,
    ))
    assert SWITCHED_BACKUP in text
    assert "hermes fallback add" not in text
    assert text.endswith("\n\nProvider said: upstream unavailable")


def test_all_backup_candidates_skipped():
    text = _exhausted(_agent(
        _fallback_chain=[{"provider": "openrouter", "model": "b"}], _fallback_index=1,
    ))
    assert NO_BACKUP in text
    assert "hermes fallback add" not in text


def test_unwalked_chain_drops_advice_and_preserves_reset_window(monkeypatch):
    monkeypatch.setattr("agent.turn_recovery.time.time", lambda: 100)
    text = _exhausted(_agent(
        _fallback_chain=[{"provider": "openrouter", "model": "b"}],
    ), reset=True)
    assert "its usage limit resets in ~1h. Send /retry after that, or switch models with /model." in text
    assert "add a backup provider" not in text
    assert "fallback list" not in text
    assert "/model.\n\nProvider said:" in text
    assert "  " not in text


def test_empty_chain_preserves_add_backup_advice():
    assert ADD_BACKUP in _exhausted(_agent())


def test_exhausted_copy_without_new_kwarg_preserves_exact_text():
    assert exhausted_copy("server_error", label="OpenRouter", attempts=3, summary="unavailable") == (
        "OpenRouter returned a server error on all 3 attempts — it looks temporarily unavailable. "
        "Wait a minute and send /retry, or switch models with /model. "
        + ADD_BACKUP + "\n\nProvider said: unavailable"
    )


@pytest.mark.parametrize("walked", [True, False])
def test_signed_in_nous_guard_backup_advice(monkeypatch, walked):
    monkeypatch.setattr("hermes_cli.anon_auth.apply_model_switch", lambda agent: None)
    monkeypatch.setattr("hermes_cli.anon_auth.is_anonymous_agent", lambda agent: False)
    monkeypatch.setattr("agent.nous_rate_guard.nous_rate_limit_remaining", lambda **kwargs: 600)
    agent = _agent(provider="nous")
    if walked:
        agent._fallback_chain = [{"provider": "openrouter", "model": "b"}]
        agent._fallback_index = 1
    verdict = nous_rate_limit_guard(
        agent, _retry=None, api_messages=[], messages=[], conversation_history=None,
        active_system_prompt="system", retry_count=0, compression_attempts=0, api_call_count=0,
    )
    assert verdict.action == "return"
    text = verdict.result["final_response"]
    assert "Your Nous account has hit its rate limit; it resets in" in text
    if walked:
        assert NO_BACKUP in text
        assert "fallback add" not in text
    else:
        assert text.endswith(NOUS_COPY)
        from agent.turn_failure_copy import nous_rate_limit_copy
        assert nous_rate_limit_copy(_agent()) == NOUS_COPY


@pytest.mark.parametrize("agent", [object(), SimpleNamespace(_fallback_chain=None)])
def test_clause_missing_state_preserves_legacy_advice(agent):
    from agent.turn_failure_copy import backup_provider_clause
    assert backup_provider_clause(agent) == ADD_BACKUP


@pytest.mark.parametrize("snapshot", [{}, {"model": "claude-x"}, {"provider": "anthropic"}])
def test_active_backup_with_incomplete_primary_uses_generic_clause(snapshot):
    from agent.turn_failure_copy import backup_provider_clause
    assert backup_provider_clause(_agent(
        _fallback_chain=[{"provider": "openrouter", "model": "b"}], _fallback_index=1,
        _provider_fallback_active=True, _primary_runtime=snapshot,
    )) == NO_BACKUP


def test_clause_state_read_exception_preserves_legacy_advice():
    from agent.turn_failure_copy import backup_provider_clause

    class _BrokenAgent:
        @property
        def _fallback_chain(self):
            raise RuntimeError("state unavailable")

    assert backup_provider_clause(_BrokenAgent()) == ADD_BACKUP
