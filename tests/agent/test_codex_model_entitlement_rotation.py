"""A Codex ChatGPT-account model entitlement 400 rotates to the next pool credential (#71970).

The exact normalized rejection benches only (credential, model) and hands the next eligible entry
back; every other 400 stays a plain request failure. Once every entry rejects the model, the
single-credential handling from #106475 takes over.
"""
import json
import time
import types
from unittest.mock import MagicMock

import pytest

from agent.agent_runtime_helpers import recover_with_credential_pool
from agent.error_classifier import FailoverReason, classify_api_error

MODEL = "gpt-5.3-codex"
OTHER_MODEL = "gpt-5.3-codex-mini"
TOKENS = ("tok-account-a", "tok-account-b")


class _Err(Exception):
    def __init__(self, status, body):
        self.status_code = status
        self.body = body
        self.response = types.SimpleNamespace(status_code=status, headers={}, text=json.dumps(body), json=lambda: body)
        self.message = f"Error code: {status} - {json.dumps(body)}"
        super().__init__(self.message)


def _entitlement_400():
    return _Err(400, {"detail": f"The '{MODEL}' model is not supported when using Codex with a ChatGPT account."})


@pytest.fixture
def pool(tmp_path, monkeypatch):
    root = tmp_path / "hermes-root"
    root.mkdir()
    (tmp_path / "fakehome").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "fakehome"))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import hermes_constants
    hermes_constants._default_hermes_root_memo = None  # type: ignore[attr-defined]
    (root / "auth.json").write_text(json.dumps({"credential_pool": {"openai-codex": [
        {"id": f"cred-{i}", "label": f"acct-{i}", "auth_type": "oauth", "priority": i, "source": "manual",
         "access_token": tok, "refresh_token": f"rt-{i}", "expires_at_ms": 4_000_000_000_000}
        for i, tok in enumerate(TOKENS)
    ]}}))
    from agent.credential_pool import load_pool
    return load_pool("openai-codex")


def test_entitlement_400_benches_only_that_model_and_rotates(pool):
    verdict = classify_api_error(_entitlement_400(), provider="openai-codex", model=MODEL)
    assert verdict.reason == FailoverReason.model_entitlement
    assert verdict.should_rotate_credential and verdict.should_fallback and not verdict.retryable

    generic = classify_api_error(_Err(400, {"detail": "Invalid request: bad field"}), provider="openai-codex", model=MODEL)
    assert generic.reason == FailoverReason.format_error and not generic.should_rotate_credential

    # Drive the production recovery entry point (turn recovery -> recover_with_credential_pool),
    # not the pool directly: the classifier verdict must reach the model-scoped bench.
    assert pool.select(model=MODEL).id == "cred-0"
    agent = types.SimpleNamespace(
        provider="openai-codex", model=MODEL, base_url="https://chatgpt.com/backend-api/codex",
        api_key=TOKENS[0], _credential_pool=pool, _credential_pool_entry_id="cred-0",
        _swap_credential=MagicMock(return_value=True),
    )
    assert recover_with_credential_pool(
        agent, status_code=400, has_retried_429=False, classified_reason=verdict.reason,
    ) == (True, False)
    agent._swap_credential.assert_called_once()
    assert agent._swap_credential.call_args.args[0].id == "cred-1"
    first = pool.entries()[0]
    assert first.last_status is None  # credential-wide state untouched: other models stay usable
    assert set(first.model_cooldowns) == {MODEL}
    # An entitlement is a plan property, not a window: no hourly re-probe, only reset clears it.
    assert first.model_cooldowns[MODEL] > time.time() + 24 * 3600
    assert pool.select(model=OTHER_MODEL).id == "cred-0"
    assert pool.reset_statuses() >= 1 and not pool.entries()[0].model_cooldowns


def test_all_entries_rejecting_falls_back_to_session_marker(pool):
    from agent.fallback_cooldown import _is_entitlement_rejected, _mark_entitlement_rejected_model

    agent = types.SimpleNamespace(
        provider="openai-codex", model=MODEL, _credential_pool=pool,
        _buffer_diagnostic_status=lambda *_a, **_k: None,
    )
    pool.mark_exhausted_and_rotate(
        status_code=400, api_key_hint=TOKENS[0], credential_id="cred-0", failure_reason="model_entitlement", model=MODEL,
    )
    # cred-1 is still eligible for the model: rotation owns the recovery, no session-wide marker.
    assert _mark_entitlement_rejected_model(agent, _entitlement_400()) is False
    assert not _is_entitlement_rejected(agent, "openai-codex", MODEL)

    assert pool.mark_exhausted_and_rotate(
        status_code=400, api_key_hint=TOKENS[1], credential_id="cred-1", failure_reason="model_entitlement", model=MODEL,
    ) is None
    assert _mark_entitlement_rejected_model(agent, _entitlement_400()) is True
    assert _is_entitlement_rejected(agent, "openai-codex", MODEL)


def test_provider_level_availability_ignores_per_model_benches(pool):
    """A per-model entitlement bench must not make the provider unusable for pickers.

    The provider-level read (`has_available(ignore_model_cooldowns=True)`, used by
    `_credential_pool_is_usable` for every model picker) stays True while the benched
    model itself stays unselectable; the conservative unscoped default is unchanged.
    """
    pool.mark_exhausted_and_rotate(
        status_code=400, api_key_hint=TOKENS[0], credential_id="cred-0",
        failure_reason="model_entitlement", model=MODEL,
    )
    # Provider-level: both credentials remain visible despite the (credential, model) bench.
    assert pool.has_credentials()
    assert pool.has_available(ignore_model_cooldowns=True)
    # Leasing for the benched model rotates away; for any other model it still leases.
    assert pool.select(model=MODEL).id == "cred-1"
    assert pool.select(model=OTHER_MODEL).id == "cred-0"

    # Sole-credential case: bench every entry for the model — provider stays listed.
    pool.mark_exhausted_and_rotate(
        status_code=400, api_key_hint=TOKENS[1], credential_id="cred-1",
        failure_reason="model_entitlement", model=MODEL,
    )
    assert pool.has_available(ignore_model_cooldowns=True)
    assert pool.select(model=MODEL) is None
    assert pool.select(model=OTHER_MODEL).id in {"cred-0", "cred-1"}
