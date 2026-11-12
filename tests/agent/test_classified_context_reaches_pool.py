"""A classification's ``error_context`` (from a ``transform_api_error_classification`` hook or a
provider profile's ``classify_api_error``) reaches credential-pool recovery.

Before: recovery read only what ``extract_api_error_context`` scraped off the exception, so a
classifier's ``reset_at`` never sized the bench, and the per-model cooldown that already exists
for Anthropic could not be asked for by anyone else.
"""

import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.error_classifier import ClassifiedError, FailoverReason
from agent.turn_api_error import merge_classified_error_context

PROVIDER = "openrouter"
_PROVIDER_ENV = ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY")


def _classified(reason=FailoverReason.rate_limit, **ctx):
    return ClassifiedError(reason=reason, error_context=dict(ctx))


@pytest.fixture
def make_pool(tmp_path, monkeypatch):
    def _make(entries):
        home = tmp_path / "hermes"
        home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        for env_var in _PROVIDER_ENV:
            monkeypatch.delenv(env_var, raising=False)
        monkeypatch.setattr("hermes_cli.auth.is_provider_explicitly_configured", lambda provider: False)
        (home / "auth.json").write_text(
            json.dumps({"version": 1, "credential_pool": {PROVIDER: entries}}), encoding="utf-8",
        )
        from agent.credential_pool import load_pool

        pool = load_pool(PROVIDER)
        assert [e.id for e in pool.entries()] == [e["id"] for e in entries], "fixture leaked host keys"
        return pool

    return _make


def _entry(idx):
    return {
        "id": f"cred-{idx}", "label": f"key-{idx}", "auth_type": "api_key",
        "priority": idx, "source": "manual", "access_token": f"key-{idx}",
    }


def _agent(pool, failing_key, model="model-a"):
    return SimpleNamespace(
        provider=PROVIDER, api_key=failing_key, model=model, base_url=None,
        _credential_pool=pool, _credential_pool_entry_id=None,
        _credential_pool_revert_id=None, _swap_credential=MagicMock(),
    )


def _by_id(pool, cred_id):
    return next(e for e in pool.entries() if e.id == cred_id)


# --- merge -------------------------------------------------------------------------------------

def test_classifier_reset_and_scope_win_over_extracted_context():
    extracted = {"reason": "RESOURCE_EXHAUSTED", "message": "quota", "reset_at": 100.0}
    merged = merge_classified_error_context(
        extracted, _classified(reset_at=500.0, quota_scope="model", unrelated="ignored"),
    )
    assert merged == {"reason": "RESOURCE_EXHAUSTED", "message": "quota", "reset_at": 500.0,
                      "quota_scope": "model"}
    assert extracted["reset_at"] == 100.0, "the turn's own dict is not mutated"


def test_classifier_without_context_leaves_extracted_context_alone():
    extracted = {"message": "slow down", "reset_at": 100.0}
    assert merge_classified_error_context(extracted, _classified()) == extracted
    assert merge_classified_error_context(None, _classified()) == {}
    assert merge_classified_error_context(extracted, _classified(reset_at="")) == extracted


# --- per-model scope ---------------------------------------------------------------------------

def test_model_scoped_throttle_benches_only_that_model(make_pool):
    from agent.agent_runtime_helpers import recover_with_credential_pool

    pool = make_pool([_entry(0), _entry(1)])
    reset_at = time.time() + 600
    agent = _agent(pool, failing_key="key-0", model="model-a")

    recovered, _ = recover_with_credential_pool(
        agent, status_code=429, has_retried_429=True, classified_reason=FailoverReason.rate_limit,
        error_context={"reset_at": reset_at, "quota_scope": "model"},
    )

    assert recovered is True
    failed = _by_id(pool, "cred-0")
    assert failed.last_status != "exhausted", "credential stays usable for other models"
    assert failed.model_cooldowns.get("model-a") == pytest.approx(reset_at)
    assert pool.has_available(model="model-b")
    assert agent._swap_credential.call_args[0][0].id == "cred-1"


def test_throttle_without_scope_still_benches_the_credential(make_pool):
    from agent.agent_runtime_helpers import recover_with_credential_pool

    pool = make_pool([_entry(0), _entry(1)])
    reset_at = time.time() + 600
    agent = _agent(pool, failing_key="key-0")

    recover_with_credential_pool(
        agent, status_code=429, has_retried_429=True, classified_reason=FailoverReason.rate_limit,
        error_context={"reset_at": reset_at},
    )

    failed = _by_id(pool, "cred-0")
    assert failed.last_status == "exhausted"
    assert not failed.model_cooldowns


def test_auth_failure_never_narrows_to_one_model(make_pool):
    from agent.credential_pool import load_pool

    pool = make_pool([_entry(0), _entry(1)])
    pool.mark_exhausted_and_rotate(
        status_code=401, error_context={"quota_scope": "model"}, api_key_hint="key-0",
        failure_reason="auth", model="model-a",
    )

    failed = _by_id(load_pool(PROVIDER), "cred-0")
    assert failed.last_status in ("exhausted", "dead")
    assert not failed.model_cooldowns


def test_billing_never_narrows_to_one_model(make_pool):
    from agent.agent_runtime_helpers import recover_with_credential_pool

    pool = make_pool([_entry(0), _entry(1)])
    agent = _agent(pool, failing_key="key-0")

    recover_with_credential_pool(
        agent, status_code=402, has_retried_429=False, classified_reason=FailoverReason.billing,
        error_context={"quota_scope": "model"},
    )

    failed = _by_id(pool, "cred-0")
    assert failed.last_status == "exhausted"
    assert not failed.model_cooldowns


# --- stated reset rotates now ------------------------------------------------------------------

def test_first_429_with_stated_reset_rotates_without_a_retry(make_pool):
    from agent.agent_runtime_helpers import recover_with_credential_pool

    pool = make_pool([_entry(0), _entry(1)])
    reset_at = time.time() + 30
    agent = _agent(pool, failing_key="key-0")

    recovered, has_retried = recover_with_credential_pool(
        agent, status_code=429, has_retried_429=False, classified_reason=FailoverReason.rate_limit,
        error_context={"reset_at": reset_at},
    )

    assert (recovered, has_retried) == (True, False)
    failed = _by_id(pool, "cred-0")
    assert failed.last_status == "exhausted"
    assert failed.last_error_reset_at == pytest.approx(reset_at)
    assert agent._swap_credential.call_args[0][0].id == "cred-1"


@pytest.mark.parametrize("error_context", [
    None,
    {},
    {"message": "Too Many Requests"},
    {"reset_at": time.time() - 5},
])
def test_first_429_without_a_future_reset_keeps_retry_once(make_pool, error_context):
    from agent.agent_runtime_helpers import recover_with_credential_pool

    pool = make_pool([_entry(0), _entry(1)])
    agent = _agent(pool, failing_key="key-0")

    recovered, has_retried = recover_with_credential_pool(
        agent, status_code=429, has_retried_429=False, classified_reason=FailoverReason.rate_limit,
        error_context=error_context,
    )

    assert (recovered, has_retried) == (False, True)
    assert _by_id(pool, "cred-0").last_status != "exhausted"
    agent._swap_credential.assert_not_called()


def test_sole_credential_with_stated_reset_keeps_retry_once(make_pool):
    from agent.agent_runtime_helpers import recover_with_credential_pool

    pool = make_pool([_entry(0)])
    agent = _agent(pool, failing_key="key-0")

    recovered, has_retried = recover_with_credential_pool(
        agent, status_code=429, has_retried_429=False, classified_reason=FailoverReason.rate_limit,
        error_context={"reset_at": time.time() + 30},
    )

    assert (recovered, has_retried) == (False, True)
    assert _by_id(pool, "cred-0").last_status != "exhausted"
