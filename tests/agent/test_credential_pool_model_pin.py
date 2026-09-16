"""Model-pinned pool rows: ``allowed_models`` keeps a cheap-tier account off premium models.

Two ChatGPT accounts in one pool (a pro plan and a plus plan) must not share model
routing: the premium slug may only spend the pro account's quota, and the plus account
is reserved for the cheaper slug. Without the pin, credential rotation is model-blind and
a premium 429 walks straight onto the plus account with the same premium model.
"""

from __future__ import annotations

import json
import time

import pytest


def _write_auth_store(tmp_path, payload: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps(payload, indent=2))


def _two_account_store(*, pro_exhausted: bool = False) -> dict:
    pro: dict = {
        "id": "pro001",
        "label": "pro-acct1",
        "auth_type": "oauth",
        "priority": 0,
        "source": "manual:device_code",
        "access_token": "tok-pro",
    }
    if pro_exhausted:
        pro.update(
            {
                "last_status": "exhausted",
                "last_status_at": time.time(),
                "last_error_code": 429,
                "last_error_reason": "usage_limit_reached",
                "last_error_reset_at": time.time() + 3600,
            }
        )
    plus = {
        "id": "plus01",
        "label": "plus-acct2",
        "auth_type": "oauth",
        "priority": 1,
        "source": "device_code",
        "access_token": "tok-plus",
        "allowed_models": ["gpt-5.6-terra"],
    }
    return {"version": 1, "credential_pool": {"openai-codex": [pro, plus]}}


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    # Host Codex CLI tokens must not seed a third row into the pool under test.
    monkeypatch.setattr(
        "agent.credential_pool._seed_from_singletons",
        lambda provider, entries: (False, set()),
    )


def test_premium_model_never_selects_the_pinned_cheap_account(tmp_path):
    """Premium slug + exhausted pro account = no capacity, NOT a rotation onto plus."""
    _write_auth_store(tmp_path, _two_account_store(pro_exhausted=True))
    from agent.credential_pool import load_pool

    pool = load_pool("openai-codex")
    pool.model_scope = "gpt-5.6-sol"

    assert pool.select() is None, "sol must not borrow the terra-pinned plus account"


def test_pinned_account_serves_its_own_model_when_the_other_is_exhausted(tmp_path):
    """The cheap slug falls through to the pinned account once pro is spent."""
    _write_auth_store(tmp_path, _two_account_store(pro_exhausted=True))
    from agent.credential_pool import load_pool

    pool = load_pool("openai-codex")
    pool.model_scope = "gpt-5.6-terra"

    entry = pool.select()
    assert entry is not None and entry.label == "plus-acct2"


def test_unpinned_account_keeps_priority_for_the_pinned_model(tmp_path):
    """Pinning plus to terra does not make it outrank a healthy unpinned pro row."""
    _write_auth_store(tmp_path, _two_account_store())
    from agent.credential_pool import load_pool

    pool = load_pool("openai-codex")
    pool.model_scope = "gpt-5.6-terra"

    entry = pool.select()
    assert entry is not None and entry.label == "pro-acct1"


def test_unknown_model_scope_leaves_every_row_eligible(tmp_path):
    """Auxiliary callers that never set a scope must not be starved by pins."""
    _write_auth_store(tmp_path, _two_account_store(pro_exhausted=True))
    from agent.credential_pool import load_pool

    pool = load_pool("openai-codex")
    assert pool.model_scope is None

    entry = pool.select()
    assert entry is not None and entry.label == "plus-acct2"


def test_model_scope_matching_ignores_case_and_padding(tmp_path):
    _write_auth_store(tmp_path, _two_account_store(pro_exhausted=True))
    from agent.credential_pool import load_pool

    pool = load_pool("openai-codex")
    pool.model_scope = "  GPT-5.6-Terra  "

    entry = pool.select()
    assert entry is not None and entry.label == "plus-acct2"


def test_allowed_models_survives_a_pool_round_trip(tmp_path):
    """The pin is persisted, not a runtime-only flag — a reload must keep enforcing it."""
    _write_auth_store(tmp_path, _two_account_store())
    from agent.credential_pool import load_pool

    pool = load_pool("openai-codex")
    plus = next(entry for entry in pool.entries() if entry.label == "plus-acct2")
    assert plus.allowed_models == ["gpt-5.6-terra"]
    assert plus.to_dict()["allowed_models"] == ["gpt-5.6-terra"]

    # Mutating an unrelated field must not drop the pin.
    pool.mark_exhausted_and_rotate(
        status_code=429, credential_id=plus.id, error_context={"reason": "usage_limit_reached"},
    )
    reloaded = load_pool("openai-codex")
    plus_again = next(entry for entry in reloaded.entries() if entry.label == "plus-acct2")
    assert plus_again.allowed_models == ["gpt-5.6-terra"]


def test_empty_allowed_models_is_treated_as_unrestricted(tmp_path):
    store = _two_account_store(pro_exhausted=True)
    store["credential_pool"]["openai-codex"][1]["allowed_models"] = []
    _write_auth_store(tmp_path, store)
    from agent.credential_pool import load_pool

    pool = load_pool("openai-codex")
    pool.model_scope = "gpt-5.6-sol"

    entry = pool.select()
    assert entry is not None and entry.label == "plus-acct2"
