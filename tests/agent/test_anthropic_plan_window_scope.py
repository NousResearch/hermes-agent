"""A spent Claude plan window refuses every model on the account (#149112).

Anthropic names the window that refused a subscription request in its unified rate-limit headers.
``five_hour`` and ``seven_day`` cover the whole account: the pool must bench that credential for
every model until the window resets, and the first such 429 must rotate instead of retrying the
spent account after up to ``RETRY_AFTER_CAP_S``. A family claim (``seven_day_opus``) and a 429
without the headers (API-key per-model limits) keep the per-model cooldown of #111769.

Real ``agent.credential_pool`` against a temp auth store; the 429 is a real ``anthropic`` error.
"""
import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic
import httpx
import pytest

SPENT, SPARE = "sk-ant-api03-spent-seat", "sk-ant-api03-spare-seat"
FABLE, OPUS = "claude-fable-5-2", "claude-opus-5-5"
RATE_LIMIT_MESSAGE = "This request would exceed your account's rate limit. Please try again later."


def _plan_429(claim: str, reset: float) -> anthropic.RateLimitError:
    """The 429 a spent subscription window returns (headers as captured in #147070 / #149112)."""
    body = {"type": "error", "error": {"type": "rate_limit_error", "message": RATE_LIMIT_MESSAGE}}
    response = httpx.Response(
        429, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"), json=body,
        headers={
            "retry-after": str(int(reset - time.time())),
            "anthropic-ratelimit-unified-status": "rejected",
            "anthropic-ratelimit-unified-representative-claim": claim,
            "anthropic-ratelimit-unified-reset": str(int(reset)),
            "anthropic-ratelimit-unified-5h-status": "allowed",
        },
    )
    return anthropic.RateLimitError(f"Error code: 429 - {body}", response=response, body=body)


@pytest.fixture
def pool(tmp_path, monkeypatch):
    root = tmp_path / "hermes-root"
    root.mkdir()
    (tmp_path / "fakehome").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "fakehome"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "fakehome"))
    for var in ("ANTHROPIC_TOKEN", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(root))
    import hermes_constants
    hermes_constants._default_hermes_root_memo = None  # type: ignore[attr-defined]
    seats = [("spent", SPENT), ("spare", SPARE)]
    (root / "auth.json").write_text(json.dumps({"credential_pool": {"anthropic": [
        {"id": seat, "label": seat, "auth_type": "api_key", "priority": priority,
         "source": "manual", "access_token": key}
        for priority, (seat, key) in enumerate(seats)
    ]}}))
    from agent.credential_pool import load_pool
    return load_pool("anthropic")


def _bench_end(pool) -> float:
    from agent.credential_pool_timestamps import _parse_absolute_timestamp
    spent = next(e for e in pool.entries() if e.id == "spent")
    return _parse_absolute_timestamp(spent.last_error_reset_at)


@pytest.mark.parametrize("claim, account_wide", [
    ("seven_day", True), ("five_hour", True), ("seven_day_opus", False),
])
def test_spent_plan_window_benches_every_model_until_its_reset(pool, claim, account_wide):
    from agent.agent_runtime_helpers import extract_api_error_context
    from agent.credential_pool import load_pool

    reset = time.time() + 63 * 3600
    context = extract_api_error_context(_plan_429(claim, reset))
    rotated = pool.mark_exhausted_and_rotate(
        status_code=429, error_context=context, api_key_hint=SPENT, failure_reason="rate_limit", model=FABLE,
    )

    assert rotated.id == "spare"
    # The next request for ANOTHER model: fill_first must skip a spent account, here and in
    # every other process reading the store; a family claim leaves the account to other models.
    for view in (pool, load_pool("anthropic")):
        assert view.select(model=OPUS).id == ("spare" if account_wide else "spent")
    if account_wide:
        assert abs(_bench_end(pool) - reset) < 5  # benched until the window resets, not one hour


@pytest.mark.parametrize("plan_window, rotates_now", [(True, True), (False, False)])
def test_first_spent_window_429_rotates_instead_of_waiting(pool, plan_window, rotates_now):
    from agent.agent_runtime_helpers import extract_api_error_context, recover_with_credential_pool

    context = (extract_api_error_context(_plan_429("seven_day", time.time() + 63 * 3600)) if plan_window
               else {"reason": "rate_limit_error", "message": RATE_LIMIT_MESSAGE})
    agent = SimpleNamespace(
        provider="anthropic", api_key=SPENT, model=FABLE, _credential_pool=pool,
        _credential_pool_entry_id="spent", _swap_credential=MagicMock(),
    )

    recovered, has_retried_429 = recover_with_credential_pool(
        agent, status_code=429, has_retried_429=False, error_context=context,
    )

    # A spent window cannot reopen within any retry wait; a plain per-minute 429 keeps its one retry.
    assert (recovered, has_retried_429) == ((True, False) if rotates_now else (False, True))
    if rotates_now:
        assert agent._swap_credential.call_args[0][0].id == "spare"


def test_auxiliary_429_keeps_the_plan_window_and_its_reset(pool, monkeypatch):
    """Titles and compression recover through the auxiliary client; its bench must match the main loop's."""
    from agent import auxiliary_client as ac
    from agent.auxiliary_error_policy import classify_attempt

    reset = time.time() + 63 * 3600
    error = _plan_429("seven_day", reset)
    monkeypatch.setattr(ac, "load_pool", lambda provider: pool)
    monkeypatch.setattr(ac, "_evict_cached_clients", lambda provider: None)

    assert ac._recover_provider_pool(
        "anthropic", error, failed_api_key=SPENT, verdict=classify_attempt(error, "anthropic", model=FABLE),
    )

    assert pool.select(model=OPUS).id == "spare"
    assert abs(_bench_end(pool) - reset) < 5
