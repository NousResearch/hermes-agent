"""A spent Anthropic subscription window benches the credential that reported it.

Anthropic subscription traffic carries ``anthropic-ratelimit-unified-*`` on every response.
These tests drive a real ``CredentialPool`` loaded from a temp ``HERMES_HOME`` and assert the
contract: a window reported ``rejected`` keeps the credential out of selection until that
window's reset, for every Claude model, whether the report arrived on a 200 or a 429.
Header shapes mirror live responses (a weekly window spent, overage disabled by the org).
"""

import json
import time
from types import SimpleNamespace

import pytest

from agent.credential_pool_quota_window import spent_quota_reset_at

WEEK_RESET = int(time.time()) + 3 * 86400
SESSION_RESET = int(time.time()) + 2 * 3600


def _unified(*, week_status="allowed", week_util="0.4", overall=None):
    headers = {
        "anthropic-ratelimit-unified-5h-status": "allowed",
        "anthropic-ratelimit-unified-5h-utilization": "0.0",
        "anthropic-ratelimit-unified-5h-reset": str(SESSION_RESET),
        "anthropic-ratelimit-unified-7d-status": week_status,
        "anthropic-ratelimit-unified-7d-utilization": week_util,
        "anthropic-ratelimit-unified-7d-reset": str(WEEK_RESET),
        # Present on every account whose org disables overage; says nothing about quota.
        "anthropic-ratelimit-unified-overage-status": "rejected",
        "anthropic-ratelimit-unified-overage-disabled-reason": "org_level_disabled",
        "anthropic-ratelimit-unified-status": overall or week_status,
        "anthropic-ratelimit-unified-reset": str(WEEK_RESET if week_status == "rejected" else SESSION_RESET),
    }
    return headers


class TestSpentQuotaResetAt:
    def test_rejected_window_yields_its_reset(self):
        assert spent_quota_reset_at(_unified(week_status="rejected", week_util="1.0")) == WEEK_RESET

    def test_healthy_account_with_overage_disabled_is_not_spent(self):
        assert spent_quota_reset_at(_unified()) is None

    def test_full_utilization_without_rejection_is_not_spent(self):
        """Overage-enabled accounts keep serving past 100%; only the provider's verdict counts."""
        assert spent_quota_reset_at(_unified(week_util="1.0")) is None

    def test_reset_already_past_is_ignored(self):
        headers = _unified(week_status="rejected", week_util="1.0")
        headers["anthropic-ratelimit-unified-reset"] = "1000"
        headers["anthropic-ratelimit-unified-7d-reset"] = "1000"
        assert spent_quota_reset_at(headers) is None

    def test_non_anthropic_headers_are_ignored(self):
        assert spent_quota_reset_at({"x-ratelimit-remaining-requests": "0"}) is None
        assert spent_quota_reset_at(None) is None


def _seed_pool(tmp_path, monkeypatch, *, twin=False):
    """``twin`` adds a second entry backed by the spent entry's token (one account, two rows)."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    entry = lambda cid, prio, token=None: {  # noqa: E731
        "id": cid, "label": cid, "auth_type": "oauth", "priority": prio,
        "source": "manual:hermes_pkce", "access_token": token or f"tok-{cid}",
        "refresh_token": f"ref-{cid}", "expires_at_ms": int((time.time() + 86400) * 1000),
    }
    entries = [entry("spent", 0), entry("fresh", 2)]
    if twin:
        entries.insert(1, entry("twin", 1, token="tok-spent"))
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1, "providers": {},
        "credential_pool": {"anthropic": entries},
    }))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    from agent.credential_pool import load_pool
    return load_pool("anthropic")


def _status(pool, cid):
    return next(e for e in pool._entries if e.id == cid)


class TestBenchOnSuccessfulResponse:
    def _agent(self, pool, entry_id, token):
        from agent.rate_limit_credits import RateLimitCreditsMixin

        agent = RateLimitCreditsMixin()
        agent._credential_pool = pool
        agent._credential_pool_entry_id = entry_id
        agent._anthropic_api_key = token
        agent.api_key = token
        return agent

    def test_rejected_window_on_a_200_benches_until_reset_across_reload(self, tmp_path, monkeypatch):
        pool = _seed_pool(tmp_path, monkeypatch)
        assert pool.select().id == "spent"
        agent = self._agent(pool, "spent", "tok-spent")

        agent._bench_spent_subscription_window(
            SimpleNamespace(headers=_unified(week_status="rejected", week_util="1.0")))

        from agent.credential_pool import _exhausted_until, load_pool
        assert _exhausted_until(_status(pool, "spent")) == WEEK_RESET
        # Persisted: a fresh process skips the spent credential at selection.
        reloaded = load_pool("anthropic")
        assert reloaded.select().id == "fresh"
        assert reloaded.select(model="claude-sonnet-x").id == "fresh"

    def test_bench_covers_every_entry_backed_by_the_spent_token(self, tmp_path, monkeypatch):
        """The window belongs to the account, so a second row on the same token is spent too;
        leaving it selectable hands the spent account straight back."""
        pool = _seed_pool(tmp_path, monkeypatch, twin=True)
        agent = self._agent(pool, "spent", "tok-spent")

        agent._bench_spent_subscription_window(
            SimpleNamespace(headers=_unified(week_status="rejected", week_util="1.0")))

        from agent.credential_pool import _exhausted_until, load_pool
        assert _exhausted_until(_status(pool, "twin")) == WEEK_RESET
        assert _status(pool, "fresh").last_status in (None, "ok")
        assert load_pool("anthropic").select().id == "fresh"

    def test_healthy_headers_change_nothing(self, tmp_path, monkeypatch):
        pool = _seed_pool(tmp_path, monkeypatch)
        agent = self._agent(pool, "spent", "tok-spent")
        agent._bench_spent_subscription_window(SimpleNamespace(headers=_unified()))
        assert _status(pool, "spent").last_status in (None, "ok")

    def test_benches_the_key_that_was_sent_not_a_stale_entry_id(self, tmp_path, monkeypatch):
        """The per-request refresh can swap the token under a stale entry id; the bench must land
        on the credential whose quota the headers describe."""
        pool = _seed_pool(tmp_path, monkeypatch)
        agent = self._agent(pool, "fresh", "tok-spent")
        agent._bench_spent_subscription_window(
            SimpleNamespace(headers=_unified(week_status="rejected", week_util="1.0")))
        assert _status(pool, "spent").last_status == "exhausted"
        assert _status(pool, "fresh").last_status in (None, "ok")


class TestSpentWindowOn429:
    def _error(self):
        err = Exception("Error code: 429")
        err.status_code = 429
        err.response = SimpleNamespace(
            headers={**_unified(week_status="rejected", week_util="1.0"), "retry-after": "259200"})
        err.body = {"type": "error", "error": {
            "type": "rate_limit_error",
            "message": "This request would exceed your account's rate limit. Please try again later.",
        }}
        return err

    def test_bench_is_account_wide_not_per_model(self, tmp_path, monkeypatch):
        """A generic Anthropic 429 cools down one model; a spent plan window must not, or the
        next Claude model is handed the same spent credential."""
        from agent.agent_runtime_helpers import extract_api_error_context
        from agent.credential_pool import _exhausted_until

        pool = _seed_pool(tmp_path, monkeypatch)
        assert pool.select().id == "spent"
        next_entry = pool.mark_exhausted_and_rotate(
            status_code=429, error_context=extract_api_error_context(self._error()),
            credential_id="spent", model="claude-opus-x",
        )
        assert next_entry is not None and next_entry.id == "fresh"
        assert _exhausted_until(_status(pool, "spent")) == WEEK_RESET
        assert pool.select(model="claude-sonnet-x").id == "fresh"

    @pytest.mark.parametrize("has_retried", [False, True])
    def test_first_429_rotates_without_a_same_key_retry(self, tmp_path, monkeypatch, has_retried):
        from agent.agent_runtime_helpers import _recover_rate_limit, extract_api_error_context

        pool = _seed_pool(tmp_path, monkeypatch)
        rotated = []
        recovered, _ = _recover_rate_limit(
            pool, has_retried_429=has_retried, error_context=extract_api_error_context(self._error()),
            api_key_hint="tok-spent", credential_id="spent",
            rotate_and_swap=lambda status, label: rotated.append(label) or True,
        )
        assert recovered is True and rotated
