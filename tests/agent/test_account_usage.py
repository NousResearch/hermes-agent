from types import SimpleNamespace

import pytest

from agent import account_usage


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, calls, payload):
        self.calls = calls
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, headers):
        self.calls.append({"url": url, "headers": headers})
        return _FakeResponse(self.payload)


@pytest.fixture
def codex_usage_payload():
    return {
        "plan_type": "plus",
        "rate_limit": {
            "primary_window": {
                "used_percent": 21,
                "reset_at": 1779846359,
            },
            "secondary_window": {
                "used_percent": 4,
                "reset_at": 1780230796,
            },
        },
        "credits": {"has_credits": False},
    }


def test_codex_usage_prefers_explicit_live_agent_credentials(monkeypatch, codex_usage_payload):
    calls = []
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeClient(calls, codex_usage_payload),
    )
    monkeypatch.setattr(
        account_usage,
        "resolve_codex_runtime_credentials",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("legacy auth should not be used")),
    )

    snapshot = account_usage.fetch_account_usage(
        "openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="live-agent-token",
    )

    assert snapshot is not None
    assert snapshot.provider == "openai-codex"
    assert snapshot.plan == "Plus"
    assert [w.label for w in snapshot.windows] == ["Session", "Weekly"]
    assert snapshot.windows[0].used_percent == 21
    assert calls[0]["url"] == "https://chatgpt.com/backend-api/wham/usage"
    assert calls[0]["headers"]["Authorization"] == "Bearer live-agent-token"


def test_codex_usage_falls_back_to_native_credential_pool(monkeypatch, codex_usage_payload):
    calls = []
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeClient(calls, codex_usage_payload),
    )
    # Pool fallback fires only on AuthError (the documented "no creds" mode of
    # the resolver), NOT on arbitrary exceptions — see the transient-error guard
    # test below.
    monkeypatch.setattr(
        account_usage,
        "resolve_codex_runtime_credentials",
        lambda **kwargs: (_ for _ in ()).throw(
            account_usage.AuthError("no singleton auth", provider="openai-codex", code="codex_auth_missing")
        ),
    )

    pool_entry = SimpleNamespace(
        runtime_api_key="pooled-token",
        runtime_base_url="https://chatgpt.com/backend-api/codex",
    )
    pool = SimpleNamespace(select=lambda: pool_entry)

    import agent.credential_pool as credential_pool

    monkeypatch.setattr(credential_pool, "load_pool", lambda provider: pool)

    snapshot = account_usage.fetch_account_usage("openai-codex")

    assert snapshot is not None
    assert snapshot.windows[0].label == "Session"
    assert snapshot.windows[1].label == "Weekly"
    assert calls[0]["url"] == "https://chatgpt.com/backend-api/wham/usage"
    assert calls[0]["headers"]["Authorization"] == "Bearer pooled-token"
    # Pool creds have no account_id concept — the ChatGPT-Account-Id header must
    # be omitted rather than sent stale/wrong.
    assert "ChatGPT-Account-Id" not in calls[0]["headers"]




def test_codex_usage_account_id_read_failure_keeps_singleton_token(monkeypatch, codex_usage_payload):
    """When the resolver succeeds but the separate account_id read raises, the
    working singleton token must still be used (best-effort account_id), NOT
    abandoned in favor of a header-less pool credential."""
    calls = []
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeClient(calls, codex_usage_payload),
    )
    monkeypatch.setattr(
        account_usage,
        "resolve_codex_runtime_credentials",
        lambda **kwargs: {
            "api_key": "singleton-token",
            "base_url": "https://chatgpt.com/backend-api/codex",
        },
    )
    monkeypatch.setattr(
        account_usage,
        "_read_codex_tokens",
        lambda *a, **k: (_ for _ in ()).throw(
            account_usage.AuthError("partial store", provider="openai-codex", code="codex_auth_invalid_shape")
        ),
    )

    import agent.credential_pool as credential_pool

    monkeypatch.setattr(
        credential_pool,
        "load_pool",
        lambda provider: (_ for _ in ()).throw(AssertionError("pool must not be consulted")),
    )

    snapshot = account_usage.fetch_account_usage("openai-codex")

    assert snapshot is not None
    assert calls[0]["headers"]["Authorization"] == "Bearer singleton-token"
    # account_id read failed → header omitted, but the singleton token is kept.
    assert "ChatGPT-Account-Id" not in calls[0]["headers"]




# ── Banked rate-limit reset credits (`/usage reset`) ─────────────────────────


class _FakeResetClient:
    """GET returns the usage payload; POST returns the consume payload."""

    def __init__(self, calls, usage_payload, consume_payload=None):
        self.calls = calls
        self.usage_payload = usage_payload
        self.consume_payload = consume_payload or {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, headers):
        self.calls.append({"method": "GET", "url": url, "headers": headers})
        return _FakeResponse(self.usage_payload)

    def post(self, url, headers=None, json=None):
        self.calls.append({"method": "POST", "url": url, "headers": headers, "json": json})
        return _FakeResponse(self.consume_payload)


def _usage_payload_with_resets(primary_used, secondary_used, banked):
    return {
        "plan_type": "plus",
        "rate_limit": {
            "primary_window": {"used_percent": primary_used, "reset_at": 1779846359},
            "secondary_window": {"used_percent": secondary_used, "reset_at": 1780230796},
        },
        "rate_limit_reset_credits": {"available_count": banked},
        "credits": {"has_credits": False},
    }
















def test_redeem_missing_credentials_reports_unavailable(monkeypatch):
    """When an explicit api_key IS present but credential resolution fails
    (e.g. refresh error), the helper reports unavailable with a hermes-auth
    hint. The no-api_key path is covered by the account-binding guard tests
    below (which assert the resolver is never even called)."""
    monkeypatch.setattr(
        account_usage,
        "_resolve_codex_usage_credentials",
        lambda base_url, api_key: (_ for _ in ()).throw(RuntimeError("no creds")),
    )

    result = account_usage.redeem_codex_reset_credit(api_key="explicit-but-unresolvable")

    assert result.status == "unavailable"
    assert "hermes auth" in result.message


# ── Account-binding guard: refuse destructive reset without explicit api_key ──
#
# redeem_codex_reset_credit consumes a scarce banked reset. Unlike the read-only
# /usage display (which may fall back to singleton/pool state), the destructive
# path must be bound to an explicit, account-identified api_key forwarded by the
# invoking surface. Without it the helper would silently resolve singleton/pool
# state and spend a reset belonging to an account the caller never authenticated
# as in this session.

def test_redeem_refuses_without_explicit_api_key(monkeypatch):
    """No explicit api_key → fail-closed BEFORE any credential resolution or
    network call. The singleton/pool fallback is acceptable for read-only
    /usage display but NOT for consuming a banked reset."""
    monkeypatch.setattr(
        account_usage,
        "_resolve_codex_usage_credentials",
        lambda base_url, api_key: (_ for _ in ()).throw(
            AssertionError("credential resolution must not run without explicit api_key")
        ),
    )
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("network must not run")),
    )

    result = account_usage.redeem_codex_reset_credit()

    assert result.status == "unavailable"
    assert not result.redeemed
    # User-facing message must guide toward establishing the active account.
    assert "send a message" in result.message.lower() or "authenticate" in result.message.lower()


def test_redeem_refuses_with_empty_api_key(monkeypatch):
    """Empty/whitespace api_key is treated as absent — same fail-closed guard."""
    monkeypatch.setattr(
        account_usage,
        "_resolve_codex_usage_credentials",
        lambda base_url, api_key: (_ for _ in ()).throw(AssertionError("must not resolve")),
    )

    result = account_usage.redeem_codex_reset_credit(api_key="   ")

    assert result.status == "unavailable"
    assert not result.redeemed


def test_redeem_forwards_explicit_api_key_and_base_url(monkeypatch):
    """When an explicit api_key IS provided, the helper resolves credentials
    using it (tier-1 short-circuit) and proceeds normally — proving the guard
    does not block the legitimate path."""
    calls = []
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeResetClient(
            calls,
            _usage_payload_with_resets(100, 100, 1),
            consume_payload={"code": "reset", "windows_reset": 2},
        ),
    )
    # If the guard incorrectly blocks, this would never be reached.
    monkeypatch.setattr(
        account_usage,
        "resolve_codex_runtime_credentials",
        lambda **kw: (_ for _ in ()).throw(AssertionError("resolver should not run with explicit key")),
    )

    result = account_usage.redeem_codex_reset_credit(
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="explicit-session-token",
    )

    assert result.redeemed
    assert calls[0]["headers"]["Authorization"] == "Bearer explicit-session-token"
    assert calls[0]["url"] == "https://chatgpt.com/backend-api/wham/usage"


def test_redeem_clears_codex_pool_cooldowns(monkeypatch):
    """A redeemed reset must unfreeze the same account's stale pool cooldown."""
    calls = []
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeResetClient(
            calls,
            _usage_payload_with_resets(100, 100, 1),
            consume_payload={"code": "reset", "windows_reset": 2},
        ),
    )
    cooldown_clears = []
    monkeypatch.setattr(
        "hermes_cli.auth.clear_codex_pool_quota_cooldowns",
        lambda: cooldown_clears.append(True),
    )

    result = account_usage.redeem_codex_reset_credit(
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="explicit-session-token",
    )

    assert result.redeemed
    assert cooldown_clears == [True]


def test_redeem_request_id_is_fresh_per_call_not_durable(monkeypatch):
    """Clarify: redeem_request_id is a fresh UUID per request (request identity
    for the backend's idempotency check), NOT a durable cross-command
    idempotency key. Two calls produce two different IDs; we do not persist."""
    import uuid

    generated_ids = []
    real_uuid4 = uuid.uuid4

    def _capture_uuid():
        val = real_uuid4()
        generated_ids.append(val)
        return val

    monkeypatch.setattr(account_usage.uuid, "uuid4", _capture_uuid) if hasattr(account_usage, "uuid") else None
    # uuid is imported inside the function body; patch the module-level uuid.
    import agent.account_usage as au_mod
    # The function does `import uuid` locally; patch the global uuid module.
    monkeypatch.setattr("uuid.uuid4", _capture_uuid)

    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeResetClient(
            [],
            _usage_payload_with_resets(100, 100, 2),
            consume_payload={"code": "reset", "windows_reset": 2},
        ),
    )

    r1 = account_usage.redeem_codex_reset_credit(
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="tok",
    )
    r2 = account_usage.redeem_codex_reset_credit(
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="tok",
    )

    assert r1.redeemed
    assert r2.redeemed
    # Two calls → two distinct fresh request IDs (not durable idempotency).
    assert len(generated_ids) >= 2
    assert generated_ids[0] != generated_ids[1]
