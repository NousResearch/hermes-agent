import base64
import json
import threading
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


def _codex_access_token(account_id):
    header = base64.urlsafe_b64encode(b'{"alg":"none","typ":"JWT"}').decode().rstrip("=")
    payload = base64.urlsafe_b64encode(
        json.dumps(
            {
                "https://api.openai.com/auth": {
                    "chatgpt_account_id": account_id,
                }
            }
        ).encode()
    ).decode().rstrip("=")
    return f"{header}.{payload}.signature"


def test_codex_usage_explicit_token_account_id_beats_singleton(monkeypatch, codex_usage_payload):
    calls = []
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeClient(calls, codex_usage_payload),
    )
    monkeypatch.setattr(
        account_usage,
        "_read_codex_tokens",
        lambda: {"tokens": {"account_id": "singleton-account"}},
    )

    account_usage.fetch_account_usage(
        "openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key=_codex_access_token("active-account"),
    )

    assert calls[0]["headers"]["ChatGPT-Account-Id"] == "active-account"


def test_codex_usage_malformed_explicit_token_omits_account_id(monkeypatch, codex_usage_payload):
    calls = []
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeClient(calls, codex_usage_payload),
    )

    snapshot = account_usage.fetch_account_usage(
        "openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="not-a-jwt",
    )

    assert snapshot is not None
    assert "ChatGPT-Account-Id" not in calls[0]["headers"]


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


def test_codex_usage_does_not_swap_to_pool_on_transient_resolver_error(monkeypatch, codex_usage_payload):
    """A transient refresh/network failure (non-AuthError) must NOT silently
    downgrade to a possibly-different pool account. It fails open (no snapshot)
    instead of reporting the wrong account's usage."""
    calls = []
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeClient(calls, codex_usage_payload),
    )
    monkeypatch.setattr(
        account_usage,
        "resolve_codex_runtime_credentials",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("refresh endpoint 503")),
    )

    pool_entry = SimpleNamespace(
        runtime_api_key="pooled-token-WRONG-ACCOUNT",
        runtime_base_url="https://chatgpt.com/backend-api/codex",
    )
    pool = SimpleNamespace(select=lambda: pool_entry)

    import agent.credential_pool as credential_pool

    # If the guard regressed, this pool would be consulted and return a snapshot
    # for the wrong account. It must NOT be.
    monkeypatch.setattr(credential_pool, "load_pool", lambda provider: pool)

    snapshot = account_usage.fetch_account_usage("openai-codex")

    assert snapshot is None
    assert calls == []  # HTTP usage endpoint never hit with a wrong-account token


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


def test_credential_rotation_updates_identity_with_runtime_credential():
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent.api_mode = "chat_completions"
    agent.api_key = "old-token"
    agent.base_url = "https://old.example/v1"
    agent.credential_id = "old-id"
    agent.credential_label = "old-label"
    agent._client_lock = threading.RLock()
    agent._client_kwargs = {}
    agent._apply_client_headers_for_base_url = lambda _base_url: None
    agent._replace_primary_openai_client = lambda **_kwargs: None
    entry = SimpleNamespace(
        runtime_api_key="new-token",
        runtime_base_url="https://new.example/v1/",
        id="new-id",
        label="new-label",
    )

    agent._swap_credential(entry)

    assert agent.api_key == "new-token"
    assert agent.base_url == "https://new.example/v1"
    assert agent.credential_id == "new-id"
    assert agent.credential_label == "new-label"
    assert agent._client_kwargs == {
        "api_key": "new-token",
        "base_url": "https://new.example/v1",
    }


def test_codex_usage_treats_wham_used_percent_as_used_not_remaining(monkeypatch):
    """ChatGPT UI says "left"; /wham/usage.used_percent is already used."""
    payload = {
        "plan_type": "plus",
        "rate_limit": {
            "primary_window": {
                "used_percent": 85,
                "reset_at": 1779846359,
            },
            "secondary_window": {
                "used_percent": 14,
                "reset_at": 1780230796,
            },
        },
        "credits": {"has_credits": False},
    }
    calls = []
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeClient(calls, payload),
    )
    monkeypatch.setattr(
        account_usage,
        "resolve_codex_runtime_credentials",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("explicit auth should be used")),
    )

    snapshot = account_usage.fetch_account_usage(
        "openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="live-agent-token",
    )

    assert snapshot is not None
    assert [window.used_percent for window in snapshot.windows] == [85, 14]
    rendered = "\n".join(account_usage.render_account_usage_lines(snapshot, markdown=True))
    assert "85% used" in rendered
    assert "14% used" in rendered
    assert "15% used" not in rendered
    assert "86% used" not in rendered


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


def test_usage_snapshot_shows_banked_resets_hint(monkeypatch):
    calls = []
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeResetClient(calls, _usage_payload_with_resets(21, 4, 2)),
    )

    snapshot = account_usage.fetch_account_usage(
        "openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="live-agent-token",
    )

    assert snapshot is not None
    rendered = "\n".join(account_usage.render_account_usage_lines(snapshot))
    assert "You have 2 resets banked - use /usage reset to activate" in rendered


def test_usage_snapshot_hides_reset_hint_when_none_banked(monkeypatch, codex_usage_payload):
    calls = []
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeResetClient(calls, codex_usage_payload),
    )

    snapshot = account_usage.fetch_account_usage(
        "openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="live-agent-token",
    )

    assert snapshot is not None
    rendered = "\n".join(account_usage.render_account_usage_lines(snapshot))
    assert "banked" not in rendered


def test_redeem_blocked_when_limits_not_exhausted(monkeypatch):
    calls = []
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeResetClient(calls, _usage_payload_with_resets(60, 30, 2)),
    )

    result = account_usage.redeem_codex_reset_credit(
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="live-agent-token",
    )

    assert result.status == "not_exhausted"
    assert not result.redeemed
    assert "--force" in result.message
    assert "60% used" in result.message
    assert result.available_count == 2
    # The consume endpoint must never be hit — the credit is protected.
    assert [c["method"] for c in calls] == ["GET"]


def test_redeem_force_bypasses_exhaustion_guard(monkeypatch):
    calls = []
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeResetClient(
            calls,
            _usage_payload_with_resets(60, 30, 2),
            consume_payload={"code": "reset", "windows_reset": 2},
        ),
    )

    result = account_usage.redeem_codex_reset_credit(
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="live-agent-token",
        force=True,
    )

    assert result.redeemed
    assert result.windows_reset == 2
    assert result.available_count == 1  # 2 banked - 1 spent
    assert "1 banked reset remaining" in result.message
    post = [c for c in calls if c["method"] == "POST"][0]
    assert post["url"] == "https://chatgpt.com/backend-api/wham/rate-limit-reset-credits/consume"
    assert post["json"]["redeem_request_id"]  # idempotency key present
    assert "credit_id" not in post["json"]


def test_redeem_allowed_without_force_when_window_exhausted(monkeypatch):
    calls = []
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeResetClient(
            calls,
            _usage_payload_with_resets(100, 42, 1),
            consume_payload={"code": "reset", "windows_reset": 2},
        ),
    )

    result = account_usage.redeem_codex_reset_credit(
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="live-agent-token",
    )

    assert result.redeemed
    assert result.available_count == 0
    assert "0 banked resets remaining" in result.message


def test_redeem_refuses_when_no_credits_banked(monkeypatch):
    calls = []
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeResetClient(calls, _usage_payload_with_resets(100, 100, 0)),
    )

    result = account_usage.redeem_codex_reset_credit(
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="live-agent-token",
    )

    assert result.status == "no_credits_banked"
    assert [c["method"] for c in calls] == ["GET"]


def test_redeem_nothing_to_reset_reports_credit_not_spent(monkeypatch):
    calls = []
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeResetClient(
            calls,
            _usage_payload_with_resets(100, 100, 3),
            consume_payload={"code": "nothing_to_reset"},
        ),
    )

    result = account_usage.redeem_codex_reset_credit(
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="live-agent-token",
    )

    assert result.status == "nothing_to_reset"
    assert not result.redeemed
    assert "NOT spent" in result.message
    assert result.available_count == 3


def test_redeem_missing_credentials_reports_unavailable(monkeypatch):
    monkeypatch.setattr(
        account_usage,
        "_resolve_codex_usage_credentials",
        lambda base_url, api_key: (_ for _ in ()).throw(RuntimeError("no creds")),
    )

    result = account_usage.redeem_codex_reset_credit()

    assert result.status == "unavailable"
    assert "hermes auth" in result.message
