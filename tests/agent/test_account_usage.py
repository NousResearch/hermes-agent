import base64
import json
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


def test_codex_usage_for_token_uses_explicit_account_id(monkeypatch, codex_usage_payload):
    calls = []
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeClient(calls, codex_usage_payload),
    )

    snapshot = account_usage.fetch_codex_account_usage_for_token(
        " access-token ",
        base_url="https://chatgpt.com/backend-api/codex",
        account_id="acct_456",
        timeout_seconds=2.5,
    )

    assert snapshot.plan == "Plus"
    assert calls[0]["url"] == "https://chatgpt.com/backend-api/wham/usage"
    assert calls[0]["headers"]["Authorization"] == "Bearer access-token"
    assert calls[0]["headers"]["ChatGPT-Account-Id"] == "acct_456"


def test_codex_usage_for_token_prefers_jwt_account_id_over_stored_hint(
    monkeypatch, codex_usage_payload
):
    calls = []
    payload = {
        "https://api.openai.com/auth": {"chatgpt_account_id": "acct_from_jwt"},
    }
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    token = f"header.{encoded}.signature"
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeClient(calls, codex_usage_payload),
    )

    snapshot = account_usage.fetch_codex_account_usage_for_token(
        token,
        account_id="acct_stale",
    )

    assert snapshot.windows[0].used_percent == 21
    assert calls[0]["headers"]["ChatGPT-Account-Id"] == "acct_from_jwt"


def test_codex_usage_for_token_refuses_untrusted_host(monkeypatch):
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("untrusted host must not receive OAuth token")
        ),
    )

    snapshot = account_usage.fetch_codex_account_usage_for_token(
        "access-token",
        base_url="https://example.invalid/backend-api/codex",
    )

    assert snapshot.unavailable_reason is not None
    assert "non-ChatGPT" in snapshot.unavailable_reason


def test_codex_existing_usage_refuses_untrusted_explicit_base_url(monkeypatch):
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("untrusted host must not receive OAuth token")
        ),
    )

    snapshot = account_usage.fetch_account_usage(
        "openai-codex",
        base_url="https://example.invalid/backend-api/codex",
        api_key="access-token",
    )

    assert snapshot is not None
    assert snapshot.unavailable_reason is not None
    assert "non-ChatGPT" in snapshot.unavailable_reason


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
        account_id="acct-required",
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
    assert calls[0]["headers"]["ChatGPT-Account-Id"] == "acct-required"


def test_codex_usage_runtime_pool_envelope_preserves_selected_account_id(
    monkeypatch, codex_usage_payload
):
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
            "api_key": "shared-opaque-token",
            "account_id": "acct-selected",
            "credential_id": "usable-alias",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "source": "credential_pool",
        },
    )
    monkeypatch.setattr(
        account_usage,
        "_read_codex_tokens",
        lambda: {
            "tokens": {
                "access_token": "shared-opaque-token",
                "account_id": "acct-wrong-singleton-alias",
            }
        },
    )

    snapshot = account_usage.fetch_account_usage("openai-codex")

    assert snapshot is not None
    assert calls[0]["headers"]["Authorization"] == "Bearer shared-opaque-token"
    assert calls[0]["headers"]["ChatGPT-Account-Id"] == "acct-selected"


def test_codex_usage_real_resolver_keeps_duplicate_alias_envelope(
    tmp_path, monkeypatch, codex_usage_payload
):
    calls = []
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "auth.json").write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {},
                "credential_pool": {
                    "openai-codex": [
                        {
                            "id": "cooldown-alias",
                            "source": "manual:device_code",
                            "auth_type": "oauth",
                            "access_token": "shared-opaque-token",
                            "account_id": "acct-wrong",
                            "last_status": "exhausted",
                            "last_error_reset_at": 4_102_444_800,
                        },
                        {
                            "id": "usable-alias",
                            "source": "manual:device_code",
                            "auth_type": "oauth",
                            "access_token": "shared-opaque-token",
                            "account_id": "acct-selected",
                            "last_status": "ok",
                        },
                    ]
                },
            }
        )
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "empty-codex-home"))
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeClient(calls, codex_usage_payload),
    )

    snapshot = account_usage.fetch_account_usage("openai-codex")

    assert snapshot is not None
    assert calls[0]["headers"]["Authorization"] == "Bearer shared-opaque-token"
    assert calls[0]["headers"]["ChatGPT-Account-Id"] == "acct-selected"


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
