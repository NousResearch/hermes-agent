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
    monkeypatch.setattr(
        account_usage,
        "_resolve_codex_usage_credentials",
        lambda base_url, api_key: (_ for _ in ()).throw(RuntimeError("no creds")),
    )

    result = account_usage.redeem_codex_reset_credit()

    assert result.status == "unavailable"
    assert "hermes auth" in result.message


# ── Google Antigravity account quota (`/usage`) ──────────────────────────────


def _antigravity_windows():
    from agent.gemini_cloudcode_adapter import parse_quota_summary

    return parse_quota_summary(
        {
            "buckets": [
                {
                    "bucketId": "gemini-weekly",
                    "remainingFraction": 0.62,
                    "window": "weekly",
                    "resetTime": "2030-08-25T09:00:00Z",
                    "description": "62% left for user@example.com",
                },
                {
                    "bucketId": "gemini-5h",
                    "remainingFraction": 0.25,
                    "window": "5h",
                    "resetTime": "2030-08-22T14:30:00Z",
                },
            ]
        }
    )


def test_antigravity_usage_returns_snapshot_from_resolved_runtime(monkeypatch):
    """`/usage` for google-antigravity: resolved OAuth runtime in, normalized
    quota windows out. Token/project never leave the in-memory call."""
    seen = {}

    def _fake_fetch(*, access_token, project, base_url, timeout=8.0, http_client=None):
        seen["has_token"] = bool(access_token)
        seen["project"] = project
        seen["base_url"] = base_url
        seen["timeout"] = timeout
        return _antigravity_windows()

    monkeypatch.setattr(
        account_usage,
        "_resolve_antigravity_usage_credentials",
        lambda: ("tok-abc", "proj-1", "https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal"),
    )
    monkeypatch.setattr(account_usage, "_antigravity_fetch_quota_summary", _fake_fetch)

    snapshot = account_usage.fetch_account_usage("google-antigravity")

    assert snapshot is not None
    assert snapshot.provider == "google-antigravity"
    assert snapshot.available
    assert [w.label for w in snapshot.windows] == [
        "Gemini Models (weekly)",
        "Gemini Models (5h)",
    ]
    assert seen["has_token"] is True
    assert seen["project"] == "proj-1"
    assert seen["timeout"] > 0
    # No account identity anywhere in the snapshot.
    assert "example.com" not in repr(snapshot)
    assert "proj-1" not in repr(snapshot)


def test_antigravity_usage_unavailable_when_quota_summary_missing(monkeypatch):
    monkeypatch.setattr(
        account_usage,
        "_resolve_antigravity_usage_credentials",
        lambda: ("tok-abc", "proj-1", "https://example.invalid/v1internal"),
    )
    monkeypatch.setattr(
        account_usage,
        "_antigravity_fetch_quota_summary",
        lambda **kwargs: None,
    )

    snapshot = account_usage.fetch_account_usage("google-antigravity")

    assert snapshot is not None
    assert snapshot.windows == ()
    assert not snapshot.available
    assert snapshot.unavailable_reason


def test_antigravity_usage_fails_closed_when_not_signed_in(monkeypatch):
    def _raise():
        raise RuntimeError("not signed in")

    monkeypatch.setattr(
        account_usage, "_resolve_antigravity_usage_credentials", _raise
    )

    assert account_usage.fetch_account_usage("google-antigravity") is None


def test_other_providers_are_unchanged_by_antigravity_wiring(monkeypatch, codex_usage_payload):
    """The Antigravity branch must not intercept any existing provider."""
    monkeypatch.setattr(
        account_usage,
        "_resolve_antigravity_usage_credentials",
        lambda: (_ for _ in ()).throw(AssertionError("antigravity path must not run")),
    )
    calls = []
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeClient(calls, codex_usage_payload),
    )

    snapshot = account_usage.fetch_account_usage(
        "openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="live-agent-token",
    )

    assert snapshot is not None and snapshot.provider == "openai-codex"
    assert account_usage.fetch_account_usage("gemini") is None
    assert account_usage.fetch_account_usage("auto") is None


def test_antigravity_usage_lines_show_windows_without_identity(monkeypatch):
    monkeypatch.setattr(
        account_usage,
        "_resolve_antigravity_usage_credentials",
        lambda: ("tok-abc", "proj-1", "https://example.invalid/v1internal"),
    )
    monkeypatch.setattr(
        account_usage,
        "_antigravity_fetch_quota_summary",
        lambda **kwargs: _antigravity_windows(),
    )

    snapshot = account_usage.fetch_account_usage("google-antigravity")
    lines = account_usage.render_account_usage_lines(snapshot)
    blob = "\n".join(lines)

    assert "google-antigravity" in blob
    assert "Gemini Models (weekly): 62% remaining (38% used)" in blob
    assert "Gemini Models (5h): 25% remaining (75% used)" in blob
    assert blob.count("resets ") == 2
    assert "example.com" not in blob
    assert "proj-1" not in blob
    assert "tok-abc" not in blob
