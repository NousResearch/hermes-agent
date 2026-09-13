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


@pytest.fixture
def kimi_usage_payload():
    """Minimal synthetic shape returned by the Coding Plan usage endpoint."""
    return {
        "user": {"membership": {"level": "LEVEL_ADVANCED"}},
        "usage": {
            "limit": "100",
            "used": "1",
            "remaining": "99",
            "resetTime": "2026-08-05T21:53:56Z",
        },
        "limits": [
            {
                "window": {"duration": 300, "timeUnit": "TIME_UNIT_MINUTE"},
                "detail": {
                    "limit": "100",
                    "used": "3",
                    "remaining": "97",
                    "resetTime": "2026-07-30T02:53:56Z",
                },
            }
        ],
        "parallel": {"limit": "30"},
    }


def _install_fake_kimi_client(monkeypatch, calls, payload):
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeClient(calls, payload),
    )


def _write_saved_moonshot_provider(base_url):
    """Store a provider under its legacy alias in the isolated HERMES_HOME."""
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "providers:\n"
        "  moonshot:\n"
        "    name: Moonshot saved runtime\n"
        f"    api: {base_url}\n"
        "    api_key: synthetic-test-key\n",
        encoding="utf-8",
    )


def test_kimi_usage_maps_quota_via_real_runtime_resolution(
    monkeypatch, kimi_usage_payload
):
    calls = []
    _install_fake_kimi_client(monkeypatch, calls, kimi_usage_payload)

    snapshot = account_usage.fetch_account_usage(
        "kimi-coding",
        base_url="https://api.kimi.com/coding",
        api_key="synthetic-test-key",
    )

    assert snapshot is not None
    assert snapshot.provider == "kimi-coding"
    assert snapshot.plan == "Level Advanced"
    assert [window.label for window in snapshot.windows] == ["Weekly", "5-hour"]
    assert [window.used_percent for window in snapshot.windows] == pytest.approx([
        1.0,
        3.0,
    ])
    assert snapshot.windows[0].reset_at is not None
    assert snapshot.details == ("Parallel requests: 30 max",)
    assert calls == [
        {
            "url": "https://api.kimi.com/coding/v1/usages",
            "headers": {
                "Authorization": "Bearer synthetic-test-key",
                "Accept": "application/json",
            },
        }
    ]


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.kimi.com/coding",
        "https://api.kimi.com/coding/",
        "https://api.kimi.com/coding/v1",
        "https://api.kimi.com:443/coding",
    ],
)
def test_kimi_coding_base_url_gate_accepts_only_canonical_variants(base_url):
    from utils import is_kimi_coding_base_url

    assert is_kimi_coding_base_url(base_url)


@pytest.mark.parametrize(
    "base_url",
    [
        None,
        "http://api.kimi.com/coding",
        "https://api.moonshot.ai/v1",
        "https://proxy.example.com/coding",
        "https://api.kimi.com.example/coding",
        "https://api.kimi.com/coding/extra",
        "https://api.kimi.com/coding?mode=legacy",
        "https://user@api.kimi.com/coding",
    ],
)
def test_kimi_coding_base_url_gate_rejects_legacy_custom_and_ambiguous_urls(base_url):
    from utils import is_kimi_coding_base_url

    assert not is_kimi_coding_base_url(base_url)


@pytest.mark.parametrize(
    ("base_url", "expect_snapshot"),
    [
        ("https://api.kimi.com/coding", True),
        ("https://api.moonshot.ai/v1", False),
    ],
)
def test_kimi_legacy_alias_uses_its_saved_runtime_record(
    monkeypatch,
    kimi_usage_payload,
    base_url,
    expect_snapshot,
):
    """Exercise the real resolver; remapping to kimi-coding would miss this row."""
    calls = []
    _install_fake_kimi_client(monkeypatch, calls, kimi_usage_payload)
    _write_saved_moonshot_provider(base_url)

    snapshot = account_usage.fetch_account_usage("moonshot")

    if expect_snapshot:
        assert snapshot is not None
        assert snapshot.provider == "kimi-coding"
        assert calls[0]["url"] == "https://api.kimi.com/coding/v1/usages"
    else:
        assert snapshot is None
        assert calls == []


@pytest.mark.parametrize(
    ("api_key", "expect_snapshot"),
    [
        ("sk-kimi-synthetic-test", True),
        ("legacy-moonshot-test-key", False),
    ],
)
def test_kimi_builtin_alias_distinguishes_coding_plan_from_legacy_key(
    monkeypatch,
    kimi_usage_payload,
    api_key,
    expect_snapshot,
):
    """Use the real auth/runtime chain rather than replacing its resolver."""
    calls = []
    _install_fake_kimi_client(monkeypatch, calls, kimi_usage_payload)
    monkeypatch.setenv("KIMI_API_KEY", api_key)
    monkeypatch.delenv("KIMI_CODING_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_BASE_URL", raising=False)

    snapshot = account_usage.fetch_account_usage("moonshot")

    assert (snapshot is not None) is expect_snapshot
    if expect_snapshot:
        assert calls[0]["url"] == "https://api.kimi.com/coding/v1/usages"
    else:
        assert calls == []


def test_kimi_custom_runtime_fails_closed_without_request(monkeypatch):
    calls = []
    _install_fake_kimi_client(monkeypatch, calls, {})

    snapshot = account_usage.fetch_account_usage(
        "kimi",
        base_url="https://proxy.example.com/coding",
        api_key="synthetic-test-key",
    )

    assert snapshot is None
    assert calls == []


def test_kimi_over_quota_percent_is_preserved_and_renders_safely(monkeypatch):
    calls = []
    _install_fake_kimi_client(
        monkeypatch,
        calls,
        {"usage": {"limit": "100", "used": "130"}},
    )

    snapshot = account_usage.fetch_account_usage(
        "kimi-coding",
        base_url="https://api.kimi.com/coding",
        api_key="synthetic-test-key",
    )

    assert account_usage._percent_from_counts({"limit": "100", "used": "130"}) == 130.0
    assert snapshot is not None
    assert snapshot.windows[0].used_percent == 130.0
    assert (
        "Weekly: 0% remaining (130% used)"
        in account_usage.render_account_usage_lines(snapshot)
    )


def test_kimi_usage_tolerates_missing_optional_sections(monkeypatch):
    calls = []
    _install_fake_kimi_client(
        monkeypatch,
        calls,
        {"usage": {"limit": "100", "used": "40"}},
    )

    snapshot = account_usage.fetch_account_usage(
        "kimi-coding",
        base_url="https://api.kimi.com/coding/v1",
        api_key="synthetic-test-key",
    )

    assert snapshot is not None
    assert [window.label for window in snapshot.windows] == ["Weekly"]
    assert snapshot.windows[0].used_percent == pytest.approx(40.0)
    assert snapshot.windows[0].reset_at is None
    assert snapshot.details == ()
