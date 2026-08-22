"""RED/GREEN tests for Claude-p subscription usage-aware routing.

Covers:
  * a narrow subscription-only token resolver for ``claude -p`` (never an
    API key or an ambiguous/unrecognized token; never persists/logs it),
  * ``fetch_account_usage("claude-p")`` mapping to the Anthropic OAuth usage
    endpoint via that resolver and projecting to provider ``claude-p``,
  * the delegation usage cache treating ``claude-p`` like any other provider
    (safe source/window allowlists, no forced-unknown special case, at most
    one background refresh),
  * route-scoped ``usage_window_prefixes`` binding Opus/Sonnet/Haiku routes
    to the correct windows without cross-family leakage,
  * the selector preferring a known-remaining claude-p route over lower
    remaining OpenAI/Antigravity routes for an equivalent eligible task.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timedelta, timezone

import pytest

from agent import account_usage
from agent import delegation_usage_cache as duc
from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
from agent.delegation_routing import (
    DelegationRoute,
    ModelClass,
    ProviderUsage,
    RouteCatalog,
    RouteRequest,
    TaskDifficulty,
    UsageView,
    select_route,
)


# ---------------------------------------------------------------------------
# 1. Narrow subscription-only token resolver
# ---------------------------------------------------------------------------


class TestResolveClaudeSubscriptionToken:
    def test_rejects_bare_api_key_even_when_only_source_available(self, monkeypatch):
        monkeypatch.setattr(account_usage, "resolve_anthropic_token", lambda: None, raising=False)
        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials", lambda: None
        )
        monkeypatch.setattr("agent.anthropic_adapter._get_secret", lambda name, default=None: (
            "sk-ant-api03-realkeyvalue" if name == "ANTHROPIC_API_KEY" else default
        ))
        from agent.anthropic_adapter import resolve_claude_subscription_token

        assert resolve_claude_subscription_token() is None

    def test_prefers_claude_code_credential_file_over_env(self, monkeypatch):
        from agent import anthropic_adapter as aa

        monkeypatch.setattr(
            aa, "read_claude_code_credentials",
            lambda: {"accessToken": "cc-file-token", "expiresAt": 0, "source": "claude_code_credentials_file"},
        )
        monkeypatch.setattr(aa, "is_claude_code_token_valid", lambda creds: True)
        monkeypatch.setattr(aa, "_getenv", lambda name, default="": "")

        token = aa.resolve_claude_subscription_token()
        assert token == "cc-file-token"

    def test_prefers_macos_keychain_credential_via_existing_helper(self, monkeypatch):
        from agent import anthropic_adapter as aa

        monkeypatch.setattr(
            aa, "read_claude_code_credentials",
            lambda: {"accessToken": "cc-keychain-token", "expiresAt": 0, "source": "macos_keychain"},
        )
        monkeypatch.setattr(aa, "is_claude_code_token_valid", lambda creds: True)
        monkeypatch.setattr(aa, "_getenv", lambda name, default="": "")

        token = aa.resolve_claude_subscription_token()
        assert token == "cc-keychain-token"

    def test_rejects_api_key_shape_even_from_claude_credential_store(self, monkeypatch):
        from agent import anthropic_adapter as aa

        monkeypatch.setattr(
            aa,
            "read_claude_code_credentials",
            lambda: {
                "accessToken": "sk-ant-api-not-subscription",
                "expiresAt": 0,
                "source": "macos_keychain",
            },
        )
        monkeypatch.setattr(aa, "is_claude_code_token_valid", lambda creds: True)
        monkeypatch.setattr(aa, "_getenv", lambda name, default="": "")

        assert aa.resolve_claude_subscription_token() is None

    def test_rejects_managed_key_shape_from_claude_credential_store(self, monkeypatch):
        from agent import anthropic_adapter as aa

        monkeypatch.setattr(
            aa,
            "read_claude_code_credentials",
            lambda: {
                "accessToken": "sk-ant-managed-not-subscription",
                "expiresAt": 0,
                "source": "macos_keychain",
            },
        )
        monkeypatch.setattr(aa, "is_claude_code_token_valid", lambda creds: True)
        monkeypatch.setattr(aa, "_getenv", lambda name, default="": "")

        assert aa.resolve_claude_subscription_token() is None

    def test_accepts_claude_setup_token_shape_from_credential_store(self, monkeypatch):
        from agent import anthropic_adapter as aa

        monkeypatch.setattr(
            aa,
            "read_claude_code_credentials",
            lambda: {
                "accessToken": "sk-ant-oat01-subscription",
                "expiresAt": 0,
                "source": "claude_code_credentials_file",
            },
        )
        monkeypatch.setattr(aa, "is_claude_code_token_valid", lambda creds: True)
        monkeypatch.setattr(aa, "_getenv", lambda name, default="": "")

        assert aa.resolve_claude_subscription_token() == "sk-ant-oat01-subscription"

    def test_accepts_claude_code_oauth_token_env_when_recognized_oauth(self, monkeypatch):
        from agent import anthropic_adapter as aa

        monkeypatch.setattr(aa, "read_claude_code_credentials", lambda: None)

        def _getenv(name, default=""):
            if name == "CLAUDE_CODE_OAUTH_TOKEN":
                return "cc-oauth-abc123"
            return default

        monkeypatch.setattr(aa, "_getenv", _getenv)
        token = aa.resolve_claude_subscription_token()
        assert token == "cc-oauth-abc123"

    def test_rejects_claude_code_oauth_token_env_when_not_recognized_oauth_shape(self, monkeypatch):
        from agent import anthropic_adapter as aa

        monkeypatch.setattr(aa, "read_claude_code_credentials", lambda: None)

        def _getenv(name, default=""):
            if name == "CLAUDE_CODE_OAUTH_TOKEN":
                return "not-an-oauth-shaped-value"
            return default

        monkeypatch.setattr(aa, "_getenv", _getenv)
        token = aa.resolve_claude_subscription_token()
        assert token is None

    def test_never_falls_back_to_ambiguous_anthropic_token_env(self, monkeypatch):
        from agent import anthropic_adapter as aa

        monkeypatch.setattr(aa, "read_claude_code_credentials", lambda: None)

        def _getenv(name, default=""):
            if name == "ANTHROPIC_TOKEN":
                return "sk-ant-oat-ambiguous-source"
            return default

        monkeypatch.setattr(aa, "_getenv", _getenv)
        token = aa.resolve_claude_subscription_token()
        assert token is None

    def test_never_falls_back_to_anthropic_api_key_env(self, monkeypatch):
        from agent import anthropic_adapter as aa

        monkeypatch.setattr(aa, "read_claude_code_credentials", lambda: None)

        def _getenv(name, default=""):
            if name == "ANTHROPIC_API_KEY":
                return "sk-ant-api03-shouldneverbeused"
            return default

        monkeypatch.setattr(aa, "_getenv", _getenv)
        token = aa.resolve_claude_subscription_token()
        assert token is None

    def test_never_falls_back_to_credential_pool(self, monkeypatch):
        """The subscription resolver must not silently pick up a pooled API-key/OAuth entry."""
        from agent import anthropic_adapter as aa

        monkeypatch.setattr(aa, "read_claude_code_credentials", lambda: None)
        monkeypatch.setattr(aa, "_getenv", lambda name, default="": "")
        pool_called = []
        monkeypatch.setattr(
            aa, "_resolve_anthropic_pool_token", lambda: (pool_called.append(1) or "pool-token")
        )

        token = aa.resolve_claude_subscription_token()
        assert token is None
        assert pool_called == []

    def test_does_not_refresh_expired_keychain_credential(self, monkeypatch):
        from agent import anthropic_adapter as aa

        expired = {
            "accessToken": "cc-old",
            "refreshToken": "cc-refresh",
            "expiresAt": 1,
            "source": "macos_keychain",
        }
        monkeypatch.setattr(aa, "read_claude_code_credentials", lambda: expired)
        monkeypatch.setattr(aa, "is_claude_code_token_valid", lambda creds: False)
        refreshes = []
        monkeypatch.setattr(
            aa, "_refresh_oauth_token", lambda creds: refreshes.append(creds) or "cc-new"
        )
        monkeypatch.setattr(aa, "_getenv", lambda name, default="": "")

        assert aa.resolve_claude_subscription_token() is None
        assert refreshes == []

    def test_refreshes_expired_claude_code_credential_file(self, monkeypatch):
        from agent import anthropic_adapter as aa

        expired = {"accessToken": "old", "refreshToken": "r1", "expiresAt": 1, "source": "claude_code_credentials_file"}
        monkeypatch.setattr(aa, "read_claude_code_credentials", lambda: expired)
        monkeypatch.setattr(aa, "is_claude_code_token_valid", lambda creds: creds.get("accessToken") != "old")
        monkeypatch.setattr(aa, "_refresh_oauth_token", lambda creds: "cc-refreshed-token")
        monkeypatch.setattr(aa, "_getenv", lambda name, default="": "")

        token = aa.resolve_claude_subscription_token()
        assert token == "cc-refreshed-token"

    def test_refresh_returns_none_when_rotated_credentials_cannot_persist(
        self, monkeypatch
    ):
        from agent import anthropic_adapter as aa

        stale = {
            "accessToken": "cc-old",
            "refreshToken": "cc-refresh",
            "expiresAt": 1,
        }
        monkeypatch.setattr(aa, "read_claude_code_credentials", lambda: stale)
        monkeypatch.setattr(
            aa,
            "refresh_anthropic_oauth_pure",
            lambda *_args, **_kwargs: {
                "access_token": "cc-new",
                "refresh_token": "cc-new-refresh",
                "expires_at_ms": 9999999999999,
            },
        )
        monkeypatch.setattr(
            aa, "_write_claude_code_credentials", lambda *_args, **_kwargs: False
        )

        assert aa._refresh_oauth_token(stale) is None

    def test_refresh_failure_does_not_log_secret_bearing_exception(
        self, monkeypatch, caplog
    ):
        from agent import anthropic_adapter as aa

        expired = {
            "accessToken": "cc-old",
            "refreshToken": "cc-refresh-secret",
            "expiresAt": 1,
        }
        monkeypatch.setattr(aa, "read_claude_code_credentials", lambda: expired)
        monkeypatch.setattr(aa, "is_claude_code_token_valid", lambda creds: False)
        monkeypatch.setattr(
            aa,
            "refresh_anthropic_oauth_pure",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("account@example.com cc-refresh-secret")
            ),
        )
        monkeypatch.setattr(aa, "_getenv", lambda name, default="": "")

        with caplog.at_level("DEBUG"):
            assert aa.resolve_claude_subscription_token() is None

        assert "account@example.com" not in caplog.text
        assert "cc-refresh-secret" not in caplog.text

    def test_refresh_endpoint_failure_logs_exception_type_only(
        self, monkeypatch, caplog
    ):
        from agent import anthropic_adapter as aa

        def _boom(*_args, **_kwargs):
            raise RuntimeError("account@example.com cc-refresh-secret")

        monkeypatch.setattr(urllib.request, "urlopen", _boom)

        with caplog.at_level("DEBUG"), pytest.raises(RuntimeError):
            aa.refresh_anthropic_oauth_pure("cc-refresh-secret")

        assert "RuntimeError" in caplog.text
        assert "account@example.com" not in caplog.text
        assert "cc-refresh-secret" not in caplog.text

    def test_authorization_exchange_failure_does_not_disclose_exception(
        self, monkeypatch, caplog, capsys
    ):
        from agent import anthropic_adapter as aa
        from hermes_cli import auth

        monkeypatch.setattr(aa, "_generate_pkce", lambda: ("verifier", "challenge"))
        monkeypatch.setattr("secrets.token_urlsafe", lambda _size: "fixed-state")
        monkeypatch.setattr("builtins.input", lambda _prompt: "auth-code#fixed-state")
        monkeypatch.setattr(auth, "_can_open_graphical_browser", lambda: False)

        def _boom(*_args, **_kwargs):
            raise RuntimeError("account@example.com exchange-secret")

        monkeypatch.setattr(urllib.request, "urlopen", _boom)

        with caplog.at_level("DEBUG"):
            assert aa.run_hermes_oauth_login_pure() is None

        output = capsys.readouterr().out + caplog.text
        assert "account@example.com" not in output
        assert "exchange-secret" not in output


# ---------------------------------------------------------------------------
# 2. fetch_account_usage("claude-p") projection
# ---------------------------------------------------------------------------


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _Client:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, headers=None):
        return _Response(self._payload)


class TestFetchAccountUsageClaudeP:
    def test_returns_none_without_a_subscription_token(self, monkeypatch):
        monkeypatch.setattr(account_usage, "resolve_claude_subscription_token", lambda: None, raising=False)
        assert account_usage.fetch_account_usage("claude-p") is None

    def test_never_calls_the_ordinary_anthropic_resolver(self, monkeypatch):
        """claude-p must use its own subscription-only resolver, not resolve_anthropic_token
        (which can prefer ANTHROPIC_API_KEY)."""
        called = []
        monkeypatch.setattr(
            account_usage, "resolve_anthropic_token", lambda: (called.append(1) or "sk-ant-api03-x"), raising=False
        )
        monkeypatch.setattr(
            account_usage, "resolve_claude_subscription_token", lambda: "cc-subscription-token", raising=False
        )
        monkeypatch.setattr(
            account_usage.httpx, "Client",
            lambda timeout=15.0: _Client({}),
        )
        account_usage.fetch_account_usage("claude-p")
        assert called == []

    def test_maps_oauth_usage_payload_to_claude_p_provider(self, monkeypatch):
        monkeypatch.setattr(
            account_usage, "resolve_claude_subscription_token", lambda: "cc-subscription-token", raising=False
        )
        payload = {
            "five_hour": {"utilization": 42.0, "resets_at": "2026-08-22T18:00:00Z"},
            "seven_day": {"utilization": 0.42},
            "seven_day_opus": {"utilization": 55.0},
            "seven_day_sonnet": {"utilization": 5.0},
        }
        monkeypatch.setattr(account_usage.httpx, "Client", lambda timeout=15.0: _Client(payload))

        snapshot = account_usage.fetch_account_usage("claude-p")
        assert snapshot is not None
        assert snapshot.provider == "claude-p"
        labels = {w.label: w.used_percent for w in snapshot.windows}
        assert labels["Current session"] == pytest.approx(42.0)
        assert labels["Current week"] == pytest.approx(0.42)
        assert labels["Opus week"] == pytest.approx(55.0)
        assert labels["Sonnet week"] == pytest.approx(5.0)

    def test_drops_non_finite_and_out_of_range_utilization(self, monkeypatch):
        monkeypatch.setattr(
            account_usage,
            "resolve_claude_subscription_token",
            lambda: "cc-subscription-token",
            raising=False,
        )
        payload = {
            "five_hour": {"utilization": float("nan")},
            "seven_day": {"utilization": -1},
            "seven_day_opus": {"utilization": 101},
            "seven_day_sonnet": {"utilization": 1.0},
        }
        monkeypatch.setattr(
            account_usage.httpx, "Client", lambda timeout=15.0: _Client(payload)
        )

        snapshot = account_usage.fetch_account_usage("claude-p")
        assert snapshot is not None
        assert [(w.label, w.used_percent) for w in snapshot.windows] == [
            ("Sonnet week", 1.0)
        ]

    def test_preserves_existing_anthropic_provider_behavior(self, monkeypatch):
        """provider='anthropic' must still use resolve_anthropic_token, unaffected by the new resolver."""
        called = []
        monkeypatch.setattr(
            account_usage, "resolve_anthropic_token", lambda: (called.append(1) or "sk-ant-oat-abc"), raising=False
        )
        monkeypatch.setattr(account_usage, "_is_oauth_token", lambda tok: True, raising=False)
        monkeypatch.setattr(
            account_usage.httpx, "Client",
            lambda timeout=15.0: _Client({"five_hour": {"utilization": 0.2}}),
        )
        snapshot = account_usage.fetch_account_usage("anthropic")
        assert called == [1]
        assert snapshot is not None
        assert snapshot.provider == "anthropic"

    def test_network_failure_returns_none_never_raises(self, monkeypatch):
        monkeypatch.setattr(
            account_usage, "resolve_claude_subscription_token", lambda: "cc-subscription-token", raising=False
        )

        class _BoomClient:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def get(self, *a, **k):
                raise RuntimeError("network down: https://api?key=SECRET")

        monkeypatch.setattr(account_usage.httpx, "Client", lambda timeout=15.0: _BoomClient())
        assert account_usage.fetch_account_usage("claude-p") is None


# ---------------------------------------------------------------------------
# 3. Cache security — safe source/window allowlists for claude-p
# ---------------------------------------------------------------------------


def _claude_p_snapshot(*, five_hour=20.0, seven_day=10.0, opus=30.0, sonnet=15.0, fetched_at=None):
    return AccountUsageSnapshot(
        provider="claude-p",
        source="oauth_usage_api",
        fetched_at=fetched_at or datetime.now(timezone.utc),
        windows=(
            AccountUsageWindow(label="Current session", used_percent=five_hour),
            AccountUsageWindow(label="Current week", used_percent=seven_day),
            AccountUsageWindow(label="Opus week", used_percent=opus),
            AccountUsageWindow(label="Sonnet week", used_percent=sonnet),
        ),
    )


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(duc, "_cache_path", lambda: tmp_path / "usage.json")
    duc.reset_refresh_state()
    return duc


class TestClaudePCacheProjection:
    def test_safe_source_and_window_labels_survive_projection(self, cache):
        record = cache.project_snapshot(_claude_p_snapshot())
        assert record["source"] == "oauth_usage_api"
        labels = {w["label"] for w in record["windows"]}
        assert labels == {"Current session", "Current week", "Opus week", "Sonnet week"}

    def test_unrecognized_source_and_labels_are_dropped(self, cache):
        snapshot = AccountUsageSnapshot(
            provider="claude-p",
            source="account bob@example.com",
            fetched_at=datetime.now(timezone.utc),
            windows=(
                AccountUsageWindow(label="bob@example.com session", used_percent=10.0),
            ),
        )
        record = cache.project_snapshot(snapshot)
        assert record["source"] == ""
        assert record["windows"][0]["label"] == ""
        assert "bob@example.com" not in json.dumps(record)

    def test_only_allowlisted_fields_persisted(self, cache):
        record = cache.project_snapshot(_claude_p_snapshot())
        assert set(record) == {"provider", "fetched_at", "source", "windows"}
        for window in record["windows"]:
            assert set(window) == {"label", "used_percent", "remaining_percent", "reset_at"}


# ---------------------------------------------------------------------------
# 4. build_route_usage_view no longer forces claude-p to unknown
# ---------------------------------------------------------------------------


class _Route:
    def __init__(self, route_id, provider, backend="native", prefixes=()):
        self.id = route_id
        self.provider = provider
        self.backend = backend
        self.usage_window_prefixes = prefixes


class TestBuildRouteUsageViewClaudeP:
    def test_claude_p_reads_cache_like_any_other_provider(self, cache):
        cache.store_snapshot(_claude_p_snapshot(five_hour=20.0))
        view = cache.build_route_usage_view(
            [_Route("claude-sonnet-coding", "claude-p", backend="claude-p", prefixes=("Current",))],
            ttl_seconds=300,
            stale_seconds=1800,
            refresh=False,
        )
        entry = view.entries["claude-sonnet-coding"]
        assert entry.freshness == "fresh"
        assert entry.remaining_percent == 80.0

    def test_claude_p_schedules_at_most_one_background_refresh(self, cache, monkeypatch):
        scheduled = []
        monkeypatch.setattr(cache, "_spawn_refresh", lambda p: scheduled.append(p))
        cache.build_route_usage_view(
            [
                _Route("claude-sonnet-coding", "claude-p", backend="claude-p"),
                _Route("claude-opus-coding", "claude-p", backend="claude-p"),
            ],
            ttl_seconds=300,
            stale_seconds=1800,
            refresh=True,
        )
        assert scheduled == ["claude-p"]

    def test_claude_p_missing_cache_is_unknown_not_fabricated(self, cache):
        view = cache.build_route_usage_view(
            [_Route("claude-sonnet-coding", "claude-p", backend="claude-p")],
            ttl_seconds=300,
            stale_seconds=1800,
            refresh=False,
        )
        entry = view.entries["claude-sonnet-coding"]
        assert entry.freshness == "unknown"
        assert entry.remaining_percent is None

    def test_claude_p_failed_refresh_retains_stale_cache(self, cache, monkeypatch):
        fetched = datetime.now(timezone.utc) - timedelta(seconds=900)
        cache.store_snapshot(_claude_p_snapshot(five_hour=20.0, fetched_at=fetched))

        def _boom(provider):
            raise RuntimeError("network down")

        monkeypatch.setattr(cache, "_fetch_account_usage", _boom)
        cache.refresh_provider_now("claude-p")

        view = cache.build_route_usage_view(
            [_Route("claude-sonnet-coding", "claude-p", backend="claude-p", prefixes=("Current",))],
            ttl_seconds=300,
            stale_seconds=1800,
            refresh=False,
        )
        entry = view.entries["claude-sonnet-coding"]
        assert entry.freshness == "stale"
        assert entry.remaining_percent == 80.0


# ---------------------------------------------------------------------------
# 5. Route-scoped windows for Claude model families — no cross-family leak
# ---------------------------------------------------------------------------


class TestClaudeModelFamilyWindowScoping:
    def _store(self, cache):
        cache.store_snapshot(
            _claude_p_snapshot(five_hour=10.0, seven_day=20.0, opus=95.0, sonnet=5.0)
        )

    def test_opus_route_bound_by_general_and_opus_windows(self, cache):
        self._store(cache)
        view = cache.build_route_usage_view(
            [_Route("claude-opus-5-coding", "claude-p", backend="claude-p",
                     prefixes=("Current session", "Current week", "Opus week"))],
            ttl_seconds=300, stale_seconds=1800, refresh=False,
        )
        # worst-of: min(90, 80, 5) = 5 (Opus week is the binding constraint)
        assert view.entries["claude-opus-5-coding"].remaining_percent == 5.0

    def test_sonnet_route_bound_by_general_and_sonnet_windows(self, cache):
        self._store(cache)
        view = cache.build_route_usage_view(
            [_Route("claude-sonnet-coding", "claude-p", backend="claude-p",
                     prefixes=("Current session", "Current week", "Sonnet week"))],
            ttl_seconds=300, stale_seconds=1800, refresh=False,
        )
        # worst-of: min(90, 80, 95) = 80 (Current week is binding, NOT Opus's 5)
        assert view.entries["claude-sonnet-coding"].remaining_percent == 80.0

    def test_haiku_and_general_routes_bound_only_by_general_windows(self, cache):
        self._store(cache)
        view = cache.build_route_usage_view(
            [_Route("claude-haiku-coding", "claude-p", backend="claude-p",
                     prefixes=("Current session", "Current week"))],
            ttl_seconds=300, stale_seconds=1800, refresh=False,
        )
        # worst-of: min(90, 80) = 80 — depleted Opus (5%) must NOT leak in
        assert view.entries["claude-haiku-coding"].remaining_percent == 80.0

    def test_opus_depletion_does_not_starve_sonnet_or_haiku_routes(self, cache):
        cache.store_snapshot(
            _claude_p_snapshot(five_hour=50.0, seven_day=50.0, opus=100.0, sonnet=10.0)
        )
        view = cache.build_route_usage_view(
            [
                _Route("claude-opus-5-coding", "claude-p", backend="claude-p",
                       prefixes=("Current session", "Current week", "Opus week")),
                _Route("claude-sonnet-coding", "claude-p", backend="claude-p",
                       prefixes=("Current session", "Current week", "Sonnet week")),
                _Route("claude-haiku-coding", "claude-p", backend="claude-p",
                       prefixes=("Current session", "Current week")),
            ],
            ttl_seconds=300, stale_seconds=1800, refresh=False,
        )
        assert view.entries["claude-opus-5-coding"].remaining_percent == 0.0
        assert view.entries["claude-sonnet-coding"].remaining_percent == 50.0
        assert view.entries["claude-haiku-coding"].remaining_percent == 50.0


# ---------------------------------------------------------------------------
# 6. Selector: known Claude remaining beats lower remaining OpenAI/Agy
# ---------------------------------------------------------------------------


class TestSelectorPrefersKnownClaudeRemaining:
    def _catalog(self):
        return RouteCatalog(
            enabled=True,
            routes=(
                DelegationRoute(
                    id="claude-sonnet-coding",
                    provider="claude-p",
                    model="claude-sonnet-5",
                    model_class=ModelClass.ADVANCED,
                    task_difficulties=(TaskDifficulty.STANDARD,),
                    capabilities=frozenset({"coding"}),
                    priority=10,
                    backend="claude-p",
                    tool_profile="coding",
                ),
                DelegationRoute(
                    id="openai-5-5",
                    provider="openai-codex",
                    model="gpt-5.5",
                    model_class=ModelClass.ADVANCED,
                    task_difficulties=(TaskDifficulty.STANDARD,),
                    capabilities=frozenset({"coding"}),
                    priority=10,
                ),
                DelegationRoute(
                    id="agy-advanced",
                    provider="google-antigravity",
                    model="gemini-3-pro",
                    model_class=ModelClass.ADVANCED,
                    task_difficulties=(TaskDifficulty.STANDARD,),
                    capabilities=frozenset({"coding"}),
                    priority=10,
                ),
            ),
        )

    def test_claude_p_wins_when_its_remaining_is_higher(self):
        usage = UsageView(
            {
                "claude-sonnet-coding": ProviderUsage(
                    provider="claude-p", remaining_percent=80.0, freshness="fresh"
                ),
                "openai-5-5": ProviderUsage(
                    provider="openai-codex", remaining_percent=30.0, freshness="fresh"
                ),
                "agy-advanced": ProviderUsage(
                    provider="google-antigravity", remaining_percent=20.0, freshness="fresh"
                ),
            }
        )
        decision = select_route(
            self._catalog(),
            RouteRequest(difficulty=TaskDifficulty.STANDARD, required_capabilities=frozenset({"coding"})),
            usage=usage,
            available_providers=frozenset({"claude-p", "openai-codex", "google-antigravity"}),
        )
        assert decision.selected
        assert decision.route_id == "claude-sonnet-coding"
        assert decision.provider == "claude-p"
