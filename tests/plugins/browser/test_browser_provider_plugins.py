"""Plugin-side tests for the browser provider migration (PR #25214).

Covers:

- All three bundled plugins (browserbase, browser-use, firecrawl)
  instantiate and self-report the expected ABC defaults.
- Each plugin's ``is_available()`` correctly reflects env-var presence.
- The browser_registry resolves an active provider in the documented
  scenarios:
    * explicit config wins ignoring availability (so dispatcher surfaces
      a typed credentials error)
    * legacy preference walk: browser-use → browserbase (filtered by
      availability)
    * firecrawl is NOT in the legacy walk — explicit-only
    * unknown name falls through to auto-detect
    * ``local`` short-circuits to None

These tests use *real* imports from the plugin modules — no mocking of
provider classes themselves — so the test catches drift in the ABC
interface, the registry, and the plugin glue layer simultaneously.
Mirrors ``tests/plugins/web/test_web_search_provider_plugins.py`` from
PR #25182.
"""
from __future__ import annotations

import logging

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clear_browser_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every browser-provider env var so is_available() returns False."""
    for k in (
        "BROWSERBASE_API_KEY",
        "BROWSERBASE_PROJECT_ID",
        "BROWSERBASE_BASE_URL",
        "BROWSERBASE_CONTEXT_ID",
        "BROWSERBASE_PROXIES",
        "BROWSERBASE_ADVANCED_STEALTH",
        "BROWSERBASE_KEEP_ALIVE",
        "BROWSERBASE_SESSION_TIMEOUT",
        "BROWSER_USE_API_KEY",
        "BROWSER_USE_GATEWAY_URL",
        "FIRECRAWL_API_KEY",
        "FIRECRAWL_API_URL",
        "FIRECRAWL_BROWSER_TTL",
        "TOOL_GATEWAY_DOMAIN",
        "TOOL_GATEWAY_USER_TOKEN",
    ):
        monkeypatch.delenv(k, raising=False)


def _ensure_plugins_loaded() -> None:
    """Idempotently load plugins so the registry is populated."""
    from hermes_cli.plugins import _ensure_plugins_discovered

    _ensure_plugins_discovered()


# ---------------------------------------------------------------------------
# Per-test isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts with a clean browser-provider env."""
    _clear_browser_env(monkeypatch)


# ---------------------------------------------------------------------------
# Bundled plugins register
# ---------------------------------------------------------------------------


class TestBundledPluginsRegister:
    """All three bundled browser plugins discover and register correctly."""

    def test_all_three_plugins_present_in_registry(self) -> None:
        _ensure_plugins_loaded()
        from agent.browser_registry import list_providers

        names = sorted(p.name for p in list_providers())
        assert names == ["browser-use", "browserbase", "firecrawl"]

    @pytest.mark.parametrize(
        "plugin_name,expected_display",
        [
            ("browserbase", "Browserbase"),
            ("browser-use", "Browser Use"),
            ("firecrawl", "Firecrawl"),
        ],
    )
    def test_each_plugin_has_name_and_display_name(
        self, plugin_name: str, expected_display: str
    ) -> None:
        _ensure_plugins_loaded()
        from agent.browser_registry import get_provider

        provider = get_provider(plugin_name)
        assert provider is not None, f"plugin {plugin_name!r} not registered"
        assert provider.name == plugin_name
        assert provider.display_name == expected_display


# ---------------------------------------------------------------------------
# is_available() behavior
# ---------------------------------------------------------------------------


class TestIsAvailable:
    """Each plugin's ``is_available()`` reflects env-var presence accurately."""

    def test_browserbase_requires_both_api_key_and_project_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ensure_plugins_loaded()
        from agent.browser_registry import get_provider

        p = get_provider("browserbase")
        assert p is not None
        assert p.is_available() is False

        # API key alone is insufficient.
        monkeypatch.setenv("BROWSERBASE_API_KEY", "key")
        assert p.is_available() is False

        # Both env vars set → available.
        monkeypatch.setenv("BROWSERBASE_PROJECT_ID", "proj")
        assert p.is_available() is True


    def test_browser_use_satisfied_by_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ensure_plugins_loaded()
        from agent.browser_registry import get_provider

        p = get_provider("browser-use")
        assert p is not None
        assert p.is_available() is False
        monkeypatch.setenv("BROWSER_USE_API_KEY", "key")
        assert p.is_available() is True

    def test_firecrawl_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _ensure_plugins_loaded()
        from agent.browser_registry import get_provider

        p = get_provider("firecrawl")
        assert p is not None
        assert p.is_available() is False
        monkeypatch.setenv("FIRECRAWL_API_KEY", "key")
        assert p.is_available() is True


# ---------------------------------------------------------------------------
# Registry resolution semantics
# ---------------------------------------------------------------------------


class TestRegistryResolution:
    """``_resolve()`` implements the documented three-rule precedence."""

    def test_resolve_none_with_no_creds_returns_none(self) -> None:
        """No config, no env → local mode (None)."""
        _ensure_plugins_loaded()
        from agent.browser_registry import _resolve

        assert _resolve(None) is None

    def test_explicit_local_returns_none(self) -> None:
        """``cloud_provider: local`` is a positive choice; short-circuits to None."""
        _ensure_plugins_loaded()
        from agent.browser_registry import _resolve

        assert _resolve("local") is None


    def test_legacy_walk_prefers_browser_use_over_browserbase(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rule 3: walk order is browser-use → browserbase."""
        _ensure_plugins_loaded()
        from agent.browser_registry import _resolve

        # Both available — browser-use should win.
        monkeypatch.setenv("BROWSER_USE_API_KEY", "k1")
        monkeypatch.setenv("BROWSERBASE_API_KEY", "k2")
        monkeypatch.setenv("BROWSERBASE_PROJECT_ID", "p")

        provider = _resolve(None)
        assert provider is not None
        assert provider.name == "browser-use"


# ---------------------------------------------------------------------------
# Legacy ABC backward-compat aliases (is_configured / provider_name)
# ---------------------------------------------------------------------------


class TestLegacyAbcAliases:
    """is_configured() and provider_name() delegate to the new API."""

    @pytest.mark.parametrize(
        "plugin_name",
        ["browserbase", "browser-use", "firecrawl"],
    )
    def test_is_configured_delegates_to_is_available(self, plugin_name: str) -> None:
        _ensure_plugins_loaded()
        from agent.browser_registry import get_provider

        p = get_provider(plugin_name)
        assert p is not None
        assert p.is_configured() is p.is_available()

    @pytest.mark.parametrize(
        "plugin_name,expected_label",
        [
            ("browserbase", "Browserbase"),
            ("browser-use", "Browser Use"),
            ("firecrawl", "Firecrawl"),
        ],
    )
    def test_provider_name_returns_display_name(
        self, plugin_name: str, expected_label: str
    ) -> None:
        _ensure_plugins_loaded()
        from agent.browser_registry import get_provider

        p = get_provider(plugin_name)
        assert p is not None
        assert p.provider_name() == expected_label


# ---------------------------------------------------------------------------
# Picker integration
# ---------------------------------------------------------------------------


class TestPickerIntegration:
    """`_plugin_browser_providers()` exposes all three plugins as picker rows."""

    def test_picker_rows_match_registered_plugins(self) -> None:
        _ensure_plugins_loaded()
        from hermes_cli.tools_config import _plugin_browser_providers

        rows = _plugin_browser_providers()
        names = sorted(r.get("browser_provider") for r in rows)
        assert names == ["browser-use", "browserbase", "firecrawl"]


# ---------------------------------------------------------------------------
# Browserbase session payload — persistent Contexts
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal stand-in for the bits of ``requests.Response`` the provider uses."""

    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 400
        self.text = ""

    def json(self) -> dict:
        return self._payload


class _PostLog(list):
    """Captured session-create calls, plus a queue of canned responses.

    Tests read ``log[-1]["json"]`` for the payload actually sent and push onto
    ``log.responses`` to drive the provider's 402 fallback path.
    """

    def __init__(self) -> None:
        super().__init__()
        self.responses: list = []


@pytest.fixture
def bb_posts(monkeypatch: pytest.MonkeyPatch) -> _PostLog:
    """Record every session-create payload instead of calling Browserbase."""
    from plugins.browser.browserbase import provider as bb_provider

    calls = _PostLog()

    def _fake_post(url, headers=None, json=None, timeout=None):
        # The provider mutates and re-posts the same dict on 402 fallback, so
        # snapshot it — otherwise every recorded call aliases the final state.
        calls.append({"url": url, "headers": dict(headers or {}), "json": dict(json or {})})
        if calls.responses:
            return calls.responses.pop(0)
        return _FakeResponse({"id": "sess-1", "connectUrl": "wss://connect.example/sess-1"})

    monkeypatch.setattr(bb_provider.requests, "post", _fake_post)
    return calls


def _create_session(task_id: str = "task-1") -> dict:
    _ensure_plugins_loaded()
    from agent.browser_registry import get_provider

    provider = get_provider("browserbase")
    assert provider is not None
    return provider.create_session(task_id)


class TestBrowserbaseSessionPayload:
    """``BROWSERBASE_CONTEXT_ID`` maps onto the documented Contexts payload.

    Browserbase persists cookies/localStorage across *separate* sessions when
    a session is attached to a Context with ``persist: true``
    (https://docs.browserbase.com/features/contexts). These tests pin the
    request body Hermes sends, because the payload — not our internal state —
    is the actual contract with the API.
    """

    @pytest.fixture(autouse=True)
    def _credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BROWSERBASE_API_KEY", "bb-key")
        monkeypatch.setenv("BROWSERBASE_PROJECT_ID", "bb-proj")

    def test_without_context_id_payload_carries_no_browser_settings(
        self, bb_posts: _PostLog
    ) -> None:
        """Unset → byte-identical payload to the pre-Context behaviour."""
        _create_session()

        body = bb_posts[-1]["json"]
        assert "browserSettings" not in body
        assert body == {"projectId": "bb-proj", "keepAlive": True, "proxies": True}
        # Pin the default endpoint too: a typo here would silently point every
        # user without a BROWSERBASE_BASE_URL override at the wrong host.
        assert bb_posts[-1]["url"] == "https://api.browserbase.com/v1/sessions"

    def test_context_id_attaches_persistent_context(
        self, monkeypatch: pytest.MonkeyPatch, bb_posts: _PostLog
    ) -> None:
        monkeypatch.setenv("BROWSERBASE_CONTEXT_ID", "context-123")

        _create_session()

        assert bb_posts[-1]["json"]["browserSettings"] == {
            "context": {"id": "context-123", "persist": True}
        }

    def test_context_and_advanced_stealth_coexist(
        self, monkeypatch: pytest.MonkeyPatch, bb_posts: _PostLog
    ) -> None:
        """Both knobs live in one ``browserSettings``; neither clobbers the other."""
        monkeypatch.setenv("BROWSERBASE_CONTEXT_ID", "context-123")
        monkeypatch.setenv("BROWSERBASE_ADVANCED_STEALTH", "true")

        _create_session()

        assert bb_posts[-1]["json"]["browserSettings"] == {
            "advancedStealth": True,
            "context": {"id": "context-123", "persist": True},
        }

    @pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
    def test_blank_context_id_is_ignored(
        self, monkeypatch: pytest.MonkeyPatch, bb_posts: _PostLog, blank: str
    ) -> None:
        """An empty/whitespace value must not send ``context: {"id": ""}``."""
        monkeypatch.setenv("BROWSERBASE_CONTEXT_ID", blank)

        result = _create_session()

        assert "browserSettings" not in bb_posts[-1]["json"]
        assert result["features"]["persistent_context"] is False

    def test_context_id_is_stripped(
        self, monkeypatch: pytest.MonkeyPatch, bb_posts: _PostLog
    ) -> None:
        monkeypatch.setenv("BROWSERBASE_CONTEXT_ID", "  context-123\n")

        _create_session()

        context = bb_posts[-1]["json"]["browserSettings"]["context"]
        assert context["id"] == "context-123"

    def test_context_survives_the_402_paid_feature_fallback(
        self, monkeypatch: pytest.MonkeyPatch, bb_posts: _PostLog
    ) -> None:
        """Dropping keepAlive/proxies on 402 must not drop the Context too.

        The provider retries by mutating and re-posting the same dict, so a
        careless ``pop`` or rebuild here would silently un-persist the login.
        """
        monkeypatch.setenv("BROWSERBASE_CONTEXT_ID", "context-123")
        bb_posts.responses.extend(
            [
                _FakeResponse({}, status_code=402),
                _FakeResponse({}, status_code=402),
                _FakeResponse({"id": "s", "connectUrl": "wss://connect.example/s"}),
            ]
        )

        result = _create_session()

        assert len(bb_posts) == 3
        for call in bb_posts:
            assert call["json"]["browserSettings"] == {
                "context": {"id": "context-123", "persist": True}
            }
        # The fallbacks themselves still happened.
        assert "keepAlive" not in bb_posts[-1]["json"]
        assert "proxies" not in bb_posts[-1]["json"]
        assert result["features"]["keep_alive"] is False
        assert result["features"]["proxies"] is False
        assert result["features"]["persistent_context"] is True

    def test_features_report_persistent_context_without_leaking_it(
        self, monkeypatch: pytest.MonkeyPatch, bb_posts: _PostLog
    ) -> None:
        """``features`` is surfaced to callers/logs — flags only, no values."""
        monkeypatch.setenv("BROWSERBASE_CONTEXT_ID", "context-123")

        features = _create_session()["features"]

        assert features["persistent_context"] is True
        assert all(isinstance(v, bool) for v in features.values())
        assert "context-123" not in repr(features)
        assert "bb-key" not in repr(features)

    def test_context_id_and_api_key_never_reach_the_log(
        self,
        monkeypatch: pytest.MonkeyPatch,
        bb_posts: _PostLog,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setenv("BROWSERBASE_CONTEXT_ID", "context-123")

        with caplog.at_level(logging.DEBUG, logger="plugins.browser.browserbase.provider"):
            _create_session()

        logged = "\n".join(record.getMessage() for record in caplog.records)
        assert "persistent_context" in logged
        assert "context-123" not in logged
        assert "bb-key" not in logged

    def test_context_id_resolves_from_the_profile_secret_scope(
        self, monkeypatch: pytest.MonkeyPatch, bb_posts: _PostLog
    ) -> None:
        """Under a multiplexed gateway the value lives in the secret scope.

        A profile's ``.env`` is installed as a contextvar scope, *not* into
        ``os.environ`` — reading the Context ID with ``os.environ.get`` would
        silently ignore the profile's setting (or, worse, pick up another
        profile's). Pin the scoped read: ``os.environ`` here holds a decoy.
        """
        from agent import secret_scope

        monkeypatch.setenv("BROWSERBASE_CONTEXT_ID", "env-decoy-context")

        token = secret_scope.set_secret_scope(
            {
                "BROWSERBASE_API_KEY": "scoped-key",
                "BROWSERBASE_PROJECT_ID": "scoped-proj",
                "BROWSERBASE_CONTEXT_ID": "scoped-context",
            }
        )
        secret_scope.set_multiplex_active(True)
        try:
            _create_session()
        finally:
            secret_scope.set_multiplex_active(False)
            secret_scope.reset_secret_scope(token)

        body = bb_posts[-1]["json"]
        assert body["projectId"] == "scoped-proj"
        assert body["browserSettings"]["context"]["id"] == "scoped-context"

    def test_every_knob_resolves_from_the_profile_secret_scope(
        self, monkeypatch: pytest.MonkeyPatch, bb_posts: _PostLog
    ) -> None:
        """All Browserbase knobs are per-profile, not process-global.

        Configuring a profile's ``.env`` must actually configure *that*
        profile. Reading any one of these from ``os.environ`` would silently
        ignore the profile's value under a multiplexed gateway — the operator
        sets ``BROWSERBASE_KEEP_ALIVE=false`` and keep-alive stays on.
        """
        from agent import secret_scope

        for key, decoy in (
            ("BROWSERBASE_PROXIES", "true"),
            ("BROWSERBASE_ADVANCED_STEALTH", "false"),
            ("BROWSERBASE_KEEP_ALIVE", "true"),
            ("BROWSERBASE_SESSION_TIMEOUT", "60"),
        ):
            monkeypatch.setenv(key, decoy)

        token = secret_scope.set_secret_scope(
            {
                "BROWSERBASE_API_KEY": "scoped-key",
                "BROWSERBASE_PROJECT_ID": "scoped-proj",
                # Every value here is the opposite of the os.environ decoy.
                "BROWSERBASE_PROXIES": "false",
                "BROWSERBASE_ADVANCED_STEALTH": "true",
                "BROWSERBASE_KEEP_ALIVE": "false",
                "BROWSERBASE_SESSION_TIMEOUT": "1800",
            }
        )
        secret_scope.set_multiplex_active(True)
        try:
            features = _create_session()["features"]
        finally:
            secret_scope.set_multiplex_active(False)
            secret_scope.reset_secret_scope(token)

        body = bb_posts[-1]["json"]
        assert "proxies" not in body
        assert "keepAlive" not in body
        assert body["timeout"] == 1800
        assert body["browserSettings"] == {"advancedStealth": True}
        assert features["proxies"] is False
        assert features["keep_alive"] is False
        assert features["advanced_stealth"] is True
        assert features["custom_timeout"] is True

    def test_multiplexed_scope_miss_takes_the_default_not_os_environ(
        self, monkeypatch: pytest.MonkeyPatch, bb_posts: _PostLog
    ) -> None:
        """Under an ACTIVE multiplexer the scope is authoritative, not an overlay.

        This pins a deliberate behaviour change: a knob set only in the
        process environment (systemd ``Environment=``, a shell export) and
        absent from the profile's scope resolves to its DEFAULT here, where a
        plain ``os.environ`` read would have returned the process value.
        Falling through would hand one profile another profile's setting,
        which is the isolation boundary ``get_secret`` exists to enforce —
        the API key already behaves this way. Multiplexed operators must
        configure these per profile.
        """
        from agent import secret_scope

        # Process-level values that must NOT win under multiplexing — every
        # knob, so none of them can regress to a fallthrough independently.
        monkeypatch.setenv("BROWSERBASE_KEEP_ALIVE", "false")
        monkeypatch.setenv("BROWSERBASE_PROXIES", "false")
        monkeypatch.setenv("BROWSERBASE_ADVANCED_STEALTH", "true")
        monkeypatch.setenv("BROWSERBASE_SESSION_TIMEOUT", "60")
        monkeypatch.setenv("BROWSERBASE_BASE_URL", "https://decoy.example")

        token = secret_scope.set_secret_scope(
            {"BROWSERBASE_API_KEY": "scoped-key", "BROWSERBASE_PROJECT_ID": "scoped-proj"}
        )
        secret_scope.set_multiplex_active(True)
        try:
            _create_session()
        finally:
            secret_scope.set_multiplex_active(False)
            secret_scope.reset_secret_scope(token)

        body = bb_posts[-1]["json"]
        assert body["keepAlive"] is True  # default, not the os.environ "false"
        assert body["proxies"] is True  # default, not the os.environ "false"
        assert "browserSettings" not in body  # default, not the os.environ "true"
        assert "timeout" not in body  # default (none), not the os.environ "60"
        assert bb_posts[-1]["url"] == "https://api.browserbase.com/v1/sessions"

    def test_base_url_override_resolves_from_the_profile_secret_scope(
        self, monkeypatch: pytest.MonkeyPatch, bb_posts: _PostLog
    ) -> None:
        from agent import secret_scope

        monkeypatch.setenv("BROWSERBASE_BASE_URL", "https://decoy.example")

        token = secret_scope.set_secret_scope(
            {
                "BROWSERBASE_API_KEY": "scoped-key",
                "BROWSERBASE_PROJECT_ID": "scoped-proj",
                "BROWSERBASE_BASE_URL": "https://scoped.example/",
            }
        )
        secret_scope.set_multiplex_active(True)
        try:
            _create_session()
        finally:
            secret_scope.set_multiplex_active(False)
            secret_scope.reset_secret_scope(token)

        assert bb_posts[-1]["url"] == "https://scoped.example/v1/sessions"


