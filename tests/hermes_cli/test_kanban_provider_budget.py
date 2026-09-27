"""Pure table-driven tests for the Kanban per-provider concurrency budget key
and config parsing (#123654): hermes_cli/kanban_provider_budget.py.

T1, T2–T9, T4b, T4c, T6, T6b per the approved plan
(~/.hermes/plans/2026-09-26-kanban-provider-concurrency.md) and the O1/O2/O3
addendum.
"""

from __future__ import annotations

import logging

import pytest

from hermes_cli.kanban_provider_budget import (
    ProviderBudgets,
    normalize_endpoint_key,
    parse_provider_concurrency,
    route_key,
)
from hermes_cli.model_route import resolve_requested_route


def _route(model=None, provider=None, model_config=None, env_provider=None,
           user_providers=None, custom_providers=None):
    return resolve_requested_route(
        model=model, provider=provider,
        model_config=model_config or {}, user_providers=user_providers,
        custom_providers=custom_providers, env_provider=env_provider,
    )


def _key(model=None, provider=None, model_config=None, env_provider=None,
         user_providers=None, custom_providers=None, route=None):
    if route is None:
        route = _route(
            model=model, provider=provider, model_config=model_config,
            env_provider=env_provider, user_providers=user_providers,
            custom_providers=custom_providers,
        )
    return route_key(
        route, model_config or {},
        user_providers=user_providers, custom_providers=custom_providers,
    )


# ---------------------------------------------------------------------------
# T1 — parse
# ---------------------------------------------------------------------------


class TestParse:
    def test_none_empty_nonmapping_disabled(self):
        assert parse_provider_concurrency(None) is None
        assert parse_provider_concurrency({}) is None
        assert parse_provider_concurrency("nope") is None
        assert parse_provider_concurrency(["anthropic"]) is None

    def test_all_invalid_disabled(self):
        assert parse_provider_concurrency({"x": 0, "y": -1, "z": "x", "w": True}) is None

    def test_valid_entries_kept(self):
        budgets = parse_provider_concurrency({"anthropic": 12})
        assert budgets is not None
        assert budgets.cap_for("anthropic") == 12
        assert budgets.cap_for("openrouter") is None  # unlisted, no default

    def test_bad_values_dropped_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            budgets = parse_provider_concurrency(
                {"anthropic": 12, "badbool": True, "zero": 0, "neg": -1, "str": "x", "float": 1.5})
        assert budgets is not None
        assert budgets.cap_for("anthropic") == 12
        warnings = [r for r in caplog.records if "provider_concurrency" in r.getMessage()]
        # one per distinct bad (key, value)
        assert len(warnings) == 5

    def test_null_value_means_no_budget(self):
        budgets = parse_provider_concurrency({"anthropic": None, "default": 3})
        assert budgets.cap_for("anthropic") is None
        assert budgets.cap_for("openrouter") == 3
        assert budgets.cap_for("anthropic") is None

    def test_default_honored(self):
        budgets = parse_provider_concurrency({"default": 4})
        assert budgets.cap_for("anything-unlisted") == 4

    def test_reserved_keys_before_canonicalization(self, caplog):
        """auto/moa/unknown/custom/default never reach resolve_provider (B5/M12)."""
        with caplog.at_level(logging.WARNING):
            budgets = parse_provider_concurrency(
                {"auto": 2, "moa": 1, "unknown": 1, "custom": 1, "default": 3})
        assert budgets is not None
        assert budgets.cap_for("auto") == 2
        assert budgets.cap_for("moa") == 1
        assert budgets.cap_for("unknown") == 1
        assert budgets.cap_for("custom") == 1
        assert budgets.default == 3
        # And no warnings fired despite no canonical provider lookup succeeding.
        assert not [r for r in caplog.records if "provider_concurrency" in r.getMessage()]

    def test_reserved_keys_never_reach_resolve_provider(self, monkeypatch):
        """B5/M12: `auto`/`moa`/`unknown`/`custom` are buckets — resolving
        them as providers would hit credential detection (``auto``)."""
        from hermes_cli import kanban_provider_budget as kpb

        seen: list[str] = []

        def _spy(name):
            seen.append(str(name))
            raise RuntimeError("no resolution in tests")

        monkeypatch.setattr("hermes_cli.auth.resolve_provider", _spy)
        kpb.parse_provider_concurrency(
            {"auto": 2, "moa": 1, "unknown": 1, "custom": 1, "default": 3})
        assert seen == []  # reserved keys are buckets, never resolved

    def test_custom_url_keys_normalized(self):
        budgets = parse_provider_concurrency(
            {"custom:https://LLM.Example.INTERNAL:443/v1/": 20})
        assert budgets.cap_for("custom:https://llm.example.internal/v1") == 20

    def test_custom_name_key_rejected(self, caplog):
        """custom:<name> keys are rejected in v1 (O2)."""
        with caplog.at_level(logging.WARNING):
            budgets = parse_provider_concurrency({"custom:my-endpoint": 5, "anthropic": 2})
        assert budgets is not None
        assert "custom:my-endpoint" not in budgets.caps
        assert budgets.cap_for("anthropic") == 2
        assert any("custom keys are keyed by URL" in r.getMessage() for r in caplog.records)

    def test_duplicate_after_normalization_smaller_wins(self, caplog):
        with caplog.at_level(logging.WARNING):
            budgets = parse_provider_concurrency(
                {"custom:https://llm.example.internal/v1": 20,
                 "custom:https://LLM.example.internal/v1/": 5})
        assert budgets.cap_for("custom:https://llm.example.internal/v1") == 5
        assert any("normalize" in r.getMessage() for r in caplog.records)

    def test_known_provider_canonicalized(self):
        budgets = parse_provider_concurrency({"ANTHROPIC  ": 7})
        assert budgets.cap_for("anthropic") == 7

    def test_unknown_provider_kept_verbatim_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            budgets = parse_provider_concurrency({"totally-unknown-provider": 3})
        assert budgets is not None
        # kept verbatim — a plugin provider not loaded here can still match
        assert budgets.cap_for("totally-unknown-provider") == 3


# ---------------------------------------------------------------------------
# T6 — normalize_endpoint_key
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_case_and_trailing_slash(self):
        assert normalize_endpoint_key("https://LLM.Example.INTERNAL/v1/") == "https://llm.example.internal/v1"

    def test_default_port_dropped(self):
        assert normalize_endpoint_key("https://example.com:443/v1") == "https://example.com/v1"
        assert normalize_endpoint_key("http://example.com:80/v1") == "http://example.com/v1"

    def test_nondefault_port_kept(self):
        assert normalize_endpoint_key("http://localhost:11434/v1") == "http://localhost:11434/v1"

    def test_userinfo_query_fragment_removed(self):
        assert normalize_endpoint_key(
            "https://user:pass@LLM.Example.internal/v1/?x=1#frag"
        ) == "https://llm.example.internal/v1"

    def test_invalid(self):
        assert normalize_endpoint_key("") == "<invalid url>"
        assert normalize_endpoint_key("::::") == "<invalid url>"


# ---------------------------------------------------------------------------
# T2/T3/T4/T4b/T4c/T5/T7 — route -> key
# ---------------------------------------------------------------------------


class TestRouteKey:
    def test_task_override_provider_wins_over_profile(self):
        """T2: model_override + provider_override -> that provider's canonical id."""
        key = _key(
            model="claude-opus-4.6", provider="anthropic",
            model_config={"default": "m", "provider": "openrouter"},
        )
        assert key == "anthropic"

    def test_named_custom_entry_keys_by_url(self):
        """T3: a named providers: entry -> custom:<normalized url>."""
        key = _key(
            model="m", provider="gmk-lan",
            model_config={"default": "m"},
            user_providers={"gmk-lan": {"name": "GMK", "base_url": "http://gmk.lan:9931/v1"}},
        )
        assert key == "custom:http://gmk.lan:9931/v1"

    def test_two_named_entries_one_url_same_key(self):
        a = _key(model="m", provider="one", model_config={"default": "m"},
                 user_providers={"one": {"base_url": "https://llm.example.internal/v1"}})
        b = _key(model="m", provider="two", model_config={"default": "m"},
                 user_providers={"two": {"base_url": "https://LLM.example.internal/v1/"}})
        assert a == b == "custom:https://llm.example.internal/v1"

    def test_bare_custom_with_model_base_url(self):
        key = _key(model="m", provider="custom",
                   model_config={"default": "m", "base_url": "https://llm.example.internal/v1"})
        assert key == "custom:https://llm.example.internal/v1"

    def test_bare_custom_without_url(self):
        key = _key(model="m", provider="custom", model_config={"default": "m"})
        assert key == "custom"

    def test_nested_default_provider(self):
        """T4: dict-valued model.default with nested provider."""
        key = _key(model_config={"default": {"model": "m", "provider": "anthropic"}})
        assert key == "anthropic"

    def test_model_provider_key(self):
        key = _key(model_config={"default": "m", "provider": "gemini"})
        assert key == "gemini"

    def test_env_provider_scoped_value(self):
        key = _key(model_config={"default": "m"}, env_provider="zai")
        assert key == "zai"

    def test_moa_preset(self):
        key = _key(model="moa:fast", model_config={"default": "m"})
        assert key == "moa"

    def test_nothing_pinned_auto(self):
        key = _key(model_config={"default": "m"})
        assert key == "auto"

    def test_local_base_url_auto_is_custom(self):
        """T4b: auto + local base_url -> custom:<url> (mirrors _local_endpoint_bypass)."""
        key = _key(model_config={"default": "m", "base_url": "http://localhost:11434/v1"})
        assert key == "custom:http://localhost:11434/v1"

    def test_cloud_base_url_auto_stays_auto(self):
        key = _key(model_config={"default": "m", "base_url": "https://api.openai.com/v1"})
        assert key == "auto"

    def test_bare_model_override_uses_profile_provider(self):
        """T4c: model override without provider override -> the profile's provider."""
        key = _key(model="some-model", model_config={"default": "m", "provider": "anthropic"})
        assert key == "anthropic"

    def test_alias_base_url(self):
        """T5: a direct alias with base_url -> the provider/URL the CLI would request."""
        from hermes_cli import model_switch

        key = _key(model="myalias", model_config={"default": "fallback", "provider": "openrouter"})
        # no alias seeded -> falls through to the config provider
        assert key == "openrouter"

    def test_unknown_provider_name(self):
        """T7: unknown provider -> unknown."""
        key = _key(model="m", provider="no-such-provider", model_config={"default": "m"})
        assert key == "unknown"

    def test_resolver_exception_is_unknown(self, monkeypatch):
        """T7: a resolver exception -> 'unknown', never a raise."""
        from hermes_cli import kanban_provider_budget as kpb

        def _boom(*a, **kw):
            raise RuntimeError("boom")

        monkeypatch.setattr(kpb, "route_key", _boom)
        from hermes_cli.kanban_provider_budget import RouteKeyResolver

        resolver = RouteKeyResolver(
            profile_exists=lambda a: True,
            profile_inputs=lambda a: {"model_config": {}, "user_providers": None,
                                      "custom_providers": None, "env_provider": ""},
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(kpb, "resolve_requested_route", _boom)
            assert resolver.resolve("alpha", "m", None) == "unknown"


# ---------------------------------------------------------------------------
# T8 — no network / no credential access
# ---------------------------------------------------------------------------


class TestPurity:
    impure_calls: list[str] = []

    @pytest.fixture(autouse=True)
    def _forbid_impure(self, monkeypatch):
        """Key resolution + parsing must never touch runtime resolution,
        credential pools, HTTP auto-detect, or auth status. A call spy (not a
        booby-trap) so a swallowing try/except cannot hide the escape."""
        calls = self.impure_calls

        def _spy(name):
            def _record(*a, **kw):
                calls.append(name)
                raise RuntimeError(f"no impure deps in tests: {name}")
            return _record

        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider", _spy("resolve_runtime_provider"))
        monkeypatch.setattr(
            "hermes_cli.runtime_provider._auto_detect_local_model", _spy("_auto_detect_local_model"))
        monkeypatch.setattr(
            "hermes_cli.runtime_provider._get_model_config", _spy("_get_model_config"))
        monkeypatch.setattr(
            "hermes_cli.auth.get_auth_status", _spy("get_auth_status"))
        # CredentialPool: import path varies; patch the class attribute if importable.
        try:
            from hermes_cli import auth as _auth

            if hasattr(_auth, "CredentialPool"):
                monkeypatch.setattr(_auth.CredentialPool, "select", _spy("CredentialPool.select"))
        except Exception:
            pass
        calls.clear()
        yield
        assert not calls, f"impure calls escaped: {calls}"

    def test_key_resolution_paths_stay_pure(self):
        assert _key(model="m", provider="anthropic", model_config={"default": "m"}) == "anthropic"
        assert _key(model_config={"default": "m"}) == "auto"
        assert _key(model="moa:fast", model_config={"default": "m"}) == "moa"
        assert _key(model_config={"default": "m", "base_url": "http://localhost:11434/v1"}) == \
            "custom:http://localhost:11434/v1"
        assert _key(model="m", provider="no-such", model_config={"default": "m"}) == "unknown"

    def test_parsing_reserved_keys_stay_pure(self, caplog):
        with caplog.at_level(logging.WARNING):
            budgets = parse_provider_concurrency(
                {"auto": 2, "moa": 1, "unknown": 1, "custom": 1, "default": 3})
        assert budgets is not None
        assert budgets.default == 3
        assert not [r for r in caplog.records if "provider_concurrency" in r.getMessage()]


# ---------------------------------------------------------------------------
# T9 — profile scope (adversarial, MoA B6)
# ---------------------------------------------------------------------------


class TestProfileScope:
    def _resolver_with_inputs(self, inputs_by_profile):
        from hermes_cli.kanban_provider_budget import RouteKeyResolver

        return RouteKeyResolver(
            profile_exists=lambda a: a in inputs_by_profile or a == "alpha",
            profile_inputs=lambda a: inputs_by_profile.get(a),
        )

    def test_dispatcher_env_never_leaks_into_profile_route(self):
        """Dispatcher process env pins X; profile's own config pins Y -> Y."""
        resolver = self._resolver_with_inputs(
            {"alpha": {"model_config": {"default": "m", "provider": "openrouter"},
                       "user_providers": None, "custom_providers": None,
                       "env_provider": ""}})
        assert resolver.resolve("alpha", None, None) == "openrouter"

    def test_unpinned_profile_resolves_auto_even_when_dispatcher_pinned(self):
        resolver = self._resolver_with_inputs(
            {"alpha": {"model_config": {"default": "m"},
                       "user_providers": None, "custom_providers": None,
                       "env_provider": ""}})
        assert resolver.resolve("alpha", None, None) == "auto"

    def test_scoped_env_provider_wins_over_dispatcher_env(self):
        """Routed profile .env HERMES_INFERENCE_PROVIDER=Y beats dispatcher X."""
        resolver = self._resolver_with_inputs(
            {"alpha": {"model_config": {"default": "m"},
                       "user_providers": None, "custom_providers": None,
                       "env_provider": "zai"}})
        assert resolver.resolve("alpha", None, None) == "zai"


# ---------------------------------------------------------------------------
# T6b — secret hygiene (MoA I13)
# ---------------------------------------------------------------------------


class TestSecretHygiene:
    def test_malformed_custom_key_warning_hides_url(self, caplog):
        with caplog.at_level(logging.WARNING):
            parse_provider_concurrency({"custom:not-a-url": 3})
        for r in caplog.records:
            assert "not-a-url" not in r.getMessage()

    def test_userinfo_never_in_warnings(self, caplog):
        with caplog.at_level(logging.WARNING):
            budgets = parse_provider_concurrency(
                {"custom:https://user:secret@llm.example.internal:4439/v1/": 20})
        assert budgets is not None
        assert budgets.cap_for("custom:https://llm.example.internal:4439/v1") == 20
        for r in caplog.records:
            assert "user:secret" not in r.getMessage()
