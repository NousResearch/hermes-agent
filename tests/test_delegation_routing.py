"""Tests for the multi-model native delegation route catalog (PR3).

Covers config parsing/validation, the difficulty/model-class taxonomies,
capability filtering, deterministic selection, and the secret-free usage
cache.  Every provider/usage/auth call is mocked: these tests never touch
the network, an OAuth flow, the auth store, or a real credential.
"""

import pytest

from agent import delegation_routing as dr


def _route(**overrides):
    base = {
        "id": "codex-standard",
        "backend": "native",
        "provider": "openai-codex",
        "model": "gpt-5.6-sol",
        "model_class": "advanced",
        "task_difficulties": ["standard", "complex"],
        "capabilities": ["coding", "reasoning", "tool_use"],
        "priority": 20,
        "reserve_remaining_percent": 15,
    }
    base.update(overrides)
    return base


class TestRouteCatalogParsing:
    def test_parses_contract_example(self):
        cfg = {
            "routing": {
                "enabled": True,
                "usage_ttl_seconds": 300,
                "usage_stale_seconds": 1800,
                "unknown_usage": "fixed_priority",
            },
            "routes": [
                _route(),
                {
                    "id": "gemini-routine",
                    "backend": "native",
                    "provider": "google-antigravity",
                    "model": "gemini-3-flash-agent",
                    "model_class": "balanced",
                    "task_difficulties": ["routine", "standard"],
                    "capabilities": ["reasoning", "tool_use", "long_context"],
                    "priority": 30,
                    "reserve_remaining_percent": 10,
                    "usage_window_prefixes": ["Gemini Models"],
                },
            ],
        }
        catalog = dr.load_route_catalog(cfg)

        assert catalog.enabled is True
        assert catalog.usage_ttl_seconds == 300
        assert catalog.usage_stale_seconds == 1800
        assert catalog.unknown_usage == "fixed_priority"
        assert [r.id for r in catalog.routes] == ["codex-standard", "gemini-routine"]

        gemini = catalog.routes[1]
        assert gemini.provider == "google-antigravity"
        assert gemini.model_class is dr.ModelClass.BALANCED
        assert gemini.task_difficulties == (
            dr.TaskDifficulty.ROUTINE,
            dr.TaskDifficulty.STANDARD,
        )
        assert gemini.capabilities == frozenset({"reasoning", "tool_use", "long_context"})
        assert gemini.usage_window_prefixes == ("Gemini Models",)
        assert gemini.reserve_remaining_percent == 10.0

    def test_absent_routing_block_defaults_to_disabled_empty_catalog(self):
        catalog = dr.load_route_catalog({"provider": "openai-codex", "model": "gpt-5"})
        assert catalog.enabled is False
        assert catalog.routes == ()

    def test_disabled_routing_ignores_malformed_future_routes(self):
        catalog = dr.load_route_catalog(
            {
                "provider": "openai-codex",
                "model": "gpt-5",
                "routing": {"enabled": False},
                "routes": [{"backend": "cli", "provider": "opencode"}],
            }
        )
        assert catalog.enabled is False
        assert catalog.routes == ()

    def test_empty_routes_preserve_legacy_even_with_unused_policy_values(self):
        catalog = dr.load_route_catalog(
            {
                "provider": "openai-codex",
                "model": "gpt-5",
                "routing": {
                    "enabled": True,
                    "unknown_usage": "future-policy",
                    "usage_ttl_seconds": float("inf"),
                },
                "routes": [],
            }
        )
        assert catalog.enabled is False
        assert catalog.routes == ()

    def test_accepts_ten_plus_route_entries(self):
        routes = [
            _route(id=f"route-{i:02d}", priority=i, model=f"model-{i}")
            for i in range(12)
        ]
        catalog = dr.load_route_catalog({"routing": {"enabled": True}, "routes": routes})
        assert len(catalog.routes) == 12
        assert len({r.id for r in catalog.routes}) == 12

    @pytest.mark.parametrize(
        "bad, msg",
        [
            ({"id": ""}, "id"),
            ({"provider": "opencode"}, "provider"),
            ({"backend": "cli"}, "backend"),
            ({"model": ""}, "model"),
            ({"model_class": "turbo"}, "model_class"),
            ({"task_difficulties": ["trivial"]}, "task_difficulties"),
            ({"reserve_remaining_percent": 150}, "reserve_remaining_percent"),
        ],
    )
    def test_invalid_route_is_rejected(self, bad, msg):
        cfg = {"routing": {"enabled": True}, "routes": [_route(**bad)]}
        with pytest.raises(dr.RouteConfigError) as exc:
            dr.load_route_catalog(cfg)
        assert msg in str(exc.value)

    def test_duplicate_route_ids_rejected(self):
        cfg = {"routing": {"enabled": True}, "routes": [_route(), _route()]}
        with pytest.raises(dr.RouteConfigError) as exc:
            dr.load_route_catalog(cfg)
        assert "duplicate" in str(exc.value).lower()

    @pytest.mark.parametrize("reserve", [10**1000, float("inf"), float("nan")])
    def test_non_finite_or_overflowing_reserve_is_rejected(self, reserve):
        cfg = {
            "routing": {"enabled": True},
            "routes": [_route(reserve_remaining_percent=reserve)],
        }
        with pytest.raises(dr.RouteConfigError):
            dr.load_route_catalog(cfg)

    def test_stale_window_cannot_be_shorter_than_fresh_ttl(self):
        cfg = {
            "routing": {
                "enabled": True,
                "usage_ttl_seconds": 600,
                "usage_stale_seconds": 300,
            },
            "routes": [_route()],
        }
        with pytest.raises(dr.RouteConfigError) as exc:
            dr.load_route_catalog(cfg)
        assert "usage_stale_seconds" in str(exc.value)

    def test_infinite_ttl_is_rejected_as_config_error(self):
        cfg = {
            "routing": {"enabled": True, "usage_ttl_seconds": float("inf")},
            "routes": [_route()],
        }
        with pytest.raises(dr.RouteConfigError):
            dr.load_route_catalog(cfg)

    def test_unknown_capability_is_allowed_but_builtins_documented(self):
        cfg = {"routing": {"enabled": True}, "routes": [_route(capabilities=["sql"])]}
        catalog = dr.load_route_catalog(cfg)
        assert catalog.routes[0].capabilities == frozenset({"sql"})
        assert {"coding", "reasoning", "tool_use", "long_context", "vision", "review"} <= set(
            dr.BUILTIN_CAPABILITIES
        )

    def test_model_class_ordering(self):
        assert (
            dr.ModelClass.FAST
            < dr.ModelClass.BALANCED
            < dr.ModelClass.ADVANCED
            < dr.ModelClass.FRONTIER
        )

    def test_task_difficulty_enum_members(self):
        assert [d.value for d in dr.TaskDifficulty] == [
            "routine",
            "standard",
            "complex",
            "frontier",
        ]


def _catalog(*routes, **routing):
    cfg = {"routing": {"enabled": True, **routing}, "routes": list(routes)}
    return dr.load_route_catalog(cfg)


_GEMINI = {
    "id": "gemini-routine",
    "backend": "native",
    "provider": "google-antigravity",
    "model": "gemini-3-flash-agent",
    "model_class": "balanced",
    "task_difficulties": ["routine", "standard"],
    "capabilities": ["reasoning", "tool_use", "long_context"],
    "priority": 30,
    "reserve_remaining_percent": 10,
    "usage_window_prefixes": ["Gemini Models"],
}


def _usage(**per_provider):
    """Build a UsageView from {provider: (remaining_percent, freshness, age)}."""
    entries = {}
    for provider, spec in per_provider.items():
        remaining, freshness, age = spec
        entries[provider.replace("_", "-")] = dr.ProviderUsage(
            provider=provider.replace("_", "-"),
            remaining_percent=remaining,
            freshness=freshness,
            age_seconds=age,
        )
    return dr.UsageView(entries)


_ALL_AVAILABLE = frozenset({"openai-codex", "google-antigravity"})


class TestSelection:
    def test_selects_matching_difficulty_and_returns_decision(self):
        catalog = _catalog(_route(), _GEMINI)
        decision = dr.select_route(
            catalog,
            dr.RouteRequest(difficulty=dr.TaskDifficulty.ROUTINE),
            usage=_usage(google_antigravity=(80.0, "fresh", 10.0)),
            available_providers=_ALL_AVAILABLE,
        )
        assert decision.selected is True
        assert decision.route_id == "gemini-routine"
        assert decision.provider == "google-antigravity"
        assert decision.model == "gemini-3-flash-agent"
        assert decision.model_class == "balanced"
        assert decision.difficulty == "routine"
        assert decision.usage_freshness == "fresh"
        assert decision.usage_age_seconds == 10.0
        assert decision.reason

    def test_difficulty_equivalence_across_providers(self):
        """Equivalent routes with equal usage fall back to priority."""
        catalog = _catalog(_route(priority=10), _GEMINI)
        decision = dr.select_route(
            catalog,
            dr.RouteRequest(difficulty=dr.TaskDifficulty.STANDARD),
            usage=_usage(
                openai_codex=(90.0, "fresh", 1.0),
                google_antigravity=(90.0, "fresh", 1.0),
            ),
            available_providers=_ALL_AVAILABLE,
        )
        assert decision.route_id == "codex-standard"
        assert [c.id for c in catalog.routes if dr.TaskDifficulty.STANDARD in c.task_difficulties] == [
            "codex-standard",
            "gemini-routine",
        ]

    def test_equivalent_route_with_more_remaining_usage_wins_before_priority(self):
        catalog = _catalog(_route(priority=1), _GEMINI)
        decision = dr.select_route(
            catalog,
            dr.RouteRequest(difficulty=dr.TaskDifficulty.STANDARD),
            usage=_usage(
                openai_codex=(35.0, "fresh", 1.0),
                google_antigravity=(80.0, "fresh", 1.0),
            ),
            available_providers=_ALL_AVAILABLE,
        )
        assert decision.route_id == "gemini-routine"

    def test_minimum_model_class_filters_weaker_routes(self):
        catalog = _catalog(_route(priority=50), _GEMINI)
        decision = dr.select_route(
            catalog,
            dr.RouteRequest(
                difficulty=dr.TaskDifficulty.STANDARD,
                minimum_model_class=dr.ModelClass.ADVANCED,
            ),
            usage=_usage(
                openai_codex=(90.0, "fresh", 1.0),
                google_antigravity=(90.0, "fresh", 1.0),
            ),
            available_providers=_ALL_AVAILABLE,
        )
        assert decision.route_id == "codex-standard"

    def test_capabilities_filter_is_subset_match(self):
        catalog = _catalog(_route(), _GEMINI)
        decision = dr.select_route(
            catalog,
            dr.RouteRequest(
                difficulty=dr.TaskDifficulty.STANDARD,
                required_capabilities=frozenset({"long_context"}),
            ),
            usage=_usage(google_antigravity=(90.0, "fresh", 1.0)),
            available_providers=_ALL_AVAILABLE,
        )
        assert decision.route_id == "gemini-routine"

    def test_no_candidate_returns_unselected_decision_with_reason(self):
        catalog = _catalog(_route())
        decision = dr.select_route(
            catalog,
            dr.RouteRequest(
                difficulty=dr.TaskDifficulty.STANDARD,
                required_capabilities=frozenset({"vision"}),
            ),
            usage=_usage(),
            available_providers=_ALL_AVAILABLE,
        )
        assert decision.selected is False
        assert decision.route_id is None
        assert "capabilit" in decision.reason.lower()

    def test_unavailable_provider_excluded(self):
        catalog = _catalog(_route(), _GEMINI)
        decision = dr.select_route(
            catalog,
            dr.RouteRequest(difficulty=dr.TaskDifficulty.STANDARD),
            usage=_usage(google_antigravity=(90.0, "fresh", 1.0)),
            available_providers=frozenset({"google-antigravity"}),
        )
        assert decision.route_id == "gemini-routine"

    def test_disabled_route_excluded(self):
        catalog = _catalog(_route(enabled=False), _GEMINI)
        decision = dr.select_route(
            catalog,
            dr.RouteRequest(difficulty=dr.TaskDifficulty.COMPLEX),
            usage=_usage(),
            available_providers=_ALL_AVAILABLE,
        )
        assert decision.selected is False

    def test_deterministic_tie_break_priority_then_id(self):
        a = _route(id="zzz-first", priority=5)
        b = _route(id="aaa-second", priority=5)
        c = _route(id="mmm-third", priority=5)
        catalog = _catalog(a, b, c)
        usage = _usage(openai_codex=(90.0, "fresh", 1.0))
        req = dr.RouteRequest(difficulty=dr.TaskDifficulty.COMPLEX)
        picks = {
            dr.select_route(catalog, req, usage=usage, available_providers=_ALL_AVAILABLE).route_id
            for _ in range(5)
        }
        assert picks == {"aaa-second"}

    def test_ranked_candidates_are_stably_ordered(self):
        catalog = _catalog(_route(priority=20), _GEMINI)
        decision = dr.select_route(
            catalog,
            dr.RouteRequest(difficulty=dr.TaskDifficulty.STANDARD),
            usage=_usage(
                openai_codex=(90.0, "fresh", 1.0),
                google_antigravity=(90.0, "fresh", 1.0),
            ),
            available_providers=_ALL_AVAILABLE,
        )
        assert decision.considered_route_ids == ("codex-standard", "gemini-routine")


class TestExplicitOverride:
    def test_explicit_route_wins_and_is_marked(self):
        catalog = _catalog(_route(priority=1), _GEMINI)
        decision = dr.select_route(
            catalog,
            dr.RouteRequest(difficulty=dr.TaskDifficulty.STANDARD, route_id="gemini-routine"),
            usage=_usage(google_antigravity=(90.0, "fresh", 1.0)),
            available_providers=_ALL_AVAILABLE,
        )
        assert decision.route_id == "gemini-routine"
        assert decision.explicit_override is True

    def test_explicit_route_still_validates_capabilities(self):
        catalog = _catalog(_route(), _GEMINI)
        decision = dr.select_route(
            catalog,
            dr.RouteRequest(
                difficulty=dr.TaskDifficulty.STANDARD,
                route_id="gemini-routine",
                required_capabilities=frozenset({"coding"}),
            ),
            usage=_usage(google_antigravity=(90.0, "fresh", 1.0)),
            available_providers=_ALL_AVAILABLE,
        )
        assert decision.selected is False
        assert "capabilit" in decision.reason.lower()

    def test_explicit_route_rejected_when_provider_unavailable(self):
        catalog = _catalog(_route(), _GEMINI)
        decision = dr.select_route(
            catalog,
            dr.RouteRequest(difficulty=dr.TaskDifficulty.STANDARD, route_id="gemini-routine"),
            usage=_usage(),
            available_providers=frozenset({"openai-codex"}),
        )
        assert decision.selected is False
        assert "available" in decision.reason.lower()

    def test_explicit_route_rejected_when_disabled(self):
        catalog = _catalog(_route(), dict(_GEMINI, enabled=False))
        decision = dr.select_route(
            catalog,
            dr.RouteRequest(difficulty=dr.TaskDifficulty.STANDARD, route_id="gemini-routine"),
            usage=_usage(),
            available_providers=_ALL_AVAILABLE,
        )
        assert decision.selected is False
        assert "disabled" in decision.reason.lower()

    def test_unknown_explicit_route_id_is_rejected(self):
        catalog = _catalog(_route())
        decision = dr.select_route(
            catalog,
            dr.RouteRequest(difficulty=dr.TaskDifficulty.STANDARD, route_id="nope"),
            usage=_usage(),
            available_providers=_ALL_AVAILABLE,
        )
        assert decision.selected is False
        assert "nope" in decision.reason

    def test_explicit_override_bypasses_reserve_but_says_so(self):
        catalog = _catalog(_GEMINI)
        decision = dr.select_route(
            catalog,
            dr.RouteRequest(difficulty=dr.TaskDifficulty.ROUTINE, route_id="gemini-routine"),
            usage=_usage(google_antigravity=(2.0, "fresh", 5.0)),
            available_providers=_ALL_AVAILABLE,
        )
        assert decision.selected is True
        assert decision.reserve_bypassed is True
        assert "reserve" in decision.reason.lower()


class TestReserveThreshold:
    def test_route_below_reserve_is_excluded(self):
        catalog = _catalog(_route(priority=1), _GEMINI)
        decision = dr.select_route(
            catalog,
            dr.RouteRequest(difficulty=dr.TaskDifficulty.STANDARD),
            usage=_usage(
                openai_codex=(5.0, "fresh", 1.0),
                google_antigravity=(90.0, "fresh", 1.0),
            ),
            available_providers=_ALL_AVAILABLE,
        )
        assert decision.route_id == "gemini-routine"

    def test_remaining_exactly_at_reserve_is_allowed(self):
        catalog = _catalog(_route(priority=1))
        decision = dr.select_route(
            catalog,
            dr.RouteRequest(difficulty=dr.TaskDifficulty.STANDARD),
            usage=_usage(openai_codex=(15.0, "fresh", 1.0)),
            available_providers=_ALL_AVAILABLE,
        )
        assert decision.route_id == "codex-standard"

    def test_stale_usage_below_reserve_also_excludes(self):
        catalog = _catalog(_route(priority=1), _GEMINI)
        decision = dr.select_route(
            catalog,
            dr.RouteRequest(difficulty=dr.TaskDifficulty.STANDARD),
            usage=_usage(
                openai_codex=(1.0, "stale", 900.0),
                google_antigravity=(90.0, "fresh", 1.0),
            ),
            available_providers=_ALL_AVAILABLE,
        )
        assert decision.route_id == "gemini-routine"

    def test_known_healthy_usage_outranks_unknown_usage(self):
        """Unknown usage stays eligible but is never treated as unlimited."""
        catalog = _catalog(_route(priority=1), _GEMINI)
        decision = dr.select_route(
            catalog,
            dr.RouteRequest(difficulty=dr.TaskDifficulty.STANDARD),
            usage=_usage(google_antigravity=(90.0, "fresh", 1.0)),
            available_providers=_ALL_AVAILABLE,
        )
        assert decision.route_id == "gemini-routine"
        assert decision.usage_freshness == "fresh"

    def test_all_unknown_usage_uses_fixed_priority(self):
        catalog = _catalog(_route(priority=1), _GEMINI)
        decision = dr.select_route(
            catalog,
            dr.RouteRequest(difficulty=dr.TaskDifficulty.STANDARD),
            usage=_usage(),
            available_providers=_ALL_AVAILABLE,
        )
        assert decision.route_id == "codex-standard"
        assert decision.usage_freshness == "unknown"

    def test_unknown_usage_skip_policy_excludes_route(self):
        catalog = _catalog(_route(priority=1), _GEMINI, unknown_usage="skip")
        decision = dr.select_route(
            catalog,
            dr.RouteRequest(difficulty=dr.TaskDifficulty.STANDARD),
            usage=_usage(google_antigravity=(90.0, "fresh", 1.0)),
            available_providers=_ALL_AVAILABLE,
        )
        assert decision.route_id == "gemini-routine"

    def test_unknown_usage_does_not_outrank_known_healthy_route(self):
        """Known healthy wins even when the unknown route has better priority."""
        catalog = _catalog(_route(priority=1), _GEMINI)
        decision = dr.select_route(
            catalog,
            dr.RouteRequest(difficulty=dr.TaskDifficulty.STANDARD),
            usage=_usage(google_antigravity=(90.0, "fresh", 1.0)),
            available_providers=_ALL_AVAILABLE,
        )
        assert decision.route_id == "gemini-routine"


class TestSelectorPurity:
    def test_selector_makes_no_network_or_fetch_calls(self, monkeypatch):
        import agent.account_usage as au

        def _boom(*a, **k):  # pragma: no cover - must never run
            raise AssertionError("selector must not fetch usage")

        monkeypatch.setattr(au, "fetch_account_usage", _boom)
        catalog = _catalog(_route(), _GEMINI)
        decision = dr.select_route(
            catalog,
            dr.RouteRequest(difficulty=dr.TaskDifficulty.STANDARD),
            usage=_usage(openai_codex=(90.0, "fresh", 1.0)),
            available_providers=_ALL_AVAILABLE,
        )
        assert decision.selected is True
