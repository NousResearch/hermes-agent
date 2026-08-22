"""Integration tests: route catalog -> delegate_tool credential resolution.

All provider resolution, auth probing, and usage fetching is mocked. Nothing
here touches the network, an OAuth flow, the auth store, or a credential
value.
"""

from datetime import datetime, timedelta, timezone

import pytest

from agent import delegation_routing as dr
from agent import delegation_usage_cache as duc
from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
import tools.delegate_tool as dt


CODEX_ROUTE = {
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

GEMINI_ROUTE = {
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

ROUTED_CFG = {
    "routing": {"enabled": True, "usage_ttl_seconds": 300, "usage_stale_seconds": 1800},
    "routes": [CODEX_ROUTE, GEMINI_ROUTE],
}


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(duc, "_cache_path", lambda: tmp_path / "usage.json")
    duc.reset_refresh_state()
    monkeypatch.setattr(duc, "_spawn_refresh", lambda provider: None)


@pytest.fixture
def fake_runtime(monkeypatch):
    """Mock the runtime provider resolver; record what it was asked for."""
    calls = []

    def _resolve(requested=None, target_model=None, **kwargs):
        calls.append({"requested": requested, "target_model": target_model})
        if requested == "openai-codex":
            return {
                "provider": "openai-codex",
                "model": target_model,
                "base_url": "https://chatgpt.com/backend-api/codex",
                "api_key": "test-codex-key",
                "api_mode": "codex_responses",
            }
        if requested == "google-antigravity":
            return {
                "provider": "google-antigravity",
                "model": target_model,
                "base_url": "https://cloudcode.example/v1",
                "api_key": "test-agy-key",
                "api_mode": "gemini_cloudcode",
                "project_id": "agy-project-123",
            }
        raise RuntimeError(f"no such provider {requested}")

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider", _resolve
    )
    return calls


@pytest.fixture
def all_available(monkeypatch):
    monkeypatch.setattr(
        dt, "_available_route_providers", lambda routes: frozenset(
            {"openai-codex", "google-antigravity"}
        )
    )


def _store_usage(provider, label, used, age=10):
    duc.store_snapshot(
        AccountUsageSnapshot(
            provider=provider,
            source="usage_api",
            fetched_at=datetime.now(timezone.utc) - timedelta(seconds=age),
            windows=(AccountUsageWindow(label=label, used_percent=used),),
        )
    )


class TestLegacyBehaviorUnchanged:
    def test_no_routes_uses_legacy_provider_model(self, fake_runtime):
        cfg = {"provider": "openai-codex", "model": "gpt-5.6-sol"}
        creds = dt._resolve_delegation_credentials(cfg, parent_agent=None)
        assert creds["provider"] == "openai-codex"
        assert creds["model"] == "gpt-5.6-sol"
        assert creds.get("route_decision") is None

    def test_no_config_at_all_inherits_from_parent(self):
        creds = dt._resolve_delegation_credentials({}, parent_agent=None)
        assert creds["provider"] is None
        assert creds["model"] is None
        assert creds["base_url"] is None

    def test_routing_disabled_falls_back_to_legacy(self, fake_runtime):
        cfg = {
            "provider": "openai-codex",
            "model": "legacy-model",
            "routing": {"enabled": False},
            "routes": [CODEX_ROUTE, GEMINI_ROUTE],
        }
        creds = dt._resolve_delegation_credentials(cfg, parent_agent=None)
        assert creds["model"] == "legacy-model"
        assert creds.get("route_decision") is None

    def test_routed_config_without_request_still_resolves(self, fake_runtime, all_available):
        """A routed config with no difficulty hint uses the 'standard' default."""
        creds = dt._resolve_delegation_credentials(ROUTED_CFG, parent_agent=None)
        assert creds["provider"] == "openai-codex"
        assert creds["route_decision"]["route_id"] == "codex-standard"


class TestRoutedResolution:
    def test_runtime_resolver_exception_text_is_not_exposed(
        self, all_available, monkeypatch
    ):
        def _boom(**kwargs):
            raise RuntimeError(
                "account=bob@example.com project=secret-proj https://x/?key=SECRET"
            )

        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider", _boom
        )
        with pytest.raises(ValueError) as exc:
            dt._resolve_delegation_credentials(
                ROUTED_CFG,
                parent_agent=None,
                request=dr.RouteRequest(difficulty=dr.TaskDifficulty.COMPLEX),
            )
        message = str(exc.value)
        assert "RuntimeError" in message
        for secret in ("bob@example.com", "secret-proj", "SECRET", "https://"):
            assert secret not in message

    def test_routine_task_routes_to_gemini(self, fake_runtime, all_available):
        creds = dt._resolve_delegation_credentials(
            ROUTED_CFG,
            parent_agent=None,
            request=dr.RouteRequest(
                difficulty=dr.TaskDifficulty.ROUTINE,
                difficulty_reason="mechanical file rename",
            ),
        )
        assert creds["provider"] == "google-antigravity"
        assert creds["model"] == "gemini-3-flash-agent"
        assert creds["api_key"] == "test-agy-key"
        decision = creds["route_decision"]
        assert decision["route_id"] == "gemini-routine"
        assert decision["difficulty"] == "routine"
        assert decision["difficulty_reason"] == "mechanical file rename"

    def test_complex_task_routes_to_codex(self, fake_runtime, all_available):
        creds = dt._resolve_delegation_credentials(
            ROUTED_CFG,
            parent_agent=None,
            request=dr.RouteRequest(difficulty=dr.TaskDifficulty.COMPLEX),
        )
        assert creds["provider"] == "openai-codex"
        assert creds["model"] == "gpt-5.6-sol"

    def test_antigravity_project_is_propagated_to_child(self, fake_runtime, all_available):
        creds = dt._resolve_delegation_credentials(
            ROUTED_CFG,
            parent_agent=None,
            request=dr.RouteRequest(difficulty=dr.TaskDifficulty.ROUTINE),
        )
        assert creds["provider_project_id"] == "agy-project-123"

    def test_resolver_asks_for_the_selected_pair_only(self, fake_runtime, all_available):
        dt._resolve_delegation_credentials(
            ROUTED_CFG,
            parent_agent=None,
            request=dr.RouteRequest(difficulty=dr.TaskDifficulty.ROUTINE),
        )
        assert fake_runtime == [
            {"requested": "google-antigravity", "target_model": "gemini-3-flash-agent"}
        ]

    def test_depleted_route_is_skipped_using_cached_usage(self, fake_runtime, all_available):
        _store_usage("openai-codex", "Session", used=99.0)
        _store_usage("google-antigravity", "Gemini Models (5h)", used=5.0)
        creds = dt._resolve_delegation_credentials(
            ROUTED_CFG,
            parent_agent=None,
            request=dr.RouteRequest(difficulty=dr.TaskDifficulty.STANDARD),
        )
        assert creds["provider"] == "google-antigravity"
        assert "reserve" in creds["route_decision"]["reason"]

    def test_window_prefixes_isolate_the_pool(self, fake_runtime, all_available):
        """A depleted Claude pool must not disqualify the Gemini route."""
        duc.store_snapshot(
            AccountUsageSnapshot(
                provider="google-antigravity",
                source="quota_summary",
                fetched_at=datetime.now(timezone.utc),
                windows=(
                    AccountUsageWindow(label="Gemini Models (5h)", used_percent=10.0),
                    AccountUsageWindow(
                        label="Claude and GPT models (5h)", used_percent=100.0
                    ),
                ),
            )
        )
        creds = dt._resolve_delegation_credentials(
            ROUTED_CFG,
            parent_agent=None,
            request=dr.RouteRequest(difficulty=dr.TaskDifficulty.ROUTINE),
        )
        assert creds["provider"] == "google-antigravity"

    def test_explicit_route_override_wins(self, fake_runtime, all_available):
        creds = dt._resolve_delegation_credentials(
            ROUTED_CFG,
            parent_agent=None,
            request=dr.RouteRequest(
                difficulty=dr.TaskDifficulty.STANDARD, route_id="gemini-routine"
            ),
        )
        assert creds["provider"] == "google-antigravity"
        assert creds["route_decision"]["explicit_override"] is True

    def test_unavailable_provider_is_not_selected(self, fake_runtime, monkeypatch):
        monkeypatch.setattr(
            dt, "_available_route_providers", lambda routes: frozenset({"openai-codex"})
        )
        creds = dt._resolve_delegation_credentials(
            ROUTED_CFG,
            parent_agent=None,
            request=dr.RouteRequest(difficulty=dr.TaskDifficulty.STANDARD),
        )
        assert creds["provider"] == "openai-codex"

    def test_no_eligible_route_raises_actionable_error(self, fake_runtime, all_available):
        with pytest.raises(ValueError) as exc:
            dt._resolve_delegation_credentials(
                ROUTED_CFG,
                parent_agent=None,
                request=dr.RouteRequest(
                    difficulty=dr.TaskDifficulty.FRONTIER,
                ),
            )
        assert "frontier" in str(exc.value)

    def test_selection_makes_no_usage_fetch(self, fake_runtime, all_available, monkeypatch):
        def _boom(*a, **k):  # pragma: no cover
            raise AssertionError("must not fetch usage during selection")

        monkeypatch.setattr(duc, "_fetch_account_usage", _boom)
        monkeypatch.setattr("agent.account_usage.fetch_account_usage", _boom)
        creds = dt._resolve_delegation_credentials(
            ROUTED_CFG,
            parent_agent=None,
            request=dr.RouteRequest(difficulty=dr.TaskDifficulty.ROUTINE),
        )
        assert creds["provider"] == "google-antigravity"

    def test_stale_usage_is_used_and_schedules_one_refresh(
        self, fake_runtime, all_available, monkeypatch
    ):
        _store_usage("google-antigravity", "Gemini Models (5h)", used=5.0, age=900)
        scheduled = []
        monkeypatch.setattr(duc, "_spawn_refresh", lambda p: scheduled.append(p))
        creds = dt._resolve_delegation_credentials(
            ROUTED_CFG,
            parent_agent=None,
            request=dr.RouteRequest(difficulty=dr.TaskDifficulty.ROUTINE),
        )
        assert creds["route_decision"]["usage_freshness"] == "stale"
        assert scheduled.count("google-antigravity") == 1


class TestRequestFromArgs:
    def test_builds_request_from_task_fields(self):
        request = dt._route_request_from_args(
            {
                "difficulty": "complex",
                "difficulty_reason": "multi-file refactor with hidden coupling",
                "required_capabilities": ["coding", "reasoning"],
                "minimum_model_class": "advanced",
                "route": "codex-standard",
            }
        )
        assert request.difficulty is dr.TaskDifficulty.COMPLEX
        assert request.difficulty_reason == "multi-file refactor with hidden coupling"
        assert request.required_capabilities == frozenset({"coding", "reasoning"})
        assert request.minimum_model_class is dr.ModelClass.ADVANCED
        assert request.route_id == "codex-standard"

    def test_task_fields_override_top_level(self):
        request = dt._route_request_from_args(
            {"difficulty": "routine"}, top_level={"difficulty": "complex"}
        )
        assert request.difficulty is dr.TaskDifficulty.ROUTINE

    def test_top_level_used_when_task_silent(self):
        request = dt._route_request_from_args(
            {}, top_level={"difficulty": "frontier", "required_capabilities": ["vision"]}
        )
        assert request.difficulty is dr.TaskDifficulty.FRONTIER
        assert request.required_capabilities == frozenset({"vision"})

    def test_defaults_to_standard_difficulty(self):
        request = dt._route_request_from_args({})
        assert request.difficulty is dr.TaskDifficulty.STANDARD
        assert request.route_id is None

    def test_invalid_difficulty_degrades_to_standard(self):
        request = dt._route_request_from_args({"difficulty": "impossible"})
        assert request.difficulty is dr.TaskDifficulty.STANDARD

    def test_invalid_model_class_is_ignored(self):
        request = dt._route_request_from_args({"minimum_model_class": "turbo"})
        assert request.minimum_model_class is None


class TestBatchIndependentRoutes:
    def test_batch_tasks_select_different_routes(self, fake_runtime, all_available):
        tasks = [
            {"goal": "rename symbols", "difficulty": "routine"},
            {"goal": "design the scheduler", "difficulty": "complex"},
        ]
        picked = [
            dt._resolve_delegation_credentials(
                ROUTED_CFG,
                parent_agent=None,
                request=dt._route_request_from_args(task),
            )
            for task in tasks
        ]
        assert [c["provider"] for c in picked] == ["google-antigravity", "openai-codex"]
        assert [c["model"] for c in picked] == ["gemini-3-flash-agent", "gpt-5.6-sol"]

    def test_per_task_credentials_are_isolated(self, fake_runtime, all_available):
        a = dt._resolve_delegation_credentials(
            ROUTED_CFG, parent_agent=None,
            request=dr.RouteRequest(difficulty=dr.TaskDifficulty.ROUTINE),
        )
        b = dt._resolve_delegation_credentials(
            ROUTED_CFG, parent_agent=None,
            request=dr.RouteRequest(difficulty=dr.TaskDifficulty.COMPLEX),
        )
        assert a["api_key"] != b["api_key"]
        assert a["base_url"] != b["base_url"]
        assert a["provider_project_id"] == "agy-project-123"
        assert b.get("provider_project_id") is None


class TestSchema:
    def test_schema_exposes_routing_fields(self):
        props = dt.DELEGATE_TASK_SCHEMA["parameters"]["properties"]
        for field in ("route", "difficulty", "difficulty_reason", "required_capabilities"):
            assert field in props, f"top-level '{field}' missing from schema"
        task_props = props["tasks"]["items"]["properties"]
        for field in ("route", "difficulty", "difficulty_reason", "required_capabilities"):
            assert field in task_props, f"tasks[].'{field}' missing from schema"

    def test_difficulty_enum_matches_taxonomy(self):
        props = dt.DELEGATE_TASK_SCHEMA["parameters"]["properties"]
        assert props["difficulty"]["enum"] == ["routine", "standard", "complex", "frontier"]
        assert props["minimum_model_class"]["enum"] == [
            "fast",
            "balanced",
            "advanced",
            "frontier",
        ]

    def test_descriptions_define_taxonomies_and_ask_for_reasoning(self):
        props = dt.DELEGATE_TASK_SCHEMA["parameters"]["properties"]
        difficulty_doc = props["difficulty"]["description"].lower()
        for term in ("routine", "standard", "complex", "frontier"):
            assert term in difficulty_doc
        assert "why" in props["difficulty_reason"]["description"].lower()
        caps_doc = props["required_capabilities"]["description"].lower()
        for cap in ("coding", "reasoning", "tool_use", "long_context", "vision", "review"):
            assert cap in caps_doc

    def test_schema_carries_no_dynamic_usage_data(self, monkeypatch, all_available):
        """Quota numbers must never enter the tool schema (prompt-cache churn)."""
        _store_usage("openai-codex", "5h limit", used=42.0)
        _store_usage("google-antigravity", "Gemini Models", used=77.0)
        monkeypatch.setattr(dt, "_load_config", lambda: ROUTED_CFG)

        import json

        overrides = dt._build_dynamic_schema_overrides()
        blob = json.dumps(overrides) + json.dumps(dt.DELEGATE_TASK_SCHEMA)
        for leaked in ("42", "77", "remaining", "% left", "quota"):
            assert leaked not in blob.lower(), f"schema leaked usage token {leaked!r}"

    def test_schema_is_stable_across_usage_changes(self, monkeypatch, all_available):
        import json

        monkeypatch.setattr(dt, "_load_config", lambda: ROUTED_CFG)
        _store_usage("openai-codex", "5h limit", used=1.0)
        before = json.dumps(dt._build_dynamic_schema_overrides(), sort_keys=True)
        _store_usage("openai-codex", "5h limit", used=99.0)
        after = json.dumps(dt._build_dynamic_schema_overrides(), sort_keys=True)
        assert before == after
