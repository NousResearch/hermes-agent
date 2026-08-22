"""delegate_task must resolve credentials PER TASK when routing is enabled.

Child construction is stubbed out; these tests assert only which
provider:model each task was built with, and that parent/child isolation
holds. No network, OAuth, auth store, or credential values are involved.
"""

import json

import pytest

from agent import delegation_usage_cache as duc
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
    "routing": {"enabled": True},
    "routes": [CODEX_ROUTE, GEMINI_ROUTE],
    "max_concurrent_children": 4,
}


class _Parent:
    """Minimal stand-in for the parent agent."""

    provider = "anthropic"
    model = "claude-opus-5"
    api_key = "parent-secret-key"
    base_url = "https://api.anthropic.com"
    session_id = "parent-session"
    _delegate_depth = 0


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.setattr(duc, "_cache_path", lambda: tmp_path / "usage.json")
    duc.reset_refresh_state()
    monkeypatch.setattr(duc, "_spawn_refresh", lambda provider: None)
    monkeypatch.setattr(dt, "_load_config", lambda: ROUTED_CFG)
    monkeypatch.setattr(
        dt,
        "_available_route_providers",
        lambda routes: frozenset({"openai-codex", "google-antigravity"}),
    )

    def _resolve(requested=None, target_model=None, **kwargs):
        table = {
            "openai-codex": {
                "provider": "openai-codex",
                "base_url": "https://chatgpt.com/backend-api/codex",
                "api_key": "codex-key",
                "api_mode": "codex_responses",
            },
            "google-antigravity": {
                "provider": "google-antigravity",
                "base_url": "https://cloudcode.example/v1",
                "api_key": "agy-key",
                "api_mode": "gemini_cloudcode",
                "project_id": "agy-project-123",
            },
        }
        if requested not in table:
            raise RuntimeError(f"unknown provider {requested}")
        return {**table[requested], "model": target_model}

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", _resolve)

    built = []

    class _Child:
        def __init__(self, kwargs):
            self.kwargs = kwargs
            self.session_id = f"child-{len(built)}"

    def _fake_build(**kwargs):
        built.append(kwargs)
        return _Child(kwargs)

    monkeypatch.setattr(dt, "_build_child_preserving_parent_tools", _fake_build)

    def _fake_finalize(results, *a, **kw):
        return results

    monkeypatch.setattr(
        dt, "_run_child_lifecycle",
        lambda *a, **kw: {"task": 0, "status": "success", "summary": "done"},
    )
    return built


def _run(harness, **kwargs):
    dt.delegate_task(parent_agent=_Parent(), background=False, **kwargs)
    return harness


class TestPerTaskRouting:
    def test_batch_tasks_build_children_on_independent_routes(self, harness):
        built = _run(
            harness,
            tasks=[
                {
                    "goal": "Rename the deprecated helper across the utils package",
                    "difficulty": "routine",
                },
                {
                    "goal": "Redesign the scheduler to remove the global lock",
                    "difficulty": "complex",
                },
            ],
        )
        assert len(built) == 2
        assert built[0]["override_provider"] == "google-antigravity"
        assert built[0]["model"] == "gemini-3-flash-agent"
        assert built[1]["override_provider"] == "openai-codex"
        assert built[1]["model"] == "gpt-5.6-sol"

    def test_batch_shares_one_availability_and_usage_preflight(
        self, harness, monkeypatch
    ):
        availability_calls = []
        usage_calls = []
        original_build_usage_view = duc.build_route_usage_view

        def _available(routes):
            availability_calls.append(tuple(route.id for route in routes))
            return frozenset({"openai-codex", "google-antigravity"})

        def _usage(*args, **kwargs):
            usage_calls.append(tuple(args[0]))
            return original_build_usage_view(*args, **kwargs)

        monkeypatch.setattr(dt, "_available_route_providers", _available)
        monkeypatch.setattr(duc, "build_route_usage_view", _usage)
        _run(
            harness,
            tasks=[
                {"goal": "Rename the deprecated helper everywhere", "difficulty": "routine"},
                {"goal": "Redesign the scheduler without global locks", "difficulty": "complex"},
            ],
        )
        assert len(availability_calls) == 1
        assert len(usage_calls) == 1

    def test_single_goal_uses_top_level_routing_fields(self, harness):
        built = _run(
            harness,
            goal="Summarize the changelog entries for the release notes",
            difficulty="routine",
            difficulty_reason="mechanical summarization",
        )
        assert built[0]["override_provider"] == "google-antigravity"

    def test_explicit_per_task_route_overrides_difficulty(self, harness):
        built = _run(
            harness,
            tasks=[
                {
                    "goal": "Rename the deprecated helper across the utils package",
                    "difficulty": "routine",
                    "route": "codex-standard",
                },
                {
                    "goal": "Redesign the scheduler to remove the global lock",
                    "difficulty": "complex",
                },
            ],
        )
        assert built[0]["override_provider"] == "openai-codex"

    def test_capabilities_route_the_task(self, harness):
        built = _run(
            harness,
            goal="Read the entire 400k-token log bundle and extract failures",
            difficulty="standard",
            required_capabilities=["long_context"],
        )
        assert built[0]["override_provider"] == "google-antigravity"


class TestChildIsolation:
    def test_child_never_inherits_parent_credentials_when_routed(self, harness):
        built = _run(
            harness,
            goal="Redesign the scheduler to remove the global lock cleanly",
            difficulty="complex",
        )
        kwargs = built[0]
        assert kwargs["override_api_key"] == "codex-key"
        assert kwargs["override_api_key"] != _Parent.api_key
        assert kwargs["override_base_url"] == "https://chatgpt.com/backend-api/codex"
        assert kwargs["override_provider"] != _Parent.provider

    def test_public_route_metadata_omits_raw_usage_and_reason_text(self):
        class _Child:
            _delegate_route_decision = {
                "route_id": "codex-standard",
                "provider": "openai-codex",
                "model": "gpt-5.6-sol",
                "model_class": "advanced",
                "difficulty": "standard",
                "usage_freshness": "stale",
                "usage_remaining_percent": 12.3456,
                "usage_age_seconds": 678.9,
                "reason": "account alice@example.com has 12.3456% remaining",
                "exclusions": (
                    "route 'other' has 2% remaining, below its 10% reserve",
                ),
            }

        metadata = dt._public_route_metadata(_Child())
        assert metadata == {
            "id": "codex-standard",
            "provider": "openai-codex",
            "backend": "native",
            "model": "gpt-5.6-sol",
            "model_class": "advanced",
            "difficulty": "standard",
            "usage_freshness": "stale",
            "explicit_override": False,
            "reserve_bypassed": False,
            "exclusion_codes": ["usage_reserve"],
        }
        blob = json.dumps(metadata)
        assert "alice@example.com" not in blob
        assert "12.3456" not in blob
        assert "678.9" not in blob

    def test_antigravity_project_propagates_only_to_its_own_child(self, harness):
        built = _run(
            harness,
            tasks=[
                {
                    "goal": "Rename the deprecated helper across the utils package",
                    "difficulty": "routine",
                },
                {
                    "goal": "Redesign the scheduler to remove the global lock",
                    "difficulty": "complex",
                },
            ],
        )
        assert built[0]["override_provider_project_id"] == "agy-project-123"
        assert built[1]["override_provider_project_id"] is None

    def test_cross_provider_children_do_not_share_api_mode(self, harness):
        built = _run(
            harness,
            tasks=[
                {
                    "goal": "Rename the deprecated helper across the utils package",
                    "difficulty": "routine",
                },
                {
                    "goal": "Redesign the scheduler to remove the global lock",
                    "difficulty": "complex",
                },
            ],
        )
        assert built[0]["override_api_mode"] == "gemini_cloudcode"
        assert built[1]["override_api_mode"] == "codex_responses"


class TestLegacyDispatchUnchanged:
    def test_legacy_config_resolves_once_for_all_tasks(self, tmp_path, monkeypatch):
        monkeypatch.setattr(duc, "_cache_path", lambda: tmp_path / "usage.json")
        legacy_cfg = {"provider": "openai-codex", "model": "pinned-model"}
        monkeypatch.setattr(dt, "_load_config", lambda: legacy_cfg)

        calls = []

        def _resolve(requested=None, target_model=None, **kwargs):
            calls.append(requested)
            return {
                "provider": "openai-codex",
                "model": target_model,
                "base_url": "https://chatgpt.com/backend-api/codex",
                "api_key": "codex-key",
                "api_mode": "codex_responses",
            }

        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider", _resolve
        )
        built = []
        monkeypatch.setattr(
            dt,
            "_build_child_preserving_parent_tools",
            lambda **kw: built.append(kw) or type("C", (), {"session_id": "c"})(),
        )
        monkeypatch.setattr(
            dt, "_run_child_lifecycle",
            lambda *a, **kw: {"task": 0, "status": "success", "summary": "done"},
        )

        dt.delegate_task(
            parent_agent=_Parent(),
            background=False,
            tasks=[
                {"goal": "Rename the deprecated helper across the utils package"},
                {"goal": "Redesign the scheduler to remove the global lock"},
            ],
        )
        assert [b["model"] for b in built] == ["pinned-model", "pinned-model"]
        assert calls == ["openai-codex"]  # resolved once, not per task
