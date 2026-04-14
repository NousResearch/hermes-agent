from __future__ import annotations

from agent.routing_governance import promote_route
from agent.routing_policy_bridge import build_routing_governance_snapshot, build_task_class_route_plan


REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]


def test_build_task_class_route_plan_prefers_policy_family_and_reports_coding_eligibility(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    promote_route(provider="openai-codex", model="gpt-5.4", reason="qualified coding route")

    plan = build_task_class_route_plan(
        task_class="implementation",
        primary_route={"provider": "openai-codex", "model": "gpt-5.4"},
        fallback_routes=[
            {"provider": "google", "model": "gemini-2.0-flash"},
            {"provider": "moonshot", "model": "kimi-k2"},
        ],
        repo_root=REPO_ROOT,
        probe_summary={
            "results": [
                {"provider": "openai-codex", "model": "gpt-5.4", "ok": True, "classification": "ok"},
                {"provider": "google", "model": "gemini-2.0-flash", "ok": True, "classification": "ok"},
            ]
        },
    )

    assert plan["selected_route"]["family"] == "Codex"
    assert plan["selected_route"]["provider"] == "openai-codex"
    assert plan["preferred_families"][:2] == ["Codex", "Gemini"]
    assert plan["coding_policy"]["route_is_qualified"] is True
    assert plan["coding_policy"]["eligible_for_high_risk_rollout"] is True


def test_build_task_class_route_plan_falls_back_when_default_family_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    plan = build_task_class_route_plan(
        task_class="research",
        primary_route={"provider": "anthropic", "model": "claude-sonnet-4"},
        fallback_routes=[{"provider": "moonshot", "model": "kimi-k2"}],
        repo_root=REPO_ROOT,
    )

    assert plan["selected_route"]["family"] == "Kimi"
    assert "Gemini" in plan["missing_preferred_families"]
    assert plan["parity"]["default_family_available"] is False


def test_build_routing_governance_snapshot_writes_snapshot_and_cost_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    promote_route(provider="google", model="gemini-2.0-flash", reason="qualified research route")

    snapshot = build_routing_governance_snapshot(
        primary_route={"provider": "openai-codex", "model": "gpt-5.4"},
        fallback_routes=[
            {"provider": "google", "model": "gemini-2.0-flash"},
            {"provider": "moonshot", "model": "kimi-k2"},
        ],
        repo_root=REPO_ROOT,
        probe_summary={
            "results": [
                {"provider": "openai-codex", "model": "gpt-5.4", "ok": True, "classification": "ok"},
                {"provider": "google", "model": "gemini-2.0-flash", "ok": False, "classification": "rate_limit"},
                {"provider": "moonshot", "model": "kimi-k2", "ok": True, "classification": "ok"},
            ]
        },
    )

    assert snapshot["schema"] == "hermes.session_topology_routing_snapshot.v1"
    assert snapshot["policy"]["policy_id"] == "session_topology_routing_policy_v1_2026-04-04"
    assert snapshot["cost_governance"]["family_counts"]["Codex"] >= 1
    assert snapshot["health_summary"]["classification_counts"]["ok"] >= 1
    assert snapshot["snapshot_path"].endswith("session_topology_snapshot.json")
