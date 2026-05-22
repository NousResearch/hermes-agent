"""Tests for the Marketing Agent Factory plugin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_store_initializes_schema_and_seed_profiles(isolate_home):
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    result = MarketingFactoryPipeline(store).initialize_samples()

    state = store.load()
    assert state["schema_version"] == 1
    assert {app["slug"] for app in result["apps"]} == {"pupular", "setvenue"}
    assert "pupular" in state["brand_memories"]
    assert "setvenue" in state["brand_memories"]
    assert store.paths.audit_file.exists()


def test_brand_profile_isolation_prevents_cross_app_draft_access(isolate_home):
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()
    generated = pipe.generate_campaign("pupular", days=1)
    draft_id = generated["drafts"][0]["id"]

    assert store.get_draft(draft_id, app_slug="pupular")["app_slug"] == "pupular"
    with pytest.raises(PermissionError):
        store.get_draft(draft_id, app_slug="setvenue")


def test_pipeline_generates_campaign_drafts_with_model_routing_and_safety(isolate_home):
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()

    result = pipe.generate_campaign("setvenue", days=3)

    assert result["campaign"]["app_slug"] == "setvenue"
    assert result["campaign"]["model_route"] == "premium"
    assert len(result["drafts"]) == 3
    assert all(d["app_slug"] == "setvenue" for d in result["drafts"])
    assert all(d["safety"]["passed"] for d in result["drafts"])
    assert {d["model_route"] for d in result["drafts"]}


def test_approval_workflow_blocks_scheduling_until_approved(isolate_home):
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()
    draft = pipe.generate_campaign("pupular", days=1)["drafts"][0]

    with pytest.raises(ValueError):
        store.schedule_draft(draft["id"], "2026-01-01T12:00:00+00:00")

    approval = store.set_approval(draft["id"], "approved", reviewer="tester")
    schedule = store.schedule_draft(draft["id"], "2026-01-01T12:00:00+00:00")

    assert approval["status"] == "approved"
    assert schedule["status"] == "scheduled"


def test_dry_run_publisher_never_marks_public_posted(isolate_home):
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()
    result = pipe.run_full_dry_run("pupular", days=2, reviewer="tester")

    assert len(result["publish_events"]) == 2
    assert all(event["mode"] == "dry_run" for event in result["publish_events"])
    assert all(event["would_post"] is True for event in result["publish_events"])
    assert all(event["posted"] is False for event in result["publish_events"])


def test_analytics_feedback_loop_is_app_scoped(isolate_home):
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()
    pipe.run_full_dry_run("setvenue", days=1, reviewer="tester")

    state = store.load()
    assert state["brand_memories"]["setvenue"]["learnings"]
    assert state["brand_memories"]["pupular"]["learnings"] == []


def test_model_routing_and_budget_controls_are_present(isolate_home):
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    store.initialize()
    summary = store.summary()

    assert summary["budgets"]["daily_tokens"] > 0
    policy = summary["model_routing_policy"]
    assert set(policy) == {"cheap", "mid", "premium"}
    assert "strategy" in policy["premium"]["tasks"]
    assert "duplicate_check" in policy["cheap"]["tasks"]


def test_audit_log_records_major_events(isolate_home):
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()
    pipe.run_full_dry_run("pupular", days=1, reviewer="tester")

    actions = [event["action"] for event in store.list_audit(app_slug="pupular", limit=100)]
    assert "brand_profile.upserted" in actions
    assert "campaign.generated" in actions
    assert "draft.approved" in actions
    assert "draft.scheduled" in actions
    assert "publish.dry_run" in actions
    assert "analytics.recorded" in actions


def test_tool_handler_runs_full_dry_run(isolate_home):
    from plugins.marketing_factory.tools import handle_marketing_factory

    init = json.loads(handle_marketing_factory({"action": "init"}))
    assert init["success"] is True
    result = json.loads(handle_marketing_factory({"action": "full_dry_run", "app_slug": "setvenue", "days": 1, "reviewer": "tester"}))
    assert result["success"] is True
    assert len(result["result"]["publish_events"]) == 1


def test_cli_smoke_path_outputs_status(tmp_path, capsys):
    from plugins.marketing_factory.cli import marketing_command

    store_path = tmp_path / "mf"
    init_args = argparse.Namespace(store_path=str(store_path), marketing_command="init")
    assert marketing_command(init_args) == 0

    status_args = argparse.Namespace(store_path=str(store_path), marketing_command="status")
    assert marketing_command(status_args) == 0
    captured = capsys.readouterr()
    assert '"apps": 2' in captured.out


def test_cli_full_dry_run_for_two_apps(tmp_path, capsys):
    from plugins.marketing_factory.cli import marketing_command

    store_path = tmp_path / "mf"
    assert marketing_command(argparse.Namespace(store_path=str(store_path), marketing_command="init")) == 0
    for app in ("pupular", "setvenue"):
        args = argparse.Namespace(store_path=str(store_path), marketing_command="full-dry-run", app=app, days=2, reviewer="tester")
        assert marketing_command(args) == 0

    status_args = argparse.Namespace(store_path=str(store_path), marketing_command="status")
    assert marketing_command(status_args) == 0
    out = capsys.readouterr().out
    assert '"dry_run_publish_events": 4' in out


def test_dashboard_manifest_and_assets_exist():
    manifest = Path("plugins/marketing_factory/dashboard/manifest.json")
    bundle = Path("plugins/marketing_factory/dashboard/dist/index.js")
    api_file = Path("plugins/marketing_factory/dashboard/plugin_api.py")

    data = json.loads(manifest.read_text(encoding="utf-8"))

    assert data["name"] == "marketing_factory"
    assert data["tab"]["path"] == "/marketing-factory"
    assert data["entry"] == "dist/index.js"
    assert data["api"] == "plugin_api.py"
    assert bundle.exists()
    assert api_file.exists()


def test_dashboard_api_overview_and_dry_run_actions(isolate_home):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from plugins.marketing_factory.dashboard.plugin_api import router

    app = FastAPI()
    app.include_router(router, prefix="/api/plugins/marketing_factory")
    client = TestClient(app)

    init = client.post("/api/plugins/marketing_factory/init")
    assert init.status_code == 200
    assert init.json()["overview"]["summary"]["apps"] == 2

    generated = client.post(
        "/api/plugins/marketing_factory/campaigns/generate",
        json={"app_slug": "pupular", "days": 1},
    )
    assert generated.status_code == 200
    draft = generated.json()["result"]["drafts"][0]

    approved = client.post(
        f"/api/plugins/marketing_factory/drafts/{draft['id']}/approve",
        json={"reviewer": "tester", "reason": "dashboard test"},
    )
    assert approved.status_code == 200
    assert approved.json()["overview"]["draft_status_counts"]["approved"] == 1

    scheduled = client.post(f"/api/plugins/marketing_factory/drafts/{draft['id']}/schedule", json={})
    assert scheduled.status_code == 200
    assert scheduled.json()["overview"]["summary"]["scheduled"] == 1

    published = client.post(f"/api/plugins/marketing_factory/drafts/{draft['id']}/publish-dry-run")
    assert published.status_code == 200
    event = published.json()["result"]
    assert event["mode"] == "dry_run"
    assert event["would_post"] is True
    assert event["posted"] is False
    assert published.json()["overview"]["summary"]["dry_run_publish_events"] == 1


def test_approval_writes_structured_brand_memory_and_invalidates_steering(isolate_home):
    """Phase 3: every approve/reject must capture a structured memory entry and
    invalidate any cached steering so the next campaign re-summarizes."""
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()

    generated = pipe.generate_campaign("pupular", days=2)
    drafts = generated["drafts"]
    assert len(drafts) == 2

    store.set_approval(drafts[0]["id"], "approved", reviewer="tester", reason="crisp and on-brand")
    store.set_approval(drafts[1]["id"], "rejected", reviewer="tester", reason="too generic for tiktok")

    state = store.load()
    learnings = state["brand_memories"]["pupular"]["learnings"]
    kinds = [entry.get("kind") for entry in learnings if isinstance(entry, dict)]
    assert "draft_approved" in kinds
    assert "draft_rejected" in kinds

    structured = [entry for entry in learnings if entry.get("kind") in {"draft_approved", "draft_rejected"}]
    assert all(entry.get("reason") and entry.get("excerpt") for entry in structured)
    assert all(entry.get("channel") for entry in structured)

    # Trigger steering (template path because PYTEST_CURRENT_TEST is set), persist it,
    # then prove a subsequent approval invalidates the cache.
    steering = pipe.brand_memory.get_steering(store, "pupular")
    assert steering is not None
    assert steering["method"] == "fallback"
    assert steering["approved_count"] == 1
    assert steering["rejected_count"] == 1
    assert state["brand_memories"]["pupular"].get("steering") is None  # we used pre-write state above

    after_state = store.load()
    assert after_state["brand_memories"]["pupular"].get("steering") is not None

    new_campaign = pipe.generate_campaign("pupular", days=1)
    store.set_approval(new_campaign["drafts"][0]["id"], "rejected", reviewer="tester", reason="off-tone")
    final_state = store.load()
    assert final_state["brand_memories"]["pupular"].get("steering") is None


def test_dashboard_bulk_approve_and_reject_endpoints(isolate_home):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from plugins.marketing_factory.dashboard.plugin_api import router

    app = FastAPI()
    app.include_router(router, prefix="/api/plugins/marketing_factory")
    client = TestClient(app)

    client.post("/api/plugins/marketing_factory/init")
    client.post(
        "/api/plugins/marketing_factory/campaigns/generate",
        json={"app_slug": "pupular", "days": 3},
    )
    client.post(
        "/api/plugins/marketing_factory/campaigns/generate",
        json={"app_slug": "setvenue", "days": 2},
    )

    # Bulk-approve only Pupular's queue; SetVenue should remain pending.
    bulk = client.post("/api/plugins/marketing_factory/drafts/approve-all?app_slug=pupular")
    assert bulk.status_code == 200
    approved = bulk.json()["result"]
    assert len(approved) == 3
    assert all(record["status"] == "approved" for record in approved)
    counts = bulk.json()["overview"]["draft_status_counts"]
    assert counts.get("approved") == 3
    assert counts.get("needs_review") == 2  # SetVenue still pending

    # Bulk-reject SetVenue's queue.
    bulk_reject = client.post("/api/plugins/marketing_factory/drafts/reject-all?app_slug=setvenue")
    assert bulk_reject.status_code == 200
    rejected = bulk_reject.json()["result"]
    assert len(rejected) == 2
    final_counts = bulk_reject.json()["overview"]["draft_status_counts"]
    assert final_counts.get("rejected") == 2
    assert final_counts.get("needs_review", 0) == 0
