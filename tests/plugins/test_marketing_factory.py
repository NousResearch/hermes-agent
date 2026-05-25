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


def test_tool_handler_covers_extended_agent_surface(isolate_home):
    """Phase 18: every new action surfaced on the agent tool must dispatch successfully
    so an agentic caller can drive the factory end-to-end without CLI/dashboard."""
    from plugins.marketing_factory.tools import handle_marketing_factory

    json.loads(handle_marketing_factory({"action": "init"}))

    # generate + read drafts
    gen = json.loads(handle_marketing_factory({"action": "generate", "app_slug": "pupular", "days": 2}))
    assert gen["success"] is True
    draft_id = gen["result"]["drafts"][0]["id"]

    # advise
    advise = json.loads(handle_marketing_factory({"action": "advise"}))
    assert advise["success"] is True

    # regenerate one draft
    regen = json.loads(handle_marketing_factory({"action": "regenerate", "draft_id": draft_id}))
    assert regen["success"] is True
    new_draft_id = regen["result"]["new_draft"]["id"]

    # variants (count=2)
    variants = json.loads(handle_marketing_factory({"action": "variants", "draft_id": new_draft_id, "count": 2}))
    assert variants["success"] is True
    assert variants["result"]["count_generated"] == 2

    # edit body
    edit = json.loads(handle_marketing_factory({"action": "edit", "draft_id": new_draft_id, "body": "Edited via agent surface. https://pupular.example"}))
    assert edit["success"] is True

    # reschedule
    resched = json.loads(handle_marketing_factory({"action": "reschedule", "draft_id": new_draft_id, "scheduled_for": "2026-06-15T18:00:00+00:00"}))
    assert resched["success"] is True

    # set_channel_mode + set_auto_generate
    mode = json.loads(handle_marketing_factory({"action": "set_channel_mode", "app_slug": "pupular", "channel": "x", "mode": "live"}))
    assert mode["success"] is True
    auto = json.loads(handle_marketing_factory({"action": "set_auto_generate", "app_slug": "pupular", "enabled": True, "threshold": 5}))
    assert auto["success"] is True

    # add_app
    add = json.loads(handle_marketing_factory({"action": "add_app", "profile": {"slug": "agent-test-brand", "name": "Agent Test Brand", "channels": ["x"], "positioning": "test brand", "icp": "tests", "tone": "neutral", "cta": "Test CTA"}}))
    assert add["success"] is True
    assert add["result"]["slug"] == "agent-test-brand"

    # remove_app
    rem = json.loads(handle_marketing_factory({"action": "remove_app", "app_slug": "agent-test-brand"}))
    assert rem["success"] is True

    # poll + digest
    poll = json.loads(handle_marketing_factory({"action": "poll"}))
    assert poll["success"] is True
    digest = json.loads(handle_marketing_factory({"action": "digest", "app_slug": "pupular", "days": 7}))
    assert digest["success"] is True
    assert "# Pupular" in digest["result"]["markdown"]


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


def test_channel_modes_seed_dry_run_for_every_channel(isolate_home):
    """Phase 4: every brand profile starts with channel_modes={ch: dry_run for ch in channels}."""
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    MarketingFactoryPipeline(store).initialize_samples()
    state = store.load()
    for slug in ("pupular", "setvenue"):
        app = state["apps"][slug]
        assert app.get("channel_modes"), f"{slug} missing channel_modes"
        assert all(mode == "dry_run" for mode in app["channel_modes"].values()), f"{slug} has non-dry_run default"
        assert set(app["channel_modes"].keys()) == set(app["channels"]), f"{slug} mode keys don't match channels"


def test_set_channel_mode_audits_and_rejects_unknown_channels(isolate_home):
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    MarketingFactoryPipeline(store).initialize_samples()

    result = store.set_channel_mode("pupular", "x", "live", reviewer="tester")
    assert result == {"app_slug": "pupular", "channel": "x", "previous": "dry_run", "mode": "live"}
    assert store.require_app("pupular")["channel_modes"]["x"] == "live"

    actions = [event["action"] for event in store.list_audit(app_slug="pupular", limit=100)]
    assert "channel_mode.changed" in actions

    with pytest.raises(ValueError):
        store.set_channel_mode("pupular", "linkedin", "live")  # not in pupular's channels
    with pytest.raises(ValueError):
        store.set_channel_mode("pupular", "x", "bogus")  # invalid mode


def test_publish_scheduled_falls_back_to_dry_run_when_no_live_connector(isolate_home):
    """Phase 4 safety contract: channel_mode=live without a registered connector
    must publish as dry_run and emit a `publish.dry_run.fallback` audit event
    so the dashboard can flag it."""
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()
    generated = pipe.generate_campaign("pupular", days=1)
    draft = generated["drafts"][0]
    store.set_approval(draft["id"], "approved", reviewer="tester")
    schedule = pipe.scheduler.schedule_approved(store, app_slug="pupular")
    assert schedule, "scheduler should produce at least one schedule"

    # Flip the channel to live — but no real connector is registered, so the
    # publisher must fall back gracefully instead of raising or posting.
    store.set_channel_mode("pupular", draft["channel"], "live", reviewer="tester")

    events = pipe.publisher.publish_scheduled(store, app_slug="pupular")
    assert len(events) == 1
    event = events[0]
    assert event["mode"] == "dry_run"
    assert event["posted"] is False
    assert event["would_post"] is True
    assert event["fallback_reason"] == "no_live_connector"

    final_state = store.load()
    final_draft = final_state["drafts"][draft["id"]]
    assert final_draft["status"] == "dry_run_posted"  # NOT "posted"

    audit_actions = [e["action"] for e in store.list_audit(app_slug="pupular", limit=200)]
    assert "publish.dry_run.fallback" in audit_actions


def test_publish_scheduled_uses_registered_live_connector_when_present(isolate_home):
    """Phase 4: when a real connector IS registered, publish_scheduled goes live."""
    from typing import Any, Dict
    from plugins.marketing_factory import connectors
    from plugins.marketing_factory.connectors.base import BaseChannelConnector
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    class StubLiveX(BaseChannelConnector):
        mode = "live"
        channel = "x"

        def publish(self, draft: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "mode": "live",
                "would_post": False,
                "posted": True,
                "channel": "x",
                "body": draft.get("body"),
                "payload": {"fake_tweet_id": "12345"},
            }

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()
    generated = pipe.generate_campaign("pupular", days=1)
    draft = generated["drafts"][0]
    # Force the draft channel to X so our stub connector matches.
    state = store.load()
    state["drafts"][draft["id"]]["channel"] = "x"
    store._write_state(state)
    store.set_approval(draft["id"], "approved", reviewer="tester")
    pipe.scheduler.schedule_approved(store, app_slug="pupular")
    store.set_channel_mode("pupular", "x", "live", reviewer="tester")

    # Register the stub for the duration of this test.
    previous = connectors._REGISTRY.pop("x", None)
    connectors.register("x", StubLiveX())
    try:
        events = pipe.publisher.publish_scheduled(store, app_slug="pupular")
    finally:
        connectors._REGISTRY.pop("x", None)
        if previous is not None:
            connectors._REGISTRY["x"] = previous

    assert len(events) == 1
    event = events[0]
    assert event["mode"] == "live"
    assert event["posted"] is True
    assert event["payload"].get("fake_tweet_id") == "12345"
    assert event["fallback_reason"] is None

    audit_actions = [e["action"] for e in store.list_audit(app_slug="pupular", limit=200)]
    assert "publish.live" in audit_actions

    final_draft = store.load()["drafts"][draft["id"]]
    assert final_draft["status"] == "posted"


def test_image_gen_attaches_pollinations_url_to_visual_drafts(isolate_home, monkeypatch):
    """Phase 10: instagram/tiktok/app_store drafts get a Pollinations URL on draft.images."""
    monkeypatch.setenv("MF_AUTO_IMAGES", "1")
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()
    result = pipe.generate_campaign("pupular", days=3)

    visual = [d for d in result["drafts"] if d["channel"] in {"instagram", "tiktok", "app_store"}]
    assert visual, "Pupular's 3-day campaign should include a visual-channel draft"
    for draft in visual:
        images = draft.get("images") or []
        assert images, f"Expected images on {draft['channel']} draft"
        first = images[0]
        assert first["kind"] == "image_prompt"
        assert first["url"].startswith("https://image.pollinations.ai/prompt/")
        assert first["backend"] == "pollinations"


def test_image_gen_disabled_via_env_skips_image_attachment(isolate_home, monkeypatch):
    monkeypatch.setenv("MF_AUTO_IMAGES", "0")
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()
    result = pipe.generate_campaign("pupular", days=2)
    for draft in result["drafts"]:
        assert not (draft.get("images") or []), "MF_AUTO_IMAGES=0 should suppress image generation"


def test_image_gen_skips_non_visual_channels(isolate_home, monkeypatch):
    """Phase 10: blog/email/linkedin/x drafts must NOT get auto-generated images."""
    monkeypatch.setenv("MF_AUTO_IMAGES", "1")
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    # SetVenue's channels are linkedin/x/blog/email — none are visual.
    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()
    result = pipe.generate_campaign("setvenue", days=4)
    for draft in result["drafts"]:
        assert not (draft.get("images") or []), f"{draft['channel']} should not get auto images"


def test_image_gen_url_uses_default_template_when_llm_disabled(isolate_home, monkeypatch):
    """The pollinations URL must reflect the brand-templated fallback prompt when LLM is off."""
    monkeypatch.setenv("MF_AUTO_IMAGES", "1")
    from plugins.marketing_factory.pipeline import ImageGenAgent

    agent = ImageGenAgent()
    result = agent.generate(
        app={"slug": "pupular", "name": "Pupular", "positioning": "adopt-a-pet"},
        item={"channel": "instagram", "pillar": "shelter support"},
        body="cute caption body for testing",
    )
    assert result["url"]
    assert "pollinations" in result["url"]
    # Brand-templated fallback should mention pupular's framing
    assert "adoptable" in result["prompt"].lower() or "pet" in result["prompt"].lower()


def test_weekly_digest_renders_markdown_with_expected_sections(isolate_home):
    """Phase 9: digest produces markdown with activity counts, rejected reasons, steering."""
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()
    gen = pipe.generate_campaign("pupular", days=2)
    drafts = gen["drafts"]
    store.set_approval(drafts[0]["id"], "approved", reviewer="tester", reason="crisp opener and CTA")
    store.set_approval(drafts[1]["id"], "rejected", reviewer="tester", reason="too generic, no specific pet detail")

    md = pipe.weekly_digest("pupular", days=7)
    assert "# Pupular — weekly digest" in md
    assert "## Activity" in md
    assert "Campaigns generated" in md
    assert "Approved" in md
    assert "Rejected" in md
    assert "too generic, no specific pet detail" in md
    # Steering may or may not exist via the template path; both states are valid renderings
    assert "Brand steering" in md or "Current brand steering" in md


def test_weekly_digest_via_api_endpoint(isolate_home):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from plugins.marketing_factory.dashboard.plugin_api import router

    app = FastAPI()
    app.include_router(router, prefix="/api/plugins/marketing_factory")
    client = TestClient(app)
    client.post("/api/plugins/marketing_factory/init")
    response = client.get("/api/plugins/marketing_factory/apps/pupular/digest?days=14")
    assert response.status_code == 200
    payload = response.json()
    assert payload["app_slug"] == "pupular"
    assert payload["days"] == 14
    assert payload["markdown"].startswith("# Pupular")
    # 404 on unknown slug
    response_404 = client.get("/api/plugins/marketing_factory/apps/bogus/digest")
    assert response_404.status_code == 404


def test_progress_bus_publish_to_subscribers(isolate_home):
    """Phase 18: in-loop subscriber receives published events."""
    import asyncio
    from plugins.marketing_factory import progress_bus

    progress_bus.clear()

    async def _run():
        q = progress_bus.subscribe()
        progress_bus.publish("agent.start", agent="research", detail="hello")
        progress_bus.publish("agent.end", agent="research", detail="bye")
        e1 = await asyncio.wait_for(q.get(), timeout=1.0)
        e2 = await asyncio.wait_for(q.get(), timeout=1.0)
        progress_bus.unsubscribe(q)
        return e1, e2

    e1, e2 = asyncio.run(_run())
    assert e1["type"] == "agent.start"
    assert e1["agent"] == "research"
    assert e1["detail"] == "hello"
    assert e2["type"] == "agent.end"
    assert e2["seq"] > e1["seq"]


def test_progress_bus_cross_thread_delivery(isolate_home):
    """Phase 18: pipeline runs in a worker thread; events must still hit the main-loop subscriber."""
    import asyncio
    import threading
    from plugins.marketing_factory import progress_bus

    progress_bus.clear()

    async def _run():
        q = progress_bus.subscribe()

        def _worker():
            progress_bus.publish("agent.start", agent="copy", detail="from worker")

        t = threading.Thread(target=_worker)
        t.start()
        t.join()
        event = await asyncio.wait_for(q.get(), timeout=2.0)
        progress_bus.unsubscribe(q)
        return event

    event = asyncio.run(_run())
    assert event["agent"] == "copy"
    assert event["detail"] == "from worker"


def test_progress_bus_recent_returns_buffer(isolate_home):
    from plugins.marketing_factory import progress_bus

    progress_bus.clear()
    for i in range(5):
        progress_bus.publish("agent.start", agent="research", detail=f"step {i}")
    last3 = progress_bus.recent(limit=3)
    assert len(last3) == 3
    assert [e["detail"] for e in last3] == ["step 2", "step 3", "step 4"]


def test_generate_campaign_publishes_expected_event_sequence(isolate_home):
    """Phase 18: a campaign run must emit campaign.start, agent.{start,end} for each agent boundary, and campaign.end."""
    from plugins.marketing_factory import progress_bus
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    progress_bus.clear()
    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()
    pipe.generate_campaign("pupular", days=2)

    events = progress_bus.recent(limit=200)
    types = [e["type"] for e in events]
    agents_seen = {e.get("agent") for e in events if e["type"] in {"agent.start", "agent.end"}}

    assert "campaign.start" in types
    assert "campaign.end" in types
    # All agent boundaries we instrumented should appear at least once
    assert {"brand_memory", "research", "strategy", "copy", "safety"}.issubset(agents_seen)
    # The order: campaign.start before any agent.start, campaign.end last
    first_campaign_idx = types.index("campaign.start")
    last_campaign_idx = len(types) - 1 - list(reversed(types)).index("campaign.end")
    assert first_campaign_idx < last_campaign_idx


def test_app_analytics_returns_expected_shape_on_fresh_init(isolate_home):
    """Phase 17: fresh init has no drafts → analytics returns zero counts and null rates."""
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()
    result = pipe.app_analytics("pupular", days=30)
    assert result["app_slug"] == "pupular"
    assert result["total_drafts"] == 0
    assert result["approval_rate"] is None
    assert result["avg_freshness"] is None
    assert set(result["by_channel"].keys()) == set(store.require_app("pupular")["channels"])
    assert result["auto_generate"] is False


def test_app_analytics_computes_approval_rate_correctly(isolate_home):
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()
    gen = pipe.generate_campaign("pupular", days=4)
    # Approve 3, reject 1, leave 0 in needs_review (pupular generates 4 drafts here)
    for i, draft in enumerate(gen["drafts"][:4]):
        if i < 3:
            store.set_approval(draft["id"], "approved", reviewer="tester", reason=f"good {i}")
        else:
            store.set_approval(draft["id"], "rejected", reviewer="tester", reason="bad")

    result = pipe.app_analytics("pupular", days=30)
    assert result["approval_rate"] == 0.75
    assert result["by_status"]["approved"] == 3
    assert result["by_status"]["rejected"] == 1


def test_app_analytics_via_api_endpoint(isolate_home):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from plugins.marketing_factory.dashboard.plugin_api import router

    app = FastAPI()
    app.include_router(router, prefix="/api/plugins/marketing_factory")
    client = TestClient(app)
    client.post("/api/plugins/marketing_factory/init")
    response = client.get("/api/plugins/marketing_factory/apps/pupular/analytics?days=14")
    assert response.status_code == 200
    payload = response.json()
    assert payload["app_slug"] == "pupular"
    assert payload["period_days"] == 14
    assert "by_channel" in payload
    # 404 on unknown
    bad = client.get("/api/plugins/marketing_factory/apps/bogus/analytics")
    assert bad.status_code == 404


def test_draft_checklist_x_channel_signals(isolate_home):
    """Phase 16: per-channel checklist computes correct passed/failed for X drafts."""
    from plugins.marketing_factory.pipeline import draft_checklist

    app = {
        "slug": "pupular", "name": "Pupular",
        "channels": ["x"], "links": ["https://apps.apple.com/us/app/pupular"],
        "tone": "cute warm playful",
    }
    short_x_with_link = {
        "channel": "x",
        "body": "find your new best friend 🐾 download pupular https://apps.apple.com/us/app/pupular",
    }
    items = draft_checklist(short_x_with_link, app)
    labels_passed = {item["label"]: item["passed"] for item in items}
    assert labels_passed["≤280 chars"] is True
    assert labels_passed["has link"] is True
    assert labels_passed["hashtags ≤2"] is True
    assert labels_passed["has emoji"] is True

    no_link_no_emoji = {"channel": "x", "body": "Pupular helps you discover adoptable pets near you. #adoption #pets #shelter #rescue"}
    items = draft_checklist(no_link_no_emoji, app)
    labels_passed = {item["label"]: item["passed"] for item in items}
    assert labels_passed["has link"] is False
    assert labels_passed["hashtags ≤2"] is False
    assert labels_passed["has emoji"] is False


def test_draft_checklist_instagram_requires_image(isolate_home):
    from plugins.marketing_factory.pipeline import draft_checklist

    app = {"slug": "pupular", "channels": ["instagram"], "links": [], "tone": ""}
    no_image = {"channel": "instagram", "body": "x" * 500, "images": []}
    items = draft_checklist(no_image, app)
    has_image_check = next(i for i in items if i["label"] == "has image")
    assert has_image_check["passed"] is False
    assert has_image_check["severity"] == "error"

    with_image = {"channel": "instagram", "body": "x" * 500, "images": [{"kind": "image_prompt", "url": "https://example.com/x.png"}]}
    items = draft_checklist(with_image, app)
    has_image_check = next(i for i in items if i["label"] == "has image")
    assert has_image_check["passed"] is True


def test_draft_checklist_email_subject_line(isolate_home):
    from plugins.marketing_factory.pipeline import draft_checklist

    app = {"slug": "setvenue", "channels": ["email"], "links": [], "tone": ""}
    bad = {"channel": "email", "body": "Hey there\n\nWelcome to SetVenue."}
    items = draft_checklist(bad, app)
    subject_check = next(i for i in items if i["label"] == "has Subject:")
    assert subject_check["passed"] is False

    good = {"channel": "email", "body": "Subject: Welcome to SetVenue\n\nHey there..."}
    items = draft_checklist(good, app)
    subject_check = next(i for i in items if i["label"] == "has Subject:")
    assert subject_check["passed"] is True


def test_auto_generate_default_off_means_poll_does_not_generate(isolate_home):
    """Phase 15: by default, poll() never auto-generates — opt-in only to avoid runaway spend."""
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()
    # No campaigns yet, both apps default to auto_generate=False
    result = pipe.poll()
    assert result["auto_generated_apps"] == []


def test_auto_generate_on_with_empty_queue_fires_campaign(isolate_home):
    from datetime import datetime, timedelta, timezone
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()

    # Flip pupular's auto_generate ON with threshold=3
    store.set_auto_generate("pupular", True, threshold=3, reviewer="tester")

    # No existing campaigns or drafts — queue is empty (0 < 3), should fire
    result = pipe.poll()
    auto = result["auto_generated_apps"]
    assert any(a["app_slug"] == "pupular" for a in auto), "Expected pupular to be auto-generated"
    record = next(a for a in auto if a["app_slug"] == "pupular")
    assert record["pending_before"] == 0
    assert record["threshold"] == 3
    assert record["draft_count"] > 0


def test_auto_generate_skips_when_queue_full(isolate_home):
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()
    # Generate 7 drafts (fills the queue past threshold=3)
    pipe.generate_campaign("pupular", days=7)
    store.set_auto_generate("pupular", True, threshold=3, reviewer="tester")

    result = pipe.poll()
    assert all(a["app_slug"] != "pupular" for a in result["auto_generated_apps"])


def test_auto_generate_cooldown_blocks_second_poll(isolate_home):
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()
    store.set_auto_generate("pupular", True, threshold=3, reviewer="tester")

    first = pipe.poll()
    assert any(a["app_slug"] == "pupular" for a in first["auto_generated_apps"])

    # Reject all the drafts so the queue empties again — cooldown should still block
    for draft in store.list_drafts(app_slug="pupular"):
        if draft["status"] == "needs_review":
            store.set_approval(draft["id"], "rejected", reviewer="tester", reason="cooldown test")

    second = pipe.poll()
    assert all(a["app_slug"] != "pupular" for a in second["auto_generated_apps"]), "Cooldown should block second auto-generation"


def test_reschedule_updates_draft_and_schedule_record(isolate_home):
    """Phase 14: reschedule on an approved draft updates both draft.scheduled_for AND the schedules record."""
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()
    gen = pipe.generate_campaign("pupular", days=1)
    draft = gen["drafts"][0]
    store.set_approval(draft["id"], "approved", reviewer="tester")
    pipe.scheduler.schedule_approved(store, app_slug="pupular")

    new_time = "2026-06-15T18:00:00+00:00"
    result = store.update_draft_scheduled_for(draft["id"], new_time)
    assert result["scheduled_for"] == new_time

    state = store.load()
    assert state["schedules"][draft["id"]]["scheduled_for"] == new_time

    actions = [event["action"] for event in store.list_audit(app_slug="pupular", limit=200)]
    assert "draft.rescheduled" in actions


def test_reschedule_on_needs_review_updates_advisory_only(isolate_home):
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()
    gen = pipe.generate_campaign("pupular", days=1)
    draft = gen["drafts"][0]
    assert draft["status"] == "needs_review"

    new_time = "2026-07-01T15:00:00+00:00"
    result = store.update_draft_scheduled_for(draft["id"], new_time)
    assert result["scheduled_for"] == new_time
    # No schedule record yet — that only gets created on schedule_approved
    state = store.load()
    assert state["schedules"].get(draft["id"]) is None


def test_reschedule_rejects_invalid_iso(isolate_home):
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()
    gen = pipe.generate_campaign("pupular", days=1)
    with pytest.raises(ValueError):
        store.update_draft_scheduled_for(gen["drafts"][0]["id"], "not-an-iso-string")


def test_reschedule_refuses_already_published(isolate_home):
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()
    gen = pipe.generate_campaign("pupular", days=1)
    draft = gen["drafts"][0]
    store.set_approval(draft["id"], "approved", reviewer="tester")
    pipe.scheduler.schedule_approved(store, app_slug="pupular")
    pipe.publisher.dry_run_publish_scheduled(store, app_slug="pupular")
    with pytest.raises(ValueError):
        store.update_draft_scheduled_for(draft["id"], "2026-06-15T18:00:00+00:00")


def test_freshness_jaccard_identical_scores_zero(isolate_home):
    """Phase 13: identical bodies must score 0 freshness (= 0% novel)."""
    from plugins.marketing_factory.pipeline import _compute_freshness
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    store.initialize()
    # Seed pupular app + a draft directly
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    MarketingFactoryPipeline(store).initialize_samples()
    state = store.load()
    seed_body = "Tiny paws huge main-character energy. Adoptable pets are out there waiting."
    state["drafts"]["draft_seed1"] = {
        "id": "draft_seed1", "app_slug": "pupular", "channel": "x", "body": seed_body,
        "campaign_id": "test", "content_type": "short_social", "status": "dry_run_posted",
        "created_at": "2026-05-20T00:00:00+00:00",
    }
    store._write_state(state)

    result = _compute_freshness(store, "pupular", "x", seed_body)
    assert result["score"] == 0.0
    assert result["most_similar_id"] == "draft_seed1"
    assert result["compared_against"] == 1


def test_freshness_jaccard_totally_different_scores_high(isolate_home):
    from plugins.marketing_factory.pipeline import _compute_freshness
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    MarketingFactoryPipeline(store).initialize_samples()
    state = store.load()
    state["drafts"]["draft_seed1"] = {
        "id": "draft_seed1", "app_slug": "pupular", "channel": "x",
        "body": "Tiny paws huge main-character energy. Adoptable pets are out there waiting.",
        "campaign_id": "test", "content_type": "short_social", "status": "dry_run_posted",
        "created_at": "2026-05-20T00:00:00+00:00",
    }
    store._write_state(state)

    completely_different = "Premium architectural venues for production shoots — book unique spaces with clear approval timelines."
    result = _compute_freshness(store, "pupular", "x", completely_different)
    assert result["score"] > 0.7  # mostly novel


def test_freshness_scoring_is_per_channel_scoped(isolate_home):
    """Identical body on a DIFFERENT channel must not lower freshness."""
    from plugins.marketing_factory.pipeline import _compute_freshness
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    MarketingFactoryPipeline(store).initialize_samples()
    state = store.load()
    seed_body = "Tiny paws huge main-character energy. Adoptable pets are out there waiting."
    state["drafts"]["draft_ig"] = {
        "id": "draft_ig", "app_slug": "pupular", "channel": "instagram", "body": seed_body,
        "campaign_id": "test", "content_type": "visual_caption", "status": "dry_run_posted",
        "created_at": "2026-05-20T00:00:00+00:00",
    }
    store._write_state(state)

    # Same body checked against channel=x — no x drafts exist, so freshness should be 1.0
    result = _compute_freshness(store, "pupular", "x", seed_body)
    assert result["score"] == 1.0
    assert result["compared_against"] == 0


def test_generate_campaign_attaches_freshness_to_new_drafts(isolate_home):
    """Phase 13: every draft from generate_campaign carries a freshness_score field."""
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()
    result = pipe.generate_campaign("pupular", days=2)
    for draft in result["drafts"]:
        assert "freshness_score" in draft
        assert 0.0 <= draft["freshness_score"] <= 1.0


def test_edit_draft_updates_body_and_reruns_safety(isolate_home):
    """Phase 12: editing a draft updates body, sets edited_by/at, re-runs safety, audits."""
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()
    gen = pipe.generate_campaign("pupular", days=1)
    draft = gen["drafts"][0]
    original_body = draft["body"]
    new_body = "Edited body for pupular — meet adoptable pets near you and download Pupular today. https://apps.apple.com/us/app/pupular"

    edited = pipe.edit_draft(draft["id"], new_body, editor="tester")
    assert edited["body"] == new_body
    assert edited["edited_by"] == "tester"
    assert edited["edited_at"]
    assert (edited.get("safety") or {}).get("passed") is True

    audit_actions = [event["action"] for event in store.list_audit(app_slug="pupular", limit=200)]
    assert "draft.edited" in audit_actions

    # Re-reading from store reflects the edit
    persisted = store.get_draft(draft["id"])
    assert persisted["body"] == new_body
    assert persisted["body"] != original_body


def test_edit_draft_rejects_empty_body(isolate_home):
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()
    gen = pipe.generate_campaign("pupular", days=1)
    with pytest.raises(ValueError):
        pipe.edit_draft(gen["drafts"][0]["id"], "   ", editor="tester")


def test_edit_draft_refuses_when_already_published(isolate_home):
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()
    gen = pipe.generate_campaign("pupular", days=1)
    draft = gen["drafts"][0]
    store.set_approval(draft["id"], "approved", reviewer="tester")
    pipe.scheduler.schedule_approved(store, app_slug="pupular")
    pipe.publisher.dry_run_publish_scheduled(store, app_slug="pupular")
    # Draft is now in dry_run_posted — editing should be refused
    with pytest.raises(ValueError):
        pipe.edit_draft(draft["id"], "new body", editor="tester")


def test_edit_draft_via_api_endpoint(isolate_home):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from plugins.marketing_factory.dashboard.plugin_api import router

    app = FastAPI()
    app.include_router(router, prefix="/api/plugins/marketing_factory")
    client = TestClient(app)
    client.post("/api/plugins/marketing_factory/init")
    gen = client.post("/api/plugins/marketing_factory/campaigns/generate", json={"app_slug": "pupular", "days": 1}).json()
    draft_id = gen["result"]["drafts"][0]["id"]

    response = client.patch(f"/api/plugins/marketing_factory/drafts/{draft_id}", json={"body": "Tweaked body for pupular adoption.", "editor": "dashboard"})
    assert response.status_code == 200
    assert response.json()["result"]["body"] == "Tweaked body for pupular adoption."

    bad_response = client.patch(f"/api/plugins/marketing_factory/drafts/{draft_id}", json={"body": "", "editor": "dashboard"})
    assert bad_response.status_code == 400


def test_summary_includes_cost_estimate(isolate_home):
    """Phase 20: summary exposes USD cost estimate computed from spent_by_route."""
    from plugins.marketing_factory.store import MarketingFactoryStore, estimate_costs

    store = MarketingFactoryStore()
    store.initialize()
    state = store.load()
    state["budgets"]["spent_by_route"] = {"cheap": 1_000_000, "mid": 500_000, "premium": 250_000}
    store._write_state(state)

    summary = store.summary()
    assert "cost_estimate" in summary
    ce = summary["cost_estimate"]
    assert ce["by_route_usd"]["cheap"] == 0.0  # local = free
    assert ce["by_route_usd"]["mid"] == 0.0
    # Premium = (250_000 / 1_000_000) * 6.0 = $1.50
    assert abs(ce["by_route_usd"]["premium"] - 1.50) < 0.001
    assert abs(ce["total_usd"] - 1.50) < 0.001
    # estimate_costs helper directly
    direct = estimate_costs({"premium": 1_000_000})
    assert abs(direct["total_usd"] - 6.0) < 0.001


def test_cost_estimate_respects_env_override(isolate_home, monkeypatch):
    from plugins.marketing_factory.store import estimate_costs
    monkeypatch.setenv("MF_PRICE_USD_PER_M_PREMIUM", "15.0")
    result = estimate_costs({"premium": 1_000_000})
    assert abs(result["total_usd"] - 15.0) < 0.001


def test_resolve_variant_winner_auto_rejects_siblings(isolate_home):
    """Phase 19: approving one variant auto-rejects the others with a structured reason
    so the brand memory loop learns the COMPARISON, not just yes/no per draft."""
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()
    gen = pipe.generate_campaign("pupular", days=1)
    source = gen["drafts"][0]
    variants_result = pipe.generate_variants(source["id"], count=3)
    variant_ids = [v["id"] for v in variants_result["variants"]]
    assert len(variant_ids) == 3

    chosen = variant_ids[0]
    result = pipe.resolve_variant_winner(chosen, reviewer="tester", reason="best opener")
    assert result["approved"]["status"] == "approved"
    assert result["loser_count"] == 2

    # Other 2 variants must now be rejected
    state = store.load()
    for vid in variant_ids[1:]:
        assert state["drafts"][vid]["status"] == "rejected"
        reason = state["approvals"][vid]["reason"]
        assert chosen in reason  # rejection reason references the winner

    # The brand memory captured both the approval and the rejections
    learnings = state["brand_memories"]["pupular"]["learnings"]
    kinds = [e.get("kind") for e in learnings]
    assert kinds.count("draft_approved") >= 1
    assert kinds.count("draft_rejected") >= 2

    # Audit event recorded
    actions = [e["action"] for e in store.list_audit(app_slug="pupular", limit=200)]
    assert "variants.resolved" in actions


def test_resolve_variant_winner_with_no_siblings_just_approves(isolate_home):
    """When there are no siblings to reject, behaves identically to plain approve."""
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()
    gen = pipe.generate_campaign("pupular", days=1)
    draft_id = gen["drafts"][0]["id"]

    result = pipe.resolve_variant_winner(draft_id, reviewer="tester")
    assert result["approved"]["status"] == "approved"
    assert result["loser_count"] == 0
    assert result["auto_rejected"] == []


def test_generate_variants_produces_n_distinct_drafts(isolate_home):
    """Phase 11: generate_variants(draft_id, count=3) returns 3 distinct drafts all linked to the source."""
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()
    gen = pipe.generate_campaign("pupular", days=1)
    source = gen["drafts"][0]

    result = pipe.generate_variants(source["id"], count=3)
    assert result["count_generated"] == 3
    variant_ids = {v["id"] for v in result["variants"]}
    assert len(variant_ids) == 3
    for variant in result["variants"]:
        assert variant["regenerated_from"] == source["id"]
        assert variant["status"] == "needs_review"
        assert variant["channel"] == source["channel"]

    # audit captures the batch
    actions = [event["action"] for event in store.list_audit(app_slug="pupular", limit=200)]
    assert "variants.generated" in actions


def test_generate_variants_rejects_invalid_counts(isolate_home):
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()
    gen = pipe.generate_campaign("pupular", days=1)

    with pytest.raises(ValueError):
        pipe.generate_variants(gen["drafts"][0]["id"], count=0)
    with pytest.raises(ValueError):
        pipe.generate_variants(gen["drafts"][0]["id"], count=10)


def test_regenerate_draft_creates_new_draft_preserves_old(isolate_home):
    """Phase 8: regenerate produces a new draft with regenerated_from lineage, leaving the old draft intact."""
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()
    gen = pipe.generate_campaign("pupular", days=1)
    old = gen["drafts"][0]
    old_status = old["status"]

    result = pipe.regenerate_draft(old["id"])
    new_draft = result["new_draft"]

    assert new_draft["id"] != old["id"]
    assert new_draft["regenerated_from"] == old["id"]
    assert new_draft["channel"] == old["channel"]
    assert new_draft["app_slug"] == old["app_slug"]

    # Old draft preserved unchanged
    preserved = store.get_draft(old["id"])
    assert preserved["status"] == old_status

    # Audit trail records the regeneration
    actions = [event["action"] for event in store.list_audit(app_slug="pupular", limit=200)]
    assert "draft.regenerated" in actions


def test_regenerate_draft_via_api_endpoint(isolate_home):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from plugins.marketing_factory.dashboard.plugin_api import router

    app = FastAPI()
    app.include_router(router, prefix="/api/plugins/marketing_factory")
    client = TestClient(app)
    client.post("/api/plugins/marketing_factory/init")
    gen = client.post("/api/plugins/marketing_factory/campaigns/generate", json={"app_slug": "pupular", "days": 1}).json()
    draft_id = gen["result"]["drafts"][0]["id"]

    response = client.post(f"/api/plugins/marketing_factory/drafts/{draft_id}/regenerate")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["result"]["new_draft"]["regenerated_from"] == draft_id
    # Both drafts now visible in the queue (old preserved)
    overview_drafts = {d["id"] for d in payload["overview"]["drafts"]}
    assert draft_id in overview_drafts
    assert payload["result"]["new_draft"]["id"] in overview_drafts


def test_advise_returns_empty_for_fresh_factory(isolate_home):
    """Phase 7: fresh init triggers only `info` items for never-generated apps; healthy=False but no warnings."""
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()
    result = pipe.advise()
    # Both seeded apps will trigger the "never had a campaign" info item, but no warnings.
    warnings = [item for item in result["items"] if item["severity"] == "warning"]
    assert warnings == []
    assert {item["app_slug"] for item in result["items"]} == {"pupular", "setvenue"}


def test_advise_warns_on_live_channel_without_connector(isolate_home):
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()
    store.set_channel_mode("pupular", "x", "live", reviewer="tester")

    result = pipe.advise()
    live_warnings = [item for item in result["items"] if item["severity"] == "warning" and "live" in item["message"]]
    assert live_warnings, "Expected a warning about live channel without connector"
    assert any("x" in w["message"] for w in live_warnings)


def test_advise_warns_when_pending_queue_exceeds_threshold(isolate_home):
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()
    # Generate 15 days = 15 drafts pending review
    pipe.generate_campaign("pupular", days=15)

    result = pipe.advise()
    pending_warnings = [item for item in result["items"] if "pending" in item["message"].lower()]
    assert pending_warnings, "Expected a warning about backed-up pending queue"
    assert pending_warnings[0]["severity"] == "warning"
    assert pending_warnings[0]["app_slug"] == "pupular"


def test_advise_warns_when_poller_stale_and_schedules_exist(isolate_home):
    from datetime import datetime, timedelta, timezone
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()
    gen = pipe.generate_campaign("pupular", days=1)
    store.set_approval(gen["drafts"][0]["id"], "approved", reviewer="tester")
    pipe.scheduler.schedule_approved(store, app_slug="pupular")
    # Backdate last_poll_at by 3 hours
    state = store.load()
    state["poll"] = {"last_poll_at": (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat(), "last_poll_fired": 0, "last_poll_due": 0, "total_polls": 1}
    store._write_state(state)

    result = pipe.advise()
    poller_warnings = [item for item in result["items"] if "poller" in item["message"].lower() and item["severity"] == "warning"]
    assert poller_warnings, "Expected a warning about stale poller"


def test_advise_via_api_endpoint_returns_items(isolate_home):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from plugins.marketing_factory.dashboard.plugin_api import router

    app = FastAPI()
    app.include_router(router, prefix="/api/plugins/marketing_factory")
    client = TestClient(app)
    client.post("/api/plugins/marketing_factory/init")
    response = client.get("/api/plugins/marketing_factory/advise")
    assert response.status_code == 200
    payload = response.json()
    assert "items" in payload
    assert "healthy" in payload
    assert "checked_at" in payload


def test_add_app_via_api_creates_new_brand_profile(isolate_home):
    """Phase 6 / multi-tenant: POST /apps lets us add Wingman/Hardline/etc without editing Python."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from plugins.marketing_factory.dashboard.plugin_api import router

    app = FastAPI()
    app.include_router(router, prefix="/api/plugins/marketing_factory")
    client = TestClient(app)
    client.post("/api/plugins/marketing_factory/init")

    response = client.post("/api/plugins/marketing_factory/apps", json={
        "slug": "wingman",
        "name": "Wingman",
        "positioning": "iOS app for men replying in dating apps in their voice",
        "icp": "men 25-40 using Hinge / Bumble / SMS",
        "tone": "confident, witty, brief",
        "cta": "Download Wingman from the App Store",
        "channels": ["x", "tiktok"],
        "content_pillars": ["voice-matched replies", "dating wins"],
        "forbidden_claims": ["guaranteed dates"],
        "links": ["https://apps.apple.com/us/app/wingman/idplaceholder"],
        "claims": [], "competitors": [], "assets": [],
    })
    assert response.status_code == 200, response.text
    overview = response.json()["overview"]
    slugs = {app["slug"] for app in overview["apps"]}
    assert slugs == {"pupular", "setvenue", "wingman"}

    # New app gets channel_modes seeded to dry_run for every channel
    wingman = next(app for app in overview["apps"] if app["slug"] == "wingman")
    assert set(wingman["channel_modes"]) == {"x", "tiktok"}
    assert all(mode == "dry_run" for mode in wingman["channel_modes"].values())


def test_patch_app_partial_update_preserves_other_fields(isolate_home):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from plugins.marketing_factory.dashboard.plugin_api import router

    app = FastAPI()
    app.include_router(router, prefix="/api/plugins/marketing_factory")
    client = TestClient(app)
    client.post("/api/plugins/marketing_factory/init")

    response = client.patch("/api/plugins/marketing_factory/apps/pupular", json={
        "tone": "Even cuter, even warmer, slightly mischievous",
    })
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["tone"] == "Even cuter, even warmer, slightly mischievous"
    # Untouched fields still present
    assert result["icp"]
    assert result["channels"]
    assert result["channel_modes"]


def test_delete_app_cascades_drafts_campaigns_schedules(isolate_home):
    """Phase 6: DELETE /apps/{slug}?cascade=true wipes all dependent records."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from plugins.marketing_factory.dashboard.plugin_api import router

    app = FastAPI()
    app.include_router(router, prefix="/api/plugins/marketing_factory")
    client = TestClient(app)
    client.post("/api/plugins/marketing_factory/init")

    # Seed some pupular activity
    client.post("/api/plugins/marketing_factory/campaigns/generate", json={"app_slug": "pupular", "days": 2})
    client.post("/api/plugins/marketing_factory/drafts/approve-all?app_slug=pupular")
    overview_before = client.get("/api/plugins/marketing_factory/overview").json()
    drafts_before = sum(1 for d in overview_before["drafts"] if d["app_slug"] == "pupular")
    campaigns_before = sum(1 for c in overview_before["campaigns"] if c["app_slug"] == "pupular")
    assert drafts_before > 0
    assert campaigns_before > 0

    response = client.delete("/api/plugins/marketing_factory/apps/pupular?cascade=true")
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["app_slug"] == "pupular"
    assert result["removed"]["drafts"] == drafts_before
    assert result["removed"]["campaigns"] == campaigns_before

    overview_after = response.json()["overview"]
    assert all(a["slug"] != "pupular" for a in overview_after["apps"])
    assert all(d["app_slug"] != "pupular" for d in overview_after["drafts"])
    assert all(c["app_slug"] != "pupular" for c in overview_after["campaigns"])


def test_delete_app_no_cascade_refuses_when_dependents_exist(isolate_home):
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()
    pipe.generate_campaign("pupular", days=1)

    with pytest.raises(ValueError):
        store.remove_app("pupular", cascade=False)


def test_delete_unknown_app_returns_404(isolate_home):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from plugins.marketing_factory.dashboard.plugin_api import router

    app = FastAPI()
    app.include_router(router, prefix="/api/plugins/marketing_factory")
    client = TestClient(app)
    client.post("/api/plugins/marketing_factory/init")

    response = client.delete("/api/plugins/marketing_factory/apps/bogus-slug")
    assert response.status_code == 404


def test_poll_skips_drafts_scheduled_in_the_future(isolate_home):
    """Phase 5: poll(due_only=True) must NOT publish drafts whose scheduled_for is in the future."""
    from datetime import datetime, timedelta, timezone
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()

    generated = pipe.generate_campaign("pupular", days=1)
    draft = generated["drafts"][0]
    store.set_approval(draft["id"], "approved", reviewer="tester")
    pipe.scheduler.schedule_approved(store, app_slug="pupular")

    # Pin the schedule into the future relative to our fake `now`.
    future_dt = datetime.now(timezone.utc) + timedelta(hours=2)
    state = store.load()
    sched_record = next(iter(state["schedules"].values()))
    sched_record["scheduled_for"] = future_dt.isoformat()
    store._write_state(state)

    result = pipe.poll(now=datetime.now(timezone.utc))
    assert result["due_count"] == 0
    assert result["fired_count"] == 0
    assert len(result["events"]) == 0

    # last_poll_at gets updated even when nothing fires
    assert result["last_poll"]["last_poll_at"] is not None
    assert result["last_poll"]["total_polls"] == 1


def test_poll_fires_drafts_scheduled_in_the_past(isolate_home):
    """Phase 5: poll() fires publish on drafts whose scheduled_for has passed."""
    from datetime import datetime, timedelta, timezone
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()

    generated = pipe.generate_campaign("pupular", days=2)
    for draft in generated["drafts"]:
        store.set_approval(draft["id"], "approved", reviewer="tester")
    pipe.scheduler.schedule_approved(store, app_slug="pupular")

    # Pin the schedules into the past relative to `now`.
    past_dt = datetime.now(timezone.utc) - timedelta(hours=1)
    state = store.load()
    for sched in state["schedules"].values():
        sched["scheduled_for"] = past_dt.isoformat()
    store._write_state(state)

    result = pipe.poll(now=datetime.now(timezone.utc))
    assert result["fired_count"] == 2
    assert all(e["mode"] == "dry_run" for e in result["events"])

    # Idempotent: polling again fires nothing — drafts are already published.
    second = pipe.poll(now=datetime.now(timezone.utc))
    assert second["fired_count"] == 0
    assert second["last_poll"]["total_polls"] == 2


def test_poll_walks_every_app(isolate_home):
    """Phase 5: poll() iterates all apps, not just one."""
    from datetime import datetime, timedelta, timezone
    from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
    from plugins.marketing_factory.store import MarketingFactoryStore

    store = MarketingFactoryStore()
    pipe = MarketingFactoryPipeline(store)
    pipe.initialize_samples()

    past_dt = datetime.now(timezone.utc) - timedelta(hours=1)
    for slug in ("pupular", "setvenue"):
        gen = pipe.generate_campaign(slug, days=1)
        store.set_approval(gen["drafts"][0]["id"], "approved", reviewer="tester")
        pipe.scheduler.schedule_approved(store, app_slug=slug)

    state = store.load()
    for sched in state["schedules"].values():
        sched["scheduled_for"] = past_dt.isoformat()
    store._write_state(state)

    result = pipe.poll(now=datetime.now(timezone.utc))
    assert result["polled_apps"] == 2
    assert result["fired_count"] == 2
    app_slugs = {event["app_slug"] for event in result["events"]}
    assert app_slugs == {"pupular", "setvenue"}


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
