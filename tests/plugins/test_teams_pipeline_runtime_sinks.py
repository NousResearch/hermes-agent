"""Exercise runtime-created sinks with local HTTP fixtures and a durable store."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from plugins.teams_pipeline import pipeline as pipeline_module
from plugins.teams_pipeline import runtime
from plugins.teams_pipeline.models import MeetingArtifact, TeamsMeetingRef, TeamsMeetingSummaryPayload
from plugins.teams_pipeline.store import TeamsPipelineStore


@pytest.fixture
def runtime_factory(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "build_graph_client", lambda: object())
    monkeypatch.setattr(runtime, "resolve_teams_pipeline_store_path", lambda: tmp_path / "jobs.json")

    async def resolve(*args, **kwargs):
        return TeamsMeetingRef(meeting_id="meeting-1")

    async def transcript(*args, **kwargs):
        return MeetingArtifact(artifact_type="transcript", artifact_id="tx"), "A useful transcript. " * 10

    async def call_record(*args, **kwargs):
        return None

    async def summary(self, *, resolved_meeting, transcript_text, artifacts):
        return TeamsMeetingSummaryPayload(meeting_ref=resolved_meeting, summary="Useful summary")

    monkeypatch.setattr(pipeline_module, "resolve_meeting_reference", resolve)
    monkeypatch.setattr(pipeline_module, "fetch_preferred_transcript_text", transcript)
    monkeypatch.setattr(pipeline_module, "enrich_meeting_with_call_record", call_record)
    monkeypatch.setattr(pipeline_module.TeamsMeetingPipeline, "_generate_summary_payload", summary)

    def build(config):
        gateway = SimpleNamespace(config=GatewayConfig(platforms={
            Platform("teams"): PlatformConfig(enabled=False, extra={"meeting_pipeline": config})
        }))
        return runtime.build_pipeline_runtime(gateway)

    return build


def run(pipeline):
    return asyncio.run(pipeline.run_notification({
        "id": "notification-1", "changeType": "updated",
        "resource": "communications/onlineMeetings/meeting-1",
        "resourceData": {"id": "meeting-1"},
    }))


@pytest.mark.parametrize("sink,key,target,response,record_key,record_value", [
    ("notion", "NOTION_API_KEY", {"database_id": "db"}, {"id": "page"}, "page_id", "page"),
    ("linear", "LINEAR_API_KEY", {"team_id": "team"},
     {"data": {"issueCreate": {"issue": {"id": "issue"}}}}, "issue_id", "issue"),
])
def test_runtime_enabled_sink_writes_and_reuses_durable_record(
    runtime_factory, monkeypatch, sink, key, target, response, record_key, record_value
):
    monkeypatch.setenv(key, "fixture-key")
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=response)

    transport = httpx.MockTransport(respond)
    original = pipeline_module._HttpSinkWriter.__init__

    def init(self, **kwargs):
        original(self, transport=transport, **kwargs)

    monkeypatch.setattr(pipeline_module._HttpSinkWriter, "__init__", init)
    pipeline = runtime_factory({sink: {"enabled": True, **target}})
    job = run(pipeline)
    assert job.status == "completed", job.error_info
    assert len(requests) == 1, "enabled runtime sink must actually write, not silently complete"
    record = TeamsPipelineStore(pipeline.store.path).get_sink_record(f"{sink}:meeting-1")
    assert record[record_key] == record_value
    replay = asyncio.run(pipeline.run_job(job.job_id))
    assert replay.status == "completed", replay.error_info
    assert len(requests) == 2
    if sink == "notion":
        assert requests[1].method == "PATCH"
        assert requests[1].url.path.endswith("/pages/page")
    else:
        assert b"issueUpdate" in requests[1].content


@pytest.mark.parametrize("sink,key,target", [
    ("notion", "NOTION_API_KEY", {"database_id": "db"}),
    ("linear", "LINEAR_API_KEY", {"team_id": "team"}),
])
def test_runtime_missing_sink_credential_fails_job(runtime_factory, monkeypatch, sink, key, target):
    monkeypatch.delenv(key, raising=False)
    pipeline = runtime_factory({sink: {"enabled": True, **target}})
    job = run(pipeline)
    assert job.status == "failed"
    assert key in job.error_info["message"]
    assert TeamsPipelineStore(pipeline.store.path).get_job(job.job_id)["status"] == "failed"


@pytest.mark.parametrize("config", [{}, {"notion": {"enabled": False}, "linear": {"enabled": False}}])
def test_disabled_sinks_do_not_construct_writers(runtime_factory, monkeypatch, config):
    def unexpected(*args, **kwargs):
        pytest.fail("disabled sinks must not load credentials or construct a writer")

    monkeypatch.setattr(pipeline_module._HttpSinkWriter, "__init__", unexpected)
    pipeline = runtime_factory(config)
    assert run(pipeline).status == "completed"


@pytest.mark.parametrize("sink,key", [("notion", "NOTION_API_KEY"), ("linear", "LINEAR_API_KEY")])
def test_unscoped_multiplex_startup_uses_runtime_home(runtime_factory, monkeypatch, tmp_path, sink, key):
    from agent import secret_scope
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / ".env").write_text(f"{key}=owner-key\n", encoding="utf-8")
    monkeypatch.setenv(key, "wrong-process-key")
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    token = secret_scope.set_secret_scope(None)
    try:
        pipeline = runtime_factory({sink: {"enabled": True}})
        assert getattr(pipeline, f"{sink}_writer").api_key == "owner-key"
        assert secret_scope.current_secret_scope() is None
    finally:
        secret_scope.reset_secret_scope(token)


@pytest.mark.parametrize("sink,key", [("notion", "NOTION_API_KEY"), ("linear", "LINEAR_API_KEY")])
def test_multiplex_runtime_preserves_scope_and_never_borrows_process_key(runtime_factory, monkeypatch, sink, key):
    from agent import secret_scope

    monkeypatch.setenv(key, "wrong-process-key")
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    first_scope = {key: "profile-a-key"}
    token = secret_scope.set_secret_scope(first_scope)
    try:
        first = runtime_factory({sink: {"enabled": True}})
        assert secret_scope.current_secret_scope() is first_scope
        second_token = secret_scope.set_secret_scope({})
        try:
            second = runtime_factory({sink: {"enabled": True}})
            assert getattr(second, f"{sink}_writer").api_key == ""
            assert getattr(first, f"{sink}_writer").api_key == "profile-a-key"
            job = run(second)
            assert job.status == "failed"
            assert key in job.error_info["message"]
        finally:
            secret_scope.reset_secret_scope(second_token)
    finally:
        secret_scope.reset_secret_scope(token)
