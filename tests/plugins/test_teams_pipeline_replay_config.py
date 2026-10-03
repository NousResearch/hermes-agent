"""Continuation contracts for config-aware CLI replay and missing writers."""
import json
from argparse import Namespace

import httpx
import pytest
import yaml

from hermes_constants import get_hermes_home
from plugins.teams_pipeline import cli, pipeline as pipeline_module, runtime
from plugins.teams_pipeline.store import TeamsPipelineStore
import asyncio
from types import SimpleNamespace
from gateway.config import GatewayConfig, Platform, PlatformConfig
from plugins.teams_pipeline.models import MeetingArtifact, TeamsMeetingRef, TeamsMeetingSummaryPayload


@pytest.fixture
def runtime_factory(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "build_graph_client", lambda: object())
    monkeypatch.setattr(runtime, "resolve_teams_pipeline_store_path", lambda: tmp_path / "default.json")

    async def resolve(*args, **kwargs):
        return TeamsMeetingRef(meeting_id="meeting-1")

    async def transcript(*args, **kwargs):
        return MeetingArtifact(artifact_type="transcript", artifact_id="tx"), "Useful transcript. " * 10

    async def call_record(*args, **kwargs):
        return None

    async def summary(self, **kwargs):
        return TeamsMeetingSummaryPayload(meeting_ref=kwargs["resolved_meeting"], summary="Summary")

    monkeypatch.setattr(pipeline_module, "resolve_meeting_reference", resolve)
    monkeypatch.setattr(pipeline_module, "fetch_preferred_transcript_text", transcript)
    monkeypatch.setattr(pipeline_module, "enrich_meeting_with_call_record", call_record)
    monkeypatch.setattr(pipeline_module.TeamsMeetingPipeline, "_generate_summary_payload", summary)

    def build(config):
        return runtime.build_pipeline_runtime(SimpleNamespace(config=GatewayConfig(platforms={
            Platform("teams"): PlatformConfig(enabled=False, extra={"meeting_pipeline": config})
        })))

    return build


def run(pipeline):
    return asyncio.run(pipeline.run_notification({
        "id": "notification-1", "changeType": "updated",
        "resource": "communications/onlineMeetings/meeting-1", "resourceData": {"id": "meeting-1"},
    }))


@pytest.mark.parametrize("sink,key,target,response", [
    ("notion", "NOTION_API_KEY", {"database_id": "cli-db"}, {"id": "cli-page"}),
    ("linear", "LINEAR_API_KEY", {"team_id": "cli-team"},
     {"data": {"issueCreate": {"issue": {"id": "cli-issue"}}}}),
])
def test_cli_replay_loads_config_and_preserves_selected_store(
    runtime_factory, monkeypatch, tmp_path, capsys, sink, key, target, response
):
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    path = home / "config.yaml"
    config = {"platforms": {"teams": {"enabled": False, "extra": {"meeting_pipeline": {
        "transcript_min_chars": 9000, "transcript_required": True,
        sink: {"enabled": True, **target},
    }}}}}
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv(key, "fixture-key")
    monkeypatch.setattr(cli, "build_graph_client", lambda: object())
    original = pipeline_module._HttpSinkWriter.__init__
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=response)

    def init(self, **kwargs):
        original(self, transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(pipeline_module._HttpSinkWriter, "__init__", init)
    source = runtime_factory({})
    selected_store = TeamsPipelineStore(tmp_path / "selected.json")
    source.store = selected_store
    job = run(source)
    args = Namespace(teams_pipeline_action="run", job_id=job.job_id, store_path=str(selected_store.path))
    assert cli.teams_pipeline_command(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] != "completed", "CLI must preserve configured transcript options"
    assert not requests
    config["platforms"]["teams"]["extra"]["meeting_pipeline"]["transcript_min_chars"] = 80
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert cli.teams_pipeline_command(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "completed", result.get("error_info")
    assert len(requests) == 1
    assert selected_store.path != runtime.resolve_teams_pipeline_store_path()
    assert TeamsPipelineStore(selected_store.path).get_sink_record(f"{sink}:meeting-1")


@pytest.mark.parametrize("sink", ["notion", "linear", "teams_delivery"])
def test_enabled_missing_writer_fails_durably(runtime_factory, sink):
    pipeline = runtime_factory({})
    setattr(pipeline.config, sink, {"enabled": True})
    job = run(pipeline)
    assert job.status == "failed", "enabled sink without writer must not silently succeed"
    assert "writer" in job.error_info["message"].lower()
    assert TeamsPipelineStore(pipeline.store.path).get_job(job.job_id)["status"] == "failed"
