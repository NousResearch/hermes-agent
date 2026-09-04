"""Tests for the Teams pipeline plugin package."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from pathlib import Path

import pytest

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from gateway.config import GatewayConfig, Platform, PlatformConfig
from plugins.teams_pipeline import register
from plugins.teams_pipeline.meetings import (
    TeamsMeetingError,
    looks_like_transcript_id,
    parse_graph_meeting_resource,
    resolve_meeting_reference,
)
from plugins.teams_pipeline.models import MeetingArtifact
from plugins.teams_pipeline.pipeline import TeamsMeetingPipeline
from plugins.teams_pipeline.store import TeamsPipelineStore


class FakeGraphClient:
    def __init__(self) -> None:
        self.downloaded = False


async def _transcript_meeting_resolver(
    client, *, meeting_id=None, join_web_url=None, tenant_id=None, organizer_user_id=None
):
    from plugins.teams_pipeline.models import TeamsMeetingRef

    return TeamsMeetingRef(
        meeting_id=str(meeting_id),
        organizer_user_id=organizer_user_id,
        tenant_id=tenant_id,
        metadata={"subject": "Weekly Sync", "participants": [{"displayName": "Ada"}]},
    )


async def _no_call_record(*args, **kwargs):
    return None


def test_register_adds_cli_only():
    mgr = PluginManager()
    manifest = PluginManifest(name="teams_pipeline")
    ctx = PluginContext(manifest, mgr)

    register(ctx)

    assert "teams-pipeline" in mgr._cli_commands
    entry = mgr._cli_commands["teams-pipeline"]
    assert entry["plugin"] == "teams_pipeline"
    assert callable(entry["setup_fn"])
    assert callable(entry["handler_fn"])


def test_runtime_config_uses_existing_teams_platform_settings():
    from plugins.teams_pipeline.runtime import build_pipeline_runtime_config

    gateway_config = GatewayConfig(
        platforms={
            Platform("teams"): PlatformConfig(
                enabled=True,
                extra={
                    "delivery_mode": "graph",
                    "team_id": "team-1",
                    "channel_id": "channel-1",
                    "meeting_pipeline": {
                        "transcript_min_chars": 120,
                        "notion": {"enabled": True, "database_id": "db-1"},
                    },
                },
            )
        }
    )

    runtime_config = build_pipeline_runtime_config(gateway_config)

    assert runtime_config["transcript_min_chars"] == 120
    assert runtime_config["notion"]["database_id"] == "db-1"
    assert runtime_config["teams_delivery"] == {
        "enabled": True,
        "mode": "graph",
        "team_id": "team-1",
        "channel_id": "channel-1",
    }


def test_build_pipeline_runtime_reuses_existing_teams_adapter_surface(monkeypatch, tmp_path):
    from plugins.teams_pipeline import runtime as runtime_module

    class FakeWriter:
        def __init__(self, platform_config=None, **kwargs) -> None:
            self.platform_config = platform_config

    monkeypatch.setattr(runtime_module, "build_graph_client", lambda: object())
    monkeypatch.setattr(runtime_module, "resolve_teams_pipeline_store_path", lambda: tmp_path / "teams-store.json")
    monkeypatch.setattr("plugins.platforms.teams.adapter.TeamsSummaryWriter", FakeWriter)

    gateway = SimpleNamespace(
        config=GatewayConfig(
            platforms={
                Platform("teams"): PlatformConfig(
                    enabled=True,
                    extra={
                        "delivery_mode": "incoming_webhook",
                        "incoming_webhook_url": "https://example.com/hook",
                    },
                )
            }
        )
    )

    runtime = runtime_module.build_pipeline_runtime(gateway)

    assert isinstance(runtime.teams_sender, FakeWriter)
    assert runtime.teams_sender.platform_config is gateway.config.platforms[Platform("teams")]


@pytest.mark.anyio
async def test_bind_gateway_runtime_attaches_scheduler(monkeypatch, tmp_path):
    from plugins.teams_pipeline import runtime as runtime_module

    class FakeAdapter:
        def __init__(self) -> None:
            self.scheduler = None

        def set_notification_scheduler(self, scheduler) -> None:
            self.scheduler = scheduler

    class FakePipeline:
        def __init__(self) -> None:
            self.notifications = []

        async def run_notification(self, notification):
            self.notifications.append(notification)

    adapter = FakeAdapter()
    pipeline = FakePipeline()
    gateway = SimpleNamespace(
        adapters={Platform.MSGRAPH_WEBHOOK: adapter},
        config=GatewayConfig(platforms={}),
        _teams_pipeline_runtime=None,
        _teams_pipeline_runtime_error=None,
    )

    monkeypatch.setattr(runtime_module, "build_pipeline_runtime", lambda gateway_runner: pipeline)

    bound = runtime_module.bind_gateway_runtime(gateway)

    assert bound is True
    assert gateway._teams_pipeline_runtime is pipeline
    assert callable(adapter.scheduler)

    notification = {"id": "notif-1"}
    await adapter.scheduler(notification, object())
    assert pipeline.notifications == [notification]


def test_store_persists_subscription_event_and_job_state(tmp_path):
    store_path = tmp_path / "teams-store.json"
    store = TeamsPipelineStore(store_path)
    store.upsert_subscription(
        "sub-1",
        {"client_state": "abc", "resource": "communications/onlineMeetings"},
    )
    store.record_event_timestamp("evt-1", "2026-05-03T19:30:00Z")
    store.upsert_job("job-1", {"status": "received", "event_id": "evt-1"})
    store.upsert_sink_record("notion:meeting-1", {"page_id": "page-1"})

    reloaded = TeamsPipelineStore(store_path)
    subscription = reloaded.get_subscription("sub-1")
    job = reloaded.get_job("job-1")
    sink = reloaded.get_sink_record("notion:meeting-1")

    assert subscription is not None
    assert subscription["subscription_id"] == "sub-1"
    assert subscription["client_state"] == "abc"
    assert reloaded.get_event_timestamp("evt-1") == "2026-05-03T19:30:00Z"
    assert job is not None
    assert job["status"] == "received"
    assert sink is not None
    assert sink["page_id"] == "page-1"


@pytest.mark.anyio
class TestTeamsMeetingPipeline:
    async def test_transcript_first_path_persists_state_and_skips_recording(self, tmp_path, monkeypatch):
        from plugins.teams_pipeline import pipeline as pipeline_module

        monkeypatch.setattr(pipeline_module, "resolve_meeting_reference", _transcript_meeting_resolver)

        async def _fetch_transcript(client, meeting_ref):
            return (
                MeetingArtifact(artifact_type="transcript", artifact_id="tx-1", display_name="meeting.vtt"),
                "Action: Send draft by Friday.\nDecision: Ship the transcript-first path.\nDetailed transcript content.",
            )

        async def _call_record(client, meeting_ref, *, call_record_id=None, allow_permission_errors=True):
            return MeetingArtifact(
                artifact_type="call_record",
                artifact_id="call-1",
                metadata={"metrics": {"participant_count": 4}},
            )

        async def _summarize(**kwargs):
            return pipeline_module.TeamsMeetingSummaryPayload(
                meeting_ref=kwargs["resolved_meeting"],
                title="Weekly Sync",
                transcript_text=kwargs["transcript_text"],
                summary="Short summary",
                key_decisions=["Ship the transcript-first path."],
                action_items=["Send draft by Friday."],
                risks=["Timeline risk."],
                confidence="high",
                confidence_notes="Transcript available.",
                source_artifacts=kwargs["artifacts"],
            )

        monkeypatch.setattr(pipeline_module, "fetch_preferred_transcript_text", _fetch_transcript)
        monkeypatch.setattr(pipeline_module, "enrich_meeting_with_call_record", _call_record)

        store = TeamsPipelineStore(tmp_path / "teams-store.json")
        pipeline = TeamsMeetingPipeline(
            graph_client=FakeGraphClient(),
            store=store,
            config={"transcript_min_chars": 20},
            summarize_fn=_summarize,
        )

        job = await pipeline.run_notification(
            {
                "id": "notif-1",
                "changeType": "updated",
                "resource": "communications/onlineMeetings/meeting-123",
                "resourceData": {"id": "meeting-123"},
            }
        )

        assert job.status == "completed"
        assert job.selected_artifact_strategy == "transcript_first"
        assert job.summary_payload is not None
        assert job.summary_payload.summary == "Short summary"
        stored = store.get_job(job.job_id)
        assert stored is not None
        assert stored["status"] == "completed"

    async def test_recording_fallback_uses_stt_and_updates_sink_records(self, tmp_path, monkeypatch):
        from plugins.teams_pipeline import pipeline as pipeline_module

        monkeypatch.setattr(pipeline_module, "resolve_meeting_reference", _transcript_meeting_resolver)

        async def _no_transcript(client, meeting_ref):
            return None, None

        async def _recordings(client, meeting_ref):
            return [
                MeetingArtifact(
                    artifact_type="recording",
                    artifact_id="rec-1",
                    display_name="../../nested/recording.mp4",
                    download_url="https://files.example/recording.mp4",
                )
            ]

        downloaded_targets = []

        async def _download(client, meeting_ref, recording, destination):
            target = Path(destination)
            downloaded_targets.append(target)
            target.write_bytes(b"video-bytes")
            return {"path": str(target), "size_bytes": 11, "content_type": "video/mp4"}

        async def _prepare_audio(self, recording_path):
            audio_path = recording_path.with_suffix(".wav")
            audio_path.write_bytes(b"audio-bytes")
            return audio_path

        def _transcribe(file_path, model):
            return {"success": True, "transcript": "Action: Follow up with Legal.\nRisk: Budget approval pending.", "provider": "local"}

        async def _summarize(**kwargs):
            return pipeline_module.TeamsMeetingSummaryPayload(
                meeting_ref=kwargs["resolved_meeting"],
                title="Weekly Sync",
                transcript_text=kwargs["transcript_text"],
                summary="Fallback summary",
                key_decisions=[],
                action_items=["Follow up with Legal."],
                risks=["Budget approval pending."],
                confidence="medium",
                confidence_notes="Generated from STT fallback.",
                source_artifacts=kwargs["artifacts"],
            )

        class FakeNotionWriter:
            async def write_summary(self, payload, config, existing_record=None):
                return {"page_id": existing_record.get("page_id") if existing_record else "page-1", "url": "https://notion.so/page-1"}

        async def _teams_sender(payload, config, existing_record=None):
            return {"message_id": existing_record.get("message_id") if existing_record else "msg-1"}

        monkeypatch.setattr(pipeline_module, "fetch_preferred_transcript_text", _no_transcript)
        monkeypatch.setattr(pipeline_module, "list_recording_artifacts", _recordings)
        monkeypatch.setattr(pipeline_module, "download_recording_artifact", _download)
        monkeypatch.setattr(pipeline_module.TeamsMeetingPipeline, "_prepare_audio_path", _prepare_audio)
        monkeypatch.setattr(pipeline_module, "enrich_meeting_with_call_record", _no_call_record)

        store = TeamsPipelineStore(tmp_path / "teams-store.json")
        pipeline = TeamsMeetingPipeline(
            graph_client=FakeGraphClient(),
            store=store,
            config={
                "notion": {"enabled": True, "database_id": "db-1"},
                "teams_delivery": {"enabled": True, "channel_id": "channel-1"},
            },
            transcribe_fn=_transcribe,
            summarize_fn=_summarize,
            notion_writer=FakeNotionWriter(),
            teams_sender=_teams_sender,
        )

        job = await pipeline.run_notification(
            {
                "id": "notif-2",
                "changeType": "updated",
                "resource": "communications/onlineMeetings/meeting-456",
                "resourceData": {"id": "meeting-456"},
            }
        )

        assert job.status == "completed"
        assert job.selected_artifact_strategy == "recording_stt_fallback"
        assert job.summary_payload is not None
        assert job.summary_payload.summary == "Fallback summary"
        assert downloaded_targets
        assert downloaded_targets[0].name == "recording.mp4"
        assert "nested" not in str(downloaded_targets[0])
        notion_record = store.get_sink_record("notion:meeting-456")
        teams_record = store.get_sink_record("teams:meeting-456")
        assert notion_record is not None
        assert notion_record["page_id"] == "page-1"
        assert teams_record is not None
        assert teams_record["message_id"] == "msg-1"


    async def test_missing_transcript_and_recording_schedules_retry(self, tmp_path, monkeypatch):
        from plugins.teams_pipeline import pipeline as pipeline_module

        monkeypatch.setattr(pipeline_module, "resolve_meeting_reference", _transcript_meeting_resolver)
        monkeypatch.setattr(pipeline_module, "fetch_preferred_transcript_text", lambda *a, **kw: asyncio.sleep(0, result=(None, None)))
        monkeypatch.setattr(pipeline_module, "list_recording_artifacts", lambda *a, **kw: asyncio.sleep(0, result=[]))

        store = TeamsPipelineStore(tmp_path / "teams-store.json")
        pipeline = TeamsMeetingPipeline(
            graph_client=FakeGraphClient(),
            store=store,
            config={},
            summarize_fn=lambda **kwargs: asyncio.sleep(0, result=None),
        )

        job = await pipeline.run_notification(
            {
                "id": "notif-3",
                "changeType": "updated",
                "resource": "communications/onlineMeetings/meeting-789",
                "resourceData": {"id": "meeting-789"},
            }
        )

        assert job.status == "retry_scheduled"
        assert job.error_info["retryable"] is True
        assert "Recording unavailable" in job.error_info["message"]

    async def test_duplicate_notification_reuses_completed_job(self, tmp_path, monkeypatch):
        from plugins.teams_pipeline import pipeline as pipeline_module

        monkeypatch.setattr(pipeline_module, "resolve_meeting_reference", _transcript_meeting_resolver)

        async def _fetch_transcript(client, meeting_ref):
            return (
                MeetingArtifact(artifact_type="transcript", artifact_id="tx-dup", display_name="meeting.vtt"),
                "Decision: Keep duplicate notifications idempotent.\nAction: Verify the cached job is reused.",
            )

        summarize_calls = 0

        async def _summarize(**kwargs):
            nonlocal summarize_calls
            summarize_calls += 1
            return pipeline_module.TeamsMeetingSummaryPayload(
                meeting_ref=kwargs["resolved_meeting"],
                title="Weekly Sync",
                transcript_text=kwargs["transcript_text"],
                summary="Duplicate-safe summary",
                key_decisions=["Keep duplicate notifications idempotent."],
                action_items=["Verify the cached job is reused."],
                confidence="high",
                confidence_notes="Transcript available.",
                source_artifacts=kwargs["artifacts"],
            )

        monkeypatch.setattr(pipeline_module, "fetch_preferred_transcript_text", _fetch_transcript)
        monkeypatch.setattr(pipeline_module, "enrich_meeting_with_call_record", _no_call_record)

        store = TeamsPipelineStore(tmp_path / "teams-store.json")
        pipeline = TeamsMeetingPipeline(
            graph_client=FakeGraphClient(),
            store=store,
            config={"transcript_min_chars": 20},
            summarize_fn=_summarize,
        )
        notification = {
            "id": "notif-dup",
            "changeType": "updated",
            "resource": "communications/onlineMeetings/meeting-dup",
            "resourceData": {"id": "meeting-dup"},
        }

        first_job = await pipeline.run_notification(notification)
        second_job = await pipeline.run_notification(notification)

        assert first_job.status == "completed"
        assert second_job.status == "completed"
        assert second_job.job_id == first_job.job_id
        assert summarize_calls == 1
        assert len(store.list_jobs()) == 1
        receipt_key = TeamsPipelineStore.build_notification_receipt_key(notification)
        assert store.has_notification_receipt(receipt_key) is True


def test_parse_graph_meeting_resource_reads_quoted_users_transcript_path():
    parsed = parse_graph_meeting_resource(
        "users/org-1/onlineMeetings('MSo-meeting')/transcripts('ktViz-transcript-TranscriptV2=')"
    )

    assert parsed["organizer_user_id"] == "org-1"
    assert parsed["meeting_id"] == "MSo-meeting"
    assert parsed["transcript_id"] == "ktViz-transcript-TranscriptV2="
    assert parsed["recording_id"] is None


def test_parse_graph_meeting_resource_reads_quoted_user_key_odata_id():
    parsed = parse_graph_meeting_resource(
        "users('org-1')/onlineMeetings('MSo-meeting')/transcripts('ktViz-transcript-TranscriptV2=')"
    )

    assert parsed["organizer_user_id"] == "org-1"
    assert parsed["meeting_id"] == "MSo-meeting"
    assert parsed["transcript_id"] == "ktViz-transcript-TranscriptV2="
    assert parsed["recording_id"] is None


def test_parse_graph_meeting_resource_reads_graph_url_quoted_user_key():
    parsed = parse_graph_meeting_resource(
        "https://graph.microsoft.com/v1.0/users('org-1')/onlineMeetings('MSo-meeting')"
        "/transcripts('tx-1')"
    )

    assert parsed["organizer_user_id"] == "org-1"
    assert parsed["meeting_id"] == "MSo-meeting"
    assert parsed["transcript_id"] == "tx-1"


def test_parse_graph_meeting_resource_ignores_get_all_transcripts_sentinel():
    parsed = parse_graph_meeting_resource("communications/onlineMeetings/getAllTranscripts")

    assert parsed["meeting_id"] is None
    assert parsed["organizer_user_id"] is None
    assert parsed["transcript_id"] is None


def test_looks_like_transcript_id_detects_graph_call_transcripts():
    assert looks_like_transcript_id(
        "ktVizInGAAAA-TranscriptV2=",
        odata_type="#Microsoft.Graph.callTranscript",
    )
    assert looks_like_transcript_id("abcTranscriptV2=")
    assert not looks_like_transcript_id("MSo-meeting")


def test_looks_like_transcript_id_detects_base64_encoded_marker():
    # Real-world getAllTranscripts resourceData.id shape: base64url blob whose
    # DECODED payload ends in "-TranscriptV2" while the encoded form has no
    # readable marker (from a field report, Aug 2026).
    import base64

    encoded = base64.urlsafe_b64encode(
        b"...meeting_ABC@thread.v2#55d6e479-2db7-4297-8590-98f6011a9613-1787004872-TranscriptV2"
    ).decode().rstrip("=") + "="
    assert "transcript" not in encoded.lower()
    assert looks_like_transcript_id(encoded)
    # Meeting ids (base64 of "0#...#0**19:meeting_...@thread.v2") must NOT match.
    meeting_encoded = base64.b64encode(
        b"0#af8b854e-e592-4d5a-b04c-6e63a3b502e7#0**19:meeting_ABC@thread.v2"
    ).decode()
    assert not looks_like_transcript_id(meeting_encoded)


def test_create_job_from_get_all_transcripts_uses_meeting_id_not_transcript_id(tmp_path):
    store = TeamsPipelineStore(tmp_path / "teams-store.json")
    pipeline = TeamsMeetingPipeline(graph_client=FakeGraphClient(), store=store)
    transcript_id = "ktVizInGAAAAi_B6lATZRTE5Om1lZXRpbmdfTranscriptV2="

    job = pipeline.create_job_from_notification(
        {
            "id": "notif-tx",
            "changeType": "created",
            "resource": "communications/onlineMeetings/getAllTranscripts",
            "resourceData": {
                "id": transcript_id,
                "@odata.type": "#Microsoft.Graph.callTranscript",
                "@odata.id": (
                    "users/976f4b31-fd01-4e0b-9178-29cc40c14438/"
                    f"onlineMeetings('MSo-meeting-id')/transcripts('{transcript_id}')"
                ),
            },
            "tenantId": "tenant-1",
        }
    )

    assert job.meeting_ref is not None
    assert job.meeting_ref.meeting_id == "MSo-meeting-id"
    assert job.meeting_ref.organizer_user_id == "976f4b31-fd01-4e0b-9178-29cc40c14438"
    assert job.meeting_ref.meeting_id != transcript_id
    assert job.meeting_ref.metadata.get("transcript_id") == transcript_id


def test_create_job_from_quoted_user_odata_id_uses_meeting_id_not_transcript_id(tmp_path):
    store = TeamsPipelineStore(tmp_path / "teams-store.json")
    pipeline = TeamsMeetingPipeline(graph_client=FakeGraphClient(), store=store)
    transcript_id = "ktVizInGAAAAi_B6lATZRTE5Om1lZXRpbmdfTranscriptV2="

    job = pipeline.create_job_from_notification(
        {
            "id": "notif-quoted-user",
            "changeType": "created",
            "resource": "communications/onlineMeetings/getAllTranscripts",
            "resourceData": {
                "id": transcript_id,
                "@odata.type": "#Microsoft.Graph.callTranscript",
                "@odata.id": (
                    "users('976f4b31-fd01-4e0b-9178-29cc40c14438')"
                    f"/onlineMeetings('MSo-meeting-id')/transcripts('{transcript_id}')"
                ),
            },
            "tenantId": "tenant-1",
        }
    )

    assert job.meeting_ref is not None
    assert job.meeting_ref.meeting_id == "MSo-meeting-id"
    assert job.meeting_ref.organizer_user_id == "976f4b31-fd01-4e0b-9178-29cc40c14438"
    assert job.meeting_ref.meeting_id != transcript_id
    assert job.meeting_ref.metadata.get("transcript_id") == transcript_id


@pytest.mark.anyio
async def test_run_job_reparses_quoted_user_odata_id_when_stored_meeting_id_is_transcript(
    tmp_path, monkeypatch
):
    from plugins.teams_pipeline import pipeline as pipeline_module
    from plugins.teams_pipeline.models import TeamsMeetingRef

    captured: dict[str, str | None] = {}

    async def _resolve(client, *, meeting_id=None, join_web_url=None, tenant_id=None, organizer_user_id=None):
        captured["meeting_id"] = meeting_id
        captured["organizer_user_id"] = organizer_user_id
        return TeamsMeetingRef(
            meeting_id=str(meeting_id),
            organizer_user_id=organizer_user_id,
            tenant_id=tenant_id,
            metadata={"subject": "Standup"},
        )

    async def _fetch_transcript(client, meeting_ref):
        return (
            MeetingArtifact(artifact_type="transcript", artifact_id="tx-1", display_name="meeting.vtt"),
            "Decision: Re-parse stored Graph notifications on replay.\nAction: Confirm the meeting id is used.",
        )

    async def _summarize(**kwargs):
        return pipeline_module.TeamsMeetingSummaryPayload(
            meeting_ref=kwargs["resolved_meeting"],
            title="Standup",
            transcript_text=kwargs["transcript_text"],
            summary="Reparsed",
            source_artifacts=kwargs["artifacts"],
        )

    monkeypatch.setattr(pipeline_module, "resolve_meeting_reference", _resolve)
    monkeypatch.setattr(pipeline_module, "fetch_preferred_transcript_text", _fetch_transcript)
    monkeypatch.setattr(pipeline_module, "enrich_meeting_with_call_record", _no_call_record)

    store = TeamsPipelineStore(tmp_path / "teams-store.json")
    pipeline = TeamsMeetingPipeline(
        graph_client=FakeGraphClient(),
        store=store,
        config={"transcript_min_chars": 20},
        summarize_fn=_summarize,
    )
    transcript_id = "ktVizInGAAAAi_B6lATZRTE5Om1lZXRpbmdfTranscriptV2="
    notification = {
        "id": "notif-replay",
        "changeType": "created",
        "resource": "communications/onlineMeetings/getAllTranscripts",
        "resourceData": {
            "id": transcript_id,
            "@odata.type": "#Microsoft.Graph.callTranscript",
            "@odata.id": (
                "users('org-1')/onlineMeetings('MSo-from-odata')/transcripts("
                f"'{transcript_id}')"
            ),
        },
    }
    job = pipeline.create_job_from_notification(notification)
    assert job.meeting_ref is not None
    job.meeting_ref.meeting_id = transcript_id
    store.upsert_job(job.job_id, job.to_dict())

    ran = await pipeline.run_job(job.job_id)

    assert captured["meeting_id"] == "MSo-from-odata"
    assert captured["organizer_user_id"] == "org-1"
    assert ran.status == "completed"


def test_create_job_from_users_resource_path_without_odata_id(tmp_path):
    store = TeamsPipelineStore(tmp_path / "teams-store.json")
    pipeline = TeamsMeetingPipeline(graph_client=FakeGraphClient(), store=store)

    job = pipeline.create_job_from_notification(
        {
            "id": "notif-resource-path",
            "changeType": "created",
            "resource": "users/org-2/onlineMeetings('MSo-from-resource')/transcripts('tx-99')",
            "resourceData": {
                "id": "tx-99",
                "@odata.type": "#Microsoft.Graph.callTranscript",
            },
        }
    )

    assert job.meeting_ref is not None
    assert job.meeting_ref.meeting_id == "MSo-from-resource"
    assert job.meeting_ref.organizer_user_id == "org-2"
    assert job.meeting_ref.metadata.get("transcript_id") == "tx-99"


@pytest.mark.anyio
async def test_resolve_meeting_reference_uses_organizer_scoped_graph_path():
    class RecordingGraphClient:
        def __init__(self) -> None:
            self.paths: list[str] = []

        async def get_json(self, path, *, params=None, headers=None):
            self.paths.append(path)
            return {
                "id": "MSo-meeting-id",
                "subject": "Standup",
                "organizer": {"identity": {"user": {"id": "org-1"}}},
            }

    client = RecordingGraphClient()
    meeting_ref = await resolve_meeting_reference(
        client,
        meeting_id="MSo-meeting-id",
        organizer_user_id="org-1",
        tenant_id="tenant-1",
    )

    assert client.paths == ["/users/org-1/onlineMeetings/MSo-meeting-id"]
    assert meeting_ref.meeting_id == "MSo-meeting-id"
    assert meeting_ref.organizer_user_id == "org-1"


@pytest.mark.anyio
async def test_resolve_meeting_reference_refuses_transcript_id_without_join_url():
    class BoomGraphClient:
        async def get_json(self, path, *, params=None, headers=None):
            raise AssertionError(f"Graph should not be called, got {path}")

    with pytest.raises(TeamsMeetingError, match="transcript id"):
        await resolve_meeting_reference(
            BoomGraphClient(),
            meeting_id="ktVizInGAAAA-TranscriptV2=",
        )
