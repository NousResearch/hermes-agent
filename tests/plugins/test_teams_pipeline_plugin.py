"""Tests for the Teams pipeline plugin package."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from gateway.config import GatewayConfig, Platform, PlatformConfig
from plugins.teams_pipeline import register
from plugins.teams_pipeline.pipeline import TeamsMeetingPipeline
from plugins.teams_pipeline.store import TeamsPipelineStore
from plugins.teams_pipeline.models import MeetingArtifact


class FakeGraphClient:
    def __init__(self) -> None:
        self.downloaded = False


async def _transcript_meeting_resolver(client, *, meeting_id=None, join_web_url=None, tenant_id=None):
    from plugins.teams_pipeline.models import TeamsMeetingRef

    return TeamsMeetingRef(
        meeting_id=str(meeting_id),
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


@pytest.mark.anyio
async def test_bind_gateway_runtime_drops_notifications_when_unavailable(monkeypatch):
    from plugins.teams_pipeline import runtime as runtime_module
    from tools.microsoft_graph_auth import MicrosoftGraphConfigError

    class FakeAdapter:
        def __init__(self) -> None:
            self.scheduler = None

        def set_notification_scheduler(self, scheduler) -> None:
            self.scheduler = scheduler

    adapter = FakeAdapter()
    gateway = SimpleNamespace(
        adapters={Platform.MSGRAPH_WEBHOOK: adapter},
        config=GatewayConfig(platforms={}),
        _teams_pipeline_runtime=None,
        _teams_pipeline_runtime_error=None,
    )

    def _raise(_gateway_runner):
        raise MicrosoftGraphConfigError("missing graph env")

    monkeypatch.setattr(runtime_module, "build_pipeline_runtime", _raise)

    bound = runtime_module.bind_gateway_runtime(gateway)

    assert bound is False
    assert "missing graph env" in gateway._teams_pipeline_runtime_error
    assert callable(adapter.scheduler)
    await adapter.scheduler({"id": "notif-2"}, object())


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


def test_store_notification_receipts_are_idempotent(tmp_path):
    store = TeamsPipelineStore(tmp_path / "teams-store.json")
    notification = {
        "subscriptionId": "sub-1",
        "resource": "communications/onlineMeetings/meeting-1",
        "changeType": "updated",
    }
    receipt_key = TeamsPipelineStore.build_notification_receipt_key(notification)

    assert store.record_notification_receipt(receipt_key, notification) is True
    assert store.record_notification_receipt(receipt_key, notification) is False
    assert store.has_notification_receipt(receipt_key) is True

    reloaded = TeamsPipelineStore(tmp_path / "teams-store.json")
    assert reloaded.has_notification_receipt(receipt_key) is True


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
                    display_name="recording.mp4",
                    download_url="https://files.example/recording.mp4",
                )
            ]

        async def _download(client, meeting_ref, recording, destination):
            target = Path(destination)
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
