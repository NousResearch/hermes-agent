"""Pipeline orchestration for Microsoft Teams meeting summaries."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import httpx

from agent.auxiliary_client import async_call_llm, extract_content_or_reasoning
from hermes_constants import get_hermes_home
from plugins.teams_pipeline.meetings import (
    TeamsMeetingArtifactNotFoundError,
    download_recording_artifact,
    enrich_meeting_with_call_record,
    fetch_preferred_transcript_text,
    list_recording_artifacts,
    resolve_meeting_reference,
)
from plugins.teams_pipeline.models import (
    MeetingArtifact,
    TeamsMeetingPipelineJob,
    TeamsMeetingRef,
    TeamsMeetingSummaryPayload,
)
from plugins.teams_pipeline.store import TeamsPipelineStore
from tools.transcription_tools import transcribe_audio

logger = logging.getLogger(__name__)

TERMINAL_PIPELINE_STATES = {"completed", "failed", "retry_scheduled"}
ACTIVE_PIPELINE_STATES = {
    "received",
    "resolving_meeting",
    "fetching_transcript",
    "downloading_recording",
    "transcribing_audio",
    "summarizing",
    "writing_notion",
    "writing_linear",
    "sending_teams",
}


class TeamsPipelineError(RuntimeError):
    """Base class for Teams meeting pipeline failures."""


class TeamsPipelineRetryableError(TeamsPipelineError):
    """Raised when the pipeline should be retried later."""


class TeamsPipelineSinkError(TeamsPipelineError):
    """Raised when an output sink fails."""


class TeamsPipelineArtifactNotFoundError(TeamsPipelineRetryableError):
    """Raised when meeting artifacts are not yet available."""


TranscribeFn = Callable[[str, Optional[str]], dict[str, Any]]
SummarizeFn = Callable[..., Awaitable[dict[str, Any] | TeamsMeetingSummaryPayload]]
SinkFn = Callable[
    [TeamsMeetingSummaryPayload, dict[str, Any], Optional[dict[str, Any]]],
    Awaitable[dict[str, Any]],
]


@dataclass
class TeamsPipelineConfig:
    transcript_preferred: bool = True
    transcript_required: bool = False
    transcription_fallback: bool = True
    stt_model: str | None = None
    ffmpeg_extract_audio: bool = True
    transcript_min_chars: int = 80
    tmp_dir: Path | None = None
    notion: dict[str, Any] | None = None
    linear: dict[str, Any] | None = None
    teams_delivery: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, payload: Optional[dict[str, Any]]) -> "TeamsPipelineConfig":
        data = dict(payload or {})
        tmp_dir = data.get("tmp_dir") or data.get("tmpDir")
        return cls(
            transcript_preferred=bool(data.get("transcript_preferred", True)),
            transcript_required=bool(data.get("transcript_required", False)),
            transcription_fallback=bool(data.get("transcription_fallback", True)),
            stt_model=data.get("stt_model") or data.get("sttModel"),
            ffmpeg_extract_audio=bool(data.get("ffmpeg_extract_audio", True)),
            transcript_min_chars=int(data.get("transcript_min_chars", 80)),
            tmp_dir=Path(tmp_dir) if tmp_dir else None,
            notion=data.get("notion"),
            linear=data.get("linear"),
            teams_delivery=data.get("teams_delivery") or data.get("teamsDelivery"),
        )


class NotionWriter:
    API_BASE = "https://api.notion.com/v1"
    API_VERSION = "2025-09-03"

    def __init__(self, *, api_key: str | None = None, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.api_key = (api_key or os.getenv("NOTION_API_KEY", "")).strip()
        self._transport = transport

    async def write_summary(
        self,
        payload: TeamsMeetingSummaryPayload,
        config: dict[str, Any],
        existing_record: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise TeamsPipelineSinkError("NOTION_API_KEY is not configured.")

        database_id = str(config.get("database_id") or config.get("databaseId") or "").strip()
        page_id = (existing_record or {}).get("page_id")
        if not database_id and not page_id:
            raise TeamsPipelineSinkError("Notion sink requires database_id or an existing page_id.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Notion-Version": self.API_VERSION,
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=30.0, transport=self._transport) as client:
            if page_id:
                response = await client.patch(
                    f"{self.API_BASE}/pages/{page_id}",
                    headers=headers,
                    json={"properties": self._build_properties(payload, config)},
                )
                response.raise_for_status()
                record = response.json()
            else:
                response = await client.post(
                    f"{self.API_BASE}/pages",
                    headers=headers,
                    json={
                        "parent": {"database_id": database_id},
                        "properties": self._build_properties(payload, config),
                        "children": self._build_blocks(payload),
                    },
                )
                response.raise_for_status()
                record = response.json()

        return {"page_id": record["id"], "url": record.get("url")}

    def _build_properties(self, payload: TeamsMeetingSummaryPayload, config: dict[str, Any]) -> dict[str, Any]:
        title_property = config.get("title_property", "Name")
        summary_property = config.get("summary_property")
        meeting_id_property = config.get("meeting_id_property")

        properties: dict[str, Any] = {
            title_property: {
                "title": [{"text": {"content": payload.title or f"Meeting {payload.meeting_ref.meeting_id}"}}]
            }
        }
        if summary_property:
            properties[summary_property] = {
                "rich_text": [{"text": {"content": (payload.summary or "")[:1900]}}]
            }
        if meeting_id_property:
            properties[meeting_id_property] = {
                "rich_text": [{"text": {"content": payload.meeting_ref.meeting_id}}]
            }
        return properties

    def _build_blocks(self, payload: TeamsMeetingSummaryPayload) -> list[dict[str, Any]]:
        sections = [
            ("Summary", payload.summary or ""),
            ("Key Decisions", "\n".join(f"- {item}" for item in payload.key_decisions)),
            ("Action Items", "\n".join(f"- {item}" for item in payload.action_items)),
            ("Risks", "\n".join(f"- {item}" for item in payload.risks)),
        ]
        blocks: list[dict[str, Any]] = []
        for heading, body in sections:
            blocks.append(
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {"rich_text": [{"text": {"content": heading}}]},
                }
            )
            blocks.append(
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"text": {"content": body or "None"}}]},
                }
            )
        return blocks


class LinearWriter:
    API_URL = "https://api.linear.app/graphql"

    def __init__(self, *, api_key: str | None = None, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.api_key = (api_key or os.getenv("LINEAR_API_KEY", "")).strip()
        self._transport = transport

    async def write_summary(
        self,
        payload: TeamsMeetingSummaryPayload,
        config: dict[str, Any],
        existing_record: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise TeamsPipelineSinkError("LINEAR_API_KEY is not configured.")

        headers = {"Authorization": self.api_key, "Content-Type": "application/json"}
        team_id = str(config.get("team_id") or config.get("teamId") or "").strip()
        title = payload.title or f"Meeting Summary: {payload.meeting_ref.meeting_id}"
        description = _render_summary_markdown(payload)
        existing_issue_id = (existing_record or {}).get("issue_id")

        async with httpx.AsyncClient(timeout=30.0, transport=self._transport) as client:
            if existing_issue_id:
                response = await client.post(
                    self.API_URL,
                    headers=headers,
                    json={
                        "query": (
                            "mutation($id: String!, $input: IssueUpdateInput!) "
                            "{ issueUpdate(id: $id, input: $input) { success issue { id identifier url } } }"
                        ),
                        "variables": {
                            "id": existing_issue_id,
                            "input": {"title": title, "description": description},
                        },
                    },
                )
            else:
                if not team_id:
                    raise TeamsPipelineSinkError("Linear sink requires team_id when creating a new issue.")
                response = await client.post(
                    self.API_URL,
                    headers=headers,
                    json={
                        "query": (
                            "mutation($input: IssueCreateInput!) "
                            "{ issueCreate(input: $input) { success issue { id identifier url } } }"
                        ),
                        "variables": {"input": {"teamId": team_id, "title": title, "description": description}},
                    },
                )
            response.raise_for_status()
            payload_json = response.json()

        issue = (
            (((payload_json.get("data") or {}).get("issueUpdate") or {}).get("issue"))
            or (((payload_json.get("data") or {}).get("issueCreate") or {}).get("issue"))
        )
        if not isinstance(issue, dict) or not issue.get("id"):
            raise TeamsPipelineSinkError(f"Linear write failed: {payload_json}")

        return {"issue_id": issue["id"], "identifier": issue.get("identifier"), "url": issue.get("url")}


class TeamsMeetingPipeline:
    """Transcript-first Teams meeting pipeline with durable lifecycle state."""

    def __init__(
        self,
        *,
        graph_client: Any,
        store: TeamsPipelineStore,
        config: TeamsPipelineConfig | dict[str, Any] | None = None,
        transcribe_fn: TranscribeFn = transcribe_audio,
        summarize_fn: Optional[SummarizeFn] = None,
        notion_writer: Optional[NotionWriter] = None,
        linear_writer: Optional[LinearWriter] = None,
        teams_sender: Optional[SinkFn] = None,
    ) -> None:
        self.graph_client = graph_client
        self.store = store
        self.config = config if isinstance(config, TeamsPipelineConfig) else TeamsPipelineConfig.from_dict(config)
        self.transcribe_fn = transcribe_fn
        self.summarize_fn = summarize_fn or self._generate_summary_payload
        self.notion_writer = notion_writer
        self.linear_writer = linear_writer
        self.teams_sender = teams_sender

    def create_job_from_notification(self, notification: dict[str, Any]) -> TeamsMeetingPipelineJob:
        event_id = TeamsPipelineStore.build_notification_receipt_key(notification)
        self.store.record_notification_receipt(event_id, notification)
        existing_job = self._find_job_by_dedupe_key(event_id)
        if existing_job is not None:
            return existing_job
        resource_data = notification.get("resourceData") or {}
        meeting_id = (
            resource_data.get("id")
            or notification.get("meetingId")
            or _extract_meeting_id_from_resource(str(notification.get("resource") or ""))
            or notification.get("resource")
            or event_id
        )
        job = TeamsMeetingPipelineJob(
            job_id=f"teams-job-{uuid.uuid4().hex[:12]}",
            event_id=event_id,
            source_event_type=str(notification.get("changeType") or "graph.notification"),
            dedupe_key=event_id,
            status="received",
            meeting_ref=TeamsMeetingRef(
                meeting_id=str(meeting_id),
                tenant_id=resource_data.get("tenantId") or notification.get("tenantId"),
                metadata={
                    "notification": dict(notification),
                    "join_web_url": resource_data.get("joinWebUrl"),
                    "call_record_id": resource_data.get("callRecordId") or notification.get("callRecordId"),
                },
            ),
        )
        self.store.upsert_job(job.job_id, job.to_dict())
        return job

    async def run_notification(self, notification: dict[str, Any]) -> TeamsMeetingPipelineJob:
        job = self.create_job_from_notification(notification)
        if job.status in TERMINAL_PIPELINE_STATES or job.status in ACTIVE_PIPELINE_STATES - {"received"}:
            return job
        return await self.run_job(job.job_id)

    async def run_job(self, job_or_id: TeamsMeetingPipelineJob | str) -> TeamsMeetingPipelineJob:
        job = self._coerce_job(job_or_id)
        meeting_ref = job.meeting_ref
        if meeting_ref is None:
            raise TeamsPipelineError(f"Job {job.job_id} has no meeting_ref.")

        artifacts: list[MeetingArtifact] = []

        try:
            job = self._persist_job(job, status="resolving_meeting")
            notification = meeting_ref.metadata.get("notification") if isinstance(meeting_ref.metadata, dict) else {}
            resolved_meeting = await resolve_meeting_reference(
                self.graph_client,
                meeting_id=meeting_ref.meeting_id,
                join_web_url=meeting_ref.join_web_url or meeting_ref.metadata.get("join_web_url"),
                tenant_id=meeting_ref.tenant_id,
            )
            job.meeting_ref = resolved_meeting
            job = self._persist_job(job, meeting_ref=resolved_meeting.to_dict())

            transcript_text: str | None = None
            if self.config.transcript_preferred:
                job = self._persist_job(job, status="fetching_transcript")
                transcript_artifact, transcript_text = await fetch_preferred_transcript_text(
                    self.graph_client, resolved_meeting
                )
                if transcript_artifact and transcript_text:
                    artifacts.append(transcript_artifact)
                    if len(transcript_text.strip()) < self.config.transcript_min_chars:
                        transcript_text = None

            if not transcript_text:
                if self.config.transcript_required:
                    raise TeamsPipelineRetryableError(
                        f"Transcript unavailable for meeting {resolved_meeting.meeting_id}."
                    )
                if not self.config.transcription_fallback:
                    raise TeamsPipelineArtifactNotFoundError(
                        "No transcript available and transcription fallback disabled "
                        f"for {resolved_meeting.meeting_id}."
                    )
                job = self._persist_job(job, status="downloading_recording")
                recordings = await list_recording_artifacts(self.graph_client, resolved_meeting)
                if not recordings:
                    raise TeamsPipelineRetryableError(
                        f"Recording unavailable for meeting {resolved_meeting.meeting_id}."
                    )
                recording = recordings[0]
                artifacts.append(recording)
                transcript_text = await self._transcribe_recording(job, resolved_meeting, recording)
                job = self._persist_job(job, selected_artifact_strategy="recording_stt_fallback")
            else:
                job = self._persist_job(job, selected_artifact_strategy="transcript_first")

            call_record_id = notification.get("callRecordId") or (meeting_ref.metadata or {}).get("call_record_id")
            call_record = await enrich_meeting_with_call_record(
                self.graph_client,
                resolved_meeting,
                call_record_id=call_record_id,
            )
            if call_record is not None:
                artifacts.append(call_record)

            job = self._persist_job(job, status="summarizing")
            generated = await self.summarize_fn(
                resolved_meeting=resolved_meeting,
                transcript_text=transcript_text or "",
                artifacts=artifacts,
            )
            summary_payload = (
                generated
                if isinstance(generated, TeamsMeetingSummaryPayload)
                else TeamsMeetingSummaryPayload.from_dict(generated)
            )
            job.summary_payload = summary_payload
            job = self._persist_job(job, summary_payload=summary_payload.to_dict())

            await self._write_sinks(job, summary_payload)
            job = self._persist_job(job, status="completed")
            return job
        except TeamsPipelineRetryableError as exc:
            job = self._persist_job(
                job,
                status="retry_scheduled",
                error_info={"message": str(exc), "retryable": True},
            )
            return job
        except Exception as exc:
            job = self._persist_job(
                job,
                status="failed",
                error_info={"message": str(exc), "type": type(exc).__name__},
            )
            return job

    def _coerce_job(self, job_or_id: TeamsMeetingPipelineJob | str) -> TeamsMeetingPipelineJob:
        if isinstance(job_or_id, TeamsMeetingPipelineJob):
            return job_or_id
        payload = self.store.get_job(str(job_or_id))
        if not payload:
            raise TeamsPipelineError(f"Unknown Teams pipeline job: {job_or_id}")
        return TeamsMeetingPipelineJob.from_dict(payload)

    def _find_job_by_dedupe_key(self, dedupe_key: str) -> TeamsMeetingPipelineJob | None:
        for payload in self.store.list_jobs().values():
            if not isinstance(payload, dict):
                continue
            if str(payload.get("dedupe_key") or "") != dedupe_key:
                continue
            return TeamsMeetingPipelineJob.from_dict(payload)
        return None

    def _persist_job(self, job: TeamsMeetingPipelineJob, **updates: Any) -> TeamsMeetingPipelineJob:
        payload = job.to_dict()
        payload.update(updates)
        stored = self.store.upsert_job(job.job_id, payload)
        return TeamsMeetingPipelineJob.from_dict(stored)

    async def _transcribe_recording(
        self,
        job: TeamsMeetingPipelineJob,
        meeting_ref: TeamsMeetingRef,
        recording: MeetingArtifact,
    ) -> str:
        temp_root = self.config.tmp_dir or (get_hermes_home() / "tmp" / "teams_pipeline")
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(temp_root), prefix="teams-recording-") as tmp_dir:
            recording_name = recording.display_name or f"{recording.artifact_id}.mp4"
            recording_path = Path(tmp_dir) / recording_name
            await download_recording_artifact(
                self.graph_client,
                meeting_ref,
                recording,
                recording_path,
            )
            audio_path = await self._prepare_audio_path(recording_path)
            job = self._persist_job(job, status="transcribing_audio")
            result = await asyncio.to_thread(self.transcribe_fn, str(audio_path), self.config.stt_model)
            if not result.get("success"):
                raise TeamsPipelineRetryableError(str(result.get("error") or "Unknown STT failure"))
            transcript = str(result.get("transcript") or "").strip()
            if not transcript:
                raise TeamsPipelineRetryableError("STT returned an empty transcript.")
            return transcript

    async def _prepare_audio_path(self, recording_path: Path) -> Path:
        if recording_path.suffix.lower() in {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".webm"}:
            return recording_path
        if not self.config.ffmpeg_extract_audio:
            return recording_path
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise TeamsPipelineRetryableError(
                "Recording fallback requires ffmpeg for audio extraction, but ffmpeg was not found."
            )
        audio_path = recording_path.with_suffix(".wav")
        proc = await asyncio.create_subprocess_exec(
            ffmpeg,
            "-y",
            "-i",
            str(recording_path),
            str(audio_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise TeamsPipelineRetryableError(f"ffmpeg audio extraction failed: {detail}")
        return audio_path

    async def _generate_summary_payload(
        self,
        *,
        resolved_meeting: TeamsMeetingRef,
        transcript_text: str,
        artifacts: list[MeetingArtifact],
    ) -> TeamsMeetingSummaryPayload:
        prompt = _build_summary_prompt(resolved_meeting, transcript_text, artifacts)
        try:
            response = await async_call_llm(
                task="call",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You summarize meeting transcripts. Return only valid JSON with keys: "
                            "summary, key_decisions, action_items, risks, confidence, confidence_notes."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=900,
            )
            content = extract_content_or_reasoning(response)
            parsed = _parse_summary_json(content)
        except Exception as exc:
            logger.info("Teams pipeline LLM summary unavailable, using heuristic summary: %s", exc)
            parsed = _heuristic_summary(transcript_text)

        metrics = _collect_call_metrics(artifacts)
        return TeamsMeetingSummaryPayload(
            meeting_ref=resolved_meeting,
            title=str(resolved_meeting.metadata.get("subject") or f"Meeting {resolved_meeting.meeting_id}"),
            start_time=resolved_meeting.metadata.get("startDateTime"),
            end_time=resolved_meeting.metadata.get("endDateTime"),
            participants=_collect_participants(resolved_meeting),
            transcript_text=transcript_text,
            summary=parsed.get("summary"),
            key_decisions=list(parsed.get("key_decisions") or []),
            action_items=list(parsed.get("action_items") or []),
            risks=list(parsed.get("risks") or []),
            call_metrics=metrics,
            source_artifacts=artifacts,
            confidence=parsed.get("confidence"),
            confidence_notes=parsed.get("confidence_notes"),
            notion_target=(self.config.notion or {}).get("database_id"),
            linear_target=(self.config.linear or {}).get("team_id"),
            teams_target=(
                (self.config.teams_delivery or {}).get("channel_id")
                or (self.config.teams_delivery or {}).get("chat_id")
            ),
        )

    async def _write_sinks(self, job: TeamsMeetingPipelineJob, payload: TeamsMeetingSummaryPayload) -> None:
        if self.config.notion and self.config.notion.get("enabled") and self.notion_writer:
            job = self._persist_job(job, status="writing_notion")
            sink_key = f"notion:{payload.meeting_ref.meeting_id}"
            existing = self.store.get_sink_record(sink_key)
            result = await self.notion_writer.write_summary(payload, self.config.notion, existing)
            self.store.upsert_sink_record(sink_key, result)

        if self.config.linear and self.config.linear.get("enabled") and self.linear_writer:
            job = self._persist_job(job, status="writing_linear")
            sink_key = f"linear:{payload.meeting_ref.meeting_id}"
            existing = self.store.get_sink_record(sink_key)
            result = await self.linear_writer.write_summary(payload, self.config.linear, existing)
            self.store.upsert_sink_record(sink_key, result)

        if self.config.teams_delivery and self.config.teams_delivery.get("enabled") and self.teams_sender:
            job = self._persist_job(job, status="sending_teams")
            sink_key = f"teams:{payload.meeting_ref.meeting_id}"
            existing = self.store.get_sink_record(sink_key)
            if hasattr(self.teams_sender, "write_summary"):
                result = await self.teams_sender.write_summary(payload, self.config.teams_delivery, existing)
            else:
                result = await self.teams_sender(payload, self.config.teams_delivery, existing)
            self.store.upsert_sink_record(sink_key, result)


def _collect_call_metrics(artifacts: list[MeetingArtifact]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for artifact in artifacts:
        if artifact.artifact_type == "call_record":
            metrics.update(dict(artifact.metadata.get("metrics") or {}))
    metrics["artifact_count"] = len(artifacts)
    return metrics


def _collect_participants(meeting_ref: TeamsMeetingRef) -> list[str]:
    participants = meeting_ref.metadata.get("participants") or []
    result: list[str] = []
    if isinstance(participants, list):
        for item in participants:
            if isinstance(item, dict):
                name = item.get("displayName") or (((item.get("identity") or {}).get("user") or {}).get("displayName"))
                if name:
                    result.append(str(name))
    return result


def _extract_meeting_id_from_resource(resource: str) -> str | None:
    if not resource:
        return None
    parts = [part for part in resource.split("/") if part]
    if not parts:
        return None
    if "onlineMeetings" in parts:
        index = parts.index("onlineMeetings")
        if index + 1 < len(parts):
            return parts[index + 1]
    return parts[-1]


def _build_summary_prompt(
    meeting_ref: TeamsMeetingRef,
    transcript_text: str,
    artifacts: list[MeetingArtifact],
) -> str:
    artifact_lines = [f"- {artifact.artifact_type}:{artifact.artifact_id}:{artifact.display_name or ''}" for artifact in artifacts]
    return (
        f"Meeting ID: {meeting_ref.meeting_id}\n"
        f"Title: {meeting_ref.metadata.get('subject') or 'Unknown'}\n"
        f"Artifacts:\n{chr(10).join(artifact_lines) or '- none'}\n\n"
        "Transcript:\n"
        f"{transcript_text[:18000]}"
    )


def _parse_summary_json(content: str) -> dict[str, Any]:
    text = (content or "").strip()
    if not text:
        return _heuristic_summary("")
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    payload = json.loads(text)
    return {
        "summary": str(payload.get("summary") or "").strip(),
        "key_decisions": [str(item).strip() for item in payload.get("key_decisions", []) if str(item).strip()],
        "action_items": [str(item).strip() for item in payload.get("action_items", []) if str(item).strip()],
        "risks": [str(item).strip() for item in payload.get("risks", []) if str(item).strip()],
        "confidence": str(payload.get("confidence") or "medium").strip(),
        "confidence_notes": str(payload.get("confidence_notes") or "").strip(),
    }


def _heuristic_summary(transcript_text: str) -> dict[str, Any]:
    lines = [line.strip(" -*\t") for line in transcript_text.splitlines() if line.strip()]
    summary = " ".join(lines[:3])[:1200] or "Transcript unavailable or too sparse for a confident summary."
    action_items = [
        line for line in lines if line.lower().startswith(("action:", "todo:", "next step:", "follow up:"))
    ][:8]
    risks = [line for line in lines if "risk" in line.lower() or "blocker" in line.lower()][:6]
    decisions = [line for line in lines if "decide" in line.lower() or "decision" in line.lower()][:6]
    confidence = "low" if len(transcript_text.strip()) < 300 else "medium"
    return {
        "summary": summary,
        "key_decisions": decisions,
        "action_items": action_items,
        "risks": risks,
        "confidence": confidence,
        "confidence_notes": "Generated with heuristic fallback because no LLM summary response was available.",
    }


def _render_summary_markdown(payload: TeamsMeetingSummaryPayload) -> str:
    lines = [
        f"# {payload.title or f'Meeting {payload.meeting_ref.meeting_id}'}",
        "",
        "## Summary",
        payload.summary or "No summary available.",
        "",
        "## Key Decisions",
        *([f"- {item}" for item in payload.key_decisions] or ["- None"]),
        "",
        "## Action Items",
        *([f"- {item}" for item in payload.action_items] or ["- None"]),
        "",
        "## Risks",
        *([f"- {item}" for item in payload.risks] or ["- None"]),
        "",
        f"Confidence: {payload.confidence or 'unknown'}",
        payload.confidence_notes or "",
    ]
    return "\n".join(lines).strip()
