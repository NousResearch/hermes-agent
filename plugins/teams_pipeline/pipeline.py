"""Pipeline orchestration for Microsoft Teams meeting summaries."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import httpx

from agent.auxiliary_client import async_call_llm, extract_content_or_reasoning
from agent.secret_scope import get_secret
from hermes_constants import get_hermes_home
from plugins.teams_pipeline.meetings import (
    download_recording_artifact,
    enrich_meeting_with_call_record,
    fetch_preferred_transcript_text,
    list_recording_artifacts,
    looks_like_transcript_id,
    parse_graph_meeting_resource,
    resolve_meeting_reference)
from plugins.teams_pipeline.models import (
    MeetingArtifact, TeamsMeetingPipelineJob, TeamsMeetingRef, TeamsMeetingSummaryPayload)
from plugins.teams_pipeline.store import TeamsPipelineStore
from tools.transcription_tools import transcribe_audio

logger = logging.getLogger(__name__)

TERMINAL_PIPELINE_STATES = {"completed", "failed", "retry_scheduled"}
ACTIVE_PIPELINE_STATES = {
    "received", "resolving_meeting", "fetching_transcript", "downloading_recording",
    "transcribing_audio", "summarizing", "writing_notion", "writing_linear", "sending_teams"}
_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".webm"}
_SUMMARY_SYSTEM_PROMPT = (
    "You summarize meeting transcripts. Return only valid JSON with keys: "
    "summary, key_decisions, action_items, risks, confidence, confidence_notes.")


class TeamsPipelineError(RuntimeError): """Base class for Teams meeting pipeline failures."""
class TeamsPipelineRetryableError(TeamsPipelineError): """Raised when the pipeline should be retried later."""
class TeamsPipelineSinkError(TeamsPipelineError): """Raised when an output sink fails."""
class TeamsPipelineArtifactNotFoundError(TeamsPipelineRetryableError): """Raised when meeting artifacts are not yet available."""


TranscribeFn = Callable[[str, Optional[str]], dict[str, Any]]
SummarizeFn = Callable[..., Awaitable[dict[str, Any] | TeamsMeetingSummaryPayload]]
SinkFn = Callable[[TeamsMeetingSummaryPayload, dict[str, Any], Optional[dict[str, Any]]], Awaitable[dict[str, Any]]]


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
        flags = {"transcript_preferred": True, "transcript_required": False, "transcription_fallback": True, "ffmpeg_extract_audio": True}
        return cls(
            **{name: bool(data.get(name, default)) for name, default in flags.items()},
            stt_model=data.get("stt_model") or data.get("sttModel"), transcript_min_chars=int(data.get("transcript_min_chars", 80)),
            tmp_dir=Path(tmp_dir) if tmp_dir else None, notion=data.get("notion"), linear=data.get("linear"),
            teams_delivery=data.get("teams_delivery") or data.get("teamsDelivery"))


def _rich_text(content: str) -> dict[str, Any]:
    return {"rich_text": [{"text": {"content": content}}]}


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _sections(payload: TeamsMeetingSummaryPayload) -> list[tuple[str, str]]:
    return [("Summary", payload.summary or ""), ("Key Decisions", _bullets(payload.key_decisions)),
            ("Action Items", _bullets(payload.action_items)), ("Risks", _bullets(payload.risks))]


class _HttpSinkWriter:
    """Shared API-key + httpx transport plumbing for the Notion/Linear sinks."""

    SECRET_NAME = ""

    def __init__(self, *, api_key: str | None = None, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.api_key = (api_key or get_secret(self.SECRET_NAME, "") or "").strip()
        self._transport = transport

    def _require_api_key(self) -> None:
        if not self.api_key:
            raise TeamsPipelineSinkError(f"{self.SECRET_NAME} is not configured.")

    async def _request(self, method: str, url: str, *, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30.0, transport=self._transport) as client:
            response = await client.request(method, url, headers=headers, json=body)
        response.raise_for_status()
        return response.json()


class NotionWriter(_HttpSinkWriter):
    API_BASE = "https://api.notion.com/v1"
    API_VERSION = "2025-09-03"
    SECRET_NAME = "NOTION_API_KEY"

    async def write_summary(
        self, payload: TeamsMeetingSummaryPayload, config: dict[str, Any], existing_record: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        self._require_api_key()
        database_id = str(config.get("database_id") or config.get("databaseId") or "").strip()
        page_id = (existing_record or {}).get("page_id")
        if not database_id and not page_id:
            raise TeamsPipelineSinkError("Notion sink requires database_id or an existing page_id.")
        headers = {"Authorization": f"Bearer {self.api_key}", "Notion-Version": self.API_VERSION, "Content-Type": "application/json"}
        properties = self._build_properties(payload, config)
        # Re-runs update the existing page's properties only; body blocks are written once on create.
        if page_id:
            record = await self._request("PATCH", f"{self.API_BASE}/pages/{page_id}", headers=headers, body={"properties": properties})
        else:
            body = {"parent": {"database_id": database_id}, "properties": properties, "children": self._build_blocks(payload)}
            record = await self._request("POST", f"{self.API_BASE}/pages", headers=headers, body=body)
        return {"page_id": record["id"], "url": record.get("url")}

    def _build_properties(self, payload: TeamsMeetingSummaryPayload, config: dict[str, Any]) -> dict[str, Any]:
        title = payload.title or f"Meeting {payload.meeting_ref.meeting_id}"
        properties: dict[str, Any] = {config.get("title_property", "Name"): {"title": [{"text": {"content": title}}]}}
        if summary_property := config.get("summary_property"):
            properties[summary_property] = _rich_text((payload.summary or "")[:1900])
        if meeting_id_property := config.get("meeting_id_property"):
            properties[meeting_id_property] = _rich_text(payload.meeting_ref.meeting_id)
        return properties

    def _build_blocks(self, payload: TeamsMeetingSummaryPayload) -> list[dict[str, Any]]:
        return [block for heading, body in _sections(payload) for block in (
            {"object": "block", "type": "heading_2", "heading_2": _rich_text(heading)},
            {"object": "block", "type": "paragraph", "paragraph": _rich_text(body or "None")})]


class LinearWriter(_HttpSinkWriter):
    API_URL = "https://api.linear.app/graphql"
    SECRET_NAME = "LINEAR_API_KEY"
    _UPDATE_MUTATION = "mutation($id: String!, $input: IssueUpdateInput!) { issueUpdate(id: $id, input: $input) { success issue { id identifier url } } }"
    _CREATE_MUTATION = "mutation($input: IssueCreateInput!) { issueCreate(input: $input) { success issue { id identifier url } } }"

    async def write_summary(
        self, payload: TeamsMeetingSummaryPayload, config: dict[str, Any], existing_record: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        self._require_api_key()
        headers = {"Authorization": self.api_key, "Content-Type": "application/json"}
        team_id = str(config.get("team_id") or config.get("teamId") or "").strip()
        issue_input = {"title": payload.title or f"Meeting Summary: {payload.meeting_ref.meeting_id}", "description": _render_summary_markdown(payload)}
        if existing_issue_id := (existing_record or {}).get("issue_id"):
            body = {"query": self._UPDATE_MUTATION, "variables": {"id": existing_issue_id, "input": issue_input}}
        elif not team_id:
            raise TeamsPipelineSinkError("Linear sink requires team_id when creating a new issue.")
        else:
            body = {"query": self._CREATE_MUTATION, "variables": {"input": {"teamId": team_id, **issue_input}}}
        payload_json = await self._request("POST", self.API_URL, headers=headers, body=body)
        data = payload_json.get("data") or {}
        issue = (data.get("issueUpdate") or {}).get("issue") or (data.get("issueCreate") or {}).get("issue")
        if not isinstance(issue, dict) or not issue.get("id"):
            raise TeamsPipelineSinkError(f"Linear write failed: {payload_json}")
        return {"issue_id": issue["id"], "identifier": issue.get("identifier"), "url": issue.get("url")}


class TeamsMeetingPipeline:
    """Transcript-first Teams meeting pipeline with durable lifecycle state."""

    def __init__(
        self, *, graph_client: Any, store: TeamsPipelineStore,
        config: TeamsPipelineConfig | dict[str, Any] | None = None, transcribe_fn: TranscribeFn = transcribe_audio,
        summarize_fn: Optional[SummarizeFn] = None, notion_writer: Optional[NotionWriter] = None,
        linear_writer: Optional[LinearWriter] = None, teams_sender: Optional[SinkFn] = None) -> None:
        self.graph_client, self.store, self.transcribe_fn = graph_client, store, transcribe_fn
        self.config = config if isinstance(config, TeamsPipelineConfig) else TeamsPipelineConfig.from_dict(config)
        self.summarize_fn = summarize_fn or self._generate_summary_payload
        self.notion_writer, self.linear_writer, self.teams_sender = notion_writer, linear_writer, teams_sender

    def create_job_from_notification(self, notification: dict[str, Any]) -> TeamsMeetingPipelineJob:
        event_id = TeamsPipelineStore.build_notification_receipt_key(notification)
        self.store.record_notification_receipt(event_id, notification)
        existing_job = self._find_job_by_dedupe_key(event_id)
        if existing_job is not None:
            return existing_job
        resource_data = notification.get("resourceData") or {}
        meeting_id, organizer_user_id, extra_metadata = _meeting_ids_from_notification(notification)
        job = TeamsMeetingPipelineJob(
            job_id=f"teams-job-{uuid.uuid4().hex[:12]}", event_id=event_id, dedupe_key=event_id, status="received",
            source_event_type=str(notification.get("changeType") or "graph.notification"),
            meeting_ref=TeamsMeetingRef(
                meeting_id=str(meeting_id), organizer_user_id=organizer_user_id,
                tenant_id=resource_data.get("tenantId") or notification.get("tenantId"),
                metadata={"notification": dict(notification), "join_web_url": resource_data.get("joinWebUrl"),
                          "call_record_id": resource_data.get("callRecordId") or notification.get("callRecordId"), **extra_metadata},
            ),
        )
        self.store.upsert_job(job.job_id, job.to_dict())
        return job

    async def run_notification(self, notification: dict[str, Any]) -> TeamsMeetingPipelineJob:
        job = self.create_job_from_notification(notification)
        # Only freshly "received" jobs run; terminal or in-flight duplicates are returned as-is.
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
            job, resolved_meeting, notification = await self._resolve_meeting(job, meeting_ref)
            job, transcript_text = await self._obtain_transcript(job, resolved_meeting, artifacts)
            call_record_id = notification.get("callRecordId") or (meeting_ref.metadata or {}).get("call_record_id")
            call_record = await enrich_meeting_with_call_record(self.graph_client, resolved_meeting, call_record_id=call_record_id)
            if call_record is not None:
                artifacts.append(call_record)
            job = self._persist_job(job, status="summarizing")
            summary_payload = await self.summarize_fn(resolved_meeting=resolved_meeting, transcript_text=transcript_text or "", artifacts=artifacts)
            if not isinstance(summary_payload, TeamsMeetingSummaryPayload):
                summary_payload = TeamsMeetingSummaryPayload.from_dict(summary_payload)
            job.summary_payload = summary_payload
            job = self._persist_job(job, summary_payload=summary_payload.to_dict())
            await self._write_sinks(job, summary_payload)
            return self._persist_job(job, status="completed")
        except TeamsPipelineRetryableError as exc:
            return self._persist_job(job, status="retry_scheduled", error_info={"message": str(exc), "retryable": True})
        except Exception as exc:
            return self._persist_job(job, status="failed", error_info={"message": str(exc), "type": type(exc).__name__})

    async def _resolve_meeting(
        self, job: TeamsMeetingPipelineJob, meeting_ref: TeamsMeetingRef) -> tuple[TeamsMeetingPipelineJob, TeamsMeetingRef, Any]:
        """Phase 1: resolve the Graph meeting; returns (job, resolved_meeting, stored notification)."""
        job = self._persist_job(job, status="resolving_meeting")
        notification = meeting_ref.metadata.get("notification") if isinstance(meeting_ref.metadata, dict) else {}
        meeting_id, organizer_user_id = meeting_ref.meeting_id, meeting_ref.organizer_user_id
        # Re-parse the stored notification: older jobs may have persisted a transcript id as meeting_id.
        if isinstance(notification, dict) and notification:
            parsed_id, parsed_org, _extra = _meeting_ids_from_notification(notification)
            if parsed_org:
                organizer_user_id = organizer_user_id or parsed_org
            if parsed_id and not looks_like_transcript_id(parsed_id):
                meeting_id = parsed_id
        resolved_meeting = await resolve_meeting_reference(
            self.graph_client, meeting_id=meeting_id, tenant_id=meeting_ref.tenant_id,
            join_web_url=meeting_ref.join_web_url or meeting_ref.metadata.get("join_web_url"), organizer_user_id=organizer_user_id)
        if meeting_ref.metadata:
            resolved_meeting.metadata = {**meeting_ref.metadata, **resolved_meeting.metadata}
        resolved_meeting.organizer_user_id = resolved_meeting.organizer_user_id or meeting_ref.organizer_user_id
        job.meeting_ref = resolved_meeting
        return self._persist_job(job, meeting_ref=resolved_meeting.to_dict()), resolved_meeting, notification

    async def _obtain_transcript(
        self, job: TeamsMeetingPipelineJob, resolved_meeting: TeamsMeetingRef, artifacts: list[MeetingArtifact]
    ) -> tuple[TeamsMeetingPipelineJob, str | None]:
        """Phase 2: transcript first, then the recording->STT fallback (appends chosen artifacts)."""
        transcript_text: str | None = None
        if self.config.transcript_preferred:
            job = self._persist_job(job, status="fetching_transcript")
            transcript_artifact, transcript_text = await fetch_preferred_transcript_text(self.graph_client, resolved_meeting)
            if transcript_artifact and transcript_text:
                artifacts.append(transcript_artifact)
                if len(transcript_text.strip()) < self.config.transcript_min_chars:
                    transcript_text = None
        if transcript_text:
            return self._persist_job(job, selected_artifact_strategy="transcript_first"), transcript_text
        if self.config.transcript_required:
            raise TeamsPipelineRetryableError(f"Transcript unavailable for meeting {resolved_meeting.meeting_id}.")
        if not self.config.transcription_fallback:
            raise TeamsPipelineArtifactNotFoundError(
                f"No transcript available and transcription fallback disabled for {resolved_meeting.meeting_id}.")
        job = self._persist_job(job, status="downloading_recording")
        recordings = await list_recording_artifacts(self.graph_client, resolved_meeting)
        if not recordings:
            raise TeamsPipelineRetryableError(f"Recording unavailable for meeting {resolved_meeting.meeting_id}.")
        artifacts.append(recordings[0])
        transcript_text = await self._transcribe_recording(job, resolved_meeting, recordings[0])
        return self._persist_job(job, selected_artifact_strategy="recording_stt_fallback"), transcript_text

    def _coerce_job(self, job_or_id: TeamsMeetingPipelineJob | str) -> TeamsMeetingPipelineJob:
        if isinstance(job_or_id, TeamsMeetingPipelineJob):
            return job_or_id
        payload = self.store.get_job(str(job_or_id))
        if not payload:
            raise TeamsPipelineError(f"Unknown Teams pipeline job: {job_or_id}")
        return TeamsMeetingPipelineJob.from_dict(payload)

    def _find_job_by_dedupe_key(self, dedupe_key: str) -> TeamsMeetingPipelineJob | None:
        for payload in self.store.list_jobs().values():
            if isinstance(payload, dict) and str(payload.get("dedupe_key") or "") == dedupe_key:
                return TeamsMeetingPipelineJob.from_dict(payload)
        return None

    def _persist_job(self, job: TeamsMeetingPipelineJob, **updates: Any) -> TeamsMeetingPipelineJob:
        return TeamsMeetingPipelineJob.from_dict(self.store.upsert_job(job.job_id, {**job.to_dict(), **updates}))

    async def _transcribe_recording(self, job: TeamsMeetingPipelineJob, meeting_ref: TeamsMeetingRef, recording: MeetingArtifact) -> str:
        temp_root = self.config.tmp_dir or (get_hermes_home() / "tmp" / "teams_pipeline")
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(temp_root), prefix="teams-recording-") as tmp_dir:
            # display_name is organizer-controlled Graph data: keep only the basename so a crafted
            # "../../etc/cron.d/evil" cannot escape tmp_dir. Path(...).name leaves "." / ".." / ""
            # unchanged (joining "tmp/.." resolves to the parent), so reject those explicitly.
            fallback_name = f"{recording.artifact_id}.mp4"
            recording_name = Path(recording.display_name or fallback_name).name
            recording_path = Path(tmp_dir) / (fallback_name if recording_name in ("", ".", "..") else recording_name)
            await download_recording_artifact(self.graph_client, meeting_ref, recording, recording_path)
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
        if recording_path.suffix.lower() in _AUDIO_SUFFIXES or not self.config.ffmpeg_extract_audio:
            return recording_path
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise TeamsPipelineRetryableError("Recording fallback requires ffmpeg for audio extraction, but ffmpeg was not found.")
        audio_path = recording_path.with_suffix(".wav")
        proc = await asyncio.create_subprocess_exec(
            ffmpeg, "-y", "-i", str(recording_path), str(audio_path), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        _stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise TeamsPipelineRetryableError(f"ffmpeg audio extraction failed: {stderr.decode('utf-8', errors='replace').strip()}")
        return audio_path

    async def _generate_summary_payload(
        self, *, resolved_meeting: TeamsMeetingRef, transcript_text: str, artifacts: list[MeetingArtifact]) -> TeamsMeetingSummaryPayload:
        prompt = _build_summary_prompt(resolved_meeting, transcript_text, artifacts)
        try:
            response = await async_call_llm(
                task="call", temperature=0.2, max_tokens=900,
                messages=[{"role": "system", "content": _SUMMARY_SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
            parsed = _parse_summary_json(extract_content_or_reasoning(response))
        except Exception as exc:
            logger.info("Teams pipeline LLM summary unavailable, using heuristic summary: %s", exc)
            parsed = _heuristic_summary(transcript_text)
        teams_delivery = self.config.teams_delivery or {}
        return TeamsMeetingSummaryPayload(
            meeting_ref=resolved_meeting, transcript_text=transcript_text, source_artifacts=artifacts,
            title=str(resolved_meeting.metadata.get("subject") or f"Meeting {resolved_meeting.meeting_id}"),
            start_time=resolved_meeting.metadata.get("startDateTime"), end_time=resolved_meeting.metadata.get("endDateTime"),
            participants=_collect_participants(resolved_meeting), call_metrics=_collect_call_metrics(artifacts),
            summary=parsed.get("summary"), confidence=parsed.get("confidence"), confidence_notes=parsed.get("confidence_notes"),
            **{key: list(parsed.get(key) or []) for key in ("key_decisions", "action_items", "risks")},
            notion_target=(self.config.notion or {}).get("database_id"), linear_target=(self.config.linear or {}).get("team_id"),
            teams_target=teams_delivery.get("channel_id") or teams_delivery.get("chat_id"))

    async def _write_sinks(self, job: TeamsMeetingPipelineJob, payload: TeamsMeetingSummaryPayload) -> None:
        # Sink order is part of the contract: Notion, then Linear, then Teams delivery.
        sinks = (
            ("notion", "writing_notion", self.config.notion, self.notion_writer),
            ("linear", "writing_linear", self.config.linear, self.linear_writer),
            ("teams", "sending_teams", self.config.teams_delivery, self.teams_sender))
        for name, status, config, sink in sinks:
            if not (config and config.get("enabled") and sink):
                continue
            job = self._persist_job(job, status=status)
            sink_key = f"{name}:{payload.meeting_ref.meeting_id}"
            existing = self.store.get_sink_record(sink_key)
            # Sinks may be writer objects (write_summary) or bare async callables.
            write: Any = getattr(sink, "write_summary", sink)
            self.store.upsert_sink_record(sink_key, await write(payload, config, existing))


def _collect_call_metrics(artifacts: list[MeetingArtifact]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for artifact in artifacts:
        if artifact.artifact_type == "call_record":
            metrics.update(dict(artifact.metadata.get("metrics") or {}))
    return {**metrics, "artifact_count": len(artifacts)}


def _collect_participants(meeting_ref: TeamsMeetingRef) -> list[str]:
    participants = meeting_ref.metadata.get("participants") or []
    if not isinstance(participants, list):
        return []
    names = (item.get("displayName") or (((item.get("identity") or {}).get("user") or {}).get("displayName"))
             for item in participants if isinstance(item, dict))
    return [str(name) for name in names if name]


def _odata_field(payload: dict[str, Any], name: str) -> Any:
    return payload.get(f"@{name}") or payload.get(name)


def _organizer_user_id_from_payload(payload: dict[str, Any]) -> str | None:
    organizer = payload.get("meetingOrganizer") or payload.get("organizer")
    if isinstance(organizer, dict):
        user = organizer.get("user")
        if user is None and isinstance(organizer.get("identity"), dict):
            user = organizer["identity"].get("user")
        if isinstance(user, dict) and user.get("id"):
            return str(user["id"]).strip() or None
    return str(payload.get("organizerUserId") or payload.get("organizer_user_id") or "").strip() or None


def _resource_data_id_is_artifact(notification: dict[str, Any], resource_data: dict[str, Any]) -> bool:
    odata_type = str(_odata_field(resource_data, "odata.type") or "")
    if "calltranscript" in odata_type.lower() or "callrecording" in odata_type.lower():
        return True
    resource = str(notification.get("resource") or "").lower()
    if any(marker in resource for marker in ("getalltranscripts", "getallrecordings", "/transcripts", "/recordings")):
        return True
    return looks_like_transcript_id(str(resource_data.get("id") or ""), odata_type=odata_type)


def _meeting_ids_from_notification(notification: dict[str, Any]) -> tuple[str, str | None, dict[str, Any]]:
    """Return (meeting_id, organizer_user_id, extra_metadata) from a Graph change notification: parsed
    resource paths win over flat fields; resourceData.id counts as a meeting id only when it is not an
    artifact id; meeting_id never comes back empty (falls back to an artifact id, then the receipt key)."""
    resource_data = notification.get("resourceData")
    resource_data = resource_data if isinstance(resource_data, dict) else {}
    odata_type = str(_odata_field(resource_data, "odata.type") or "")
    parsed_paths = [parse_graph_meeting_resource(str(raw or "")) for raw in (
        _odata_field(resource_data, "odata.id"), notification.get("resource"), resource_data.get("transcriptContentUrl"))]
    first = {key: next((parsed[key] for parsed in parsed_paths if parsed.get(key)), None)
             for key in ("organizer_user_id", "meeting_id", "transcript_id", "recording_id")}
    organizer_user_id = first["organizer_user_id"] or _organizer_user_id_from_payload(resource_data) or _organizer_user_id_from_payload(notification)
    meeting_id = first["meeting_id"] or (str(resource_data.get("meetingId") or notification.get("meetingId") or "").strip() or None)
    transcript_id, recording_id = first["transcript_id"], first["recording_id"]
    resource_data_id = str(resource_data.get("id") or "").strip() or None
    if resource_data_id and not _resource_data_id_is_artifact(notification, resource_data):
        meeting_id = meeting_id or resource_data_id
    elif resource_data_id and not transcript_id and looks_like_transcript_id(resource_data_id, odata_type=odata_type):
        transcript_id = resource_data_id
    meeting_id = meeting_id or transcript_id or recording_id or TeamsPipelineStore.build_notification_receipt_key(notification)
    extra_metadata = {k: v for k, v in (("transcript_id", transcript_id), ("recording_id", recording_id)) if v}
    return str(meeting_id), organizer_user_id, extra_metadata


def _build_summary_prompt(meeting_ref: TeamsMeetingRef, transcript_text: str, artifacts: list[MeetingArtifact]) -> str:
    artifact_lines = [f"- {artifact.artifact_type}:{artifact.artifact_id}:{artifact.display_name or ''}" for artifact in artifacts]
    return (
        f"Meeting ID: {meeting_ref.meeting_id}\n"
        f"Title: {meeting_ref.metadata.get('subject') or 'Unknown'}\n"
        f"Artifacts:\n{chr(10).join(artifact_lines) or '- none'}\n\n"
        f"Transcript:\n{transcript_text[:18000]}")


def _clean_items(values: Any) -> list[str]:
    return [str(item).strip() for item in values if str(item).strip()]


def _parse_summary_json(content: str) -> dict[str, Any]:
    text = (content or "").strip()
    if not text:
        return _heuristic_summary("")
    # Tolerate prose or code fences around the JSON object.
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    payload = json.loads(text)
    return {
        "summary": str(payload.get("summary") or "").strip(),
        **{key: _clean_items(payload.get(key, [])) for key in ("key_decisions", "action_items", "risks")},
        "confidence": str(payload.get("confidence") or "medium").strip(),
        "confidence_notes": str(payload.get("confidence_notes") or "").strip()}


def _heuristic_summary(transcript_text: str) -> dict[str, Any]:
    lines = [line.strip(" -*\t") for line in transcript_text.splitlines() if line.strip()]
    lowered = [line.lower() for line in lines]
    return {
        "summary": " ".join(lines[:3])[:1200] or "Transcript unavailable or too sparse for a confident summary.",
        "key_decisions": [line for line, low in zip(lines, lowered) if "decide" in low or "decision" in low][:6],
        "action_items": [line for line, low in zip(lines, lowered) if low.startswith(("action:", "todo:", "next step:", "follow up:"))][:8],
        "risks": [line for line, low in zip(lines, lowered) if "risk" in low or "blocker" in low][:6],
        "confidence": "low" if len(transcript_text.strip()) < 300 else "medium",
        "confidence_notes": "Generated with heuristic fallback because no LLM summary response was available."}


def _render_summary_markdown(payload: TeamsMeetingSummaryPayload) -> str:
    lines = [f"# {payload.title or f'Meeting {payload.meeting_ref.meeting_id}'}"]
    for heading, body in _sections(payload):
        lines += ["", f"## {heading}", body or ("No summary available." if heading == "Summary" else "- None")]
    lines += ["", f"Confidence: {payload.confidence or 'unknown'}", payload.confidence_notes or ""]
    return "\n".join(lines).strip()


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import os  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
