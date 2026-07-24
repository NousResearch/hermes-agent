"""HEGI v2 end-to-end pipeline."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from .actions import persist_new_actions
from .analyzer import build_minutes, minimal_minutes
from .archive import ArchiveManager
from .collector import HermesSQLiteCollector
from .config import HegiConfig, validate_config
from .episode import EpisodeDetector
from .llm import HermesLLMClient, HierarchicalMeetingAnalyzer
from .memory import MCPMemoryBackend, MemoryEvaluator
from .models import (
    AgentPosition,
    ConceptDefinition,
    DiscussionStage,
    EvidenceItem,
    MeetingEpisode,
    MeetingMinutes,
    MemoryEvaluation,
    MemoryMatch,
    SourceMessage,
    as_jsonable,
)
from .notify import TelegramReporter, load_env_value
from .observability import RunLogger
from .quality import audit_minutes
from .state import StateStore


def episode_from_dict(payload: dict[str, Any]) -> MeetingEpisode:
    return MeetingEpisode(
        meeting_id=payload["meeting_id"],
        chat_id=payload["chat_id"],
        started_at=float(payload["started_at"]),
        ended_at=float(payload["ended_at"]),
        participants=list(payload["participants"]),
        messages=[SourceMessage(**item) for item in payload["messages"]],
        episode_hash=payload["episode_hash"],
        topic_hint=payload.get("topic_hint"),
        status=payload.get("status", "quiet"),
    )


def minutes_from_dict(payload: dict[str, Any]) -> MeetingMinutes:
    memory_payload = payload.get("memory_evaluation")
    memory = None
    if isinstance(memory_payload, dict):
        memory = MemoryEvaluation(
            searched_queries=list(memory_payload.get("searched_queries", [])),
            matched_memories=[
                MemoryMatch(**item) for item in memory_payload.get("matched_memories", [])
            ],
            duplicate_score=memory_payload.get("duplicate_score"),
            novelty_score=memory_payload.get("novelty_score"),
            significance_score=memory_payload.get("significance_score"),
            durability_score=memory_payload.get("durability_score"),
            recommendation=memory_payload.get("recommendation", "needs_professor_review"),
            candidate_memory_title=memory_payload.get("candidate_memory_title"),
            candidate_memory_summary=memory_payload.get("candidate_memory_summary"),
            reasons=list(memory_payload.get("reasons", [])),
        )
    from .models import ActionItem

    return MeetingMinutes(
        meeting_id=payload["meeting_id"],
        title=payload["title"],
        background=payload["background"],
        agenda=list(payload.get("agenda", [])),
        discussion_flow=[
            DiscussionStage(**item) for item in payload.get("discussion_flow", [])
        ],
        agent_positions=[
            AgentPosition(**item) for item in payload.get("agent_positions", [])
        ],
        professor_positions=list(payload.get("professor_positions", [])),
        agreements=list(payload.get("agreements", [])),
        disagreements=list(payload.get("disagreements", [])),
        unresolved_questions=list(payload.get("unresolved_questions", [])),
        new_concepts=[
            ConceptDefinition(**item) for item in payload.get("new_concepts", [])
        ],
        evidence_and_sources=[
            EvidenceItem(**item) for item in payload.get("evidence_and_sources", [])
        ],
        research_direction=list(payload.get("research_direction", [])),
        action_items=[ActionItem(**item) for item in payload.get("action_items", [])],
        memory_evaluation=memory,
        confidence=float(payload.get("confidence", 0)),
        warnings=list(payload.get("warnings", [])),
        recommendation=payload.get("recommendation", ""),
        metadata=dict(payload.get("metadata", {})),
    )


class HegiPipeline:
    def __init__(
        self,
        config: HegiConfig,
        *,
        state: StateStore | None = None,
        llm_client: HermesLLMClient | None = None,
        memory_backend: Any | None = None,
        telegram_sender: Any | None = None,
    ):
        self.config = config
        self.state = state or StateStore(config.state_db)
        episode_config = config.section("episode")
        self.collector = HermesSQLiteCollector(
            config.agents,
            self.state,
            timestamp_bucket_seconds=int(
                episode_config.get("timestamp_bucket_seconds", 3)
            ),
        )
        self.detector = EpisodeDetector(
            quiet_minutes=int(episode_config["quiet_minutes"]),
            max_gap_minutes=int(episode_config["max_gap_minutes"]),
            minimum_agents=int(episode_config["minimum_agents"]),
            minimum_messages=int(episode_config["minimum_messages"]),
            maximum_messages=int(episode_config.get("maximum_messages", 10000)),
        )
        analysis = config.section("analysis")
        self.llm_client = llm_client or HermesLLMClient(
            provider=str(analysis.get("provider", "")),
            model=str(analysis.get("model", "")),
            max_tokens=int(analysis.get("max_output_tokens", 10000)),
            timeout_seconds=int(analysis.get("timeout_seconds", 180)),
        )
        self.analyzer = HierarchicalMeetingAnalyzer(
            self.llm_client,
            max_input_chars=int(analysis.get("max_input_chars", 100000)),
            chunk_chars=int(analysis.get("chunk_chars", 30000)),
        )
        memory = config.section("memory")
        self.memory_backend = memory_backend or MCPMemoryBackend(
            read_server=str(memory.get("read_server", "memory-forest-read")),
            search_tool=str(memory.get("search_tool", "")),
            draft_server=str(memory.get("draft_server", "")),
            draft_tool=str(memory.get("draft_tool", "")),
        )
        self.telegram_sender = telegram_sender
        self.logger = RunLogger(config.state_db.parent / "runs.jsonl")

    def collect(
        self, *, since: float | None = None, now: float | None = None
    ) -> list[SourceMessage]:
        if since is None and not self.state.has_cursors():
            episode = self.config.section("episode")
            reference_time = time.time() if now is None else now
            since = reference_time - int(
                episode.get("initial_lookback_minutes", 240)
            ) * 60
        return self.collector.collect(self.config.chat_id, since=since)

    def run_once(
        self, *, dry_run: bool = True, now: float | None = None
    ) -> list[dict[str, Any]]:
        errors = validate_config(self.config, require_runtime=True)
        if errors:
            raise ValueError("; ".join(errors))
        if not self.config.enabled:
            raise RuntimeError("HEGI가 설정에서 비활성화되어 있습니다.")
        started = time.monotonic()
        run_id = uuid.uuid4().hex
        messages = self.collect(now=now)
        episodes = self.detector.detect(messages, now=now)
        results: list[dict[str, Any]] = []
        for episode in episodes:
            if episode.status != "quiet":
                continue
            existing = self.state.episode_by_id(episode.meeting_id)
            if existing and existing["status"] not in {"failed", "analyzing"}:
                continue
            if not existing:
                self.state.save_episode(
                    episode.meeting_id,
                    episode.episode_hash,
                    as_jsonable(episode),
                    "analyzing",
                )
            else:
                self.state.update_episode(episode.meeting_id, status="analyzing")
            result = self._process_episode(episode, dry_run=dry_run)
            result["run_id"] = run_id
            results.append(result)
        self.logger.write(
            {
                "run_id": run_id,
                "collector_count": len(messages),
                "meeting_count": len(results),
                "latency_ms": int((time.monotonic() - started) * 1000),
                "dry_run": dry_run,
            }
        )
        return results

    def analyze_episode(self, episode: MeetingEpisode) -> MeetingMinutes:
        analysis = self.config.section("analysis")
        try:
            payload = self.analyzer.analyze_payload(episode)
            minutes = build_minutes(payload, episode)
        except Exception as exc:
            minutes = minimal_minutes(episode, str(exc))
            self.state.add_dead_letter(
                "analysis",
                {"episode": as_jsonable(episode)},
                str(exc),
                episode.meeting_id,
            )
            raise AnalysisFallback(minutes) from exc
        minutes.metadata = {
            "meeting_id": episode.meeting_id,
            "episode_hash": episode.episode_hash,
            "source_message_ids": episode.source_message_ids,
            "source_session_ids": episode.source_session_ids,
            "model": analysis.get("model", ""),
            "prompt_version": analysis.get("prompt_version", "v2.0.0"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "hegi_version": "2.0.1",
        }
        return minutes

    def _process_episode(
        self, episode: MeetingEpisode, *, dry_run: bool
    ) -> dict[str, Any]:
        try:
            minutes = self.analyze_episode(episode)
        except AnalysisFallback as fallback:
            minutes = fallback.minutes
            self.state.update_episode(
                episode.meeting_id,
                status="failed",
                minutes=as_jsonable(minutes),
                error=minutes.warnings[0],
            )
            return {
                "meeting_id": episode.meeting_id,
                "status": "failed",
                "warnings": minutes.warnings,
            }
        memory_cfg = self.config.section("memory")
        if memory_cfg.get("enabled", True):
            try:
                minutes.memory_evaluation = MemoryEvaluator(
                    self.memory_backend
                ).evaluate(minutes)
            except Exception as exc:
                minutes.memory_evaluation = MemoryEvaluation(
                    recommendation="needs_professor_review",
                    reasons=[f"Memory Forest 검색 실패: {exc}"],
                )
                minutes.warnings.append("Memory Forest 검색 실패")
        minutes.action_items = persist_new_actions(
            self.state, episode.meeting_id, minutes.action_items
        )
        minutes.warnings.extend(audit_minutes(minutes, episode))
        archive = ArchiveManager(self.config.local_spool, self.config.nas_root).archive(
            minutes, episode
        )
        sent_ids: list[str] = []
        telegram_config = self.config.section("telegram")
        if telegram_config.get("enabled") and self.config.section("reports").get(
            "telegram", True
        ):
            token = load_env_value(self.config.curator_env, "TELEGRAM_BOT_TOKEN")
            sent_ids = TelegramReporter(
                self.state,
                token=token,
                chat_id=self.config.chat_id,
                sender=self.telegram_sender,
            ).send(minutes, dry_run=dry_run)
        if dry_run:
            status = "dry_run"
        else:
            status = "reported"
            self.state.update_episode(
                episode.meeting_id, status="reported", minutes=as_jsonable(minutes)
            )
            self.state.consume_range(
                episode.chat_id, episode.started_at, episode.ended_at
            )
        event = {
            "meeting_id": episode.meeting_id,
            "episode_hash": episode.episode_hash,
            "status": status,
            "participants": episode.participants,
            "episode_start": episode.started_at,
            "episode_end": episode.ended_at,
            "archive_path": archive["markdown"],
            "telegram_message_ids": sent_ids,
            "memory_search_count": len(
                minutes.memory_evaluation.searched_queries
                if minutes.memory_evaluation
                else []
            ),
            "draft_status": "not_created",
            "warnings": minutes.warnings,
        }
        self.logger.write(event)
        return event


class AnalysisFallback(Exception):
    def __init__(self, minutes: MeetingMinutes):
        super().__init__("structured analysis failed")
        self.minutes = minutes
