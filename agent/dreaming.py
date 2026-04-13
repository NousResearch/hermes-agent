from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home
from agent.memory_search import LocalMemorySearch
from tools.memory_tool import MemoryStore

_MAX_STABLE_CANDIDATES = 5
_MAX_OPEN_LOOPS = 5
_MAX_TOMORROW_CUES = 5
_MAX_HOLD_ITEMS = 5

_WORK_VERBS = ("fix", "build", "add", "implement", "investigate", "debug", "review", "continue", "finish", "ship", "wire", "tune", "improve")
_LOW_SIGNAL_PATTERNS = (
    r"^how do i do step \d+",
    r"^open terminal",
    r"^what will happen",
    r"^right now:",
    r"^i dont know which folder",
    r"^now give me one recommendation",
)
_ASSISTANT_META_PATTERNS = (
    r"^i will ",
    r"^let me ",
    r"^i can ",
    r"^here('?s| is) how",
    r"^open terminal",
)


@dataclass
class DreamPaths:
    root: Path
    state: Path
    journal: Path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_line(text: str, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", str(text or "").strip())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _content_text(content: Any) -> str:
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return _normalize_line(" ".join(parts), limit=220)
    if isinstance(content, str):
        return _normalize_line(content, limit=220)
    return ""


def _looks_low_signal(text: str) -> bool:
    lowered = text.casefold().strip()
    if not lowered:
        return True
    if len(lowered) < 12:
        return True
    return any(re.search(pattern, lowered) for pattern in _LOW_SIGNAL_PATTERNS)


def _looks_assistant_meta(text: str) -> bool:
    lowered = text.casefold().strip()
    if not lowered:
        return True
    return any(re.search(pattern, lowered) for pattern in _ASSISTANT_META_PATTERNS)


def _looks_task_relevant(text: str) -> bool:
    lowered = text.casefold()
    return any(verb in lowered for verb in _WORK_VERBS) or any(token in lowered for token in ("auth", "login", "repo", "build", "dream", "cron", "memory", "note", "review"))


class DreamingEngine:
    """Conservative local-only dreaming MVP.

    Produces a bounded dream artifact from recent session messages, writes local
    state under ~/.hermes/dreams/, and can render a review note for Obsidian.
    Auto-promotion is supported but off by default.
    """

    def __init__(
        self,
        hermes_home: Optional[Path] = None,
        memory_store: Optional[MemoryStore] = None,
        session_db: Any = None,
        auto_promote: bool = False,
    ):
        self.hermes_home = Path(hermes_home or get_hermes_home())
        self.paths = DreamPaths(
            root=self.hermes_home / "dreams",
            state=self.hermes_home / "dreams" / "state.json",
            journal=self.hermes_home / "dreams" / "dreams.jsonl",
        )
        self.memory_store = memory_store or MemoryStore()
        self.session_db = session_db
        self.auto_promote = auto_promote
        self._ensure_paths()
        self._load_memory_store_if_needed()

    def run(
        self,
        messages: List[Dict[str, Any]],
        *,
        session_id: str = "",
        platform: str = "cli",
        workspace: str = "",
    ) -> Dict[str, Any]:
        recent_user, recent_assistant = self._recent_messages(messages)
        query = self._dream_query(recent_user)
        recall = self._build_recall(query)

        open_loops = self._open_loops(recent_user)
        stable_candidates = self._stable_candidates(recall, query=query, open_loops=open_loops)
        tomorrow_cue = self._tomorrow_cue(stable_candidates, open_loops, recent_assistant)
        hold_items = self._hold_items(recent_user, recent_assistant, stable_candidates, open_loops)
        promotions = self._propose_promotions(stable_candidates)
        applied_promotions = self._apply_promotions(promotions) if self.auto_promote else []

        artifact = {
            "generated_at": _utc_now_iso(),
            "session_id": session_id,
            "platform": platform,
            "workspace": workspace,
            "query": query,
            "stable_candidates": stable_candidates,
            "open_loops": open_loops,
            "tomorrow_cue": tomorrow_cue,
            "do_not_promote_yet": hold_items,
            "promotion_candidates": promotions,
            "applied_promotions": applied_promotions,
        }
        artifact["artifact_hash"] = hashlib.sha256(
            json.dumps(artifact, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:16]

        self._append_journal(artifact)
        self._write_state(artifact)
        return artifact

    def collect_recent_messages(self, *, session_limit: int = 5, per_session_limit: int = 8) -> List[Dict[str, Any]]:
        if self.session_db is None:
            return []
        try:
            sessions = self.session_db.list_sessions_rich(limit=session_limit)
        except Exception:
            return []

        collected: List[Dict[str, Any]] = []
        for session in sessions:
            session_id = session.get("id")
            source = str(session.get("source") or "")
            if not session_id or source.startswith("cron") or source == "tool":
                continue
            try:
                messages = self.session_db.get_messages(session_id)
            except Exception:
                continue
            for msg in messages[-per_session_limit:]:
                role = msg.get("role")
                if role not in {"user", "assistant"}:
                    continue
                text = _content_text(msg.get("content"))
                if not text:
                    continue
                collected.append({"role": role, "content": text})
        return collected[-(session_limit * per_session_limit):]

    def run_nightly(self, *, session_limit: int = 5, per_session_limit: int = 8) -> Dict[str, Any]:
        messages = self.collect_recent_messages(session_limit=session_limit, per_session_limit=per_session_limit)
        return self.run(messages, session_id="nightly-dream", platform="cron", workspace="nightly")

    def render_dream_review(self, artifact: Dict[str, Any]) -> str:
        stable = artifact.get("stable_candidates") or []
        open_loops = artifact.get("open_loops") or []
        tomorrow = artifact.get("tomorrow_cue") or []
        hold = artifact.get("do_not_promote_yet") or []
        promotions = artifact.get("applied_promotions") or []

        def bullets(items: List[str], empty: str) -> str:
            return "\n".join(f"- {item}" for item in items) if items else f"- {empty}"

        body = (
            "## Stable candidates\n\n"
            + bullets(stable, "No strong stable candidates this run.")
            + "\n\n## Open loops\n\n"
            + bullets(open_loops, "No clear open loops captured.")
            + "\n\n## Tomorrow cue\n\n"
            + bullets(tomorrow, "No tomorrow cue generated.")
            + "\n\n## Do not promote yet\n\n"
            + bullets(hold, "No hold items.")
            + "\n\n## Applied promotions\n\n"
            + bullets(promotions, "No automatic promotions this run.")
            + "\n"
        )
        return (
            "---\n"
            "title: Dream Review\n"
            f"updated: {artifact.get('generated_at', _utc_now_iso())}\n"
            "managed_by: hermes-dreaming\n"
            "retention: overwrite-rolling\n"
            "---\n\n"
            "# Dream Review\n\n"
            "> Local-only bounded synthesis of recent Hermes activity. Reviewable, not append-only.\n\n"
            f"- Session ID: {artifact.get('session_id') or 'unknown'}\n"
            f"- Platform: {artifact.get('platform') or 'unknown'}\n"
            f"- Workspace: {artifact.get('workspace') or 'unknown'}\n"
            f"- Query seed: {artifact.get('query') or 'n/a'}\n\n"
            f"{body}"
        )

    def write_obsidian_review(self, path: Path, artifact: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render_dream_review(artifact), encoding="utf-8")

    def _ensure_paths(self) -> None:
        self.paths.root.mkdir(parents=True, exist_ok=True)

    def _load_memory_store_if_needed(self) -> None:
        if self.memory_store and not (self.memory_store.memory_entries or self.memory_store.user_entries):
            try:
                self.memory_store.load_from_disk()
            except Exception:
                pass

    def _recent_messages(self, messages: List[Dict[str, Any]]) -> tuple[List[str], List[str]]:
        user_msgs: List[str] = []
        assistant_msgs: List[str] = []
        for msg in messages[-20:]:
            role = msg.get("role")
            if role not in {"user", "assistant"}:
                continue
            text = _content_text(msg.get("content"))
            if not text or text.startswith("[System:"):
                continue
            if role == "user":
                if _looks_low_signal(text) and not _looks_task_relevant(text):
                    continue
                user_msgs.append(text)
            else:
                if _looks_assistant_meta(text) and not _looks_task_relevant(text):
                    continue
                assistant_msgs.append(text)
        return user_msgs[-5:], assistant_msgs[-4:]

    def _dream_query(self, recent_user: List[str]) -> str:
        if not recent_user:
            return ""
        query = " ".join(recent_user[-3:])
        query = re.sub(r"[^a-zA-Z0-9\s/_-]", " ", query)
        query = re.sub(r"\s+", " ", query).strip()
        return query[:240]

    def _build_recall(self, query: str) -> Dict[str, Any]:
        if not query:
            return {"durable": [], "recent": [], "rendered": ""}
        try:
            search = LocalMemorySearch(memory_store=self.memory_store, session_db=self.session_db)
            return search.build_recall_context(query, mode="full", durable_limit=4, recent_limit=3)
        except Exception:
            return {"durable": [], "recent": [], "rendered": ""}

    def _stable_candidates(self, recall: Dict[str, Any], *, query: str = "", open_loops: Optional[List[str]] = None) -> List[str]:
        ranked: List[tuple[int, str]] = []
        seen = set()
        query_tokens = {tok for tok in re.findall(r"[a-zA-Z0-9_/-]+", query.casefold()) if len(tok) >= 4}
        loop_tokens = {
            tok
            for line in (open_loops or [])
            for tok in re.findall(r"[a-zA-Z0-9_/-]+", line.casefold())
            if len(tok) >= 4
        }
        has_open_loops = bool(open_loops)
        for source_weight, bucket in ((4, (recall.get("recent") or [])), (1, (recall.get("durable") or []))):
            for item in bucket:
                content = _normalize_line(item.get("content", ""), limit=180)
                if not content or _looks_low_signal(content):
                    continue
                canonical = content.casefold()
                if canonical in seen:
                    continue
                seen.add(canonical)
                score = source_weight
                content_tokens = {tok for tok in re.findall(r"[a-zA-Z0-9_/-]+", canonical) if len(tok) >= 4}
                if _looks_task_relevant(content):
                    score += 4
                overlap = len(content_tokens & (query_tokens | loop_tokens))
                score += min(overlap, 4)
                if any(token in canonical for token in ("repo", "workspace", "build command", "login", "auth", "memory", "dream", "cron")):
                    score += 2
                if has_open_loops and canonical.startswith("user prefers"):
                    score -= 4
                if len(content) > 170:
                    score -= 1
                ranked.append((score, content))
        ranked.sort(key=lambda item: (-item[0], len(item[1])))
        return [content for _, content in ranked[:_MAX_STABLE_CANDIDATES]]

    def _open_loops(self, recent_user: List[str]) -> List[str]:
        ranked: List[tuple[int, str]] = []
        seen = set()
        for line in reversed(recent_user):
            lowered = line.casefold()
            if _looks_low_signal(line):
                continue
            score = 0
            if any(v in lowered for v in _WORK_VERBS):
                score += 3
            if any(token in lowered for token in ("need to", "still", "next", "todo", "blocker", "auth", "login", "cron", "dream", "memory")):
                score += 2
            if score <= 0:
                continue
            canonical = line.casefold()
            if canonical in seen:
                continue
            seen.add(canonical)
            ranked.append((score, line))
        ranked.sort(key=lambda item: (-item[0], len(item[1])))
        return [line for _, line in ranked[:_MAX_OPEN_LOOPS]]

    def _tomorrow_cue(self, stable: List[str], open_loops: List[str], recent_assistant: List[str]) -> List[str]:
        cues: List[str] = []
        seen = set()
        assistant_candidates = [item for item in recent_assistant[-2:] if _looks_task_relevant(item) and not _looks_assistant_meta(item)]
        stable_work = [item for item in stable if _looks_task_relevant(item)]
        stable_other = [item for item in stable if item not in stable_work]
        ordered = open_loops + stable_work[:2] + assistant_candidates + stable_other[:1]
        for item in ordered:
            item = _normalize_line(item, limit=160)
            if not item or _looks_low_signal(item):
                continue
            canonical = item.casefold()
            if canonical in seen:
                continue
            seen.add(canonical)
            cues.append(item)
            if len(cues) >= _MAX_TOMORROW_CUES:
                break
        return cues

    def _hold_items(
        self,
        recent_user: List[str],
        recent_assistant: List[str],
        stable: List[str],
        open_loops: List[str],
    ) -> List[str]:
        held: List[str] = []
        seen = {item.casefold() for item in stable + open_loops}
        assistant_candidates = [item for item in recent_assistant[-2:] if _looks_task_relevant(item) and not _looks_assistant_meta(item)]
        for item in recent_user[-2:] + assistant_candidates:
            item = _normalize_line(item, limit=160)
            if not item or _looks_low_signal(item):
                continue
            canonical = item.casefold()
            if canonical in seen:
                continue
            seen.add(canonical)
            held.append(item)
            if len(held) >= _MAX_HOLD_ITEMS:
                break
        return held

    def _propose_promotions(self, stable_candidates: List[str]) -> List[Dict[str, str]]:
        proposals: List[Dict[str, str]] = []
        existing = {
            entry.casefold()
            for entry in (self.memory_store.memory_entries + self.memory_store.user_entries)
        }
        for item in stable_candidates:
            lowered = item.casefold()
            if lowered in existing:
                continue
            target = "user" if any(token in lowered for token in ("prefers", "likes", "wants", "benefits from", "never publish", "concise")) else "memory"
            confidence = "high" if any(token in lowered for token in ("prefers", "always", "never", "build command", "repo", "workspace")) else "medium"
            if confidence != "high":
                continue
            proposals.append({"target": target, "content": item, "confidence": confidence})
            if len(proposals) >= 3:
                break
        return proposals

    def _apply_promotions(self, proposals: List[Dict[str, str]]) -> List[str]:
        applied: List[str] = []
        for proposal in proposals[:3]:
            try:
                result = self.memory_store.add(proposal["target"], proposal["content"])
            except Exception:
                continue
            if result.get("success"):
                applied.append(f"[{proposal['target']}] {proposal['content']}")
        return applied

    def _append_journal(self, artifact: Dict[str, Any]) -> None:
        with self.paths.journal.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(artifact, ensure_ascii=False) + "\n")

    def _write_state(self, artifact: Dict[str, Any]) -> None:
        state = {
            "last_run_at": artifact.get("generated_at"),
            "last_session_id": artifact.get("session_id"),
            "last_artifact_hash": artifact.get("artifact_hash"),
            "run_count": self._read_run_count() + 1,
            "last_promotions": artifact.get("applied_promotions") or [],
        }
        self.paths.state.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _read_run_count(self) -> int:
        try:
            payload = json.loads(self.paths.state.read_text(encoding="utf-8"))
            return int(payload.get("run_count", 0))
        except Exception:
            return 0
