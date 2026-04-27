"""Local Brain memory provider for Hermes.

This provider connects Hermes' pluggable memory lifecycle to the generic
Agent Brain API added to zep-ia/brain.  It is intentionally local-first and
sidecar-style: Hermes keeps its built-in memory as source of truth, while this
provider captures turn/tool/memory events, asks Brain to rank candidates during
caller-authorized idle windows, and injects the ranked candidates on later
turns.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

DEFAULT_ITERATIONS = 90
DEFAULT_TOP_K = 5
DEFAULT_MAX_EVENTS = 500
DEFAULT_HIPPOCAMPUS_ENABLED = True
STATE_VERSION = 1

STATUS_SCHEMA = {
    "name": "brain_status",
    "description": "Show local Brain memory provider status, event counts, and last 90-iteration experiment summary.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

SEARCH_SCHEMA = {
    "name": "brain_search",
    "description": "Search Brain-ranked local memory candidates captured from Hermes events.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search text."},
            "top_k": {"type": "integer", "description": "Max results (default: 5, max: 20)."},
        },
        "required": ["query"],
    },
}

RUN_SCHEMA = {
    "name": "brain_run_experiment",
    "description": "Run the local zep-ia/brain Agent Brain experiment over captured Hermes events using a caller-authorized idle window.",
    "parameters": {
        "type": "object",
        "properties": {
            "iterations": {"type": "integer", "description": "PageRank max iteration budget (default: 90)."},
            "top_k": {"type": "integer", "description": "Long-term candidate count (default: 5)."},
        },
        "required": [],
    },
}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_text(value: Any, *, limit: int = 4000) -> str:
    text = "" if value is None else str(value)
    text = _strip_memory_context(text)
    # Avoid persisting obvious credentials in the local event log. Brain also
    # applies its own boundary, but the provider should minimize raw capture.
    text = re.sub(r"(?i)(api[_-]?key|token|password|secret)\s*[:=]\s*\S+", r"\1=[REDACTED]", text)
    text = re.sub(r"sk-[A-Za-z0-9_-]{16,}", "[REDACTED_API_KEY]", text)
    return text[:limit].strip()


def _strip_memory_context(text: str) -> str:
    """Remove recalled memory blocks so they cannot feed back into Brain state."""
    text = re.sub(r"(?is)<memory-context>.*?</memory-context>", "", text or "")
    text = re.sub(r"(?is)##\s*Brain Memory\b.*?(?=\n##\s|\Z)", "", text)
    return text.strip()


def _is_noise_event(kind: str, role: str, content: str) -> bool:
    """Filter internal scaffolding and known assistant echoes before persistence."""
    lowered = (content or "").lower()
    normalized_role = (role or "").lower()
    if normalized_role in {"system", "developer", "tool"}:
        return True
    internal_markers = (
        "review the conversation above and consider whether a skill",
        "skill save/update",
        "skill-reflection",
        "context compaction",
        "reference only",
    )
    if any(marker in lowered for marker in internal_markers):
        return True
    if normalized_role == "assistant":
        assistant_echo_markers = (
            "brain provider connected as local hermes memory sidecar",
            "skill update summary",
            "hermes skill update summary",
        )
        if any(marker in lowered for marker in assistant_echo_markers):
            return True
    return False


def _contains(text: str, query: str) -> bool:
    terms = [t.lower() for t in re.findall(r"[\w가-힣]+", query or "") if len(t) > 1]
    if not terms:
        return True
    haystack = (text or "").lower()
    return any(term in haystack for term in terms)


def _load_brain_config() -> Dict[str, Any]:
    """Load the optional top-level ``brain:`` config block from config.yaml."""
    try:
        from hermes_cli.config import load_config

        config = load_config()
        brain = config.get("brain", {}) if isinstance(config, dict) else {}
        return dict(brain) if isinstance(brain, dict) else {}
    except Exception:
        return {}


class BrainMemoryProvider(MemoryProvider):
    """Hermes memory provider backed by local zep-ia/brain experiments."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        experiment_runner: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        loaded_config = _load_brain_config()
        if config:
            loaded_config.update(config)
        self.config = loaded_config
        self._experiment_runner = experiment_runner
        self._session_id = ""
        self._platform = ""
        self._agent_identity = "hermes"
        self._hermes_home = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))).expanduser()
        self._state_dir = self._hermes_home / "brain"
        self._state_path = self._state_dir / "state.json"
        self._brain_repo = self._resolve_brain_repo()
        self._state: Dict[str, Any] = self._empty_state()

    @property
    def name(self) -> str:
        return "brain"

    def is_available(self) -> bool:
        repo = self._resolve_brain_repo()
        return repo is not None and (repo / "src" / "index.js").exists()

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id or "default"
        self._platform = str(kwargs.get("platform") or "")
        self._agent_identity = str(kwargs.get("agent_identity") or "hermes") or "hermes"
        hermes_home = kwargs.get("hermes_home") or os.environ.get("HERMES_HOME")
        if hermes_home:
            self._hermes_home = Path(str(hermes_home)).expanduser()
        self._state_dir = self._hermes_home / "brain"
        self._state_path = self._state_dir / "state.json"
        self._brain_repo = self._resolve_brain_repo()
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._state = self._load_state()
        self._save_state()

    def system_prompt_block(self) -> str:
        if not self.is_available():
            return ""
        return (
            "Brain memory sidecar is enabled. It captures Hermes events locally, "
            "runs zep-ia/brain Agent Brain experiments only in caller-authorized "
            "idle windows, and uses a 90-iteration default PageRank budget for "
            "candidate ranking."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._state:
            self._state = self._load_state()
        candidates = self._search_candidates(query, top_k=int(self.config.get("top_k", DEFAULT_TOP_K)))
        if not candidates:
            return ""
        lines = ["## Brain Memory", "Local zep-ia/brain ranked candidates (90-iteration default):"]
        for item in candidates:
            score = item.get("score")
            score_text = f" score={score:.3f}" if isinstance(score, (int, float)) else ""
            content = _safe_text(item.get("content") or item.get("text") or item.get("memory") or "", limit=500)
            if content:
                lines.append(f"- {content}{score_text}")
        return "\n".join(lines) if len(lines) > 2 else ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        # Synchronous local provider: work is already done during idle sync_turn.
        return None

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        sid = session_id or self._session_id or "default"
        self._append_event("message", "user", user_content, sid)
        self._append_event("message", "assistant", assistant_content, sid)
        if bool(self.config.get("auto_consolidate", True)):
            self._run_and_store(iterations=int(self.config.get("iterations", DEFAULT_ITERATIONS)), top_k=int(self.config.get("top_k", DEFAULT_TOP_K)))

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        metadata = metadata or {}
        sid = str(metadata.get("session_id") or self._session_id or "default")
        self._append_event(
            "memory_write",
            target or "memory",
            content,
            sid,
            extra={"action": action, "target": target, "metadata": metadata},
        )
        if bool(self.config.get("auto_consolidate", True)):
            self._run_and_store(iterations=int(self.config.get("iterations", DEFAULT_ITERATIONS)), top_k=int(self.config.get("top_k", DEFAULT_TOP_K)))

    def on_delegation(self, task: str, result: str, *, child_session_id: str = "", **kwargs) -> None:
        sid = self._session_id or "default"
        self._append_event("delegation", "task", task, sid, extra={"child_session_id": child_session_id})
        self._append_event("delegation", "result", result, sid, extra={"child_session_id": child_session_id})

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if messages:
            sid = self._session_id or "default"
            for idx, msg in enumerate(messages[-20:]):
                role = str(msg.get("role") or msg.get("sender") or "message")
                content = msg.get("content") or msg.get("text") or ""
                if content:
                    self._append_event("session_end", role, content, sid, extra={"index": idx})
        self._run_and_store(iterations=int(self.config.get("iterations", DEFAULT_ITERATIONS)), top_k=int(self.config.get("top_k", DEFAULT_TOP_K)))

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [STATUS_SCHEMA, SEARCH_SCHEMA, RUN_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        args = args or {}
        if tool_name == "brain_status":
            return json.dumps(self._status(), ensure_ascii=False)
        if tool_name == "brain_search":
            query = str(args.get("query") or "")
            top_k = max(1, min(20, int(args.get("top_k") or DEFAULT_TOP_K)))
            results = self._search_candidates(query, top_k=top_k)
            return json.dumps({"results": results, "count": len(results)}, ensure_ascii=False)
        if tool_name == "brain_run_experiment":
            iterations = max(1, min(1000, int(args.get("iterations") or DEFAULT_ITERATIONS)))
            top_k = max(1, min(50, int(args.get("top_k") or DEFAULT_TOP_K)))
            result = self._run_and_store(iterations=iterations, top_k=top_k)
            return json.dumps(result, ensure_ascii=False)
        return json.dumps({"error": f"Unknown Brain tool: {tool_name}"}, ensure_ascii=False)

    def shutdown(self) -> None:
        try:
            self._save_state()
        except Exception:
            logger.debug("Failed to save Brain state during shutdown", exc_info=True)

    def _resolve_brain_repo(self) -> Optional[Path]:
        candidates = []
        configured = self.config.get("repo_path") or os.environ.get("HERMES_BRAIN_REPO")
        if configured:
            candidates.append(Path(str(configured)).expanduser())
        candidates.extend([
            Path.home() / ".hermes" / "brain-repos" / "zep-brain",
            Path("/tmp/zep-brain"),
        ])
        for path in candidates:
            if (path / "src" / "index.js").exists():
                return path
        return Path(str(configured)).expanduser() if configured else None

    def _empty_state(self) -> Dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "events": [],
            "long_term_candidates": [],
            "last_experiment": None,
        }

    def _load_state(self) -> Dict[str, Any]:
        if not self._state_path.exists():
            return self._empty_state()
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return self._empty_state()
            data.setdefault("version", STATE_VERSION)
            data.setdefault("events", [])
            data.setdefault("long_term_candidates", [])
            data.setdefault("last_experiment", None)
            data["events"] = self._sanitize_events(data.get("events") or [])
            return data
        except Exception:
            logger.warning("Failed to read Brain state; starting empty", exc_info=True)
            return self._empty_state()

    def _save_state(self) -> None:
        self._state_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._state_path)

    def _sanitize_events(self, events: List[Any]) -> List[Dict[str, Any]]:
        sanitized: List[Dict[str, Any]] = []
        for raw in events:
            if not isinstance(raw, dict):
                continue
            event = dict(raw)
            metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
            role = str(event.get("role") or metadata.get("role") or "")
            kind = str(event.get("type") or event.get("kind") or "event")
            content = _safe_text(event.get("content") or "")
            if not content or _is_noise_event(kind, role, content):
                continue
            event["content"] = content
            sanitized.append(event)
        return sanitized

    def _append_event(self, kind: str, role: str, content: Any, session_id: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        if not self._state:
            self._state = self._load_state()
        content_text = _safe_text(content)
        if not content_text.strip() or _is_noise_event(kind, role, content_text):
            return
        event_id = f"{session_id}:{len(self._state.get('events', [])) + 1}:{kind}:{role}"
        event = {
            "id": event_id,
            "type": kind,
            "role": role,
            "content": content_text,
            "timestamp": _now_ms(),
            "sessionId": session_id,
            "scope": self._platform or "hermes",
        }
        if extra:
            event["metadata"] = extra
        events = list(self._state.get("events") or [])
        events.append(event)
        max_events = max(10, int(self.config.get("max_events", DEFAULT_MAX_EVENTS)))
        self._state["events"] = events[-max_events:]
        self._save_state()

    def _build_payload(self, *, iterations: int, top_k: int) -> Dict[str, Any]:
        events = []
        for raw in self._state.get("events") or []:
            event = {
                "id": str(raw.get("id")),
                "type": str(raw.get("type") or "event"),
                "content": _safe_text(raw.get("content") or ""),
                "timestamp": raw.get("timestamp") or _now_ms(),
                "scope": raw.get("scope") or self._platform or "hermes",
                "metadata": {
                    "role": raw.get("role"),
                    "sessionId": raw.get("sessionId"),
                    **(raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}),
                },
            }
            events.append(event)
        return {
            "agentId": self._agent_identity or "hermes",
            "events": events,
            "toolCalls": [],
            "iterations": iterations,
            "topK": top_k,
            "runtime": {"phase": "idle", "authority": "caller"},
            "hippocampus": {"enabled": bool(self.config.get("hippocampus_enabled", DEFAULT_HIPPOCAMPUS_ENABLED))},
        }

    def _run_and_store(self, *, iterations: int, top_k: int) -> Dict[str, Any]:
        if not self._state.get("events"):
            return {"status": "skipped", "reason": "no_events"}
        payload = self._build_payload(iterations=iterations, top_k=top_k)
        try:
            result = self._run_experiment(payload)
        except Exception as exc:
            logger.warning("Brain experiment failed: %s", exc)
            result = {"status": "error", "error": str(exc)}
        self._state["last_experiment"] = result
        if isinstance(result, dict) and result.get("status") in {"completed", "ok", None}:
            candidates = result.get("longTermCandidates") or result.get("long_term_candidates") or []
            if isinstance(candidates, list):
                self._state["long_term_candidates"] = candidates[:top_k]
        self._save_state()
        return result

    def _run_experiment(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._experiment_runner:
            return self._experiment_runner(payload)
        repo = self._resolve_brain_repo()
        if repo is None or not (repo / "src" / "index.js").exists():
            raise RuntimeError("zep-ia/brain repo not found; set HERMES_BRAIN_REPO to a checkout containing src/index.js")
        script = """
import { runAgentBrainExperiment } from './src/index.js';
const chunks = [];
for await (const chunk of process.stdin) chunks.push(chunk);
const payload = JSON.parse(Buffer.concat(chunks).toString('utf8'));
const result = runAgentBrainExperiment(payload);
process.stdout.write(JSON.stringify(result));
"""
        proc = subprocess.run(
            ["node", "--input-type=module", "-e", script],
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            cwd=str(repo),
            timeout=float(self.config.get("timeout_seconds", 20)),
            check=False,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"node Brain experiment exited {proc.returncode}: {err[:500]}")
        return json.loads(proc.stdout or "{}")

    def _search_candidates(self, query: str, *, top_k: int) -> List[Dict[str, Any]]:
        if not self._state:
            self._state = self._load_state()
        candidates = self._state.get("long_term_candidates") or []
        results = []
        for item in candidates:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or item.get("text") or item.get("memory") or "")
            if _contains(content, query):
                results.append(item)
        if not results and query:
            for event in self._state.get("events") or []:
                content = str(event.get("content") or "")
                if _contains(content, query):
                    results.append({
                        "memoryId": event.get("id"),
                        "content": content,
                        "score": 0.0,
                        "source": "captured_event",
                    })
        return results[: max(1, top_k)]

    def _status(self) -> Dict[str, Any]:
        last = self._state.get("last_experiment") if isinstance(self._state, dict) else None
        last_graph = last.get("graph") if isinstance(last, dict) and isinstance(last.get("graph"), dict) else {}
        last_hippocampus = last.get("hippocampus") if isinstance(last, dict) and isinstance(last.get("hippocampus"), dict) else {}
        return {
            "provider": "brain",
            "brain_repo": str(self._resolve_brain_repo() or ""),
            "brain_repo_available": self.is_available(),
            "events": len(self._state.get("events") or []),
            "long_term_candidates": len(self._state.get("long_term_candidates") or []),
            "default_iterations": DEFAULT_ITERATIONS,
            "hippocampus_enabled": bool(last_hippocampus.get("enabled", self.config.get("hippocampus_enabled", DEFAULT_HIPPOCAMPUS_ENABLED))),
            "last_graph_nodes": len(last_graph.get("nodes") or []),
            "last_graph_edges": len(last_graph.get("edges") or []),
            "last_experiment": last,
            "state_path": str(self._state_path),
        }


def register(ctx) -> None:
    ctx.register_memory_provider(BrainMemoryProvider())
