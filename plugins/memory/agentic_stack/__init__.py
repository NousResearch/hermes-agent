"""Agentic-stack memory provider.

Wires Hermes's MemoryProvider hooks to the portable brain at ``~/.agent/``:
  - sync_turn         -> auto-log each turn as an episodic entry
  - on_delegation     -> log pepessimo-side delegation outcomes
  - on_session_end    -> heuristic rollup entry at higher importance
  - on_pre_compress   -> tell the compressor to preserve memory-curation content
  - system_prompt_block -> inject REVIEW_QUEUE status when pending > 0
  - prefetch          -> ripgrep the semantic tier for relevant entries

Profile tagging is automatic via the ``HERMES_HOME``-driven provenance.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

from plugins.memory.agentic_stack import context as ctx
from plugins.memory.agentic_stack import client as client
from plugins.memory.agentic_stack import reflector

logger = logging.getLogger(__name__)


# -- Tool schemas ----------------------------------------------------------

SEARCH_SCHEMA = {
    "name": "brain_search",
    "description": (
        "Search the portable brain's semantic tier (entities, concepts, "
        "LESSONS.md, DECISIONS.md) for a keyword or phrase. Fast ripgrep-"
        "backed lookup; no LLM reasoning. Use for 'does the brain already "
        "know about X?' type queries before reading files by hand."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Keyword or phrase to search for."},
            "max_results": {
                "type": "integer",
                "description": "Max hits to return (default 5, cap 15).",
            },
        },
        "required": ["query"],
    },
}

REVIEW_QUEUE_SCHEMA = {
    "name": "brain_review_queue",
    "description": (
        "Return pending candidate lessons from the review queue, produced "
        "by the nightly dream cycle. Use to show the user what is waiting "
        "for curation. Does not graduate or reject - that needs his call."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

GRADUATE_SCHEMA = {
    "name": "brain_graduate",
    "description": (
        "Promote a candidate lesson into LESSONS.md with a rationale. "
        "Only call after the user explicitly approves the graduation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "candidate_id": {"type": "string"},
            "rationale": {"type": "string"},
        },
        "required": ["candidate_id", "rationale"],
    },
}

REJECT_SCHEMA = {
    "name": "brain_reject",
    "description": (
        "Reject a candidate lesson with a reason. Only call after the user "
        "explicitly approves the rejection."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "candidate_id": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["candidate_id", "reason"],
    },
}

LOG_SCHEMA = {
    "name": "brain_log",
    "description": (
        "Explicitly log a memory entry, overriding the per-turn heuristic. "
        "Use when you want to force a high-importance entry for something "
        "the auto-log would have skipped, or to record an observation that "
        "does not fit one-turn framing."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "skill": {"type": "string", "description": "Skill or domain bucket."},
            "action": {"type": "string", "description": "What happened."},
            "outcome": {"type": "string", "description": "Result or reflection."},
            "importance": {
                "type": "integer",
                "description": "1-10; 7+ for failures and user corrections.",
            },
            "success": {"type": "boolean"},
            "reflection": {"type": "string"},
        },
        "required": ["skill", "action", "outcome"],
    },
}


class AgenticStackProvider(MemoryProvider):
    """Memory provider wired to the agentic-stack portable brain."""

    def __init__(self) -> None:
        self._brain_path: Path = client.resolve_brain_path(None)
        self._config: Dict[str, Any] = {}
        self._session_id: str = ""
        self._platform: str = ""
        self._agent_context: str = "primary"
        self._hermes_home: str = ""
        self._log_execution = None
        self._on_failure = None
        self._auto_log: bool = True
        self._log_threshold: int = 4
        self._log_delegations: bool = True
        self._session_rollup: bool = True
        self._prefetch_enabled: bool = True
        self._review_surface: bool = True
        self._disabled_for_session: bool = False
        # Camofox health probe state: "healthy"|"unhealthy"|"error"|"disabled"
        self._camofox_status: str = "disabled"
        self._camofox_detail: str = ""

    @property
    def name(self) -> str:
        return "agentic_stack"

    # -- Core lifecycle ----------------------------------------------------

    def is_available(self) -> bool:
        """Plugin is available iff the brain directory + harness hooks exist."""
        try:
            brain = client.resolve_brain_path(self._config.get("brain_path") if self._config else None)
            return (brain / "harness").is_dir() and (brain / "memory").is_dir()
        except Exception:
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._hermes_home = kwargs.get("hermes_home", "")
        self._platform = kwargs.get("platform", "")
        self._agent_context = kwargs.get("agent_context", "primary")

        # Load plugin-specific config from Hermes config.yaml
        try:
            from hermes_cli.config import load_config
            full_cfg = load_config()
            plugin_cfg = full_cfg.get("agentic_stack", {}) or {}
        except Exception:
            plugin_cfg = {}
        self._config = plugin_cfg
        self._brain_path = client.resolve_brain_path(plugin_cfg.get("brain_path"))
        self._auto_log = bool(plugin_cfg.get("auto_log", True))
        self._log_threshold = int(plugin_cfg.get("log_threshold_importance", 4))
        self._log_delegations = bool(plugin_cfg.get("log_delegations", True))
        self._session_rollup = bool(plugin_cfg.get("session_rollup", True))
        self._prefetch_enabled = bool(plugin_cfg.get("prefetch_enabled", True))
        self._review_surface = bool(plugin_cfg.get("review_surface", True))

        # Cron/flush contexts must not write (would corrupt user representations)
        if self._agent_context in ("cron", "flush"):
            self._disabled_for_session = True

        self._log_execution = client.get_log_execution(self._brain_path)
        self._on_failure = client.get_on_failure(self._brain_path)

        if self._log_execution is None:
            logger.warning(
                "agentic_stack: log_execution unavailable; auto-logging disabled "
                "for this session (brain_path=%s)", self._brain_path,
            )
            self._disabled_for_session = True

        # Camofox health probe. Runs even when session writes are disabled
        # (cron/flush) so the status is consistent; but skipping for those
        # contexts keeps the probe cost off the cron path.
        if self._agent_context == "primary":
            self._probe_camofox()

    def _probe_camofox(self) -> None:
        """Probe camofox /health once at session start.

        Sets ``self._camofox_status`` so ``system_prompt_block`` can warn
        the agent when the stealth browser backend is unreachable or
        degraded. Fail-open: any exception sets status to "error" and
        the session continues normally.
        """
        import json as _json
        import os
        import urllib.error
        import urllib.request

        url = os.environ.get("CAMOFOX_URL", "").rstrip("/")
        if not url:
            self._camofox_status = "disabled"
            return
        try:
            req = urllib.request.Request(f"{url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            data = _json.loads(body)
            if (data.get("ok") and data.get("browserConnected")
                    and data.get("browserRunning")):
                self._camofox_status = "healthy"
                self._camofox_detail = ""
            else:
                self._camofox_status = "unhealthy"
                flags = [
                    f"ok={data.get('ok')}",
                    f"browserConnected={data.get('browserConnected')}",
                    f"browserRunning={data.get('browserRunning')}",
                    f"consecutiveFailures={data.get('consecutiveFailures', '?')}",
                ]
                self._camofox_detail = ", ".join(flags)
        except urllib.error.URLError as e:
            self._camofox_status = "error"
            self._camofox_detail = f"unreachable: {str(e.reason)[:80]}"
        except Exception as e:
            self._camofox_status = "error"
            self._camofox_detail = str(e)[:120]

    @staticmethod
    def _read_text_safe(path: Path, max_chars: int) -> str:
        """Read a memory file, truncate if oversized, swallow errors."""
        try:
            if not path.exists():
                return ""
            txt = path.read_text(encoding="utf-8")
        except Exception:
            return ""
        if len(txt) <= max_chars:
            return txt
        # Keep head, mark truncation
        return txt[:max_chars].rstrip() + "\n\n[...truncated for prompt budget...]"

    def system_prompt_block(self) -> str:
        """Inject the portable brain's durable context into every session.

        Closes the distribute arm of the feedback loop: PREFERENCES.md and
        LESSONS.md get read automatically instead of relying on SOUL-level
        guidance. Review queue appears only when the nightly dream cycle
        has staged candidates.
        """
        if self._disabled_for_session:
            return ""
        sections: List[str] = []

        # PREFERENCES.md: small, canonical identity + collab rules. Always inject.
        prefs = self._read_text_safe(
            self._brain_path / "memory" / "personal" / "PREFERENCES.md",
            max_chars=8000,
        )
        if prefs.strip():
            sections.append(
                "## Portable brain: PREFERENCES\n\n"
                "Who the user is, how he wants to collaborate. Canonical source. "
                "If something here conflicts with a specialist's local memory, "
                "this wins for collaboration style.\n\n"
                + prefs.strip()
            )

        # LESSONS.md: curated patterns graduated from the dream cycle.
        lessons = self._read_text_safe(
            self._brain_path / "memory" / "semantic" / "LESSONS.md",
            max_chars=4000,
        )
        if lessons.strip():
            sections.append(
                "## Portable brain: LESSONS\n\n"
                "Distilled patterns. Each entry was graduated with a rationale; "
                "treat them as standing guidance unless the user says otherwise.\n\n"
                + lessons.strip()
            )

        # Review queue: only when the dream cycle staged something.
        if self._review_surface:
            try:
                queue_text = ctx.read_review_queue(self._brain_path)
            except Exception:
                queue_text = ""
            if queue_text:
                head = "\n".join(queue_text.splitlines()[:12])
                sections.append(
                    "## Portable brain: review queue\n\n"
                    "The nightly dream cycle staged candidate lessons; the user's "
                    "judgment is pending. Surface this to him once in your "
                    "first substantive reply if you have not already this "
                    "session. Do not auto-graduate; he decides.\n\n"
                    "```\n" + head + "\n```"
                )

        # Camofox status: only mention when something is wrong. Silent on
        # healthy and disabled (nothing actionable for the agent in those
        # cases). Keeps system prompt quiet when the browser works.
        if self._camofox_status in ("unhealthy", "error"):
            sections.append(
                "## Browser backend: camofox is degraded\n\n"
                "The stealth browser (camofox at $CAMOFOX_URL) reported a "
                f"non-healthy state at session start: {self._camofox_status} "
                f"({self._camofox_detail}). `browser_navigate`, `browser_click`, "
                "and other `browser_*` tools may fail or return stale content. "
                "Surface this to the user if a task needs the browser, and consider "
                "alternatives: `terminal` + `curl` for plain HTTP, `web_search` "
                "for search results, or ask the user to check `curl -s $CAMOFOX_URL/health`."
            )

        if not sections:
            return ""
        return "\n\n".join(sections) + "\n"

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._disabled_for_session or not self._prefetch_enabled or not query:
            return ""
        try:
            res = ctx.search(self._brain_path, query, max_results=3)
        except Exception:
            return ""
        if not res.get("ok") or not res.get("hits"):
            return ""
        lines = ["## Portable brain: relevant entries"]
        for hit in res["hits"][:3]:
            f = hit.get("file", "")
            n = hit.get("line", 0)
            t = (hit.get("text") or "").strip()
            lines.append(f"- {f}:{n} - {t[:140]}")
        return "\n".join(lines) + "\n"

    def sync_turn(
        self, user_content: str, assistant_content: str, *, session_id: str = ""
    ) -> None:
        if self._disabled_for_session or not self._auto_log:
            return
        if self._log_execution is None:
            return
        try:
            importance = reflector.infer_importance(user_content, assistant_content)
            if importance < self._log_threshold:
                return
            success = reflector.infer_success(assistant_content)
            skill = reflector.infer_skill(self._platform, self._agent_context)
            self._log_execution(
                skill_name=skill,
                action=(user_content or "")[:200],
                result=(assistant_content or "")[:500],
                success=success,
                reflection="",
                importance=importance,
                confidence=0.5,
            )
        except Exception as e:
            logger.debug("agentic_stack: sync_turn failed: %s", e)

    def on_delegation(
        self, task: str, result: str, *, child_session_id: str = "", **kwargs
    ) -> None:
        if self._disabled_for_session or not self._log_delegations:
            return
        if self._log_execution is None:
            return
        try:
            self._log_execution(
                skill_name="orchestration",
                action=("delegated: " + (task or ""))[:200],
                result=(result or "")[:500],
                success=reflector.infer_success(result or ""),
                reflection=(f"child_session={child_session_id}" if child_session_id else ""),
                importance=6,
                confidence=0.5,
            )
        except Exception as e:
            logger.debug("agentic_stack: on_delegation failed: %s", e)

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if self._disabled_for_session or not self._session_rollup:
            return
        if self._log_execution is None:
            return
        # Skip trivial sessions
        user_turns = [m for m in messages if m.get("role") == "user"]
        if len(user_turns) < 2:
            return
        try:
            first_q = (
                (user_turns[0].get("content") if isinstance(user_turns[0].get("content"), str)
                 else "")
            )[:180]
            last_asst = next(
                (m for m in reversed(messages) if m.get("role") == "assistant"),
                None,
            )
            last_text = ""
            if last_asst:
                c = last_asst.get("content")
                if isinstance(c, str):
                    last_text = c[:300]
            summary = (
                f"{len(user_turns)}-turn session; first question: '{first_q}'; "
                f"closing reply: '{last_text}'"
            )
            self._log_execution(
                skill_name="session_rollup",
                action="session ended",
                result=summary,
                success=True,
                reflection="heuristic rollup v1",
                importance=7,
                confidence=0.4,
            )
        except Exception as e:
            logger.debug("agentic_stack: on_session_end failed: %s", e)

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        if self._disabled_for_session:
            return ""
        return (
            "Preserve: any candidate-lesson-worthy patterns, user corrections, "
            "graduated decisions, architectural trade-offs discussed, specialist "
            "delegation outcomes. These feed the portable brain's curation loop."
        )

    # -- Tools --------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            SEARCH_SCHEMA,
            REVIEW_QUEUE_SCHEMA,
            GRADUATE_SCHEMA,
            REJECT_SCHEMA,
            LOG_SCHEMA,
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        try:
            if tool_name == "brain_search":
                res = ctx.search(
                    self._brain_path,
                    args.get("query", ""),
                    max_results=int(args.get("max_results", 5)),
                )
                return json.dumps(res)
            if tool_name == "brain_review_queue":
                return json.dumps(ctx.review_queue(self._brain_path))
            if tool_name == "brain_graduate":
                return json.dumps(
                    ctx.graduate(
                        self._brain_path,
                        args.get("candidate_id", ""),
                        args.get("rationale", ""),
                    )
                )
            if tool_name == "brain_reject":
                return json.dumps(
                    ctx.reject(
                        self._brain_path,
                        args.get("candidate_id", ""),
                        args.get("reason", ""),
                    )
                )
            if tool_name == "brain_log":
                return json.dumps(
                    ctx.log_via_cli(
                        self._brain_path,
                        skill=args.get("skill", "manual"),
                        action=args.get("action", ""),
                        outcome=args.get("outcome", ""),
                        importance=int(args.get("importance", 5)),
                        reflection=args.get("reflection", ""),
                        success=bool(args.get("success", True)),
                    )
                )
            return tool_error(f"Unknown tool: {tool_name}")
        except Exception as e:
            logger.exception("agentic_stack: tool %s failed", tool_name)
            return tool_error(f"{tool_name} failed: {e}")

    # -- Config wizard (optional) ------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "brain_path",
                "description": "Path to the agentic-stack brain (.agent directory)",
                "default": "~/.agent",
                "required": False,
            },
            {
                "key": "log_threshold_importance",
                "description": "Minimum inferred importance (1-10) to auto-log a turn",
                "default": 4,
                "required": False,
            },
        ]

    def shutdown(self) -> None:
        return None


# -- Plugin entry point ----------------------------------------------------

def register(ctx) -> None:
    """Register the agentic-stack memory provider."""
    ctx.register_memory_provider(AgenticStackProvider())
