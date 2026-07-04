"""
Hermes Context Governor

Enforces context engineering for long tool-heavy workflows:
- Retain only recent whole tool call/response pairs (default: last 5-8)
- Summarize evicted tool interactions into a compact task-state ledger
- Preserve task-level situational awareness via structured summaries
- Verify completion by independent read-back of target state

Research basis: arXiv:2606.10209v1 — last 5 tool pairs + summarization
beats full-context retention (91.6% vs 71.0% completion, 553K vs 1.48M tokens,
5.79h vs 14.56h).
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolCallRecord:
    """A single tool call/response pair with reducer output."""
    tool_name: str
    tool_args: Dict[str, Any]
    tool_result: str
    timestamp: float
    reducer_summary: str
    raw_result_chars: int
    turn_index: int


RawToolCall = ToolCallRecord  # Alias for clarity in the Context Governor API


@dataclass
class TaskStateLedger:
    """Compact task-state ledger per the Context Governor spec."""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    repo: str = ""
    objective: str = ""
    current_branch: str = ""
    last_verified_commit: str = ""
    known_constraints: List[str] = field(default_factory=list)
    completed_actions: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    stale_or_discarded_assumptions: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    verification_evidence: Dict[str, List[str]] = field(default_factory=lambda: {
        "tests_run": [],
        "files_read": [],
        "external_state_checked": [],
    })
    next_action: str = ""
    next_required_verification: str = ""

    def to_yaml(self) -> str:
        """Export as YAML-like string for context injection."""
        lines = ["task_state:"]
        for key, value in asdict(self).items():
            if isinstance(value, list):
                if value:
                    lines.append(f"  {key}:")
                    for item in value:
                        lines.append(f"    - {item}")
                else:
                    lines.append(f"  {key}: []")
            elif isinstance(value, dict):
                lines.append(f"  {key}:")
                for k, v in value.items():
                    if isinstance(v, list):
                        if v:
                            lines.append(f"    {k}:")
                            for item in v:
                                lines.append(f"      - {item}")
                        else:
                            lines.append(f"    {k}: []")
                    else:
                        lines.append(f"    {k}: {v}")
            else:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)


@dataclass
class ContextSummary:
    """Semantic context summary for evicted tool interactions."""
    objective: str = ""
    verified_current_state: List[str] = field(default_factory=list)
    actions_completed: List[str] = field(default_factory=list)
    files_or_resources_touched: List[str] = field(default_factory=list)
    decisions_made: List[str] = field(default_factory=list)
    stale_or_discarded_assumptions: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    next_required_verification: str = ""

    def to_text(self) -> str:
        """Format as compact human-readable text for context injection."""
        lines = []
        if self.objective:
            lines.append(f"Goal: {self.objective}")
        # Compact facts: combine actions, files, and verified state into one deduplicated list
        facts = []
        facts.extend(f"A:{a}" for a in self.actions_completed)
        facts.extend(f"F:{f}" for f in self.files_or_resources_touched)
        facts.extend(f"V:{v}" for v in self.verified_current_state)
        if facts:
            lines.append("Facts: " + "; ".join(facts))
        if self.blockers:
            lines.append(f"Blockers: {', '.join(self.blockers)}")
        if self.next_required_verification:
            lines.append(f"Next: {self.next_required_verification}")
        return "\n".join(lines) if lines else ""


class TelemetryCollector:
    """Lightweight in-memory telemetry collector for the Context Governor.

    The collector is intentionally dependency-free. Observers can register a
    callback via `set_observer(fn)` to flush counters to an external sink
    (e.g. Prometheus, PostHog, or a debug log).
    """

    def __init__(self) -> None:
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, int] = {}
        self._observer: Optional[Callable[[str, int, str], None]] = None

    def set_observer(self, fn: Optional[Callable[[str, int, str], None]]) -> None:
        """Register a callback(name, value, kind) for each metric update."""
        self._observer = fn

    def inc(self, name: str, delta: int = 1) -> None:
        self._counters[name] = self._counters.get(name, 0) + delta
        if self._observer:
            self._observer(name, self._counters[name], "counter")

    def gauge(self, name: str, value: int) -> None:
        self._gauges[name] = value
        if self._observer:
            self._observer(name, value, "gauge")

    def snapshot(self) -> Dict[str, Dict[str, int]]:
        return {"counters": dict(self._counters), "gauges": dict(self._gauges)}

    def log_metrics(self) -> None:
        """Emit a single log line with the current telemetry snapshot."""
        snap = self.snapshot()
        if not snap["counters"] and not snap["gauges"]:
            return
        parts = ["Context Governor telemetry:"]
        for name, value in sorted(snap["counters"].items()):
            parts.append(f"{name}={value}")
        for name, value in sorted(snap["gauges"].items()):
            parts.append(f"{name}={value}")
        logger.info(" ".join(parts))

    def write_prometheus_file(self, path: Path) -> None:
        """Write current metrics in Prometheus text exposition format to a file."""
        lines: List[str] = []
        lines.append("# HELP context_governor_counters Context Governor counter telemetry")
        lines.append("# TYPE context_governor_counters counter")
        for name, value in sorted(self._counters.items()):
            safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
            lines.append(f"context_governor_{{name=\"{safe_name}\"}} {value}")
        lines.append("# HELP context_governor_gauges Context Governor gauge telemetry")
        lines.append("# TYPE context_governor_gauges gauge")
        for name, value in sorted(self._gauges.items()):
            safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
            lines.append(f"context_governor_{{name=\"{safe_name}\"}} {value}")
        path.write_text("\n".join(lines))


class ToolOutputReducer:
    """Reduces tool outputs to structured summaries per the Context Governor spec."""

    @staticmethod
    def reduce_github_pr_list(content: str) -> str:
        """Reduce GitHub PR list to key fields."""
        # PR number, title, branch, draft state, mergeability, checks, labels
        return "[GitHub PR list] PRs summarized: number, title, branch, draft, checks, labels"

    @staticmethod
    def reduce_ci_logs(content: str) -> str:
        """Reduce CI logs to failing job, command, first error, implicated files."""
        return "[CI logs] Failing job, failing command, first error, changed files implicated"

    @staticmethod
    def reduce_deploy_logs(content: str) -> str:
        """Reduce deploy logs to environment, version, commit SHA, health check, rollback marker."""
        return "[Deploy logs] Environment, version, commit SHA, health check, rollback marker"

    @staticmethod
    def reduce_obsidian_search(content: str) -> str:
        """Reduce Obsidian search to note path, heading, excerpt, last modified."""
        return "[Obsidian search] Note path, heading, relevant excerpt, last modified"

    @staticmethod
    def reduce_browser_screenshots(content: str) -> str:
        """Reduce browser screenshots to URL, viewport, visible issues, layout defects."""
        return "[Browser] URL, viewport, visible issues, measured layout defects"

    @staticmethod
    def reduce_ses_email(content: str) -> str:
        """Reduce SES/email test to provider, recipient class, send result, message ID if safe."""
        return "[SES/Email] Provider, recipient class, send result, message ID (if safe)"

    @staticmethod
    def reduce_git_diff(content: str) -> str:
        """Reduce git diff to files changed, intent, risky hunks, generated files excluded."""
        return "[git diff] Files changed, intent, risky hunks, generated files excluded"

    @staticmethod
    def reduce_terminal(content: str, args: Dict) -> str:
        """Reduce terminal output to a short command signature and exit code."""
        cmd = args.get("command", "")
        # Keep only executable + subcommand (drop args/flags)
        parts = []
        for part in cmd.split():
            if part.startswith("-"):
                continue
            parts.append(part)
            if len(parts) >= 3:
                break
        short_cmd = " ".join(parts) or "terminal"
        import re
        exit_match = re.search(r'"exit_code"\s*:\s*(-?\d+)', content)
        exit_code = exit_match.group(1) if exit_match else "?"
        return f"[terminal] {short_cmd} -> exit {exit_code}"

    @staticmethod
    def reduce_read_file(content: str, args: Dict) -> str:
        """Reduce read_file to path, offset, char count."""
        path = args.get("path", "?")
        offset = args.get("offset", 1)
        return f"[read_file] read {path} from line {offset} ({len(content):,} chars)"

    @staticmethod
    def reduce_write_file(content: str, args: Dict) -> str:
        """Reduce write_file to path, line count."""
        path = args.get("path", "?")
        written_lines = args.get("content", "").count("\n") + 1 if args.get("content") else "?"
        return f"[write_file] wrote to {path} ({written_lines} lines)"

    @staticmethod
    def reduce_search_files(content: str, args: Dict) -> str:
        """Reduce search_files to pattern, path, target, match count."""
        pattern = args.get("pattern", "?")
        path = args.get("path", ".")
        target = args.get("target", "content")
        import re
        match_count = re.search(r'"total_count"\s*:\s*(\d+)', content)
        count = match_count.group(1) if match_count else "?"
        return f"[search_files] {target} search for '{pattern}' in {path} -> {count} matches"

    @staticmethod
    def reduce_patch(content: str, args: Dict) -> str:
        """Reduce patch to path, mode, result size."""
        path = args.get("path", "?")
        mode = args.get("mode", "replace")
        return f"[patch] {mode} in {path} ({len(content):,} chars result)"

    @staticmethod
    def reduce_web_search(content: str, args: Dict) -> str:
        """Reduce web_search to query."""
        query = args.get("query", "?")
        return f"[web_search] query='{query}' ({len(content):,} chars result)"

    @staticmethod
    def reduce_web_extract(content: str, args: Dict) -> str:
        """Reduce web_extract to URL and char count."""
        urls = args.get("urls", [])
        url_desc = urls[0] if isinstance(urls, list) and urls else "?"
        if isinstance(urls, list) and len(urls) > 1:
            url_desc += f" (+{len(urls) - 1} more)"
        return f"[web_extract] {url_desc} ({len(content):,} chars)"

    @staticmethod
    def reduce_delegate_task(content: str, args: Dict) -> str:
        """Reduce delegate_task to goal summary."""
        goal = args.get("goal", "")
        if len(goal) > 60:
            goal = goal[:57] + "..."
        return f"[delegate_task] '{goal}' ({len(content):,} chars result)"

    @staticmethod
    def reduce_execute_code(content: str, args: Dict) -> str:
        """Reduce execute_code to code preview and line count."""
        code_preview = (args.get("code") or "")[:60].replace("\n", " ")
        if len(args.get("code", "")) > 60:
            code_preview += "..."
        line_count = content.count("\n") + 1 if content.strip() else 0
        return f"[execute_code] `{code_preview}` ({line_count} lines output)"

    @classmethod
    def reduce(cls, tool_name: str, content: str, args: Dict) -> str:
        """Dispatch to the appropriate reducer."""
        reducer_map = {
            "terminal": cls.reduce_terminal,
            "read_file": cls.reduce_read_file,
            "write_file": cls.reduce_write_file,
            "search_files": cls.reduce_search_files,
            "patch": cls.reduce_patch,
            "web_search": cls.reduce_web_search,
            "web_extract": cls.reduce_web_extract,
            "delegate_task": cls.reduce_delegate_task,
            "execute_code": cls.reduce_execute_code,
            "browser_navigate": lambda c, a: f"[browser_navigate] {a.get('url', '')} ({len(c):,} chars)",
            "browser_click": lambda c, a: f"[browser_click] ref={a.get('ref', '')} ({len(c):,} chars)",
            "browser_snapshot": lambda c, a: f"[browser_snapshot] ({len(c):,} chars)",
            "browser_type": lambda c, a: f"[browser_type] ({len(c):,} chars)",
            "browser_scroll": lambda c, a: f"[browser_scroll] ({len(c):,} chars)",
            "browser_vision": lambda c, a: f"[browser_vision] {a.get('question', '')[:50]} ({len(c):,} chars)",
            "vision_analyze": lambda c, a: f"[vision_analyze] '{a.get('question', '')[:50]}' ({len(c):,} chars)",
            "memory": lambda c, a: f"[memory] {a.get('action', '?')} on {a.get('target', '?')}",
            "todo": lambda c, a: "[todo] updated task list",
            "clarify": lambda c, a: "[clarify] asked user a question",
            "text_to_speech": lambda c, a: f"[text_to_speech] generated audio ({len(c):,} chars)",
            "cronjob": lambda c, a: f"[cronjob] {a.get('action', '?')}",
            "process": lambda c, a: f"[process] {a.get('action', '?')} session={a.get('session_id', '?')}",
            "skill_view": lambda c, a: f"[skill_view] name={a.get('name', '?')} ({len(c):,} chars)",
            "skills_list": lambda c, a: f"[skills_list] ({len(c):,} chars)",
            "skill_manage": lambda c, a: f"[skill_manage] action={a.get('action', '?')} ({len(c):,} chars)",
            # Third-party / MCP integrations (common tools)
            "mcp_linear_create_issue": lambda c, a: f"[Linear] create issue team={a.get('teamId', a.get('teamName', '?'))} title={str(a.get('title', ''))[:50]} ({len(c):,} chars)",
            "mcp_linear_get_issues": lambda c, a: f"[Linear] get issues {a.get('filter', '')[:60]} ({len(c):,} chars)",
            "mcp_linear_update_issue": lambda c, a: f"[Linear] update issue {a.get('issueId', '?')} ({len(c):,} chars)",
            "mcp_linear_get_teams": lambda c, a: f"[Linear] get teams ({len(c):,} chars)",
            "mcp_sentry_get_issue": lambda c, a: f"[Sentry] issue {a.get('issueId', a.get('issue_id', '?'))} ({len(c):,} chars)",
            "mcp_sentry_list_issues": lambda c, a: f"[Sentry] list issues project={a.get('projectSlug', a.get('project_slug', '?'))} ({len(c):,} chars)",
            "mcp_sentry_create_release": lambda c, a: f"[Sentry] create release {a.get('version', '')[:40]} ({len(c):,} chars)",
            "mcp_posthog_query": lambda c, a: f"[PostHog] query {str(a.get('query', ''))[:60]} ({len(c):,} chars)",
            "mcp_posthog_get_insight": lambda c, a: f"[PostHog] insight {a.get('insightId', a.get('insight_id', '?'))} ({len(c):,} chars)",
            "mcp_posthog_list_insights": lambda c, a: f"[PostHog] list insights ({len(c):,} chars)",
            "mcp_cloudflare_list_zones": lambda c, a: f"[Cloudflare] list zones ({len(c):,} chars)",
            "mcp_cloudflare_dns_records": lambda c, a: f"[Cloudflare] DNS records zone={a.get('zoneId', a.get('zone_id', '?'))} ({len(c):,} chars)",
            "mcp_cloudflare_purge_cache": lambda c, a: f"[Cloudflare] purge cache zone={a.get('zoneId', a.get('zone_id', '?'))} ({len(c):,} chars)",
            "mcp_google_workspace_gmail_search": lambda c, a: f"[Gmail] search {str(a.get('query', ''))[:50]} ({len(c):,} chars)",
            "mcp_google_workspace_gmail_get": lambda c, a: f"[Gmail] get message {a.get('message_id', '?')} ({len(c):,} chars)",
            "mcp_google_workspace_calendar_list": lambda c, a: f"[Calendar] list events ({len(c):,} chars)",
            "mcp_google_workspace_drive_search": lambda c, a: f"[Drive] search {str(a.get('query', ''))[:50]} ({len(c):,} chars)",
            "mcp_obsidian_search": lambda c, a: f"[Obsidian] search {str(a.get('query', ''))[:50]} ({len(c):,} chars)",
            "mcp_obsidian_read": lambda c, a: f"[Obsidian] read {a.get('path', '?')} ({len(c):,} chars)",
            "mcp_1password_get_item": lambda c, a: f"[1Password] get item {a.get('vault', '?')}/{a.get('item', '?')} ({len(c):,} chars)",
            "mcp_stripe_list_customers": lambda c, a: f"[Stripe] list customers ({len(c):,} chars)",
            "mcp_stripe_create_invoice": lambda c, a: f"[Stripe] create invoice customer={a.get('customer', '?')} ({len(c):,} chars)",
        }

        if tool_name in reducer_map:
            return reducer_map[tool_name](content, args)

        # Generic fallback
        first_arg = ""
        for k, v in list(args.items())[:2]:
            sv = str(v)[:40]
            first_arg += f" {k}={sv}"
        return f"[{tool_name}]{first_arg} ({len(content):,} chars result)"


class ContextGovernor:
    """
    Context Governor for Hermes agent runs.

    Manages:
    - Task-state ledger (durable across session)
    - Recent raw tool call/response window (configurable N)
    - Summarized older tool interactions
    - Independent verification gates
    """

    def __init__(
        self,
        raw_tool_window: int = 5,
        summary_window: int = 3,
        max_state_summary_words: int = 700,
        max_raw_log_lines: int = 200,
        verification_required: bool = True,
    ):
        self.raw_tool_window = raw_tool_window
        self.summary_window = summary_window
        self.max_state_summary_words = max_state_summary_words
        self.max_raw_log_lines = max_raw_log_lines
        self.verification_required = verification_required
        self.ledger = TaskStateLedger()
        self.raw_tool_calls: List[RawToolCall] = []
        self.summarized_window: List[ContextSummary] = []
        self._session_id: Optional[str] = None
        self.telemetry = TelemetryCollector()

    def on_session_start(self, session_id: str, **kwargs) -> None:
        """Called when a new conversation session begins."""
        self._session_id = session_id
        loaded = self._load_state(session_id)
        self.telemetry.inc("context_governor_load_hits_total" if loaded else "context_governor_load_misses_total")

    def on_session_reset(self) -> None:
        """Reset all per-session state for /new or /reset."""
        self.ledger = TaskStateLedger()
        self.raw_tool_calls = []
        self.summarized_window = []
        self._session_id = None
        self.telemetry.inc("context_governor_reset_total")

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """Persist durable state at a real session boundary, then clear ephemeral state."""
        try:
            self._persist_state(session_id)
            self.telemetry.inc("context_governor_persist_total")
        except Exception as exc:
            self.telemetry.inc("context_governor_persist_failures_total")
            logger.debug("Context Governor persist on session end failed: %s", exc)
        self.telemetry.log_metrics()
        self.raw_tool_calls = []
        self.summarized_window = []

    def _persist_state(self, session_id: str) -> None:
        """Persist the task-state ledger and full internal state to the session DB."""
        try:
            from hermes_state import SessionDB
            from hermes_constants import get_hermes_home
            import json

            records = []
            for r in self.raw_tool_calls:
                records.append({
                    "tool_name": r.tool_name,
                    "tool_args": r.tool_args,
                    "tool_result": r.tool_result,
                    "reducer_summary": r.reducer_summary,
                    "timestamp": r.timestamp,
                    "raw_result_chars": r.raw_result_chars,
                    "turn_index": r.turn_index,
                })

            db = SessionDB(get_hermes_home() / "sessions.db")
            state = {
                "ledger": asdict(self.ledger),
                "raw_tool_calls": records,
                "summarized_window": [asdict(s) for s in self.summarized_window[-(self.summary_window * 2):]],
                "session_id": session_id,
                "persisted_at": time.time(),
            }
            db.update_context_governor_state(session_id, json.dumps(state))
            db.close()
        except Exception as exc:
            logger.debug("Context Governor _persist_state failed: %s", exc)

    def _load_state(self, session_id: str) -> bool:
        """Load the task-state ledger and full internal state from the session DB."""
        try:
            from hermes_state import SessionDB
            from hermes_constants import get_hermes_home
            import json

            db = SessionDB(get_hermes_home() / "sessions.db")
            state_json = db.get_context_governor_state(session_id)
            db.close()
            if state_json:
                state = json.loads(state_json)
                ledger_dict = state.get("ledger", {})
                if ledger_dict:
                    self.ledger = TaskStateLedger(**ledger_dict)
                self.summarized_window = [ContextSummary(**s) for s in state.get("summarized_window", [])]
                for r in state.get("raw_tool_calls", []):
                    self.raw_tool_calls.append(RawToolCall(
                        tool_name=r["tool_name"],
                        tool_args=r["tool_args"],
                        tool_result=r["tool_result"],
                        reducer_summary=r.get("reducer_summary", ""),
                        timestamp=r.get("timestamp", 0.0),
                        raw_result_chars=r.get("raw_result_chars", 0),
                        turn_index=r.get("turn_index", 0),
                    ))
                return True
        except Exception as exc:
            logger.debug("Context Governor _load_state failed: %s", exc)
        return False

    def record_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_result: str,
        turn_index: int,
    ) -> ToolCallRecord:
        """Record a tool call and its reduced summary."""
        reducer_summary = ToolOutputReducer.reduce(tool_name, tool_result, tool_args)
        record = ToolCallRecord(
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result,
            timestamp=time.time(),
            reducer_summary=reducer_summary,
            raw_result_chars=len(tool_result),
            turn_index=turn_index,
        )
        self.raw_tool_calls.append(record)
        self.telemetry.inc("context_governor_tool_calls_recorded_total")

        # Prune old raw tool calls, move to summarized window
        if len(self.raw_tool_calls) > self.raw_tool_window:
            evicted = self.raw_tool_calls[:-self.raw_tool_window]
            self.raw_tool_calls = self.raw_tool_calls[-self.raw_tool_window:]
            summary = self._summarize_evicted(evicted)
            self.summarized_window.append(summary)

            # Compact: merge adjacent summaries with the same objective
            self._compact_summaries()

            # Prune summarized window too
            if len(self.summarized_window) > self.summary_window:
                self.summarized_window = self.summarized_window[-self.summary_window:]

        return record

    def _compact_summaries(self) -> None:
        """Merge adjacent summaries that share the same objective to reduce repetition."""
        if not self.summarized_window:
            return
        compacted = [self.summarized_window[0]]
        for s in self.summarized_window[1:]:
            prev = compacted[-1]
            if s.objective == prev.objective:
                # Merge into prev, keeping newest values when duplicated
                prev.actions_completed = list(dict.fromkeys(s.actions_completed + prev.actions_completed))[-5:]
                prev.files_or_resources_touched = list(dict.fromkeys(s.files_or_resources_touched + prev.files_or_resources_touched))[-10:]
                prev.verified_current_state = list(dict.fromkeys(s.verified_current_state + prev.verified_current_state))[-3:]
                if s.blockers:
                    prev.blockers = list(dict.fromkeys(prev.blockers + s.blockers))[-5:]
                if s.next_required_verification:
                    prev.next_required_verification = s.next_required_verification
            else:
                compacted.append(s)
        self.summarized_window = compacted

    def _summarize_evicted(self, evicted: List[RawToolCall]) -> ContextSummary:
        """Build a compact semantic summary from evicted raw tool calls.

        Keeps only recent, non-redundant actions and state facts so that
        repetitive log/terminal dumps do not bloat the context.
        """
        files_touched: List[str] = []
        actions: List[str] = []
        state_facts: List[str] = []
        terminal_runs = 0
        log_checks: Dict[str, int] = {}
        read_files: Dict[str, int] = {}

        for r in evicted:
            if r.tool_name in {"read_file", "write_file", "patch"}:
                path = r.tool_args.get("path", "?")
                if path and path not in files_touched:
                    files_touched.append(path)
                if r.tool_name == "read_file":
                    read_files[path] = read_files.get(path, 0) + 1

            if r.tool_name == "patch":
                actions.append(f"applied patch to {r.tool_args.get('path', '?')}")
            elif r.tool_name == "write_file":
                actions.append(f"wrote {r.tool_args.get('path', '?')}")
            elif r.tool_name == "execute_code":
                actions.append(f"ran code: {r.reducer_summary}")
            elif r.tool_name == "search_files":
                actions.append(f"searched: {r.reducer_summary}")
            elif r.tool_name == "terminal":
                terminal_runs += 1
                cmd = r.reducer_summary
                # Collapse repeated docker logs / docker inspect checks into a count
                if "docker logs" in cmd or "docker inspect" in cmd or "docker ps" in cmd:
                    log_checks[cmd] = log_checks.get(cmd, 0) + 1
                else:
                    actions.append(f"shell: {cmd}")
            if r.tool_name in {"execute_code", "terminal"} and "exit" in r.reducer_summary.lower():
                state_facts.append(r.reducer_summary)

        # Collapse repeated log checks
        for cmd, count in log_checks.items():
            if count > 1:
                actions.append(f"shell: {cmd} (×{count})")
            else:
                actions.append(f"shell: {cmd}")

        # Collapse repeated read_file of the same path
        for path, count in read_files.items():
            if count > 1:
                # Prefer the latest explicit mention of this path
                for i, a in enumerate(actions):
                    if a.startswith(f"read_file: {path}") or a.endswith(str(path)):
                        actions[i] = f"read {path} (×{count})"
                        break

        # Cap to avoid summary bloat; keep most recent
        max_actions = 5
        max_state = 3
        max_files = 10
        actions = actions[-max_actions:]
        state_facts = state_facts[-max_state:]
        files_touched = files_touched[-max_files:]

        new_summary = ContextSummary(
            objective=self.ledger.objective,
            actions_completed=actions,
            files_or_resources_touched=files_touched,
            verified_current_state=state_facts,
        )

        # Merge with the most recent summary if it shares the same objective to reduce repetition
        if self.summarized_window and self.summarized_window[-1].objective == new_summary.objective:
            prev = self.summarized_window[-1]
            for f in prev.files_or_resources_touched[-max_files:]:
                if f not in new_summary.files_or_resources_touched:
                    new_summary.files_or_resources_touched.insert(0, f)
            for a in prev.actions_completed[-max_actions:]:
                if a not in new_summary.actions_completed:
                    new_summary.actions_completed.insert(0, a)
            for s in prev.verified_current_state[-max_state:]:
                if s not in new_summary.verified_current_state:
                    new_summary.verified_current_state.insert(0, s)
            # Re-apply caps after merge
            new_summary.files_or_resources_touched = new_summary.files_or_resources_touched[-max_files:]
            new_summary.actions_completed = new_summary.actions_completed[-max_actions:]
            new_summary.verified_current_state = new_summary.verified_current_state[-max_state:]
            self.summarized_window[-1] = new_summary
            return new_summary

        return new_summary

    def update_ledger(
        self,
        *,
        repo: Optional[str] = None,
        objective: Optional[str] = None,
        current_branch: Optional[str] = None,
        last_verified_commit: Optional[str] = None,
        known_constraints: Optional[List[str]] = None,
        completed_actions: Optional[List[str]] = None,
        open_questions: Optional[List[str]] = None,
        blockers: Optional[List[str]] = None,
        stale_or_discarded_assumptions: Optional[List[str]] = None,
        risks: Optional[List[str]] = None,
        verification_evidence: Optional[Dict[str, List[str]]] = None,
        next_action: Optional[str] = None,
        next_required_verification: Optional[str] = None,
    ) -> None:
        """Update the task-state ledger."""
        if repo is not None:
            self.ledger.repo = repo
        if objective is not None:
            self.ledger.objective = objective
        if current_branch is not None:
            self.ledger.current_branch = current_branch
        if last_verified_commit is not None:
            self.ledger.last_verified_commit = last_verified_commit
        if known_constraints is not None:
            self.ledger.known_constraints = known_constraints
        if completed_actions is not None:
            self.ledger.completed_actions = completed_actions
        if open_questions is not None:
            self.ledger.open_questions = open_questions
        if blockers is not None:
            self.ledger.blockers = blockers
        if stale_or_discarded_assumptions is not None:
            self.ledger.stale_or_discarded_assumptions = stale_or_discarded_assumptions
        if risks is not None:
            self.ledger.risks = risks
        if verification_evidence is not None:
            self.ledger.verification_evidence.update(verification_evidence)
        if next_action is not None:
            self.ledger.next_action = next_action
        if next_required_verification is not None:
            self.ledger.next_required_verification = next_required_verification

    def build_context_summary(self) -> ContextSummary:
        """Build a ContextSummary from current state."""
        summary = ContextSummary()
        summary.objective = self.ledger.objective
        summary.verified_current_state = [f"Branch: {self.ledger.current_branch}"] if self.ledger.current_branch else []
        if self.ledger.last_verified_commit:
            summary.verified_current_state.append(f"Last verified commit: {self.ledger.last_verified_commit[:8]}")
        summary.actions_completed = self.ledger.completed_actions.copy()
        summary.files_or_resources_touched = [
            f"{r.tool_name}: {list(r.tool_args.keys())[:2]}"
            for r in self.raw_tool_calls + [ToolCallRecord(**s) for s in self.summarized_window[-self.summary_window:]]
        ][:10]
        summary.decisions_made = []  # Could be extracted from ledger
        summary.stale_or_discarded_assumptions = self.ledger.stale_or_discarded_assumptions.copy()
        summary.blockers = self.ledger.blockers.copy()
        summary.risks = self.ledger.risks.copy()
        summary.next_required_verification = self.ledger.next_required_verification
        return summary

    def get_context_for_model(self) -> str:
        """
        Build the context block to inject into the model call.

        Returns:
            Formatted context string with:
            - Task-state ledger
            - Recent raw tool calls (last N)
            - Summarized older tool calls
        """
        parts = []

        # 1. Task-state ledger
        if self.ledger.objective or self.ledger.completed_actions:
            parts.append(self.ledger.to_yaml())

        # 2. Summarized older tool interactions
        if self.summarized_window:
            for summary in self.summarized_window[-self.summary_window:]:
                parts.append(summary.to_text())

        # 3. Recent raw tool calls
        if self.raw_tool_calls:
            raw_lines = ["RECENT RAW TOOL CALLS (last {}):".format(len(self.raw_tool_calls))]
            for record in self.raw_tool_calls:
                raw_lines.append(f"- {record.reducer_summary}")
            parts.append("\n".join(raw_lines))

        # 4. Next required verification
        if self.ledger.next_required_verification:
            parts.append(f"NEXT REQUIRED VERIFICATION: {self.ledger.next_required_verification}")

        context = "\n\n".join(parts) if parts else ""
        self.telemetry.gauge("context_governor_prompt_chars", len(context))
        return context

    def verify_completion(self, task_type: str) -> Dict[str, Any]:
        """
        Independent read-back verification per Context Governor spec.

        Returns dict with verification results and any failures.
        """
        verifiers = {
            "pr_merged": self._verify_pr_merged,
            "staging_deploy": self._verify_staging_deploy,
            "production_release": self._verify_production_release,
            "obsidian_doc_update": self._verify_obsidian_doc,
            "admin_email_notification": self._verify_admin_email,
            "ui_change": self._verify_ui_change,
            "cleanup_refactor": self._verify_cleanup_refactor,
        }

        verifier = verifiers.get(task_type)
        if not verifier:
            return {"verified": False, "error": f"Unknown task type: {task_type}"}

        return verifier()

    def _verify_pr_merged(self) -> Dict[str, Any]:
        # This would call gh pr view, check branch state, checks, merge commit
        return {"verified": False, "note": "Implement gh pr view verification"}

    def _verify_staging_deploy(self) -> Dict[str, Any]:
        return {"verified": False, "note": "Implement staging deploy verification"}

    def _verify_production_release(self) -> Dict[str, Any]:
        return {"verified": False, "note": "Implement production release verification"}

    def _verify_obsidian_doc(self) -> Dict[str, Any]:
        return {"verified": False, "note": "Implement Obsidian doc verification"}

    def _verify_admin_email(self) -> Dict[str, Any]:
        return {"verified": False, "note": "Implement admin email verification"}

    def _verify_ui_change(self) -> Dict[str, Any]:
        return {"verified": False, "note": "Implement UI change verification"}

    def _verify_cleanup_refactor(self) -> Dict[str, Any]:
        return {"verified": False, "note": "Implement cleanup/refactor verification"}


# Global context governor instance for the session
_global_governor: Optional[ContextGovernor] = None


def get_context_governor() -> ContextGovernor:
    """Get or create the global context governor."""
    global _global_governor
    if _global_governor is None:
        # Read config from Hermes config.yaml
        try:
            import yaml
            from pathlib import Path
            config_path = Path.home() / ".hermes" / "config.yaml"
            if config_path.exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f) or {}
                cg = config.get("context_governor", {})
                _global_governor = ContextGovernor(
                    raw_tool_window=cg.get("raw_tool_window", 5),
                    summary_window=cg.get("summary_window", 3),
                    max_state_summary_words=cg.get("max_state_summary_words", 700),
                    max_raw_log_lines=cg.get("max_raw_log_lines", 200),
                    verification_required=cg.get("verification_required", True),
                )
            else:
                _global_governor = ContextGovernor()
        except Exception:
            _global_governor = ContextGovernor()
    return _global_governor


def reset_context_governor() -> None:
    """Reset the global context governor (e.g., on /new)."""
    global _global_governor
    _global_governor = None