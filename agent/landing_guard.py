"""No-landing streak guard and act-now directive pin (bounded progress tracking).

The identity-based loop guards in ``agent.tool_guardrails`` only see a call that repeats
with identical arguments. Varied exploration (forty different ``python -c`` probes, the same
file re-read at shifted offsets) never repeats exactly, so a turn could run 90+ calls without
changing anything. This module counts calls since the last *landing* action (a file edit,
commit, PR, message, task update, service change) and escalates:

* ``notice_after`` calls: stop exploring, state what is known, land or report.
* ``final_after`` calls: the next action must land something or be the final reply.
* ``halt_after`` calls: stop the turn (attended platforms with hard stops only).

Two side notices ride the same observation, once per turn each: after ``probe_notice_after``
inline interpreter probes (write one scratch script instead), and when a ``read_file`` mostly
repeats an earlier read of a file nothing has changed since.

When the user's message is an explicit act-now directive ("do a PR right now", "just fix it"),
the thresholds tighten and a turn note pins the directive from the first call.

Pure and side-effect free: callers feed observations and apply the returned notices.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional

# Tools whose successful call changes something the user can see.
LANDING_TOOL_NAMES = frozenset({
    "write_file", "patch", "send_message", "delegate_task", "cronjob", "cronjob_manage",
    "kanban_complete", "kanban_block", "kanban_comment", "kanban_create", "kanban_request_review",
    "browser_click", "browser_type", "browser_press", "skill_manage",
})

# Neither progress nor exploration: polling a running job, bookkeeping, asking the user.
NEUTRAL_TOOL_NAMES = frozenset({"process", "process_manage", "todo", "todo_list", "clarify"})
_NEUTRAL_SUFFIXES = ("_get_result", "_poll")

# MCP tools are named mcp_<server>_<verb...>; these verbs mutate.
_MCP_MUTATING = re.compile(
    r"^mcp_.*_(set|create|update|delete|remove|write|call|send|add|apply|restart|reload|manage|"
    r"bulk|upload|publish|post|put|patch|merge|comment|close)(_|$)", re.I)

# Terminal commands that change state. Deliberately broad: a false "landing" only delays a
# notice, a false "no landing" nags someone who is working.
_LANDING_COMMAND = re.compile(
    r"(\bgit\s+(commit|push|merge|rebase|apply|am|cherry-pick|tag)\b"
    r"|\bgh\s+(pr|issue|release|repo|api)\s+(create|merge|comment|edit|close|reopen|review|-X)\b"
    r"|\bgh\s+api\b.*\s-X\s*(POST|PUT|PATCH|DELETE)"
    r"|\bsed\s+(-[a-zA-Z]*i|--in-place)"
    r"|\btee\s+(?!/tmp/|/dev/)|(^|[^0-9&>])>>?\s*(?!/tmp/|/dev/|&)[^\s&|>]"
    r"|\b(mv|rm|chmod|chown|install|truncate)\s"
    r"|\bsystemctl\s+(--user\s+)?(restart|start|stop|reload|enable|disable|daemon-reload)\b"
    r"|\bdocker\s+(compose\s+)?(restart|start|stop|run|up|down|rm|create|update|exec\s+\S+\s+(sh|bash)\s+-c)\b"
    r"|\b(pip|uv|npm|pnpm|yarn|apt|dnf|brew)\s+(install|add|remove|uninstall)\b"
    r"|\bhermes\s+kanban\s+(complete|block|comment|create|reopen|unblock|specify)\b"
    r"|\bcurl\b.*\s-X\s*(POST|PUT|PATCH|DELETE)"
    r"|\bcurl\b.*\s(-d|--data\S*)\s)",
    re.I | re.S,
)

_INLINE_PROBE = re.compile(r"(\bpython[0-9.]*|/python[0-9.]*|\bnode|\bperl|\bruby)\s+-[ce]\b|<<\s*['\"]?(PY|EOF|PYTHON)", re.I)
_DELIVERY_COMMAND = re.compile(r"\bgit\s+(commit|push)\b|\bgh\s+(pr|issue)\s+(create|merge|comment)\b", re.I)

# Explicit act-now directives from the user. Kept narrow: these are instructions to stop
# investigating and deliver, not ordinary task requests.
_DIRECTIVE = re.compile(
    r"\b((do|fix|make|open|ship|push|send|merge|print|start)\b[^.\n]{0,40}\bright now|do (a|the) pr|open (a|the) pr|make (a|the) pr|just (do|fix|make|ship|open|push) (it|the|a)\b"
    r"|stop (investigating|digging|researching|exploring|testing|poking)"
    r"|(ship|push|merge) it( now)?\b|do it now|now,? please|asap|no more (investigating|digging|testing))",
    re.I,
)

# Intermediate steps toward a deliverable: they land locally but deliver nothing yet. Under an
# act-now directive they do not reset the streak (edit, re-test, edit again is how a PR never ships).
_INTERMEDIATE_TOOL_NAMES = frozenset({"write_file", "patch", "browser_click", "browser_type", "browser_press"})

DIRECTIVE_TURN_NOTE = (
    "[hermes note — act-now directive: the user said \"{phrase}\". Deliver that first. Land the "
    "requested change (edit, commit, PR, or reply) within the next few tool calls; do not open a new "
    "investigation thread. Anything still uncertain goes into the PR description or your reply as an "
    "open question instead of more probing. A safety check is never weakened to make a case pass.]"
)

_NOTICE = (
    "[hermes note: {count} tool calls this turn without landing anything (no file edit, commit, PR, "
    "message or task update). Stop exploring. In your next reply state in two lines what you have "
    "established, then either make the change now or report status to the user and stop. Do not start "
    "another investigation thread; reuse results you already have instead of re-reading or re-probing.{directive}]"
)
_FINAL = (
    "[hermes note: {count} tool calls without landing anything. Your NEXT action must land something "
    "(edit, commit, PR, message, task update) or be your final reply to the user. Exploration stops "
    "at {halt} calls.{directive}]"
)
_DIRECTIVE_SUFFIX = " The user asked: \"{phrase}\" — that deliverable comes first."

_PROBE_NOTICE = (
    "[hermes note: {count} one-off inline interpreter probes this turn (python -c / heredoc), each "
    "re-loading everything from scratch. Write one scratch script in a scratch directory and iterate on "
    "it (edit, re-run) instead, and decide what question the next probe actually answers before running it.]"
)
_REREAD_NOTICE = (
    "[hermes note: you already read lines {start}-{end} of {path} earlier this turn (call {call}) and "
    "nothing has changed that file since. Use that earlier result instead of re-reading; use search_files "
    "for a specific symbol if you need a different spot.]"
)
_READ_DEFAULT_LIMIT = 2000  # tools/file_tools.py READ_FILE_SCHEMA default

# Worker mode : for a dispatched Kanban worker or a delegated subagent, reading
# is often the deliverable (a 30-record audit needs ~60 reads). Telling it to "report status and
# stop" at 20 calls made the #11 review workers file a third of the work as done. Workers get a
# checkpoint instead: persist what you have, then keep going. Their run length stays bounded by the
# iteration budget, the Kanban budget notices and the identical-call guards.
_WORKER_NOTICE = (
    "[hermes note: {count} tool calls since you last saved anything. If reading is the job (a review, "
    "audit or comparison), that is fine: append what you have established so far to your report or "
    "scratch file now, so nothing is lost, then continue with the next items. Do not stop early and do "
    "not file partial work as complete. If you are going in circles instead (re-reading, re-probing the "
    "same thing), change approach.]"
)

HALT_RESPONSE = (
    "I stopped after {count} tool calls that didn't change anything, because I was investigating "
    "in circles instead of delivering. Everything I found is in the tool results above. Tell me what "
    "to land first, or send `continue` and I'll make the change with what I have."
)


@dataclass(frozen=True)
class LandingStreakConfig:
    """Thresholds; ``0`` disables a stage. Directive thresholds apply after an act-now message."""

    enabled: bool = True
    notice_after: int = 20
    final_after: int = 40
    halt_after: int = 50
    directive_notice_after: int = 10
    directive_final_after: int = 20
    directive_halt_after: int = 35
    probe_notice_after: int = 6
    reread_notice: bool = True
    # Kanban workers and subagents: one checkpoint notice, no final order, no halt. 0 disables it.
    worker_mode: bool = True
    worker_notice_after: int = 40

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "LandingStreakConfig":
        if not isinstance(data, Mapping):
            return cls()
        out: dict[str, Any] = {}
        for name, default in cls().__dict__.items():
            value = data.get(name, default)
            if isinstance(default, bool):
                out[name] = value if isinstance(value, bool) else default
            else:
                try:
                    v = int(value)
                except (TypeError, ValueError):
                    v = default
                out[name] = v if v >= 0 else default
        return cls(**out)


@dataclass(frozen=True)
class LandingObservation:
    notice: Optional[str] = None
    halt: bool = False
    count: int = 0


def detect_directive(text: Any) -> Optional[str]:
    """Return the act-now phrase in the user's message (with a little context), else None."""
    if not isinstance(text, str) or not text.strip():
        return None
    # Only the user's own words: ignore harness-injected blocks.
    if text.lstrip().startswith(("[CONTEXT COMPACTION", "[IMPORTANT:", "[Your active task list", "✔ [")):
        return None
    m = _DIRECTIVE.search(text)
    if not m:
        return None
    start = max(0, text.rfind(".", 0, m.start()) + 1, text.rfind("\n", 0, m.start()) + 1)
    end_candidates = [i for i in (text.find(".", m.end()), text.find("\n", m.end()), text.find(",", m.end())) if i != -1]
    end = min(end_candidates) if end_candidates else len(text)
    phrase = " ".join(text[start:end].split())
    return phrase[:160] or m.group(0)


def directive_turn_note(phrase: Optional[str]) -> str:
    return DIRECTIVE_TURN_NOTE.format(phrase=phrase) if phrase else ""


def is_landing(tool_name: str, args: Mapping[str, Any] | None, *, failed: bool) -> bool:
    """Whether a completed call changed something. Failed calls never land."""
    if failed or not tool_name:
        return False
    if tool_name in LANDING_TOOL_NAMES or _MCP_MUTATING.search(tool_name):
        return True
    if tool_name == "terminal":
        cmd = (args or {}).get("command") if isinstance(args, Mapping) else None
        if not isinstance(cmd, str):
            return False
        # Inline interpreter probes are exploration; their code ("x > 0.95") must not read as a redirect.
        if _INLINE_PROBE.search(cmd) and not _DELIVERY_COMMAND.search(cmd):
            return False
        return bool(_LANDING_COMMAND.search(cmd))
    return False


class LandingStreak:
    """Per-turn counter of calls since the last landing action."""

    def __init__(self, config: LandingStreakConfig | None = None, *, can_halt: bool = False, worker: bool = False):
        self.config = config or LandingStreakConfig()
        self.worker = bool(worker and self.config.worker_mode)
        self.can_halt = can_halt and not self.worker
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.directive: Optional[str] = None
        self._fired: set[str] = set()
        self._probes = 0
        self._probe_noticed = False
        # path -> [(start, end, call_number)] of lines read since anything could have changed them.
        self._reads: dict[str, list[tuple[int, int, int]]] = {}
        self._reread_noticed: set[str] = set()
        self._calls = 0

    def set_directive(self, phrase: Optional[str]) -> None:
        self.directive = phrase or None

    def pin_steer(self, text: Any) -> Optional[str]:
        """A mid-turn steer that is an act-now directive pins it and restarts the count from now,
        so the tightened thresholds measure calls since the user said it. Other steers change nothing."""
        phrase = detect_directive(text)
        if phrase:
            self.directive = phrase
            self.count = 0
            self._fired.clear()
        return phrase

    def _thresholds(self) -> tuple[int, int, int]:
        c = self.config
        if self.worker:
            return c.worker_notice_after, 0, 0
        if self.directive:
            return c.directive_notice_after, c.directive_final_after, c.directive_halt_after
        return c.notice_after, c.final_after, c.halt_after

    def _probe_notice(self, tool_name: str, args: Mapping[str, Any] | None) -> Optional[str]:
        """Nudge toward one scratch script after N inline interpreter probes (once per turn)."""
        at = self.config.probe_notice_after
        if not at or tool_name != "terminal" or not isinstance(args, Mapping):
            return None
        cmd = args.get("command")
        if not isinstance(cmd, str) or not _INLINE_PROBE.search(cmd):
            return None
        self._probes += 1
        if self._probes >= at and not self._probe_noticed:
            self._probe_noticed = True
            return _PROBE_NOTICE.format(count=self._probes)
        return None

    def _reread_notice(self, tool_name: str, args: Mapping[str, Any] | None, *, failed: bool) -> Optional[str]:
        """Flag a read_file whose range mostly repeats an earlier read of an unchanged file."""
        if not isinstance(args, Mapping):
            return None
        path = args.get("path")
        if tool_name in ("write_file", "patch") and isinstance(path, str):
            self._reads.pop(path, None)  # that file changed; earlier reads are stale
            self._reread_noticed.discard(path)
            return None
        if tool_name != "read_file" or failed or not isinstance(path, str) or not self.config.reread_notice:
            return None
        try:
            start = max(1, int(args.get("offset") or 1))
            limit = max(1, int(args.get("limit") or _READ_DEFAULT_LIMIT))
        except (TypeError, ValueError):
            return None
        end = start + limit - 1
        earlier = self._reads.setdefault(path, [])
        hit = None
        for s, e, n in earlier:
            overlap = min(end, e) - max(start, s) + 1
            if overlap > 0 and overlap * 2 >= (end - start + 1):
                hit = (s, e, n)
                break
        earlier.append((start, end, self._calls))
        if hit and path not in self._reread_noticed:
            self._reread_noticed.add(path)
            s, e, n = hit
            return _REREAD_NOTICE.format(path=path, start=s, end=e, call=n)
        return None

    def observe(self, tool_name: str, args: Mapping[str, Any] | None, *, failed: bool) -> LandingObservation:
        if not self.config.enabled:
            return LandingObservation()
        if tool_name in NEUTRAL_TOOL_NAMES or tool_name.endswith(_NEUTRAL_SUFFIXES):
            return LandingObservation(count=self.count)
        self._calls += 1
        side = [n for n in (self._probe_notice(tool_name, args),
                            self._reread_notice(tool_name, args, failed=failed)) if n]
        obs = self._observe_streak(tool_name, args, failed=failed)
        notices = side + ([obs.notice] if obs.notice else [])
        return LandingObservation("\n\n".join(notices) or None, obs.halt, obs.count)

    def _observe_streak(self, tool_name: str, args: Mapping[str, Any] | None, *, failed: bool) -> LandingObservation:
        landed = is_landing(tool_name, args, failed=failed)
        if landed and tool_name == "terminal":
            self._reads.clear()  # a shell change may have touched any file read earlier
            self._reread_noticed.clear()
        if landed and self.directive and tool_name in _INTERMEDIATE_TOOL_NAMES:
            landed = False
        if landed:
            self.count = 0
            self._fired.clear()  # each stretch of exploration gets its own escalation
            return LandingObservation()
        self.count += 1
        notice_at, final_at, halt_at = self._thresholds()
        suffix = _DIRECTIVE_SUFFIX.format(phrase=self.directive) if self.directive else ""
        if self.can_halt and halt_at and self.count >= halt_at and "halt" not in self._fired:
            self._fired.add("halt")
            return LandingObservation(halt=True, count=self.count)
        if final_at and self.count >= final_at and "final" not in self._fired:
            self._fired.update({"final", "notice"})
            halt_txt = str(halt_at) if (self.can_halt and halt_at) else "the turn budget"
            return LandingObservation(_FINAL.format(count=self.count, halt=halt_txt, directive=suffix), count=self.count)
        if notice_at and self.count >= notice_at and "notice" not in self._fired:
            self._fired.add("notice")
            if self.worker:
                return LandingObservation(_WORKER_NOTICE.format(count=self.count), count=self.count)
            return LandingObservation(_NOTICE.format(count=self.count, directive=suffix), count=self.count)
        return LandingObservation(count=self.count)
