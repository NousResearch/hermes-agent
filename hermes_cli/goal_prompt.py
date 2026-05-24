"""Shared helpers for loading GOAL_PROMPT.md into /goal."""

from __future__ import annotations

import re
from pathlib import Path

DEFAULT_GOAL_PROMPT_FILENAMES: tuple[str, ...] = (
    "GOAL_PROMPT.md",
    "goal_prompt.md",
    "GO_PROMPT.md",
    "go_prompt.md",
)

ONESHOT_GOAL_INSTRUCTIONS = """\

[Hermes /goal_prompt_oneshot mode]
The user invoked the high-autonomy one-shot project loop. Behave like repeated
/goal_prompt runs chained together inside this standing /goal:

1. Before acting, inspect the project context: README.md, GOAL.md,
   docs/status/NEXT_ACTIONS.md, docs/security/SAFETY.md, AGENTS.md or handoff
   docs when present, and current git status.
2. Treat this invocation as approval for sustained autonomous development
   within the selected project: inspect files, edit code/docs, add tests, run
   builds/tests/linters/scripts, start local dev services, use local/mock/
   simulation/paper paths, and commit/push when project policy allows.
3. Take the next concrete incomplete slice from GOAL.md/NEXT_ACTIONS.md. After
   each slice, verify it, refresh NEXT_ACTIONS.md and GOAL_PROMPT.md with the
   exact current frontier, commit/push if allowed, then immediately re-read the
   updated frontier and continue without stopping for routine status reports.
4. End every non-final slice report with this machine-readable block so Hermes
   can continue deterministically instead of inferring from prose:

   /goal_prompt_oneshot continuation decision: CONTINUE
   GOAL.md definition of done: NOT SATISFIED
   Completed slice: <one-line summary>
   Next safe autonomous slice: <exact next action from NEXT_ACTIONS.md>
   Operator input needed before next slice: None
   Hard stop: No

   Use STOP_FOR_OPERATOR only for true non-bypassable gates:

   /goal_prompt_oneshot continuation decision: STOP_FOR_OPERATOR
   GOAL.md definition of done: NOT SATISFIED
   Reason: <exact non-bypassable gate>
   No safe autonomous slice remains because: <why>
   Operator decision needed: <specific bounded approval/input>

   Use COMPLETE only when GOAL.md's definition of done is actually satisfied:

   /goal_prompt_oneshot continuation decision: COMPLETE
   GOAL.md definition of done: SATISFIED
   Evidence: <tests/docs/git status/commit ids if relevant>
5. Long-runtime freshness: after the configured context-compaction refresh
   interval (default: 5), Hermes may finish the current slice, update the
   frontier docs, start a fresh /new session, and re-run /goal_prompt_oneshot
   from the updated GOAL_PROMPT.md so work continues from disk truth instead of
   stale compacted memory.
6. Continue until GOAL.md's definition of done is satisfied, the configured
   goal/agent loop limit is reached, tooling/context fails irrecoverably, or a
   non-bypassable blocker is reached.
7. If this session cannot finish the whole goal, leave the repo ready for a
   normal /goal_prompt continuation: document the exact frontier, blockers,
   verification state, and first next action in NEXT_ACTIONS.md and
   GOAL_PROMPT.md, and commit/push that handoff if project policy allows.

High autonomy is not permission to ignore system/developer policy, expose or
fabricate secrets, exfiltrate credentials, bypass OS/tool approval mechanisms,
perform destructive unrelated filesystem operations, or execute live/funded/
production actions whose gates are absent. For trading bots and live-execution
systems, continue through local tests, mocks, simulations, dry-runs, paper/
shadow modes, fail-closed gates, and runbook scaffolding. Stop at the first
point where real private keys, real funds, token approvals, transaction signing
or broadcasting, production deployment, irreversible on-chain actions, or
scale-up decisions are required unless the current project docs plus current
user instruction provide concrete bounded authorization for that exact action.
When stopped at such a boundary, document the blocked live step and the safe
/goal_prompt continuation path."""


def extract_goal_prompt_text(raw: str) -> str:
    """Extract the executable /goal prompt from a GOAL_PROMPT-style file.

    If the file contains a fenced code block, use the first text/markdown/md
    fence. Otherwise, use the whole file.
    """
    text = (raw or "").strip()
    if not text:
        return ""

    fenced = re.search(
        r"```(?:text|markdown|md)?\s*\n(.*?)\n```",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        return fenced.group(1).strip()
    return text


def _prompt_candidates_for_dir(directory: Path) -> list[Path]:
    """Return likely prompt-file locations for a project/root directory."""
    candidates: list[Path] = []
    for filename in DEFAULT_GOAL_PROMPT_FILENAMES:
        candidates.append(directory / "docs" / "runbooks" / filename)
    for filename in DEFAULT_GOAL_PROMPT_FILENAMES:
        candidates.append(directory / filename)
    return candidates


def resolve_goal_prompt_path(
    arg: str = "",
    cwd: Path | None = None,
    *,
    search_parents: bool = True,
) -> Path | None:
    """Find a GOAL_PROMPT-style file from an optional path or current cwd.

    Args:
        arg: Optional explicit project root, directory, or markdown file path.
        cwd: Base directory to resolve relative paths from. Defaults to
            ``Path.cwd()``.
        search_parents: When ``True`` and no argument is provided, search
            upward from ``cwd`` for an existing prompt file. When ``False``,
            treat ``cwd`` itself as the target project root.

    Returns:
        The existing prompt path when found. If an explicit file path was given,
        returns that candidate even when missing so callers can show a useful
        error. If nothing is found while searching from cwd, returns the
        conventional ``cwd/docs/runbooks/GOAL_PROMPT.md`` missing path.
    """
    base_cwd = (cwd or Path.cwd()).expanduser().resolve()
    candidate_arg = (arg or "").strip()

    if candidate_arg:
        candidate = Path(candidate_arg).expanduser()
        if not candidate.is_absolute():
            candidate = base_cwd / candidate
        candidate = candidate.resolve()
        if candidate.is_dir():
            for path in _prompt_candidates_for_dir(candidate):
                if path.exists():
                    return path
            return candidate / "docs" / "runbooks" / "GOAL_PROMPT.md"
        return candidate

    search_bases = (base_cwd, *base_cwd.parents) if search_parents else (base_cwd,)
    for base in search_bases:
        for path in _prompt_candidates_for_dir(base):
            if path.exists():
                return path
    return base_cwd / "docs" / "runbooks" / "GOAL_PROMPT.md"


def _goal_text_without_slash_command(prompt_text: str) -> str:
    """Return bare goal text, accepting either plain text or ``/goal ...``."""
    text = (prompt_text or "").strip()
    if not text:
        return ""
    if text.lstrip().startswith("/goal"):
        parts = text.lstrip().split(None, 1)
        if parts and parts[0] == "/goal":
            return parts[1].strip() if len(parts) > 1 else ""
    return text


def oneshot_goal_text_from_prompt_text(prompt_text: str) -> str:
    """Return high-autonomy /goal text from extracted GOAL_PROMPT content."""
    base = _goal_text_without_slash_command(prompt_text)
    if not base:
        return ""
    return f"{base}\n\n{ONESHOT_GOAL_INSTRUCTIONS}"


def goal_command_from_prompt_text(prompt_text: str, *, oneshot: bool = False) -> str:
    """Return text suitable for /goal dispatch from extracted prompt text."""
    if oneshot:
        text = oneshot_goal_text_from_prompt_text(prompt_text)
        return f"/goal {text}" if text else ""

    text = (prompt_text or "").strip()
    if not text:
        return ""
    if text.lstrip().startswith("/goal"):
        return text
    return f"/goal {text}"
