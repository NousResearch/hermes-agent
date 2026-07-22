"""disk-cleanup plugin — auto-cleanup of ephemeral Hermes session files.

Wires three behaviours:

1. ``post_tool_call`` hook — consumes explicit ``created_paths`` metadata
   emitted by ``write_file``, then tracks paths matching test/temp patterns
   under ``HERMES_HOME`` silently.  Terminal and patch calls are excluded
   because their current result metadata cannot prove exclusive creation.

2. ``on_session_end`` hook — when any test files were auto-tracked
   during the just-finished turn, runs :func:`disk_cleanup.quick` and
   logs a single line to ``$HERMES_HOME/disk-cleanup/cleanup.log``.

3. ``/disk-cleanup`` slash command — manual ``status``, ``dry-run``,
   ``quick``, ``deep``, ``track``, ``forget``.

Replaces PR #12212's skill-plus-script design: the agent no longer
needs to remember to run commands.
"""

from __future__ import annotations

import json
import logging
import re
import shlex
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from . import disk_cleanup as dg

logger = logging.getLogger(__name__)


# Per-session set of "test files newly tracked this turn".  Keyed by session_id
# (or task_id only when no session identifier exists) so on_session_end can
# clean only the paths owned by the ending session. Guarded by a lock —
# post_tool_call can fire concurrently on
# parallel tool calls.
_recent_test_tracks: Dict[str, Set[str]] = {}
_lock = threading.Lock()


# Tool-call result shapes we can parse
_TERMINAL_PATH_REGEX = re.compile(r"(?:^|\s)(/[^\s'\"`]+|\~/[^\s'\"`]+)")
_WINDOWS_PATH_REGEX = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z]:[\\/][^\s'\"`]+)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tracker_key(task_id: str, session_id: str) -> str:
    return session_id or task_id or "default"


def _record_track(task_id: str, session_id: str, path: Path, category: str) -> None:
    """Record that we tracked *path* as *category* during this turn."""
    if category != "test":
        return
    key = _tracker_key(task_id, session_id)
    with _lock:
        _recent_test_tracks.setdefault(key, set()).add(str(path))


def _drain(task_id: str, session_id: str) -> Set[str]:
    """Pop the set of test paths tracked during this turn."""
    key = _tracker_key(task_id, session_id)
    with _lock:
        return _recent_test_tracks.pop(key, set())


def _attempt_track(
    path_str: str,
    expected_identity: Dict[str, int],
    task_id: str,
    session_id: str,
) -> None:
    """Best-effort auto-track. Never raises."""
    try:
        p = Path(path_str).expanduser()
        if not p.exists():
            return
        category = dg.guess_category(p)
        if category is None:
            return
        newly = dg.track(
            str(p),
            category,
            silent=True,
            expected_identity=expected_identity,
        )
        if newly:
            _record_track(task_id, session_id, p, category)
    except Exception:
        # Hook inspection is advisory; malformed paths or unavailable state
        # must never interrupt the agent loop.
        return


def _extract_paths_from_terminal(args: Dict[str, Any], result: str) -> Set[str]:
    """Extract display/debug candidates from terminal command text.

    This helper deliberately is *not* used for auto-tracking.  Keeping the
    Windows-aware extractor available preserves callers that use it for
    diagnostics, while raw command/output text has no destructive authority.
    """
    paths: Set[str] = set()
    cmd = args.get("command") or ""
    if isinstance(cmd, str) and cmd:
        for match in _WINDOWS_PATH_REGEX.finditer(cmd):
            paths.add(match.group(1))
        # Tokenise the command — catches `touch /tmp/hermes-x/test_foo.py`
        try:
            for tok in shlex.split(cmd, posix=True):
                if tok.startswith(("/", "~/")):
                    paths.add(tok)
        except ValueError:
            pass
    # Preserve diagnostic extraction (including Windows drive paths) for
    # callers that display candidates.  This result is never fed to
    # ``_on_post_tool_call`` and therefore has no cleanup authority.
    if isinstance(result, str) and len(result) < 4096:
        paths.update(_TERMINAL_PATH_REGEX.findall(result))
        paths.update(match.group(1) for match in _WINDOWS_PATH_REGEX.finditer(result))
    return paths


def _extract_trusted_created_paths(
    result: Any,
) -> List[Tuple[str, Dict[str, int]]]:
    """Return path identities bound to a successful ``write_file`` result.

    A path string alone is not durable creation evidence: another process can
    replace it before this hook runs. ``write_file`` therefore emits the exact
    post-write identity sampled inside its path lock, and tracking requires the
    current object to match. Malformed or legacy result shapes fail closed.
    """
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (TypeError, ValueError):
            return []
    if not isinstance(result, dict):
        return []

    paths: List[Tuple[str, Dict[str, int]]] = []
    values = result.get("created_path_identities")
    if not isinstance(values, list):
        return paths
    for value in values:
        if not isinstance(value, dict):
            continue
        path = value.get("path")
        identity = value.get("identity")
        if isinstance(path, str) and path and isinstance(identity, dict):
            paths.append((path, identity))
    return paths


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

def _on_post_tool_call(
    tool_name: str = "",
    args: Optional[Dict[str, Any]] = None,
    result: Any = None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    **_: Any,
) -> None:
    """Auto-track ephemeral files created by recent tool calls."""
    if not isinstance(args, dict):
        return

    if tool_name != "write_file":
        return

    # Only write_file currently emits creation evidence sampled inside its
    # serialized write boundary. Terminal and patch metadata remain ambiguous.
    candidates = _extract_trusted_created_paths(result)

    for path_str, expected_identity in candidates:
        _attempt_track(path_str, expected_identity, task_id, session_id)


def _on_session_end(
    session_id: str = "",
    task_id: str = "",
    completed: bool = True,
    interrupted: bool = False,
    **_: Any,
) -> None:
    """Run quick cleanup if any test files were tracked during this turn."""
    drained_session = _drain(task_id, session_id)
    if not drained_session:
        return

    try:
        summary = dg.quick(paths=drained_session)
    except Exception as exc:
        logger.debug("disk-cleanup quick cleanup failed: %s", exc)
        return

    if summary["deleted"] or summary["empty_dirs"]:
        dg._log(
            f"AUTO_QUICK (session_end): deleted={summary['deleted']} "
            f"dirs={summary['empty_dirs']} freed={dg.fmt_size(summary['freed'])}"
        )


# ---------------------------------------------------------------------------
# Slash command
# ---------------------------------------------------------------------------

_HELP_TEXT = """\
/disk-cleanup — ephemeral-file cleanup

Subcommands:
  status                     Per-category breakdown + top-10 largest
  dry-run                    Preview what quick/deep would delete
  quick                      Run safe cleanup now (no prompts)
  deep                       Run quick, then list items that need prompts
  track <path> <category>    Manually add a path to tracking
  forget <path>              Stop tracking a path (does not delete)

Categories: temp | test | research | download | chrome-profile | cron-output | other

All operations are scoped to HERMES_HOME and /tmp/hermes-*.
Test files created by write_file are auto-tracked and auto-cleaned at session end.
"""


def _fmt_summary(summary: Dict[str, Any]) -> str:
    base = (
        f"[disk-cleanup] Cleaned {summary['deleted']} files + "
        f"{summary['empty_dirs']} empty dirs, freed {dg.fmt_size(summary['freed'])}."
    )
    if summary.get("errors"):
        base += f"\n  {len(summary['errors'])} error(s); see cleanup.log."
    return base


def _handle_slash(raw_args: str) -> Optional[str]:
    argv = raw_args.strip().split()
    if not argv or argv[0] in {"help", "-h", "--help"}:
        return _HELP_TEXT

    sub = argv[0]

    if sub == "status":
        return dg.format_status(dg.status())

    if sub == "dry-run":
        auto, prompt = dg.dry_run()
        auto_size = sum(i["size"] for i in auto)
        prompt_size = sum(i["size"] for i in prompt)
        lines = [
            "Dry-run preview (nothing deleted):",
            f"  Auto-delete : {len(auto)} files ({dg.fmt_size(auto_size)})",
        ]
        for item in auto:
            lines.append(f"    [{item['category']}] {item['path']}")
        lines.append(
            f"  Needs prompt: {len(prompt)} files ({dg.fmt_size(prompt_size)})"
        )
        for item in prompt:
            lines.append(f"    [{item['category']}] {item['path']}")
        lines.append(
            f"\n  Total potential: {dg.fmt_size(auto_size + prompt_size)}"
        )
        return "\n".join(lines)

    if sub == "quick":
        return _fmt_summary(dg.quick())

    if sub == "deep":
        # In-session deep can't prompt the user interactively — show what
        # quick cleaned plus the items that WOULD need confirmation.
        quick_summary = dg.quick()
        _auto, prompt_items = dg.dry_run()
        lines = [_fmt_summary(quick_summary)]
        if prompt_items:
            size = sum(i["size"] for i in prompt_items)
            lines.append(
                f"\n{len(prompt_items)} item(s) need confirmation "
                f"({dg.fmt_size(size)}):"
            )
            for item in prompt_items:
                lines.append(f"  [{item['category']}] {item['path']}")
            lines.append(
                "\nRun `/disk-cleanup forget <path>` to skip, or delete "
                "manually via terminal."
            )
        return "\n".join(lines)

    if sub == "track":
        if len(argv) < 3:
            return "Usage: /disk-cleanup track <path> <category>"
        path_arg = argv[1]
        category = argv[2]
        if category not in dg.ALLOWED_CATEGORIES:
            return (
                f"Unknown category '{category}'. "
                f"Allowed: {sorted(dg.ALLOWED_CATEGORIES)}"
            )
        if dg.track(path_arg, category, silent=True):
            return f"Tracked {path_arg} as '{category}'."
        return (
            f"Not tracked (already present, missing, or outside HERMES_HOME): "
            f"{path_arg}"
        )

    if sub == "forget":
        if len(argv) < 2:
            return "Usage: /disk-cleanup forget <path>"
        n = dg.forget(argv[1])
        return (
            f"Removed {n} tracking entr{'y' if n == 1 else 'ies'} for {argv[1]}."
            if n else f"Not found in tracking: {argv[1]}"
        )

    return f"Unknown subcommand: {sub}\n\n{_HELP_TEXT}"


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("on_session_end", _on_session_end)
    ctx.register_command(
        "disk-cleanup",
        handler=_handle_slash,
        description="Track and clean up ephemeral Hermes session files.",
    )
