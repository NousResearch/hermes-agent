"""disk-cleanup plugin — auto-cleanup of ephemeral Hermes session files.

Wires three behaviours:

1. ``post_tool_call`` hook — inspects ``write_file`` and ``terminal``
   tool results for newly-created paths matching test/temp patterns
   under ``HERMES_HOME`` and tracks them silently.  Zero agent
   compliance required.

2. ``on_session_end`` hook — when any test files were auto-tracked
   during the just-finished turn, runs :func:`disk_cleanup.quick` and
   logs a single line to ``$HERMES_HOME/disk-cleanup/cleanup.log``.

3. ``/disk-cleanup`` slash command — manual ``status``, ``dry-run``,
   ``quick``, ``deep``, ``track``, ``forget``.

Replaces PR #12212's skill-plus-script design: the agent no longer
needs to remember to run commands.
"""

from __future__ import annotations

import logging
import re
import shlex
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Set

from . import disk_cleanup as dg

logger = logging.getLogger(__name__)


# Per-task set of "test files newly tracked this turn".  Keyed by task_id
# (or session_id as fallback) so on_session_end can decide whether to run
# cleanup.  Guarded by a lock — post_tool_call can fire concurrently on
# parallel tool calls.
_recent_test_tracks: Dict[str, Set[str]] = {}
_lock = threading.Lock()

# #75403: per-tool-call snapshot of candidate paths that already existed
# BEFORE the tool ran.  A tool that merely *edits* a pre-existing durable
# file must never auto-track it for deletion.  Keyed by session_id + call
# identity so session-end can remove only matching session entries without
# disturbing another in-flight session's snapshots.
_preexisting_paths: Dict[str, Set[str]] = {}


def _snap_key(tool_call_id: str, task_id: str, session_id: str) -> str:
    """Return a snap key that always encodes session_id for scoped cleanup."""
    call_id = tool_call_id or _tracker_key(task_id, session_id)
    return f"{session_id}::{call_id}"


# Tool-call result shapes we can parse
_WRITE_FILE_PATH_KEY = "path"
_TERMINAL_PATH_REGEX = re.compile(r"(?:^|\s)(/[^\s'\"`]+|\~/[^\s'\"`]+)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tracker_key(task_id: str, session_id: str) -> str:
    return task_id or session_id or "default"


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
    path_str: str, task_id: str, session_id: str, tool_call_id: str = ""
) -> None:
    """Best-effort auto-track. Never raises.

    A file that already existed *before* the tool ran is never auto-tracked
    (#75403): editing a durable file must not mark it disposable.  Only
    genuinely newly-created files are tracked.
    """
    try:
        p = Path(path_str).expanduser()
    except Exception:
        return
    if not p.exists():
        return

    # #75403: consult the pre_tool_call existence snapshot for this call.
    # A pre-existing path the tool merely edited is not disposable.
    try:
        resolved = str(p.resolve())
    except Exception:
        resolved = str(p)
    snap_k = _snap_key(tool_call_id, task_id, session_id)
    with _lock:
        preexisting = _preexisting_paths.get(snap_k)
    if preexisting and resolved in preexisting:
        dg._log(f"SKIP pre-existing file: {p} (edited, not auto-tracked #75403)")
        with _lock:
            snap = _preexisting_paths.get(snap_k)
            if snap is not None:
                snap.discard(resolved)
                if not snap:
                    _preexisting_paths.pop(snap_k, None)
        return

    category = dg.guess_category(p)
    if category is None:
        return
    newly = dg.track(str(p), category, silent=True)
    if newly:
        _record_track(task_id, session_id, p, category)


def _extract_paths_from_write_file(args: Dict[str, Any]) -> Set[str]:
    path = args.get(_WRITE_FILE_PATH_KEY)
    return {path} if isinstance(path, str) and path else set()


def _extract_paths_from_patch(args: Dict[str, Any]) -> Set[str]:
    # The patch tool creates new files via the `mode="patch"` path too, but
    # most of its use is editing existing files — we only care about new
    # ephemeral creations, so treat patch conservatively and only pick up
    # the single-file `path` arg.  Track-then-cleanup is idempotent, so
    # re-tracking an already-tracked file is a no-op (dedup in track()).
    path = args.get("path")
    return {path} if isinstance(path, str) and path else set()


def _extract_paths_from_terminal(args: Dict[str, Any], result: str) -> Set[str]:
    """Best-effort: pull candidate filesystem paths from a terminal command.

    ONLY command-arg paths are eligible — result-only paths are NEVER
    candidates because their pre-existence is unknowable at pre-call time
    (#75403 concurrency/data-loss blocker).
    """
    paths: Set[str] = set()
    cmd = args.get("command") or ""
    if isinstance(cmd, str) and cmd:
        # Tokenise the command — catches `touch /tmp/hermes-x/test_foo.py`
        try:
            for tok in shlex.split(cmd, posix=True):
                if tok.startswith(("/", "~")):
                    paths.add(tok)
        except ValueError:
            pass
    # NOTE: result text is intentionally NOT scanned — paths that appear
    # only in terminal output were not snapshotted pre-call, so their
    # pre-existence is unknowable.  Auto-tracking them risks data loss.
    return paths


def _candidate_paths_for_tracking(
    tool_name: str, args: Dict[str, Any]
) -> Set[str]:
    """Paths a tool call may have created/touched that are worth tracking.

    Shared by the ``pre_tool_call`` existence snapshot and the
    ``post_tool_call`` track decision so the two stay in sync (#75403).
    Terminal only extracts from command args — result paths are never
    candidates because their pre-existence is unknowable.
    """
    if tool_name == "write_file":
        return _extract_paths_from_write_file(args)
    if tool_name == "patch":
        return _extract_paths_from_patch(args)
    if tool_name == "terminal":
        return _extract_paths_from_terminal(args, "")  # never scan result
    return set()


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

def _on_pre_tool_call(
    tool_name: str = "",
    args: Optional[Dict[str, Any]] = None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    **_: Any,
) -> None:
    """Snapshot which candidate paths already exist before the tool runs.

    ``post_tool_call`` consults this snapshot so editing a pre-existing
    durable file never auto-tracks it for deletion (#75403).  Observer-only
    — returns None and never blocks the tool.
    """
    if not isinstance(args, dict):
        return None
    candidates = _candidate_paths_for_tracking(tool_name, args)
    if not candidates:
        return None
    existing: Set[str] = set()
    for path_str in candidates:
        try:
            cp = Path(path_str).expanduser()
            if cp.exists():
                existing.add(str(cp.resolve()))
        except Exception:
            pass
    if existing:
        snap_k = _snap_key(tool_call_id, task_id, session_id)
        with _lock:
            _preexisting_paths.setdefault(snap_k, set()).update(existing)
    return None

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

    candidates = _candidate_paths_for_tracking(tool_name, args)

    for path_str in candidates:
        _attempt_track(path_str, task_id, session_id, tool_call_id)

    # Drain this call's pre-existence snapshot so it can't leak across
    # turns (pre_tool_call and post_tool_call are paired per call).
    snap_k = _snap_key(tool_call_id, task_id, session_id)
    with _lock:
        _preexisting_paths.pop(snap_k, None)


def _on_session_end(
    session_id: str = "",
    completed: bool = True,
    interrupted: bool = False,
    **_: Any,
) -> None:
    """Run quick cleanup if any test files were tracked during this turn."""
    # #75403 safety net: clear pre-existence snapshots left behind when a
    # pre_tool_call fired without a matching post_tool_call (blocked/errored
    # tool).  Only remove entries matching *this* session — never disturb
    # another in-flight session's snapshots.
    prefix = f"{session_id}::"
    with _lock:
        stale_keys = [k for k in _preexisting_paths if k.startswith(prefix)]
        for k in stale_keys:
            _preexisting_paths.pop(k, None)
    # Drain both task-level and session-level buckets.  In practice only one
    # is populated per turn; the other is empty.
    drained_session = _drain("", session_id)
    # Also drain any task-scoped buckets that happen to exist.  This is a
    # cheap sweep: if an agent spawned subagents (each with their own
    # task_id) they'll have recorded into separate buckets; we want to
    # cleanup them all at session end.
    with _lock:
        task_buckets = list(_recent_test_tracks.keys())
    for key in task_buckets:
        if key and key != session_id:
            _recent_test_tracks.pop(key, None)

    if not drained_session and not task_buckets:
        return

    try:
        summary = dg.quick()
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
Test files are auto-tracked on write_file / terminal and auto-cleaned at session end.
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
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("on_session_end", _on_session_end)
    ctx.register_command(
        "disk-cleanup",
        handler=_handle_slash,
        description="Track and clean up ephemeral Hermes session files.",
    )
