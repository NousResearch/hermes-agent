"""``/undo``'s line naming the files the undone turns changed, so the user knows ``/rollback`` can bring them back.

``/undo`` only rewinds the conversation; it never touches files. With checkpoints on, it names the
files that differ from the first snapshot the undone span took, and says to run ``/rollback``. It
names no checkpoint: which one is right cannot be proven from git history alone (whole-second
stamps, a take that reused an older snapshot, another chat's takes in the same project, pruning), so
the user picks from the list ``/rollback`` shows.

The base is the newest retained snapshot at or before the span start (a first undone turn whose take
found nothing changed reused it), else the oldest retained one. The line is printed only when some
snapshot was taken in the span: with none, no undone turn wrote through a checkpointing tool, so the
surface prints main's output. The list says "changed", never "will be restored": it is best effort and
promises nothing, and when unsure it prints nothing.

Read-only over the store: ``git log`` plus a name-only diff of the snapshot's tree against the
working tree, through a throwaway index; it writes no objects, refs or project index.
"""
from __future__ import annotations

import logging
import math
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_MAX_NAMED_FILES = 8


def _epoch(value) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) and value > 0 else None
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.strip()).timestamp()
        except ValueError:
            return None
    return None


def undo_span_start(target_timestamp, kept_history=()) -> Optional[float]:
    """When the undone span began: the undone user message's stamp, or the newest kept message's when that is
    later. A gateway stamps a message with its platform send time, so a prompt queued while the previous turn
    ran is stamped before that turn's checkpoints; the kept turn's last row bounds them out."""
    from hermes_agent.agent.context_compressor import is_compaction_summary_message

    start = _epoch(target_timestamp)
    for message in kept_history or ():
        if not isinstance(message, dict) or is_compaction_summary_message(message):
            continue  # a summary row is stamped when the compaction ran, not when its turns did
        if (when := _epoch(message.get("timestamp"))) is not None:
            start = when if start is None else max(start, when)
    return start


def _span_base_tree(mgr, abs_dir: str, since: float) -> Optional[str]:
    """Tree to diff the working tree against (see the module docstring); None when no snapshot was taken in
    the span, so there is nothing to report."""
    from hermes_agent.tools import checkpoint_manager as cm

    store = cm._store_path()
    if not (store / "HEAD").exists():
        return None
    ok, out, _ = cm._run_git(["log", cm._ref_name(cm._project_hash(abs_dir)), "--format=%T|%at",
                              "-n", str(mgr.max_snapshots)], store, abs_dir, allowed_returncodes={128, 129})
    rows = [(tree, int(epoch)) for tree, _, epoch in (line.partition("|") for line in (out.splitlines() if ok else []))
            if tree and epoch.isdigit()]  # newest first
    if not any(epoch >= math.floor(since) for _tree, epoch in rows):
        return None
    return next((tree for tree, epoch in rows if epoch <= since), rows[-1][0])


def _changed_files(abs_dir: str, tree: str) -> Optional[list[str]]:
    """Paths whose working-tree state differs from ``tree``: tracked changes and deletions plus new files,
    through a throwaway index (no object, ref or project index is written), None on failure."""
    from hermes_agent.tools import checkpoint_manager as cm

    store = cm._store_path()
    excludes = cm._exclude_store_args(store.parent, abs_dir)
    with tempfile.TemporaryDirectory(prefix="undo-hint-", dir=store) as scratch:
        index = Path(scratch) / "index"
        ok, _, _ = cm._run_git(["read-tree", tree], store, abs_dir, index_file=index)
        if not ok:
            return None
        ok, changed, _ = cm._run_git(["diff", "--name-only", "-z", *excludes], store, abs_dir, index_file=index)
        ok_new, new, _ = cm._run_git(["ls-files", "--others", "--exclude-standard", "-z", *excludes],
                                     store, abs_dir, index_file=index)
    if not (ok and ok_new):
        return None
    return sorted({p for p in (changed + "\x00" + new).split("\x00") if p})


def undo_changed_files(mgr, cwd: str, since, *, task_id: Optional[str] = "default") -> list[str]:
    """Files the turns from ``since`` (``undo_span_start``) onward changed under ``cwd``, best effort. ``[]``:
    checkpoints off or refused here, no timestamp, no snapshot of the span, or nothing changed.
    ``task_id=None``: the caller already applied ``unsupported_backend_reason`` under its own scope."""
    from hermes_agent.tools import checkpoint_manager as cm

    if mgr is None or not getattr(mgr, "enabled", False) or (since_ts := _epoch(since)) is None:
        return []
    try:
        if task_id is not None and mgr.unsupported_backend_reason(task_id):
            return []
        abs_dir = str(cm._normalize_path(cwd))
        tree = _span_base_tree(mgr, abs_dir, since_ts)
        return (_changed_files(abs_dir, tree) or []) if tree else []
    except Exception as exc:  # health: allow BLE001 -- an advisory line must never fail the /undo it follows
        logger.debug("undo rollback hint skipped: %s", exc)
        return []


def format_undo_rollback_hint(files: list[str]) -> str:
    """The one localized line every surface prints after its own ``/undo`` output; ``""`` for no files."""
    if not files:
        return ""
    from hermes_agent.agent.i18n import t

    named = ", ".join(files[:_MAX_NAMED_FILES])
    if len(files) > _MAX_NAMED_FILES:
        named += " " + t("shared.undo_rollback.more", count=len(files) - _MAX_NAMED_FILES)
    return t("shared.undo_rollback.list", files=named)


def undo_hint_payload(mgr, cwd: str, since, *, task_id: Optional[str] = "default") -> Optional[dict]:
    """The wire shape (``UndoRollbackHint``): the changed files plus the rendered ``text``; None: no line."""
    files = undo_changed_files(mgr, cwd, since, task_id=task_id)
    return {"files": files, "text": format_undo_rollback_hint(files)} if files else None


def checkpoint_taken_at(cwd: str, commit: str) -> Optional[tuple[float, bool]]:
    """``(when, exact)`` checkpoint ``commit`` was taken: the exact time its body records, else its git
    author second (``exact`` False: older snapshots); None when unreadable."""
    from hermes_agent.tools import checkpoint_manager as cm

    store = cm._store_path()
    if not commit or not (store / "HEAD").exists():
        return None
    ok, out, _ = cm._run_git(["log", "-1", "--format=%at%x00%B", commit], store, str(cm._normalize_path(cwd)),
                             allowed_returncodes={128})
    second, _, body = out.partition("\x00") if ok else ("", "", "")
    for line in body.splitlines():
        key, _, value = line.partition(":")
        if key.strip() == cm.TAKEN_AT_TRAILER and (when := _epoch(_float(value))) is not None:
            return when, True
    return (float(second.strip()), False) if second.strip().isdigit() else None


def _float(value: str) -> Optional[float]:
    try:
        return float(value.strip())
    except ValueError:
        return None


def rollback_rewinds_newest_turn(history: list, user_indices: list[int], taken) -> bool:
    """Whether a whole-tree ``/rollback`` to a checkpoint taken at ``taken`` (``checkpoint_taken_at``) also
    undoes the newest kept user turn (main's "so the agent's context matches the restored files"): only when
    that turn made the checkpoint or came after it, i.e. one of its rows is no older than the snapshot. After
    ``/undo`` already removed the turn that made it, the newest kept turn ended before the checkpoint and stays.

    A snapshot from an older build records only its git second, so a row inside that same second may be
    either side of it: that is ambiguous, and an ambiguous turn is KEPT (losing a turn the user kept is the
    worse mistake). Unknown times keep main's behaviour."""
    from hermes_agent.agent.context_compressor import is_compaction_summary_message

    if not user_indices or taken is None:
        return bool(user_indices)
    when, exact = taken
    stamps = [_epoch(m.get("timestamp")) for m in history[user_indices[-1]:]
              if isinstance(m, dict) and not is_compaction_summary_message(m)]
    known = [s for s in stamps if s is not None]
    if not stamps or len(known) != len(stamps):
        return True
    return max(known) >= (when if exact else math.floor(when) + 1)
