"""Archive-before-stub for the proactive tool-result prune: no shortened result loses its only original.

Every tool result the prune is about to shorten is first written, byte-verified, to
``$HERMES_HOME/cache/pruned/`` and its stub names that file, so ``read_file`` gets the exact output
back instead of the model re-running the tool. A result whose write fails keeps its full body. The
files share the spillover sweeper (``tools.tool_result_storage``): kept while any ``state.db`` row
points at them, retired with their session.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

ARCHIVE_POINTER_SEP = " — full result: "
_ARCHIVE_POINTER_RE = re.compile(r" — full result: \S*cache/pruned/[A-Za-z0-9_.-]+")


def has_archive_pointer(content: str) -> bool:
    """True for a prune stub that names its archived original: already shortened, never re-summarized
    (the pointer is the only way back to the body)."""
    return len(content) < 2_000 and ARCHIVE_POINTER_SEP in content and _ARCHIVE_POINTER_RE.search(content) is not None


def _with_pointer(content: Any, display_path: str) -> Any:
    """Stub *content* with the archive pointer, kept before an attached Skill Safety Rule."""
    pointer = f"{ARCHIVE_POINTER_SEP}{display_path}"
    if isinstance(content, list):
        return [*content, {"type": "text", "text": pointer.strip(" —")}]
    from agent.context_compressor_tail import _SKILL_SAFETY_SUFFIX

    head, sep, rule = str(content).partition(_SKILL_SAFETY_SUFFIX)
    return f"{head}{pointer}{sep}{rule}"


def archive_pruned_originals(
    messages: list[dict[str, Any]], pruned: list[dict[str, Any]], session_id: str,
) -> tuple[list[dict[str, Any]], list[str], int]:
    """Write the original of every tool row *pruned* changed; returns ``(rows, written host paths, reverted)``.

    A changed row whose archive write fails is put back to its original (``reverted`` counts them),
    so the caller re-checks its reclaim gate and never commits a stub without a file behind it. A row
    that already names its archive is left as the prune made it: its original is on disk.
    """
    from agent.context_compressor import _tool_calls_by_id
    from tools.tool_result_storage import write_pruned_original

    call_id_to_tool = _tool_calls_by_id(messages)
    rows, written, reverted = list(pruned), [], 0
    for i, (original, replacement) in enumerate(zip(messages, pruned)):
        if (not isinstance(original, dict) or original.get("role") != "tool" or not isinstance(replacement, dict)
                or replacement.get("content") == original.get("content")):
            continue
        body = original.get("content")
        if isinstance(body, str) and has_archive_pointer(body):
            continue
        text = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
        call_id = str(original.get("tool_call_id") or "")
        tool_name = call_id_to_tool.get(call_id, ("unknown", ""))[0]
        archived = write_pruned_original(text, session_id or "session", tool_name, call_id)
        if archived is None:
            rows[i], reverted = dict(original), reverted + 1
            continue
        written.append(archived[0])
        rows[i] = {**replacement, "content": _with_pointer(replacement.get("content"), archived[1])}
    return rows, written, reverted


def archive_compaction_drops(
    messages: list[dict[str, Any]], dropped: list[dict[str, Any]], session_id: str,
) -> tuple[list[dict[str, Any]], list[str], int]:
    """:func:`archive_pruned_originals` for a compaction's tool-result drops (window prune, lean tail).

    A compaction must still reclaim where ``read_file`` cannot open host paths (sandboxed backends), so
    there the plain stubs stand and nothing is written; on the local backend a result whose write fails
    keeps its original, and the compaction proceeds with the rest.
    """
    from tools.tool_result_storage import _backend_reads_host_files

    if not _backend_reads_host_files():
        return dropped, [], 0
    return archive_pruned_originals(messages, dropped, session_id)


def discard_archives(paths: list[str]) -> None:
    """Unlink archives written for a prune that did not commit (nothing points at them)."""
    for path in paths:
        try:
            os.unlink(path)
        except OSError:
            pass  # unreferenced: the retention sweep reclaims it
