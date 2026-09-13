"""Manual ``/compress`` core shared by the CLI, gateway, TUI and ACP surfaces.

Manual compression is the ONE sanctioned history mutation (prompt-cache invariant): each surface parses
its own flags and renders its own text, but the sequence — split for ``here [N]``, estimate, run
``agent._compress_context(force=True)``, detect a lock-skip, rejoin the verbatim tail, summarize — lives
here so ``--preview`` / ``--aggressive`` and the lock-skip wording cannot drift per surface again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

#: Every surface renders the same refusal; hard truncation has no persistence path outside the guarded
#: ``_compress_context`` rotation, so ``--aggressive`` is refused rather than mis-parsed as a focus topic.
AGGRESSIVE_UNSUPPORTED = (
    "--aggressive is not supported; use '/compress here [N]' to keep only recent exchanges, "
    "or /undo to drop turns.")
MIN_MESSAGES = 4


@dataclass
class CompressRequest:
    """Parsed ``/compress`` arguments (``extract_compress_flags`` + ``parse_partial_compress_args``)."""
    preview: bool = False
    aggressive: bool = False
    partial: bool = False
    keep_last: int = 2
    focus_topic: Optional[str] = None


@dataclass
class CompressResult:
    status: str  # "preview" | "compressed" | "lock_skipped" | "nothing_to_do"
    before_messages: List[Dict[str, Any]]
    after_messages: List[Dict[str, Any]]
    before_tokens: int
    after_tokens: int
    request: CompressRequest
    lines: List[str] = field(default_factory=list)  # preview report lines (status == "preview")
    lock_holder: Any = None
    summary: Optional[Dict[str, Any]] = None  # ``summarize_manual_compression`` payload when compressed

    @property
    def removed(self) -> int:
        return len(self.before_messages) - len(self.after_messages)


def parse_compress_args(raw_args: str) -> CompressRequest:
    """One parser for every surface: flags anywhere, then the boundary-aware / focus positional forms."""
    from hermes_cli.partial_compress import extract_compress_flags, parse_partial_compress_args
    rest, preview, aggressive = extract_compress_flags((raw_args or "").strip())
    partial, keep_last, focus_topic = parse_partial_compress_args(rest)
    return CompressRequest(preview=preview, aggressive=aggressive, partial=partial, keep_last=keep_last,
                           focus_topic=focus_topic or None)


def estimate_request_tokens(agent: Any, messages: Sequence[Dict[str, Any]]) -> int:
    """Transcript + system prompt + tool schemas: a transcript-only figure understates real request pressure
    and can even appear to grow after a dense handoff summary replaces many short turns (#6217)."""
    from agent.model_metadata import estimate_request_tokens_rough
    if not messages:
        return 0
    return estimate_request_tokens_rough(
        list(messages), system_prompt=getattr(agent, "_cached_system_prompt", "") or "",
        tools=getattr(agent, "tools", None) or None)


def compress_now(
    agent: Any, history: Sequence[Dict[str, Any]], request: CompressRequest, *,
    system_message: Any = None, task_id: str = "default", skip_without_window: bool = False,
) -> CompressResult:
    """Run one manual compression of ``history`` on ``agent`` and return the outcome; the caller installs
    ``after_messages`` (and re-anchors session ids) — history is never mutated here.

    ``preview=True`` performs no compression and leaves ``agent`` untouched. A held compression lock
    yields ``lock_skipped`` with the agent's signal cleared and the deferred context-engine notification
    discarded; otherwise the caller must call ``finalize_context_engine_compression_notification(agent,
    committed=True)`` once its own history transaction commits (``committed=False`` on failure).
    ``system_message=None`` makes ``_compress_context`` rebuild the prompt; passing the cached prompt
    duplicated the identity block (#15281). ``skip_without_window`` (gateway) answers ``nothing_to_do``
    when the local compressor sees no summarizable middle; the in-process surfaces leave it off because
    ``_compress_context`` still does useful work there — codex_app_server native compaction, and the
    phase-1 tool-result prune / blank-echo drop that ``ContextCompressor.compress`` commits even when no
    summary window exists."""
    from agent.conversation_compression import finalize_context_engine_compression_notification
    from agent.manual_compression_feedback import summarize_manual_compression
    from hermes_cli.partial_compress import (
        rejoin_compressed_head_and_tail, split_history_for_partial_compress, summarize_compress_preview)

    before = list(history)
    before_tokens = estimate_request_tokens(agent, before)
    head, tail = before, []
    if request.partial:
        head, tail = split_history_for_partial_compress(before, request.keep_last)
        if not tail:  # degenerate split: nothing to keep verbatim → full compression
            head = before
    if request.preview:
        report = summarize_compress_preview(before, request.partial, request.keep_last, request.focus_topic, before_tokens)
        return CompressResult("preview", before, before, before_tokens, before_tokens, request, lines=report["lines"])

    compressor = getattr(agent, "context_compressor", None)
    has_content = getattr(compressor, "has_content_to_compress", None)
    if skip_without_window and callable(has_content) and has_content(head) is False:
        return CompressResult("nothing_to_do", before, before, before_tokens, before_tokens, request)
    try:
        compressed, _ = agent._compress_context(
            head, system_message, approx_tokens=before_tokens, focus_topic=request.focus_topic, force=True,
            defer_context_engine_notification=True, **({"task_id": task_id} if task_id != "default" else {}))
    except Exception:
        finalize_context_engine_compression_notification(agent, committed=False)
        raise
    # Type-pinned (is True / str): bare truthiness is fooled by MagicMock auto-attributes on test doubles.
    lock_signal = getattr(agent, "_compression_skipped_due_to_lock", None)
    if lock_signal is True or isinstance(lock_signal, str):
        agent._compression_skipped_due_to_lock = None
        finalize_context_engine_compression_notification(agent, committed=False)
        return CompressResult("lock_skipped", before, before, before_tokens, before_tokens, request,
                              lock_holder=lock_signal if isinstance(lock_signal, str) else None)
    if tail:
        compressed = rejoin_compressed_head_and_tail(compressed, tail)
    after_tokens = estimate_request_tokens(agent, compressed)
    summary = summarize_manual_compression(before, compressed, before_tokens, after_tokens, compression_state=compressor)
    return CompressResult("compressed", before, list(compressed), before_tokens, after_tokens, request, summary=summary)


def render_compress_result(result: CompressResult, *, prefix: str = "") -> List[str]:
    """Surface-neutral text lines for a result (each surface may add its own icon/prefix)."""
    if result.status == "preview":
        return [f"{prefix}{line}" for line in result.lines]
    if result.status == "lock_skipped":
        from agent.manual_compression_feedback import describe_compression_lock_skip
        return [f"{prefix}{describe_compression_lock_skip(result.lock_holder or True)}"]
    if result.status == "nothing_to_do":
        return [f"{prefix}Nothing to compress yet."]
    summary = result.summary or {}
    return [f"{prefix}{line}" for line in (summary.get("headline"), summary.get("token_line"), summary.get("note")) if line]
