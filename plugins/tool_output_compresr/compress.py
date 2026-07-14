"""Tool-output compression: compress, cache the original, point back.

Call the Compresr API, persist the verbatim original to Hermes's managed cache,
and return the API's compressed text unchanged plus a footer pointing the agent
at the full original (recoverable via ``read_file``/``search_files``). Fail-open:
any API error, or an output that isn't meaningfully shorter once the footer is
added, returns the original content unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from . import cache
from .client import CompresrToolOutputClient

logger = logging.getLogger(__name__)

# Marks our own footer so the hook never re-compresses an output it produced.
FOOTER_MARKER = "[compresr:recover]"

# Conservative footer token budget (path embedded twice: in prose + read_file hint).
# The exact net-size check below catches any long remote-home path that exceeds it.
FOOTER_TOKEN_BUDGET = 90

# Rough chars-per-token used for dependency-free gating estimates.
_CHARS_PER_TOKEN = 4


def count_tokens(s: str) -> int:
    """Cheap, dependency-free token estimate for gating."""
    return (len(s) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN


def _footer(path: str, base_tok: int, out_tok: int) -> str:
    saved = max(0, base_tok - out_tok)
    pct = int(round(100 * saved / base_tok)) if base_tok else 0
    return (
        f"\n\n{FOOTER_MARKER} Tool output compressed {base_tok}→{out_tok} tokens "
        f"(~{pct}% saved). The full verbatim original is cached at {path} — "
        f"if you need exact details that were summarized away, recover them with "
        f'read_file("{path}") or search_files.'
    )


def compress_tool_output(
    query: str,
    content: str,
    tool_name: str,
    cache_id: str,
    client: CompresrToolOutputClient,
    task_id: str = "default",
    max_cache_mb: int = 256,
    target_ratio: float = 2.0,
    cache_content: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Compress via Compresr, store the original, append a recovery footer.

    Returns ``(output_text, info)`` and never raises: on any failure it returns
    the original ``content`` with ``info["shortened"]`` False.

    ``cache_content`` is what gets persisted for recovery (defaults to
    ``content``); it may differ, e.g. a de-numbered copy of already-line-numbered
    read_file output. The API call and size gate always use ``content``.
    """
    if cache_content is None:
        cache_content = content
    base_tok = count_tokens(content)
    info: Dict[str, Any] = {
        "called_api": False,
        "base_tokens": base_tok,
        "out_tokens": base_tok,
        "shortened": False,
    }

    try:
        compressed, stats = client.compress(
            tool_output=content, query=query, tool_name=tool_name,
            coarse=True, target_ratio=target_ratio,
        )
    except Exception as e:  # fail-open to the original tool output
        logger.warning("tool_output_compresr: API failed (%s) — leaving original", e)
        info["error"] = str(e)
        return content, info

    info["called_api"] = True
    info["api_stats"] = stats

    # A whitespace-only body is truthy at the client layer but would replace the
    # output with near-nothing — treat as failure and fail open.
    if not compressed or not compressed.strip():
        info["error"] = "empty compressed output"
        return content, info

    # Cheap pre-filter before paying for the cache write / remote sync.
    body_tok = count_tokens(compressed)
    if body_tok + FOOTER_TOKEN_BUDGET >= base_tok:
        # Benign "no net win", NOT a failure — leave info["error"] unset so the
        # caller doesn't back off / count it as an error.
        info["skipped_reason"] = "not smaller"
        return content, info

    # store_original returns an agent-visible path, or None if the active backend
    # can't prove one — fail open rather than point at an unreadable file.
    cache_path = cache.store_original(cache_id, cache_content, task_id, max_cache_mb=max_cache_mb)
    if cache_path is None:
        info["error"] = "cache write failed"
        return content, info

    # Report the honest post-footer size so the footer never understates cost.
    reported_out_tok = body_tok + FOOTER_TOKEN_BUDGET
    out = compressed + _footer(cache_path, base_tok, reported_out_tok)
    # Exact gate on the REAL returned size: a long remote-home cache path can push
    # the footer past FOOTER_TOKEN_BUDGET, so fail open if it isn't a net win.
    out_tok = count_tokens(out)
    if out_tok >= base_tok:
        # Don't delete the cache entry — cache_id is content-addressed, so a
        # sibling may already hold this path; the pruner reclaims it if unused.
        # Benign "no net win" (not a failure): leave info["error"] unset.
        info["skipped_reason"] = "not smaller after footer"
        return content, info
    info.update({
        "shortened": True,
        "out_tokens": out_tok,
        "saved": max(0, base_tok - out_tok),
        "cache_path": cache_path,
    })
    return out, info
