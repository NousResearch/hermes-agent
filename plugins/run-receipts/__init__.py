"""run-receipts plugin — tamper-evident per-run evidence records for Hermes.

Every agent turn produces one JSON receipt appended to
``<HERMES_HOME>/receipts/runs.ndjson``: session/turn identity, model, provider,
platform, timings, outcome, and per-call records carrying only sha256 digests
of arguments, results and error messages — never raw content. Each record is
self-hashed and linked to its predecessor (``prev_sha256``), so ``hermes
receipts verify`` (or ``/receipts verify`` in-session) detects edited,
deleted, or reordered history.

Observer-only: callbacks accumulate in memory and the single file append
happens at ``on_session_end``. Nothing on the token path, no core changes.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from . import receipts

logger = logging.getLogger(__name__)


def _home():
    """Resolve the owning profile's home at fire time (context-local, multiplex-safe)."""
    from hermes_constants import get_hermes_home

    return get_hermes_home()


def _on_session_start(
    session_id: str = "", model: str = "", platform: str = "", **_: Any
) -> None:
    try:
        receipts.note_session_start(session_id, model=model, platform=platform)
    except Exception:
        logger.debug("run-receipts: session_start note failed", exc_info=True)


def _on_post_tool_call(
    tool_name: str = "",
    args: Any = None,
    result: Any = None,
    session_id: str = "",
    task_id: str = "",
    tool_call_id: str = "",
    duration_ms: Any = None,
    turn_id: str = "",
    **_: Any,
) -> None:
    try:
        receipts.note_tool_call(
            session_id,
            task_id=task_id,
            turn_id=turn_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            args=args,
            result=result,
            duration_ms=duration_ms,
        )
    except Exception:
        logger.debug("run-receipts: post_tool_call note failed", exc_info=True)


def _on_post_api_request(
    session_id: str = "",
    task_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    model: str = "",
    provider: str = "",
    api_duration: Any = None,
    started_at: Any = None,
    ended_at: Any = None,
    api_call_count: Any = None,
    **_: Any,
) -> None:
    try:
        receipts.note_api_call(
            session_id,
            task_id=task_id,
            turn_id=turn_id,
            api_request_id=api_request_id,
            model=model,
            provider=provider,
            api_duration=api_duration,
            started_at=started_at,
            ended_at=ended_at,
            api_call_count=api_call_count,
        )
    except Exception:
        logger.debug("run-receipts: post_api_request note failed", exc_info=True)


def _on_api_request_error(
    session_id: str = "",
    task_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    model: str = "",
    provider: str = "",
    api_duration: Any = None,
    api_call_count: Any = None,
    status_code: Any = None,
    retryable: Any = None,
    reason: str = "",
    error: Any = None,
    **_: Any,
) -> None:
    try:
        receipts.note_api_error(
            session_id,
            task_id=task_id,
            turn_id=turn_id,
            api_request_id=api_request_id,
            model=model,
            provider=provider,
            api_duration=api_duration,
            api_call_count=api_call_count,
            status_code=status_code,
            retryable=retryable,
            reason=reason,
            error=error,
        )
    except Exception:
        logger.debug("run-receipts: api_request_error note failed", exc_info=True)


def _on_session_end(
    session_id: str = "",
    task_id: str = "",
    turn_id: str = "",
    completed: bool = False,
    failed: bool = False,
    interrupted: bool = False,
    turn_exit_reason: str = "",
    model: str = "",
    platform: str = "",
    **_: Any,
) -> None:
    try:
        record = receipts.finalize_run(
            session_id,
            task_id=task_id,
            turn_id=turn_id,
            completed=completed,
            failed=failed,
            interrupted=interrupted,
            turn_exit_reason=turn_exit_reason,
            model=model,
            platform=platform,
        )
        if record is not None:
            receipts.write_receipt(_home(), record)
    except Exception:
        logger.warning("run-receipts: failed to write run receipt", exc_info=True)


def _on_session_finalize(session_id: str = "", **_: Any) -> None:
    # Session close without a turn boundary — drop any pending residue so it
    # cannot leak into a later session reusing the id space.
    try:
        receipts.drop_pending(session_id)
    except Exception:
        logger.debug("run-receipts: session_finalize cleanup failed", exc_info=True)


# --------------------------------------------------------------------------
# Commands


def _resolve_path(file_arg: Optional[str]):
    from pathlib import Path

    if file_arg:
        return Path(file_arg)
    return receipts.receipts_path(_home())


def _receipts_slash(raw_args: str) -> Optional[str]:
    argv = raw_args.strip().split()
    sub = argv[0] if argv else "latest"
    file_arg = None
    if "--file" in argv:
        i = argv.index("--file")
        if i + 1 < len(argv):
            file_arg = argv[i + 1]
    path = _resolve_path(file_arg)
    if sub == "verify":
        return receipts.format_verify(receipts.verify_chain(path))
    if sub == "stats":
        return receipts.format_stats(path)
    if sub == "latest":
        return receipts.format_latest(path)
    return (
        "Unknown subcommand.\n\n"
        "Usage: /receipts [latest|stats|verify] [--file <path>]\n"
        "  latest  — print the most recent run receipt\n"
        "  stats   — outcome/tool/api totals across the chain\n"
        "  verify  — walk the hash chain, report the first break"
    )


def _cli_setup(subparser) -> None:
    subparser.add_argument(
        "action",
        nargs="?",
        default="latest",
        choices=["latest", "stats", "verify"],
        help="latest (default): print most recent receipt; stats: totals; verify: walk the hash chain",
    )
    subparser.add_argument(
        "--file",
        default=None,
        help="receipts file (default: <HERMES_HOME>/receipts/runs.ndjson)",
    )


def _cli_handler(args) -> int:
    path = _resolve_path(getattr(args, "file", None))
    action = getattr(args, "action", "latest")
    if action == "verify":
        report = receipts.verify_chain(path)
        print(receipts.format_verify(report))
        return 0 if report["ok"] else 1
    if action == "stats":
        print(receipts.format_stats(path))
        return 0
    print(receipts.format_latest(path))
    return 0


def register(ctx) -> None:
    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("post_api_request", _on_post_api_request)
    ctx.register_hook("api_request_error", _on_api_request_error)
    ctx.register_hook("on_session_end", _on_session_end)
    ctx.register_hook("on_session_finalize", _on_session_finalize)
    ctx.register_command(
        "receipts",
        handler=_receipts_slash,
        description="Show or verify hash-chained run receipts.",
        args_hint="[latest|stats|verify] [--file <path>]",
    )
    ctx.register_cli_command(
        "receipts",
        help="Show or verify hash-chained run receipts",
        setup_fn=_cli_setup,
        handler_fn=_cli_handler,
        description="Inspect the append-only run receipt ledger (latest / stats / verify).",
    )
