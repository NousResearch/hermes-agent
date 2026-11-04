#!/usr/bin/env python3
"""replay_refusal.py — Replay a saved session's message history against the
Anthropic provider to confirm whether a specific tool result triggers
``stop_reason="refusal"`` (finish_reason=content_filter).

Usage examples::

    # Replay full session from state.db
    python scripts/replay_refusal.py --session 20261104_112940_1a4d06

    # Replay up to (and including) the 25th tool result
    python scripts/replay_refusal.py --session 20261104_112940_1a4d06 --up-to-tool-result 25

    # Replay from a JSON message-list file
    python scripts/replay_refusal.py --messages-file /tmp/session_msgs.json

    # Binary-search mode: exclude tool results one-by-one to isolate the trigger
    python scripts/replay_refusal.py --session 20261104_112940_1a4d06 --bisect

    # Specify a different state.db path
    python scripts/replay_refusal.py --session 20261104_112940_1a4d06 \\
        --db ~/.hermes/state.db

Output columns (TSV):
    run_index   tool_result_excluded   stop_reason   latency_ms   refusal_len
    sha256s of retained tool results are logged separately.

Security contract:
    Tool result RAW CONTENT is NEVER written to stdout, stderr, or any log.
    Only tool_name, content length (chars), and sha256 are reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrap — allow running from anywhere inside the repo
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent  # scripts/ is one level below root
sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "claude-fable-5-1"
_DEFAULT_MAX_TOKENS = 256  # small — we only need to see stop_reason
_ANTHROPIC_BASE_URL = "https://api.anthropic.com"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(text: str | bytes) -> str:
    if isinstance(text, str):
        text = text.encode()
    return hashlib.sha256(text).hexdigest()


def _safe_content_summary(msg: dict[str, Any]) -> dict[str, Any]:
    """Return a dict with tool name, char count, sha256 — never raw content."""
    raw = msg.get("content") or ""
    if isinstance(raw, list):
        # OpenAI-style content blocks — concatenate text parts
        raw = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in raw
        )
    char_count = len(raw)
    return {
        "tool_name": msg.get("tool_name") or msg.get("name") or "unknown",
        "char_count": char_count,
        "sha256": _sha256(raw),
    }


# ---------------------------------------------------------------------------
# State.db loading (read-only)
# ---------------------------------------------------------------------------


def _open_db_readonly(db_path: str) -> sqlite3.Connection:
    """Open an SQLite DB in read-only mode via URI."""
    uri = f"file:{Path(db_path).expanduser().resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _load_messages_from_db(
    session_id: str,
    db_path: str,
) -> list[dict[str, Any]]:
    """Load active messages for *session_id* from state.db.

    Returns raw DB rows as dicts — the caller decides how to filter/truncate.
    The DB is opened read-only; the original session is never modified.
    """
    conn = _open_db_readonly(db_path)
    try:
        rows = conn.execute(
            "SELECT role, content, tool_call_id, tool_calls, tool_name "
            "FROM messages "
            "WHERE session_id = ? AND active = 1 "
            "ORDER BY id",
            (session_id,),
        ).fetchall()
    finally:
        conn.close()

    result: list[dict[str, Any]] = []
    for row in rows:
        msg: dict[str, Any] = {
            "role": row["role"],
            "content": row["content"],
        }
        if row["tool_call_id"]:
            msg["tool_call_id"] = row["tool_call_id"]
        if row["tool_name"]:
            msg["tool_name"] = row["tool_name"]
        if row["tool_calls"]:
            try:
                msg["tool_calls"] = json.loads(row["tool_calls"])
            except (json.JSONDecodeError, TypeError):
                msg["tool_calls"] = []
        result.append(msg)
    return result


def _load_messages_from_file(path: str) -> list[dict[str, Any]]:
    """Load a JSON message list from a file."""
    with open(Path(path).expanduser(), encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}")
    return data


# ---------------------------------------------------------------------------
# Message filtering / truncation
# ---------------------------------------------------------------------------


def _tool_result_indices(messages: list[dict[str, Any]]) -> list[int]:
    """Return the list indices of all tool-result messages (role='tool')."""
    return [i for i, m in enumerate(messages) if m.get("role") == "tool"]


def _truncate_to_tool_result_n(
    messages: list[dict[str, Any]],
    n: int,
) -> list[dict[str, Any]]:
    """Return messages up to and including the *n*-th tool result (1-based).

    If the session has fewer than *n* tool results, the full list is returned.
    """
    tr_indices = _tool_result_indices(messages)
    if n <= 0 or not tr_indices:
        return messages
    if n > len(tr_indices):
        return messages
    cutoff_idx = tr_indices[n - 1]
    return messages[: cutoff_idx + 1]


def _exclude_tool_result(
    messages: list[dict[str, Any]],
    tool_result_index: int,
) -> list[dict[str, Any]]:
    """Return a copy of *messages* with the *tool_result_index*-th tool result
    (0-based among tool results) replaced by a short placeholder.

    We do NOT remove the message entirely because that would create an
    orphaned tool_use block in the preceding assistant message, causing an
    HTTP 400.  Instead we replace the content with a neutral placeholder so
    the conversation structure stays valid.

    Tool result raw content is never exposed.
    """
    tr_indices = _tool_result_indices(messages)
    if tool_result_index >= len(tr_indices):
        return messages
    target_msg_index = tr_indices[tool_result_index]
    out = [dict(m) for m in messages]
    out[target_msg_index] = {
        **out[target_msg_index],
        "content": "[tool result excluded by replay_refusal.py bisect]",
    }
    return out


# ---------------------------------------------------------------------------
# Anthropic message conversion (minimal — no tools sent, no thinking)
# ---------------------------------------------------------------------------


def _convert_to_anthropic_messages(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert OpenAI-style messages to Anthropic Messages API format.

    Returns ``(system_prompt, anthropic_messages)``.

    Rules:
    - ``role=system``  → extracted as system param (last one wins if multiple)
    - ``role=user``    → {"role": "user", "content": [{type: text, ...}]}
    - ``role=tool``    → merged into preceding or new user turn as tool_result
    - ``role=assistant`` with tool_calls → {"role": "assistant",
                           "content": [{type: tool_use, ...}, ...]}
    - ``role=assistant`` without tool_calls → {"role": "assistant",
                           "content": [{type: text, ...}]}

    This is a local minimal conversion for replay purposes.  The full
    hermes conversion (convert_messages_to_anthropic) is not called here
    because it imports agent/ internals that may not be available outside
    the full hermes environment.  For production use, prefer that function.
    """
    system: str | None = None
    result: list[dict[str, Any]] = []

    def _text_block(text: str) -> dict[str, Any]:
        return {"type": "text", "text": text or ""}

    def _content_to_str(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    parts.append(block.get("text") or block.get("content") or "")
                else:
                    parts.append(str(block))
            return "\n".join(parts)
        return str(content)

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            system = _content_to_str(content)
            continue

        if role == "user":
            result.append({
                "role": "user",
                "content": [_text_block(_content_to_str(content))],
            })
            continue

        if role == "tool":
            # Merge into the last user turn or create a new user turn.
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", "unknown"),
                "content": _content_to_str(content),
            }
            if result and result[-1]["role"] == "user":
                result[-1]["content"].append(tool_result_block)
            else:
                result.append({"role": "user", "content": [tool_result_block]})
            continue

        if role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            asst_content: list[dict[str, Any]] = []
            # Text content first
            text_part = _content_to_str(content)
            if text_part:
                asst_content.append(_text_block(text_part))
            # Tool use blocks
            for tc in tool_calls:
                fn = tc.get("function", {})
                raw_args = fn.get("arguments", "{}")
                try:
                    input_obj = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except (json.JSONDecodeError, TypeError):
                    input_obj = {"raw": raw_args}
                asst_content.append({
                    "type": "tool_use",
                    "id": tc.get("id") or tc.get("call_id", "unknown"),
                    "name": fn.get("name", "unknown"),
                    "input": input_obj,
                })
            if not asst_content:
                asst_content.append(_text_block(""))
            result.append({"role": "assistant", "content": asst_content})
            continue

        # Fallback: treat unknown roles as user
        result.append({
            "role": "user",
            "content": [_text_block(_content_to_str(content))],
        })

    # Ensure the conversation starts with a user turn
    if result and result[0]["role"] != "user":
        result.insert(0, {"role": "user", "content": [_text_block("(context)")]} )

    # Merge consecutive same-role turns (Anthropic requires strict alternation)
    merged: list[dict[str, Any]] = []
    for turn in result:
        if merged and merged[-1]["role"] == turn["role"]:
            # Extend the content list of the last turn
            prev_content = merged[-1]["content"]
            if isinstance(prev_content, list):
                merged[-1]["content"] = prev_content + turn["content"]
            else:
                merged[-1]["content"] = [_text_block(str(prev_content))] + turn["content"]
        else:
            merged.append(turn)

    return system, merged


# ---------------------------------------------------------------------------
# Credential loading
# ---------------------------------------------------------------------------


def _get_access_token() -> str | None:
    """Try to read a Claude Code OAuth access token from the standard locations.

    Returns the access token string, or None if unavailable.
    Does NOT raise — callers fall back to ANTHROPIC_API_KEY.
    """
    # Keychain (macOS) — delegate to anthropic_adapter if importable
    try:
        from agent.anthropic_adapter import read_claude_code_credentials  # type: ignore[import]

        creds = read_claude_code_credentials()
        if creds and creds.get("accessToken"):
            return creds["accessToken"]
    except Exception:
        pass

    # File fallback
    cred_path = Path.home() / ".claude" / ".credentials.json"
    try:
        data = json.loads(cred_path.read_text(encoding="utf-8"))
        return data.get("accessToken") or None
    except Exception:
        pass

    return None


def _build_client(model: str, base_url: str | None) -> Any:  # returns anthropic.Anthropic
    """Build an Anthropic SDK client using available credentials."""
    try:
        import anthropic  # type: ignore[import]
    except ImportError as exc:
        raise SystemExit(
            "anthropic SDK not installed. Run: pip install 'anthropic>=0.39.0'"
        ) from exc

    access_token = _get_access_token()
    api_key = os.environ.get("ANTHROPIC_API_KEY") or ""

    if access_token:
        # Claude Code OAuth path — Bearer auth
        import httpx

        def _add_auth(request: httpx.Request) -> httpx.Request:
            request.headers["Authorization"] = f"Bearer {access_token}"
            request.headers["anthropic-version"] = "2023-06-01"
            # Claude Code flag
            request.headers["anthropic-beta"] = (
                "claude-code-20250219,interleaved-thinking-2025-05-14"
            )
            return request

        http_client = httpx.Client(
            event_hooks={"request": [_add_auth]},
            timeout=httpx.Timeout(timeout=60.0, connect=10.0),
        )
        return anthropic.Anthropic(
            api_key="sk-ant-placeholder",  # SDK requires non-empty; overridden by hook
            http_client=http_client,
            base_url=base_url or _ANTHROPIC_BASE_URL,
            max_retries=0,
        )

    if api_key:
        return anthropic.Anthropic(
            api_key=api_key,
            base_url=base_url or _ANTHROPIC_BASE_URL,
            max_retries=0,
        )

    raise SystemExit(
        "No Anthropic credentials found.\n"
        "Set ANTHROPIC_API_KEY or ensure Claude Code credentials are present "
        "at ~/.claude/.credentials.json"
    )


# ---------------------------------------------------------------------------
# Single replay run
# ---------------------------------------------------------------------------


def _replay_once(
    client: Any,
    messages: list[dict[str, Any]],
    model: str,
    max_tokens: int,
    *,
    run_label: str = "replay",
    tool_results_summary: list[dict[str, Any]] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Send *messages* to the API and return a result dict.

    Keys: stop_reason, latency_ms, refusal_len, refusal_text_snippet.
    Raw content is never included in the return value.
    """
    system, anthropic_msgs = _convert_to_anthropic_messages(messages)

    if verbose:
        print(
            f"[{run_label}] Sending {len(anthropic_msgs)} turns "
            f"(system={'yes' if system else 'no'}) to model={model}",
            file=sys.stderr,
        )

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": anthropic_msgs,
        "max_tokens": max_tokens,
    }
    if system:
        kwargs["system"] = system

    t0 = time.monotonic()
    try:
        response = client.messages.create(**kwargs)
        latency_ms = int((time.monotonic() - t0) * 1000)

        stop_reason = response.stop_reason or "unknown"
        # Collect any text content
        refusal_text = ""
        for block in response.content or []:
            if hasattr(block, "type") and block.type == "text":
                refusal_text += block.text or ""
        refusal_len = len(refusal_text)
        return {
            "stop_reason": stop_reason,
            "latency_ms": latency_ms,
            "refusal_len": refusal_len,
            "refusal_text_snippet": refusal_text[:80] + ("…" if len(refusal_text) > 80 else ""),
            "error": None,
        }

    except Exception as exc:
        latency_ms = int((time.monotonic() - t0) * 1000)
        # Check for content_policy / refusal inside exception
        err_str = str(exc)
        stop_reason = "error"
        # Anthropic SDK raises APIStatusError for 4xx with status_code
        if hasattr(exc, "status_code") and getattr(exc, "status_code", None) == 400:
            stop_reason = "http_400"
        return {
            "stop_reason": stop_reason,
            "latency_ms": latency_ms,
            "refusal_len": 0,
            "refusal_text_snippet": err_str[:160],
            "error": err_str[:300],
        }


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


def _print_tool_results_summary(messages: list[dict[str, Any]]) -> None:
    """Print sha256/size for each retained tool result — no raw content."""
    tr_indices = _tool_result_indices(messages)
    if not tr_indices:
        print("  (no tool results in message list)", file=sys.stderr)
        return
    for rank, idx in enumerate(tr_indices, start=1):
        s = _safe_content_summary(messages[idx])
        print(
            f"  tool_result[{rank}] tool={s['tool_name']!r} "
            f"chars={s['char_count']} sha256={s['sha256'][:16]}…",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--session",
        metavar="SESSION_ID",
        help="Session ID to load from state.db (e.g. 20261104_112940_1a4d06).",
    )
    src.add_argument(
        "--messages-file",
        metavar="PATH",
        help="Path to a JSON file containing a message list (OpenAI format).",
    )

    p.add_argument(
        "--db",
        metavar="PATH",
        default=str(Path.home() / ".hermes" / "state.db"),
        help="Path to state.db (default: ~/.hermes/state.db).",
    )
    p.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=f"Model to use (default: {_DEFAULT_MODEL}).",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=_DEFAULT_MAX_TOKENS,
        help=f"Max output tokens (default: {_DEFAULT_MAX_TOKENS}). "
             "Small values are fine — we only need stop_reason.",
    )
    p.add_argument(
        "--base-url",
        metavar="URL",
        default=None,
        help="Anthropic API base URL override.",
    )
    p.add_argument(
        "--up-to-tool-result",
        type=int,
        metavar="N",
        default=0,
        help="Include messages only up to (and including) the N-th tool result "
             "(1-based). 0 = include all (default).",
    )
    p.add_argument(
        "--bisect",
        action="store_true",
        help="Bisect mode: for each tool result, replay the history with that "
             "tool result replaced by a neutral placeholder, then print which "
             "runs trigger a refusal.",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-run progress to stderr.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # --- Load messages ---
    if args.session:
        print(
            f"Loading session {args.session!r} from {args.db} (read-only)…",
            file=sys.stderr,
        )
        raw_messages = _load_messages_from_db(args.session, args.db)
    else:
        print(f"Loading messages from {args.messages_file!r}…", file=sys.stderr)
        raw_messages = _load_messages_from_file(args.messages_file)

    # --- Optional truncation ---
    if args.up_to_tool_result > 0:
        raw_messages = _truncate_to_tool_result_n(raw_messages, args.up_to_tool_result)
        print(
            f"Truncated to first {args.up_to_tool_result} tool result(s): "
            f"{len(raw_messages)} messages remain.",
            file=sys.stderr,
        )

    tr_indices = _tool_result_indices(raw_messages)
    n_tool_results = len(tr_indices)
    print(
        f"Message list: {len(raw_messages)} messages, "
        f"{n_tool_results} tool result(s).",
        file=sys.stderr,
    )

    # --- Print tool result summary (sha256/size, never raw content) ---
    print("\nTool results summary (no raw content):", file=sys.stderr)
    _print_tool_results_summary(raw_messages)
    print("", file=sys.stderr)

    # --- Build client ---
    client = _build_client(args.model, args.base_url)

    # --- TSV header ---
    print("\t".join(["run_index", "excluded_tool_result_rank", "stop_reason",
                     "latency_ms", "refusal_len", "refusal_snippet_or_error"]))

    if not args.bisect:
        # --- Single replay ---
        result = _replay_once(
            client,
            raw_messages,
            model=args.model,
            max_tokens=args.max_tokens,
            run_label="full",
            verbose=args.verbose,
        )
        row = "\t".join([
            "0",
            "none",
            result["stop_reason"],
            str(result["latency_ms"]),
            str(result["refusal_len"]),
            result.get("refusal_text_snippet") or result.get("error") or "",
        ])
        print(row)

        # Print tool result SHA256 for the offending run
        print("\nTool result hashes for this run:", file=sys.stderr)
        _print_tool_results_summary(raw_messages)

        if result["stop_reason"] in ("refusal", "content_filter"):
            print(
                "\n[REFUSAL CONFIRMED] stop_reason indicates API-level refusal. "
                "See tool result hashes above for forensics.",
                file=sys.stderr,
            )
        elif result["stop_reason"] == "stop":
            print(
                "\n[NO REFUSAL] stop_reason=stop — session replays cleanly.",
                file=sys.stderr,
            )
        else:
            print(
                f"\n[INCONCLUSIVE] stop_reason={result['stop_reason']!r}",
                file=sys.stderr,
            )

    else:
        # --- Bisect mode: exclude each tool result in turn ---
        print(
            f"Bisect mode: {n_tool_results} tool result(s) to probe.",
            file=sys.stderr,
        )
        # First run: baseline (no exclusion)
        baseline = _replay_once(
            client,
            raw_messages,
            model=args.model,
            max_tokens=args.max_tokens,
            run_label="baseline",
            verbose=args.verbose,
        )
        baseline_refused = baseline["stop_reason"] in ("refusal", "content_filter")
        print("\t".join([
            "0",
            "none (baseline)",
            baseline["stop_reason"],
            str(baseline["latency_ms"]),
            str(baseline["refusal_len"]),
            baseline.get("refusal_text_snippet") or baseline.get("error") or "",
        ]))

        refusal_trigger: list[int] = []  # 1-based ranks of tool results that trigger refusal

        for rank in range(1, n_tool_results + 1):
            excluded_msgs = _exclude_tool_result(raw_messages, rank - 1)
            # Print what we're excluding (hash/size, never content)
            excl_idx = tr_indices[rank - 1]
            excl_summary = _safe_content_summary(raw_messages[excl_idx])
            if args.verbose:
                print(
                    f"  [bisect {rank}/{n_tool_results}] excluding tool_result[{rank}]: "
                    f"tool={excl_summary['tool_name']!r} chars={excl_summary['char_count']} "
                    f"sha256={excl_summary['sha256'][:16]}…",
                    file=sys.stderr,
                )
            result = _replay_once(
                client,
                excluded_msgs,
                model=args.model,
                max_tokens=args.max_tokens,
                run_label=f"bisect-{rank}",
                verbose=False,
            )
            is_refusal = result["stop_reason"] in ("refusal", "content_filter")
            # If excluding this TR removes the refusal → it's the trigger
            baseline_refused = baseline["stop_reason"] in ("refusal", "content_filter")
            if baseline_refused and not is_refusal:
                refusal_trigger.append(rank)
                print(
                    f"  [bisect {rank}] REFUSAL CLEARED when tool_result[{rank}] is excluded "
                    f"→ LIKELY TRIGGER  "
                    f"tool={excl_summary['tool_name']!r} "
                    f"chars={excl_summary['char_count']} "
                    f"sha256={excl_summary['sha256']}",
                    file=sys.stderr,
                )
            print("\t".join([
                str(rank),
                str(rank),
                result["stop_reason"],
                str(result["latency_ms"]),
                str(result["refusal_len"]),
                result.get("refusal_text_snippet") or result.get("error") or "",
            ]))

        # --- Summary ---
        print("\n=== Bisect Summary ===", file=sys.stderr)
        if not baseline_refused:
            print(
                "Baseline did NOT trigger a refusal — bisect results show "
                "no refusal trigger to identify.",
                file=sys.stderr,
            )
        elif refusal_trigger:
            print(
                f"Likely refusal trigger(s) [1-based rank among tool results]: "
                f"{refusal_trigger}",
                file=sys.stderr,
            )
            for rank in refusal_trigger:
                excl_idx = tr_indices[rank - 1]
                s = _safe_content_summary(raw_messages[excl_idx])
                print(
                    f"  rank={rank} tool={s['tool_name']!r} "
                    f"chars={s['char_count']} sha256={s['sha256']}",
                    file=sys.stderr,
                )
        else:
            print(
                "Baseline refused but removing individual tool results did not clear "
                "the refusal — trigger may be additive (multiple messages together) "
                "or not a tool result.",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
