"""Opt-in log-reconstruction desync check (dev invariant).

Detects **silent loss after known wire transforms** — not full ``api_kwargs``
re-derivation. Projects send-path history shaping onto the live transcript,
then requires that the non-request-only portion of ``api_messages`` is a
contiguous content-compatible **suffix** of that projection.

Covers: ``api_content`` prefer; thinking-only drop + adjacent-user merge;
string strip; cache-control shape (text flatten); skip system/prefills;
current-user **prefix** injection only; strip ``_compressed_summary``.

Default off (zero cost). Soft log by default; raise only if
``agent.log_reconstruction_check_raise``. Model meta never raises.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

_COMPRESSED_SUMMARY_KEY = "_compressed_summary"  # context_compressor constant


class LogReconstructionDesyncError(RuntimeError):
    """Raised when outgoing LLM history diverges from the live transcript."""

    def __init__(self, message: str, *, diff: Optional[List[str]] = None):
        self.diff = list(diff or [])
        full = message
        if self.diff:
            full = message + "\n" + "\n".join(f"  - {line}" for line in self.diff)
        super().__init__(full)


@dataclass
class ReconstructionReport:
    ok: bool
    mismatches: List[str] = field(default_factory=list)
    expected_turns: int = 0
    actual_turns: int = 0


def is_log_reconstruction_check_enabled(agent: Any) -> bool:
    return bool(getattr(agent, "log_reconstruction_check", False))


def is_log_reconstruction_raise_enabled(agent: Any) -> bool:
    return bool(getattr(agent, "log_reconstruction_check_raise", False))


def _stable_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        return repr(value)


def _normalize_content_for_compare(content: Any) -> str:
    """Canonical text across cache_control shape rewrites + strip()."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                if block.strip():
                    parts.append(block.strip())
                continue
            if not isinstance(block, Mapping):
                parts.append(_stable_json(block))
                continue
            btype = block.get("type")
            if btype in (None, "text"):
                text = block.get("text", "")
                parts.append(text if isinstance(text, str) else _stable_json(text))
            elif btype in ("thinking", "redacted_thinking"):
                continue
            else:
                cleaned = {k: v for k, v in block.items() if k != "cache_control"}
                parts.append(_stable_json(cleaned))
        return "".join(parts).strip()
    return _stable_json(content).strip()


def _content_fingerprint(content: Any) -> str:
    raw = _normalize_content_for_compare(content)
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()


def _tool_call_fingerprints(msg: Mapping[str, Any]) -> tuple[str, ...]:
    """Integrity: id + function name + arguments."""
    tcs = msg.get("tool_calls") or []
    if not isinstance(tcs, list):
        return ()
    out: list[str] = []
    for tc in tcs:
        if not isinstance(tc, Mapping):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), Mapping) else {}
        fn = fn or {}
        args = fn.get("arguments")
        if not isinstance(args, str):
            args = _stable_json(args)
        else:
            try:
                args = json.dumps(
                    json.loads(args), separators=(",", ":"), sort_keys=True
                )
            except Exception:
                args = args.strip()
        payload = {
            "id": str(tc.get("id") or ""),
            "name": str(fn.get("name") or ""),
            "arguments": args,
        }
        out.append(_content_fingerprint(payload)[:24])
    return tuple(out)


def _effective_content(msg: Mapping[str, Any], *, prefer_api_content: bool) -> Any:
    content = msg.get("content")
    if not prefer_api_content:
        return content
    api_content = msg.get("api_content")
    if isinstance(api_content, str) and api_content:
        return api_content
    if api_content is not None and not isinstance(api_content, str):
        return api_content
    return content


def project_turn(
    msg: Mapping[str, Any], *, prefer_api_content: bool = True
) -> dict[str, Any]:
    """Project one message to durable comparison fields."""
    return {
        "role": str(msg.get("role") or ""),
        "content_fingerprint": _content_fingerprint(
            _effective_content(msg, prefer_api_content=prefer_api_content)
        ),
        "tool_call_fingerprints": _tool_call_fingerprints(msg),
        "tool_call_id": str(msg.get("tool_call_id") or ""),
    }


def extract_api_history(
    api_messages: Sequence[Mapping[str, Any]] | None,
    *,
    prefill_count: int = 0,
) -> List[Mapping[str, Any]]:
    """Skip leading system + ``prefill_count`` ephemeral prefills."""
    msgs = [m for m in (api_messages or []) if isinstance(m, Mapping)]
    if not msgs:
        return []
    idx = 1 if msgs[0].get("role") == "system" else 0
    idx += max(0, int(prefill_count or 0))
    return list(msgs[idx:]) if idx <= len(msgs) else []


def _is_compressed_summary_msg(msg: Mapping[str, Any]) -> bool:
    if msg.get(_COMPRESSED_SUMMARY_KEY):
        return True
    try:
        from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY as _k

        return bool(msg.get(_k))
    except Exception:
        return False


def project_live_through_wire_transforms(
    messages: Sequence[Mapping[str, Any]] | None,
    *,
    drop_codex_reasoning_items: bool = True,
) -> List[dict]:
    """api_content prefer → drop thinking-only + merge users → strip strings."""
    projected: List[dict] = []
    for msg in messages or []:
        if not isinstance(msg, Mapping):
            continue
        role = msg.get("role")
        if role is None or role == "system":
            continue
        row: dict[str, Any] = {
            "role": role,
            "content": _effective_content(msg, prefer_api_content=True),
        }
        for key in ("tool_calls", "tool_call_id", "name"):
            if msg.get(key) is not None:
                row[key] = msg.get(key)
        for key in (
            "reasoning",
            "reasoning_content",
            "reasoning_details",
            "codex_reasoning_items",
            "_thinking_prefill",
        ):
            if key in msg:
                row[key] = msg[key]
        projected.append(row)

    from agent.agent_runtime_helpers import drop_thinking_only_and_merge_users

    projected = drop_thinking_only_and_merge_users(
        projected, drop_codex_reasoning_items=drop_codex_reasoning_items
    )
    for am in projected:
        if isinstance(am.get("content"), str):
            am["content"] = am["content"].strip()
    return projected


def _content_compatible(
    expected_msg: Mapping[str, Any],
    actual_msg: Mapping[str, Any],
    *,
    is_current_user: bool,
) -> bool:
    if _content_fingerprint(expected_msg.get("content")) == _content_fingerprint(
        actual_msg.get("content")
    ):
        return True
    if not (is_current_user and expected_msg.get("role") == "user"):
        return False
    exp_raw = _normalize_content_for_compare(expected_msg.get("content"))
    act_raw = _normalize_content_for_compare(actual_msg.get("content"))
    # Empty expected: only empty api (stamped empty↔empty). Injection must not
    # mask a wiped current-user body.
    if not exp_raw:
        return not act_raw
    # Prefix only — not substring (short "a" must not match arbitrary text).
    return act_raw == exp_raw or act_raw.startswith(exp_raw)


def _turns_compatible(
    live_msg: Mapping[str, Any],
    act_msg: Mapping[str, Any],
    *,
    is_current_user: bool,
) -> tuple[bool, str]:
    role = str(live_msg.get("role") or "")
    if str(act_msg.get("role") or "") != role:
        return False, f"role mismatch live={role!r} api={act_msg.get('role')!r}"
    if not _content_compatible(live_msg, act_msg, is_current_user=is_current_user):
        live_fp = project_turn(live_msg, prefer_api_content=False)[
            "content_fingerprint"
        ][:12]
        api_fp = project_turn(act_msg, prefer_api_content=False)[
            "content_fingerprint"
        ][:12]
        return False, f"content fingerprint drift (live_fp={live_fp} api_fp={api_fp})"
    if role == "assistant" and _tool_call_fingerprints(
        live_msg
    ) != _tool_call_fingerprints(act_msg):
        return False, (
            f"tool_calls integrity mismatch "
            f"(live={_tool_call_fingerprints(live_msg)} "
            f"api={_tool_call_fingerprints(act_msg)})"
        )
    if role == "tool":
        live_tcid = str(live_msg.get("tool_call_id") or "")
        act_tcid = str(act_msg.get("tool_call_id") or "")
        if live_tcid and act_tcid and live_tcid != act_tcid:
            return False, f"tool_call_id mismatch (live={live_tcid!r} api={act_tcid!r})"
    return True, ""


def _resolve_projected_current_user(
    messages: Sequence[Mapping[str, Any]],
    live: Sequence[Mapping[str, Any]],
    current_turn_user_idx: int,
) -> Optional[int]:
    orig = list(messages)
    if not (0 <= current_turn_user_idx < len(orig)):
        return None
    cur = orig[current_turn_user_idx]
    if not (isinstance(cur, Mapping) and cur.get("role") == "user"):
        return None
    cur_fp = project_turn(cur)["content_fingerprint"]
    cur_norm = _normalize_content_for_compare(
        cur.get("api_content") or cur.get("content")
    )
    for i in range(len(live) - 1, -1, -1):
        if live[i].get("role") != "user":
            continue
        if (
            project_turn(live[i], prefer_api_content=False)["content_fingerprint"]
            == cur_fp
        ):
            return i
        live_norm = _normalize_content_for_compare(live[i].get("content"))
        if cur_norm and live_norm.endswith(cur_norm):
            return i
    for i in range(len(live) - 1, -1, -1):
        if live[i].get("role") == "user":
            return i
    return None


def compare_history_to_api_messages(
    messages: Sequence[Mapping[str, Any]] | None,
    api_messages: Sequence[Mapping[str, Any]] | None,
    *,
    prefill_count: int = 0,
    current_turn_user_idx: Optional[int] = None,
    drop_codex_reasoning_items: bool = True,
    apply_wire_transforms: bool = True,
) -> ReconstructionReport:
    """Require non-summary API history ⊆ contiguous suffix of projected live."""
    if apply_wire_transforms:
        live = project_live_through_wire_transforms(
            messages, drop_codex_reasoning_items=drop_codex_reasoning_items
        )
    else:
        live = [dict(m) for m in (messages or []) if isinstance(m, Mapping)]

    api_hist = [
        m
        for m in extract_api_history(api_messages, prefill_count=prefill_count)
        if not _is_compressed_summary_msg(m)
    ]
    mismatches: List[str] = []

    projected_current: Optional[int] = None
    if current_turn_user_idx is not None and messages is not None:
        projected_current = _resolve_projected_current_user(
            messages, live, current_turn_user_idx
        )

    live_tool_ids = {
        str(m.get("tool_call_id"))
        for m in live
        if m.get("role") == "tool" and m.get("tool_call_id")
    }
    api_for_suffix: List[Mapping[str, Any]] = []
    for m in api_hist:
        if m.get("role") == "tool":
            tcid = str(m.get("tool_call_id") or "")
            if tcid and tcid not in live_tool_ids:
                continue  # wire-only stub
        api_for_suffix.append(m)

    if not api_for_suffix:
        if live:
            mismatches.append(
                "outgoing api history empty while projected live transcript "
                f"has {len(live)} turn(s) (silent loss)"
            )
        return ReconstructionReport(
            ok=not mismatches,
            mismatches=mismatches,
            expected_turns=len(live),
            actual_turns=0,
        )

    if len(api_for_suffix) > len(live):
        mismatches.append(
            f"outgoing api history longer than projected live "
            f"(api={len(api_for_suffix)} live={len(live)})"
        )
        return ReconstructionReport(
            ok=False,
            mismatches=mismatches,
            expected_turns=len(live),
            actual_turns=len(api_for_suffix),
        )

    suffix = live[-len(api_for_suffix) :]
    for i, (live_msg, act_msg) in enumerate(zip(suffix, api_for_suffix)):
        live_i = len(live) - len(api_for_suffix) + i
        is_cur = projected_current is not None and live_i == projected_current
        ok, reason = _turns_compatible(live_msg, act_msg, is_current_user=is_cur)
        if not ok:
            mismatches.append(f"live[{live_i}] role={live_msg.get('role')}: {reason}")

    return ReconstructionReport(
        ok=not mismatches,
        mismatches=mismatches,
        expected_turns=len(live),
        actual_turns=len(api_for_suffix),
    )


def compare_request_meta(
    *,
    agent: Any = None,
    api_kwargs: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    """Soft-only notes; never fed into the raise path."""
    notes: List[str] = []
    if not api_kwargs or agent is None:
        return notes
    model = getattr(agent, "model", None)
    sent_model = api_kwargs.get("model")
    if model is not None and sent_model is not None and str(sent_model) != str(model):
        notes.append(
            f"model differs (informational, not a desync): "
            f"agent={model!r} request={sent_model!r}"
        )
    return notes


def check_log_reconstruction(
    agent: Any,
    *,
    messages: Sequence[Mapping[str, Any]] | None,
    api_messages: Sequence[Mapping[str, Any]] | None,
    api_kwargs: Optional[Mapping[str, Any]] = None,
    current_turn_user_idx: Optional[int] = None,
    raise_on_desync: Optional[bool] = None,
) -> ReconstructionReport:
    """Opt-in desync check. Soft by default; raise only when raise flag is on."""
    if not is_log_reconstruction_check_enabled(agent):
        return ReconstructionReport(ok=True)

    if raise_on_desync is None:
        raise_on_desync = is_log_reconstruction_raise_enabled(agent)

    prefill = getattr(agent, "prefill_messages", None) or []
    prefill_count = len(prefill) if isinstance(prefill, list) else 0
    drop_codex = getattr(agent, "api_mode", None) != "codex_responses"

    report = compare_history_to_api_messages(
        messages,
        api_messages,
        prefill_count=prefill_count,
        current_turn_user_idx=current_turn_user_idx,
        drop_codex_reasoning_items=drop_codex,
        apply_wire_transforms=True,
    )

    meta_notes = compare_request_meta(agent=agent, api_kwargs=api_kwargs)
    if meta_notes:
        logger.info("log-reconstruction meta: %s", "; ".join(meta_notes))

    if not report.ok:
        logger.warning(
            "log-reconstruction desync (soft=%s): %s",
            not raise_on_desync,
            "; ".join(report.mismatches[:8]),
        )
        if raise_on_desync:
            raise LogReconstructionDesyncError(
                "log-reconstruction desync: outgoing LLM history diverges from "
                "the live session transcript after known wire transforms",
                diff=report.mismatches,
            )
    return report


def maybe_check_before_request(
    agent: Any,
    *,
    messages: Sequence[Mapping[str, Any]] | None,
    api_messages: Sequence[Mapping[str, Any]] | None,
    api_kwargs: Optional[Mapping[str, Any]] = None,
    current_turn_user_idx: Optional[int] = None,
) -> Optional[ReconstructionReport]:
    """Request-finalization entrypoint. Single ``if`` when disabled."""
    if not is_log_reconstruction_check_enabled(agent):
        return None
    return check_log_reconstruction(
        agent,
        messages=messages,
        api_messages=api_messages,
        api_kwargs=api_kwargs,
        current_turn_user_idx=current_turn_user_idx,
        raise_on_desync=None,
    )
