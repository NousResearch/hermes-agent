"""Agent Workflow Synthesis (AWF) local-file integration.

v0 is intentionally offline-only: it reads configured JSON/text files and only
mutates the configured approval-events JSONL ledger.
"""

from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:  # pragma: no cover - Windows fallback; gateways here are POSIX/macOS/Linux.
    import fcntl
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

from hermes_constants import get_hermes_home

RESOLVED_DECISIONS = {"approved", "rejected"}


@dataclass(frozen=True)
class AwfConfig:
    status_path: Path
    details_dir: Path
    pending_gates_path: Path
    approval_events_path: Path
    approver_telegram_user_ids: frozenset[str]
    auto_send_cards: bool
    card_requests_path: Path
    card_results_path: Path
    card_chat_id: str
    card_thread_id: str
    card_poll_interval_seconds: float


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def extract_awf_config(config: Any) -> dict[str, Any]:
    """Return the raw ``awf`` config mapping from a runner/GatewayConfig/dict."""

    if config is None:
        return {}
    raw = _cfg_get(config, "awf", {})
    return raw if isinstance(raw, dict) else {}


def _resolve_path(value: Any, *, home: Path, default: str) -> Path:
    raw = str(value or default).strip() or default
    path = Path(os.path.expandvars(raw)).expanduser()
    if not path.is_absolute():
        path = home / path
    return path


def _resolve_optional_path(value: Any, *, home: Path) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    path = Path(os.path.expandvars(raw)).expanduser()
    if not path.is_absolute():
        path = home / path
    return path


def _bool_config(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _float_config(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def load_awf_config(config: Any = None) -> AwfConfig:
    """Load AWF local-file config from a runner/GatewayConfig/dict.

    If ``config`` is omitted, load ``gateway.config.load_gateway_config()`` so
    Telegram callback handlers can resolve gates after process restarts without
    relying on in-memory button state.
    """

    if config is None:
        from gateway.config import load_gateway_config

        config = load_gateway_config()
    raw = extract_awf_config(config)
    home = get_hermes_home()
    approvers_raw = raw.get("approver_telegram_user_ids", [])
    if isinstance(approvers_raw, str):
        approvers = [part.strip() for part in approvers_raw.split(",")]
    elif isinstance(approvers_raw, Iterable):
        approvers = [str(part).strip() for part in approvers_raw]
    else:
        approvers = []
    pending_gates_path = _resolve_path(
        raw.get("pending_gates_path"),
        home=home,
        default="awf/pending-gates.json",
    )
    card_requests_path = _resolve_optional_path(
        raw.get("card_requests_path") or raw.get("telegram_card_requests_path"),
        home=home,
    ) or (pending_gates_path.parent / "telegram-card-requests.jsonl")
    card_results_path = _resolve_optional_path(
        raw.get("card_results_path") or raw.get("telegram_card_results_path"),
        home=home,
    ) or (card_requests_path.parent / "telegram-card-results.jsonl")
    card_chat_id = str(
        raw.get("card_chat_id")
        or raw.get("telegram_card_chat_id")
        or next((v for v in approvers if v), "")
    ).strip()
    return AwfConfig(
        status_path=_resolve_path(raw.get("status_path"), home=home, default="awf/status.json"),
        details_dir=_resolve_path(raw.get("details_dir"), home=home, default="awf/details"),
        pending_gates_path=pending_gates_path,
        approval_events_path=_resolve_path(raw.get("approval_events_path"), home=home, default="awf/approval-events.jsonl"),
        approver_telegram_user_ids=frozenset(v for v in approvers if v),
        auto_send_cards=_bool_config(raw.get("auto_send_cards"), default=False),
        card_requests_path=card_requests_path,
        card_results_path=card_results_path,
        card_chat_id=card_chat_id,
        card_thread_id=str(raw.get("card_thread_id") or raw.get("telegram_card_thread_id") or "").strip(),
        card_poll_interval_seconds=_float_config(raw.get("card_poll_interval_seconds"), default=2.0),
    )


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing AWF file: {path}") from None


def _load_json_or_text(path: Path) -> Any:
    text = _read_text(path)
    stripped = text.strip()
    if not stripped:
        return {}
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return text


def _safe_detail_candidates(identifier: str, details_dir: Path) -> list[Path]:
    ident = identifier.strip()
    if not ident or "/" in ident or "\\" in ident or ident in {".", ".."} or ".." in Path(ident).parts:
        return []
    return [details_dir / f"{ident}.json", details_dir / f"{ident}.md", details_dir / f"{ident}.txt"]


def _first_str(data: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, tuple):
        return [str(v) for v in value]
    return [str(value)]


def _gate_id(gate: dict[str, Any]) -> str:
    return _first_str(gate, "gate_id", "id", "key", "approval_id")


def _gate_issue(gate: dict[str, Any]) -> str:
    return _first_str(gate, "issue", "issue_id", "issue_key", "linear_issue", "run", "run_id", "id")


def _gate_stage(gate: dict[str, Any]) -> str:
    return _first_str(gate, "stage", "gate", "phase")


def _gate_linear_url(gate: dict[str, Any]) -> str:
    return _first_str(gate, "linear_url", "linear", "issue_url", "url")


def _format_lines(title: str, lines: list[str]) -> str:
    if not lines:
        return ""
    return "\n" + title + "\n" + "\n".join(f"- `{line}`" for line in lines)


def load_pending_gates(config: AwfConfig) -> list[dict[str, Any]]:
    data = _load_json_or_text(config.pending_gates_path)
    if isinstance(data, list):
        return [g for g in data if isinstance(g, dict)]
    if isinstance(data, dict):
        for key in ("gates", "pending_gates", "pending", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return [g for g in value if isinstance(g, dict)]
        if _gate_id(data) or _gate_issue(data):
            return [data]
    return []


def find_gate(config: AwfConfig, identifier: str, stage: str | None = None) -> dict[str, Any] | None:
    identifier = str(identifier or "").strip()
    stage_norm = str(stage or "").strip().lower()
    if not identifier:
        return None
    for gate in load_pending_gates(config):
        ids = {
            _gate_id(gate),
            _gate_issue(gate),
            _first_str(gate, "run", "run_id"),
            _first_str(gate, "issue", "issue_id", "issue_key"),
        }
        if identifier not in ids:
            continue
        if stage_norm and _gate_stage(gate).lower() != stage_norm:
            continue
        return gate
    return None


def load_gate_by_id(config: AwfConfig, gate_id: str) -> dict[str, Any] | None:
    gate_id = str(gate_id or "").strip()
    for gate in load_pending_gates(config):
        if _gate_id(gate) == gate_id:
            return gate
    return None


def _load_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    result: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            result.append(item)
    return result


def _card_request_id(request: dict[str, Any]) -> str:
    return _first_str(request, "request_id", "id")


def _card_request_gate_id(request: dict[str, Any]) -> str:
    return _first_str(request, "gate_id", "gate_event_id")


def load_card_requests(config: AwfConfig) -> list[dict[str, Any]]:
    """Return AWF Telegram card send-intent requests from the configured outbox."""

    requests: list[dict[str, Any]] = []
    for request in _load_jsonl_objects(config.card_requests_path):
        if _first_str(request, "type") != "awf.telegram.card.requested":
            continue
        if not _card_request_id(request) or not _card_request_gate_id(request):
            continue
        status = _first_str(request, "status").lower()
        if status and status != "pending":
            continue
        requests.append(request)
    return requests


def sent_card_request_ids(config: AwfConfig) -> set[str]:
    sent: set[str] = set()
    for event in _load_jsonl_objects(config.card_results_path):
        if _first_str(event, "type") != "awf.telegram.card.send_result":
            continue
        if _first_str(event, "result") != "sent":
            continue
        request_id = _first_str(event, "request_id")
        if request_id:
            sent.add(request_id)
    return sent


def pending_card_requests(config: AwfConfig) -> list[dict[str, Any]]:
    sent_ids = sent_card_request_ids(config)
    return [request for request in load_card_requests(config) if _card_request_id(request) not in sent_ids]


def gate_for_card_request(config: AwfConfig, request: dict[str, Any]) -> dict[str, Any]:
    """Resolve the best gate payload for a card request.

    AWF writes compact send-intent events.  Hermes should prefer the pending
    gate object, because that contains proof commands, denied commands, and
    Linear URL.  If the pending gate is unavailable, the request itself still
    carries enough identity for a minimal approval card.
    """

    gate_id = _card_request_gate_id(request)
    gate = load_gate_by_id(config, gate_id)
    if gate is not None:
        return gate
    fallback = dict(request)
    if "issue" not in fallback and request.get("issue_id"):
        fallback["issue"] = request.get("issue_id")
    if "run" not in fallback and request.get("run_id"):
        fallback["run"] = request.get("run_id")
    return fallback


def card_request_chat_id(config: AwfConfig, request: dict[str, Any]) -> str:
    return _first_str(request, "chat_id") or config.card_chat_id


def card_request_metadata(config: AwfConfig, request: dict[str, Any]) -> dict[str, str]:
    thread_id = _first_str(request, "thread_id") or config.card_thread_id
    return {"thread_id": thread_id} if thread_id else {}


def append_card_send_result_once(
    config: AwfConfig,
    *,
    request: dict[str, Any],
    chat_id: str,
    message_id: str,
) -> tuple[bool, dict[str, Any] | None]:
    """Record a successful Telegram card send once per request_id."""

    request_id = _card_request_id(request)
    gate_id = _card_request_gate_id(request)
    if not request_id:
        raise ValueError("AWF card request must include request_id")

    config.card_results_path.parent.mkdir(parents=True, exist_ok=True)
    with config.card_results_path.open("a+", encoding="utf-8") as fh:
        if fcntl is not None:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.seek(0)
            for line in fh:
                if not line.strip():
                    continue
                try:
                    existing = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    isinstance(existing, dict)
                    and _first_str(existing, "type") == "awf.telegram.card.send_result"
                    and _first_str(existing, "result") == "sent"
                    and _first_str(existing, "request_id") == request_id
                ):
                    return False, existing
            event = {
                "type": "awf.telegram.card.send_result",
                "result": "sent",
                "request_id": request_id,
                "gate_id": gate_id,
                "chat_id": chat_id,
                "message_id": message_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            fh.seek(0, os.SEEK_END)
            fh.write(json.dumps(event, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            return True, event
        finally:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _find_details_json(config: AwfConfig, identifier: str) -> Any:
    for candidate in _safe_detail_candidates(identifier, config.details_dir):
        if candidate.exists():
            return _load_json_or_text(candidate)
    # Fallback: scan small details directory for matching id/issue/run.
    if config.details_dir.is_dir():
        for path in sorted(config.details_dir.glob("*.json")):
            try:
                data = _load_json_or_text(path)
            except Exception:
                continue
            if isinstance(data, dict):
                ids = {
                    _first_str(data, "id", "run", "run_id"),
                    _first_str(data, "issue", "issue_id", "issue_key"),
                    _gate_id(data),
                }
                if identifier in ids:
                    return data
    return None


def _format_mapping(title: str, data: dict[str, Any]) -> str:
    lines = [f"## {title}"]
    for key in ("summary", "status", "id", "run", "issue", "stage"):
        value = data.get(key)
        if value not in (None, "", []):
            label = key.replace("_", " ").title()
            lines.append(f"- {label}: {value}")
    proof = _as_list(data.get("proof_commands"))
    denied = _as_list(data.get("denied_commands"))
    if proof:
        lines.append("- Proof commands:")
        lines.extend(f"  - `{cmd}`" for cmd in proof)
    if denied:
        lines.append("- Denied commands:")
        lines.extend(f"  - `{cmd}`" for cmd in denied)
    linear_url = _gate_linear_url(data)
    if linear_url:
        lines.append(f"- Linear: {linear_url}")
    return "\n".join(lines)


def format_status(config: AwfConfig) -> str:
    data = _load_json_or_text(config.status_path)
    if isinstance(data, str):
        return f"## AWF status\n{data.strip()}"
    if not isinstance(data, dict):
        return "## AWF status\nNo status data found."
    lines = ["## AWF status"]
    summary = data.get("summary")
    if summary:
        lines.append(f"- Summary: {summary}")
    for key in ("status", "updated_at", "pending_count", "approved_count", "rejected_count"):
        if key in data:
            lines.append(f"- {key.replace('_', ' ').title()}: {data[key]}")
    runs = data.get("runs") or data.get("items") or []
    if isinstance(runs, list) and runs:
        lines.append("- Runs:")
        for run in runs[:10]:
            if isinstance(run, dict):
                rid = _first_str(run, "id", "run", "run_id") or "unknown"
                status = _first_str(run, "status") or "unknown"
                stage = _first_str(run, "stage")
                suffix = f" ({stage})" if stage else ""
                lines.append(f"  - {rid}: {status}{suffix}")
            else:
                lines.append(f"  - {run}")
    return "\n".join(lines)


def format_details(config: AwfConfig, identifier: str) -> str:
    data = _find_details_json(config, identifier)
    if data is None:
        return f"AWF details not found for `{identifier}`."
    if isinstance(data, str):
        return f"## AWF details: {identifier}\n{data.strip()}"
    if isinstance(data, dict):
        return _format_mapping(f"AWF details: {identifier}", data)
    return f"## AWF details: {identifier}\n{json.dumps(data, indent=2)}"


def format_gate_card(gate: dict[str, Any]) -> str:
    issue = _gate_issue(gate) or "unknown"
    stage = _gate_stage(gate) or "unknown"
    summary = _first_str(gate, "summary", "title", "description") or "No summary provided."
    proof = _as_list(gate.get("proof_commands"))
    denied = _as_list(gate.get("denied_commands"))
    text = f"## AWF approval gate\n- Issue: {issue}\n- Stage: {stage}\n- Summary: {summary}"
    text += _format_lines("\nProof commands:", proof)
    text += _format_lines("\nDenied commands:", denied)
    return text


def actor_from_event(event: Any) -> dict[str, str]:
    source = getattr(event, "source", None)
    return {
        "platform": str(getattr(getattr(source, "platform", None), "value", getattr(source, "platform", "")) or ""),
        "telegram_user_id": str(getattr(source, "user_id", "") or ""),
        "telegram_username": str(getattr(source, "user_name", "") or ""),
        "chat_id": str(getattr(source, "chat_id", "") or ""),
        "message_id": str(getattr(event, "message_id", "") or ""),
    }


def actor_from_callback_query(query: Any) -> dict[str, str]:
    user = getattr(query, "from_user", None)
    message = getattr(query, "message", None)
    return {
        "platform": "telegram",
        "telegram_user_id": str(getattr(user, "id", "") or ""),
        "telegram_username": str(getattr(user, "username", "") or ""),
        "telegram_first_name": str(getattr(user, "first_name", "") or ""),
        "chat_id": str(getattr(message, "chat_id", "") or ""),
        "message_id": str(getattr(message, "message_id", "") or ""),
    }


def is_approver(config: AwfConfig, actor: dict[str, str]) -> bool:
    return bool(actor.get("telegram_user_id") in config.approver_telegram_user_ids)


def _event_matches(event: dict[str, Any], *, gate_id: str, issue: str, stage: str) -> bool:
    if str(event.get("decision", "")).lower() not in RESOLVED_DECISIONS:
        return False
    if gate_id and str(event.get("gate_id", "")) == gate_id:
        return True
    return (
        str(event.get("issue", "")).strip() == issue
        and str(event.get("stage", "")).strip().lower() == stage.lower()
    )


def append_resolution_event(
    config: AwfConfig,
    *,
    gate: dict[str, Any],
    decision: str,
    actor: dict[str, str],
    reason: str = "",
) -> tuple[bool, dict[str, Any] | None]:
    """Append an approval/rejection event once.

    Returns ``(True, event)`` when appended, ``(False, existing_event)`` when the
    gate has already been resolved.
    """

    decision = decision.strip().lower()
    if decision not in RESOLVED_DECISIONS:
        raise ValueError(f"unsupported AWF decision: {decision}")
    issue = _gate_issue(gate)
    stage = _gate_stage(gate)
    gate_id = _gate_id(gate)
    if not issue or not stage:
        raise ValueError("AWF gate must include issue/run and stage")

    config.approval_events_path.parent.mkdir(parents=True, exist_ok=True)
    with config.approval_events_path.open("a+", encoding="utf-8") as fh:
        if fcntl is not None:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.seek(0)
            for line in fh:
                if not line.strip():
                    continue
                try:
                    existing = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(existing, dict) and _event_matches(existing, gate_id=gate_id, issue=issue, stage=stage):
                    return False, existing
            event = {
                "type": "awf.gate.resolved",
                "decision": decision,
                "gate_id": gate_id,
                "issue": issue,
                "stage": stage,
                "reason": reason,
                "actor": actor,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source": "telegram",
            }
            fh.seek(0, os.SEEK_END)
            fh.write(json.dumps(event, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            return True, event
        finally:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


async def handle_awf_command(event: Any, gateway_config: Any) -> str:
    cfg = load_awf_config(gateway_config)
    args = str(event.get_command_args() if hasattr(event, "get_command_args") else "").strip()
    try:
        tokens = shlex.split(args)
    except ValueError as exc:
        return f"Invalid /awf arguments: {exc}"
    if not tokens:
        return "Usage: /awf status | details <issue-or-run> | approve <issue-or-run> <stage> | reject <issue-or-run> <stage> <reason>"
    sub = tokens[0].lower()
    if sub == "status":
        try:
            return format_status(cfg)
        except FileNotFoundError as exc:
            return str(exc)
    if sub == "details":
        if len(tokens) < 2:
            return "Usage: /awf details <issue-or-run>"
        return format_details(cfg, tokens[1])
    if sub in {"approve", "reject"}:
        if len(tokens) < 3 or (sub == "reject" and len(tokens) < 4):
            if sub == "approve":
                return "Usage: /awf approve <issue-or-run> <stage>"
            return "Usage: /awf reject <issue-or-run> <stage> <reason>"
        actor = actor_from_event(event)
        if not is_approver(cfg, actor):
            return "⛔ You are not authorized to approve/reject AWF gates."
        gate = find_gate(cfg, tokens[1], tokens[2])
        if gate is None:
            return f"AWF gate not found for `{tokens[1]}` stage `{tokens[2]}`."
        reason = " ".join(tokens[3:]) if sub == "reject" else ""
        appended, _ = append_resolution_event(
            cfg,
            gate=gate,
            decision="approved" if sub == "approve" else "rejected",
            actor=actor,
            reason=reason,
        )
        if not appended:
            return "AWF gate already resolved."
        return f"✅ AWF gate {'approved' if sub == 'approve' else 'rejected'}: {_gate_issue(gate)} / {_gate_stage(gate)}"
    return "Usage: /awf status | details <issue-or-run> | approve <issue-or-run> <stage> | reject <issue-or-run> <stage> <reason>"


async def handle_awf_callback(query: Any, data: str, gateway_config: Any = None) -> tuple[str, str, bool]:
    """Resolve an ``awf:*`` Telegram callback.

    Returns ``(answer_text, edit_text, remove_buttons)``.
    """

    cfg = load_awf_config(gateway_config)
    parts = data.split(":", 2)
    if len(parts) != 3:
        return "Invalid AWF callback.", "Invalid AWF callback.", False
    _, action, gate_id = parts
    gate = load_gate_by_id(cfg, gate_id)
    if gate is None:
        return "AWF gate not found.", f"AWF gate `{gate_id}` was not found on disk.", False
    if action == "d":
        return "AWF details", format_gate_card(gate), False
    if action == "o":
        url = _gate_linear_url(gate)
        return ("Open Linear", f"Linear: {url}" if url else "No Linear URL configured for this AWF gate.", False)
    if action not in {"a", "r"}:
        return "Invalid AWF action.", "Invalid AWF action.", False
    actor = actor_from_callback_query(query)
    if not is_approver(cfg, actor):
        return "⛔ You are not authorized for this AWF gate.", "⛔ You are not authorized for this AWF gate.", False
    decision = "approved" if action == "a" else "rejected"
    reason = "Rejected from Telegram button." if action == "r" else ""
    appended, _ = append_resolution_event(cfg, gate=gate, decision=decision, actor=actor, reason=reason)
    if not appended:
        return "This gate is already resolved.", "AWF gate already resolved.", True
    label = "✅ AWF approved" if decision == "approved" else "❌ AWF rejected"
    return label, f"{label}: {_gate_issue(gate)} / {_gate_stage(gate)}", True


__all__ = [
    "AwfConfig",
    "append_card_send_result_once",
    "append_resolution_event",
    "card_request_chat_id",
    "card_request_metadata",
    "extract_awf_config",
    "find_gate",
    "format_gate_card",
    "format_status",
    "format_details",
    "gate_for_card_request",
    "handle_awf_callback",
    "handle_awf_command",
    "load_card_requests",
    "load_awf_config",
    "load_gate_by_id",
    "pending_card_requests",
    "sent_card_request_ids",
]
