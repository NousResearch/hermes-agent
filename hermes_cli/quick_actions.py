"""Telegram Quick Actions local control plane.

This module intentionally keeps promotion conservative: it reviews and marks
routing candidates, but it does not directly mutate Cortex, brain-sync, or
Kanban. Downstream workers can consume ``promotions.jsonl`` after a human or
operator explicitly promotes a captured candidate.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Iterable

from hermes_constants import get_hermes_home


QA_DIRNAME = "telegram_quick_actions"
ROUTING_CANDIDATES = "routing_candidates.jsonl"
PROMOTIONS = "promotions.jsonl"
DISCARDS = "discards.jsonl"
ACTIVE_ACTIONS = "active_actions.json"


def _qa_dir(home: Path | None = None) -> Path:
    root = home or get_hermes_home()
    path = root / QA_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _jsonl_path(name: str, home: Path | None = None) -> Path:
    return _qa_dir(home) / name


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            item = {"_invalid": True, "_line_no": line_no, "raw": line}
        if isinstance(item, dict):
            item.setdefault("_line_no", line_no)
            rows.append(item)
    return rows


def _append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    tmp.replace(path)


def _candidate_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("token") or row.get("_line_no") or "")


def _shorten(text: Any, limit: int = 96) -> str:
    value = str(text or "").replace("\n", " ").strip()
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "…"


def _target_alias(target: str) -> str:
    aliases = {
        "cortex_memory": "memory",
        "cortex_todo": "todo",
        "brain_sync_wiki_candidate": "wiki",
        "kanban_candidate": "kanban",
    }
    return aliases.get(target, target)


def _candidate_targets(row: dict[str, Any]) -> list[str]:
    targets = [str(t) for t in (row.get("recommended_targets") or []) if t]
    promoted_to = row.get("promoted_to")
    if promoted_to and not targets:
        targets = [str(promoted_to)]
    return [_target_alias(t) for t in targets]


def _candidate_time(row: dict[str, Any]) -> str:
    captured = str(row.get("captured_at") or row.get("promoted_at") or row.get("discarded_at") or "")
    if not captured:
        return "time unknown"
    try:
        dt = datetime.fromisoformat(captured.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%m-%d %H:%MZ")
    except Exception:
        return captured[:16]


def format_candidate_card(row: dict[str, Any], *, index: int | None = None, verbose: bool = False) -> str:
    """Format one routing candidate for compact chat surfaces.

    This intentionally avoids TSV tables: Telegram renders dense tabular text
    poorly, so the gateway uses small review cards with the actionable command
    beside the candidate id.
    """
    cid = _candidate_id(row)
    status = str(row.get("status") or "candidate")
    action = str(row.get("action") or "?")
    prefix = f"{index}. " if index is not None else ""
    title = _shorten(row.get("title") or row.get("content") or "(untitled)", 110)
    targets = ", ".join(_candidate_targets(row)) or "no target"
    lines = [
        f"{prefix}`{cid}` · {action} · {targets}",
        title,
        f"state: {status} · captured: {_candidate_time(row)}",
    ]
    if verbose:
        content = _shorten(row.get("content") or "", 240)
        if content and content != title:
            lines.append(f"content: {content}")
    if status == "candidate":
        default_target = (_candidate_targets(row) or ["cortex_memory"])[0]
        target_for_command = {
            "memory": "cortex_memory",
            "todo": "cortex_todo",
            "wiki": "brain_sync_wiki_candidate",
            "kanban": "kanban_candidate",
        }.get(default_target, default_target)
        lines.append(f"actions: `/qa promote {cid} --to {target_for_command}` · `/qa discard {cid}`")
    else:
        lines.append(f"details: `/qa show {cid}`")
    return "\n".join(lines)


def format_candidate_digest(rows: list[dict[str, Any]], *, status: str, limit: int, verbose: bool = False) -> str:
    if not rows:
        return f"Quick Actions: no {status} candidates."
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("action") or "?")
        counts[key] = counts.get(key, 0) + 1
    mix = ", ".join(f"{key} {value}" for key, value in sorted(counts.items()))
    lines = [
        "**Quick Actions review**",
        f"showing: {len(rows)} latest · status: {status} · mix: {mix}",
        "",
    ]
    for idx, row in enumerate(rows, start=1):
        if idx > 1:
            lines.append("")
        lines.append(format_candidate_card(row, index=idx, verbose=verbose))
    lines.extend([
        "",
        f"More: `/qa list --status {status} --limit {min(max(limit * 2, 5), 25)}`",
        "Detail: `/qa show <id>`",
    ])
    return "\n".join(lines)


def _find_candidate(identifier: str, *, home: Path | None = None) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
    rows = _read_jsonl(_jsonl_path(ROUTING_CANDIDATES, home))
    ident = str(identifier)
    for idx, row in enumerate(rows):
        if _candidate_id(row) == ident or str(row.get("token") or "") == ident or str(row.get("_line_no") or "") == ident:
            return idx, row, rows
    raise SystemExit(f"No Quick Action routing candidate found for id/token/line: {identifier}")


def _format_candidate(row: dict[str, Any], *, verbose: bool = False) -> str:
    cid = _candidate_id(row)
    action = row.get("action", "?")
    status = row.get("status", "candidate")
    title = str(row.get("title") or "(untitled)").replace("\n", " ")
    targets = ",".join(str(t) for t in (row.get("recommended_targets") or [])) or "-"
    captured_at = row.get("captured_at") or "-"
    line = f"{cid}\t{status}\t{action}\t{targets}\t{captured_at}\t{title}"
    if not verbose:
        return line
    source_ref = ""
    todo = row.get("todo") if isinstance(row.get("todo"), dict) else {}
    if todo:
        source_ref = str(todo.get("source_ref") or "")
    elif row.get("chat_id") or row.get("message_id"):
        source_ref = f"telegram:{row.get('chat_id') or ''}:{row.get('thread_id') or ''}:{row.get('message_id') or ''}"
    content = str(row.get("content") or "")
    if len(content) > 500:
        content = content[:497].rstrip() + "…"
    return "\n".join([
        line,
        f"  source: {source_ref or '-'}",
        f"  content: {content}",
    ])


def list_candidates(*, status: str = "candidate", limit: int = 20, home: Path | None = None, verbose: bool = False) -> list[dict[str, Any]]:
    rows = _read_jsonl(_jsonl_path(ROUTING_CANDIDATES, home))
    if status != "all":
        rows = [r for r in rows if str(r.get("status") or "candidate") == status]
    return rows[-limit:] if limit > 0 else rows


def promote_candidate(identifier: str, *, target: str, home: Path | None = None, actor: str = "cli") -> dict[str, Any]:
    idx, row, rows = _find_candidate(identifier, home=home)
    now = datetime.now(timezone.utc).isoformat()
    updated = dict(row)
    updated["status"] = "promoted"
    updated["promoted_to"] = target
    updated["promoted_at"] = now
    updated["promoted_by"] = actor
    rows[idx] = updated
    _write_jsonl_atomic(_jsonl_path(ROUTING_CANDIDATES, home), rows)
    event = {
        "candidate_id": _candidate_id(row),
        "token": row.get("token"),
        "target": target,
        "title": row.get("title"),
        "action": row.get("action"),
        "content": row.get("content"),
        "source": row.get("source") or {},
        "promoted_at": now,
        "promoted_by": actor,
        "status": "pending_execution",
    }
    _append_jsonl(_jsonl_path(PROMOTIONS, home), event)
    return updated


def discard_candidate(identifier: str, *, reason: str = "", home: Path | None = None, actor: str = "cli") -> dict[str, Any]:
    idx, row, rows = _find_candidate(identifier, home=home)
    now = datetime.now(timezone.utc).isoformat()
    updated = dict(row)
    updated["status"] = "discarded"
    updated["discarded_at"] = now
    updated["discarded_by"] = actor
    if reason:
        updated["discard_reason"] = reason
    rows[idx] = updated
    _write_jsonl_atomic(_jsonl_path(ROUTING_CANDIDATES, home), rows)
    _append_jsonl(_jsonl_path(DISCARDS, home), {
        "candidate_id": _candidate_id(row),
        "token": row.get("token"),
        "title": row.get("title"),
        "reason": reason,
        "discarded_at": now,
        "discarded_by": actor,
    })
    return updated


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def prune_active_actions(*, older_than_days: int = 14, drop_undated: bool = False, home: Path | None = None) -> dict[str, int]:
    path = _qa_dir(home) / ACTIVE_ACTIONS
    if not path.exists():
        return {"kept": 0, "removed": 0, "total": 0}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Expected {path} to contain a JSON object")

    cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
    kept: dict[str, Any] = {}
    removed = 0
    for token, payload in data.items():
        if not isinstance(payload, dict):
            removed += 1
            continue
        created = _parse_dt(payload.get("created_at") or payload.get("recorded_at") or payload.get("captured_at"))
        if created is None:
            if drop_undated:
                removed += 1
                continue
            kept[token] = payload
            continue
        if created < cutoff:
            removed += 1
            continue
        kept[token] = payload

    if removed:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    return {"kept": len(kept), "removed": removed, "total": len(data)}


def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hermes qa", description="Review Telegram Quick Action routing candidates")
    sub = parser.add_subparsers(dest="action", required=True)

    p_list = sub.add_parser("list", aliases=["ls"], help="List routing candidates")
    p_list.add_argument("--status", default="candidate", choices=["candidate", "promoted", "discarded", "all"])
    p_list.add_argument("--limit", type=int, default=20)
    p_list.add_argument("--verbose", "-v", action="store_true")

    p_show = sub.add_parser("show", help="Show one candidate as JSON")
    p_show.add_argument("id")

    p_promote = sub.add_parser("promote", help="Mark a candidate promoted and append promotions.jsonl")
    p_promote.add_argument("id")
    p_promote.add_argument("--to", required=True, choices=["cortex", "wiki", "kanban", "cortex_memory", "cortex_todo", "brain_sync_wiki_candidate", "kanban_candidate"])

    p_discard = sub.add_parser("discard", help="Discard a routing candidate")
    p_discard.add_argument("id")
    p_discard.add_argument("--reason", default="")

    p_prune = sub.add_parser("prune-active", help="Prune stale active button payloads")
    p_prune.add_argument("--older-than-days", type=int, default=14)
    p_prune.add_argument("--drop-undated", action="store_true", help="Also remove legacy active payloads without timestamps")

    args = parser.parse_args(argv)
    if args.action in {"list", "ls"}:
        rows = list_candidates(status=args.status, limit=args.limit, verbose=args.verbose)
        print(format_candidate_digest(rows, status=args.status, limit=args.limit, verbose=args.verbose))
        return 0
    if args.action == "show":
        _, row, _ = _find_candidate(args.id)
        print(json.dumps(row, ensure_ascii=False, indent=2))
        return 0
    if args.action == "promote":
        row = promote_candidate(args.id, target=args.to)
        print(f"Promoted {_candidate_id(row)} -> {args.to}")
        return 0
    if args.action == "discard":
        row = discard_candidate(args.id, reason=args.reason)
        print(f"Discarded {_candidate_id(row)}")
        return 0
    if args.action == "prune-active":
        result = prune_active_actions(older_than_days=args.older_than_days, drop_undated=args.drop_undated)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    parser.print_help()
    return 1


def register_cli(parent: argparse.ArgumentParser) -> None:
    parent.set_defaults(func=lambda a: sys.exit(cli_main([a.qa_action] + getattr(a, "qa_args", []))))
    parent.add_argument("qa_action", nargs="?", help="list/show/promote/discard/prune-active")
    parent.add_argument("qa_args", nargs=argparse.REMAINDER)


if __name__ == "__main__":
    raise SystemExit(cli_main())
