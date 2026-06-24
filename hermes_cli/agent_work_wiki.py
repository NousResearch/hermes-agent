from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional
import argparse

DEFAULT_MAX_EXCERPT_CHARS = 8192
DEFAULT_KEEP_LATEST_RECEIPTS = 50
DEFAULT_MAX_AGE_SECONDS = 14 * 24 * 60 * 60
_TRUNC_MARKER = "\n...[truncated]...\n"
_GITIGNORE_LINES = (
    ".hermes/work/tmp/",
    ".hermes/work/receipts/",
    ".hermes/wiki/handoffs/tmp/",
)


@dataclass(frozen=True)
class WorkWikiWriteResult:
    root: Path
    receipt_path: Path
    handoff_path: Path
    schema_path: Path
    index_path: Path
    log_path: Path


@dataclass(frozen=True)
class PruneResult:
    deleted_count: int
    bytes_deleted: int
    deleted_paths: tuple[Path, ...]


def build_parser(parent_subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = parent_subparsers.add_parser(
        "work-wiki",
        help="Write bounded agent-work receipts and durable wiki handoffs",
    )
    sub = parser.add_subparsers(dest="work_wiki_action")

    p_record = sub.add_parser("record", help="Write one receipt and handoff bundle")
    p_record.add_argument("--root", default=".", help="Repository root")
    src = p_record.add_mutually_exclusive_group(required=True)
    src.add_argument("--receipt-file", default=None, help="JSON file containing receipt input")
    src.add_argument("--receipt-json", default=None, help="Inline JSON receipt input")
    p_record.add_argument("--now", default=None, help="ISO timestamp override")
    p_record.add_argument("--max-excerpt-chars", type=int, default=DEFAULT_MAX_EXCERPT_CHARS)
    p_record.add_argument("--json", action="store_true", help="Emit JSON")

    p_gc = sub.add_parser("gc", help="Prune bounded runtime evidence")
    p_gc.add_argument("--root", default=".", help="Repository root")
    p_gc.add_argument("--now-timestamp", type=float, default=None)
    p_gc.add_argument("--max-age-seconds", type=int, default=DEFAULT_MAX_AGE_SECONDS)
    p_gc.add_argument("--keep-latest-receipts", type=int, default=DEFAULT_KEEP_LATEST_RECEIPTS)
    p_gc.add_argument("--json", action="store_true", help="Emit JSON")

    return parser


def agent_work_wiki_command(args: argparse.Namespace) -> int:
    action = getattr(args, "work_wiki_action", None)
    if action == "record":
        receipt = _load_receipt_arg(args)
        result = write_work_wiki_bundle(
            getattr(args, "root", "."),
            receipt,
            now=getattr(args, "now", None),
            max_excerpt_chars=getattr(args, "max_excerpt_chars", DEFAULT_MAX_EXCERPT_CHARS),
        )
        payload = {
            "ok": True,
            "receipt_path": result.receipt_path.as_posix(),
            "handoff_path": result.handoff_path.as_posix(),
            "schema_path": result.schema_path.as_posix(),
            "index_path": result.index_path.as_posix(),
            "log_path": result.log_path.as_posix(),
        }
        _emit(payload, json_output=bool(getattr(args, "json", False)))
        return 0
    if action == "gc":
        result = prune_work_artifacts(
            getattr(args, "root", "."),
            now_timestamp=getattr(args, "now_timestamp", None),
            max_age_seconds=getattr(args, "max_age_seconds", DEFAULT_MAX_AGE_SECONDS),
            keep_latest_receipts=getattr(args, "keep_latest_receipts", DEFAULT_KEEP_LATEST_RECEIPTS),
        )
        payload = {
            "ok": True,
            "deleted_count": result.deleted_count,
            "bytes_deleted": result.bytes_deleted,
            "deleted_paths": [p.as_posix() for p in result.deleted_paths],
        }
        _emit(payload, json_output=bool(getattr(args, "json", False)))
        return 0
    raise SystemExit("usage: hermes work-wiki {record,gc} ...")


def _load_receipt_arg(args: argparse.Namespace) -> Mapping[str, Any]:
    if getattr(args, "receipt_json", None):
        data = json.loads(args.receipt_json)
    else:
        data = json.loads(Path(args.receipt_file).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise SystemExit("receipt must be a JSON object")
    return data


def _emit(payload: Mapping[str, Any], *, json_output: bool) -> None:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if json_output:
        print(text)
    else:
        print(text)


def normalize_receipt(
    receipt: Mapping[str, Any],
    *,
    max_excerpt_chars: int = DEFAULT_MAX_EXCERPT_CHARS,
    now: Optional[str] = None,
) -> dict[str, Any]:
    goal = _clean_str(receipt.get("goal")) or "untitled"
    out: dict[str, Any] = {
        "schema": "hermes.agent_work_receipt.v1",
        "created_at": now or _now_iso(),
        "goal": goal,
        "changed_files": _clean_str_list(receipt.get("changed_files")),
        "commands_run": [
            _normalize_command(item, max_excerpt_chars=max_excerpt_chars)
            for item in _as_list(receipt.get("commands_run"))
            if isinstance(item, Mapping)
        ],
        "skipped_checks": _clean_mapping_list(receipt.get("skipped_checks")),
        "open_risks": _clean_str_list(receipt.get("open_risks")),
        "artifacts": _clean_str_list(receipt.get("artifacts")),
        "decisions": _clean_str_list(receipt.get("decisions")),
        "next_prompt": _cap_text(_clean_str(receipt.get("next_prompt")), max_excerpt_chars),
    }
    if _clean_str(receipt.get("status")):
        out["status"] = _clean_str(receipt.get("status"))
    if _as_list(receipt.get("acceptance")):
        out["acceptance"] = _clean_mapping_list(receipt.get("acceptance"))
    return out


def write_work_wiki_bundle(
    root: str | Path,
    receipt: Mapping[str, Any],
    *,
    now: Optional[str] = None,
    max_excerpt_chars: int = DEFAULT_MAX_EXCERPT_CHARS,
) -> WorkWikiWriteResult:
    root_path = Path(root)
    ts_iso = now or _now_iso()
    stamp = _stamp(ts_iso)
    normalized = normalize_receipt(
        receipt,
        max_excerpt_chars=max_excerpt_chars,
        now=ts_iso,
    )
    slug = _slug(normalized["goal"])

    hermes_dir = root_path / ".hermes"
    receipts_dir = hermes_dir / "work" / "receipts"
    wiki_dir = hermes_dir / "wiki"
    handoffs_dir = wiki_dir / "handoffs"
    for path in (receipts_dir, handoffs_dir, wiki_dir / "goals", wiki_dir / "decisions", wiki_dir / "mistakes", wiki_dir / "projects"):
        path.mkdir(parents=True, exist_ok=True)
    (hermes_dir / "work" / "tmp").mkdir(parents=True, exist_ok=True)

    receipt_path = receipts_dir / f"{stamp}-{slug}.json"
    handoff_path = handoffs_dir / f"{stamp}-{slug}.md"
    schema_path = wiki_dir / "SCHEMA.md"
    index_path = wiki_dir / "index.md"
    log_path = wiki_dir / "log.md"

    normalized["id"] = f"{stamp}-{slug}"
    normalized["receipt_path"] = _rel(root_path, receipt_path)
    normalized["handoff_path"] = _rel(root_path, handoff_path)

    _write_json(receipt_path, normalized)
    _ensure_schema(schema_path)
    _ensure_index(index_path)
    _ensure_log(log_path)
    _write_handoff(handoff_path, normalized, root_path=root_path)
    _append_json_line(index_path, {"type": "handoff", "path": _rel(root_path, handoff_path), "receipt": _rel(root_path, receipt_path), "goal": normalized["goal"], "ts": ts_iso})
    _append_json_line(log_path, {"action": "write", "type": "agent_work_wiki_bundle", "path": _rel(root_path, handoff_path), "receipt": _rel(root_path, receipt_path), "ts": ts_iso})
    _ensure_gitignore(root_path)

    return WorkWikiWriteResult(
        root=root_path,
        receipt_path=receipt_path,
        handoff_path=handoff_path,
        schema_path=schema_path,
        index_path=index_path,
        log_path=log_path,
    )


def prune_work_artifacts(
    root: str | Path,
    *,
    now_timestamp: Optional[float] = None,
    max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
    keep_latest_receipts: int = DEFAULT_KEEP_LATEST_RECEIPTS,
) -> PruneResult:
    root_path = Path(root)
    now = float(now_timestamp if now_timestamp is not None else time.time())
    cutoff = now - max(0, int(max_age_seconds))
    deleted: list[Path] = []
    bytes_deleted = 0

    tmp_dir = root_path / ".hermes" / "work" / "tmp"
    if tmp_dir.exists():
        for path in sorted((p for p in tmp_dir.rglob("*") if p.is_file()), key=lambda p: str(p)):
            if path.stat().st_mtime <= cutoff:
                size = path.stat().st_size
                path.unlink()
                deleted.append(path)
                bytes_deleted += size
        _remove_empty_dirs(tmp_dir)

    receipts_dir = root_path / ".hermes" / "work" / "receipts"
    if receipts_dir.exists():
        receipts = sorted(
            (p for p in receipts_dir.rglob("*") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        keep = set(receipts[: max(0, int(keep_latest_receipts))])
        for path in receipts:
            if path in keep:
                continue
            if path.stat().st_mtime <= cutoff:
                size = path.stat().st_size
                path.unlink()
                deleted.append(path)
                bytes_deleted += size
        _remove_empty_dirs(receipts_dir)

    return PruneResult(
        deleted_count=len(deleted),
        bytes_deleted=bytes_deleted,
        deleted_paths=tuple(deleted),
    )


def _normalize_command(item: Mapping[str, Any], *, max_excerpt_chars: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "command": _clean_str(item.get("command")),
    }
    if "exit_code" in item:
        try:
            out["exit_code"] = int(item.get("exit_code"))
        except Exception:
            out["exit_code"] = item.get("exit_code")
    if _clean_str(item.get("summary")):
        out["summary"] = _clean_str(item.get("summary"))
    if _clean_str(item.get("status")):
        out["status"] = _clean_str(item.get("status"))
    excerpt = _clean_str(item.get("excerpt"))
    if not excerpt:
        parts = []
        stdout = _clean_str(item.get("stdout"))
        stderr = _clean_str(item.get("stderr"))
        if stdout:
            parts.append(f"stdout:\n{stdout}")
        if stderr:
            parts.append(f"stderr:\n{stderr}")
        excerpt = "\n".join(parts)
    if excerpt:
        out["excerpt"] = _cap_text(excerpt, max_excerpt_chars)
    return out


def _cap_text(text: str, max_chars: int) -> str:
    text = text or ""
    max_chars = max(0, int(max_chars))
    if len(text) <= max_chars:
        return text
    if max_chars <= 0:
        return ""
    if max_chars <= len(_TRUNC_MARKER) + 2:
        return text[:max_chars]
    budget = max_chars - len(_TRUNC_MARKER)
    head = budget // 2
    tail = budget - head
    return text[:head] + _TRUNC_MARKER + text[-tail:]


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _clean_str_list(value: Any) -> list[str]:
    out = []
    for item in _as_list(value):
        clean = _clean_str(item)
        if clean:
            out.append(clean)
    return out


def _clean_mapping_list(value: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in _as_list(value):
        if isinstance(item, Mapping):
            clean = {str(k): v for k, v in item.items() if v not in (None, "")}
            if clean:
                out.append(clean)
        else:
            clean_str = _clean_str(item)
            if clean_str:
                out.append({"value": clean_str})
    return out


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_handoff(path: Path, receipt: Mapping[str, Any], *, root_path: Path) -> None:
    payload = json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2)
    content = "\n".join(
        [
            "---",
            "type: agent_work_handoff",
            f"id: {receipt.get('id', '')}",
            f"created_at: {receipt.get('created_at', '')}",
            f"receipt: {_rel(root_path, Path(str(receipt.get('receipt_path', '')))) if receipt.get('receipt_path') else ''}",
            "---",
            "```json",
            payload,
            "```",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def _ensure_schema(path: Path) -> None:
    if path.exists():
        return
    path.write_text(
        "\n".join(
            [
                "# agent_work_wiki_schema_v1",
                "domain: repo-local agent work evidence",
                "layers: work.receipts=bounded execution summaries; wiki=durable curated memory; work.tmp=gc target",
                "rules: no full logs; cap excerpts; skipped checks require reason; wiki pages preserve decisions/handoffs/mistakes only",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _ensure_index(path: Path) -> None:
    if path.exists():
        return
    path.write_text("# agent_work_wiki_index_v1\n", encoding="utf-8")


def _ensure_log(path: Path) -> None:
    if path.exists():
        return
    path.write_text("# agent_work_wiki_log_v1\n", encoding="utf-8")


def _append_json_line(path: Path, data: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def _ensure_gitignore(root: Path) -> None:
    path = root / ".gitignore"
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = current.splitlines()
    changed = False
    for item in _GITIGNORE_LINES:
        if item not in lines:
            lines.append(item)
            changed = True
    if changed or not path.exists():
        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _stamp(value: str) -> str:
    dt = _parse_iso(value)
    return dt.strftime("%Y%m%dT%H%M%SZ")


def _parse_iso(value: str) -> datetime:
    cleaned = value.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _slug(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text[:80] or "untitled"


def _rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _remove_empty_dirs(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass
