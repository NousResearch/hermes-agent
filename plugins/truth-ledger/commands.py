from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Awaitable

try:
    from .processor import process_pending as _process_pending
except ImportError:
    # commands.py is also loaded as a standalone module by focused unit tests.
    _process_pending = None

_FACT_ID_RE = re.compile(r"^fact_[A-Za-z0-9_-]+$")


def _load_local_module(module_name: str):
    module_path = Path(__file__).with_name(f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(f"truth_ledger_{module_name}", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module {module_name} from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ledger_mod():
    return _load_local_module("ledger")


def _projection_mod():
    return _load_local_module("projection")


def _mkdir_private(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass


def _chmod_private_file(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_ledger_events(root: Path):
    ledger_dir = root / "ledger"
    if not ledger_dir.exists():
        return
    for ledger_file in sorted(ledger_dir.glob("*.jsonl")):
        for raw_line in ledger_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                break
            yield ledger_file, event


def _find_fact_event(root: Path, fact_id: str) -> dict[str, Any] | None:
    # Projection rows intentionally omit evidence and fact kind. Retraction must
    # use the canonical ledger event so the appended event remains schema-valid.
    for _ledger_file, event in _iter_ledger_events(root):
        if str(event.get("fact_id") or "") == fact_id:
            return event
    return None


def _default_root(root: Path | str | None = None) -> Path:
    if root is not None:
        return Path(root)
    hermes_home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
    return hermes_home / "truth-ledger"


def status_report(root: Path | str) -> dict[str, Any]:
    root = Path(root)
    if not root.exists():
        return {
            "ok": True,
            "action": "status",
            "enabled": False,
            "reason": "truth_ledger_root_missing",
            "root": str(root),
        }

    ledger_files = 0
    ledger_events = 0
    latest_occurred_at = None
    for _ledger_file, event in _iter_ledger_events(root):
        ledger_events += 1
        latest_occurred_at = max(latest_occurred_at or "", str(event.get("occurred_at") or ""))
        if ledger_events == 1:
            ledger_files = len(list((root / "ledger").glob("*.jsonl")))

    pending = len(list((root / "spool" / "pending").glob("*.json")))
    processing = len(list((root / "spool" / "processing").glob("*.json")))
    dead_letter = len(list((root / "spool" / "dead-letter").glob("*.json")))

    active = 0
    current_path = root / "views" / "current.jsonl"
    if current_path.exists():
        for raw_line in current_path.read_text(encoding="utf-8").splitlines():
            if raw_line.strip():
                active += 1

    return {
        "ok": True,
        "action": "status",
        "enabled": True,
        "root": str(root),
        "ledger_files": ledger_files,
        "ledger_events": ledger_events,
        "active_facts": active,
        "pending": pending,
        "processing": processing,
        "dead_letter": dead_letter,
        "latest_occurred_at": latest_occurred_at,
    }


def review_report(root: Path | str, limit: int = 20) -> dict[str, Any]:
    root = Path(root)
    dead_items: list[dict[str, Any]] = []
    dead_dir = root / "spool" / "dead-letter"
    for item in sorted(dead_dir.glob("*.json"), reverse=True)[: max(1, limit)]:
        reason = None
        try:
            payload = json.loads(item.read_text(encoding="utf-8"))
            flow = payload.get("flow") if isinstance(payload.get("flow"), dict) else {}
            reason = (
                payload.get("dead_letter_reason")
                or flow.get("dead_letter_reason")
                or payload.get("reason")
            )
        except Exception:
            reason = "unparseable"
        dead_items.append({"file": item.name, "reason": str(reason or "unknown")})

    review_entries = 0
    review_path = root / "views" / "review.jsonl"
    if review_path.exists():
        review_entries = sum(1 for ln in review_path.read_text(encoding="utf-8").splitlines() if ln.strip())

    return {
        "ok": True,
        "action": "review",
        "root": str(root),
        "review_entries": review_entries,
        "dead_letter_entries": len(dead_items),
        "dead_letter_preview": dead_items,
    }


def rebuild_views(root: Path | str, *, apply: bool = False) -> dict[str, Any]:
    root = Path(root)
    current_path = root / "views" / "current.jsonl"
    before_sha = _sha256_file(current_path)

    payload: dict[str, Any] = {
        "ok": True,
        "action": "rebuild",
        "root": str(root),
        "dry_run": not apply,
        "before_sha256": before_sha,
        "ledger_files": len(list((root / "ledger").glob("*.jsonl"))),
    }
    if not apply:
        return payload

    backups_dir = root / "backups"
    _mkdir_private(backups_dir)
    backup_path: Path | None = None
    if current_path.exists():
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        backup_path = backups_dir / f"current-{ts}.jsonl"
        shutil.copy2(current_path, backup_path)
        _chmod_private_file(backup_path)

    out = _projection_mod().rebuild_current_view(root)
    after_sha = _sha256_file(current_path)
    payload.update(
        {
            "applied": int(out.get("applied", 0)),
            "active": int(out.get("active", 0)),
            "path": str(out.get("path") or current_path),
            "after_sha256": after_sha,
            "backup_path": str(backup_path) if backup_path else "",
        }
    )
    return payload


def retract_fact(root: Path | str, *, fact_id: str, apply: bool = False) -> dict[str, Any]:
    root = Path(root)
    if not _FACT_ID_RE.match(fact_id):
        return {
            "ok": False,
            "action": "retract",
            "reason": "invalid_fact_id",
            "fact_id": fact_id,
            "dry_run": not apply,
        }

    target = _find_fact_event(root, fact_id)
    if target is None:
        return {
            "ok": False,
            "action": "retract",
            "reason": "fact_not_found",
            "fact_id": fact_id,
            "dry_run": not apply,
        }

    payload: dict[str, Any] = {
        "ok": True,
        "action": "retract",
        "fact_id": fact_id,
        "dry_run": not apply,
        "target_key": str(target.get("key") or ""),
        "target_subject": str(target.get("subject") or ""),
    }
    if not apply:
        return payload

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    source_basis = f"manual-retract|{fact_id}|{now}"
    event_id = f"evt_{hashlib.sha256(source_basis.encode('utf-8')).hexdigest()[:24]}"
    raw_target_fact = target.get("fact")
    target_fact: dict[str, Any] = dict(raw_target_fact) if isinstance(raw_target_fact, dict) else {}
    retract_event = {
        "schema_version": 1,
        "event_id": event_id,
        "occurred_at": now,
        "operation": "retract",
        "fact_id": fact_id,
        "supersedes": fact_id,
        "fact": {
            "scope": target_fact.get("scope") or target.get("scope"),
            "kind": target_fact.get("kind"),
            "subject": target_fact.get("subject") or target.get("subject"),
            "key": target_fact.get("key") or target.get("key"),
            "value": None,
        },
        "evidence": {
            "type": "tool_verified",
            "profile": os.environ.get("HERMES_PROFILE", "default"),
            "platform": "operator",
            "session_id": "manual-retraction",
            "turn_id": None,
            "task_id": None,
            "speaker_id": None,
            "conversation_id": None,
            "thread_id": None,
        },
        "extraction": {
            "schema_name": "truth-ledger.fact-candidates.v1",
            "provider": "operator",
            "model": "manual",
            "prompt_version": 1,
        },
    }

    event_key = f"manual_retract|{fact_id}"
    append_out = _ledger_mod().LedgerStore(root).append_event(event=retract_event, event_key=event_key)
    if append_out.get("status") not in {"indexed", "duplicate"}:
        return {
            "ok": False,
            "action": "retract",
            "fact_id": fact_id,
            "reason": str(append_out.get("reason") or append_out.get("status") or "append_failed"),
            "dry_run": False,
        }

    _projection_mod().rebuild_current_view(root)
    payload.update(
        {
            "dry_run": False,
            "appended": append_out.get("status") == "indexed",
            "event_id": append_out.get("event_id"),
            "ledger_file": append_out.get("ledger_file"),
        }
    )
    return payload


def export_snapshot(
    root: Path | str,
    *,
    destination: Path | str | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    root = Path(root)
    if destination is None:
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        destination = root / "exports" / f"truth-ledger-export-{ts}.tar.gz"
    destination = Path(destination)
    if "://" in str(destination):
        return {
            "ok": False,
            "action": "export",
            "reason": "remote_destinations_not_allowed",
            "path": str(destination),
        }

    include_paths: list[Path] = []
    for rel in ("ledger", "views", "state/index.sqlite", "spool/dead-letter"):
        p = root / rel
        if p.exists():
            include_paths.append(p)

    payload: dict[str, Any] = {
        "ok": True,
        "action": "export",
        "dry_run": not apply,
        "path": str(destination),
        "includes": [str(p.relative_to(root)) for p in include_paths],
    }
    if not apply:
        return payload

    _mkdir_private(destination.parent)
    with tarfile.open(destination, "w:gz") as tf:
        for item in include_paths:
            tf.add(item, arcname=str(item.relative_to(root)))
    _chmod_private_file(destination)

    payload.update(
        {
            "dry_run": False,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "size_bytes": destination.stat().st_size,
        }
    )
    return payload


def dispatch_headless(action: str, root: Path | str | None = None, **kwargs) -> dict[str, Any]:
    resolved_root = _default_root(root)
    action_clean = str(action or "").strip().lower()

    if action_clean == "status":
        return status_report(resolved_root)
    if action_clean == "review":
        return review_report(resolved_root, limit=int(kwargs.get("limit", 20) or 20))
    if action_clean == "rebuild":
        return rebuild_views(resolved_root, apply=bool(kwargs.get("apply", False)))
    if action_clean == "retract":
        return retract_fact(
            resolved_root,
            fact_id=str(kwargs.get("fact_id") or ""),
            apply=bool(kwargs.get("apply", False)),
        )
    if action_clean == "export":
        return export_snapshot(
            resolved_root,
            destination=kwargs.get("destination"),
            apply=bool(kwargs.get("apply", False)),
        )

    return {
        "ok": False,
        "action": action_clean,
        "reason": "unknown_action",
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="truth-ledger", add_help=False)
    sub = parser.add_subparsers(dest="action")

    sub.add_parser("status", add_help=False).add_argument("--json", action="store_true")

    review = sub.add_parser("review", add_help=False)
    review.add_argument("--json", action="store_true")
    review.add_argument("--limit", type=int, default=20)

    process = sub.add_parser("process", add_help=False)
    process.add_argument("--json", action="store_true")
    process.add_argument("--limit", type=int, default=1)
    process.add_argument("--apply", action="store_true")

    rebuild = sub.add_parser("rebuild", add_help=False)
    rebuild.add_argument("--json", action="store_true")
    rebuild.add_argument("--apply", action="store_true")

    retract = sub.add_parser("retract", add_help=False)
    retract.add_argument("fact_id")
    retract.add_argument("--json", action="store_true")
    retract.add_argument("--apply", action="store_true")

    export = sub.add_parser("export", add_help=False)
    export.add_argument("--json", action="store_true")
    export.add_argument("--apply", action="store_true")
    export.add_argument("--destination", default="")

    return parser


def _usage_text() -> str:
    return (
        "Usage: /truth-ledger <status|review|process|rebuild|retract|export> [args]\n"
        "Safe defaults: process/rebuild/retract/export run in dry-run mode unless --apply is provided."
    )


def _format_result(result: dict[str, Any]) -> str:
    if not result.get("ok", False):
        return f"truth-ledger {result.get('action', 'unknown')}: ERROR ({result.get('reason', 'failed')})"

    action = result.get("action")
    if action == "status":
        return (
            "truth-ledger status: "
            f"enabled={result.get('enabled')} ledger_events={result.get('ledger_events', 0)} "
            f"active={result.get('active_facts', 0)} pending={result.get('pending', 0)} "
            f"processing={result.get('processing', 0)} dead_letter={result.get('dead_letter', 0)}"
        )
    if action == "review":
        return (
            "truth-ledger review: "
            f"review_entries={result.get('review_entries', 0)} "
            f"dead_letter_entries={result.get('dead_letter_entries', 0)}"
        )
    if action == "process":
        mode = "apply" if not result.get("dry_run") else "dry-run"
        return (
            "truth-ledger process: "
            f"mode={mode} limit={result.get('limit')} claimed={result.get('claimed', 0)} "
            f"appended={result.get('appended', 0)} rejected={result.get('rejected', 0)} "
            f"retried={result.get('retried', 0)} dead_lettered={result.get('dead_lettered', 0)} "
            f"pending_after={result.get('pending_after', result.get('pending_before', 0))}"
        )
    if action == "rebuild":
        mode = "apply" if not result.get("dry_run") else "dry-run"
        return (
            "truth-ledger rebuild: "
            f"mode={mode} before_sha={result.get('before_sha256')} after_sha={result.get('after_sha256')} "
            f"backup={result.get('backup_path', '')}"
        )
    if action == "retract":
        mode = "apply" if not result.get("dry_run") else "dry-run"
        return (
            "truth-ledger retract: "
            f"mode={mode} fact_id={result.get('fact_id')} appended={result.get('appended', False)}"
        )
    if action == "export":
        mode = "apply" if not result.get("dry_run") else "dry-run"
        return f"truth-ledger export: mode={mode} path={result.get('path')}"
    return json.dumps(result, separators=(",", ":"), ensure_ascii=False)


def _render_result(result: dict[str, Any], *, json_output: bool) -> str:
    if json_output:
        return json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _format_result(result)


def handle_truth_ledger_command(
    raw_args: str,
    root: Path | str | None = None,
    runtime_ctx: Any = None,
) -> str | Awaitable[str]:
    tokens = shlex.split(raw_args or "")
    if not tokens or tokens[0] in {"-h", "--help", "help"}:
        return _usage_text()

    parser = _build_parser()
    try:
        args = parser.parse_args(tokens)
    except SystemExit:
        return _usage_text()

    json_output = bool(getattr(args, "json", False))
    if str(args.action or "") == "process":
        process_limit = int(getattr(args, "limit", 1) or 1)
        if process_limit < 1 or process_limit > 3:
            return _render_result(
                {
                    "ok": False,
                    "action": "process",
                    "reason": "invalid_limit",
                    "minimum": 1,
                    "maximum": 3,
                    "dry_run": not bool(getattr(args, "apply", False)),
                },
                json_output=json_output,
            )
        processor = _process_pending
        if processor is None:
            return _render_result(
                {"ok": False, "action": "process", "reason": "processor_unavailable"},
                json_output=json_output,
            )

        async def _run_process() -> str:
            result = await processor(
                root=_default_root(root),
                ctx=runtime_ctx,
                limit=process_limit,
                apply=bool(getattr(args, "apply", False)),
            )
            return _render_result(result, json_output=json_output)

        return _run_process()

    destination = getattr(args, "destination", "") or None
    result = dispatch_headless(
        action=str(args.action or ""),
        root=_default_root(root),
        apply=bool(getattr(args, "apply", False)),
        limit=int(getattr(args, "limit", 20) or 20),
        fact_id=getattr(args, "fact_id", None),
        destination=destination,
    )

    return _render_result(result, json_output=json_output)
