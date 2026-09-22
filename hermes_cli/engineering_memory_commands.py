"""Handlers for the local engineering-memory CLI contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.engineering_memory_curator import EngineeringMemoryCurator
from agent.engineering_memory_index import EngineeringMemoryIndex
from agent.engineering_memory_ledger import EngineeringMemoryLedger, EngineeringMemoryLedgerError
from agent.engineering_memory_schema import EngineeringMemorySchemaError, parse_markdown_record

CONTRACT = "hermes.engineering_memory.v1"


def _envelope(operation: str, *, ok: bool, **payload: Any) -> dict[str, Any]:
    return {"contract": CONTRACT, "operation": operation, "ok": ok, **payload}


def _paths(args) -> tuple[Path, Path]:
    vault = getattr(args, "vault", None)
    index = getattr(args, "index_path", None)
    if not vault or not index:
        from hermes_cli.config import load_config
        settings = load_config().get("engineering_memory")
        if isinstance(settings, dict) and settings.get("enabled"):
            vault = vault or settings.get("vault_path")
            index = index or settings.get("index_path")
    if not vault or not index:
        raise ValueError("explicit --vault and --index are required; shared engineering memory is disabled until configured")
    return Path(vault).expanduser(), Path(index).expanduser()


def _emit(args, result: dict[str, Any]) -> None:
    if getattr(args, "json", False):
        print(json.dumps(result, sort_keys=True, ensure_ascii=False))
        return
    if result.get("ok"):
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"engineering-memory: {result.get('error', 'operation failed')}")


def cmd_engineering_memory(args) -> None:
    command = getattr(args, "engineering_memory_command", None)
    operation = command or "help"
    try:
        if command == "laya":
            from agent.engineering_memory_laya import LayaRuntimeConfig, SubprocessLayaRuntime, calibrate_laya, download_laya

            laya_command = getattr(args, "laya_command", None)
            if laya_command == "doctor":
                revision = args.revision.strip()
                result = _envelope(
                    "laya.doctor",
                    ok=bool(args.repo.strip() and revision and args.local_path),
                    configured={"repo": args.repo, "revision": revision, "local_path": args.local_path},
                    diagnostics=[] if revision and args.local_path else ["laya_unavailable"],
                )
                _emit(args, result)
                return
            if laya_command == "download":
                path = download_laya(args.repo, args.revision, Path(args.destination).expanduser())
                _emit(args, _envelope("laya.download", ok=True, path=str(path), diagnostics=[]))
                return
            if laya_command == "calibrate":
                runtime = SubprocessLayaRuntime(args.runtime_command, model_revision=args.revision)
                artifact = calibrate_laya(runtime, Path(args.fixtures).expanduser(), Path(args.output).expanduser(), model_revision=args.revision)
                _emit(args, _envelope("laya.calibrate", ok=True, artifact=artifact, diagnostics=[]))
                return
            raise ValueError("choose laya doctor, laya download, or laya calibrate")
        vault, index_path = _paths(args)
        ledger = EngineeringMemoryLedger(vault)
        curator = EngineeringMemoryCurator(ledger)
        index = EngineeringMemoryIndex(index_path)
        if command == "propose":
            record_path = Path(args.record).expanduser().resolve()
            if not record_path.is_file():
                raise ValueError("record path is not a regular file")
            if not record_path.is_relative_to(vault.resolve()):
                raise ValueError("record path is outside configured vault")
            record = parse_markdown_record(record_path.read_text(encoding="utf-8"), source_path=record_path)
            decision = curator.stage(record)
            _emit(args, _envelope(operation, ok=True, record_id=decision.record.record_id, status=decision.status, diagnostics=list(decision.reason_codes)))
        elif command == "review":
            decision = curator.review(args.record_id, args.action, reviewer=args.reviewer, reason=args.reason, evidence_refs=tuple(args.evidence_ref))
            _emit(args, _envelope(operation, ok=True, record_id=decision.record.record_id, status=decision.status, diagnostics=[]))
        elif command == "rebuild":
            index.rebuild(ledger.iter_records(statuses={"approved"}))
            _emit(args, _envelope(operation, ok=True, health=index.verify().__dict__, diagnostics=list(ledger.diagnostics)))
        elif command == "search":
            results = index.search(
                args.query,
                repository=args.repository,
                workspace=args.workspace,
                component=args.component,
                task_type=args.task_type,
                tags=tuple(args.tag),
                verified_only=not args.include_unverified,
                as_of_head=args.as_of_head,
                limit=args.max_results or 8,
                char_budget=args.max_chars or 6_000,
            )
            _emit(args, _envelope(operation, ok=True, records=[item.to_mapping() for item in results], diagnostics=[]))
        elif command == "verify":
            health = index.verify()
            _emit(args, _envelope(operation, ok=health.ok, health=health.__dict__, diagnostics=list(ledger.diagnostics)))
        else:
            _emit(args, _envelope(operation, ok=False, error="choose propose, review, search, rebuild, or verify", diagnostics=[]))
    except (OSError, ValueError, EngineeringMemorySchemaError, EngineeringMemoryLedgerError) as exc:
        result = _envelope(operation, ok=False, error=str(exc), diagnostics=[])
        _emit(args, result)
