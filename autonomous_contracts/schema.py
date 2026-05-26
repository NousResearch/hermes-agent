"""JSON Schema export helpers for autonomous contract artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Type

from pydantic import BaseModel

from .models import CleanupRecord, Contract, ContractLock, LedgerSeed, WorkerPacket

_SCHEMA_MODELS: dict[str, Type[BaseModel]] = {
    "contract.schema.json": Contract,
    "contract-lock.schema.json": ContractLock,
    "ledger-seed.schema.json": LedgerSeed,
    "worker-packet.schema.json": WorkerPacket,
    "cleanup-record.schema.json": CleanupRecord,
}


def schema_map() -> dict[str, dict]:
    """Return JSON-serializable schemas keyed by canonical filename."""

    return {name: model.model_json_schema() for name, model in _SCHEMA_MODELS.items()}


def write_schema_files(output_dir: str | Path) -> list[Path]:
    """Write canonical JSON schema files into ``output_dir``.

    The function overwrites only the schema filenames it owns and returns the
    written paths in deterministic order.
    """

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, schema in sorted(schema_map().items()):
        path = destination / name
        path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(path)
    return written
