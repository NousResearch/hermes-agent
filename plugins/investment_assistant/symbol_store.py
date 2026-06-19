"""File-backed symbol data store with a SQLite index.

The JSON/markdown/html files under ``symbols/<symbol>/`` are the source of
truth. SQLite is only an index for CRUD, freshness checks, and audits.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Literal

from .storage import new_id, utc_now


DEFAULT_DATA_ROOT = Path("data/investment_assistant")
SAFE_SYMBOL_RE = re.compile(r"^[A-Z0-9]+[.][A-Z0-9._-]+$")


class SymbolDataStore:
    """CRUD interface for long-lived per-symbol investment data."""

    def __init__(self, root: str | Path = DEFAULT_DATA_ROOT, db_path: str | Path | None = None):
        self.root = Path(root)
        self.symbols_root = self.root / "symbols"
        self.experiments_root = self.root / "experiments"
        self.data_runs_root = self.root / "data_runs"
        self.db_path = Path(db_path) if db_path else self.root / "store.sqlite"
        self.root.mkdir(parents=True, exist_ok=True)
        self.symbols_root.mkdir(parents=True, exist_ok=True)
        self.experiments_root.mkdir(parents=True, exist_ok=True)
        self.data_runs_root.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS symbols (
                    symbol TEXT PRIMARY KEY,
                    market TEXT NOT NULL,
                    name TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    manifest_path TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted_at TEXT
                );

                CREATE TABLE IF NOT EXISTS symbol_layers (
                    symbol TEXT NOT NULL,
                    layer TEXT NOT NULL,
                    provider TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    path TEXT NOT NULL DEFAULT '',
                    data_asof TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    run_id TEXT NOT NULL DEFAULT '',
                    checksum TEXT NOT NULL DEFAULT '',
                    warnings_json TEXT NOT NULL DEFAULT '[]',
                    error TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY(symbol, layer),
                    FOREIGN KEY(symbol) REFERENCES symbols(symbol)
                );

                CREATE INDEX IF NOT EXISTS idx_symbol_layers_layer_status
                    ON symbol_layers(layer, status);

                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    root_path TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS experiment_stages (
                    experiment_id TEXT NOT NULL,
                    stage_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    artifact_path TEXT NOT NULL DEFAULT '',
                    trace_path TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(experiment_id, stage_id),
                    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
                );

                CREATE TABLE IF NOT EXISTS data_runs (
                    run_id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    layers_json TEXT NOT NULL DEFAULT '[]',
                    symbols_json TEXT NOT NULL DEFAULT '[]',
                    output_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def create_symbol(
        self,
        symbol: str,
        *,
        market: str | None = None,
        name: str = "",
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        symbol = normalize_symbol(symbol, market or "US")
        now = utc_now()
        manifest = self._read_manifest(symbol) or {
            "artifact_type": "symbol_data_manifest",
            "symbol": symbol,
            "market": symbol.split(".", 1)[0],
            "created_at": now,
            "generated_at": now,
            "updated_at": now,
            "source_status": "missing",
            "layers": {},
            "warnings": [],
            "metadata": {},
            "tags": [],
            "deleted": False,
        }
        manifest["updated_at"] = now
        manifest["market"] = market or manifest.get("market") or symbol.split(".", 1)[0]
        if name:
            manifest["name"] = name
        if metadata:
            existing = manifest.get("metadata") if isinstance(manifest.get("metadata"), dict) else {}
            manifest["metadata"] = {**existing, **metadata}
        if tags:
            manifest["tags"] = _dedupe([*(manifest.get("tags") or []), *tags])
        manifest["deleted"] = False
        manifest.pop("deleted_at", None)
        self._write_manifest(symbol, manifest)
        self._upsert_symbol_index(manifest)
        return manifest

    def get_symbol(self, symbol: str) -> dict[str, Any] | None:
        symbol = normalize_symbol(symbol)
        return self._read_manifest(symbol)

    def list_symbols(
        self,
        *,
        include_deleted: bool = False,
        layer: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if not include_deleted:
            clauses.append("s.deleted_at IS NULL")
        if layer:
            clauses.append("l.layer = ?")
            params.append(layer)
        if status:
            if layer:
                clauses.append("l.status = ?")
            else:
                clauses.append("s.status = ?")
            params.append(status)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        join = "JOIN symbol_layers l ON s.symbol = l.symbol" if layer else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT DISTINCT s.*
                FROM symbols s
                {join}
                {where}
                ORDER BY s.symbol
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def update_symbol(self, symbol: str, patch: dict[str, Any]) -> dict[str, Any]:
        symbol = normalize_symbol(symbol)
        manifest = self._read_manifest(symbol)
        if not manifest:
            raise KeyError(f"Unknown symbol: {symbol}")
        for key in ("name", "source_status", "market"):
            if key in patch:
                manifest[key] = patch[key]
        if isinstance(patch.get("metadata"), dict):
            existing = manifest.get("metadata") if isinstance(manifest.get("metadata"), dict) else {}
            manifest["metadata"] = {**existing, **patch["metadata"]}
        if isinstance(patch.get("tags"), list):
            manifest["tags"] = _dedupe([str(item) for item in patch["tags"]])
        manifest["updated_at"] = utc_now()
        self._write_manifest(symbol, manifest)
        self._upsert_symbol_index(manifest)
        return manifest

    def delete_symbol(self, symbol: str, *, mode: Literal["soft", "hard"] = "soft") -> None:
        symbol = normalize_symbol(symbol)
        symbol_dir = self.symbol_dir(symbol)
        if mode == "hard":
            shutil.rmtree(symbol_dir, ignore_errors=True)
            with self._connect() as conn:
                conn.execute("DELETE FROM symbol_layers WHERE symbol = ?", (symbol,))
                conn.execute("DELETE FROM symbols WHERE symbol = ?", (symbol,))
            return
        manifest = self._read_manifest(symbol)
        if not manifest:
            return
        now = utc_now()
        manifest["deleted"] = True
        manifest["deleted_at"] = now
        manifest["updated_at"] = now
        self._write_manifest(symbol, manifest)
        self._upsert_symbol_index(manifest)

    def put_layer(
        self,
        symbol: str,
        layer: str,
        payload: Any,
        *,
        provider: str = "",
        status: str | None = None,
        data_asof: str = "",
        run_id: str = "",
        warnings: list[str] | None = None,
        error: str = "",
        filename: str | None = None,
    ) -> dict[str, Any]:
        symbol = normalize_symbol(symbol)
        self.create_symbol(symbol)
        filename = filename or f"{layer}.json"
        rel_path = Path("symbols") / symbol / filename
        abs_path = self.root / rel_path
        _atomic_write_json(abs_path, payload)
        checksum = _sha256_file(abs_path)
        layer_entry = {
            "layer": layer,
            "status": status or str(_payload_get(payload, "source_status", "fresh") or "fresh"),
            "provider": provider or str(_payload_get(payload, "provider", "") or _payload_get(payload, "source", "")),
            "path": rel_path.as_posix(),
            "data_asof": data_asof or _data_asof(payload),
            "updated_at": utc_now(),
            "run_id": run_id,
            "checksum": checksum,
            "warnings": warnings if warnings is not None else list(_payload_get(payload, "warnings", []) or []),
            "error": error or str(_payload_get(payload, "error", "") or ""),
        }
        manifest = self._read_manifest(symbol) or self.create_symbol(symbol)
        manifest.setdefault("layers", {})[layer] = layer_entry
        manifest["updated_at"] = layer_entry["updated_at"]
        manifest["source_status"] = _combined_manifest_status(manifest.get("layers", {}))
        manifest["warnings"] = _dedupe(
            [
                str(item)
                for entry in manifest.get("layers", {}).values()
                for item in entry.get("warnings", [])
                if item
            ]
        )
        self._write_manifest(symbol, manifest)
        self._upsert_symbol_index(manifest)
        self._upsert_layer_index(symbol, layer, layer_entry)
        return layer_entry

    def get_layer(self, symbol: str, layer: str) -> Any:
        manifest = self._read_manifest(normalize_symbol(symbol))
        if not manifest:
            return None
        entry = manifest.get("layers", {}).get(layer)
        if not entry or not entry.get("path"):
            return None
        path = self.root / entry["path"]
        if not path.exists():
            return None
        if path.suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        return path.read_text(encoding="utf-8")

    def list_layers(self, symbol: str) -> dict[str, Any]:
        manifest = self._read_manifest(normalize_symbol(symbol)) or {}
        return manifest.get("layers", {})

    def delete_layer(self, symbol: str, layer: str, *, delete_file: bool = False) -> None:
        symbol = normalize_symbol(symbol)
        manifest = self._read_manifest(symbol)
        if not manifest:
            return
        entry = manifest.get("layers", {}).pop(layer, None)
        if delete_file and entry and entry.get("path"):
            try:
                (self.root / entry["path"]).unlink()
            except FileNotFoundError:
                pass
        manifest["updated_at"] = utc_now()
        manifest["source_status"] = _combined_manifest_status(manifest.get("layers", {}))
        self._write_manifest(symbol, manifest)
        with self._connect() as conn:
            conn.execute("DELETE FROM symbol_layers WHERE symbol = ? AND layer = ?", (symbol, layer))
        self._upsert_symbol_index(manifest)

    def mark_layer_stale(self, symbol: str, layer: str, *, reason: str = "") -> dict[str, Any]:
        symbol = normalize_symbol(symbol)
        manifest = self._read_manifest(symbol)
        if not manifest or layer not in manifest.get("layers", {}):
            raise KeyError(f"Unknown layer {layer!r} for {symbol}")
        entry = manifest["layers"][layer]
        entry["status"] = "stale"
        entry["updated_at"] = utc_now()
        if reason:
            entry["warnings"] = _dedupe([*(entry.get("warnings") or []), reason])
        manifest["updated_at"] = entry["updated_at"]
        manifest["source_status"] = _combined_manifest_status(manifest.get("layers", {}))
        self._write_manifest(symbol, manifest)
        self._upsert_symbol_index(manifest)
        self._upsert_layer_index(symbol, layer, entry)
        return entry

    def ingest_symbol_dir(
        self,
        source_symbol_dir: str | Path,
        *,
        run_id: str = "",
        overwrite: bool = True,
    ) -> dict[str, Any]:
        source_symbol_dir = Path(source_symbol_dir)
        symbol = normalize_symbol(source_symbol_dir.name)
        self.create_symbol(symbol)
        dest = self.symbol_dir(symbol)
        dest.mkdir(parents=True, exist_ok=True)

        for child in source_symbol_dir.iterdir():
            if child.name == "manifest.json":
                continue
            target = dest / child.name
            if child.is_dir():
                shutil.copytree(child, target, dirs_exist_ok=True)
            elif overwrite or not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(child, target)

        source_manifest = _read_json(source_symbol_dir / "manifest.json") if (source_symbol_dir / "manifest.json").exists() else {}
        manifest = self._read_manifest(symbol) or self.create_symbol(symbol)
        manifest["symbol"] = symbol
        manifest["market"] = manifest.get("market") or symbol.split(".", 1)[0]
        manifest.setdefault("layers", {})
        manifest.setdefault("created_at", manifest.get("generated_at") or utc_now())
        manifest["updated_at"] = utc_now()
        manifest["metadata"] = {
            **(manifest.get("metadata") if isinstance(manifest.get("metadata"), dict) else {}),
            "last_ingested_from": str(source_symbol_dir),
        }

        for layer, entry in (source_manifest.get("layers", {}) or {}).items():
            merged = dict(entry)
            merged["layer"] = layer
            merged["run_id"] = run_id or merged.get("run_id", "")
            merged["updated_at"] = utc_now()
            merged["path"] = self._canonical_layer_path(symbol, layer, merged.get("path", ""))
            if merged["path"]:
                layer_path = self.root / merged["path"]
                if layer_path.exists():
                    merged["checksum"] = _sha256_file(layer_path)
            manifest["layers"][layer] = merged

        inferred = self._infer_layers_from_files(symbol)
        for layer, entry in inferred.items():
            manifest["layers"].setdefault(layer, entry)

        manifest["source_status"] = _combined_manifest_status(manifest.get("layers", {}))
        manifest["warnings"] = _dedupe(
            [
                str(item)
                for entry in manifest.get("layers", {}).values()
                for item in entry.get("warnings", [])
                if item
            ]
        )
        self._write_manifest(symbol, manifest)
        self._upsert_symbol_index(manifest)
        for layer, entry in manifest.get("layers", {}).items():
            self._upsert_layer_index(symbol, layer, entry)
        return manifest

    def ingest_symbols_root(self, source_root: str | Path, *, run_id: str = "", overwrite: bool = True) -> list[dict[str, Any]]:
        source_root = Path(source_root)
        symbols_dir = source_root / "symbols" if (source_root / "symbols").exists() else source_root
        manifests = []
        for source_symbol_dir in sorted(path for path in symbols_dir.iterdir() if path.is_dir()):
            manifests.append(self.ingest_symbol_dir(source_symbol_dir, run_id=run_id, overwrite=overwrite))
        return manifests

    def rebuild_index(self, symbols: list[str] | None = None) -> None:
        symbol_dirs = [self.symbol_dir(normalize_symbol(symbol)) for symbol in symbols] if symbols else sorted(self.symbols_root.iterdir())
        with self._connect() as conn:
            if symbols:
                for symbol in symbols:
                    norm = normalize_symbol(symbol)
                    conn.execute("DELETE FROM symbol_layers WHERE symbol = ?", (norm,))
                    conn.execute("DELETE FROM symbols WHERE symbol = ?", (norm,))
            else:
                conn.execute("DELETE FROM symbol_layers")
                conn.execute("DELETE FROM symbols")
        for symbol_dir in symbol_dirs:
            manifest = _read_json(symbol_dir / "manifest.json") if (symbol_dir / "manifest.json").exists() else None
            if not manifest:
                continue
            self._upsert_symbol_index(manifest)
            for layer, entry in (manifest.get("layers", {}) or {}).items():
                self._upsert_layer_index(manifest["symbol"], layer, entry)

    def create_experiment(
        self,
        experiment_id: str | None = None,
        *,
        name: str,
        status: str = "active",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        experiment_id = experiment_id or new_id("exp")
        now = utc_now()
        root_path = self.experiments_root / experiment_id
        root_path.mkdir(parents=True, exist_ok=True)
        payload = {
            "artifact_type": "investment_experiment",
            "experiment_id": experiment_id,
            "name": name,
            "status": status,
            "metadata": metadata or {},
            "created_at": now,
            "updated_at": now,
            "stages": {},
        }
        path = root_path / "experiment.json"
        if path.exists():
            existing = _read_json(path)
            payload["created_at"] = existing.get("created_at") or now
            payload["stages"] = existing.get("stages") or {}
        _atomic_write_json(path, payload)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO experiments (
                    experiment_id, name, status, root_path, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(experiment_id) DO UPDATE SET
                    name=excluded.name,
                    status=excluded.status,
                    root_path=excluded.root_path,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    experiment_id,
                    name,
                    status,
                    str(root_path),
                    json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
                    payload["created_at"],
                    now,
                ),
            )
        return payload

    def put_experiment_stage(
        self,
        experiment_id: str,
        stage_id: str,
        artifact: Any,
        *,
        status: str = "completed",
        trace: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        experiment_path = self.experiments_root / experiment_id
        experiment_path.mkdir(parents=True, exist_ok=True)
        stage_path = experiment_path / "stages" / stage_id
        stage_path.mkdir(parents=True, exist_ok=True)
        artifact_path = stage_path / "artifact.json"
        trace_path = stage_path / "trace.json"
        _atomic_write_json(artifact_path, artifact)
        if trace is not None:
            _atomic_write_json(trace_path, trace)
        now = utc_now()
        stage_payload = {
            "stage_id": stage_id,
            "status": status,
            "artifact_path": str(artifact_path),
            "trace_path": str(trace_path) if trace is not None else "",
            "metadata": metadata or {},
            "updated_at": now,
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO experiment_stages (
                    experiment_id, stage_id, status, artifact_path, trace_path,
                    metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(experiment_id, stage_id) DO UPDATE SET
                    status=excluded.status,
                    artifact_path=excluded.artifact_path,
                    trace_path=excluded.trace_path,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    experiment_id,
                    stage_id,
                    status,
                    str(artifact_path),
                    str(trace_path) if trace is not None else "",
                    json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
                    now,
                    now,
                ),
            )
        experiment_json = experiment_path / "experiment.json"
        if experiment_json.exists():
            experiment = _read_json(experiment_json)
            experiment.setdefault("stages", {})[stage_id] = stage_payload
            experiment["updated_at"] = now
            _atomic_write_json(experiment_json, experiment)
        return stage_payload

    def record_data_run(self, run_payload: dict[str, Any], *, job_type: str = "data_miner") -> None:
        run_id = str(run_payload.get("run_id") or new_id("dmr"))
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO data_runs (
                    run_id, job_type, status, layers_json, symbols_json,
                    output_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    job_type=excluded.job_type,
                    status=excluded.status,
                    layers_json=excluded.layers_json,
                    symbols_json=excluded.symbols_json,
                    output_json=excluded.output_json,
                    updated_at=excluded.updated_at
                """,
                (
                    run_id,
                    job_type,
                    str(run_payload.get("status") or "completed"),
                    json.dumps(run_payload.get("layers") or [], ensure_ascii=False, sort_keys=True),
                    json.dumps(run_payload.get("symbols") or [], ensure_ascii=False, sort_keys=True),
                    json.dumps(run_payload, ensure_ascii=False, sort_keys=True),
                    str(run_payload.get("generated_at") or now),
                    now,
                ),
            )

    def symbol_dir(self, symbol: str) -> Path:
        symbol = normalize_symbol(symbol)
        _validate_symbol(symbol)
        return self.symbols_root / symbol

    def _read_manifest(self, symbol: str) -> dict[str, Any] | None:
        path = self.symbol_dir(symbol) / "manifest.json"
        if not path.exists():
            return None
        return _read_json(path)

    def _write_manifest(self, symbol: str, manifest: dict[str, Any]) -> None:
        manifest["symbol"] = normalize_symbol(symbol)
        manifest["updated_at"] = manifest.get("updated_at") or utc_now()
        _atomic_write_json(self.symbol_dir(symbol) / "manifest.json", manifest)

    def _upsert_symbol_index(self, manifest: dict[str, Any]) -> None:
        symbol = normalize_symbol(manifest["symbol"])
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO symbols (
                    symbol, market, name, status, manifest_path, metadata_json,
                    tags_json, created_at, updated_at, deleted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    market=excluded.market,
                    name=excluded.name,
                    status=excluded.status,
                    manifest_path=excluded.manifest_path,
                    metadata_json=excluded.metadata_json,
                    tags_json=excluded.tags_json,
                    updated_at=excluded.updated_at,
                    deleted_at=excluded.deleted_at
                """,
                (
                    symbol,
                    str(manifest.get("market") or symbol.split(".", 1)[0]),
                    str(manifest.get("name") or ""),
                    str(manifest.get("source_status") or "missing"),
                    str(self.symbol_dir(symbol) / "manifest.json"),
                    json.dumps(manifest.get("metadata") or {}, ensure_ascii=False, sort_keys=True),
                    json.dumps(manifest.get("tags") or [], ensure_ascii=False, sort_keys=True),
                    str(manifest.get("created_at") or manifest.get("generated_at") or now),
                    str(manifest.get("updated_at") or now),
                    str(manifest.get("deleted_at") or "") or None,
                ),
            )

    def _upsert_layer_index(self, symbol: str, layer: str, entry: dict[str, Any]) -> None:
        symbol = normalize_symbol(symbol)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO symbol_layers (
                    symbol, layer, provider, status, path, data_asof, updated_at,
                    run_id, checksum, warnings_json, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, layer) DO UPDATE SET
                    provider=excluded.provider,
                    status=excluded.status,
                    path=excluded.path,
                    data_asof=excluded.data_asof,
                    updated_at=excluded.updated_at,
                    run_id=excluded.run_id,
                    checksum=excluded.checksum,
                    warnings_json=excluded.warnings_json,
                    error=excluded.error
                """,
                (
                    symbol,
                    layer,
                    str(entry.get("provider") or entry.get("source") or ""),
                    str(entry.get("status") or "missing"),
                    str(entry.get("path") or ""),
                    str(entry.get("data_asof") or entry.get("asof") or ""),
                    str(entry.get("updated_at") or entry.get("asof") or utc_now()),
                    str(entry.get("run_id") or ""),
                    str(entry.get("checksum") or ""),
                    json.dumps(entry.get("warnings") or [], ensure_ascii=False, sort_keys=True),
                    str(entry.get("error") or ""),
                ),
            )

    def _canonical_layer_path(self, symbol: str, layer: str, raw_path: str) -> str:
        if not raw_path:
            guessed = self.symbol_dir(symbol) / f"{layer}.json"
            if guessed.exists():
                return (Path("symbols") / symbol / guessed.name).as_posix()
            return ""
        name = Path(raw_path).name
        path = self.symbol_dir(symbol) / name
        if path.exists():
            return (Path("symbols") / symbol / name).as_posix()
        return raw_path if raw_path.startswith("symbols/") else ""

    def _infer_layers_from_files(self, symbol: str) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for path in self.symbol_dir(symbol).glob("*.json"):
            if path.name == "manifest.json":
                continue
            try:
                payload = _read_json(path)
            except Exception:
                payload = {}
            layer = str(payload.get("artifact_type") or path.stem)
            rel_path = (Path("symbols") / symbol / path.name).as_posix()
            result[layer] = {
                "layer": layer,
                "status": str(payload.get("source_status") or "fresh"),
                "provider": str(payload.get("provider") or payload.get("source") or ""),
                "path": rel_path,
                "data_asof": _data_asof(payload),
                "updated_at": utc_now(),
                "run_id": "",
                "checksum": _sha256_file(path),
                "warnings": list(payload.get("warnings") or []),
                "error": str(payload.get("error") or ""),
            }
        return result


def normalize_symbol(symbol: str, default_market: str = "US") -> str:
    value = str(symbol or "").strip().upper()
    if "." not in value:
        value = f"{default_market}.{value}"
    _validate_symbol(value)
    return value


def _validate_symbol(symbol: str) -> None:
    if not SAFE_SYMBOL_RE.match(symbol):
        raise ValueError(f"Unsafe or unsupported symbol: {symbol!r}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_asof(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    data_asof = payload.get("data_asof")
    if isinstance(data_asof, str):
        return data_asof
    if isinstance(data_asof, dict):
        values = [str(value) for value in data_asof.values() if value]
        return max(values) if values else ""
    for key in ("retrieved_at", "generated_at", "filing_date", "period_of_report"):
        if payload.get(key):
            return str(payload[key])
    return ""


def _payload_get(payload: Any, key: str, default: Any) -> Any:
    return payload.get(key, default) if isinstance(payload, dict) else default


def _combined_manifest_status(layers: dict[str, Any]) -> str:
    statuses = [str(entry.get("status") or "missing") for entry in layers.values()]
    if not statuses:
        return "missing"
    if all(status == "fresh" for status in statuses):
        return "fresh"
    if any(status in {"fresh", "partial", "stale", "not_implemented"} for status in statuses):
        return "partial"
    if all(status == "skipped" for status in statuses):
        return "skipped"
    if any(status == "unavailable" for status in statuses):
        return "unavailable"
    return "partial"


def _dedupe(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result
