#!/usr/bin/env python3
"""Audit and export the installed JiuwenMemory memory_server.py PoC patch.

The Hermes JiuwenMemory plugin talks to the memory-server HTTP API. The
server-side Graph/Turbo/Swarm/Dreaming/session_id PoC currently lives in the
installed jiuwen_memory package under site-packages, so this script avoids
importing the server module and inspects the file on disk instead.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import shutil
import site
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

SERVER_RELATIVE_PATH = Path("jiuwen_memory/server/memory_server.py")
POC_NOTE = (
    "JiuwenMemory server-side PoC persistence currently lives in the installed "
    "site-packages file jiuwen_memory/server/memory_server.py, not in the "
    "Hermes plugin package."
)

REQUIRED_ROUTES: tuple[tuple[str, str], ...] = (
    ("POST", "/turbo/add_messages_async"),
    ("GET", "/turbo/status"),
    ("POST", "/turbo/process_once"),
    ("POST", "/swarm/promote"),
    ("GET", "/swarm/status"),
    ("POST", "/swarm/search"),
    ("GET", "/graph/status"),
    ("POST", "/graph/extract"),
    ("POST", "/graph/query"),
    ("GET", "/dreaming/status"),
    ("POST", "/dreaming/start"),
    ("POST", "/dreaming/stop"),
)

REQUIRED_MODELS: tuple[str, ...] = (
    "DreamingStartRequest",
    "DreamingStopRequest",
    "GraphEntityRequest",
    "GraphRelationRequest",
    "GraphExtractRequest",
    "GraphQueryRequest",
    "SwarmPromoteRequest",
    "SwarmSearchRequest",
)

REQUIRED_CLASS_FIELDS: tuple[tuple[str, str], ...] = (
    ("AddMessagesRequest", "session_id"),
    ("GraphExtractRequest", "session_id"),
)

REQUIRED_HELPERS: tuple[str, ...] = (
    "_dreaming_config_from_env",
    "_dreaming_status_payload",
    "_turbo_enqueue",
    "_turbo_process_next_pending",
    "_turbo_status_payload",
    "_graph_connect",
    "_graph_extract_heuristic",
    "_swarm_insert_record",
    "_swarm_status_payload",
)

REQUIRED_ASSIGNED_NAMES: tuple[str, ...] = (
    "_GRAPH_DB_PATH",
    "_TURBO_DB_PATH",
    "_SWARM_DB_PATH",
)


@dataclass(frozen=True)
class FeatureResult:
    """One required PoC marker and whether it was found."""

    name: str
    present: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "present": self.present,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class AuditReport:
    """Structured audit result for memory_server.py."""

    server_file: str
    sha256: str
    ok: bool
    features: list[FeatureResult]

    def missing(self) -> list[FeatureResult]:
        return [feature for feature in self.features if not feature.present]

    def to_dict(self) -> dict[str, object]:
        return {
            "server_file": self.server_file,
            "sha256": self.sha256,
            "ok": self.ok,
            "missing": [feature.to_dict() for feature in self.missing()],
            "features": [feature.to_dict() for feature in self.features],
        }


@dataclass(frozen=True)
class ExportResult:
    """Paths written by an export run."""

    backup_file: str
    patch_file: str
    manifest_file: str
    audit: AuditReport

    def to_dict(self) -> dict[str, object]:
        return {
            "backup_file": self.backup_file,
            "patch_file": self.patch_file,
            "manifest_file": self.manifest_file,
            "audit": self.audit.to_dict(),
        }


def _candidate_site_paths(repo_root: Path) -> Iterable[Path]:
    seen: set[Path] = set()

    def emit(path: Path) -> Iterable[Path]:
        resolved = path.expanduser()
        if resolved not in seen:
            seen.add(resolved)
            yield resolved

    for entry in sys.path:
        if entry:
            yield from emit(Path(entry))

    try:
        for entry in site.getsitepackages():
            yield from emit(Path(entry))
    except Exception:
        pass

    try:
        yield from emit(Path(site.getusersitepackages()))
    except Exception:
        pass

    for venv_name in (".venv", "venv"):
        for candidate in (repo_root / venv_name / "lib").glob("python*/site-packages"):
            yield from emit(candidate)


def find_memory_server_path(explicit_path: str | None = None) -> Path:
    """Find the installed JiuwenMemory server file without importing it."""

    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"memory_server.py not found: {path}")
        return path

    repo_root = Path(__file__).resolve().parents[1]
    for base in _candidate_site_paths(repo_root):
        candidate = base / SERVER_RELATIVE_PATH
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find installed jiuwen_memory/server/memory_server.py. "
        "Pass --server-file to audit a specific file."
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _literal_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _route_set(tree: ast.AST) -> set[tuple[str, str]]:
    routes: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            func = decorator.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr in {"get", "post", "put", "delete", "patch"}
                and isinstance(func.value, ast.Name)
                and func.value.id == "app"
            ):
                continue
            route_path = _literal_str(decorator.args[0]) if decorator.args else None
            if route_path is None:
                for keyword in decorator.keywords:
                    if keyword.arg in {"path", "path_format"}:
                        route_path = _literal_str(keyword.value)
                        break
            if route_path:
                routes.add((func.attr.upper(), route_path))
    return routes


def _class_names(tree: ast.AST) -> set[str]:
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}


def _function_names(tree: ast.AST) -> set[str]:
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _assigned_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _class_field_names(tree: ast.AST, class_name: str) -> set[str]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        fields: set[str] = set()
        for child in node.body:
            if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                fields.add(child.target.id)
            elif isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        fields.add(target.id)
        return fields
    return set()


def _audit_tree(tree: ast.AST) -> list[FeatureResult]:
    routes = _route_set(tree)
    classes = _class_names(tree)
    functions = _function_names(tree)
    assigned = _assigned_names(tree)
    features: list[FeatureResult] = []

    for method, path in REQUIRED_ROUTES:
        present = (method, path) in routes
        features.append(
            FeatureResult(
                name=f"route:{method} {path}",
                present=present,
                detail="FastAPI route decorator",
            )
        )

    for model in REQUIRED_MODELS:
        features.append(
            FeatureResult(
                name=f"model:{model}",
                present=model in classes,
                detail="Pydantic request model",
            )
        )

    for class_name, field in REQUIRED_CLASS_FIELDS:
        present = field in _class_field_names(tree, class_name)
        features.append(
            FeatureResult(
                name=f"field:{class_name}.{field}",
                present=present,
                detail="session_id propagation field",
            )
        )

    for helper in REQUIRED_HELPERS:
        features.append(
            FeatureResult(
                name=f"helper:{helper}",
                present=helper in functions,
                detail="PoC helper function",
            )
        )

    for name in REQUIRED_ASSIGNED_NAMES:
        features.append(
            FeatureResult(
                name=f"storage:{name}",
                present=name in assigned,
                detail="PoC persistence path",
            )
        )

    return features


def audit_memory_server(server_file: str | Path | None = None) -> AuditReport:
    """Verify that memory_server.py still contains the expected PoC surface."""

    path = find_memory_server_path(str(server_file)) if server_file else find_memory_server_path()
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        feature = FeatureResult(
            name="syntax:python",
            present=False,
            detail=f"{exc.msg} at line {exc.lineno}",
        )
        return AuditReport(str(path), _sha256(path), False, [feature])

    features = _audit_tree(tree)
    return AuditReport(
        server_file=str(path),
        sha256=_sha256(path),
        ok=all(feature.present for feature in features),
        features=features,
    )


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _full_file_patch(source: str, server_file: Path) -> str:
    lines = source.splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            [],
            lines,
            fromfile="/dev/null",
            tofile=f"b/{server_file.name}",
        )
    )


def export_server_artifacts(
    server_file: str | Path | None = None,
    output_dir: str | Path = "jiuwenmemory-server-poc-backups",
    *,
    timestamp: str | None = None,
) -> ExportResult:
    """Export a timestamped backup, full-file patch, and manifest."""

    path = find_memory_server_path(str(server_file)) if server_file else find_memory_server_path()
    stamp = timestamp or _timestamp()
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)

    backup_file = output / f"memory_server.{stamp}.py"
    patch_file = output / f"memory_server.{stamp}.full-file.patch"
    manifest_file = output / f"memory_server.{stamp}.manifest.json"

    audit = audit_memory_server(path)
    shutil.copy2(path, backup_file)

    source = path.read_text(encoding="utf-8")
    patch_file.write_text(_full_file_patch(source, path), encoding="utf-8")

    manifest = {
        "note": POC_NOTE,
        "exported_at": stamp,
        "server_file": str(path),
        "backup_file": str(backup_file),
        "patch_file": str(patch_file),
        "source_sha256": audit.sha256,
        "audit": audit.to_dict(),
    }
    manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return ExportResult(
        backup_file=str(backup_file),
        patch_file=str(patch_file),
        manifest_file=str(manifest_file),
        audit=audit,
    )


def _print_verify(report: AuditReport, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return

    print(POC_NOTE)
    print(f"server_file: {report.server_file}")
    print(f"sha256: {report.sha256}")
    print(f"status: {'ok' if report.ok else 'missing required features'}")
    if not report.ok:
        print("missing:")
        for feature in report.missing():
            print(f"  - {feature.name} ({feature.detail})")


def _print_export(result: ExportResult, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return

    print(POC_NOTE)
    print(f"backup_file: {result.backup_file}")
    print(f"patch_file: {result.patch_file}")
    print(f"manifest_file: {result.manifest_file}")
    print(f"audit_status: {'ok' if result.audit.ok else 'missing required features'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify", help="verify required PoC routes/features")
    verify.add_argument("--server-file", help="path to memory_server.py; defaults to installed package")
    verify.add_argument("--json", action="store_true", help="emit JSON")

    export = subparsers.add_parser("export", help="export a timestamped backup and full-file patch")
    export.add_argument("--server-file", help="path to memory_server.py; defaults to installed package")
    export.add_argument(
        "--output-dir",
        default="jiuwenmemory-server-poc-backups",
        help="directory for backup, patch, and manifest",
    )
    export.add_argument("--timestamp", help="override timestamp for reproducible tests")
    export.add_argument("--json", action="store_true", help="emit JSON")

    locate = subparsers.add_parser("locate", help="print the installed memory_server.py path")
    locate.add_argument("--server-file", help="path to memory_server.py; validates and echoes it")
    locate.add_argument("--json", action="store_true", help="emit JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "verify":
            report = audit_memory_server(args.server_file)
            _print_verify(report, as_json=args.json)
            return 0 if report.ok else 1
        if args.command == "export":
            result = export_server_artifacts(
                args.server_file,
                args.output_dir,
                timestamp=args.timestamp,
            )
            _print_export(result, as_json=args.json)
            return 0
        if args.command == "locate":
            path = find_memory_server_path(args.server_file)
            if args.json:
                print(json.dumps({"server_file": str(path), "note": POC_NOTE}, indent=2, sort_keys=True))
            else:
                print(path)
            return 0
    except Exception as exc:
        if getattr(args, "json", False):
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True), file=sys.stderr)
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
