"""Authenticated, hash-bound report catalog for the Hermes dashboard."""
from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import stat
from pathlib import Path
from typing import Any

from hermes_cli.config import get_hermes_home

CATALOG_ENV = "HERMES_REPORT_CATALOG"
SEO_CATALOG_ENV = "HERMES_SEO_REPORT_CATALOG"
APPROVED_ROOTS_ENV = "HERMES_REPORT_APPROVED_ROOTS"
SCHEMA_VERSION = 1
_PRIMARY_ARTIFACTS = frozenset({"seo-report.html", "seo-report-package.zip", "export-manifest.json", "final-receipt.json", "SHA256SUMS"})

class ReportCatalogError(ValueError):
    pass


def _component(value: str, label: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
        raise ReportCatalogError(f"invalid {label}")
    if label in {"artifact filename", "manifest"} and not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise ReportCatalogError(f"invalid {label}")
    if any(ord(char) < 32 or char in {'"', "'"} for char in value):
        raise ReportCatalogError(f"invalid {label}")
    return value


def catalog_path() -> Path:
    return Path(os.environ.get(SEO_CATALOG_ENV) or os.environ.get(CATALOG_ENV) or (get_hermes_home() / "report-catalog.json")).expanduser().resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ReportCatalogError(f"invalid {label}")
    return value.lower()


def _trusted_roots() -> list[Path]:
    raw = os.environ.get(APPROVED_ROOTS_ENV, "").strip()
    values = [item for item in raw.split(os.pathsep) if item] if raw else [str(Path.home() / "seo_spider_mcp_server")]
    roots = [Path(item).expanduser().resolve() for item in values]
    if not all(root.is_dir() for root in roots):
        raise ReportCatalogError("trusted report root is unavailable")
    return roots


def _read_hashed(root: Path, path: Path) -> tuple[bytes, str]:
    try:
        relative = path.resolve().relative_to(root.resolve())
        if len(relative.parts) != 1:
            raise ValueError
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        expected_stat = os.stat(root, follow_symlinks=False)
        root_fd = os.open(root, directory_flags)
        try:
            actual_stat = os.fstat(root_fd)
            if (actual_stat.st_dev, actual_stat.st_ino) != (expected_stat.st_dev, expected_stat.st_ino):
                raise OSError("report root changed during validation")
            leaf_fd = os.open(relative.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=root_fd)
            if not stat.S_ISREG(os.fstat(leaf_fd).st_mode):
                os.close(leaf_fd)
                raise OSError("artifact is not a regular file")
        finally:
            os.close(root_fd)
        with os.fdopen(leaf_fd, "rb") as stream:
            content = stream.read()
    except (OSError, ValueError) as exc:
        raise ReportCatalogError("artifact is unavailable") from exc
    return content, hashlib.sha256(content).hexdigest()


def _root(entry: dict[str, Any], allowed_roots: list[Path], trusted_roots: list[Path]) -> Path:
    raw = entry.get("root")
    if not isinstance(raw, str) or not raw:
        raise ReportCatalogError("report root is missing")
    root = Path(raw).expanduser().resolve()
    if not root.is_dir() or not any(_under(parent, root) for parent in allowed_roots) or not any(_under(parent, root) for parent in trusted_roots):
        raise ReportCatalogError("report root is outside the approved report roots")
    return root


def _under(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def load_catalog(path: Path | None = None, trusted_roots: list[Path] | None = None) -> dict[str, Any]:
    source = (path or catalog_path()).resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportCatalogError("report catalog is unavailable") from exc
    if (not isinstance(payload, dict) or payload.get("schema_version") not in (SCHEMA_VERSION, 2)
            or not isinstance(payload.get("reports"), list)):
        raise ReportCatalogError("unsupported report catalog schema")
    bound_run_id: str | None = None
    bound_prompt_version: str | None = None
    if payload["schema_version"] == SCHEMA_VERSION:
        bound_run_id = payload.get("bound_run_id")
        bound_prompt_version = payload.get("bound_prompt_version")
        if not isinstance(bound_run_id, str) or not bound_run_id or not isinstance(bound_prompt_version, str) or not bound_prompt_version:
            raise ReportCatalogError("catalog binding is missing")
    else:
        if "bound_run_id" in payload or "bound_prompt_version" in payload:
            raise ReportCatalogError("version two requires per-report binding")
    trusted = trusted_roots or _trusted_roots()
    raw_roots = payload.get("allowed_roots")
    if not isinstance(raw_roots, list) or not raw_roots:
        raise ReportCatalogError("approved report roots are missing")
    allowed_roots = [Path(str(item)).expanduser().resolve() for item in raw_roots]
    for parent in allowed_roots:
        if not parent.is_dir():
            raise ReportCatalogError("approved report root is unavailable")
    seen_ids: set[str] = set()
    for entry in payload["reports"]:
        if not isinstance(entry, dict):
            raise ReportCatalogError("invalid report catalog entry")
        report_id = _component(str(entry.get("report_id", "")), "report id")
        if report_id in seen_ids:
            raise ReportCatalogError("duplicate report id")
        seen_ids.add(report_id)
        if payload["schema_version"] == SCHEMA_VERSION:
            if entry.get("run_id") != bound_run_id or entry.get("prompt_version") != bound_prompt_version:
                raise ReportCatalogError("report catalog binding mismatch")
        elif (not isinstance(entry.get("run_id"), str) or not entry["run_id"]
              or not isinstance(entry.get("prompt_version"), str) or not entry["prompt_version"]):
            raise ReportCatalogError("per-report catalog binding missing")
        _root(entry, allowed_roots, trusted)
        manifest_name = _component(str(entry.get("manifest", "export-manifest.json")), "manifest")
        manifest_path = Path(str(entry["root"])).expanduser().resolve() / manifest_name
        _manifest_bytes, manifest_digest = _read_hashed(Path(str(entry["root"])).expanduser().resolve(), manifest_path)
        if manifest_digest != _digest(entry.get("manifest_sha256"), "manifest hash"):
            raise ReportCatalogError("report manifest hash drift detected")
        entry["_allowed_roots"] = allowed_roots
        entry["_trusted_roots"] = trusted
        entry["_manifest_bytes"] = _manifest_bytes
    return payload


def get_report(report_id: str, path: Path | None = None) -> dict[str, Any]:
    wanted = _component(report_id, "report id")
    catalog = load_catalog(path)
    for entry in catalog["reports"]:
        if entry.get("report_id") == wanted:
            return entry
    raise ReportCatalogError("report not found")


def _expected_artifacts(entry: dict[str, Any]) -> dict[str, str]:
    root = _root(entry, entry.get("_allowed_roots", []), entry.get("_trusted_roots", []))
    manifest_name = _component(str(entry.get("manifest", "export-manifest.json")), "manifest")
    manifest_path = root / manifest_name
    if not _under(root, manifest_path) or not manifest_path.is_file():
        raise ReportCatalogError("report manifest is unavailable")
    try:
        manifest = json.loads(entry["_manifest_bytes"].decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportCatalogError("report manifest is invalid") from exc
    if manifest.get("run_id") != entry.get("run_id") or manifest.get("prompt_version") != entry.get("prompt_version"):
        raise ReportCatalogError("report lineage does not match catalog")
    expected: dict[str, str] = {}
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ReportCatalogError("manifest artifacts are invalid")
    for item in artifacts:
        if not isinstance(item, dict):
            raise ReportCatalogError("manifest artifact is invalid")
        name, digest = item.get("filename"), item.get("sha256")
        if not isinstance(name, str) or not isinstance(digest, str):
            raise ReportCatalogError("manifest artifact is invalid")
        _component(name, "artifact filename")
        expected[name] = _digest(digest, "artifact hash")
    primary_artifacts = entry.get("primary_artifacts", [])
    if not isinstance(primary_artifacts, list):
        raise ReportCatalogError("primary artifacts are invalid")
    for item in primary_artifacts:
        if not isinstance(item, dict):
            raise ReportCatalogError("primary artifact is invalid")
        name, digest = item.get("filename"), item.get("sha256")
        if not isinstance(name, str) or not isinstance(digest, str):
            raise ReportCatalogError("primary artifact is invalid")
        _component(name, "artifact filename")
        _digest(digest, "artifact hash")
        if name not in _PRIMARY_ARTIFACTS:
            raise ReportCatalogError("unapproved primary artifact")
        primary_digest = digest.lower()
        if name in expected and expected[name] != primary_digest:
            raise ReportCatalogError("primary artifact hash conflicts with manifest")
        expected[name] = primary_digest
    if not expected:
        raise ReportCatalogError("report has no artifacts")
    return expected


def resolve_artifact(report_id: str, filename: str, path: Path | None = None) -> tuple[str, bytes, str, str]:
    entry = get_report(report_id, path)
    name = _component(filename, "artifact filename")
    root = _root(entry, entry.get("_allowed_roots", []), entry.get("_trusted_roots", []))
    expected = _expected_artifacts(entry)
    if name not in expected:
        raise ReportCatalogError("artifact is not registered")
    target = (root / name).resolve()
    if not _under(root, target) or not target.is_file():
        raise ReportCatalogError("artifact is unavailable")
    content, actual = _read_hashed(root, target)
    if actual != expected[name]:
        raise ReportCatalogError("artifact hash drift detected")
    mime = {
        ".html": "text/html; charset=utf-8",
        ".csv": "text/csv; charset=utf-8",
        ".json": "application/json",
        ".zip": "application/zip",
        "": "text/plain; charset=utf-8",
    }.get(target.suffix.lower(), mimetypes.guess_type(target.name)[0] or "application/octet-stream")
    return target.name, content, mime, actual


def public_entry(entry: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    expected = _expected_artifacts(entry)
    return {
        "report_id": entry["report_id"],
        "run_id": entry["run_id"],
        "prompt_version": entry["prompt_version"],
        "title": entry.get("title", entry["report_id"]),
        "report_url": f"/reports/{entry['report_id']}",
        "artifacts": [{"filename": name, "sha256": digest} for name, digest in sorted(expected.items())],
    }
