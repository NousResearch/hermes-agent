"""Shared fixtures and helpers for Task 29 advisory inspection traceability tests."""

from __future__ import annotations

import ast
import json
import os
import stat
import time
from pathlib import Path
from typing import Any, Callable

import pytest

from htr import contracts, events, io, paths
from htr.advisory_inspection_constants import (
    MAX_ARTIFACT_REFERENCES_PER_MANIFEST,
    MAX_CONTROL_JSON_BYTES,
    MAX_CONTROL_RECORD_FILE_BYTES,
    MAX_RAW_READ_BYTES,
    SUPPLEMENTAL_FINDING_TOKENS,
)
from htr.advisory_inspection_models import ArtifactReferenceSelector, LinkReferenceSelector
from htr.advisory_inspection_secure import raw_sha256_digest
from htr.ids import new_attempt_id, new_run_id, new_task_id

_FORBIDDEN_NAMES = frozenset(
    {
        "read_artifact_manifest",
        "read_json",
        "evaluate_run_seal",
        "parse_strict_json_bytes",
    }
)

_ARTIFACT_AXIS_SCALARS = frozenset(
    {
        "reference_selected",
        "reference_absent_from_manifest",
        "manifest_bound",
        "path_valid_attempt_relative",
        "filesystem_observed",
        "advisory_only",
        "budget_within_limits",
    }
)


@pytest.fixture
def advisory_runs_root(tmp_path, monkeypatch):
    """Hermetic HERMES_HOME with an empty runs/ directory."""
    hermes = tmp_path / ".hermes"
    runs = hermes / "runs"
    runs.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes))
    return runs


def bootstrap_attempt(*, base_dir: Path | None = None) -> tuple[str, str, str]:
    run_id = new_run_id()
    task_id = new_task_id()
    attempt_id = new_attempt_id()
    io.create_run_workspace(run_id, base_dir=base_dir)
    io.create_task_workspace(run_id, task_id, base_dir=base_dir)
    events.register_attempt(run_id, task_id, attempt_id, actor="test", base_dir=base_dir)
    return run_id, task_id, attempt_id


def bootstrap_run(*, base_dir: Path | None = None) -> str:
    run_id = new_run_id()
    io.create_run_workspace(run_id, base_dir=base_dir)
    return run_id


def write_manifest(
    run_id: str,
    task_id: str,
    attempt_id: str,
    payload: dict[str, Any],
    *,
    trailing_lf: bool = True,
    base_dir: Path | None = None,
) -> str:
    target = paths.artifact_manifest_path(run_id, task_id, attempt_id, base_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if trailing_lf:
        raw += b"\n"
    target.write_bytes(raw)
    return raw_sha256_digest(raw)


def write_manifest_bytes(
    run_id: str,
    task_id: str,
    attempt_id: str,
    raw: bytes,
    *,
    base_dir: Path | None = None,
) -> str:
    target = paths.artifact_manifest_path(run_id, task_id, attempt_id, base_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    return raw_sha256_digest(raw)


def manifest_payload(
    run_id: str,
    task_id: str,
    attempt_id: str,
    artifacts: list[dict[str, Any]] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "1",
        "run_id": run_id,
        "task_id": task_id,
        "attempt_id": attempt_id,
        "artifacts": artifacts if artifacts is not None else [],
    }
    payload.update(extra)
    return payload


def artifact_entry(path: str, **overrides: Any) -> dict[str, Any]:
    entry = {
        "path": path,
        "kind": "file",
        "created_at": "2026-01-01T00:00:00+00:00",
        "metadata": {},
    }
    entry.update(overrides)
    return entry


def artifact_selector(
    run_id: str,
    task_id: str,
    attempt_id: str,
    digest: str,
    entry_index: int = 0,
) -> ArtifactReferenceSelector:
    return ArtifactReferenceSelector(
        run_id=run_id,
        task_id=task_id,
        attempt_id=attempt_id,
        manifest_raw_digest=digest,
        entry_index=entry_index,
    )


def write_link_record(
    run_id: str,
    filename: str,
    record: dict[str, Any],
    *,
    base_dir: Path | None = None,
) -> str:
    target = paths.run_root(run_id, base_dir) / filename
    raw = json.dumps(record, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n"
    target.write_bytes(raw)
    return raw_sha256_digest(raw)


def execution_item(**overrides: Any) -> dict[str, Any]:
    item = {
        "item_id": "exec-1",
        "source_followup_item_id": "followup-1",
        "title": "Open dashboard",
        "execution_kind": "manual_open_link",
        "command": {"url": "https://example.com"},
        "approval_reason": None,
        "metadata": {},
    }
    item.update(overrides)
    return item


def request_record(run_id: str, **kwargs: Any) -> dict[str, Any]:
    items = kwargs.pop("execution_items", [execution_item()])
    return contracts.make_run_execution_request_record(
        run_id=run_id,
        source_followup_plan_fingerprint="fp-test",
        execution_items=items,
        **kwargs,
    )


def link_selector(
    run_id: str,
    digest: str,
    *,
    item_index: int = 0,
    record_kind: str = "run_execution_request_record",
) -> LinkReferenceSelector:
    return LinkReferenceSelector(
        run_id=run_id,
        record_kind=record_kind,  # type: ignore[arg-type]
        record_raw_digest=digest,
        item_index=item_index,
    )


def collect_source_names(module_path: Path) -> set[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[-1])
    return names


def forbidden_api_names(repo_root: Path, rel_paths: tuple[str, ...]) -> set[str]:
    found: set[str] = set()
    for rel in rel_paths:
        found |= collect_source_names(repo_root / rel)
    return found


def snapshot_mtimes(paths_to_watch: list[Path]) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for p in paths_to_watch:
        if p.exists():
            st = p.stat()
            out[str(p)] = (st.st_mtime, st.st_ctime)
    return out


def assert_mtimes_unchanged(before: dict[str, tuple[float, float]]) -> None:
    for path_str, (mtime, ctime) in before.items():
        p = Path(path_str)
        assert p.exists(), f"missing watched path {path_str}"
        st = p.stat()
        assert st.st_mtime == mtime, f"mtime changed for {path_str}"
        assert st.st_ctime == ctime, f"ctime changed for {path_str}"


def collect_run_evidence_paths(run_id: str, base_dir: Path | None = None) -> list[Path]:
    root = paths.run_root(run_id, base_dir)
    watched: list[Path] = []
    if root.exists():
        for p in root.rglob("*"):
            if p.is_file():
                watched.append(p)
    control = paths.control_bounded_actions_root(base_dir)
    if control.exists():
        for p in control.rglob("*"):
            if p.is_file():
                watched.append(p)
    return watched


class ZeroWriteSpy:
    """Record write/open-for-write attempts during inspection."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.write_calls: list[tuple[Any, ...]] = []
        self._monkeypatch = monkeypatch
        self._original_open = os.open
        self._original_write = os.write
        self._original_mkdir = os.mkdir
        self._original_unlink = os.unlink
        self._original_rename = os.rename

    def install(self) -> None:
        spy = self

        def guarded_open(path, flags, *args, **kwargs):
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                spy.write_calls.append(("open", path, flags))
            return spy._original_open(path, flags, *args, **kwargs)

        def guarded_write(fd, data, *args, **kwargs):
            spy.write_calls.append(("write", fd, len(data)))
            return spy._original_write(fd, data, *args, **kwargs)

        def guarded_mkdir(path, *args, **kwargs):
            spy.write_calls.append(("mkdir", path))
            return spy._original_mkdir(path, *args, **kwargs)

        def guarded_unlink(path, *args, **kwargs):
            spy.write_calls.append(("unlink", path))
            return spy._original_unlink(path, *args, **kwargs)

        def guarded_rename(src, dst, *args, **kwargs):
            spy.write_calls.append(("rename", src, dst))
            return spy._original_rename(src, dst, *args, **kwargs)

        self._monkeypatch.setattr(os, "open", guarded_open)
        self._monkeypatch.setattr(os, "write", guarded_write)
        self._monkeypatch.setattr(os, "mkdir", guarded_mkdir)
        self._monkeypatch.setattr(os, "unlink", guarded_unlink)
        self._monkeypatch.setattr(os, "rename", guarded_rename)


def patch_read_race(
    monkeypatch: pytest.MonkeyPatch,
    *,
    filesystem_status: str,
    ok: bool = False,
) -> None:
    """Simulate secure-read race outcomes (T011–T016)."""

    from htr.advisory_inspection_secure import SecureReadResult

    def fake_read(parent_fd: int, filename: str, **kwargs: Any) -> SecureReadResult:
        return SecureReadResult(ok=ok, filesystem_status=filesystem_status)

    monkeypatch.setattr("htr.advisory_inspection_secure.read_regular_control_file", fake_read)
    monkeypatch.setattr("htr.artifact_inspection.read_regular_control_file", fake_read)


def patch_presence(
    monkeypatch: pytest.MonkeyPatch,
    presence: str,
) -> None:
    def fake_presence(parent_fd: int, filename: str):
        return presence, "filesystem_observed", "file_regular"

    monkeypatch.setattr("htr.advisory_inspection_secure.classify_regular_file_presence", fake_presence)
    monkeypatch.setattr("htr.artifact_inspection.classify_regular_file_presence", fake_presence)


def patch_hash_artifact(
    monkeypatch: pytest.MonkeyPatch,
    *,
    filesystem_status: str = "filesystem_observed",
    file_type_status: str = "file_regular",
    hardlink_status: str = "hardlink_count_one",
    budget_exceeded: bool = False,
    computed_digest: str | None = "sha256:" + "c" * 64,
    observed_size: int | None = 3,
) -> None:
    from htr.advisory_inspection_secure import HashArtifactResult

    def fake_hash(parent_fd: int, rel_components: tuple[str, ...]) -> HashArtifactResult:
        result = HashArtifactResult(
            ok=computed_digest is not None and not budget_exceeded,
            filesystem_status=filesystem_status,
            file_type_status=file_type_status,
            hardlink_status=hardlink_status,
            budget_exceeded=budget_exceeded,
            computed_digest=computed_digest,
            observed_size=observed_size,
        )
        if file_type_status == "file_symlink":
            result.filesystem_status = "filesystem_observed"
        return result

    monkeypatch.setattr("htr.advisory_inspection_secure.hash_artifact_file", fake_hash)
    monkeypatch.setattr("htr.artifact_inspection.hash_artifact_file", fake_hash)


def make_hardlink(path: Path, link_path: Path) -> None:
    if link_path.exists():
        link_path.unlink()
    os.link(path, link_path)


def make_symlink(target: Path, link_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(target)


def make_fifo(path: Path) -> None:
    import stat as stat_mod

    if path.exists():
        path.unlink()
    os.mkfifo(path, mode=0o644)


def make_socket(path: Path) -> None:
    import socket

    if path.exists():
        path.unlink()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind(str(path))
    finally:
        sock.close()


def pad_json_body(base: dict[str, Any], target_body_len: int) -> bytes:
    """Pad JSON object UTF-8 body to exact byte length (no trailing LF)."""
    payload = dict(base)
    lo, hi = 0, target_body_len * 2
    best: bytes | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        payload["pad"] = "x" * mid
        raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if len(raw) == target_body_len:
            return raw
        if len(raw) < target_body_len:
            best = raw
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        raise ValueError("unable to pad JSON to target length")
    # final linear finish from best
    pad_len = int(payload.get("pad", "").__len__() if isinstance(payload.get("pad"), str) else 0)
    for extra in range(pad_len, target_body_len * 2):
        payload["pad"] = "x" * extra
        raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if len(raw) == target_body_len:
            return raw
    raise ValueError(f"unable to pad JSON to {target_body_len} bytes")


def deep_json(depth: int) -> Any:
    if depth <= 0:
        return {"leaf": 1}
    return {"nested": deep_json(depth - 1)}


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# Re-export constants for tests
FORBIDDEN_NAMES = _FORBIDDEN_NAMES
ARTIFACT_AXIS_SCALARS = _ARTIFACT_AXIS_SCALARS
SUPPLEMENTAL_FINDINGS = SUPPLEMENTAL_FINDING_TOKENS
MAX_REFS_PER_MANIFEST = MAX_ARTIFACT_REFERENCES_PER_MANIFEST
MAX_BODY_BYTES = MAX_CONTROL_JSON_BYTES
MAX_FILE_BYTES = MAX_CONTROL_RECORD_FILE_BYTES
MAX_RAW_BYTES = MAX_RAW_READ_BYTES
