"""Storage helpers (stdlib only): atomic JSON, append-only JSONL, strict perms.

Default backend is local-json. The optional google-sheets tracker is handled in
report.py by emitting rows for the `google-workspace` skill; this module stays
dependency-free so the hermetic tests never touch the network.
"""
from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any

import crypto
import paths


@contextlib.contextmanager
def locked(target: Path, timeout: float = 10.0, stale: float = 30.0):
    """Portable advisory lock via an O_EXCL lockfile next to `target`.

    Serializes read-modify-write on shared JSON (the ledger) across concurrent
    processes - a cron re-scan overlapping a manual run, or multiple tenants -
    so one writer can't clobber another's update. A lock older than `stale`
    seconds is treated as abandoned (crashed writer) and broken, so a dead
    process can never deadlock the queue. Works on macOS/Linux/Windows (O_EXCL).
    """
    ensure_dir(target.parent)
    lock = target.with_name(target.name + ".lock")
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            try:
                os.write(fd, str(os.getpid()).encode())
            finally:
                os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > stale:
                    lock.unlink(missing_ok=True)
                    continue
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError(f"could not acquire lock {lock} within {timeout}s")
            time.sleep(0.05)
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            lock.unlink(missing_ok=True)


def _secure(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        pass  # non-POSIX / unsupported FS; HERMES_HOME directory perms still apply


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _secure(path, 0o700)
    return path


def _is_sensitive(path: Path) -> bool:
    """Per-subject docs (dossier, ledger) are sensitive; config/cache are not."""
    try:
        Path(path).resolve().relative_to(paths.subjects_dir().resolve())
        return True
    except (ValueError, OSError):
        return False


def _age_path(path: Path) -> Path:
    return path.with_name(path.name + ".age")


def _atomic_write(path: Path, data: bytes) -> Path:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    _secure(tmp, 0o600)
    os.replace(tmp, path)
    _secure(path, 0o600)
    return path


def write_json(path: Path, obj: Any) -> Path:
    ensure_dir(path.parent)
    data = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if _is_sensitive(path) and crypto.encryption_setting() == "age":
        if not crypto.age_available():
            raise RuntimeError(
                "encryption=age is configured but `age` is not available; "
                "refusing to write PII as plaintext. Install age or run `setup --encryption none`."
            )
        target = _atomic_write(_age_path(path), crypto.encrypt(data))
        if path.exists():
            path.unlink()  # migrate plaintext -> ciphertext
        return target
    target = _atomic_write(path, data)
    ap = _age_path(path)
    if ap.exists():
        ap.unlink()  # encryption turned off -> drop stale ciphertext
    return target


def read_json(path: Path, default: Any = None) -> Any:
    ap = _age_path(path)
    if ap.exists():
        return json.loads(crypto.decrypt(ap.read_bytes()).decode("utf-8"))
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def append_jsonl(path: Path, record: dict) -> Path:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    _secure(path, 0o600)
    return path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out
