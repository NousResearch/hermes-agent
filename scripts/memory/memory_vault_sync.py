#!/usr/bin/env python3
"""Sync redacted Ebbinghaus snapshots and brain docs to a git memory vault."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except ImportError:  # pragma: no cover
    AESGCM = None  # type: ignore[misc, assignment]

REPO_ROOT = Path(__file__).resolve().parents[2]
BRAIN_DIR = REPO_ROOT / "brain"

SECRET_PATTERNS = (
    "api_key",
    "token",
    "secret",
    "password",
    "bearer",
)


def _run(cmd: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def _hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home())
    except Exception:
        return Path.home() / ".hermes"


def _read_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    env_file = _hermes_home() / ".env"
    if not env_file.exists():
        return ""
    for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip().strip('"')
    return ""


def _redact(text: str) -> str:
    lowered = (text or "").lower()
    if any(marker in lowered for marker in SECRET_PATTERNS):
        return "[REDACTED]"
    return text


def _export_ebbinghaus_snapshot(db_path: Path, limit: int = 500) -> dict[str, Any]:
    if not db_path.exists():
        return {"memories": [], "stats": {"count": 0}}
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT memory_id, content, tags, salience, strength, rehearsal_count,
                   retrieval_count, source, created_at, updated_at
            FROM memories
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    finally:
        conn.close()

    memories = []
    for row in rows:
        memories.append(
            {
                "memory_id": row["memory_id"],
                "content": _redact(row["content"] or "")[:700],
                "tags": row["tags"] or "",
                "salience": row["salience"],
                "strength": row["strength"],
                "rehearsal_count": row["rehearsal_count"],
                "retrieval_count": row["retrieval_count"],
                "source": row["source"] or "",
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )
    return {
        "exported_at": datetime.now(UTC).isoformat(),
        "memories": memories,
        "stats": {"count": len(memories)},
    }


def _decode_aes_key(key_b64: str) -> bytes:
    normalized = "".join(key_b64.split())
    for decoder in (base64.urlsafe_b64decode, base64.b64decode):
        try:
            padded = normalized + ("=" * ((-len(normalized)) % 4))
            key = decoder(padded, validate=False)
            if len(key) == 32:
                return key
        except Exception:
            continue
    raise ValueError("MEMORY_VAULT_AES_KEY_B64 must decode to 32 bytes for AES-256-GCM")


def _encrypt_payload(payload: dict[str, Any], key_b64: str) -> bytes:
    if AESGCM is None:
        raise RuntimeError("cryptography package required for memory vault encryption")
    key = _decode_aes_key(key_b64)
    nonce = os.urandom(12)
    plaintext = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, None)
    return nonce + ciphertext


def _ensure_repo(local_path: Path, remote: str) -> None:
    local_path.mkdir(parents=True, exist_ok=True)
    git_dir = local_path / ".git"
    if not git_dir.exists():
        _run(["git", "init"], cwd=local_path)
        if remote:
            _run(["git", "remote", "add", "origin", remote], cwd=local_path, check=False)
    elif remote:
        existing = _run(["git", "remote", "get-url", "origin"], cwd=local_path, check=False)
        if existing.returncode != 0:
            _run(["git", "remote", "add", "origin", remote], cwd=local_path, check=False)


def _copy_brain_docs(vault_root: Path) -> int:
    if not BRAIN_DIR.is_dir():
        return 0
    target = vault_root / "brain"
    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in sorted(BRAIN_DIR.glob("*.md")):
        dst = target / src.name
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        copied += 1
    return copied


def sync_vault(
    *,
    local_path: Path,
    remote: str,
    memory_db: Path,
    encrypt: bool,
    sync_brain: bool,
    auto_push: bool,
    dry_run: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "local_path": str(local_path),
        "remote": remote,
        "dry_run": dry_run,
    }

    if dry_run:
        result["would_export"] = memory_db.exists()
        result["would_copy_brain"] = sync_brain and BRAIN_DIR.is_dir()
        return result

    _ensure_repo(local_path, remote)

    snapshot = _export_ebbinghaus_snapshot(memory_db)
    snapshots_dir = local_path / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    if encrypt:
        key_b64 = _read_env("MEMORY_VAULT_AES_KEY_B64")
        if not key_b64:
            raise RuntimeError("MEMORY_VAULT_AES_KEY_B64 missing from environment/.env")
        try:
            blob = _encrypt_payload(snapshot, key_b64)
            out_path = snapshots_dir / f"ebbinghaus-{stamp}.json.enc"
            out_path.write_bytes(blob)
            result["snapshot_file"] = str(out_path)
            result["encrypted"] = True
        except Exception as exc:
            out_path = snapshots_dir / f"ebbinghaus-{stamp}.json"
            out_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            result["snapshot_file"] = str(out_path)
            result["encrypted"] = False
            result["encryption_warning"] = str(exc)
    else:
        out_path = snapshots_dir / f"ebbinghaus-{stamp}.json"
        out_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result["snapshot_file"] = str(out_path)

    if sync_brain:
        result["brain_files_copied"] = _copy_brain_docs(local_path)

    manifest = {
        "synced_at": datetime.now(UTC).isoformat(),
        "snapshot_file": out_path.name,
        "memory_count": snapshot["stats"]["count"],
    }
    (local_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    _run(["git", "add", "-A"], cwd=local_path)
    status = _run(["git", "status", "--porcelain"], cwd=local_path)
    if status.stdout.strip():
        _run(
            ["git", "commit", "-m", f"memory-vault: sync {stamp}"],
            cwd=local_path,
        )
        result["committed"] = True
        if auto_push and remote:
            _run(["git", "push", "-u", "origin", "HEAD"], cwd=local_path)
            result["pushed"] = True
    else:
        result["committed"] = False

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-path", default="")
    parser.add_argument("--remote", default="")
    parser.add_argument("--memory-db", default="")
    parser.add_argument("--no-encrypt", action="store_true")
    parser.add_argument("--no-brain", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    hermes_home = _hermes_home()
    local_path = Path(args.local_path or _read_env("MEMORY_VAULT_LOCAL_PATH") or hermes_home / "memory-vault")
    remote = args.remote or _read_env("MEMORY_VAULT_REMOTE") or "https://github.com/zapabob/hermes-memory-vault.git"
    memory_db = Path(args.memory_db or hermes_home / "ebbinghaus_memory.db")

    try:
        result = sync_vault(
            local_path=local_path,
            remote=remote,
            memory_db=memory_db,
            encrypt=not args.no_encrypt,
            sync_brain=not args.no_brain,
            auto_push=not args.no_push,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
