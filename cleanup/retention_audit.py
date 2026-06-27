from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from hermes_constants import get_hermes_home
except Exception:  # pragma: no cover
    def get_hermes_home() -> Path:  # type: ignore[no-redef]
        return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")

POLICY_VERSION = "galt-retention-v1"
SEVERE_LOG_TERMS = (
    "critical",
    "traceback",
    "exception",
    "data loss",
    "auth failure",
    "authentication failure",
    "rate limit",
    "429",
    "crash",
    "panic",
    "fatal",
    "service restart",
)
SECRET_NAME_TERMS = (
    "token",
    "secret",
    "credential",
    "credentials",
    "apikey",
    "api_key",
    ".env",
    "auth.json",
)
SQLITE_SUFFIXES = (".db", ".sqlite", "-wal", "-shm", "-journal")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _days_old(path: Path, now: float | None = None) -> float:
    now = time.time() if now is None else now
    return max(0.0, (now - path.stat().st_mtime) / 86400.0)


def _sha256(path: Path) -> str | None:
    if not path.is_file() or path.is_symlink():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _contains_symlink(path: Path) -> bool:
    cur = Path(path.anchor) if path.is_absolute() else Path(".")
    parts = path.parts[1:] if path.is_absolute() else path.parts
    for part in parts:
        cur = cur / part
        try:
            if cur.is_symlink():
                return True
        except OSError:
            return True
    return False


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _is_sqlite(path: Path) -> bool:
    s = str(path)
    return any(s.endswith(suffix) for suffix in SQLITE_SUFFIXES)


def _looks_secret(path: Path) -> bool:
    name = path.name.lower()
    return any(term in name for term in SECRET_NAME_TERMS)


def _read_prefix(path: Path, limit: int = 65536) -> str:
    try:
        with path.open("rb") as f:
            return f.read(limit).decode("utf-8", errors="ignore").lower()
    except OSError:
        return ""


@dataclass(frozen=True)
class RetentionPolicy:
    hermes_home: Path = field(default_factory=get_hermes_home)
    cache_allowlist: tuple[Path, ...] = ()
    raw_log_days: int = 30
    compressed_log_days: int = 180
    severe_log_days: int = 365
    raw_session_days: int = 90
    cache_delete_days: int = 14
    large_file_bytes: int = 500 * 1024 * 1024

    def __post_init__(self) -> None:
        object.__setattr__(self, "hermes_home", Path(self.hermes_home).resolve())
        cache_allowlist = self.cache_allowlist or (Path(self.hermes_home) / "cache",)
        object.__setattr__(
            self,
            "cache_allowlist",
            tuple(Path(p).resolve() for p in cache_allowlist),
        )

    @property
    def cleanup_dir(self) -> Path:
        return self.hermes_home / "cleanup"

    @property
    def manifest_dir(self) -> Path:
        return self.cleanup_dir / "manifests"

    @property
    def audit_log(self) -> Path:
        return self.cleanup_dir / "audit.log"

    @property
    def lock_path(self) -> Path:
        return self.cleanup_dir / "apply.lock"

    def default_roots(self) -> list[Path]:
        roots = [
            self.hermes_home / "logs",
            self.hermes_home / "sessions",
            self.hermes_home / "cron" / "output",
            self.hermes_home / "reviews",
            self.hermes_home / "fulltime-dev" / "logs",
        ]
        profiles = self.hermes_home / "profiles"
        if profiles.exists():
            roots.extend(profiles.glob("*/logs"))
            roots.extend(profiles.glob("*/sessions"))
        kanban = self.hermes_home / "kanban" / "boards"
        if kanban.exists():
            roots.extend(kanban.glob("*/logs"))
        roots.extend(self.cache_allowlist)
        return roots


def inventory_roots(roots: Iterable[Path]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    root_list = [Path(r).resolve() for r in roots]
    for root in root_list:
        if not root.exists():
            continue
        paths = [root]
        if root.is_dir() and not root.is_symlink():
            try:
                paths.extend(root.rglob("*"))
            except OSError:
                continue
        for path in paths:
            try:
                st = path.lstat()
            except OSError as e:
                candidates.append({"path": str(path), "error": str(e)})
                continue
            if path.is_dir() and not path.is_symlink():
                continue
            candidates.append(
                {
                    "path": str(path),
                    "size_bytes": st.st_size,
                    "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
                    "ctime": datetime.fromtimestamp(st.st_ctime, timezone.utc).isoformat(),
                    "mode": st.st_mode,
                    "st_nlink": st.st_nlink,
                    "is_symlink": path.is_symlink(),
                    "is_dir": path.is_dir() and not path.is_symlink(),
                    "is_file": path.is_file() and not path.is_symlink(),
                }
            )
    return {
        "inventory_id": hashlib.sha256((utc_now() + json.dumps([str(r) for r in root_list])).encode()).hexdigest()[:16],
        "created_at": utc_now(),
        "roots": [str(r) for r in root_list],
        "candidates": candidates,
    }


def _protected_reason(path: Path, policy: RetentionPolicy) -> str | None:
    resolved = path.resolve(strict=False)
    hh = policy.hermes_home
    hard_names = {".env", "auth.json", "MEMORY.md", "USER.md", "SOUL.md", "config.yaml"}
    if path.name in hard_names or _looks_secret(path):
        return "protected secret/auth/config/identity path"
    protected_under_home = ("memories", "skills", "plugins", "hermes-agent")
    try:
        rel = resolved.relative_to(hh)
        if rel.parts and rel.parts[0] in protected_under_home:
            return f"protected Hermes {rel.parts[0]} path"
    except ValueError:
        pass
    protected_prefixes = [
        Path.home() / ".hindsight",
        Path.home() / ".pg0",
        Path("/Volumes/T7"),
        Path("/Volumes/AIStore"),
        Path("/Volumes/AIStore2"),
        Path("/Volumes/AIStore3"),
        Path("/Volumes/Shared_Drive"),
        Path("/Volumes/AIStore/galt/memory"),
        Path("/Users/johngalt/Projects/nj-legal-corpus"),
        Path("/Users/johngalt/Projects/nj-legal-corpus-case-engine"),
        Path("/Users/johngalt/Projects/nj-legal-corpus-statutes"),
    ]
    for prefix in protected_prefixes:
        if _under(resolved, prefix):
            return f"protected root {prefix}"
    return None


def _classify(path: Path, inv: dict[str, Any], policy: RetentionPolicy) -> dict[str, Any]:
    base = {
        "path": str(path),
        "relative_root_id": None,
        "file_type": "symlink" if inv.get("is_symlink") else ("file" if inv.get("is_file") else "other"),
        "size_bytes": inv.get("size_bytes", 0),
        "mtime": inv.get("mtime"),
        "ctime": inv.get("ctime"),
        "owner": "default",
        "class": "U",
        "proposed_action": "report",
        "reason": "unclassified",
        "rule_id": "unclassified",
        "reversible": True,
        "dependency_checks": [],
        "active_use": "unknown",
        "git_status": "not_checked",
        "st_nlink": inv.get("st_nlink"),
        "sha256": None,
        "autonomous": False,
    }
    if inv.get("error"):
        base.update(reason=f"stat error: {inv['error']}", rule_id="stat-error")
        return base

    protected = _protected_reason(path, policy)
    if protected:
        base.update({"class": "A", "reason": protected, "rule_id": "hard-no-touch"})
        return base
    if inv.get("is_symlink") or _contains_symlink(path):
        base.update({"reason": "symlink path is report-only", "rule_id": "symlink-refuse"})
        return base
    if (inv.get("st_nlink") or 1) > 1:
        base.update({"reason": "hardlink count > 1 is report-only", "rule_id": "hardlink-refuse"})
        return base
    if _is_sqlite(path):
        base.update({"class": "A", "reason": "SQLite/sidecar files are refused", "rule_id": "sqlite-refuse"})
        return base

    try:
        if path.is_file() and not _looks_secret(path):
            base["sha256"] = _sha256(path)
    except OSError:
        base.update(reason="hash failed; report-only", rule_id="hash-failed")
        return base

    age = _days_old(path)
    hh = policy.hermes_home
    resolved = path.resolve(strict=False)
    rel = None
    try:
        rel = resolved.relative_to(hh)
        top = rel.parts[0] if rel.parts else ""
    except ValueError:
        top = ""

    if path.stat().st_size > policy.large_file_bytes:
        base.update({"class": "G", "reason": ">500MB large file report-only", "rule_id": "large-file-report"})
        return base

    if top == "logs" or "/logs/" in str(resolved):
        text = _read_prefix(path)
        if any(term in text for term in SEVERE_LOG_TERMS):
            base.update({"class": "B", "reason": "severe log preserved for extended retention", "rule_id": "log-severe-preserve"})
            return base
        if age > policy.raw_log_days and not path.name.endswith(".gz"):
            base.update({"class": "B", "proposed_action": "compress", "reason": f"log older than {policy.raw_log_days} days", "rule_id": "log-compress-30d", "autonomous": True})
            return base
        base.update({"class": "B", "reason": "log within retention window", "rule_id": "log-keep"})
        return base

    if top == "sessions" or "/sessions/" in str(resolved):
        base.update({"class": "C", "reason": "sessions are inventory/report-only until relevance classifier is validated", "rule_id": "session-report-only"})
        return base

    if top == "cron" and rel is not None and len(rel.parts) >= 2 and rel.parts[1] == "output":
        if age > 30:
            base.update({"class": "D", "proposed_action": "compress", "reason": "cron output older than 30 days", "rule_id": "cron-output-compress-30d", "autonomous": True})
        else:
            base.update({"class": "D", "reason": "cron output within retention", "rule_id": "cron-output-keep"})
        return base

    if top == "reviews" or "plans" in path.parts:
        base.update({"class": "E", "reason": "plans/reviews/docs require reference checks before archive", "rule_id": "docs-report-only"})
        return base

    if any(_under(resolved, allow) for allow in policy.cache_allowlist):
        if age > policy.cache_delete_days:
            base.update({"class": "F", "proposed_action": "delete", "reason": f"allowlisted cache older than {policy.cache_delete_days} days", "rule_id": "cache-delete-14d", "reversible": False, "autonomous": True})
        else:
            base.update({"class": "F", "reason": "allowlisted cache within retention", "rule_id": "cache-keep"})
        return base

    base.update({"reason": "outside autonomous policy scope", "rule_id": "scope-report-only"})
    return base


def _manifest_hash(candidates: list[dict[str, Any]]) -> str:
    body = json.dumps(candidates, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


def plan_inventory(inventory: dict[str, Any], policy: RetentionPolicy | None = None) -> dict[str, Any]:
    policy = policy or RetentionPolicy()
    candidates = [_classify(Path(item["path"]), item, policy) for item in inventory.get("candidates", [])]
    manifest = {
        "manifest_id": hashlib.sha256((inventory.get("inventory_id", "") + utc_now()).encode()).hexdigest()[:16],
        "policy_version": POLICY_VERSION,
        "created_at": utc_now(),
        "autonomous_policy": "pre-approved deterministic actions only; ambiguous/protected/session/large/repo artifacts report-only",
        "inventory": {k: inventory.get(k) for k in ("inventory_id", "created_at", "roots")},
        "candidates": candidates,
    }
    manifest["manifest_sha256"] = _manifest_hash(candidates)
    return manifest


def write_manifest(manifest: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{manifest['created_at'].replace(':', '').replace('+', 'Z')}-{manifest['manifest_id']}.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def render_external_summary(manifest: dict[str, Any], manifest_path: Path | None = None) -> str:
    counts: dict[str, int] = {}
    bytes_by_action: dict[str, int] = {}
    needs_attention = 0
    for c in manifest.get("candidates", []):
        key = f"Class {c.get('class')} / {c.get('proposed_action')}"
        counts[key] = counts.get(key, 0) + 1
        action = c.get("proposed_action", "report")
        bytes_by_action[action] = bytes_by_action.get(action, 0) + int(c.get("size_bytes") or 0)
        if not c.get("autonomous") and action != "ignore":
            needs_attention += 1
    lines = [
        "Galt retention inventory-plan summary",
        f"manifest_id: {manifest.get('manifest_id')}",
        f"policy_version: {manifest.get('policy_version')}",
        f"candidates: {len(manifest.get('candidates', []))}",
        f"needs_attention: {needs_attention}",
    ]
    if manifest_path:
        lines.append(f"local_manifest: {manifest_path}")
    lines.append("counts:")
    for key in sorted(counts):
        lines.append(f"  - {key}: {counts[key]}")
    lines.append("bytes_by_action:")
    for key in sorted(bytes_by_action):
        lines.append(f"  - {key}: {bytes_by_action[key]}")
    return "\n".join(lines)


def _audit(policy: RetentionPolicy, entry: dict[str, Any]) -> None:
    policy.audit_log.parent.mkdir(parents=True, exist_ok=True)
    with policy.audit_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": utc_now(), **entry}, sort_keys=True) + "\n")


class _FileLock:
    def __init__(self, path: Path):
        self.path = path
        self.fd: int | None = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            self.fd = os.open(self.path, flags, 0o600)
            os.write(self.fd, str(os.getpid()).encode())
        except FileExistsError as e:
            raise RuntimeError(f"apply lock already exists: {self.path}") from e
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.fd is not None:
            os.close(self.fd)
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def _candidate_unchanged(candidate: dict[str, Any], path: Path) -> bool:
    try:
        st = path.lstat()
    except OSError:
        return False
    if int(candidate.get("size_bytes") or -1) != st.st_size:
        return False
    planned_mtime = candidate.get("mtime")
    if planned_mtime:
        current_mtime = datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()
        if current_mtime != planned_mtime:
            return False
    planned_hash = candidate.get("sha256")
    if planned_hash:
        try:
            if _sha256(path) != planned_hash:
                return False
        except OSError:
            return False
    return True


def _safe_unlink_file(path: Path) -> None:
    parent = path.parent
    fd = os.open(parent, os.O_RDONLY)
    try:
        os.unlink(path.name, dir_fd=fd)
    finally:
        os.close(fd)


def _compress_file(path: Path) -> Path:
    gz = path.with_name(path.name + ".gz")
    tmp = path.with_name(path.name + ".gz.tmp")
    with path.open("rb") as src, gzip.open(tmp, "wb") as dst:
        shutil.copyfileobj(src, dst)
    with gzip.open(tmp, "rb") as verify:
        while verify.read(1024 * 1024):
            pass
    tmp.replace(gz)
    _safe_unlink_file(path)
    return gz


def apply_manifest(manifest: dict[str, Any], policy: RetentionPolicy | None = None, autonomous_only: bool = True) -> dict[str, Any]:
    policy = policy or RetentionPolicy()
    result = {"deleted": 0, "compressed": 0, "skipped": 0, "errors": []}
    if manifest.get("policy_version") != POLICY_VERSION:
        raise ValueError("policy version mismatch")
    if manifest.get("manifest_sha256") != _manifest_hash(manifest.get("candidates", [])):
        raise ValueError("manifest hash mismatch")
    with _FileLock(policy.lock_path):
        for c in manifest.get("candidates", []):
            path = Path(c["path"])
            action = c.get("proposed_action")
            if autonomous_only and not c.get("autonomous"):
                result["skipped"] += 1
                continue
            if action not in {"delete", "compress"}:
                result["skipped"] += 1
                continue
            if not path.exists() or path.is_symlink() or _contains_symlink(path) or _is_sqlite(path):
                result["skipped"] += 1
                continue
            try:
                st = path.lstat()
                if st.st_nlink > 1 or not _candidate_unchanged(c, path):
                    result["skipped"] += 1
                    continue
                if action == "delete":
                    if path.is_file():
                        _safe_unlink_file(path)
                        result["deleted"] += 1
                    elif path.is_dir() and not any(path.iterdir()):
                        path.rmdir()
                        result["deleted"] += 1
                    else:
                        result["skipped"] += 1
                        continue
                elif action == "compress" and path.is_file():
                    _compress_file(path)
                    result["compressed"] += 1
                _audit(policy, {"event": "apply", "action": action, "path": str(path), "status": "success"})
            except OSError as e:
                result["errors"].append(f"{path}: {e}")
                _audit(policy, {"event": "apply", "action": action, "path": str(path), "status": "error", "error": str(e)})
    return result


def inventory_plan(policy: RetentionPolicy, output_dir: Path | None = None) -> tuple[dict[str, Any], Path]:
    inv = inventory_roots(policy.default_roots())
    manifest = plan_inventory(inv, policy)
    path = write_manifest(manifest, output_dir or policy.manifest_dir)
    return manifest, path


def autonomous_run(policy: RetentionPolicy | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    """Inventory, plan, and apply only pre-approved autonomous actions.

    This is the unattended operating mode. It still writes a full local manifest
    and emits only a redacted summary suitable for gateway delivery.
    """
    policy = policy or RetentionPolicy()
    manifest, path = inventory_plan(policy, output_dir)
    apply_result = apply_manifest(manifest, policy, autonomous_only=True)
    return {
        "manifest_path": str(path),
        "summary": render_external_summary(manifest, path),
        "apply_result": apply_result,
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Galt/Hermes retention audit")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ip = sub.add_parser("inventory-plan")
    ip.add_argument("--output-dir", type=Path)
    ip.add_argument("--summary-only", action="store_true")
    ip.add_argument("--cache-allowlist", action="append", type=Path, default=[])
    ar = sub.add_parser("autonomous-run")
    ar.add_argument("--output-dir", type=Path)
    ar.add_argument("--cache-allowlist", action="append", type=Path, default=[])
    ap = sub.add_parser("apply")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--autonomous-only", action="store_true", default=True)
    ap.add_argument("--cache-allowlist", action="append", type=Path, default=[])
    args = parser.parse_args(argv)

    policy = RetentionPolicy(cache_allowlist=tuple(args.cache_allowlist))
    if args.cmd == "inventory-plan":
        manifest, path = inventory_plan(policy, args.output_dir)
        if args.summary_only:
            print(render_external_summary(manifest, path))
        else:
            print(json.dumps({"manifest": str(path), "summary": render_external_summary(manifest, path)}, indent=2))
        return 0
    if args.cmd == "autonomous-run":
        outcome = autonomous_run(policy, args.output_dir)
        print(outcome["summary"])
        print("apply_result:")
        print(json.dumps(outcome["apply_result"], indent=2, sort_keys=True))
        return 0
    if args.cmd == "apply":
        result = apply_manifest(_load_manifest(args.manifest), policy, autonomous_only=args.autonomous_only)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
