from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from hermes_constants import get_hermes_home

EXPORT_SCHEMA_VERSION = 1
EXPORTER_VERSION = "2026.05.02.1"
DEFAULT_OUTPUT_ROOT = Path("/var/lib/hermes-export")
DEFAULT_KEEP_RELEASES = 96
DEFAULT_LOG_TAIL_LINES = 100
DEFAULT_HASH_CHUNK_SIZE = 1024 * 1024

DIRECT_COPY_FILES = {
    "SOUL.md": "SOUL.md",
    "BOOT.md": "BOOT.md",
    ".skills_prompt_snapshot.json": "skills_snapshot.json",
}
LOW_RISK_LOG_TAILS = (
    "disk-monitor.log",
    "ram-monitor.log",
    "disk-hygiene.log",
    "update.log",
)
SENSITIVE_EXCLUSIONS = [
    ".env",
    "auth.json",
    "*token*.json",
    "*secret*.json",
    "x_credentials.env",
    "state.db*",
    "raw sessions/ content",
    "raw memories/ content",
    "whatsapp/",
    "cron/ output content",
    "pastes/",
    "document_cache/",
    "sandboxes/",
    "raw agent.log/gateway.log/errors.log/mcp-stderr.log",
]
PROFILE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class ExporterError(RuntimeError):
    pass


@dataclass
class HashCache:
    path: Path
    data: Dict[str, Dict[str, Any]]

    @classmethod
    def load(cls, path: Path) -> "HashCache":
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    return cls(path=path, data=raw)
            except Exception:
                pass
        return cls(path=path, data={})

    def lookup(self, file_path: Path, st: os.stat_result) -> Optional[str]:
        entry = self.data.get(str(file_path))
        if not isinstance(entry, dict):
            return None
        if entry.get("size_bytes") != int(st.st_size):
            return None
        if entry.get("mtime_ns") != int(st.st_mtime_ns):
            return None
        digest = entry.get("sha256")
        return str(digest) if digest else None

    def store(self, file_path: Path, st: os.stat_result, digest: str) -> None:
        self.data[str(file_path)] = {
            "size_bytes": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
            "sha256": digest,
        }


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _release_name(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _path_meta(path: Path, *, include_sha256: bool = False, hash_cache: Optional[HashCache] = None) -> Dict[str, Any]:
    st = path.stat()
    meta: Dict[str, Any] = {
        "path": str(path),
        "size_bytes": int(st.st_size),
        "mtime": _isoformat_utc(datetime.fromtimestamp(st.st_mtime, timezone.utc)),
    }
    if include_sha256:
        digest = hash_cache.lookup(path, st) if hash_cache else None
        if digest is None:
            digest = _sha256_file(path)
            if hash_cache:
                hash_cache.store(path, st, digest)
        meta["sha256"] = digest
    return meta


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(DEFAULT_HASH_CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _ensure_dir(path: Path, *, mode: int = 0o2750) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, mode)
    except OSError as exc:
        raise ExporterError(f"Failed to enforce directory mode {oct(mode)} on {path}: {exc}") from exc


def _chmod_file(path: Path, mode: int = 0o640) -> None:
    try:
        os.chmod(path, mode)
    except OSError as exc:
        raise ExporterError(f"Failed to enforce file mode {oct(mode)} on {path}: {exc}") from exc


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_bytes(path: Path, data: bytes) -> None:
    _ensure_dir(path.parent)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o640)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        try:
            os.fchmod(handle.fileno(), 0o640)
        except OSError as exc:
            raise ExporterError(f"Failed to enforce temp file mode 0o640 on {tmp_path}: {exc}") from exc
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    _chmod_file(path)
    _fsync_dir(path.parent)


def _write_text(path: Path, text: str) -> None:
    _write_bytes(path, text.encode("utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    _write_text(path, text)


def _copy_file(src: Path, dst: Path) -> None:
    _write_bytes(dst, src.read_bytes())


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.is_dir():
        raise ExporterError(f"Expected directory tree at {src}")
    _ensure_dir(dst)
    for child in sorted(src.iterdir(), key=lambda item: item.name):
        target = dst / child.name
        if child.is_symlink():
            raise ExporterError(f"Refusing to publish symlink from staging tree: {child}")
        if child.is_dir():
            _copy_tree(child, target)
            continue
        if child.is_file():
            _copy_file(child, target)
            continue
        raise ExporterError(f"Unsupported staging entry type: {child}")
    _fsync_dir(dst)


def _remove_if_exists(path: Path) -> None:
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    except OSError:
        pass


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _reject_symlink(path: Path, *, context: str) -> None:
    if path.is_symlink():
        raise ExporterError(f"Refusing to follow symlink for {context}: {path}")


def _read_last_lines(path: Path, limit: int) -> List[str]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    return [line.rstrip("\n") for line in lines[-limit:]]


def _expected_export_group(output_root: Path) -> tuple[int, str]:
    expected_gid = output_root.stat().st_gid
    group_name = str(expected_gid)
    try:
        import grp
        group_name = grp.getgrgid(expected_gid).gr_name
    except Exception:
        pass
    return expected_gid, group_name


def _can_assign_gid(expected_gid: int) -> bool:
    if os.geteuid() == 0:
        return True
    gids = set()
    try:
        gids.update(os.getgroups())
    except OSError:
        pass
    try:
        gids.add(os.getegid())
    except OSError:
        pass
    return expected_gid in gids


def _run_checked(command: List[str], *, context: str) -> None:
    proc = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise ExporterError(f"{context} failed: {detail or f'returncode={proc.returncode}'}")


def _enforce_export_tree_contract(output_root: Path) -> None:
    expected_gid, group_name = _expected_export_group(output_root)
    try:
        os.chmod(output_root, 0o2750)
    except OSError as exc:
        raise ExporterError(f"Failed to enforce export root mode on {output_root}: {exc}") from exc

    if _can_assign_gid(expected_gid):
        roots = [output_root]
        for root, dirs, files in os.walk(output_root, followlinks=False):
            roots.extend(Path(root) / name for name in dirs + files)
        for entry in roots:
            if entry.is_symlink():
                raise ExporterError(f"symlink not allowed in export tree: {entry}")
            if entry != output_root:
                try:
                    os.chown(entry, -1, expected_gid)
                except OSError as exc:
                    raise ExporterError(f"Failed to assign group {group_name} to {entry}: {exc}") from exc
            try:
                if entry.is_dir():
                    os.chmod(entry, 0o2750)
                else:
                    os.chmod(entry, 0o640)
            except OSError as exc:
                raise ExporterError(f"Failed to enforce export mode on {entry}: {exc}") from exc
        return

    root_group = group_name
    root_str = str(output_root)
    _run_checked(["sudo", "-n", "chmod", "2750", root_str], context="sudo chmod export root")
    _run_checked(
        ["sudo", "-n", "find", root_str, "-mindepth", "1", "-exec", "chgrp", "-h", root_group, "{}", "+"],
        context="sudo chgrp export descendants",
    )
    _run_checked(
        ["sudo", "-n", "find", root_str, "-mindepth", "1", "-type", "d", "-exec", "chmod", "2750", "{}", "+"],
        context="sudo chmod export dirs",
    )
    _run_checked(
        ["sudo", "-n", "find", root_str, "-mindepth", "1", "-type", "f", "-exec", "chmod", "640", "{}", "+"],
        context="sudo chmod export files",
    )


def _verify_export_tree_contract(output_root: Path) -> None:
    expected_gid, group_name = _expected_export_group(output_root)

    problems: List[str] = []
    roots = [output_root]
    for root, dirs, files in os.walk(output_root, followlinks=False):
        roots.extend(Path(root) / name for name in dirs + files)
    for entry in roots:
        if entry.is_symlink():
            problems.append(f"symlink not allowed in export tree: {entry}")
            continue
        st = entry.stat()
        mode = stat.S_IMODE(st.st_mode)
        if st.st_gid != expected_gid:
            problems.append(f"wrong group on {entry}: gid={st.st_gid} expected={expected_gid}")
        if entry.is_dir():
            if (mode & 0o2750) != 0o2750:
                problems.append(f"wrong dir mode on {entry}: {oct(mode)} expected at least 0o2750")
        else:
            if (mode & 0o040) != 0o040:
                problems.append(f"group-read missing on file {entry}: {oct(mode)}")
    if problems:
        raise ExporterError(
            "Export tree contract verification failed for group "
            f"{group_name}: " + "; ".join(problems[:12])
        )


def _walk_regular_files(base: Path) -> Iterable[Path]:
    for root, dirs, files in os.walk(base, followlinks=False):
        dirs.sort()
        files.sort()
        root_path = Path(root)
        for name in files:
            path = root_path / name
            if path.is_file() and not path.is_symlink():
                yield path


def _inventory_files(base: Path, *, include_sha256: bool, hash_cache: Optional[HashCache]) -> Dict[str, Any]:
    if not base.exists():
        return {
            "present": False,
            "entries": [],
            "file_count": 0,
            "total_bytes": 0,
            "oldest_mtime": None,
            "newest_mtime": None,
        }
    if not base.is_dir():
        raise ExporterError(f"Expected directory at {base}")

    entries: List[Dict[str, Any]] = []
    total_bytes = 0
    mtimes: List[float] = []
    for file_path in _walk_regular_files(base):
        st = file_path.stat()
        total_bytes += int(st.st_size)
        mtimes.append(st.st_mtime)
        entry = {
            "path": str(file_path.relative_to(base)),
            "size_bytes": int(st.st_size),
            "mtime": _isoformat_utc(datetime.fromtimestamp(st.st_mtime, timezone.utc)),
        }
        if include_sha256:
            digest = hash_cache.lookup(file_path, st) if hash_cache else None
            if digest is None:
                digest = _sha256_file(file_path)
                if hash_cache:
                    hash_cache.store(file_path, st, digest)
            entry["sha256"] = digest
        entries.append(entry)

    return {
        "present": True,
        "entries": sorted(entries, key=lambda item: item["path"]),
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "oldest_mtime": _isoformat_utc(datetime.fromtimestamp(min(mtimes), timezone.utc)) if mtimes else None,
        "newest_mtime": _isoformat_utc(datetime.fromtimestamp(max(mtimes), timezone.utc)) if mtimes else None,
    }


def _aggregate_directory(base: Path) -> Dict[str, Any]:
    if not base.exists():
        return {
            "present": False,
            "file_count": 0,
            "total_bytes": 0,
            "oldest_mtime": None,
            "newest_mtime": None,
        }
    if not base.is_dir():
        raise ExporterError(f"Expected directory at {base}")

    file_count = 0
    total_bytes = 0
    mtimes: List[float] = []
    for file_path in _walk_regular_files(base):
        st = file_path.stat()
        file_count += 1
        total_bytes += int(st.st_size)
        mtimes.append(st.st_mtime)
    return {
        "present": True,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "oldest_mtime": _isoformat_utc(datetime.fromtimestamp(min(mtimes), timezone.utc)) if mtimes else None,
        "newest_mtime": _isoformat_utc(datetime.fromtimestamp(max(mtimes), timezone.utc)) if mtimes else None,
    }


def _inventory_logs(logs_dir: Path) -> Dict[str, Any]:
    if not logs_dir.exists():
        return {"present": False, "entries": []}
    entries = []
    for child in sorted(logs_dir.iterdir()):
        if child.is_file() and not child.is_symlink():
            st = child.stat()
            entries.append({
                "path": child.name,
                "size_bytes": int(st.st_size),
                "mtime": _isoformat_utc(datetime.fromtimestamp(st.st_mtime, timezone.utc)),
            })
    return {"present": True, "entries": entries}


def _inventory_checkpoints(checkpoints_dir: Path) -> Dict[str, Any]:
    if not checkpoints_dir.exists():
        return {"present": False, "root_files": [], "checkpoints": [], "checkpoint_count": 0}

    root_files = []
    checkpoints = []
    for child in sorted(checkpoints_dir.iterdir()):
        if child.is_file() and not child.is_symlink():
            st = child.stat()
            root_files.append({
                "path": child.name,
                "size_bytes": int(st.st_size),
                "mtime": _isoformat_utc(datetime.fromtimestamp(st.st_mtime, timezone.utc)),
            })
            continue
        if not child.is_dir():
            continue
        agg = _aggregate_directory(child)
        checkpoints.append({
            "path": child.name,
            **agg,
        })
    checkpoints.sort(key=lambda item: (item.get("newest_mtime") or "", item["path"]), reverse=True)
    return {
        "present": True,
        "root_files": root_files,
        "checkpoints": checkpoints,
        "checkpoint_count": len(checkpoints),
    }


def _count_skills(skills_dir: Path) -> int:
    if not skills_dir.is_dir():
        return 0
    count = 0
    for skill_path in skills_dir.rglob("SKILL.md"):
        skill_str = str(skill_path)
        if "/.hub/" in skill_str or "/.git/" in skill_str:
            continue
        count += 1
    return count


def _inventory_skills(skills_dir: Path, hash_cache: Optional[HashCache]) -> Dict[str, Any]:
    if not skills_dir.exists():
        return {"present": False, "entries": [], "skill_count": 0}

    entries = []
    for skill_path in sorted(skills_dir.rglob("SKILL.md")):
        skill_str = str(skill_path)
        if "/.hub/" in skill_str or "/.git/" in skill_str:
            continue
        st = skill_path.stat()
        digest = hash_cache.lookup(skill_path, st) if hash_cache else None
        if digest is None:
            digest = _sha256_file(skill_path)
            if hash_cache:
                hash_cache.store(skill_path, st, digest)
        entries.append({
            "skill_id": str(skill_path.parent.relative_to(skills_dir)),
            "size_bytes": int(st.st_size),
            "mtime": _isoformat_utc(datetime.fromtimestamp(st.st_mtime, timezone.utc)),
            "sha256": digest,
        })

    return {
        "present": True,
        "skill_count": len(entries),
        "entries": entries,
    }


def _snapshot_summary(snapshot_path: Path) -> Dict[str, Any]:
    if not snapshot_path.exists():
        return {"present": False, "parseable": False, "manifest_count": 0}
    try:
        payload = _read_json(snapshot_path)
    except Exception as exc:
        return {
            "present": True,
            "parseable": False,
            "manifest_count": 0,
            "error": str(exc),
        }
    manifest = payload.get("manifest") if isinstance(payload, dict) else None
    manifest_count = len(manifest) if isinstance(manifest, dict) else 0
    return {
        "present": True,
        "parseable": True,
        "manifest_count": manifest_count,
        "version": payload.get("version") if isinstance(payload, dict) else None,
    }


def _cron_delivery_mode(value: Any) -> str:
    text = str(value or "origin").strip().lower()
    if text in {"origin", "local"}:
        return text
    return "explicit"


def _collect_cron_inventory(hermes_home: Path) -> Dict[str, Any]:
    jobs_path = hermes_home / "cron" / "jobs.json"
    if not jobs_path.exists():
        return {"present": False, "job_count": 0, "jobs": []}

    payload = _read_json(jobs_path)
    jobs = payload.get("jobs", []) if isinstance(payload, dict) else payload
    safe_jobs = []
    for raw in jobs or []:
        if not isinstance(raw, dict):
            continue
        skills = raw.get("skills")
        if isinstance(skills, list):
            normalized_skills = [str(item).strip() for item in skills if str(item).strip()]
            skill_count = len(normalized_skills)
        elif raw.get("skill"):
            normalized_skills = [str(raw.get("skill")).strip()]
            skill_count = 1
        else:
            normalized_skills = []
            skill_count = 0
        prompt = str(raw.get("prompt") or "")
        name = str(raw.get("name") or "").strip() or None
        auto_label_source = (prompt or (normalized_skills[0] if normalized_skills else None) or "cron job")[:50].strip()
        if prompt and (name is None or name == auto_label_source):
            safe_name = "[auto-redacted-from-prompt]"
        else:
            safe_name = name
        safe_jobs.append({
            "id": raw.get("id"),
            "name": safe_name,
            "schedule": raw.get("schedule"),
            "schedule_display": raw.get("schedule_display"),
            "enabled": bool(raw.get("enabled", True)),
            "state": raw.get("state"),
            "last_status": raw.get("last_status"),
            "last_run_at": raw.get("last_run_at"),
            "next_run_at": raw.get("next_run_at"),
            "paused_at": raw.get("paused_at"),
            "paused_reason": raw.get("paused_reason"),
            "repeat": raw.get("repeat"),
            "has_script": bool(raw.get("script")),
            "skill_count": skill_count,
            "delivery_mode": _cron_delivery_mode(raw.get("deliver")),
        })
    safe_jobs.sort(key=lambda item: ((item.get("name") or ""), (item.get("id") or "")))
    return {
        "present": True,
        "job_count": len(safe_jobs),
        "enabled_count": len([job for job in safe_jobs if job["enabled"]]),
        "jobs": safe_jobs,
    }


def _profile_shared_symlinks(profile_dir: Path) -> List[str]:
    shared = []
    for name in (".env", "SOUL.md", "auth.json", "google_token.json", "google_client_secret.json", "skills"):
        path = profile_dir / name
        if path.is_symlink():
            shared.append(name)
    return shared


def _fingerprint(path: Path, hash_cache: Optional[HashCache]) -> Dict[str, Any]:
    if not path.exists():
        return {"present": False}
    st = path.stat()
    digest = hash_cache.lookup(path, st) if hash_cache else None
    if digest is None:
        digest = _sha256_file(path)
        if hash_cache:
            hash_cache.store(path, st, digest)
    return {
        "present": True,
        "size_bytes": int(st.st_size),
        "mtime": _isoformat_utc(datetime.fromtimestamp(st.st_mtime, timezone.utc)),
        "sha256": digest,
    }


def _collect_profile_inventory(hermes_home: Path, hash_cache: Optional[HashCache]) -> Dict[str, Any]:
    profiles = []
    default_profile = {
        "name": "default",
        "is_default": True,
        "skill_count": _count_skills(hermes_home / "skills"),
        "config": _fingerprint(hermes_home / "config.yaml", hash_cache),
        "sessions": _aggregate_directory(hermes_home / "sessions"),
        "memories": _aggregate_directory(hermes_home / "memories"),
        "shared_symlinks": [],
    }
    profiles.append(default_profile)

    profiles_root = hermes_home / "profiles"
    if profiles_root.is_dir():
        for entry in sorted(profiles_root.iterdir()):
            if not entry.is_dir():
                continue
            if not PROFILE_NAME_RE.match(entry.name):
                continue
            profiles.append({
                "name": entry.name,
                "is_default": False,
                "skill_count": _count_skills(entry / "skills"),
                "config": _fingerprint(entry / "config.yaml", hash_cache),
                "sessions": _aggregate_directory(entry / "sessions"),
                "memories": _aggregate_directory(entry / "memories"),
                "shared_symlinks": _profile_shared_symlinks(entry),
            })
    return {"profile_count": len(profiles), "profiles": profiles}


def _read_meminfo() -> Dict[str, int]:
    meminfo: Dict[str, int] = {}
    path = Path("/proc/meminfo")
    if not path.exists():
        return meminfo
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        parts = raw.strip().split()
        if not parts:
            continue
        try:
            value_kib = int(parts[0])
        except ValueError:
            continue
        meminfo[key] = value_kib * 1024
    return meminfo


def _collect_resources() -> Dict[str, Any]:
    disk = shutil.disk_usage("/")
    meminfo = _read_meminfo()
    load1, load5, load15 = os.getloadavg()
    total = int(disk.total)
    free = int(disk.free)
    used = int(disk.used)
    disk_free_pct = (free / total) if total else None
    return {
        "hostname": socket.gethostname(),
        "disk": {
            "path": "/",
            "total_bytes": total,
            "used_bytes": used,
            "free_bytes": free,
            "free_pct": round(disk_free_pct, 4) if disk_free_pct is not None else None,
        },
        "memory": {
            "mem_total_bytes": meminfo.get("MemTotal"),
            "mem_available_bytes": meminfo.get("MemAvailable"),
            "swap_total_bytes": meminfo.get("SwapTotal"),
            "swap_free_bytes": meminfo.get("SwapFree"),
        },
        "loadavg": {
            "load1": round(load1, 4),
            "load5": round(load5, 4),
            "load15": round(load15, 4),
        },
    }


def _run_command(argv: List[str]) -> Dict[str, Any]:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=30, check=False)
        return {
            "returncode": int(proc.returncode),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as exc:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": str(exc),
        }


_SECRET_OPTION_RE = re.compile(r"(?i)((?:--?|)(?:api[-_]?key|access[-_]?token|refresh[-_]?token|token|secret|password|authorization)\s*[= ]\s*)(\S+)")
_SECRET_ENV_RE = re.compile(r"\b([A-Z0-9_]*(?:TOKEN|SECRET|KEY|PASSWORD)[A-Z0-9_]*)=(\S+)")
_BEARER_RE = re.compile(r"(?i)\b(Bearer\s+)(\S+)")
_PREFIX_SECRET_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]+|rk_(?:liv|test)_[A-Za-z0-9_-]+|xox[baprs]-[A-Za-z0-9-]+|gh[pousr]_[A-Za-z0-9_]+|AIza[0-9A-Za-z\-_]+|eyJ[A-Za-z0-9._-]{20,})\b"
)


def _sanitize_command(command: str) -> str:
    command = _SECRET_OPTION_RE.sub(lambda m: f"{m.group(1)}<REDACTED>", command)
    command = _SECRET_ENV_RE.sub(lambda m: f"{m.group(1)}=<REDACTED>", command)
    command = _BEARER_RE.sub(lambda m: f"{m.group(1)}<REDACTED>", command)
    command = _PREFIX_SECRET_RE.sub("<REDACTED>", command)
    return " ".join(command.split())


def _process_label(command: str) -> str:
    sanitized = _sanitize_command(command)
    lowered = sanitized.lower()
    if "runtime_export" in lowered:
        return "hermes runtime export"
    if "gateway run" in lowered:
        return "hermes gateway"
    if "@stripe/mcp" in lowered or "stripe/mcp" in lowered:
        return "stripe mcp"
    if "enzyme" in lowered:
        return "enzyme"
    if "mcp" in lowered:
        return "mcp"
    first = sanitized.split()[0] if sanitized else "process"
    return Path(first).name


def _collect_processes() -> Dict[str, Any]:
    result = _run_command(["ps", "-eo", "pid=,user=,ppid=,etimes=,pcpu=,pmem=,args="])
    if result["returncode"] != 0:
        return {"present": False, "error": result["stderr"].strip()}

    entries = []
    for raw_line in result["stdout"].splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 6)
        if len(parts) < 7:
            continue
        pid, user, ppid, etimes, pcpu, pmem, args = parts
        lowered = args.lower()
        if not any(token in lowered for token in ("hermes", "kael", "enzyme", "mcp")):
            continue
        entries.append({
            "pid": int(pid),
            "user": user,
            "ppid": int(ppid),
            "elapsed_seconds": int(float(etimes)),
            "cpu_pct": float(pcpu),
            "mem_pct": float(pmem),
            "label": _process_label(args),
        })
    entries.sort(key=lambda item: item["pid"])
    return {"present": True, "entries": entries}


def _collect_versions() -> Dict[str, Any]:
    out = {}
    for name in ("kael", "hermes"):
        result = _run_command([name, "--version"])
        out[name] = {
            "returncode": result["returncode"],
            "stdout": result["stdout"].strip(),
            "stderr": result["stderr"].strip(),
        }
    return out


def _collect_gateway_service_state() -> Dict[str, Any]:
    show = _run_command([
        "systemctl",
        "--user",
        "show",
        "hermes-gateway.service",
        "--property=ActiveState,SubState,MainPID,UnitFileState,ExecMainStartTimestamp",
    ])
    if show["returncode"] != 0:
        return {"present": False, "error": show["stderr"].strip()}
    payload: Dict[str, Any] = {"present": True}
    for line in show["stdout"].splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        payload[key.strip()] = value.strip()
    return payload


def _collect_live_repo_state(hermes_home: Path) -> Dict[str, Any]:
    repo = hermes_home / "hermes-agent"
    if not repo.is_dir():
        return {"present": False}

    def _git(*args: str) -> Dict[str, Any]:
        return _run_command(["git", "-C", str(repo), *args])

    head = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    status = _git("status", "--short")
    return {
        "present": True,
        "repo_path": str(repo),
        "head": head["stdout"].strip() if head["returncode"] == 0 else None,
        "branch": branch["stdout"].strip() if branch["returncode"] == 0 else None,
        "dirty": bool(status["stdout"].strip()) if status["returncode"] == 0 else None,
        "dirty_paths": [line.strip() for line in status["stdout"].splitlines() if line.strip()][:50] if status["returncode"] == 0 else [],
    }


def _collect_direct_files(hermes_home: Path, release_dir: Path, warnings: List[str]) -> Dict[str, Any]:
    direct = {}
    snapshot_summary = _snapshot_summary(hermes_home / ".skills_prompt_snapshot.json")
    for source_name, target_name in DIRECT_COPY_FILES.items():
        src = hermes_home / source_name
        dst = release_dir / target_name
        if not src.exists():
            direct[target_name] = {"present": False}
            warnings.append(f"Missing direct-copy file: {source_name}")
            continue
        if src.is_symlink():
            direct[target_name] = {"present": True, "copied": False, "symlink_rejected": True}
            warnings.append(f"Rejected symlink direct-copy source: {source_name}")
            continue
        if source_name == ".skills_prompt_snapshot.json" and not snapshot_summary.get("parseable"):
            direct[target_name] = {"present": True, "copied": False, **snapshot_summary}
            warnings.append("skills snapshot exists but is not parseable; skipped direct export")
            continue
        _copy_file(src, dst)
        direct[target_name] = {
            "present": True,
            "copied": True,
            "source": str(src),
            "export_path": target_name,
        }
        if source_name == ".skills_prompt_snapshot.json":
            direct[target_name].update(snapshot_summary)
    return direct


def _collect_log_tails(logs_dir: Path, line_limit: int, warnings: List[str]) -> Dict[str, Any]:
    tails = {}
    for name in LOW_RISK_LOG_TAILS:
        path = logs_dir / name
        if not path.exists():
            tails[name] = {"present": False, "lines": []}
            continue
        if path.is_symlink():
            tails[name] = {"present": True, "lines": [], "symlink_rejected": True}
            warnings.append(f"Rejected symlink log tail source: {name}")
            continue
        tails[name] = {
            "present": True,
            "lines": _read_last_lines(path, line_limit),
            "size_bytes": path.stat().st_size,
            "mtime": _isoformat_utc(datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)),
        }
    return tails


def _build_health_flags(summary: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    resources = summary.get("ops", {}).get("resources", {})
    disk = resources.get("disk", {})
    memory = resources.get("memory", {})
    heapdumps = summary.get("inventories", {}).get("heapdumps", {})
    gateway = summary.get("ops", {}).get("gateway_service", {})
    repo = summary.get("ops", {}).get("live_repo", {})

    free_bytes = disk.get("free_bytes")
    free_pct = disk.get("free_pct")
    if isinstance(free_bytes, int) and free_bytes < 10 * 1024 * 1024 * 1024:
        flags.append("low_disk_free_bytes")
    if isinstance(free_pct, float) and free_pct < 0.15:
        flags.append("low_disk_free_pct")

    mem_avail = memory.get("mem_available_bytes")
    if isinstance(mem_avail, int) and mem_avail < 768 * 1024 * 1024:
        flags.append("low_memory_available_bytes")

    swap_total = memory.get("swap_total_bytes")
    swap_free = memory.get("swap_free_bytes")
    if isinstance(swap_total, int) and swap_total > 0 and isinstance(swap_free, int) and swap_free < 256 * 1024 * 1024:
        flags.append("low_swap_free_bytes")

    if heapdumps.get("file_count", 0):
        flags.append("heapdumps_present")

    if gateway.get("present") and gateway.get("ActiveState") != "active":
        flags.append("gateway_not_active")

    if repo.get("present") and repo.get("dirty"):
        flags.append("live_repo_dirty")

    return flags


def export_runtime_snapshot(
    *,
    output_root: Path,
    keep_releases: int = DEFAULT_KEEP_RELEASES,
    log_tail_lines: int = DEFAULT_LOG_TAIL_LINES,
) -> Dict[str, Any]:
    exported_at = _now_utc()
    exported_at_iso = _isoformat_utc(exported_at)
    release_name = _release_name(exported_at)
    hermes_home = get_hermes_home().resolve()

    if not hermes_home.is_dir():
        raise ExporterError(f"Hermes home does not exist: {hermes_home}")

    output_root = output_root.expanduser().resolve()
    _ensure_dir(output_root)
    releases_dir = output_root / "releases"
    _ensure_dir(releases_dir)

    hash_cache = HashCache.load(output_root / ".hash-cache.json")

    stage_dir = releases_dir / f".staging-{release_name}-{os.getpid()}"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    _ensure_dir(stage_dir)
    _ensure_dir(stage_dir / "inventory")
    _ensure_dir(stage_dir / "ops")
    _ensure_dir(stage_dir / "tails")

    warnings: List[str] = []

    direct_files = _collect_direct_files(hermes_home, stage_dir, warnings)

    inventories = {
        "heapdumps": _inventory_files(hermes_home / "heapdumps", include_sha256=True, hash_cache=hash_cache),
        "archives": _inventory_files(hermes_home / "archives", include_sha256=True, hash_cache=hash_cache),
        "checkpoints": _inventory_checkpoints(hermes_home / "checkpoints"),
        "logs": _inventory_logs(hermes_home / "logs"),
        "sessions": _aggregate_directory(hermes_home / "sessions"),
        "memories": _aggregate_directory(hermes_home / "memories"),
        "skills": _inventory_skills(hermes_home / "skills", hash_cache),
    }

    profiles = _collect_profile_inventory(hermes_home, hash_cache)
    config_fingerprints = {
        "config.yaml": _fingerprint(hermes_home / "config.yaml", hash_cache),
        "SOUL.md": _fingerprint(hermes_home / "SOUL.md", hash_cache),
        "BOOT.md": _fingerprint(hermes_home / "BOOT.md", hash_cache),
        ".skills_prompt_snapshot.json": _fingerprint(hermes_home / ".skills_prompt_snapshot.json", hash_cache),
        "profiles": {
            item["name"]: item["config"]
            for item in profiles["profiles"]
            if not item["is_default"]
        },
    }

    ops = {
        "versions": _collect_versions(),
        "processes": _collect_processes(),
        "cron": _collect_cron_inventory(hermes_home),
        "profiles": profiles,
        "config_fingerprints": config_fingerprints,
        "resources": _collect_resources(),
        "gateway_service": _collect_gateway_service_state(),
        "live_repo": _collect_live_repo_state(hermes_home),
    }

    tails = _collect_log_tails(hermes_home / "logs", log_tail_lines, warnings)

    summary = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "metadata": {
            "exporter_version": EXPORTER_VERSION,
            "exported_at": exported_at_iso,
            "release_name": release_name,
            "hermes_home": str(hermes_home),
            "output_root": str(output_root),
            "keep_releases": int(keep_releases),
            "log_tail_lines": int(log_tail_lines),
        },
        "direct_files": direct_files,
        "inventories": inventories,
        "ops": ops,
        "tails": tails,
        "excluded_paths": SENSITIVE_EXCLUSIONS,
        "warnings": warnings,
    }
    summary["health_flags"] = _build_health_flags(summary)

    _write_json(stage_dir / "inventory" / "heapdumps.json", inventories["heapdumps"])
    _write_json(stage_dir / "inventory" / "archives.json", inventories["archives"])
    _write_json(stage_dir / "inventory" / "checkpoints.json", inventories["checkpoints"])
    _write_json(stage_dir / "inventory" / "logs.json", inventories["logs"])
    _write_json(stage_dir / "inventory" / "sessions.json", inventories["sessions"])
    _write_json(stage_dir / "inventory" / "memories.json", inventories["memories"])
    _write_json(stage_dir / "inventory" / "skills.json", inventories["skills"])

    _write_json(stage_dir / "ops" / "versions.json", ops["versions"])
    _write_json(stage_dir / "ops" / "processes.json", ops["processes"])
    _write_json(stage_dir / "ops" / "cron.json", ops["cron"])
    _write_json(stage_dir / "ops" / "profiles.json", ops["profiles"])
    _write_json(stage_dir / "ops" / "config_fingerprints.json", ops["config_fingerprints"])
    _write_json(stage_dir / "ops" / "resources.json", ops["resources"])
    _write_json(stage_dir / "ops" / "gateway_service.json", ops["gateway_service"])
    _write_json(stage_dir / "ops" / "live_repo.json", ops["live_repo"])

    for log_name, payload in tails.items():
        _write_json(stage_dir / "tails" / f"{log_name}.json", payload)

    _write_json(stage_dir / "summary.json", summary)
    _write_text(
        stage_dir / "exported_at.txt",
        f"{exported_at_iso}\nexporter_version={EXPORTER_VERSION}\nrelease_name={release_name}\n",
    )

    final_release_dir = releases_dir / release_name
    publish_dir = releases_dir / f".publish-{release_name}-{os.getpid()}"
    if final_release_dir.exists():
        raise ExporterError(f"Refusing to overwrite existing release: {final_release_dir}")
    if publish_dir.exists():
        shutil.rmtree(publish_dir)
    _copy_tree(stage_dir, publish_dir)
    publish_dir.rename(final_release_dir)
    shutil.rmtree(stage_dir)
    _fsync_dir(releases_dir)

    _remove_if_exists(output_root / "current")
    for export_name in ("SOUL.md", "BOOT.md", "skills_snapshot.json"):
        target_path = final_release_dir / export_name
        root_path = output_root / export_name
        if target_path.exists():
            _copy_file(target_path, root_path)
        else:
            _remove_if_exists(root_path)
    _copy_file(final_release_dir / "exported_at.txt", output_root / "exported_at.txt")
    _write_text(output_root / "current_release.txt", f"{release_name}\n")
    _copy_file(final_release_dir / "summary.json", output_root / "summary.json")

    if keep_releases > 0:
        releases = sorted(
            [path for path in releases_dir.iterdir() if path.is_dir() and not path.name.startswith(".staging-")],
            key=lambda path: path.name,
            reverse=True,
        )
        keep_paths = [final_release_dir]
        for candidate in releases:
            if candidate == final_release_dir:
                continue
            if len(keep_paths) >= keep_releases:
                break
            keep_paths.append(candidate)
        keep_set = set(keep_paths)
        for stale in releases:
            if stale in keep_set:
                continue
            shutil.rmtree(stale, ignore_errors=True)
        _fsync_dir(releases_dir)

    _write_json(hash_cache.path, hash_cache.data)
    _enforce_export_tree_contract(output_root)
    _verify_export_tree_contract(output_root)

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export safe Hermes runtime metadata for read-only audit consumption.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Stable export root (default: /var/lib/hermes-export)")
    parser.add_argument("--keep-releases", type=int, default=DEFAULT_KEEP_RELEASES, help="How many immutable releases to retain under output_root/releases")
    parser.add_argument("--log-tail-lines", type=int, default=DEFAULT_LOG_TAIL_LINES, help="Tail line count for low-risk monitor logs")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = export_runtime_snapshot(
            output_root=Path(args.output_root),
            keep_releases=max(1, int(args.keep_releases)),
            log_tail_lines=max(1, int(args.log_tail_lines)),
        )
    except Exception as exc:
        print(f"runtime export failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({
        "ok": True,
        "exported_at": summary["metadata"]["exported_at"],
        "release_name": summary["metadata"]["release_name"],
        "output_root": args.output_root,
        "health_flags": summary.get("health_flags", []),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
