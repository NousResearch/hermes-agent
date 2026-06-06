"""Lightweight skill/cron reliability checks.

This module powers `hermes doctor skill|cron` and is intentionally conservative:
static checks are local-only, smoke probes run only when explicitly requested and
only for declarations marked safe by the skill/job owner.
"""
from __future__ import annotations

import importlib
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

from hermes_constants import get_hermes_home


KNOWN_TOOLSETS = {
    "web", "browser", "terminal", "file", "code_execution", "vision",
    "image_gen", "tts", "skills", "memory", "session_search", "delegation",
    "cronjob", "clarify", "moa", "homeassistant", "todo",
}


class ReliabilityUsageError(ValueError):
    pass


def _expand_path(value: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value))))


def _status(errors: list[dict], warnings: list[dict]) -> str:
    return "fail" if errors else "ok"


def _frontmatter(text: str) -> dict[str, Any]:
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    try:
        data = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as exc:
        return {"_parse_error": str(exc)}
    return data if isinstance(data, dict) else {}


def _iter_skill_files() -> Iterable[Path]:
    home = get_hermes_home()
    roots = [home / "skills", Path(__file__).resolve().parents[1] / "skills"]
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("SKILL.md"):
            rp = path.resolve()
            if rp not in seen:
                seen.add(rp)
                yield path


def find_skill(skill_name: str) -> Optional[Path]:
    wanted = skill_name.strip().lower()
    for path in _iter_skill_files():
        data = _frontmatter(path.read_text(encoding="utf-8", errors="replace"))
        name = str(data.get("name") or path.parent.name).strip().lower()
        if name == wanted or path.parent.name.lower() == wanted:
            return path
    return None


def _listify(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _prerequisites(meta: dict[str, Any]) -> dict[str, Any]:
    prereq = meta.get("prerequisites") or {}
    return prereq if isinstance(prereq, dict) else {}


def _configured_mcp_servers() -> set[str]:
    """Return MCP server names configured for this Hermes install."""
    try:
        from hermes_cli.config import load_config

        servers = (load_config() or {}).get("mcp_servers") or {}
    except Exception:  # noqa: BLE001 - diagnostic checks should degrade cleanly
        servers = {}
    if not isinstance(servers, dict):
        return set()
    return {str(name) for name in servers.keys()}


def _check_mcp_configured(name: str) -> bool:
    return name in _configured_mcp_servers()


def _check_static(prereq: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    errors: list[dict] = []
    warnings: list[dict] = []

    for name in _listify(prereq.get("env_vars")):
        name = str(name)
        if not os.getenv(name):
            errors.append({"type": "env_var", "name": name, "message": "missing"})

    for name in _listify(prereq.get("commands")):
        name = str(name)
        if not shutil.which(name):
            errors.append({"type": "command", "name": name, "message": "missing"})

    for raw in _listify(prereq.get("files")):
        path = _expand_path(str(raw))
        if not path.is_file():
            errors.append({"type": "file", "name": str(raw), "message": "missing"})

    for raw in _listify(prereq.get("directories")):
        path = _expand_path(str(raw))
        if not path.is_dir():
            errors.append({"type": "directory", "name": str(raw), "message": "missing"})

    for module in _listify(prereq.get("python_imports")):
        module = str(module)
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 - diagnostic surface
            errors.append({"type": "python_import", "name": module, "message": str(exc)})

    for server in _listify(prereq.get("mcp_servers")):
        name = str(server)
        if not _check_mcp_configured(name):
            errors.append({"type": "mcp_server", "name": name, "message": "not_configured"})

    if not prereq:
        warnings.append({"type": "dependencies", "message": "no_declared_dependencies"})

    return errors, warnings


def _normalize_smoke(smoke_decl: Any) -> tuple[bool, int, list[dict[str, Any]]]:
    if not isinstance(smoke_decl, dict):
        return False, 20, []
    safe = bool(smoke_decl.get("safe"))
    timeout = int(smoke_decl.get("timeout_seconds") or 20)
    probes = smoke_decl.get("probes")
    if probes is None and smoke_decl.get("command"):
        probes = [{"type": "command", "command": smoke_decl["command"]}]
    return safe, timeout, [p for p in _listify(probes) if isinstance(p, dict)]


def _run_probe(probe: dict[str, Any], timeout: int, safe: bool) -> dict[str, Any]:
    ptype = str(probe.get("type") or "").strip()
    label = probe.get("name") or probe.get("path") or probe.get("module") or probe.get("command") or ptype
    base = {"type": ptype, "name": str(label)}

    try:
        if ptype == "env-present":
            ok = bool(os.getenv(str(probe.get("name") or "")))
            return {**base, "status": "ok" if ok else "fail", "message": "present" if ok else "missing"}
        if ptype == "command-exists":
            ok = bool(shutil.which(str(probe.get("name") or "")))
            return {**base, "status": "ok" if ok else "fail", "message": "present" if ok else "missing"}
        if ptype == "file-exists":
            ok = _expand_path(str(probe.get("path") or "")).is_file()
            return {**base, "status": "ok" if ok else "fail", "message": "present" if ok else "missing"}
        if ptype == "directory-exists":
            ok = _expand_path(str(probe.get("path") or "")).is_dir()
            return {**base, "status": "ok" if ok else "fail", "message": "present" if ok else "missing"}
        if ptype == "python-import":
            module = str(probe.get("module") or probe.get("name") or "")
            importlib.import_module(module)
            return {**base, "status": "ok", "message": "imported"}
        if ptype == "mcp-configured":
            name = str(probe.get("name") or "")
            ok = _check_mcp_configured(name)
            return {**base, "status": "ok" if ok else "fail", "message": "configured" if ok else "not_configured"}
        if ptype == "command":
            if not safe:
                return {**base, "status": "warn", "message": "unsafe smoke skipped"}
            command = str(probe.get("command") or "")
            completed = subprocess.run(
                shlex.split(command),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return {
                **base,
                "status": "ok" if completed.returncode == 0 else "fail",
                "message": f"exit {completed.returncode}",
                "stdout": (completed.stdout or "")[-500:],
                "stderr": (completed.stderr or "")[-500:],
            }
    except subprocess.TimeoutExpired:
        return {**base, "status": "fail", "message": f"timeout after {timeout}s"}
    except Exception as exc:  # noqa: BLE001
        return {**base, "status": "fail", "message": str(exc)}

    return {**base, "status": "warn", "message": "unknown probe type"}


def _smoke(prereq: dict[str, Any]) -> tuple[list[dict], list[dict], list[dict]]:
    safe, timeout, probes = _normalize_smoke(prereq.get("smoke"))
    if not probes:
        return [], [], [{"type": "smoke", "message": "no_safe_smoke_declared"}]
    results = [_run_probe(p, timeout, safe) for p in probes]
    errors = [{"type": "smoke", "name": r["name"], "message": r.get("message", "failed")} for r in results if r["status"] == "fail"]
    warnings = [{"type": "smoke", "name": r["name"], "message": r.get("message", "warning")} for r in results if r["status"] == "warn"]
    return results, errors, warnings


def doctor_skill(skill_name: str, *, smoke: bool = False) -> Dict[str, Any]:
    path = find_skill(skill_name)
    if path is None:
        return {
            "kind": "skill",
            "skill": skill_name,
            "status": "fail",
            "errors": [{"type": "skill", "name": skill_name, "message": "not_found"}],
            "warnings": [],
            "smoke": [],
        }

    meta = _frontmatter(path.read_text(encoding="utf-8", errors="replace"))
    skill = str(meta.get("name") or path.parent.name)
    prereq = _prerequisites(meta)
    errors, warnings = _check_static(prereq)
    smoke_results: list[dict] = []
    if smoke:
        smoke_results, smoke_errors, smoke_warnings = _smoke(prereq)
        errors.extend(smoke_errors)
        warnings.extend(smoke_warnings)

    return {
        "kind": "skill",
        "skill": skill,
        "path": str(path),
        "status": _status(errors, warnings),
        "errors": errors,
        "warnings": warnings,
        "smoke": smoke_results,
    }


def doctor_all_skills(*, smoke: bool = False) -> Dict[str, Any]:
    results = []
    seen: set[str] = set()
    for path in _iter_skill_files():
        meta = _frontmatter(path.read_text(encoding="utf-8", errors="replace"))
        name = str(meta.get("name") or path.parent.name)
        if name in seen:
            continue
        seen.add(name)
        results.append(doctor_skill(name, smoke=smoke))
    errors = [{"type": "skill", "name": r["skill"], "message": "dependency_failed"} for r in results if r["status"] == "fail"]
    return {
        "kind": "skill_collection",
        "status": _status(errors, []),
        "errors": errors,
        "warnings": [],
        "results": results,
    }


def _resolve_script(script: str) -> Path:
    raw = str(script)
    path = _expand_path(raw)
    if path.is_absolute():
        return path
    return get_hermes_home() / "scripts" / raw


def _find_job(job_id_or_name: str) -> Optional[dict[str, Any]]:
    from cron.jobs import load_jobs
    wanted = str(job_id_or_name)
    for job in load_jobs():
        if str(job.get("id")) == wanted or str(job.get("name")) == wanted:
            return job
    return None


_ABSOLUTE_PROMPT_PATH_RE = re.compile(r"(?<![\w.-])(/[^\s\"'`;|&<>]+)")


def _prompt_paths(prompt: str) -> list[str]:
    """Return absolute script-like paths embedded in a cron prompt command.

    Prompt-only cron jobs often tell the agent to run an exact shell command.
    This lightweight static check catches missing local wrapper/scripts without
    trying to execute the command.
    """
    paths: list[str] = []
    for match in _ABSOLUTE_PROMPT_PATH_RE.finditer(prompt or ""):
        raw = match.group(1).rstrip(")],.")
        if raw.startswith(("/tmp/", "/var/", "/Users/", "/opt/", "/usr/local/", "/bin/", "/usr/bin/")):
            paths.append(raw)
    return list(dict.fromkeys(paths))


def doctor_cron(job_id_or_name: str, *, smoke: bool = False) -> Dict[str, Any]:
    job = _find_job(job_id_or_name)
    if job is None:
        return {
            "kind": "cron",
            "job": job_id_or_name,
            "status": "fail",
            "errors": [{"type": "cron", "name": str(job_id_or_name), "message": "not_found"}],
            "warnings": [],
            "skill_results": [],
            "smoke": [],
        }

    errors: list[dict] = []
    warnings: list[dict] = []
    skill_results: list[dict] = []
    smoke_results: list[dict] = []

    script = job.get("script")
    if script:
        script_path = _resolve_script(str(script))
        if not script_path.is_file():
            errors.append({"type": "script", "name": str(script), "message": "missing"})

    for raw_path in _prompt_paths(str(job.get("prompt") or "")):
        path = Path(raw_path)
        if not path.exists():
            errors.append({"type": "prompt_path", "name": raw_path, "message": "missing"})

    for toolset in job.get("enabled_toolsets") or []:
        if str(toolset) not in KNOWN_TOOLSETS:
            errors.append({"type": "toolset", "name": str(toolset), "message": "unknown"})

    for skill in job.get("skills") or []:
        res = doctor_skill(str(skill), smoke=smoke)
        skill_results.append(res)
        if res["status"] == "fail":
            errors.append({"type": "skill", "name": str(skill), "message": "dependency_failed"})

    if smoke and isinstance(job.get("smoke"), dict):
        smoke_results, smoke_errors, smoke_warnings = _smoke({"smoke": job["smoke"]})
        errors.extend(smoke_errors)
        warnings.extend(smoke_warnings)

    if not script and not job.get("skills") and not _prompt_paths(str(job.get("prompt") or "")) and not job.get("smoke"):
        warnings.append({"type": "dependencies", "message": "no_script_or_skills_declared"})

    return {
        "kind": "cron",
        "job": job.get("name") or job.get("id"),
        "job_id": job.get("id"),
        "status": _status(errors, warnings),
        "errors": errors,
        "warnings": warnings,
        "skill_results": skill_results,
        "smoke": smoke_results,
    }


def doctor_all_crons(*, smoke: bool = False) -> Dict[str, Any]:
    from cron.jobs import load_jobs
    results = [doctor_cron(str(job.get("id") or job.get("name")), smoke=smoke) for job in load_jobs()]
    errors = [{"type": "cron", "name": str(r.get("job")), "message": "dependency_failed"} for r in results if r["status"] == "fail"]
    return {
        "kind": "cron_collection",
        "status": _status(errors, []),
        "errors": errors,
        "warnings": [],
        "results": results,
    }


def _print_human(result: dict[str, Any]) -> None:
    icon = "✓" if result.get("status") == "ok" else "✗"
    subject = result.get("skill") or result.get("job") or result.get("kind")
    print(f"{icon} {result.get('kind')} {subject}: {result.get('status')}")
    for err in result.get("errors", []):
        print(f"  ERROR {err.get('type')} {err.get('name', '')}: {err.get('message')}")
    for warn in result.get("warnings", []):
        print(f"  WARN {warn.get('type')}: {warn.get('message')}")
    for item in result.get("results", []):
        _print_human(item)
    for skill in result.get("skill_results", []):
        _print_human(skill)


def doctor_command(args: Any) -> int:
    target = getattr(args, "doctor_target", None)
    name = getattr(args, "doctor_name", None)
    all_targets = bool(getattr(args, "all", False))
    smoke = bool(getattr(args, "smoke", False))
    as_json = bool(getattr(args, "json", False))

    if not all_targets and not name:
        raise ReliabilityUsageError(f"doctor {target} requires a name or --all")

    if target == "skill":
        result = doctor_all_skills(smoke=smoke) if all_targets else doctor_skill(name, smoke=smoke)
    elif target == "cron":
        result = doctor_all_crons(smoke=smoke) if all_targets else doctor_cron(name, smoke=smoke)
    else:
        raise ReliabilityUsageError("doctor target must be 'skill' or 'cron'")

    if as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _print_human(result)
    return 1 if result.get("status") == "fail" else 0
