"""Static reliability diagnostics and safe smoke declarations.

This module is intentionally side-effect light: it validates declarations and
returns structured results, but it does not print CLI output or mutate cron
state.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import sys
import json
import os
from dataclasses import asdict, dataclass
from importlib.machinery import PathFinder
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, cast

from cron.jobs import AmbiguousJobReference, list_jobs, resolve_job_ref
from hermes_cli.config import get_hermes_home
from tools.skills_tool import (
    _collect_prerequisite_values,
    _find_all_skills,
    _get_required_environment_variables,
    skill_view,
)


SMOKE_VERSION = 1
MAX_PROBES = 32
MAX_TEXT_LENGTH = 512
ALLOWED_PROBE_TYPES = {
    "command-exists",
    "env-present",
    "file-exists",
    "directory-exists",
    "python-import",
    "mcp-configured",
}
ALLOWED_ROOTS = {"hermes_home", "scripts_dir", "workdir", "skill_dir"}

_COMMON_FIELDS = {"type"}
_FIELDS_BY_TYPE = {
    "command-exists": _COMMON_FIELDS | {"name"},
    "env-present": _COMMON_FIELDS | {"name"},
    "file-exists": _COMMON_FIELDS | {"root", "path"},
    "directory-exists": _COMMON_FIELDS | {"root", "path"},
    "python-import": _COMMON_FIELDS | {"module"},
    "mcp-configured": _COMMON_FIELDS | {"server"},
}
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_INVISIBLE_CODEPOINTS = {
    "\u200b",
    "\u200c",
    "\u200d",
    "\u2060",
    "\ufeff",
    "\u2061",
    "\u2062",
    "\u2063",
    "\u2064",
    "\u2066",
    "\u2067",
    "\u2068",
    "\u2069",
}


class SmokeValidationError(ValueError):
    """Raised when a smoke declaration fails the safe schema."""


@dataclass(frozen=True)
class DiagnosticResult:
    subject_type: str
    subject: str
    probe_type: str
    target: str
    status: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_smoke_spec(value: object) -> dict[str, Any]:
    """Return a fresh canonical smoke declaration or raise ``SmokeValidationError``."""
    if not isinstance(value, dict):
        raise SmokeValidationError("invalid_smoke_spec")
    value_map = cast(dict[str, Any], value)
    if any(not isinstance(key, str) for key in value_map):
        raise SmokeValidationError("invalid_smoke_field")
    unknown = set(value_map) - {"version", "probes"}
    if unknown:
        raise SmokeValidationError(f"unknown_field:{sorted(unknown)[0]}")
    if value_map.get("version") != SMOKE_VERSION:
        raise SmokeValidationError("invalid_version")
    probes = value_map.get("probes")
    if not isinstance(probes, list):
        raise SmokeValidationError("invalid_probes")
    if len(probes) > MAX_PROBES:
        raise SmokeValidationError("too_many_probes")
    return {
        "version": SMOKE_VERSION,
        "probes": [_validate_probe(probe) for probe in probes],
    }


def resolve_probe_path(
    root_name: str,
    relative_path: str,
    roots: Mapping[str, Path | None],
) -> Path | None:
    """Resolve a smoke path and reject traversal/symlink escapes."""
    root = roots.get(root_name)
    if root is None:
        return None
    root_path = Path(root).expanduser().resolve()
    candidate = (root_path / relative_path).resolve()
    try:
        candidate.relative_to(root_path)
    except ValueError as exc:
        raise SmokeValidationError("path_escape") from exc
    return candidate


def evaluate_static_probes(
    smoke: dict,
    *,
    subject_type: str,
    subject: str,
    roots: Mapping[str, Path | None],
    env: Mapping[str, str] | None = None,
    mcp_servers: Mapping[str, object] | None = None,
) -> list[DiagnosticResult]:
    """Evaluate non-executing smoke probes."""
    environment = env if env is not None else os.environ
    results: list[DiagnosticResult] = []
    for probe in smoke.get("probes", []):
        probe_type = probe.get("type", "")
        if probe_type == "env-present":
            name = str(probe["name"])
            results.append(
                _result(
                    subject_type,
                    subject,
                    probe_type,
                    name,
                    "pass" if name in environment else "fail",
                    "env_present" if name in environment else "env_missing",
                )
            )
        elif probe_type == "command-exists":
            name = str(probe["name"])
            found = shutil.which(name) is not None
            results.append(
                _result(
                    subject_type,
                    subject,
                    probe_type,
                    name,
                    "pass" if found else "fail",
                    "command_found" if found else "command_missing",
                )
            )
        elif probe_type in {"file-exists", "directory-exists"}:
            results.append(_evaluate_path_probe(probe, subject_type, subject, roots))
        elif probe_type == "python-import":
            module = str(probe["module"])
            found = _module_is_findable(module)
            results.append(
                _result(
                    subject_type,
                    subject,
                    probe_type,
                    module,
                    "pass" if found else "fail",
                    "python_import_found" if found else "python_import_missing",
                )
            )
        elif probe_type == "mcp-configured":
            server = str(probe["server"])
            servers = mcp_servers if mcp_servers is not None else _load_mcp_servers()
            found = server in servers
            results.append(
                _result(
                    subject_type,
                    subject,
                    probe_type,
                    server,
                    "pass" if found else "fail",
                    "mcp_configured" if found else "mcp_missing",
                )
            )
    return results


def diagnose_skill(
    name: str,
    *,
    env: Mapping[str, str] | None = None,
) -> list[DiagnosticResult]:
    """Run static diagnostics for a discovered skill."""
    discovered = {str(skill.get("name")) for skill in _find_all_skills()}
    if name not in discovered:
        return [_result("skill", name, "skill", name, "fail", "skill_not_found")]

    frontmatter = _load_skill_frontmatter(name)
    if frontmatter is None:
        return [_result("skill", name, "skill", name, "fail", "skill_unreadable")]

    try:
        legacy_env_vars, commands = _collect_prerequisite_values(frontmatter)
        required_env = [
            item
            for item in _get_required_environment_variables(
                frontmatter, legacy_env_vars
            )
            if not item.get("optional")
        ]
    except Exception:
        return [
            _result(
                "skill",
                name,
                "skill-metadata",
                name,
                "fail",
                "invalid_skill_metadata",
            )
        ]
    probes: list[dict[str, Any]] = []
    probes.extend(
        {"type": "env-present", "name": item["name"]} for item in required_env
    )
    probes.extend({"type": "command-exists", "name": command} for command in commands)
    if not probes:
        return [_result("skill", name, "skill", name, "pass", "skill_found")]
    smoke = validate_smoke_spec({"version": SMOKE_VERSION, "probes": probes})
    return evaluate_static_probes(
        smoke,
        subject_type="skill",
        subject=name,
        roots={},
        env=env,
    )


def diagnose_all_skills() -> dict[str, list[DiagnosticResult]]:
    """Diagnose every currently discovered skill."""
    return {
        str(skill.get("name")): diagnose_skill(str(skill.get("name")))
        for skill in _find_all_skills()
    }


def diagnose_cron(ref: str) -> list[DiagnosticResult]:
    """Run static diagnostics for a cron job reference."""
    try:
        job = resolve_job_ref(ref)
    except AmbiguousJobReference:
        return [_result("cron", ref, "cron", ref, "fail", "cron_ref_ambiguous")]
    if not job:
        return [_result("cron", ref, "cron", ref, "fail", "cron_not_found")]

    subject = str(job.get("id") or ref)
    results: list[DiagnosticResult] = []
    prompt = str(job.get("prompt") or "").strip()
    skills = [str(skill) for skill in (job.get("skills") or []) if str(skill).strip()]
    script = job.get("script")
    has_script = isinstance(script, str) and bool(script.strip())
    schedule = str(job.get("schedule_display") or "")
    if prompt:
        results.append(
            _result(
                "cron", subject, "cron-prompt", "prompt", "pass", "cron_prompt_present"
            )
        )
    elif job.get("no_agent") and has_script:
        results.append(
            _result(
                "cron",
                subject,
                "cron-prompt",
                "prompt",
                "skipped",
                "cron_prompt_not_required",
            )
        )
    elif skills:
        results.append(
            _result(
                "cron",
                subject,
                "cron-prompt",
                "prompt",
                "skipped",
                "cron_prompt_from_skill",
            )
        )
    else:
        results.append(
            _result(
                "cron", subject, "cron-prompt", "prompt", "fail", "cron_prompt_missing"
            )
        )
    for skill in skills:
        results.append(
            _result("cron", subject, "cron-skill", skill, "pass", "cron_skill_attached")
        )
        results.extend(diagnose_skill(skill))
    results.append(
        _result(
            "cron",
            subject,
            "cron-schedule",
            schedule or "schedule",
            "pass" if job.get("schedule") else "fail",
            "cron_schedule_present" if job.get("schedule") else "cron_schedule_missing",
        )
    )

    if has_script:
        assert isinstance(script, str)
        results.append(_diagnose_job_script(script, subject))

    if "smoke" in job and job.get("smoke") is not None:
        try:
            smoke = validate_smoke_spec(job["smoke"])
        except SmokeValidationError:
            results.append(
                _result(
                    "cron", subject, "smoke-schema", "smoke", "fail", "invalid_smoke"
                )
            )
        else:
            results.extend(
                evaluate_static_probes(
                    smoke,
                    subject_type="cron",
                    subject=subject,
                    roots=_roots_for_job(job),
                )
            )
    return results


def render_diagnostic_json(
    results: list[DiagnosticResult] | Mapping[str, list[DiagnosticResult]],
) -> str:
    """Render structured diagnostics without raw output fields."""
    if isinstance(results, Mapping):
        result_map = cast(Mapping[str, list[DiagnosticResult]], results)
        payload = {}
        used_keys: set[str] = set()
        for key, value in result_map.items():
            safe_key = _safe_mapping_key(key, used_keys)
            used_keys.add(safe_key)
            payload[safe_key] = [item.to_dict() for item in value]
    else:
        payload = [item.to_dict() for item in results]
    return json.dumps(payload, sort_keys=True)


def render_diagnostic_text(
    results: list[DiagnosticResult] | Mapping[str, list[DiagnosticResult]],
) -> str:
    """Render compact human diagnostics without secret values."""
    flat: list[DiagnosticResult] = []
    if isinstance(results, Mapping):
        result_map = cast(Mapping[str, list[DiagnosticResult]], results)
        for items in result_map.values():
            flat.extend(items)
    else:
        flat = list(results)
    lines = []
    for item in flat:
        lines.append(
            f"{item.status.upper()} {item.subject_type}:{item.subject} "
            f"{item.probe_type}:{item.target} {item.reason}"
        )
    return "\n".join(lines)


def diagnose_all_crons() -> dict[str, list[DiagnosticResult]]:
    """Diagnose all cron jobs, including disabled jobs."""
    return {
        str(job.get("id")): diagnose_cron(str(job.get("id")))
        for job in list_jobs(include_disabled=True)
    }


def _evaluate_path_probe(
    probe: Mapping[str, Any],
    subject_type: str,
    subject: str,
    roots: Mapping[str, Path | None],
) -> DiagnosticResult:
    root = str(probe["root"])
    target = f"{root}:{probe['path']}"
    try:
        path = resolve_probe_path(root, str(probe["path"]), roots)
    except SmokeValidationError as exc:
        return _result(
            subject_type,
            subject,
            str(probe["type"]),
            target,
            "fail",
            str(exc),
        )
    if path is None:
        return _result(
            subject_type,
            subject,
            str(probe["type"]),
            target,
            "skipped",
            "root_unavailable",
        )
    if probe["type"] == "file-exists":
        exists = path.is_file()
        return _result(
            subject_type,
            subject,
            "file-exists",
            target,
            "pass" if exists else "fail",
            "file_exists" if exists else "file_missing",
        )
    exists = path.is_dir()
    return _result(
        subject_type,
        subject,
        "directory-exists",
        target,
        "pass" if exists else "fail",
        "directory_exists" if exists else "directory_missing",
    )


def _diagnose_job_script(script: str, subject: str) -> DiagnosticResult:
    scripts_dir = (get_hermes_home() / "scripts").resolve()
    raw = Path(script).expanduser()
    candidate = raw.resolve() if raw.is_absolute() else (scripts_dir / raw).resolve()
    try:
        candidate.relative_to(scripts_dir)
    except ValueError:
        return _result(
            "cron", subject, "cron-script", script, "fail", "script_path_escape"
        )
    exists = candidate.is_file()
    return _result(
        "cron",
        subject,
        "cron-script",
        script,
        "pass" if exists else "fail",
        "script_exists" if exists else "script_missing",
    )


def _load_skill_frontmatter(name: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(skill_view(name, preprocess=False, metadata_only=True))
    except Exception:
        return None
    if not isinstance(payload, dict) or not payload.get("success"):
        return None
    frontmatter = payload.get("frontmatter")
    return frontmatter if isinstance(frontmatter, dict) else None


def _roots_for_job(job: Mapping[str, Any]) -> dict[str, Path | None]:
    hermes_home = get_hermes_home()
    workdir = job.get("workdir")
    return {
        "hermes_home": hermes_home,
        "scripts_dir": hermes_home / "scripts",
        "workdir": Path(workdir).expanduser()
        if isinstance(workdir, str) and workdir
        else Path.cwd(),
        "skill_dir": None,
    }


def _module_is_findable(module: str) -> bool:
    search_paths: list[str] | None = list(sys.path)
    spec = None
    for index, part in enumerate(module.split(".")):
        fullname = ".".join(module.split(".")[: index + 1])
        spec = PathFinder.find_spec(fullname, search_paths)
        if spec is None:
            return False
        if index < len(module.split(".")) - 1:
            locations = spec.submodule_search_locations
            if locations is None:
                return False
            search_paths = list(locations)
    return spec is not None


def _load_mcp_servers() -> Mapping[str, object]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
    except Exception:
        return {}
    if not isinstance(cfg, dict):
        return {}
    mcp = cfg.get("mcp")
    if isinstance(mcp, dict):
        servers = mcp.get("servers")
        if isinstance(servers, dict):
            return servers
    servers = cfg.get("mcp_servers")
    return servers if isinstance(servers, dict) else {}


def _result(
    subject_type: str,
    subject: str,
    probe_type: str,
    target: str,
    status: str,
    reason: str,
) -> DiagnosticResult:
    return DiagnosticResult(
        subject_type=_safe_diagnostic_text(subject_type),
        subject=_safe_diagnostic_text(subject),
        probe_type=_safe_diagnostic_text(probe_type),
        target=_safe_diagnostic_text(target),
        status=_safe_diagnostic_text(status),
        reason=_safe_diagnostic_text(reason),
    )


def _safe_diagnostic_text(value: object) -> str:
    text = str(value)
    safe = "".join(
        "�" if ord(ch) < 32 or ord(ch) == 127 or ch in _INVISIBLE_CODEPOINTS else ch
        for ch in text
    )
    return safe[:MAX_TEXT_LENGTH]


def _safe_mapping_key(value: object, used: set[str]) -> str:
    raw = str(value)
    safe = _safe_diagnostic_text(raw)
    if safe == raw and safe not in used:
        return safe
    nonce = 0
    while True:
        digest_input = f"{raw}\0{nonce}".encode("utf-8", errors="replace")
        suffix = f"#{hashlib.sha256(digest_input).hexdigest()[:12]}"
        candidate = f"{safe[: MAX_TEXT_LENGTH - len(suffix)]}{suffix}"
        if candidate not in used:
            return candidate
        nonce += 1


def _validate_probe(probe: object) -> dict[str, Any]:
    if not isinstance(probe, dict):
        raise SmokeValidationError("invalid_probe")
    probe_map = cast(dict[str, Any], probe)
    if any(not isinstance(key, str) for key in probe_map):
        raise SmokeValidationError("invalid_probe_field")
    probe_type = probe_map.get("type")
    if not isinstance(probe_type, str) or probe_type not in ALLOWED_PROBE_TYPES:
        raise SmokeValidationError("unknown_probe_type")
    allowed = _FIELDS_BY_TYPE[str(probe_type)]
    unknown = set(probe_map) - allowed
    if unknown:
        raise SmokeValidationError(f"unknown_probe_field:{sorted(unknown)[0]}")
    if probe_type in {"command-exists", "env-present"}:
        return {
            "type": probe_type,
            "name": _validate_identifier(probe_map.get("name"), "name"),
        }
    if probe_type in {"file-exists", "directory-exists"}:
        return {
            "type": probe_type,
            "root": _validate_root(probe_map.get("root")),
            "path": _validate_relative_path(probe_map.get("path")),
        }
    if probe_type == "python-import":
        module = _validate_text(probe_map.get("module"), "module")
        if not _MODULE_RE.fullmatch(module):
            raise SmokeValidationError("invalid_module")
        return {"type": probe_type, "module": module}
    if probe_type == "mcp-configured":
        return {
            "type": probe_type,
            "server": _validate_identifier(probe_map.get("server"), "server"),
        }
    raise SmokeValidationError("unknown_probe_type")


def _validate_root(value: object) -> str:
    root = _validate_text(value, "root")
    if root not in ALLOWED_ROOTS:
        raise SmokeValidationError("invalid_root")
    return root


def _validate_identifier(value: object, field: str) -> str:
    text = _validate_text(value, field)
    if not _IDENT_RE.fullmatch(text):
        raise SmokeValidationError(f"invalid_{field}")
    return text


def _validate_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_TEXT_LENGTH:
        raise SmokeValidationError(f"invalid_{field}")
    if any(
        ord(ch) < 32 or ord(ch) == 127 or ch in _INVISIBLE_CODEPOINTS for ch in value
    ):
        raise SmokeValidationError("invalid_text")
    return value


def _validate_relative_path(value: object) -> str:
    text = _validate_text(value, "path").replace("\\", "/")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or ":" in path.parts[0]
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise SmokeValidationError("invalid_path")
    return path.as_posix()
