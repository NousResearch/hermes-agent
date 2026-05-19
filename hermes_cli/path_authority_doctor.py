"""Read-only Kanban path-authority diagnostics for Hermes deployments."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SEVERITY_RANK = {"OK": 0, "P2": 1, "P1": 2, "P0": 3}
UNIT_NAMES = {
    "gateway": "hermes-gateway.service",
    "webui": "hermes-webui.service",
    "watchdog": "hermes-gateway-watchdog.service",
}
PATH_ENV_KEYS = {
    "HERMES_KANBAN_AGENT_DIR",
    "HERMES_AGENT_DIR",
    "HERMES_AGENT_REPO",
    "HERMES_HOME",
    "HERMES_WEBUI_DIR",
    "HERMES_WEBUI_REPO",
    "OBSIDIAN_VAULT_PATH",
}
SECRET_KEY_FRAGMENTS = (
    "TOKEN",
    "SECRET",
    "KEY",
    "PASSWORD",
    "PASS",
    "CREDENTIAL",
    "COOKIE",
    "AUTH",
)
AUTHORITY_DIRTY_PATHS = (
    "hermes_cli/path_authority_doctor.py",
    "hermes_cli/doctor.py",
    "hermes_cli/main.py",
    "hermes_cli/gateway.py",
    "gateway/status.py",
    "gateway/run.py",
    "gateway/watchdog_check.py",
)
SECRET_ARG_NAMES = {
    "--api-key",
    "--apikey",
    "--authorization",
    "--auth",
    "--bearer-token",
    "--cookie",
    "--key",
    "--password",
    "--secret",
    "--token",
}
SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(api[_-]?key|authorization|auth|cookie|password|passwd|secret|token)="
)
GRAPH_FRESHNESS_RELATIVE_PATHS = (
    "Imports/Manifests/import-ledger.jsonl",
    "Imports/Manifests/redaction-policy.md",
    "AgentOps/Runbooks/HermesVault-Memory-ETL-Control-Plane.md",
    "AgentOps/Runbooks/Memory-Promotion-Workflow.md",
    "AgentOps/Runbooks/Agent-Startup-Query-Gate.md",
    "AgentOps/Control/HermesVault-Memory-ETL.freeze",
    "Memory/Contradictions/contradiction-register.md",
    "Graphify/graph-manifest.md",
    "Graphify/graph-manifest.json",
)


@dataclass
class ParsedPathsEnv:
    path: Path
    exists: bool
    values: dict[str, str]
    redacted: dict[str, str]

    def to_report(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "exists": self.exists,
            "values": dict(self.redacted),
        }


@dataclass
class EffectiveUnit:
    exec_start: list[str] = field(default_factory=list)
    working_directory: str | None = None
    environment: dict[str, str] = field(default_factory=dict)

    def to_report(self) -> dict[str, Any]:
        return {
            "exec_start": [_redact_command(command) for command in self.exec_start],
            "working_directory": self.working_directory,
            "environment": _redact_mapping(self.environment),
        }


@dataclass
class UnitReport:
    role: str
    name: str
    path: Path
    exists: bool
    dropins: list[Path]
    effective: EffectiveUnit

    def to_report(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "name": self.name,
            "path": str(self.path),
            "exists": self.exists,
            "dropins": [str(path) for path in self.dropins],
            "effective": self.effective.to_report(),
        }


@dataclass
class CheckResult:
    id: str
    severity: str
    status: str
    message: str
    detail: str | None = None

    def to_report(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "severity": self.severity,
            "status": self.status,
            "message": self.message,
        }
        if self.detail:
            payload["detail"] = self.detail
        return payload


@dataclass
class PathAuthorityReport:
    expected_agent_dir: str | None
    paths_env: ParsedPathsEnv
    units: dict[str, UnitReport]
    checks: list[CheckResult]
    runtime_import_authority: dict[str, Any] | None = None

    @property
    def overall_severity(self) -> str:
        severity = "OK"
        for check in self.failed_checks():
            if SEVERITY_RANK[check.severity] > SEVERITY_RANK[severity]:
                severity = check.severity
        return severity

    def failed_checks(self) -> list[CheckResult]:
        return [check for check in self.checks if check.status != "ok"]

    def exit_code(self, *, strict: bool) -> int:
        if strict:
            return 1 if self.failed_checks() else 0
        return 1 if any(check.severity == "P0" for check in self.failed_checks()) else 0

    def to_report(self) -> dict[str, Any]:
        return {
            "status": self.overall_severity,
            "expected_agent_dir": self.expected_agent_dir,
            "paths_env": self.paths_env.to_report(),
            "units": {role: unit.to_report() for role, unit in self.units.items()},
            "runtime_import_authority": self.runtime_import_authority,
            "checks": [check.to_report() for check in self.checks],
        }


def parse_paths_env(path: Path | None = None) -> ParsedPathsEnv:
    path = path or Path.home() / ".config" / "agent-env" / "paths.env"
    if not path.exists():
        return ParsedPathsEnv(path=path, exists=False, values={}, redacted={})

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_assignment(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        values[key] = value

    return ParsedPathsEnv(
        path=path,
        exists=True,
        values=values,
        redacted=_redact_mapping(values),
    )


def classify_dirty_repo(status_lines: list[str]) -> CheckResult:
    dirty_paths = [_status_path(line) for line in status_lines if line.strip()]
    if not dirty_paths:
        return CheckResult(
            id="active_repo_clean",
            severity="OK",
            status="ok",
            message="Active Kanban agent checkout is clean",
        )

    authority_dirty = [
        path for path in dirty_paths if _is_authority_dirty_path(path)
    ]
    if authority_dirty:
        return CheckResult(
            id="active_repo_dirty_authority",
            severity="P0",
            status="fail",
            message="Active Kanban agent checkout has dirty path-authority/runtime-authority files",
            detail=", ".join(authority_dirty),
        )

    return CheckResult(
        id="active_repo_dirty",
        severity="P1",
        status="warn",
        message="Active Kanban agent checkout is dirty",
        detail=", ".join(dirty_paths),
    )


def build_path_authority_report(
    *,
    paths_env: ParsedPathsEnv | None = None,
    systemd_user_dir: Path | None = None,
    include_live: bool = True,
) -> PathAuthorityReport:
    parsed_env = paths_env or parse_paths_env()
    expected_dir = parsed_env.values.get("HERMES_KANBAN_AGENT_DIR")
    systemd_dir = systemd_user_dir or Path.home() / ".config" / "systemd" / "user"
    units = {
        role: read_effective_unit(role, name, systemd_dir)
        for role, name in UNIT_NAMES.items()
    }

    checks: list[CheckResult] = []
    if expected_dir:
        checks.append(
            CheckResult(
                id="expected_agent_dir_configured",
                severity="OK",
                status="ok",
                message="HERMES_KANBAN_AGENT_DIR is configured",
                detail=expected_dir,
            )
        )
    else:
        checks.append(
            CheckResult(
                id="expected_agent_dir_missing",
                severity="P0",
                status="fail",
                message="HERMES_KANBAN_AGENT_DIR is missing from paths.env",
            )
        )

    expected_path = Path(expected_dir).expanduser() if expected_dir else None
    if expected_path is not None:
        for role, unit in units.items():
            checks.extend(_check_unit_authority(role, unit, expected_path))

        checks.append(_check_dirty_repo(expected_path))

    checks.extend(_check_webui_graphify_authority(parsed_env, units))
    checks.extend(_check_graphify_freshness(parsed_env))
    checks.extend(_check_cron_shadow_authority(parsed_env))

    runtime_authority = _read_runtime_import_authority() if include_live else None
    if expected_path is not None:
        if runtime_authority:
            checks.append(_check_runtime_authority(runtime_authority, expected_path))
        elif include_live:
            checks.append(
                CheckResult(
                    id="runtime_import_authority_missing",
                    severity="P0",
                    status="fail",
                    message="Persisted runtime import authority is missing or unreadable",
                )
            )

    return PathAuthorityReport(
        expected_agent_dir=expected_dir,
        paths_env=parsed_env,
        units=units,
        checks=checks,
        runtime_import_authority=runtime_authority,
    )


def read_effective_unit(role: str, unit_name: str, systemd_user_dir: Path) -> UnitReport:
    unit_path = systemd_user_dir / unit_name
    dropin_dir = systemd_user_dir / f"{unit_name}.d"
    dropins = sorted(dropin_dir.glob("*.conf")) if dropin_dir.exists() else []
    effective = EffectiveUnit()

    sources = [unit_path] if unit_path.exists() else []
    sources.extend(dropins)
    for source in sources:
        _apply_unit_source(effective, source.read_text(encoding="utf-8"))

    return UnitReport(
        role=role,
        name=unit_name,
        path=unit_path,
        exists=unit_path.exists(),
        dropins=dropins,
        effective=effective,
    )


def run_path_authority_doctor(args: Any) -> int:
    report = build_path_authority_report(include_live=not getattr(args, "no_live", False))
    if getattr(args, "json", False):
        print(json.dumps(report.to_report(), indent=2, sort_keys=True))
    else:
        _print_text_report(report)
    return report.exit_code(strict=bool(getattr(args, "strict", False)))


def _parse_env_assignment(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export "):].lstrip()
    if "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
        return None
    value = _strip_inline_comment(value.strip())
    try:
        parts = shlex.split(value, posix=True)
    except ValueError:
        parts = [value.strip("'\"")]
    parsed_value = parts[0] if parts else ""
    return key, parsed_value


def _strip_inline_comment(value: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(value):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            if index == 0 or value[index - 1].isspace():
                return value[:index].rstrip()
    return value


def _apply_unit_source(effective: EffectiveUnit, text: str) -> None:
    section = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            continue
        if section != "Service" or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key == "ExecStart":
            value = _unquote_systemd_value(value)
            if value == "":
                effective.exec_start.clear()
            else:
                effective.exec_start.append(value)
        elif key == "WorkingDirectory":
            value = _unquote_systemd_value(value)
            effective.working_directory = value or None
        elif key == "Environment":
            if value == "":
                effective.environment.clear()
            else:
                effective.environment.update(_parse_environment_directive(value))


def _parse_environment_directive(value: str) -> dict[str, str]:
    try:
        parts = shlex.split(value, posix=True)
    except ValueError:
        parts = [value]
    env: dict[str, str] = {}
    for part in parts:
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        if key:
            env[key] = val
    return env


def _unquote_systemd_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _check_unit_authority(role: str, unit: UnitReport, expected_path: Path) -> list[CheckResult]:
    checks: list[CheckResult] = []
    if not unit.exists and not unit.dropins:
        checks.append(
            CheckResult(
                id=f"{role}_unit_missing",
                severity="P1",
                status="warn",
                message=f"{unit.name} is not installed in the user systemd directory",
            )
        )
        return checks

    if role == "gateway":
        if any(
            token == "--replace" or token.startswith("--replace=")
            for command in unit.effective.exec_start
            for token in _split_command(command)
        ):
            checks.append(
                CheckResult(
                    id="gateway_execstart_replace",
                    severity="P0",
                    status="fail",
                    message="Effective gateway ExecStart contains --replace",
                    detail="; ".join(
                        _redact_command(command) for command in unit.effective.exec_start
                    ),
                )
            )
        else:
            checks.append(
                CheckResult(
                    id="gateway_execstart_no_replace",
                    severity="OK",
                    status="ok",
                    message="Effective gateway ExecStart does not contain --replace",
                )
            )

    if _unit_points_to_expected_dir(role, unit.effective, expected_path):
        checks.append(
            CheckResult(
                id=f"{role}_path_authority",
                severity="OK",
                status="ok",
                message=f"{unit.name} points at HERMES_KANBAN_AGENT_DIR",
            )
        )
    else:
        checks.append(
            CheckResult(
                id=f"{role}_path_authority",
                severity="P0",
                status="fail",
                message=f"{unit.name} does not point at HERMES_KANBAN_AGENT_DIR",
                detail=_unit_authority_detail(unit.effective),
            )
        )
    return checks


def _unit_points_to_expected_dir(role: str, effective: EffectiveUnit, expected_path: Path) -> bool:
    env_path_keys = {
        "HERMES_KANBAN_AGENT_DIR",
        "HERMES_AGENT_DIR",
        "HERMES_AGENT_REPO",
        "HERMES_AGENT_ROOT",
        "HERMES_WEBUI_AGENT_DIR",
        "AGENT_DIR",
    }
    authority_env = {
        key: value
        for key, value in effective.environment.items()
        if key in env_path_keys
    }
    env_matches = any(
        key in env_path_keys and _path_is_within(value, expected_path)
        for key, value in effective.environment.items()
    )
    env_clean = all(_path_is_within(value, expected_path) for value in authority_env.values())
    exec_matches = any(
        _path_is_within(token, expected_path)
        for command in effective.exec_start
        for token in _split_command(command)
        if token.startswith("/") or token.startswith("~")
    )
    cwd_matches = bool(
        effective.working_directory
        and _path_is_within(effective.working_directory, expected_path)
    )

    if role == "webui":
        return env_matches and env_clean
    return cwd_matches and exec_matches and env_clean


def _unit_authority_detail(effective: EffectiveUnit) -> str:
    detail = {
        "exec_start": [_redact_command(command) for command in effective.exec_start],
        "working_directory": effective.working_directory,
        "environment": _redact_mapping(effective.environment),
    }
    return json.dumps(detail, sort_keys=True)


def _check_dirty_repo(expected_path: Path) -> CheckResult:
    try:
        proc = subprocess.run(
            ["git", "-C", str(expected_path), "status", "--porcelain", "--untracked-files=all"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return CheckResult(
            id="active_repo_git_status",
            severity="P2",
            status="warn",
            message="Could not read active Kanban agent git status",
            detail=str(exc),
        )
    if proc.returncode != 0:
        return CheckResult(
            id="active_repo_git_status",
            severity="P2",
            status="warn",
            message="Could not read active Kanban agent git status",
            detail=(proc.stderr or proc.stdout).strip(),
        )
    return classify_dirty_repo(proc.stdout.splitlines())


def _check_runtime_authority(runtime_authority: dict[str, Any], expected_path: Path) -> CheckResult:
    project_root = runtime_authority.get("project_root")
    gateway_file = runtime_authority.get("gateway_file")
    hermes_cli_file = runtime_authority.get("hermes_cli_file")
    fields = [project_root, gateway_file, hermes_cli_file]
    if all(value and _path_is_within(str(value), expected_path) for value in fields):
        return CheckResult(
            id="runtime_import_authority",
            severity="OK",
            status="ok",
            message="Persisted runtime import authority points at HERMES_KANBAN_AGENT_DIR",
        )
    return CheckResult(
        id="runtime_import_authority",
        severity="P0",
        status="fail",
        message="Persisted runtime import authority does not point at HERMES_KANBAN_AGENT_DIR",
        detail=json.dumps(runtime_authority, sort_keys=True),
    )


def _check_webui_graphify_authority(
    paths_env: ParsedPathsEnv,
    units: dict[str, UnitReport],
) -> list[CheckResult]:
    expected = {
        key: value
        for key in ("GRAPHIFY_OUTPUT_DIR", "GRAPHIFY_GRAPH_JSON")
        if (value := paths_env.values.get(key))
    }
    if not expected:
        return []

    webui = units.get("webui")
    if webui is None or not webui.exists:
        return []

    mismatches = []
    for key, expected_value in expected.items():
        actual = webui.effective.environment.get(key)
        if actual != expected_value:
            mismatches.append(
                {
                    "key": key,
                    "expected": expected_value,
                    "actual": actual,
                }
            )
    if mismatches:
        return [
            CheckResult(
                id="webui_graphify_path_authority",
                severity="P1",
                status="fail",
                message="WebUI systemd Graphify paths differ from paths.env",
                detail=json.dumps(mismatches, sort_keys=True),
            )
        ]
    return [
        CheckResult(
            id="webui_graphify_path_authority",
            severity="OK",
            status="ok",
            message="WebUI systemd Graphify paths match paths.env",
        )
    ]


def _check_graphify_freshness(paths_env: ParsedPathsEnv) -> list[CheckResult]:
    vault = paths_env.values.get("OBSIDIAN_VAULT_PATH")
    graph = paths_env.values.get("GRAPHIFY_GRAPH_JSON")
    if not vault or not graph:
        return []

    vault_path = Path(vault).expanduser()
    graph_path = Path(graph).expanduser()
    if not graph_path.exists():
        return [
            CheckResult(
                id="graphify_freshness",
                severity="P1",
                status="fail",
                message="Graphify sidecar graph is missing",
                detail=str(graph_path),
            )
        ]

    sources = [
        candidate
        for relative in GRAPH_FRESHNESS_RELATIVE_PATHS
        if (candidate := vault_path / relative).exists()
    ]
    if not sources:
        return [
            CheckResult(
                id="graphify_freshness",
                severity="P2",
                status="warn",
                message="No Graphify governance freshness source files were found",
                detail=str(vault_path),
            )
        ]

    graph_mtime = graph_path.stat().st_mtime
    latest = max(sources, key=lambda path: path.stat().st_mtime)
    latest_mtime = latest.stat().st_mtime
    detail = json.dumps(
        {
            "graph": str(graph_path),
            "graph_mtime": graph_mtime,
            "latest_source": str(latest),
            "latest_source_mtime": latest_mtime,
        },
        sort_keys=True,
    )
    if latest_mtime > graph_mtime + 1:
        return [
            CheckResult(
                id="graphify_freshness",
                severity="P1",
                status="warn",
                message="Graphify sidecar graph is older than memory governance sources",
                detail=detail,
            )
        ]
    return [
        CheckResult(
            id="graphify_freshness",
            severity="OK",
            status="ok",
            message="Graphify sidecar graph is not older than memory governance sources",
            detail=detail,
        )
    ]


def _check_cron_shadow_authority(paths_env: ParsedPathsEnv) -> list[CheckResult]:
    hermes_home = paths_env.values.get("HERMES_HOME")
    if not hermes_home:
        return []

    cron_dir = Path(hermes_home).expanduser() / "cron"
    active = cron_dir / "jobs.json"
    shadow = cron_dir / "cron" / "jobs.json"
    marker = shadow.parent / "NON_AUTHORITATIVE"
    checks: list[CheckResult] = []

    if not active.exists():
        checks.append(
            CheckResult(
                id="cron_active_registry_missing",
                severity="P1",
                status="warn",
                message="Canonical cron registry is missing",
                detail=str(active),
            )
        )
        return checks

    checks.append(
        CheckResult(
            id="cron_active_registry_present",
            severity="OK",
            status="ok",
            message="Canonical cron registry is present",
            detail=str(active),
        )
    )

    if not shadow.exists():
        checks.append(
            CheckResult(
                id="cron_shadow_registry_absent",
                severity="OK",
                status="ok",
                message="No nested shadow cron registry exists",
            )
        )
        return checks

    detail = _cron_shadow_detail(shadow, marker)
    if marker.exists():
        checks.append(
            CheckResult(
                id="cron_shadow_registry_quarantined",
                severity="OK",
                status="ok",
                message="Nested shadow cron registry is explicitly marked non-authoritative",
                detail=detail,
            )
        )
    else:
        checks.append(
            CheckResult(
                id="cron_shadow_registry_unclassified",
                severity="P1",
                status="fail",
                message="Nested shadow cron registry exists without a non-authoritative marker",
                detail=detail,
            )
        )
    return checks


def _cron_shadow_detail(shadow: Path, marker: Path) -> str:
    enabled_count = 0
    stale_root_count = 0
    try:
        raw = shadow.read_text(encoding="utf-8")
        stale_root_count = raw.count("/root/")
        data = json.loads(raw)
        jobs = data.get("jobs", data if isinstance(data, list) else [])
        if isinstance(jobs, list):
            enabled_count = sum(1 for job in jobs if isinstance(job, dict) and job.get("enabled"))
    except Exception:
        pass
    return json.dumps(
        {
            "shadow": str(shadow),
            "marker": str(marker),
            "marker_present": marker.exists(),
            "enabled_jobs": enabled_count,
            "stale_root_refs": stale_root_count,
        },
        sort_keys=True,
    )


def _read_runtime_import_authority() -> dict[str, Any] | None:
    try:
        from gateway.status import read_runtime_status

        state = read_runtime_status() or {}
    except Exception:
        return None
    authority = state.get("runtime_import_authority")
    return authority if isinstance(authority, dict) else None


def _path_is_within(value: str, expected_path: Path) -> bool:
    if not value:
        return False
    try:
        candidate = Path(os.path.abspath(os.path.expanduser(value)))
        expected = Path(os.path.abspath(os.path.expanduser(str(expected_path))))
    except OSError:
        return False
    return candidate == expected or expected in candidate.parents


def _split_command(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _redact_command(command: str) -> str:
    tokens = _split_command(command)
    if not tokens:
        return ""

    redacted: list[str] = []
    redact_next = False
    for token in tokens:
        lowered = token.lower()
        if redact_next:
            redacted.append("[REDACTED]")
            redact_next = False
            continue
        if any(lowered == name for name in SECRET_ARG_NAMES):
            redacted.append(token)
            redact_next = True
            continue
        matched_flag = next(
            (name for name in SECRET_ARG_NAMES if lowered.startswith(f"{name}=")),
            None,
        )
        if matched_flag is not None:
            flag = token.split("=", 1)[0]
            redacted.append(f"{flag}=[REDACTED]")
            continue
        if SECRET_ASSIGNMENT_RE.search(token):
            key = token.split("=", 1)[0]
            redacted.append(f"{key}=[REDACTED]")
            continue
        redacted.append(token)
    return shlex.join(redacted)


def _status_path(line: str) -> str:
    path = line[3:] if len(line) > 3 else line.strip()
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    return path.strip()


def _is_authority_dirty_path(path: str) -> bool:
    return path in AUTHORITY_DIRTY_PATHS or path.startswith(".config/systemd/")


def _is_secret_key(key: str) -> bool:
    upper = key.upper()
    return any(fragment in upper for fragment in SECRET_KEY_FRAGMENTS)


def _redact_mapping(values: dict[str, str]) -> dict[str, str]:
    redacted: dict[str, str] = {}
    for key, value in values.items():
        if _is_secret_key(key):
            redacted[key] = "[REDACTED]"
        elif key in PATH_ENV_KEYS or key.endswith("_DIR") or key.endswith("_PATH") or key.endswith("_ROOT"):
            redacted[key] = value
        else:
            redacted[key] = "[SET]" if value else ""
    return redacted


def _print_text_report(report: PathAuthorityReport) -> None:
    print()
    print("Hermes Path Authority Doctor")
    print(f"Status: {report.overall_severity}")
    print(f"paths.env: {report.paths_env.path} ({'present' if report.paths_env.exists else 'missing'})")
    print(f"HERMES_KANBAN_AGENT_DIR: {report.expected_agent_dir or 'MISSING'}")
    print()
    for check in report.checks:
        label = check.severity if check.status != "ok" else "OK"
        print(f"[{label}] {check.id}: {check.message}")
        if check.detail:
            print(f"  {check.detail}")


if __name__ == "__main__":
    class _Args:
        json = "--json" in sys.argv
        no_live = "--no-live" in sys.argv
        strict = "--strict" in sys.argv

    raise SystemExit(run_path_authority_doctor(_Args()))
