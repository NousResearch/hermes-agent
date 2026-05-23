"""Read-only gateway startup validation helpers.

This module deliberately avoids restart, repair, log tailing, or raw launchd
environment output.  It exists to give operators a concise preflight receipt
for the startup risks documented in the Hermes reliability plan.
"""

from __future__ import annotations

import json
import math
import plistlib
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_cli.config import get_env_value, read_raw_config

CANONICAL_LAUNCHD_LABEL = "ai.hermes.gateway"
LEGACY_LAUNCHD_LABEL = "com.agent1.hermes.gateway"
DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8642


@dataclass(frozen=True)
class LaunchdLabelState:
    label: str
    loaded: bool
    running: bool | None
    pid: int | None = None
    status: int | None = None
    error: str | None = None


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _positive_timeout(value: Any, name: str) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive number of seconds") from None
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(f"{name} must be a positive number of seconds")
    return timeout


def _operator_wrapper_path() -> Path:
    try:
        from hermes_cli import gateway as gateway_cli

        home = gateway_cli._launchd_user_home()
    except Exception:
        home = Path.home()
    return home / "Operator" / "scripts" / "hermes-gateway.sh"


def _check(
    checks: list[dict[str, Any]],
    check_id: str,
    status: str,
    message: str,
    *,
    severity: str = "info",
    evidence: dict[str, Any] | None = None,
) -> None:
    checks.append(
        {
            "id": check_id,
            "status": status,
            "severity": severity,
            "message": message,
            "evidence": evidence or {},
        }
    )


def _redacted_launchctl_error(text: str | None) -> str | None:
    if not text:
        return None
    line = text.strip().splitlines()[0].strip()
    if not line:
        return None
    # launchctl errors should be non-secret, but keep output short and avoid
    # relaying arbitrary stderr bodies into diagnostics.
    return line[:180]


def _launchctl_label_state(label: str, timeout: float = 5.0) -> LaunchdLabelState:
    try:
        result = subprocess.run(
            ["launchctl", "list", label],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return LaunchdLabelState(
            label=label,
            loaded=False,
            running=None,
            error="launchctl not found",
        )
    except subprocess.TimeoutExpired:
        return LaunchdLabelState(
            label=label,
            loaded=False,
            running=None,
            error="launchctl timed out",
        )

    if result.returncode != 0:
        return LaunchdLabelState(
            label=label,
            loaded=False,
            running=False,
            error=_redacted_launchctl_error(result.stderr or result.stdout),
        )

    pid: int | None = None
    status: int | None = None
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[-1] == label:
            try:
                pid = int(parts[0])
            except ValueError:
                pid = None
            try:
                status = int(parts[1])
            except ValueError:
                status = None
            break

    if pid is None:
        match = re.search(r'"PID"\s*=\s*(\d+);', result.stdout)
        if match:
            try:
                pid = int(match.group(1))
            except ValueError:
                pid = None
    if status is None:
        match = re.search(r'"LastExitStatus"\s*=\s*(-?\d+);', result.stdout)
        if match:
            try:
                status = int(match.group(1))
            except ValueError:
                status = None

    return LaunchdLabelState(
        label=label,
        loaded=True,
        running=bool(pid and pid > 0),
        pid=pid,
        status=status,
    )


def _load_plist(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        payload = plistlib.load(fh)
    return payload if isinstance(payload, dict) else {}


def _launchd_program_summary(payload: dict[str, Any], expected_wrapper: Path) -> dict[str, Any]:
    args = payload.get("ProgramArguments")
    if not isinstance(args, list):
        args = []
    program = payload.get("Program")
    if not isinstance(program, str):
        program = None

    first_arg = str(args[0]) if args else None
    wrapper = str(expected_wrapper)
    uses_wrapper = program == wrapper or first_arg == wrapper
    if uses_wrapper:
        command_kind = "operator_wrapper"
    elif any("hermes_cli.main" in str(item) for item in args):
        command_kind = "python_module"
    else:
        command_kind = "unknown"

    return {
        "command_kind": command_kind,
        "uses_expected_wrapper": uses_wrapper,
        "program": program,
        "first_argument": first_arg,
        "argument_count": len(args),
        "stdout_path": payload.get("StandardOutPath")
        if isinstance(payload.get("StandardOutPath"), str)
        else None,
        "stderr_path": payload.get("StandardErrorPath")
        if isinstance(payload.get("StandardErrorPath"), str)
        else None,
        "has_environment_variables": isinstance(payload.get("EnvironmentVariables"), dict),
    }


def _api_server_settings() -> dict[str, Any]:
    raw_config = read_raw_config()
    platforms = raw_config.get("platforms") if isinstance(raw_config, dict) else {}
    platform_cfg = {}
    if isinstance(platforms, dict):
        candidate = platforms.get("api_server") or {}
        if isinstance(candidate, dict):
            platform_cfg = candidate
    extra = platform_cfg.get("extra") if isinstance(platform_cfg.get("extra"), dict) else {}

    env_enabled = _truthy(get_env_value("API_SERVER_ENABLED"))
    env_key_present = bool(get_env_value("API_SERVER_KEY"))
    config_enabled = _truthy(platform_cfg.get("enabled"))
    config_key_present = bool(extra.get("key"))

    host = get_env_value("API_SERVER_HOST") or extra.get("host") or DEFAULT_API_HOST
    raw_port = get_env_value("API_SERVER_PORT") or extra.get("port") or DEFAULT_API_PORT
    try:
        port = int(raw_port)
    except (TypeError, ValueError):
        port = DEFAULT_API_PORT

    return {
        "enabled": env_enabled or env_key_present or config_enabled or config_key_present,
        "host": str(host),
        "port": port,
        "auth_configured": env_key_present or config_key_present,
    }


def _probe_host(host: str) -> str | None:
    normalized = host.strip().strip("[]").lower()
    if normalized in {"127.0.0.1", "localhost", "::1"}:
        return "127.0.0.1" if normalized != "localhost" else "localhost"
    if normalized in {"0.0.0.0", "::"}:
        return "127.0.0.1"
    return None


def _http_status(url: str, timeout: float) -> tuple[int | None, str | None]:
    request = urllib.request.Request(
        url,
        method="GET",
        headers={"User-Agent": "hermes-gateway-validate"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.getcode()), None
    except urllib.error.HTTPError as exc:
        return int(exc.code), None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return None, str(exc)[:180]


def _validate_health(
    checks: list[dict[str, Any]],
    *,
    timeout: float,
) -> dict[str, Any]:
    settings = _api_server_settings()
    probe_host = _probe_host(settings["host"])
    health: dict[str, Any] = {
        "enabled": bool(settings["enabled"]),
        "host": settings["host"],
        "port": settings["port"],
        "probe_host": probe_host,
        "auth_configured": bool(settings["auth_configured"]),
        "health_status": None,
        "detailed_status": None,
    }

    if not settings["enabled"]:
        _check(
            checks,
            "api_server.optional",
            "skip",
            "API server is not configured; health probe skipped.",
        )
        return health
    if not probe_host:
        _check(
            checks,
            "api_server.local_probe",
            "skip",
            "API server host is not local; live probe skipped to avoid external access.",
            evidence={"host": settings["host"], "port": settings["port"]},
        )
        return health

    base = f"http://{probe_host}:{settings['port']}"
    status, error = _http_status(f"{base}/health", timeout)
    health["health_status"] = status
    if status == 200:
        _check(
            checks,
            "api_server.health",
            "pass",
            "Unauthenticated /health returned HTTP 200.",
            evidence={"status": status, "url": "/health"},
        )
    else:
        _check(
            checks,
            "api_server.health",
            "fail",
            "Configured API server did not return HTTP 200 from /health.",
            severity="error",
            evidence={"status": status, "url": "/health", "error": error},
        )
        return health

    detailed_status, detailed_error = _http_status(f"{base}/health/detailed", timeout)
    health["detailed_status"] = detailed_status
    if detailed_status == 200:
        _check(
            checks,
            "api_server.health_detailed",
            "pass",
            "Unauthenticated /health/detailed returned HTTP 200 because no auth gate is active.",
            evidence={"status": detailed_status, "url": "/health/detailed"},
        )
    elif detailed_status in {401, 403}:
        _check(
            checks,
            "api_server.health_detailed",
            "pass",
            "Unauthenticated /health/detailed returned an expected auth response.",
            evidence={"status": detailed_status, "url": "/health/detailed"},
        )
    else:
        _check(
            checks,
            "api_server.health_detailed",
            "warn",
            "Unauthenticated /health/detailed returned an unexpected status.",
            severity="warning",
            evidence={"status": detailed_status, "url": "/health/detailed", "error": detailed_error},
        )
    return health


def build_gateway_validation_report(
    *,
    check_health: bool = True,
    launchctl_timeout: float = 5.0,
    health_timeout: float = 2.0,
    expected_wrapper: Path | None = None,
) -> dict[str, Any]:
    """Build a redacted, read-only gateway startup validation report."""
    checks: list[dict[str, Any]] = []
    expected_wrapper = expected_wrapper or _operator_wrapper_path()
    launchd: dict[str, Any] = {
        "platform": sys.platform,
        "active_label": None,
        "canonical_label": CANONICAL_LAUNCHD_LABEL,
        "legacy_label": LEGACY_LAUNCHD_LABEL,
        "plist_path": None,
        "expected_wrapper": str(expected_wrapper),
        "active_label_state": None,
        "legacy_label_state": None,
        "program_summary": None,
    }

    if sys.platform != "darwin":
        _check(
            checks,
            "launchd.platform",
            "skip",
            "launchd validation applies only on macOS.",
        )
    else:
        from hermes_cli import gateway as gateway_cli

        active_label = gateway_cli.get_launchd_label()
        plist_path = gateway_cli.get_launchd_plist_path()
        launchd["active_label"] = active_label
        launchd["plist_path"] = str(plist_path)

        if active_label == CANONICAL_LAUNCHD_LABEL:
            _check(checks, "launchd.active_label", "pass", "Active launchd label is canonical.")
        else:
            _check(
                checks,
                "launchd.active_label",
                "warn",
                "Active launchd label is profile-scoped or non-canonical.",
                severity="warning",
                evidence={"active_label": active_label},
            )

        if plist_path.exists():
            _check(
                checks,
                "launchd.plist",
                "pass",
                "Launchd plist exists.",
                evidence={"path": str(plist_path)},
            )
            try:
                payload = _load_plist(plist_path)
            except Exception as exc:
                payload = {}
                _check(
                    checks,
                    "launchd.plist_parse",
                    "fail",
                    "Launchd plist could not be parsed.",
                    severity="error",
                    evidence={"error": str(exc)[:180]},
                )
            else:
                _check(checks, "launchd.plist_parse", "pass", "Launchd plist parsed.")
                summary = _launchd_program_summary(payload, expected_wrapper)
                launchd["program_summary"] = summary
                if summary["uses_expected_wrapper"]:
                    _check(
                        checks,
                        "launchd.wrapper",
                        "pass",
                        "Launchd service uses the expected operator wrapper.",
                        evidence={
                            "command_kind": summary["command_kind"],
                            "first_argument": summary["first_argument"],
                        },
                    )
                else:
                    _check(
                        checks,
                        "launchd.wrapper",
                        "fail",
                        "Launchd service does not use the expected operator wrapper.",
                        severity="error",
                        evidence={
                            "command_kind": summary["command_kind"],
                            "first_argument": summary["first_argument"],
                            "expected_wrapper": str(expected_wrapper),
                        },
                    )
        else:
            _check(
                checks,
                "launchd.plist",
                "fail",
                "Launchd plist is missing.",
                severity="error",
                evidence={"path": str(plist_path)},
            )

        active_state = _launchctl_label_state(active_label, timeout=launchctl_timeout)
        launchd["active_label_state"] = {
            "loaded": active_state.loaded,
            "running": active_state.running,
            "pid_present": active_state.pid is not None,
            "status": active_state.status,
            "error": active_state.error,
        }
        if not active_state.loaded:
            _check(
                checks,
                "launchd.loaded",
                "fail",
                "Active launchd label is not loaded.",
                severity="error",
                evidence={"label": active_label, "error": active_state.error},
            )
        elif not active_state.running:
            _check(
                checks,
                "launchd.loaded",
                "fail",
                "Active launchd label is loaded but not running.",
                severity="error",
                evidence={
                    "label": active_label,
                    "running": active_state.running,
                    "pid_present": active_state.pid is not None,
                    "status": active_state.status,
                },
            )
        elif active_state.status not in {None, 0}:
            _check(
                checks,
                "launchd.loaded",
                "warn",
                "Active launchd label is running; previous exit status was non-zero.",
                severity="warning",
                evidence={
                    "label": active_label,
                    "running": active_state.running,
                    "pid_present": active_state.pid is not None,
                    "status": active_state.status,
                },
            )
        else:
            _check(
                checks,
                "launchd.loaded",
                "pass",
                "Active launchd label is loaded and running.",
                evidence={
                    "label": active_label,
                    "running": active_state.running,
                    "pid_present": active_state.pid is not None,
                    "status": active_state.status,
                },
            )

        legacy_state = _launchctl_label_state(LEGACY_LAUNCHD_LABEL, timeout=launchctl_timeout)
        launchd["legacy_label_state"] = {
            "loaded": legacy_state.loaded,
            "running": legacy_state.running,
            "pid_present": legacy_state.pid is not None,
            "status": legacy_state.status,
            "error": legacy_state.error,
        }
        if legacy_state.loaded:
            _check(
                checks,
                "launchd.legacy_label",
                "warn",
                "Legacy launchd label is loaded; avoid restarting it by habit.",
                severity="warning",
                evidence={
                    "label": LEGACY_LAUNCHD_LABEL,
                    "running": legacy_state.running,
                    "pid_present": legacy_state.pid is not None,
                    "status": legacy_state.status,
                },
            )
        else:
            _check(
                checks,
                "launchd.legacy_label",
                "pass",
                "Legacy launchd label is not loaded.",
                evidence={"label": LEGACY_LAUNCHD_LABEL},
            )

    health = _validate_health(checks, timeout=health_timeout) if check_health else {"skipped": True}
    if not check_health:
        _check(checks, "api_server.health", "skip", "Health probe disabled by operator flag.")

    errors = sum(1 for item in checks if item["status"] == "fail")
    warnings = sum(1 for item in checks if item["status"] == "warn")
    overall_status = "pass" if errors == 0 else "fail"
    return {
        "schema_version": 1,
        "owner": "hermes-reliability-plane",
        "risk_tier": "R0",
        "read_only": True,
        "redacted": True,
        "overall_status": overall_status,
        "summary": {
            "checks": len(checks),
            "errors": errors,
            "warnings": warnings,
        },
        "launchd": launchd,
        "health": health,
        "checks": checks,
    }


def format_gateway_validation_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Hermes Gateway Startup Validation",
        "",
        f"- Status: `{report['overall_status'].upper()}`",
        f"- Read only: `{str(report['read_only']).lower()}`",
        f"- Risk tier: `{report['risk_tier']}`",
        f"- Checks: `{report['summary']['checks']}`",
        f"- Errors: `{report['summary']['errors']}`",
        f"- Warnings: `{report['summary']['warnings']}`",
        "",
        "## Checks",
    ]
    for item in report["checks"]:
        lines.append(
            f"- `{item['status'].upper()}` `{item['id']}`: {item['message']}"
        )
    return "\n".join(lines) + "\n"


def format_gateway_validation_text(report: dict[str, Any]) -> str:
    lines = [
        f"Hermes gateway startup validation: {report['overall_status'].upper()}",
        (
            f"Checks: {report['summary']['checks']}  "
            f"Errors: {report['summary']['errors']}  "
            f"Warnings: {report['summary']['warnings']}"
        ),
        "",
    ]
    for item in report["checks"]:
        lines.append(f"[{item['status'].upper():4}] {item['id']}: {item['message']}")
    return "\n".join(lines)


def run_gateway_validation(args: Any) -> bool:
    raw_launchctl_timeout = getattr(args, "launchctl_timeout", None)
    if raw_launchctl_timeout is None:
        raw_launchctl_timeout = 5.0
    raw_health_timeout = getattr(args, "health_timeout", None)
    if raw_health_timeout is None:
        raw_health_timeout = 2.0

    try:
        launchctl_timeout = _positive_timeout(
            raw_launchctl_timeout,
            "--launchctl-timeout",
        )
        health_timeout = _positive_timeout(
            raw_health_timeout,
            "--health-timeout",
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return False

    report = build_gateway_validation_report(
        check_health=not bool(getattr(args, "no_health", False)),
        launchctl_timeout=launchctl_timeout,
        health_timeout=health_timeout,
    )
    if getattr(args, "json", False):
        print(json.dumps(report, indent=2, sort_keys=True))
    elif getattr(args, "markdown", False):
        print(format_gateway_validation_markdown(report), end="")
    else:
        print(format_gateway_validation_text(report))
    return report["overall_status"] == "pass"
