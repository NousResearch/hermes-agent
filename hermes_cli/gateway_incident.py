"""Redacted gateway incident bundle helpers.

The incident bundle is intentionally conservative: it writes a local evidence
directory, but its observations are read-only and it never copies raw logs,
environment blocks, bearer tokens, memory files, or provider-specific facts.
"""

from __future__ import annotations

import json
import math
import os
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_cli.gateway_validation import build_gateway_validation_report


SCHEMA_VERSION = 1
KNOWN_LOG_FILES = (
    "gateway.log",
    "gateway.error.log",
    "gateway-exit-diag.log",
    "gateway-shutdown-diag.log",
    "agent.log",
    "errors.log",
)
KNOWN_HEALTH_LOOP_FILES = (
    "status.md",
    "status.json",
)


class IncidentBundleError(RuntimeError):
    """Raised when a bundle cannot be created safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _positive_timeout(value: Any, name: str) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        raise IncidentBundleError(f"{name} must be a positive number of seconds") from None
    if not math.isfinite(timeout) or timeout <= 0:
        raise IncidentBundleError(f"{name} must be a positive number of seconds")
    return timeout


def _hermes_home() -> Path:
    raw = os.environ.get("HERMES_HOME")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".hermes"


def _health_loop_root() -> Path:
    raw = os.environ.get("HERMES_HEALTH_LOOP_ROOT")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / "Operator" / "health-loop"


def _default_output_dir() -> Path:
    return Path(tempfile.gettempdir()) / (
        f"hermes-gateway-incident-{_utc_stamp()}-{os.getpid()}"
    )


def _display_path(path: Path) -> str:
    text = str(path)
    try:
        home = str(Path.home())
        if text == home:
            return "~"
        if text.startswith(home + os.sep):
            return "~" + text[len(home):]
    except Exception:
        pass
    return text


def _kind_from_mode(mode: int) -> str:
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISLNK(mode):
        return "symlink"
    return "other"


def _file_metadata(label: str, path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "label": label,
        "path": _display_path(path),
        "exists": False,
        "content_copied": False,
    }
    try:
        st = path.lstat()
    except FileNotFoundError:
        return metadata
    except OSError as exc:
        metadata["error"] = str(exc)[:160]
        return metadata

    mode = stat.S_IMODE(st.st_mode)
    metadata.update(
        {
            "exists": True,
            "kind": _kind_from_mode(st.st_mode),
            "mode": f"{mode:04o}",
            "group_or_other_access": bool(mode & 0o077),
            "size_bytes": st.st_size if stat.S_ISREG(st.st_mode) else None,
            "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
        }
    )
    return metadata


def collect_gateway_incident_metadata(
    *,
    include_log_metadata: bool = True,
    include_health_loop_metadata: bool = True,
) -> dict[str, Any]:
    """Collect metadata-only incident context without reading file contents."""
    hermes_home = _hermes_home()
    logs_root = hermes_home / "logs"
    health_root = _health_loop_root()

    artifact_metadata: list[dict[str, Any]] = []
    if include_log_metadata:
        artifact_metadata.append(_file_metadata("hermes_home.logs", logs_root))
        artifact_metadata.extend(
            _file_metadata(f"hermes_home.logs.{name}", logs_root / name)
            for name in KNOWN_LOG_FILES
        )
    if include_health_loop_metadata:
        artifact_metadata.append(_file_metadata("operator.health_loop", health_root))
        artifact_metadata.extend(
            _file_metadata(f"operator.health_loop.{name}", health_root / name)
            for name in KNOWN_HEALTH_LOOP_FILES
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "owner": "hermes-reliability-plane",
        "read_only_observations": True,
        "redacted": True,
        "content_copied": False,
        "sources": {
            "hermes_home": _display_path(hermes_home),
            "logs_root": _display_path(logs_root),
            "health_loop_root": _display_path(health_root),
        },
        "artifacts": artifact_metadata,
    }


def _prepare_output_dir(output_dir: Path | str | None, *, force: bool) -> Path:
    target = Path(output_dir).expanduser() if output_dir else _default_output_dir()
    if target.exists() and not target.is_dir():
        raise IncidentBundleError(f"output path exists and is not a directory: {target}")
    if target.exists() and any(target.iterdir()) and not force:
        raise IncidentBundleError(
            f"output directory is not empty: {target}; use --force to overwrite bundle files"
        )
    target.mkdir(parents=True, exist_ok=True)
    target.chmod(0o700)
    return target


def _write_text_private(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    path.chmod(0o600)


def _write_json_private(path: Path, payload: dict[str, Any]) -> None:
    _write_text_private(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _summary_markdown(
    *,
    manifest: dict[str, Any],
    validation_report: dict[str, Any],
    metadata: dict[str, Any],
) -> str:
    validation = validation_report["summary"]
    existing_artifacts = sum(1 for item in metadata["artifacts"] if item.get("exists"))
    wide_permissions = sum(
        1 for item in metadata["artifacts"] if item.get("group_or_other_access")
    )
    lines = [
        "# Hermes Gateway Incident Bundle",
        "",
        f"- Created: `{manifest['created_at']}`",
        f"- Runtime mutation: `{str(manifest['runtime_mutation']).lower()}`",
        f"- Redacted: `{str(manifest['redacted']).lower()}`",
        f"- Raw log content copied: `{str(manifest['raw_log_content_copied']).lower()}`",
        f"- Gateway validation: `{validation_report['overall_status'].upper()}`",
        f"- Checks: `{validation['checks']}`",
        f"- Errors: `{validation['errors']}`",
        f"- Warnings: `{validation['warnings']}`",
        f"- Metadata artifacts found: `{existing_artifacts}`",
        f"- Metadata artifacts with group/other access: `{wide_permissions}`",
        "",
        "## Safe Next Commands",
        "",
        "```bash",
        "hermes gateway validate --json",
        "hermes gateway status",
        "hermes doctor",
        "hermes logs gateway --since 30m --level WARNING",
        "```",
        "",
        "Do not attach raw logs or launchd environment dumps without a separate",
        "secret/private-data review.",
    ]
    return "\n".join(lines) + "\n"


def build_gateway_incident_bundle(
    *,
    output_dir: Path | str | None = None,
    force: bool = False,
    check_health: bool = True,
    launchctl_timeout: float = 5.0,
    health_timeout: float = 2.0,
    include_log_metadata: bool = True,
    include_health_loop_metadata: bool = True,
) -> dict[str, Any]:
    """Create a redacted local gateway incident bundle."""
    launchctl_timeout = _positive_timeout(launchctl_timeout, "--launchctl-timeout")
    health_timeout = _positive_timeout(health_timeout, "--health-timeout")
    target = _prepare_output_dir(output_dir, force=force)

    validation_report = build_gateway_validation_report(
        check_health=check_health,
        launchctl_timeout=launchctl_timeout,
        health_timeout=health_timeout,
    )
    metadata = collect_gateway_incident_metadata(
        include_log_metadata=include_log_metadata,
        include_health_loop_metadata=include_health_loop_metadata,
    )
    created_at = _utc_now()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "owner": "hermes-reliability-plane",
        "risk_tier": "R1",
        "created_at": created_at,
        "read_only_observations": True,
        "runtime_mutation": False,
        "redacted": True,
        "raw_log_content_copied": False,
        "private_memory_read": False,
        "external_side_effects": False,
        "output_dir": str(target),
        "artifacts": [
            "manifest.json",
            "gateway_validation.json",
            "artifact_metadata.json",
            "summary.md",
        ],
    }

    _write_json_private(target / "manifest.json", manifest)
    _write_json_private(target / "gateway_validation.json", validation_report)
    _write_json_private(target / "artifact_metadata.json", metadata)
    _write_text_private(
        target / "summary.md",
        _summary_markdown(
            manifest=manifest,
            validation_report=validation_report,
            metadata=metadata,
        ),
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "output_dir": str(target),
        "files": [str(target / name) for name in manifest["artifacts"]],
        "bundle_created": True,
        "validation_status": validation_report["overall_status"],
        "validation_summary": validation_report["summary"],
        "read_only_observations": True,
        "runtime_mutation": False,
        "redacted": True,
        "raw_log_content_copied": False,
        "private_memory_read": False,
    }


def run_gateway_incident_bundle(args: Any) -> bool:
    try:
        result = build_gateway_incident_bundle(
            output_dir=getattr(args, "output", None),
            force=bool(getattr(args, "force", False)),
            check_health=not bool(getattr(args, "no_health", False)),
            launchctl_timeout=getattr(args, "launchctl_timeout", 5.0),
            health_timeout=getattr(args, "health_timeout", 2.0),
            include_log_metadata=not bool(getattr(args, "no_log_metadata", False)),
            include_health_loop_metadata=not bool(
                getattr(args, "no_health_loop_metadata", False)
            ),
        )
    except IncidentBundleError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return False

    if getattr(args, "json", False):
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"Hermes gateway incident bundle: {result['output_dir']}")
        print(f"Gateway validation: {result['validation_status'].upper()}")
        print(
            "Files: "
            + ", ".join(Path(path).name for path in result["files"])
        )
        print("Raw log content copied: false")
    return True
