#!/usr/bin/env python3
"""Run pinned, read-only external audits and bind results to exact git state."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Callable, Protocol, cast

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hermes_constants import get_hermes_home


_MAX_CAPTURE_CHARS = 200_000
_LOCKFILE_NAME = ".audit-tool-lock.json"
_ZIZMOR_SEVERITIES = {"High", "Medium", "Low", "Informational"}
_ZIZMOR_CONFIDENCES = {"High", "Medium", "Low"}


@dataclass(frozen=True)
class AuditCommand:
    name: str
    argv: tuple[str, ...]
    expected_version: str
    distribution_name: str | None = None

    @property
    def version_argv(self) -> tuple[str, ...]:
        """Return the command used to observe the installed tool realization."""
        return (self.argv[0], "--version")


class AuditProcessResult(Protocol):
    """The subprocess fields consumed by the read-only audit pipeline."""

    @property
    def returncode(self) -> int: ...

    @property
    def stdout(self) -> str: ...

    @property
    def stderr(self) -> str: ...


@dataclass(frozen=True)
class SkippedAuditResult:
    """Typed result for an audit that cannot run after preparation fails."""

    returncode: int
    stdout: str
    stderr: str


def build_commands(
    *, repo_root: Path, tool_root: Path, requirements_path: Path | None = None
) -> tuple[AuditCommand, ...]:
    """Return fixed audit argv; registry or user data can never inject commands."""
    requirements_path = requirements_path or repo_root / ".audit-requirements.txt"
    return (
        AuditCommand(
            name="zizmor",
            expected_version="1.30.0",
            argv=(
                str(tool_root / "zizmor"),
                "--offline",
                "--format",
                "json",
                "--no-progress",
                "--no-exit-codes",
                ".github",
            ),
        ),
        AuditCommand(
            name="import-linter",
            expected_version="2.14",
            distribution_name="import-linter",
            argv=(
                str(tool_root / "lint-imports"),
                "--config",
                ".importlinter",
                "--no-cache",
                "--no-logo",
            ),
        ),
        AuditCommand(
            name="pip-audit",
            expected_version="2.10.1",
            distribution_name="pip-audit",
            argv=(
                str(tool_root / "pip-audit"),
                "-r",
                str(requirements_path),
                "--require-hashes",
                "--disable-pip",
                "--format",
                "json",
                "--progress-spinner",
                "off",
                "--cache-dir",
                str(Path(tempfile.gettempdir()) / "hermes-pip-audit-cache"),
            ),
        ),
    )


def build_uv_export_command(
    *, requirements_path: Path, cache_path: Path | None = None,
    uv_path: Path | None = None,
) -> tuple[str, ...]:
    """Export the existing uv lock without resolving or modifying dependencies."""
    cache_path = cache_path or Path(tempfile.gettempdir()) / "hermes-uv-audit-cache"
    return (
        str(uv_path) if uv_path is not None else (shutil.which("uv") or "uv"),
        "export",
        "--locked",
        "--cache-dir",
        str(cache_path),
        "--no-dev",
        "--no-emit-project",
        "--no-annotate",
        "--no-header",
        "--output-file",
        str(requirements_path),
    )


def _run(
    runner: Callable[..., Any],
    argv: Sequence[str],
    *,
    repo_root: Path,
    timeout: int,
) -> AuditProcessResult:
    return cast(
        AuditProcessResult,
        runner(
            list(argv),
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        ),
    )


def _git_value(runner: Callable[..., Any], repo_root: Path, *arguments: str) -> str:
    result = _run(runner, ("git", *arguments), repo_root=repo_root, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(arguments)} failed")
    return result.stdout.strip()


def _sha256(text: str) -> str:
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _file_sha256(path: Path) -> str | None:
    """Hash the exact executable bytes used for an audit."""
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"
    except (OSError, ValueError):
        return None


def _distribution_realization_digest(
    tool_root: Path, distribution_name: str
) -> str | None:
    """Hash every recorded file in one installed Python distribution.

    Console-script launchers are only entry points; their imported package can
    change without changing the launcher bytes.  Bind the receipt to the
    complete RECORD-described realization inside the tool venv instead.
    """
    tool_root = tool_root.expanduser().resolve()
    venv_root = tool_root.parent
    expected_name = re.sub(r"[-_.]+", "-", distribution_name).casefold()
    for site_packages in sorted(venv_root.glob("lib/python*/site-packages")):
        for metadata_dir in sorted(site_packages.glob("*.dist-info")):
            metadata_path = metadata_dir / "METADATA"
            try:
                metadata = metadata_path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            actual_name = next(
                (
                    line.split(":", 1)[1].strip()
                    for line in metadata.splitlines()
                    if line.casefold().startswith("name:")
                ),
                None,
            )
            if actual_name is None:
                continue
            normalized_name = re.sub(r"[-_.]+", "-", actual_name).casefold()
            if normalized_name != expected_name:
                continue
            record_path = metadata_dir / "RECORD"
            try:
                with record_path.open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.reader(stream))
            except (OSError, UnicodeError, csv.Error):
                return None
            files: list[tuple[str, str]] = []
            venv_root_resolved = venv_root.resolve()
            for row in rows:
                if not row or not row[0]:
                    return None
                relative = PurePosixPath(row[0])
                file_path = (site_packages / Path(*relative.parts)).resolve()
                try:
                    relative_to_venv = file_path.relative_to(venv_root_resolved)
                except ValueError:
                    return None
                digest = _file_sha256(file_path)
                if digest is None:
                    return None
                files.append((relative_to_venv.as_posix(), digest))
            if not files:
                return None
            serialized = json.dumps(
                {"distribution": normalized_name, "files": sorted(files)},
                sort_keys=True,
                separators=(",", ":"),
            )
            return _sha256(serialized)
    return None


def _resolve_executable(path: str | Path) -> Path | None:
    candidate = Path(path).expanduser()
    try:
        candidate = candidate.resolve()
    except OSError:
        return None
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return candidate
    return None


def _load_tool_lock(repo_root: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Load and validate repository-owned executable/version pins."""
    path = repo_root / _LOCKFILE_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"unable to read {_LOCKFILE_NAME}: {exc}"
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return None, f"invalid {_LOCKFILE_NAME} schema"
    tools = payload.get("tools")
    uv = payload.get("uv")
    if not isinstance(tools, dict) or not isinstance(uv, dict):
        return None, f"invalid {_LOCKFILE_NAME} entries"
    for entry in [uv, *[tools.get(name) for name in ("zizmor", "import-linter", "pip-audit")]]:
        if not isinstance(entry, dict) or not isinstance(entry.get("version"), str) or not isinstance(entry.get("sha256"), str):
            return None, f"invalid {_LOCKFILE_NAME} pin"
    for name in ("import-linter", "pip-audit"):
        if not isinstance(tools.get(name, {}).get("distribution_sha256"), str):
            return None, f"invalid {_LOCKFILE_NAME} distribution pin for {name}"
    return payload, None


def _bounded_output(text: str) -> str:
    if len(text) <= _MAX_CAPTURE_CHARS:
        return text
    return text[:_MAX_CAPTURE_CHARS] + "\n...[capture truncated]"


def _extract_version(stdout: str, stderr: str) -> str | None:
    """Extract a concrete semantic version from a tool's observed output."""
    match = re.search(r"(?<!\d)(\d+(?:\.\d+){1,3})(?!\d)", f"{stdout}\n{stderr}")
    return match.group(1) if match else None


def _summarize_json_output(name: str, stdout: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        return None

    if name == "zizmor" and isinstance(payload, list):
        if any(
            not isinstance(row, dict)
            or not isinstance(row.get("ident"), str)
            or not isinstance(row.get("determinations"), dict)
            or not isinstance(row.get("locations"), list)
            or row["determinations"].get("severity") not in _ZIZMOR_SEVERITIES
            or row["determinations"].get("confidence") not in _ZIZMOR_CONFIDENCES
            or any(not isinstance(location, dict) for location in row["locations"])
            for row in payload
        ):
            return None
        determinations = [row["determinations"] for row in payload]
        return {
            "finding_groups": len(payload),
            "locations": sum(len(row.get("locations", [])) for row in payload),
            "by_severity": dict(
                Counter(
                    determination.get("severity", "Unknown")
                    for determination in determinations
                )
            ),
            "by_confidence": dict(
                Counter(
                    determination.get("confidence", "Unknown")
                    for determination in determinations
                )
            ),
            "by_ident": dict(Counter(row.get("ident", "unknown") for row in payload)),
            "high_confidence_high_severity": sum(
                determination.get("severity") == "High"
                and determination.get("confidence") == "High"
                for determination in determinations
            ),
        }
    if name == "pip-audit" and isinstance(payload, dict):
        dependencies = payload.get("dependencies")
        if not isinstance(dependencies, list) or any(
            not isinstance(row, dict)
            or not isinstance(row.get("name"), str)
            or not isinstance(row.get("version"), str)
            or not isinstance(row.get("vulns"), list)
            or any(not isinstance(vulnerability, dict) for vulnerability in row["vulns"])
            for row in dependencies
        ):
            return None
        return {
            "dependencies": len(dependencies),
            "vulnerabilities": sum(len(row.get("vulns", [])) for row in dependencies),
        }
    return None


def _policy_ok(
    name: str, returncode: int, summary: dict[str, Any] | None
) -> bool:
    """Evaluate audit findings instead of treating process success as policy success."""
    if returncode != 0:
        return False
    if name == "zizmor":
        return summary is not None and summary["high_confidence_high_severity"] == 0
    if name == "pip-audit":
        return summary is not None and summary["vulnerabilities"] == 0
    return True


def run_audits(
    *,
    repo_root: Path,
    tool_root: Path,
    runner: Callable[..., Any] = subprocess.run,
    uv_path: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    head = _git_value(runner, repo_root, "rev-parse", "HEAD")
    branch = _git_value(runner, repo_root, "branch", "--show-current")
    status = _git_value(runner, repo_root, "status", "--porcelain")
    lock, lock_error = _load_tool_lock(repo_root)

    audits: list[dict[str, Any]] = []
    preparations: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="hermes-external-audit-") as temp_dir:
        requirements_path = Path(temp_dir) / "requirements.txt"
        resolved_uv = _resolve_executable(uv_path or (shutil.which("uv") or ""))
        uv_pin = lock.get("uv", {}) if lock else {}
        uv_digest = _file_sha256(resolved_uv) if resolved_uv else None
        uv_version_result: AuditProcessResult = SkippedAuditResult(1, "", "uv executable unavailable")
        if resolved_uv is not None:
            uv_version_result = _run(runner, (str(resolved_uv), "--version"), repo_root=repo_root, timeout=30)
        uv_version_stdout = str(uv_version_result.stdout or "")
        uv_version_stderr = str(uv_version_result.stderr or "")
        uv_observed_version = _extract_version(uv_version_stdout, uv_version_stderr)
        uv_version_matches = uv_version_result.returncode == 0 and uv_observed_version == uv_pin.get("version")
        uv_realization_ok = bool(resolved_uv and uv_digest == uv_pin.get("sha256") and uv_version_matches)
        export_command = build_uv_export_command(requirements_path=requirements_path, uv_path=resolved_uv)
        if uv_realization_ok:
            export_result = _run(runner, export_command, repo_root=repo_root, timeout=120)
        else:
            export_result = SkippedAuditResult(1, "", "uv realization is not locked; export was not run")
        export_stdout = str(export_result.stdout or "")
        export_stderr = str(export_result.stderr or "")
        preparations.append({
            "name": "uv-lock-export",
            "command": list(export_command),
            "returncode": int(export_result.returncode),
            "stdout_sha256": _sha256(export_stdout),
            "stderr_sha256": _sha256(export_stderr),
            "stdout": _bounded_output(export_stdout),
            "stderr": _bounded_output(export_stderr),
            "executable_path": str(resolved_uv) if resolved_uv else None,
            "executable_sha256": uv_digest,
            "expected_sha256": uv_pin.get("sha256"),
            "expected_version": uv_pin.get("version"),
            "observed_version": uv_observed_version,
            "version_matches": uv_version_matches,
            "realization_ok": uv_realization_ok,
            "version_command": [str(resolved_uv), "--version"] if resolved_uv else [],
            "version_returncode": int(uv_version_result.returncode),
            "version_stdout_sha256": _sha256(uv_version_stdout),
            "version_stderr_sha256": _sha256(uv_version_stderr),
            "requirements_sha256": _file_sha256(requirements_path) if requirements_path.exists() else None,
        })

        for command in build_commands(
            repo_root=repo_root,
            tool_root=tool_root,
            requirements_path=requirements_path,
        ):
            resolved_executable = _resolve_executable(command.argv[0])
            executable_digest = _file_sha256(resolved_executable) if resolved_executable else None
            pin = lock.get("tools", {}).get(command.name, {}) if lock else {}
            distribution_digest = (
                _distribution_realization_digest(tool_root, command.distribution_name)
                if command.distribution_name is not None
                else None
            )
            expected_distribution_digest = (
                pin.get("distribution_sha256")
                if command.distribution_name is not None
                else None
            )
            distribution_realization_ok = (
                None
                if command.distribution_name is None
                else distribution_digest == expected_distribution_digest
            )
            if resolved_executable is None:
                version_result: AuditProcessResult = SkippedAuditResult(1, "", "executable unavailable")
            else:
                version_result = _run(runner, command.version_argv, repo_root=repo_root, timeout=30)
            version_stdout = str(version_result.stdout or "")
            version_stderr = str(version_result.stderr or "")
            observed_version = _extract_version(version_stdout, version_stderr)
            version_matches = (
                version_result.returncode == 0
                and observed_version == command.expected_version
            )
            realization_ok = bool(
                resolved_executable
                and executable_digest == pin.get("sha256")
                and version_matches
                and (
                    command.distribution_name is None
                    or distribution_realization_ok is True
                )
            )
            if command.name == "pip-audit" and export_result.returncode != 0:
                result: AuditProcessResult = SkippedAuditResult(
                    returncode=1,
                    stdout="",
                    stderr="uv lock export failed; pip-audit was not run",
                )
            elif not realization_ok:
                result = SkippedAuditResult(
                    returncode=1,
                    stdout="",
                    stderr="tool realization is not locked; audit was not run",
                )
            else:
                result = _run(runner, command.argv, repo_root=repo_root, timeout=600)
            stdout = str(result.stdout or "")
            stderr = str(result.stderr or "")
            summary = _summarize_json_output(command.name, stdout)
            audits.append({
                "name": command.name,
                "expected_version": command.expected_version,
                "observed_version": observed_version,
                "version_matches": version_matches,
                "realization_ok": realization_ok,
                "distribution_name": command.distribution_name,
                "distribution_sha256": distribution_digest,
                "expected_distribution_sha256": expected_distribution_digest,
                "distribution_realization_ok": distribution_realization_ok,
                "audit_invoked": realization_ok and not (command.name == "pip-audit" and export_result.returncode != 0),
                "executable_path": str(resolved_executable) if resolved_executable else None,
                "executable_sha256": executable_digest,
                "expected_sha256": pin.get("sha256"),
                "version_command": list(command.version_argv),
                "version_returncode": int(version_result.returncode),
                "version_stdout_sha256": _sha256(version_stdout),
                "version_stderr_sha256": _sha256(version_stderr),
                "version_stdout": _bounded_output(version_stdout),
                "version_stderr": _bounded_output(version_stderr),
                "command": list(command.argv),
                "returncode": int(result.returncode),
                "stdout_sha256": _sha256(stdout),
                "stderr_sha256": _sha256(stderr),
                "summary": summary,
                "policy_ok": _policy_ok(command.name, result.returncode, summary),
                "stdout": _bounded_output(stdout),
                "stderr": _bounded_output(stderr),
            })

    return {
        "schema_version": "hermes_external_tooling_audit_v1",
        "audited_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "repository": {
            "path": str(repo_root),
            "head": head,
            "branch": branch,
            "dirty": bool(status),
            "status_porcelain": status,
        },
        "read_only": True,
        "auto_fix": False,
        "ok": not status
        and lock_error is None
        and all(row["returncode"] == 0 and row["realization_ok"] for row in preparations)
        and all(
            row["realization_ok"] and row["version_matches"] and row["policy_ok"]
            for row in audits
        ),
        "preparations": preparations,
        "audits": audits,
        "lock_error": lock_error,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--tool-root",
        type=Path,
        default=get_hermes_home() / "tooling" / "capability-audit" / "venv" / "bin",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--plan", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = args.repo.resolve()
    tool_root = args.tool_root.expanduser().resolve()
    if args.plan:
        requirements_path = repo_root / ".audit-requirements.txt"
        payload = {
            "read_only": True,
            "auto_fix": False,
            "preparations": [
                {
                    "name": "uv-lock-export",
                    "argv": list(
                        build_uv_export_command(requirements_path=requirements_path)
                    ),
                }
            ],
            "commands": [
                {"name": command.name, "argv": list(command.argv)}
                for command in build_commands(
                    repo_root=repo_root,
                    tool_root=tool_root,
                    requirements_path=requirements_path,
                )
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    receipt = run_audits(repo_root=repo_root, tool_root=tool_root)
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
