"""Agent readiness checks for profile paths and assigned credentials.

The checker is intentionally independent from the dashboard so the same
contract can be used by profile creation, ``hermes doctor``, cron/doctor jobs,
and future UI endpoints.  It does not inspect secret values; it only verifies
that the runtime is pointed at the expected profile and that configured
credential files are present/readable.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from hermes_constants import get_hermes_home

READY = "ready"
WARN = "warn"
REPAIRING = "repairing"
FIXED = "fixed"
REPAIR_FAILED = "repair_failed"
NEEDS_MANUAL_INTERVENTION = "needs_manual_intervention"

_READY_DIR = ".readiness"
_ISSUES_FILE = "issues.json"
_LOG_FILE = "repair.log.jsonl"


@dataclass
class ReadinessIssue:
    """A structured warning surfaced on an agent profile."""

    check: str
    message: str
    expected: str
    actual: str
    status: str = WARN
    severity: str = "warn"
    repairable: bool = False
    first_detected_at: str = ""
    last_checked_at: str = ""
    last_repair_attempt_at: str | None = None
    repair_error: str | None = None


@dataclass
class ReadinessResult:
    """Result of a readiness pass for one profile."""

    profile: str
    profile_dir: str
    status: str
    issues: list[ReadinessIssue] = field(default_factory=list)
    repaired: int = 0
    log_path: str | None = None
    issues_path: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in {READY, FIXED} and not self.issues


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return default


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _append_log(profile_dir: Path, event: dict[str, Any]) -> Path:
    readiness_dir = profile_dir / _READY_DIR
    readiness_dir.mkdir(parents=True, exist_ok=True)
    log_path = readiness_dir / _LOG_FILE
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    return log_path


def _profile_name_from_dir(profile_dir: Path) -> str:
    if profile_dir.parent.name == "profiles":
        return profile_dir.name
    return "default"


def _safe_relative_or_absolute(path_text: str, profile_dir: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(path_text)))
    if not path.is_absolute():
        path = profile_dir / path
    return path


def _path_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except (OSError, ValueError):
        return False


def _expected_profile_paths(profile_dir: Path) -> dict[str, Path]:
    return {
        "profile_home": profile_dir,
        "config": profile_dir / "config.yaml",
        "env": profile_dir / ".env",
        "skills": profile_dir / "skills",
        "plugins": profile_dir / "plugins",
        "cron": profile_dir / "cron",
        "memories": profile_dir / "memories",
        "sessions": profile_dir / "sessions",
        "logs": profile_dir / "logs",
        "workspace": profile_dir / "workspace",
    }


def _configured_credential_files(config: dict[str, Any], profile_dir: Path) -> list[dict[str, Any]]:
    """Return credential-file checks from config.

    The dashboard/UI can write either::

        agent_readiness:
          credential_files:
            - name: google
              path: google_token.json

    or, for convenience, ``credentials`` with the same shape.  String entries
    are treated as relative paths under the profile directory.
    """

    readiness_cfg = config.get("agent_readiness") or config.get("readiness") or {}
    if not isinstance(readiness_cfg, dict):
        return []
    entries = readiness_cfg.get("credential_files") or readiness_cfg.get("credentials") or []
    if not isinstance(entries, list):
        return []

    checks: list[dict[str, Any]] = []
    for entry in entries:
        if isinstance(entry, str):
            checks.append({"name": entry, "path": str(_safe_relative_or_absolute(entry, profile_dir))})
            continue
        if not isinstance(entry, dict):
            continue
        raw_path = entry.get("path") or entry.get("file")
        if not raw_path:
            continue
        name = str(entry.get("name") or raw_path)
        checks.append(
            {
                "name": name,
                "path": str(_safe_relative_or_absolute(str(raw_path), profile_dir)),
                "required": bool(entry.get("required", True)),
            }
        )
    return checks


def _issue(
    *,
    check: str,
    message: str,
    expected: Path | str,
    actual: Path | str,
    repairable: bool = False,
    severity: str = "warn",
    previous: dict[str, Any] | None = None,
) -> ReadinessIssue:
    now = _now()
    return ReadinessIssue(
        check=check,
        message=message,
        expected=str(expected),
        actual=str(actual),
        repairable=repairable,
        severity=severity,
        first_detected_at=(previous or {}).get("first_detected_at") or now,
        last_checked_at=now,
        last_repair_attempt_at=(previous or {}).get("last_repair_attempt_at"),
        repair_error=(previous or {}).get("repair_error"),
    )


def _merge_previous(issues: Iterable[ReadinessIssue], previous_rows: list[dict[str, Any]]) -> list[ReadinessIssue]:
    previous_by_check = {
        str(row.get("check")): row for row in previous_rows if isinstance(row, dict)
    }
    merged: list[ReadinessIssue] = []
    for issue in issues:
        prev = previous_by_check.get(issue.check)
        if prev:
            issue.first_detected_at = prev.get("first_detected_at") or issue.first_detected_at
            issue.last_repair_attempt_at = prev.get("last_repair_attempt_at")
            issue.repair_error = prev.get("repair_error")
        merged.append(issue)
    return merged


def _load_config_for_profile(profile_dir: Path) -> dict[str, Any]:
    config_path = profile_dir / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml

        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def check_agent_readiness(
    *,
    profile_name: str | None = None,
    profile_dir: Path | str | None = None,
    runtime_home: Path | str | None = None,
    config: dict[str, Any] | None = None,
    repair: bool = False,
    persist: bool = True,
) -> ReadinessResult:
    """Check one agent profile for path/credential readiness drift.

    ``profile_dir`` is the expected profile home. ``runtime_home`` is the
    home the currently running process resolved; if omitted, the live
    ``get_hermes_home()`` value is used.  For commissioning-time tests, callers
    can pass ``runtime_home=profile_dir`` to verify the new profile's intended
    environment without mutating global process state.
    """

    expected_home = Path(profile_dir) if profile_dir is not None else get_hermes_home()
    expected_home = expected_home.expanduser()
    actual_home = Path(runtime_home) if runtime_home is not None else get_hermes_home()
    actual_home = actual_home.expanduser()
    profile = profile_name or _profile_name_from_dir(expected_home)
    cfg = config if config is not None else _load_config_for_profile(expected_home)

    issues_path = expected_home / _READY_DIR / _ISSUES_FILE
    previous = _read_json(issues_path, []) if issues_path.exists() else []
    previous_by_check = {str(row.get("check")): row for row in previous if isinstance(row, dict)}

    issues: list[ReadinessIssue] = []

    if expected_home.resolve() != actual_home.resolve():
        issues.append(
            _issue(
                check="profile_home",
                message="Runtime HERMES_HOME does not match the assigned agent profile home.",
                expected=expected_home,
                actual=actual_home,
                repairable=False,
                previous=previous_by_check.get("profile_home"),
            )
        )

    for name, path in _expected_profile_paths(expected_home).items():
        if name in {"config", "env"}:
            # Freshly-created profiles may not have config.yaml until first
            # setup/config write.  .env is checked separately as a credential
            # boundary, so keep this check about directory drift.
            continue
        if not _path_under(path, expected_home):
            issues.append(
                _issue(
                    check=f"path:{name}",
                    message=f"Profile path {name!r} is outside the assigned profile home.",
                    expected=expected_home,
                    actual=path,
                    repairable=False,
                    previous=previous_by_check.get(f"path:{name}"),
                )
            )
        if not path.exists():
            issues.append(
                _issue(
                    check=f"path_exists:{name}",
                    message=f"Required profile path {name!r} is missing.",
                    expected=path,
                    actual="missing",
                    repairable=True,
                    previous=previous_by_check.get(f"path_exists:{name}"),
                )
            )

    env_path = expected_home / ".env"
    if not env_path.exists():
        issues.append(
            _issue(
                check="credential_boundary:.env",
                message="Profile credential boundary .env is missing.",
                expected=env_path,
                actual="missing",
                repairable=True,
                previous=previous_by_check.get("credential_boundary:.env"),
            )
        )
    elif not os.access(env_path, os.R_OK):
        issues.append(
            _issue(
                check="credential_boundary:.env.readable",
                message="Profile credential boundary .env is not readable by this runtime.",
                expected="readable",
                actual=env_path,
                repairable=False,
                previous=previous_by_check.get("credential_boundary:.env.readable"),
            )
        )

    for credential in _configured_credential_files(cfg, expected_home):
        path = Path(str(credential["path"]))
        check_id = f"credential:{credential['name']}"
        if not path.exists():
            issues.append(
                _issue(
                    check=check_id,
                    message=f"Assigned credential file {credential['name']!r} is missing.",
                    expected=path,
                    actual="missing",
                    repairable=False,
                    previous=previous_by_check.get(check_id),
                )
            )
        elif not os.access(path, os.R_OK):
            issues.append(
                _issue(
                    check=check_id,
                    message=f"Assigned credential file {credential['name']!r} is not readable.",
                    expected="readable",
                    actual=path,
                    repairable=False,
                    previous=previous_by_check.get(check_id),
                )
            )

    repaired = 0
    if repair and issues:
        for item in issues:
            if not item.repairable:
                continue
            item.status = REPAIRING
            item.last_repair_attempt_at = _now()
            log_event = {
                "timestamp": item.last_repair_attempt_at,
                "profile": profile,
                "check": item.check,
                "status": REPAIRING,
                "expected": item.expected,
                "actual": item.actual,
            }
            try:
                if item.check.startswith("path_exists:"):
                    Path(item.expected).mkdir(parents=True, exist_ok=True)
                elif item.check == "credential_boundary:.env":
                    env_path.parent.mkdir(parents=True, exist_ok=True)
                    env_path.write_text(
                        "# Per-profile secrets for this Hermes profile.\n",
                        encoding="utf-8",
                    )
                    try:
                        os.chmod(str(env_path), 0o600)
                    except OSError:
                        pass
                else:
                    continue
                item.status = FIXED
                item.repair_error = None
                log_event["status"] = FIXED
                repaired += 1
            except OSError as exc:
                item.status = REPAIR_FAILED
                item.repair_error = str(exc)
                log_event["status"] = REPAIR_FAILED
                log_event["error"] = str(exc)
            if persist:
                _append_log(expected_home, log_event)

        # Re-check after safe repairs so fixed issues disappear from the active
        # warning list and only remain in the audit log.
        if repaired:
            return check_agent_readiness(
                profile_name=profile,
                profile_dir=expected_home,
                runtime_home=actual_home,
                config=cfg,
                repair=False,
                persist=persist,
            )

    unresolved = [item for item in issues if item.status not in {FIXED}]
    status = READY if not unresolved else WARN
    if persist:
        if unresolved:
            _write_json(issues_path, [asdict(item) for item in unresolved])
        else:
            _write_json(issues_path, [])
    return ReadinessResult(
        profile=profile,
        profile_dir=str(expected_home),
        status=status,
        issues=unresolved,
        repaired=repaired,
        log_path=str(expected_home / _READY_DIR / _LOG_FILE),
        issues_path=str(issues_path),
    )


def check_all_agent_readiness(*, repair: bool = False, persist: bool = True) -> list[ReadinessResult]:
    """Run the readiness contract for the default profile and all named profiles."""

    from hermes_cli.profiles import iter_profiles_for_gateway

    results: list[ReadinessResult] = []
    for profile_name, profile_dir in iter_profiles_for_gateway(multiplex=True):
        runtime_home = profile_dir if profile_name != _profile_name_from_dir(get_hermes_home()) else get_hermes_home()
        results.append(
            check_agent_readiness(
                profile_name=profile_name,
                profile_dir=profile_dir,
                runtime_home=runtime_home,
                repair=repair,
                persist=persist,
            )
        )
    return results
