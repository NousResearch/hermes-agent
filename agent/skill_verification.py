"""Runtime verification gates for loaded skills.

Skills can declare a deterministic before-final verifier in SKILL.md
frontmatter.  When the agent loads such a skill via skill_view, this module
records the verifier for the current session.  turn_finalizer drains and runs
those verifiers before returning the final response.
"""

from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.skill_utils import parse_frontmatter

_MAX_OUTPUT_PREVIEW_CHARS = 1200
_DEFAULT_TIMEOUT_SECONDS = 120
_MAX_TIMEOUT_SECONDS = 3600
_PENDING_BY_SESSION: dict[str, dict[str, "SkillVerificationSpec"]] = {}
_LOCK = threading.Lock()


@dataclass(frozen=True)
class SkillVerificationSpec:
    name: str
    command: str
    skill_dir: str | None = None
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS
    success_exit_codes: tuple[int, ...] = (0,)
    on_failure: str = "block_final"


@dataclass(frozen=True)
class SkillVerificationCheck:
    spec: SkillVerificationSpec
    passed: bool
    exit_code: int | None = None
    timed_out: bool = False
    error: str = ""
    stdout: str = ""
    stderr: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.spec.name,
            "command": self.spec.command,
            "passed": self.passed,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "error": self.error,
            "on_failure": self.spec.on_failure,
        }


@dataclass(frozen=True)
class SkillVerificationResult:
    checks: tuple[SkillVerificationCheck, ...]
    blocked: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocked": self.blocked,
            "checks": [check.to_dict() for check in self.checks],
        }


def record_skill_view_payload(session_id: str | None, payload: dict[str, Any]) -> bool:
    """Record a loaded skill's before-final verifier for this session."""
    key = _session_key(session_id)
    if not key or not isinstance(payload, dict) or not payload.get("success"):
        return False

    spec = extract_before_final_spec(payload)
    if spec is None:
        return False

    with _LOCK:
        _PENDING_BY_SESSION.setdefault(key, {})[spec.name] = spec
    return True


def pending_verification_specs(session_id: str | None) -> list[SkillVerificationSpec]:
    """Return pending specs for tests and observability."""
    key = _session_key(session_id)
    if not key:
        return []
    with _LOCK:
        return list(_PENDING_BY_SESSION.get(key, {}).values())


def clear_session_verifications(session_id: str | None) -> None:
    key = _session_key(session_id)
    if not key:
        return
    with _LOCK:
        _PENDING_BY_SESSION.pop(key, None)


def run_before_final_verifications(session_id: str | None) -> SkillVerificationResult | None:
    """Run and drain pending before-final verifiers for a session."""
    specs = _drain_session_specs(session_id)
    if not specs:
        return None

    checks = tuple(_run_check(spec) for spec in specs)
    blocking_failures = [
        check
        for check in checks
        if not check.passed and check.spec.on_failure == "block_final"
    ]
    blocked = bool(blocking_failures)
    message = _format_blocked_message(blocking_failures) if blocked else ""
    return SkillVerificationResult(checks=checks, blocked=blocked, message=message)


def extract_before_final_spec(payload: dict[str, Any]) -> SkillVerificationSpec | None:
    """Extract a required before-final verifier from a skill_view payload."""
    content = str(payload.get("content") or "")
    frontmatter, _body = parse_frontmatter(content)
    verification = _extract_verification_block(frontmatter)
    if not verification:
        return None

    before_final = verification.get("before_final")
    if not isinstance(before_final, dict):
        return None
    if not bool(before_final.get("required")):
        return None

    command = before_final.get("command")
    if not isinstance(command, str) or not command.strip():
        return None

    return SkillVerificationSpec(
        name=str(payload.get("name") or frontmatter.get("name") or "unknown-skill"),
        command=command.strip(),
        skill_dir=_normalize_optional_string(payload.get("skill_dir")),
        timeout_seconds=_normalize_timeout(before_final.get("timeout_seconds")),
        success_exit_codes=_normalize_success_exit_codes(
            before_final.get("success_exit_codes")
        ),
        on_failure=_normalize_on_failure(before_final.get("on_failure")),
    )


def _extract_verification_block(frontmatter: dict[str, Any]) -> dict[str, Any] | None:
    metadata = frontmatter.get("metadata")
    if isinstance(metadata, dict):
        hermes_meta = metadata.get("hermes")
        if isinstance(hermes_meta, dict):
            verification = hermes_meta.get("verification")
            if isinstance(verification, dict):
                return verification

    verification = frontmatter.get("verification")
    if isinstance(verification, dict):
        return verification
    return None


def _drain_session_specs(session_id: str | None) -> list[SkillVerificationSpec]:
    key = _session_key(session_id)
    if not key:
        return []
    with _LOCK:
        specs_by_name = _PENDING_BY_SESSION.pop(key, {})
    return list(specs_by_name.values())


def _run_check(spec: SkillVerificationSpec) -> SkillVerificationCheck:
    cwd = _existing_skill_dir(spec.skill_dir)
    try:
        completed = subprocess.run(
            spec.command,
            cwd=str(cwd) if cwd else None,
            shell=True,
            capture_output=True,
            text=True,
            timeout=spec.timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return SkillVerificationCheck(
            spec=spec,
            passed=False,
            timed_out=True,
            stdout=_coerce_output(exc.stdout),
            stderr=_coerce_output(exc.stderr),
            error=f"timed out after {spec.timeout_seconds}s",
        )
    except Exception as exc:
        return SkillVerificationCheck(
            spec=spec,
            passed=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    return SkillVerificationCheck(
        spec=spec,
        passed=completed.returncode in spec.success_exit_codes,
        exit_code=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


def _format_blocked_message(failures: list[SkillVerificationCheck]) -> str:
    lines = [
        "Skill verification failed before the final response.",
        "",
        "The assistant's unverified final answer was blocked because one or more required skill verifiers failed:",
    ]
    for check in failures:
        lines.extend(
            [
                "",
                f"- skill: {check.spec.name}",
                f"  command: {check.spec.command}",
            ]
        )
        if check.timed_out:
            lines.append(f"  result: timed out after {check.spec.timeout_seconds}s")
        elif check.exit_code is not None:
            lines.append(f"  exit_code: {check.exit_code}")
        if check.error:
            lines.append(f"  error: {check.error}")
        output = _preview_output(check.stdout, check.stderr)
        if output:
            lines.append("  output:")
            lines.extend(f"    {line}" for line in output.splitlines())
    return "\n".join(lines)


def _preview_output(stdout: str, stderr: str) -> str:
    chunks = []
    if stdout.strip():
        chunks.append("stdout:\n" + stdout.strip())
    if stderr.strip():
        chunks.append("stderr:\n" + stderr.strip())
    out = "\n\n".join(chunks)
    if len(out) <= _MAX_OUTPUT_PREVIEW_CHARS:
        return out
    return out[: _MAX_OUTPUT_PREVIEW_CHARS - 3].rstrip() + "..."


def _normalize_timeout(raw: Any) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = _DEFAULT_TIMEOUT_SECONDS
    return max(1, min(value, _MAX_TIMEOUT_SECONDS))


def _normalize_success_exit_codes(raw: Any) -> tuple[int, ...]:
    if raw is None:
        return (0,)
    if not isinstance(raw, list):
        raw = [raw]
    values: list[int] = []
    for item in raw:
        try:
            values.append(int(item))
        except (TypeError, ValueError):
            continue
    return tuple(values) or (0,)


def _normalize_on_failure(raw: Any) -> str:
    value = str(raw or "block_final").strip().lower()
    return value if value else "block_final"


def _normalize_optional_string(raw: Any) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


def _existing_skill_dir(raw: str | None) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    try:
        return path if path.is_dir() else None
    except OSError:
        return None


def _session_key(session_id: str | None) -> str:
    return str(session_id or "").strip()


def _coerce_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
