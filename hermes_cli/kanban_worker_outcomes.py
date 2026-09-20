"""Durable, truthful outcome records for Kanban worker subprocesses.

This module deliberately has no dependency on the Kanban database.  Worker
processes and the dispatcher can import it independently, which keeps the
terminal failure record available even when the database or parent process is
unhealthy.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional


class FailureClass(str, Enum):
    """Operational class assigned to a worker terminal outcome."""

    SUCCESS = "success"
    PROVIDER_CAPACITY = "provider_capacity"
    PROVIDER_TRANSPORT = "provider_transport"
    PROVIDER_AUTH = "provider_auth"
    CONTEXT_LIMIT = "context_limit"
    TOOL_UNAVAILABLE = "tool_unavailable"
    WORKSPACE_PROVISIONING = "workspace_provisioning"
    WORKER_PROTOCOL = "worker_protocol"
    TASK_LOGIC = "task_logic"
    WORKER_EXIT_UNKNOWN = "worker_exit_unknown"

    # Compatibility aliases used by callers that want to distinguish the
    # source wording without creating a second retry policy.
    PROVIDER_CONNECTION = "provider_transport"
    CONTEXT_OUTPUT_LIMIT = "context_limit"
    TOOL_MISCONFIGURED = "tool_unavailable"
    DETERMINISTIC_TASK = "task_logic"
    UNKNOWN = "worker_exit_unknown"


_FAILURE_CLASS_ALIASES = {
    "provider_connection": FailureClass.PROVIDER_TRANSPORT,
    "provider_transport": FailureClass.PROVIDER_TRANSPORT,
    "provider_authentication": FailureClass.PROVIDER_AUTH,
    "provider_auth": FailureClass.PROVIDER_AUTH,
    "context_output_limit": FailureClass.CONTEXT_LIMIT,
    "context_limit": FailureClass.CONTEXT_LIMIT,
    "tool_misconfigured": FailureClass.TOOL_UNAVAILABLE,
    "tool_unavailable": FailureClass.TOOL_UNAVAILABLE,
    "workspace": FailureClass.WORKSPACE_PROVISIONING,
    "workspace_provisioning": FailureClass.WORKSPACE_PROVISIONING,
    "protocol_violation": FailureClass.WORKER_PROTOCOL,
    "worker_protocol": FailureClass.WORKER_PROTOCOL,
    "deterministic_task": FailureClass.TASK_LOGIC,
    "task_logic": FailureClass.TASK_LOGIC,
    "unknown": FailureClass.WORKER_EXIT_UNKNOWN,
    "worker_exit_unknown": FailureClass.WORKER_EXIT_UNKNOWN,
}


def normalize_failure_class(value: FailureClass | str | None) -> Optional[FailureClass]:
    """Normalize enum values and wire aliases without guessing unknown input."""
    if isinstance(value, FailureClass):
        return value
    key = str(value or "").strip().casefold()
    if not key:
        return None
    alias = _FAILURE_CLASS_ALIASES.get(key)
    if alias is not None:
        return alias
    try:
        return FailureClass(key)
    except ValueError:
        return None


@dataclass(frozen=True)
class FailureClassification:
    """Classification and retry-budget policy for one worker outcome."""

    failure_class: FailureClass
    consumes_task_budget: bool
    retry_scope: str


@dataclass(frozen=True)
class WorkerExitEnvelope:
    """Immutable terminal record written before a worker wrapper exits."""

    task_id: str
    run_id: Optional[int] = None
    session_id: Optional[str] = None
    pid: Optional[int] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    exit_code: Optional[int] = None
    failure_class: FailureClass = FailureClass.WORKER_EXIT_UNKNOWN
    provider_status: Optional[int] = None
    retry_after_seconds: Optional[int] = None
    context_tokens: Optional[int] = None
    messages: Optional[int] = None
    tool_calls: Optional[int] = None
    last_successful_checkpoint: Optional[str] = None
    redacted_error: Optional[str] = None
    error_fingerprint: str = ""
    stderr_tail: Optional[str] = None
    schema_version: int = 1

    def to_dict(self) -> dict:
        """Return a JSON-safe copy with the enum encoded as its wire value."""
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "pid": self.pid,
            "provider": self.provider,
            "model": self.model,
            "exit_code": self.exit_code,
            "failure_class": (
                self.failure_class.value
                if isinstance(self.failure_class, FailureClass)
                else str(self.failure_class)
            ),
            "provider_status": self.provider_status,
            "retry_after_seconds": self.retry_after_seconds,
            "context_tokens": self.context_tokens,
            "messages": self.messages,
            "tool_calls": self.tool_calls,
            "last_successful_checkpoint": self.last_successful_checkpoint,
            "redacted_error": redact_error(self.redacted_error) if self.redacted_error else None,
            "error_fingerprint": self.error_fingerprint,
            "stderr_tail": redact_error(self.stderr_tail) if self.stderr_tail else None,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "WorkerExitEnvelope":
        """Validate and decode a persisted envelope."""
        if not isinstance(raw, dict):
            raise ValueError("worker exit envelope must be a JSON object")
        task_id = str(raw.get("task_id") or "").strip()
        if not task_id:
            raise ValueError("worker exit envelope requires task_id")
        failure_class = normalize_failure_class(
            raw.get("failure_class")
        ) or FailureClass.WORKER_EXIT_UNKNOWN

        def _int(name: str) -> Optional[int]:
            value = raw.get(name)
            if value is None or value == "":
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        def _text(name: str) -> Optional[str]:
            value = raw.get(name)
            return str(value) if value is not None else None

        return cls(
            task_id=task_id,
            run_id=_int("run_id"),
            session_id=_text("session_id"),
            pid=_int("pid"),
            provider=_text("provider"),
            model=_text("model"),
            exit_code=_int("exit_code"),
            failure_class=failure_class,
            provider_status=_int("provider_status"),
            retry_after_seconds=_int("retry_after_seconds"),
            context_tokens=_int("context_tokens"),
            messages=_int("messages"),
            tool_calls=_int("tool_calls"),
            last_successful_checkpoint=_text("last_successful_checkpoint"),
            redacted_error=redact_error(_text("redacted_error")) if raw.get("redacted_error") else None,
            error_fingerprint=str(raw.get("error_fingerprint") or ""),
            stderr_tail=redact_error(_text("stderr_tail")) if raw.get("stderr_tail") else None,
            schema_version=_int("schema_version") or 1,
        )


_MAX_ERROR_CHARS = 2_000
_MAX_TAIL_CHARS = 8_000
_QUOTED_SECRET_PATTERN = re.compile(
    r'''(?ix)
    (?P<prefix>
        (?<![a-z0-9_])
        ["']?(?:authorization|api[_ -]?key|token|secret|password)["']?
        (?![a-z0-9_])\s*[:=]\s*
        (?P<quote>["'])
        (?:(?:bearer|basic|token)\s+)?
    )
    [^"'\s,;}\]]+
    (?P<suffix>\s*(?P=quote))
    '''
)
_SECRET_PATTERNS = (
    re.compile(
        r"(?i)(\bauthorization\b\s*[:=]\s*(?:(?:bearer|basic|token)\s+)?)([^\s,;\"'{}\[\]]+)"
    ),
    re.compile(
        r"(?i)((?:\b|_)(?:api[_ -]?key|token|secret|password)\b\s*[:=]\s*)([^\s,;\"'{}\[\]]+)"
    ),
    re.compile(r"(?i)(\bBearer\s+)([^\s,;\"'{}\[\]]+)"),
    re.compile(r"\bsk-[A-Za-z0-9_-]+\b"),
    re.compile(r"\b(?:AIza|ya29\.)[A-Za-z0-9_\-.]+\b"),
)
_DYNAMIC_PATTERNS = (
    (
        re.compile(
            r"\b(?:pid|process|worker_pid)\s*[=: ]\s*\d+",
            re.IGNORECASE,
        ),
        "pid N",
    ),
    (
        re.compile(
            r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?\b",
            re.IGNORECASE,
        ),
        "<TIMESTAMP>",
    ),
    (re.compile(r"\b\d{10,}\b"), "<TS>"),
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f-]{27,}\b", re.IGNORECASE), "<UUID>"),
    (re.compile(r"\b0x[0-9a-f]+\b", re.IGNORECASE), "0xN"),
)


def redact_sensitive(value: object, *, extra_secrets: tuple[str, ...] = ()) -> str:
    """Redact credentials and bound the text kept in a durable receipt."""
    text = "" if value is None else str(value)
    for secret in extra_secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    text = _QUOTED_SECRET_PATTERN.sub(
        lambda match: match.group("prefix")
        + "[REDACTED]"
        + match.group("suffix"),
        text,
    )
    for pattern in _SECRET_PATTERNS:
        if pattern.groups:
            text = pattern.sub(lambda match: match.group(1) + "[REDACTED]", text)
        else:
            text = pattern.sub("[REDACTED]", text)
    text = text.replace("\x00", "")
    if len(text) > _MAX_ERROR_CHARS:
        text = text[:_MAX_ERROR_CHARS] + "… [truncated]"
    return text


def redact_error(value: object, *, extra_secrets: tuple[str, ...] = ()) -> str:
    """Compatibility alias for :func:`redact_sensitive`."""
    return redact_sensitive(value, extra_secrets=extra_secrets)


def _stable_error_text(value: object) -> str:
    text = redact_sensitive(value)
    for pattern, replacement in _DYNAMIC_PATTERNS:
        text = pattern.sub(replacement, text)
    # Retry-after values describe timing, not incident identity.
    text = re.sub(r"(?i)\bretry\s+after\s+\d+\s*(?:seconds?|s|minutes?|m)?", "retry after N", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def incident_fingerprint(
    error: object,
    *,
    failure_class: FailureClass | str = FailureClass.WORKER_EXIT_UNKNOWN,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    tool: Optional[str] = None,
) -> str:
    """Return a stable, bounded fingerprint for incident grouping."""
    cls_value = normalize_failure_class(failure_class)
    cls = (cls_value or FailureClass.WORKER_EXIT_UNKNOWN).value
    material = "|".join(
        (
            cls.casefold(),
            str(provider or "").casefold(),
            str(model or "").casefold(),
            str(tool or "").casefold(),
            _stable_error_text(error),
        )
    ).encode("utf-8", "replace")
    return hashlib.sha256(material).hexdigest()[:24]


def parse_retry_after(error: object) -> Optional[int]:
    """Extract a conservative retry delay from a provider error."""
    text = str(error or "")
    match = re.search(r"(?i)(?:retry[- ]after|retry\s+in|after)\s*[:=]?\s*(\d+)\s*(seconds?|s|minutes?|m)?", text)
    if not match:
        return None
    amount = int(match.group(1))
    if (match.group(2) or "").casefold().startswith("m"):
        amount *= 60
    return max(0, min(amount, 7 * 24 * 3600))


_GEMINI_PROVIDERS = frozenset(
    {"gemini", "google", "google-genai", "google_generative_ai"}
)
_RESEARCH_LANES = frozenset(
    {"research", "research_only", "researcher", "grounded_research", "nlm"}
)
_NON_GEMINI_AUTHORITY_LANES = frozenset(
    {"coding", "code", "operational", "ops", "review", "code_review", "authority", "trading", "deployment", "live"}
)
_FALLBACKABLE_FAILURES = frozenset(
    {
        FailureClass.PROVIDER_CAPACITY,
        FailureClass.PROVIDER_TRANSPORT,
        FailureClass.CONTEXT_LIMIT,
        FailureClass.TOOL_UNAVAILABLE,
    }
)


def provider_fallback_eligible(
    *,
    task_lane: Optional[str],
    candidate_provider: Optional[str],
    candidate_model: Optional[str] = None,
    required_capabilities: Iterable[str] = (),
    candidate_capabilities: Iterable[str] = (),
    healthy: bool = True,
    failure_class: FailureClass | str | None = None,
) -> bool:
    """Check whether a candidate route may replace a failed worker route.

    Fallback is deliberately a compatibility check, not a provider picker.
    The caller remains responsible for ordering and circuit health.  Gemini
    routes are admitted only for research-only lanes; all candidates must
    advertise every capability required by the task.  Authority-sensitive
    lanes may use a compatible non-Gemini route, but never a research model.
    """
    provider = str(candidate_provider or "").strip().casefold()
    model = str(candidate_model or "").strip().casefold()
    if not provider or not healthy:
        return False

    if failure_class is not None:
        failure = normalize_failure_class(failure_class)
        if failure is None or failure not in _FALLBACKABLE_FAILURES:
            return False

    required = {
        str(capability).strip().casefold()
        for capability in (required_capabilities or ())
        if str(capability).strip()
    }
    available = {
        str(capability).strip().casefold()
        for capability in (candidate_capabilities or ())
        if str(capability).strip()
    }
    if not required.issubset(available):
        return False

    lane = str(task_lane or "").strip().casefold()
    is_gemini = provider in _GEMINI_PROVIDERS or "gemini" in model
    if is_gemini and lane not in _RESEARCH_LANES:
        return False
    if lane in _NON_GEMINI_AUTHORITY_LANES and is_gemini:
        return False
    return True


# Descriptive alias for callers that prefer a predicate-style name.
is_fallback_eligible = provider_fallback_eligible


def make_exit_envelope(
    *,
    task_id: str,
    run_id: Optional[int] = None,
    session_id: Optional[str] = None,
    pid: Optional[int] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    exit_code: Optional[int] = None,
    failure_class: FailureClass | str | None = None,
    provider_status: Optional[int] = None,
    retry_after_seconds: Optional[int] = None,
    context_tokens: Optional[int] = None,
    messages: Optional[int] = None,
    tool_calls: Optional[int] = None,
    last_successful_checkpoint: Optional[str] = None,
    error: object = None,
    redacted_error: Optional[str] = None,
    stderr_tail: object = None,
    tool: Optional[str] = None,
) -> WorkerExitEnvelope:
    """Build a classified, redacted terminal envelope."""
    source_error = error if error is not None else stderr_tail
    classification = classify_failure(
        error=str(source_error or ""),
        exit_code=exit_code,
        provider_status=provider_status,
        failure_class=failure_class,
    )
    cls = classification.failure_class
    safe_error = (
        redact_error(redacted_error)
        if redacted_error is not None
        else (redact_error(source_error) if source_error is not None else None)
    )
    safe_tail = redact_error(stderr_tail) if stderr_tail is not None else None
    return WorkerExitEnvelope(
        task_id=str(task_id),
        run_id=run_id,
        session_id=session_id,
        pid=pid,
        provider=provider,
        model=model,
        exit_code=exit_code,
        failure_class=cls,
        provider_status=provider_status,
        retry_after_seconds=(
            retry_after_seconds
            if retry_after_seconds is not None
            else parse_retry_after(source_error)
        ),
        context_tokens=context_tokens,
        messages=messages,
        tool_calls=tool_calls,
        last_successful_checkpoint=last_successful_checkpoint,
        redacted_error=safe_error,
        error_fingerprint=incident_fingerprint(
            source_error or "",
            failure_class=cls,
            provider=provider,
            model=model,
            tool=tool,
        ),
        stderr_tail=safe_tail,
    )


def write_exit_envelope(path: str | os.PathLike[str], envelope: WorkerExitEnvelope | dict) -> Path:
    """Atomically persist an exit envelope and fsync the replacement."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(envelope, dict):
        envelope = WorkerExitEnvelope.from_dict(envelope)
    payload = json.dumps(envelope.to_dict(), ensure_ascii=False, sort_keys=True).encode("utf-8")
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
        if os.name != "nt":
            try:
                dir_fd = os.open(str(target.parent), os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                pass
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
    return target


def atomic_write_exit_envelope(path: str | os.PathLike[str], envelope: WorkerExitEnvelope | dict) -> Path:
    """Explicitly named alias for callers emphasizing atomicity."""
    return write_exit_envelope(path, envelope)


def read_exit_envelope(path: str | os.PathLike[str]) -> Optional[WorkerExitEnvelope]:
    """Read a complete envelope, returning ``None`` for absent/corrupt files."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return WorkerExitEnvelope.from_dict(raw)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def load_exit_envelope(path: str | os.PathLike[str]) -> Optional[WorkerExitEnvelope]:
    """Compatibility alias for :func:`read_exit_envelope`."""
    return read_exit_envelope(path)


def worker_exit_envelope_path(
    log_dir: str | os.PathLike[str], task_id: str, run_id: int,
) -> Path:
    """Return the run-scoped sidecar path used by launchers and reapers."""
    return Path(log_dir) / f"{task_id}.{int(run_id)}.exit.json"


def exit_envelope_path(
    log_dir: str | os.PathLike[str], task_id: str, run_id: int,
) -> Path:
    """Compatibility alias for :func:`worker_exit_envelope_path`."""
    return worker_exit_envelope_path(log_dir, task_id, run_id)


def _env_int(env: dict[str, str], name: str) -> Optional[int]:
    value = env.get(name)
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _tail_text(data: bytes, limit: int = _MAX_TAIL_CHARS) -> str:
    """Decode a bounded child-output tail without failing on bad bytes."""
    return data[-limit:].decode("utf-8", "replace")


def run_worker(
    command: list[str],
    *,
    envelope_path: str | os.PathLike[str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    env: Optional[dict[str, str]] = None,
) -> int:
    """Run a worker command and atomically write its terminal receipt.

    The command's combined output is streamed unchanged to this process'
    stdout (the dispatcher points that stream at the worker log), while a
    bounded copy is retained for the receipt.  Returning the child's status
    lets the dispatcher retain normal process semantics.
    """
    child_env = dict(os.environ if env is None else env)
    output_tail: deque[bytes] = deque()
    output_size = 0
    return_code: Optional[int] = None
    try:
        process = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd is not None else None,
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        while True:
            chunk = process.stdout.read(4096)
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            output_tail.append(chunk)
            output_size += len(chunk)
            while output_size > _MAX_TAIL_CHARS and output_tail:
                removed = output_tail.popleft()
                output_size -= len(removed)
        return_code = process.wait()
    except OSError as exc:
        return_code = 127
        message = f"worker launch failed: {exc}"
        output_tail.append(message.encode("utf-8", "replace"))
        try:
            sys.stderr.write(message + "\n")
        except OSError:
            pass

    combined_tail = b"".join(output_tail)
    safe_tail = _tail_text(combined_tail)
    provider_status = _env_int(child_env, "HERMES_KANBAN_PROVIDER_STATUS")
    envelope = make_exit_envelope(
        task_id=child_env.get("HERMES_KANBAN_TASK_ID", "unknown"),
        run_id=_env_int(child_env, "HERMES_KANBAN_RUN_ID"),
        session_id=child_env.get("HERMES_KANBAN_SESSION_ID"),
        pid=os.getpid(),
        provider=child_env.get("HERMES_KANBAN_PROVIDER"),
        model=child_env.get("HERMES_KANBAN_MODEL"),
        exit_code=return_code,
        provider_status=provider_status,
        context_tokens=_env_int(child_env, "HERMES_KANBAN_CONTEXT_TOKENS"),
        messages=_env_int(child_env, "HERMES_KANBAN_MESSAGES"),
        tool_calls=_env_int(child_env, "HERMES_KANBAN_TOOL_CALLS"),
        last_successful_checkpoint=child_env.get("HERMES_KANBAN_LAST_CHECKPOINT"),
        error=safe_tail if return_code else None,
        stderr_tail=safe_tail if safe_tail else None,
    )
    target = envelope_path or child_env.get("HERMES_KANBAN_EXIT_ENVELOPE")
    if target:
        write_exit_envelope(target, envelope)
    return int(return_code if return_code is not None else 1)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point used by the Kanban launcher wrapper."""
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        separator = args.index("--")
        command = args[separator + 1 :]
    except ValueError:
        if args and args[0] == "--worker":
            command = args[1:]
        else:
            command = args
    if not command:
        return 64
    return run_worker(command)


def classify_failure(
    *,
    error: Optional[str] = None,
    exit_code: Optional[int] = None,
    provider_status: Optional[int] = None,
    failure_class: FailureClass | str | None = None,
) -> FailureClassification:
    """Classify a worker failure using the strongest available signal."""
    if failure_class is not None:
        explicit = normalize_failure_class(failure_class)
        if explicit is not None:
            return FailureClassification(
                explicit,
                explicit is FailureClass.TASK_LOGIC,
                {
                    FailureClass.TASK_LOGIC: "task",
                    FailureClass.WORKER_PROTOCOL: "profile",
                    FailureClass.CONTEXT_LIMIT: "profile_model",
                    FailureClass.PROVIDER_CAPACITY: "provider_model",
                    FailureClass.PROVIDER_TRANSPORT: "provider_model",
                    FailureClass.PROVIDER_AUTH: "credentials",
                    FailureClass.TOOL_UNAVAILABLE: "tool_capability",
                    FailureClass.WORKSPACE_PROVISIONING: "workspace_profile",
                }.get(explicit, "worker"),
            )
        return FailureClassification(FailureClass.WORKER_EXIT_UNKNOWN, False, "worker")
    # A wrapper that exits zero has completed its command successfully. Its
    # normal transcript may mention workspace paths, missing optional
    # providers, or other failure-marker words; those are not terminal
    # failures unless the provider supplied an explicit status or caller
    # supplied an explicit failure class.
    if exit_code == 0 and provider_status is None:
        return FailureClassification(FailureClass.SUCCESS, False, "none")
    text = (error or "").casefold()
    if provider_status in {401, 403} or any(
        marker in text
        for marker in (
            "invalid api key",
            "invalid_api_key",
            "authentication failed",
            "unauthorized",
            "forbidden",
            "api key is invalid",
        )
    ):
        return FailureClassification(FailureClass.PROVIDER_AUTH, False, "credentials")
    if provider_status == 429 or any(
        marker in text
        for marker in (
            "resourceexhausted",
            "rate limit",
            "rate_limit",
            "too many requests",
            "quota",
            "request limit",
        )
    ):
        return FailureClassification(
            FailureClass.PROVIDER_CAPACITY, False, "provider_model"
        )
    if provider_status in {408, 409, 425, 500, 502, 503, 504} or any(
        marker in text
        for marker in (
            "apiconnectionerror",
            "connection refused",
            "connection reset",
            "idle timeout",
            "read timeout",
            "gateway timeout",
            "service unavailable",
        )
    ):
        return FailureClassification(
            FailureClass.PROVIDER_TRANSPORT, False, "provider_model"
        )
    if any(
        marker in text
        for marker in (
            "context length",
            "context window",
            "maximum context",
            "too many tokens",
            "max_tokens",
            "output token limit",
        )
    ):
        return FailureClassification(FailureClass.CONTEXT_LIMIT, False, "profile_model")
    if any(
        marker in text
        for marker in (
            "tool unavailable",
            "tool not found",
            "missing tool",
            "browser connection",
            "extractor unavailable",
            "capability unavailable",
        )
    ):
        return FailureClassification(FailureClass.TOOL_UNAVAILABLE, False, "tool_capability")
    has_workspace_marker = any(
        marker in text
        for marker in (
            "workspace",
            "worktree",
            "provisioning",
        )
    )
    has_workspace_failure = any(
        marker in text
        for marker in (
            "failed",
            "error",
            "not found",
            "missing",
            "permission denied",
            "access is denied",
            "file exists",
            "invalid path",
        )
    )
    if has_workspace_marker and has_workspace_failure:
        return FailureClassification(
            FailureClass.WORKSPACE_PROVISIONING, False, "workspace_profile"
        )
    if "protocol violation" in text or "kanban_complete" in text and "without" in text:
        return FailureClassification(FailureClass.WORKER_PROTOCOL, False, "profile")

    if exit_code is not None and exit_code < 0:
        return FailureClassification(FailureClass.WORKER_EXIT_UNKNOWN, False, "worker")

    if exit_code is None:
        return FailureClassification(FailureClass.WORKER_EXIT_UNKNOWN, False, "worker")
    return FailureClassification(FailureClass.TASK_LOGIC, True, "task")


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    raise SystemExit(main())
