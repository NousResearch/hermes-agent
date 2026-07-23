from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from agent.redact import redact_sensitive_text


SAFE_EXECUTABLES = frozenset({"claude", "codex"})
SAFE_OUTPUT_MODES = frozenset({"auto", "structured"})
SAFE_TIMEOUT_MODES = frozenset({"task_budget"})
SAFE_AUTH_MODES = frozenset({"cli_managed_subscription"})
SAFE_BLOCK_KINDS = frozenset({"needs_input", "capability", "transient", "dependency"})
SAFE_METADATA_KEYS = frozenset({"changed_files", "tests_run", "evidence"})
SAFE_ENV_NAMES = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
    "XDG_CACHE_HOME",
    "TMPDIR",
)
BLOCKED_ENV_NAMES = frozenset(
    {
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SESSION_TOKEN",
        "AZURE_OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "XAI_API_KEY",
        "COHERE_API_KEY",
        "MISTRAL_API_KEY",
        "DEEPSEEK_API_KEY",
        "DASHSCOPE_API_KEY",
        "MINIMAX_API_KEY",
        "KIMI_API_KEY",
        "QWEN_API_KEY",
        "CODEX_ACCESS_TOKEN",
    }
)
FORBIDDEN_EXTERNAL_CLI_FIELDS = frozenset(
    {
        "api_key",
        "api_key_env",
        "openrouter_api_key",
        "anthropic_api_key",
        "openai_api_key",
        "credential_fingerprint",
        "resume",
        "session_id",
        "fallback",
        "fallback_model",
        "fallback_provider",
    }
)
MAX_CAPTURE_BYTES = 256 * 1024
MAX_PROMPT_BYTES = 1024 * 1024
SUMMARY_CHAR_LIMIT = 2000
METADATA_ITEM_LIMIT = 500
METADATA_LIST_LIMIT = 100
METADATA_SERIALIZED_LIMIT = 64 * 1024
MAX_STRUCTURED_MESSAGE_BYTES = 1024 * 1024
MAX_UNTERMINATED_JSONL_LINE_BYTES = 2 * MAX_STRUCTURED_MESSAGE_BYTES
STRUCTURED_RESULT_KEY = "hermes_external_cli_result"


@dataclass(frozen=True)
class ExternalCliWorkerConfig:
    execution_backend: str = "hermes_internal"
    executable: str = ""
    args: tuple[str, ...] = ()
    authentication_mode: str = ""
    output_mode: str = "auto"
    timeout_mode: str = "task_budget"
    allow_resume: bool = False


@dataclass(frozen=True)
class ExternalCliExecutionRequest:
    task_id: str
    profile_name: str
    prompt: str
    workspace_path: str
    timeout_seconds: Optional[int]
    evidence_dir: str
    model: Optional[str] = None
    cancellation_requested: Optional[Callable[[], bool]] = None


@dataclass(frozen=True)
class ExternalCliStructuredPayload:
    action: str
    summary: Optional[str] = None
    metadata: Optional[dict] = None
    reason: Optional[str] = None
    block_kind: Optional[str] = None


@dataclass(frozen=True)
class ExternalCliExecutionResult:
    status: str
    executable: str
    argv_summary: str
    cwd: str
    structured_output_status: str
    stdout_artifact_path: Optional[str]
    stdout_sha256: Optional[str]
    stdout_size: int
    stderr_artifact_path: Optional[str]
    stderr_sha256: Optional[str]
    stderr_size: int
    stdout_summary: str
    stderr_signature: str
    stdout_total_size: int = 0
    stdout_truncated: bool = False
    stderr_total_size: int = 0
    stderr_truncated: bool = False
    exit_code: Optional[int] = None
    signal: Optional[int] = None
    timed_out: bool = False
    cancelled: bool = False
    structured_payload: Optional[ExternalCliStructuredPayload] = None

    def as_metadata(self) -> Dict[str, Any]:
        payload = asdict(self)
        if self.structured_payload is not None:
            payload["structured_payload"] = asdict(self.structured_payload)
        return payload


def _sanitize_text(value: Any, *, limit: int = SUMMARY_CHAR_LIMIT) -> str:
    text = redact_sensitive_text(str(value or ""), force=True)
    text = "".join(ch if ch >= " " or ch in "\t\n" else " " for ch in text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _sanitize_path_fragment(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    cleaned = cleaned.strip(".-")
    return cleaned[:80] or "artifact"


# changed_files/evidence are model-controlled and land in canonical task
# metadata; without this, absolute paths, '..' traversal, and local
# filesystem layout (e.g. /home/<user>/...) could leak into the Kanban
# record (P0-F m1). tests_run is free-form description text, not a path, so
# it only gets the existing length/redaction bounds.
_PATH_LIKE_METADATA_KEYS = frozenset({"changed_files", "evidence"})
_DRIVE_LETTER_PATH_RE = re.compile(r"^[A-Za-z]:[/\\]")


def _normalize_relative_metadata_path(value: str) -> Optional[str]:
    """Accept only a clean, workspace-relative path: no absolute paths (POSIX
    or Windows drive-letter), no home-dir shorthand, no '..' traversal."""
    if "\x00" in value:
        return None
    candidate = value.strip().replace("\\", "/")
    if not candidate or candidate.startswith("/") or candidate.startswith("~"):
        return None
    if _DRIVE_LETTER_PATH_RE.match(value.strip()):
        return None
    parts = [part for part in candidate.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        return None
    return "/".join(parts)


def _sanitize_metadata(value: Any) -> dict[str, list[str]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("metadata must be an object")
    result: dict[str, list[str]] = {}
    for key in SAFE_METADATA_KEYS:
        raw_items = value.get(key)
        if raw_items is None:
            continue
        if not isinstance(raw_items, list):
            raise ValueError(f"metadata.{key} must be a list")
        if len(raw_items) > METADATA_LIST_LIMIT:
            raise ValueError(f"metadata.{key} exceeds item limit")
        items: list[str] = []
        for item in raw_items:
            if not isinstance(item, str):
                raise ValueError(f"metadata.{key} entries must be strings")
            sanitized = _sanitize_text(item, limit=METADATA_ITEM_LIMIT)
            if not sanitized:
                continue
            if key in _PATH_LIKE_METADATA_KEYS:
                sanitized = _normalize_relative_metadata_path(sanitized)
                if not sanitized:
                    continue
            items.append(sanitized)
        result[key] = items
    if len(json.dumps(result, ensure_ascii=False).encode("utf-8")) > METADATA_SERIALIZED_LIMIT:
        raise ValueError("metadata exceeds serialized size limit")
    return result


def sanitize_external_cli_payload(payload: ExternalCliStructuredPayload) -> ExternalCliStructuredPayload:
    action = str(payload.action or "").strip().lower()
    if action not in {"complete", "block"}:
        raise ValueError("action must be complete or block")
    block_kind = str(payload.block_kind or "").strip() or None
    if block_kind is not None and block_kind not in SAFE_BLOCK_KINDS:
        raise ValueError("block_kind is invalid")
    summary = _sanitize_text(payload.summary, limit=SUMMARY_CHAR_LIMIT) or None
    reason = _sanitize_text(payload.reason, limit=SUMMARY_CHAR_LIMIT) or None
    metadata = _sanitize_metadata(payload.metadata)
    if action == "complete" and not summary:
        summary = "Completed by external CLI worker"
    if action == "block" and not (reason or summary):
        reason = "External CLI worker reported a blocked task"
    return ExternalCliStructuredPayload(
        action=action,
        summary=summary,
        metadata=metadata,
        reason=reason,
        block_kind=block_kind,
    )


ALLOWED_EXTERNAL_CLI_FIELDS = frozenset(
    {"executable", "args", "authentication_mode", "output_mode", "timeout_mode", "allow_resume"}
)


def _default_worker_config() -> Dict[str, Any]:
    return {
        "execution_backend": "hermes_internal",
        "external_cli": {
            "executable": "",
            "args": [],
            "authentication_mode": "",
            "output_mode": "auto",
            "timeout_mode": "task_budget",
            "allow_resume": False,
        },
    }


def worker_config_defaults() -> Dict[str, Any]:
    return json.loads(json.dumps(_default_worker_config()))


def load_external_cli_worker_config(config: Optional[Dict[str, Any]]) -> ExternalCliWorkerConfig:
    worker_cfg: dict[str, Any] = {}
    if isinstance(config, dict):
        raw = config.get("worker")
        if isinstance(raw, dict):
            worker_cfg = raw
    defaults = _default_worker_config()
    execution_backend = str(worker_cfg.get("execution_backend", defaults["execution_backend"]) or "").strip() or "hermes_internal"
    if execution_backend not in {"hermes_internal", "external_cli"}:
        raise ValueError(f"worker.execution_backend must be one of ['external_cli', 'hermes_internal'], got {execution_backend!r}")

    raw_external = worker_cfg.get("external_cli", defaults["external_cli"])
    if raw_external is None:
        raw_external = {}
    if not isinstance(raw_external, dict):
        raise ValueError("worker.external_cli must be a mapping when set")

    forbidden = sorted(FORBIDDEN_EXTERNAL_CLI_FIELDS & set(raw_external.keys()))
    if forbidden:
        raise ValueError(f"worker.external_cli contains forbidden fields: {', '.join(forbidden)}")
    unknown = sorted(set(raw_external.keys()) - ALLOWED_EXTERNAL_CLI_FIELDS - FORBIDDEN_EXTERNAL_CLI_FIELDS)
    if unknown:
        raise ValueError(f"worker.external_cli contains unknown fields: {', '.join(unknown)}")

    executable = str(raw_external.get("executable", "") or "").strip()
    args = raw_external.get("args", [])
    if args is None:
        args = []
    if not isinstance(args, list) or any(not isinstance(item, str) for item in args):
        raise ValueError("worker.external_cli.args must be a list of strings")
    if args:
        raise ValueError("worker.external_cli.args must be empty in external CLI adapter v1")

    authentication_mode = str(raw_external.get("authentication_mode", "") or "").strip()
    output_mode = str(raw_external.get("output_mode", "auto") or "auto").strip().lower()
    timeout_mode = str(raw_external.get("timeout_mode", "task_budget") or "task_budget").strip().lower()
    allow_resume = bool(raw_external.get("allow_resume", False))

    if output_mode not in SAFE_OUTPUT_MODES:
        raise ValueError(f"worker.external_cli.output_mode must be one of {sorted(SAFE_OUTPUT_MODES)}")
    if timeout_mode not in SAFE_TIMEOUT_MODES:
        raise ValueError(f"worker.external_cli.timeout_mode must be one of {sorted(SAFE_TIMEOUT_MODES)}")
    if allow_resume:
        raise ValueError("worker.external_cli.allow_resume=true is not supported")

    if execution_backend == "external_cli":
        if executable not in SAFE_EXECUTABLES:
            raise ValueError("worker.external_cli.executable must be 'claude' or 'codex'")
        if authentication_mode not in SAFE_AUTH_MODES:
            raise ValueError("worker.external_cli.authentication_mode must be 'cli_managed_subscription'")
    else:
        executable = ""
        authentication_mode = ""

    return ExternalCliWorkerConfig(
        execution_backend=execution_backend,
        executable=executable,
        args=(),
        authentication_mode=authentication_mode,
        output_mode=output_mode,
        timeout_mode=timeout_mode,
        allow_resume=False,
    )


def validate_external_cli_worker_config(config: Optional[Dict[str, Any]]) -> list[str]:
    try:
        load_external_cli_worker_config(config)
    except ValueError as exc:
        return [str(exc)]
    return []


def _build_subprocess_env(cfg: ExternalCliWorkerConfig) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for name in SAFE_ENV_NAMES:
        value = os.environ.get(name)
        if value and name not in BLOCKED_ENV_NAMES:
            env[name] = value
    return env


def _build_worker_prompt(req: ExternalCliExecutionRequest) -> str:
    return (
        "You are executing one Hermes kanban task inside a pinned workspace.\n"
        "Do the work in the workspace and return ONLY one JSON object matching this contract:\n"
        "{\n"
        '  "hermes_external_cli_result": {\n'
        '    "action": "complete" | "block",\n'
        '    "summary": "short summary for complete",\n'
        '    "metadata": {"changed_files": [], "tests_run": [], "evidence": []},\n'
        '    "reason": "blocking reason for block",\n'
        '    "block_kind": "needs_input" | "capability" | "transient" | "dependency"\n'
        "  }\n"
        "}\n"
        "Do not output prose before or after the JSON object.\n"
        f"Task ID: {req.task_id}\n"
        f"Profile: {req.profile_name}\n"
        f"Workspace: {req.workspace_path}\n"
        "\nTask:\n"
        f"{req.prompt.strip()}\n"
    )


def _parse_contract_text(text: str) -> Optional[ExternalCliStructuredPayload]:
    try:
        candidate = json.loads(text.strip())
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(candidate, dict) or set(candidate) != {STRUCTURED_RESULT_KEY}:
        return None
    raw = candidate.get(STRUCTURED_RESULT_KEY)
    if not isinstance(raw, dict):
        return None
    allowed = {"action", "summary", "metadata", "reason", "block_kind"}
    if not set(raw).issubset(allowed):
        return None
    try:
        return sanitize_external_cli_payload(
            ExternalCliStructuredPayload(
                action=raw.get("action"),
                summary=raw.get("summary"),
                metadata=raw.get("metadata"),
                reason=raw.get("reason"),
                block_kind=raw.get("block_kind"),
            )
        )
    except (TypeError, ValueError):
        return None


_AUTH_FAILURE_SIGNATURES = (
    "not logged in", "authentication", "auth required", "sign in",
    "login required", "subscription required", "setup-token",
)
_QUOTA_FAILURE_SIGNATURES = (
    "quota", "rate limit", "too many requests", "billing", "usage cap",
    "credit balance", "exhausted",
)
# Multi-word, denial-specific phrases only. A bare "sandbox" or "approval"
# token also appears in unrelated runtime-initialization failures (e.g. "no
# sandbox helper found", "approval config invalid") and would misclassify
# those as an actionable permission block rather than an internal failure
# (P0-F m4).
_PERMISSION_FAILURE_SIGNATURES = (
    "permission denied", "not trusted", "approval required", "access denied",
    "denied by sandbox", "blocked by sandbox", "rejected by sandbox policy",
    "operation not permitted",
)
_INVALID_INVOCATION_SIGNATURES = (
    "invalid_request_error",
    "invalid_json_schema",
    "invalid schema for response_format",
)


def _classify_failure(output_text: str, exit_code: Optional[int], *, timed_out: bool, cancelled: bool) -> str:
    if cancelled:
        return "FAILED_CANCELLED"
    if timed_out:
        return "FAILED_TIMEOUT"
    lowered = f"{output_text}\nexit={exit_code}".lower()
    if any(token in lowered for token in _AUTH_FAILURE_SIGNATURES):
        return "BLOCKED_AUTH"
    if any(token in lowered for token in _QUOTA_FAILURE_SIGNATURES):
        return "BLOCKED_QUOTA"
    if any(token in lowered for token in _PERMISSION_FAILURE_SIGNATURES):
        return "BLOCKED_PERMISSION"
    if any(token in lowered for token in _INVALID_INVOCATION_SIGNATURES):
        return "FAILED_INVALID_INVOCATION"
    if exit_code == 127:
        return "FAILED_EXECUTABLE_MISSING"
    if exit_code == 2:
        return "FAILED_INVALID_INVOCATION"
    return "FAILED_INTERNAL"


def _ensure_secure_directory(path: Path) -> None:
    if path.exists() and path.is_symlink():
        raise OSError("evidence directory may not be a symlink")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)


def _write_secure_file(path: Path, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)


def _write_artifact(base_dir: Path, stem: str, content: bytes) -> tuple[Optional[str], Optional[str], int]:
    if not content:
        return None, None, 0
    _ensure_secure_directory(base_dir)
    path = base_dir / f"{_sanitize_path_fragment(stem)}-{secrets.token_hex(6)}.log"
    _write_secure_file(path, content)
    return str(path), sha256(content).hexdigest(), len(content)


class _StreamCapture:
    def __init__(self) -> None:
        self.total_size = 0
        self.truncated = False
        self._captured_size = 0
        self._chunks: list[bytes] = []
        self._lock = threading.Lock()

    def feed(self, chunk: bytes) -> None:
        if not chunk:
            return
        with self._lock:
            self.total_size += len(chunk)
            if self._captured_size >= MAX_CAPTURE_BYTES:
                self.truncated = True
                return
            remaining = MAX_CAPTURE_BYTES - self._captured_size
            captured = chunk[:remaining]
            if captured:
                self._chunks.append(captured)
                self._captured_size += len(captured)
            if len(chunk) > len(captured):
                self.truncated = True

    def content(self) -> bytes:
        with self._lock:
            return b"".join(self._chunks)


class _ActiveExternalCliRegistry:
    """Tracks whether this process currently owns a live external CLI child,
    so a SIGTERM handler can defer immediate process exit until the child has
    been terminated and reaped instead of orphaning it.

    Deliberately lock-free: a Python signal handler runs synchronously on the
    main thread's eval loop, so acquiring a lock that the interrupted code
    already holds would deadlock the process. Plain attribute reads/writes
    are GIL-atomic, which is all the cross-context safety this needs.
    """

    def __init__(self) -> None:
        self._active = False
        self._cancel_requested = False

    def activate(self) -> None:
        self._cancel_requested = False
        self._active = True

    def deactivate(self) -> None:
        self._active = False

    def cancellation_requested(self) -> bool:
        return self._cancel_requested

    def request_cancel(self) -> bool:
        """Call from a signal handler. Returns True if a child is active and
        will observe the cancellation, so the caller should defer exit."""
        if not self._active:
            return False
        self._cancel_requested = True
        return True


active_external_cli_registry = _ActiveExternalCliRegistry()


class _CodexIncrementalParser:
    """Streams Codex `exec --json` JSONL without holding the full transcript
    in memory. Only terminal state and the most recent agent message text
    are retained, so a valid multi-megabyte transcript still yields the
    final result instead of being silently truncated by a bounded capture
    buffer that only kept the first 256 KiB (P0-F M3). The raw stdout is
    still captured separately (and still size-capped) purely as an evidence
    artifact; this parser is the only thing that decides the result.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()
        self.line_count = 0
        self.total_bytes = 0
        self.malformed_line_seen = False
        self.fatal_error_seen = False
        self.terminal_failure_text: Optional[str] = None
        self.turn_completed_seen = False
        self.last_agent_message_text: Optional[str] = None

    def feed(self, chunk: bytes) -> None:
        if not chunk:
            return
        self.total_bytes += len(chunk)
        self._buffer.extend(chunk)
        while True:
            newline_index = self._buffer.find(b"\n")
            if newline_index < 0:
                break
            raw_line = bytes(self._buffer[:newline_index])
            del self._buffer[: newline_index + 1]
            self._consume_line(raw_line)
        if len(self._buffer) > MAX_UNTERMINATED_JSONL_LINE_BYTES:
            # A single line growing unbounded without a newline can't be a
            # valid bounded contract; drop it rather than buffering forever.
            self.malformed_line_seen = True
            self._buffer.clear()

    def finalize(self) -> None:
        if self._buffer.strip():
            self._consume_line(bytes(self._buffer))
        self._buffer.clear()

    def _consume_line(self, raw_line: bytes) -> None:
        line = raw_line.strip()
        if not line:
            return
        self.line_count += 1
        try:
            event = json.loads(line.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.malformed_line_seen = True
            return
        if not isinstance(event, dict):
            self.malformed_line_seen = True
            return
        event_type = event.get("type")
        if event_type in {"error", "turn.failed"}:
            self.fatal_error_seen = True
            raw_error = event.get("error")
            if isinstance(raw_error, dict):
                raw_error = raw_error.get("message")
            if not isinstance(raw_error, str):
                raw_error = event.get("message")
            if isinstance(raw_error, str) and raw_error.strip():
                # Prefer turn.failed: it is Codex's terminal result and is
                # more authoritative than an earlier streaming error event.
                if event_type == "turn.failed" or self.terminal_failure_text is None:
                    self.terminal_failure_text = raw_error
            return
        if event_type == "turn.completed":
            self.turn_completed_seen = True
            return
        if self.turn_completed_seen:
            return  # ignore any trailing noise after the terminal event
        if event_type == "item.completed":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                text = item["text"]
                if len(text.encode("utf-8", errors="ignore")) <= MAX_STRUCTURED_MESSAGE_BYTES:
                    self.last_agent_message_text = text
                else:
                    # Oversized single message can't be a valid bounded
                    # contract anyway; don't retain it.
                    self.last_agent_message_text = None

    def result(self) -> Optional[ExternalCliStructuredPayload]:
        self.finalize()
        if self.malformed_line_seen or self.fatal_error_seen:
            return None
        if not self.turn_completed_seen or self.last_agent_message_text is None:
            return None
        return _parse_contract_text(self.last_agent_message_text)


class _BaseCliStrategy:
    name = ""

    def build_argv(self, cfg: ExternalCliWorkerConfig, req: ExternalCliExecutionRequest) -> list[str]:
        raise NotImplementedError

    def build_prompt(self, req: ExternalCliExecutionRequest) -> str:
        return _build_worker_prompt(req)

    def parse_output(self, stdout_text: str) -> Optional[ExternalCliStructuredPayload]:
        raise NotImplementedError

    def create_incremental_parser(self) -> Optional["_CodexIncrementalParser"]:
        """Vendors whose terminal result can be identified while streaming
        (rather than only after the fact from a bounded buffer) return a
        parser here; the adapter feeds it chunks as they arrive and prefers
        its result over parse_output()'s post-hoc, capture-buffer-based one."""
        return None


class ClaudeCliStrategy(_BaseCliStrategy):
    name = "claude"

    def build_argv(self, cfg: ExternalCliWorkerConfig, req: ExternalCliExecutionRequest) -> list[str]:
        argv = [cfg.executable, "-p", "--output-format", "json"]
        if req.model:
            argv.extend(["--model", req.model])
        return argv

    def parse_output(self, stdout_text: str) -> Optional[ExternalCliStructuredPayload]:
        try:
            envelope = json.loads(stdout_text.strip())
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(envelope, dict):
            return None
        if envelope.get("type") != "result" or envelope.get("subtype") != "success":
            return None
        if envelope.get("is_error") is not False:
            return None
        result_text = envelope.get("result")
        if not isinstance(result_text, str):
            return None
        return _parse_contract_text(result_text)


class CodexCliStrategy(_BaseCliStrategy):
    name = "codex"

    @staticmethod
    def _schema_bytes() -> bytes:
        string_list = {
            "type": "array",
            "items": {"type": "string", "maxLength": METADATA_ITEM_LIMIT},
            "maxItems": METADATA_LIST_LIMIT,
        }
        schema = {
            "type": "object",
            "properties": {
                STRUCTURED_RESULT_KEY: {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["complete", "block"]},
                        "summary": {
                            "type": ["string", "null"],
                            "maxLength": SUMMARY_CHAR_LIMIT,
                        },
                        "metadata": {
                            "type": "object",
                            "properties": {key: string_list for key in sorted(SAFE_METADATA_KEYS)},
                            "required": sorted(SAFE_METADATA_KEYS),
                            "additionalProperties": False,
                        },
                        "reason": {
                            "type": ["string", "null"],
                            "maxLength": SUMMARY_CHAR_LIMIT,
                        },
                        "block_kind": {
                            "type": ["string", "null"],
                            "enum": [*sorted(SAFE_BLOCK_KINDS), None],
                        },
                    },
                    # Codex strict response formats require every property at
                    # every object level to be named in required. Nullable
                    # fields preserve the complete/block union semantics.
                    "required": [
                        "action",
                        "summary",
                        "metadata",
                        "reason",
                        "block_kind",
                    ],
                    "additionalProperties": False,
                }
            },
            "required": [STRUCTURED_RESULT_KEY],
            "additionalProperties": False,
        }
        return json.dumps(schema, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    def build_argv(self, cfg: ExternalCliWorkerConfig, req: ExternalCliExecutionRequest) -> list[str]:
        schema_dir = Path(req.evidence_dir)
        _ensure_secure_directory(schema_dir)
        schema_path = schema_dir / f"codex-output-schema-{secrets.token_hex(6)}.json"
        _write_secure_file(schema_path, self._schema_bytes())
        workspace = str(Path(req.workspace_path))
        argv = [
            cfg.executable,
            "exec",
            # Explicit execution-policy boundary (P0-F M4): a pinned Popen
            # cwd is not itself a security boundary, and codex's effective
            # sandbox authority can otherwise come from ambient
            # $CODEX_HOME config or profiles instead of this invocation.
            "--sandbox", "workspace-write",
            "--cd", workspace,
            "--ignore-user-config",
            "--ignore-rules",
            "--ephemeral",
            "--skip-git-repo-check",
            "--json",
            "--output-schema", str(schema_path),
        ]
        if req.model:
            argv.extend(["--model", req.model])
        return argv

    def parse_output(self, stdout_text: str) -> Optional[ExternalCliStructuredPayload]:
        parser = self.create_incremental_parser()
        parser.feed(stdout_text.encode("utf-8", errors="replace"))
        return parser.result()

    def create_incremental_parser(self) -> _CodexIncrementalParser:
        return _CodexIncrementalParser()


class ExternalCliAgentAdapter:
    def __init__(self) -> None:
        self._strategies = {"claude": ClaudeCliStrategy(), "codex": CodexCliStrategy()}

    @staticmethod
    def _empty_result(
        cfg: ExternalCliWorkerConfig,
        req: ExternalCliExecutionRequest,
        *,
        status: str,
        structured_output_status: str,
        stderr_signature: str,
        exit_code: Optional[int] = None,
        signal_number: Optional[int] = None,
        timed_out: bool = False,
        cancelled: bool = False,
        argv_summary: Optional[str] = None,
    ) -> ExternalCliExecutionResult:
        return ExternalCliExecutionResult(
            status=status,
            executable=cfg.executable,
            argv_summary=argv_summary or json.dumps([cfg.executable]),
            cwd=req.workspace_path,
            structured_output_status=structured_output_status,
            stdout_artifact_path=None,
            stdout_sha256=None,
            stdout_size=0,
            stderr_artifact_path=None,
            stderr_sha256=None,
            stderr_size=0,
            stdout_summary="",
            stderr_signature=_sanitize_text(stderr_signature),
            stdout_total_size=0,
            stdout_truncated=False,
            stderr_total_size=0,
            stderr_truncated=False,
            exit_code=exit_code,
            signal=signal_number,
            timed_out=timed_out,
            cancelled=cancelled,
        )

    def run(self, cfg: ExternalCliWorkerConfig, req: ExternalCliExecutionRequest) -> ExternalCliExecutionResult:
        strategy = self._strategies.get(cfg.executable)
        if strategy is None:
            return self._empty_result(cfg, req, status="FAILED_INVALID_INVOCATION", structured_output_status="invalid_strategy", stderr_signature="unsupported executable", exit_code=2)

        workspace = Path(req.workspace_path)
        if not workspace.is_absolute() or not workspace.is_dir():
            return self._empty_result(cfg, req, status="FAILED_INVALID_INVOCATION", structured_output_status="invalid_cwd", stderr_signature="workspace path is not an existing absolute directory", exit_code=2)

        prompt_bytes = strategy.build_prompt(req).encode("utf-8")
        if len(prompt_bytes) > MAX_PROMPT_BYTES:
            return self._empty_result(cfg, req, status="FAILED_INVALID_INVOCATION", structured_output_status="prompt_too_large", stderr_signature="prompt exceeds size limit", exit_code=2)

        executable_path = shutil.which(cfg.executable)
        if not executable_path:
            return self._empty_result(cfg, req, status="FAILED_EXECUTABLE_MISSING", structured_output_status="missing_executable", stderr_signature=f"{cfg.executable} not found on PATH", exit_code=127)

        try:
            _ensure_secure_directory(Path(req.evidence_dir))
            argv = strategy.build_argv(cfg, req)
        except (OSError, ValueError) as exc:
            return self._empty_result(cfg, req, status="FAILED_INTERNAL", structured_output_status="artifact_setup_failed", stderr_signature=f"artifact setup failed: {type(exc).__name__}")

        argv_summary = json.dumps([Path(argv[0]).name, *["<path>" if str(arg).startswith(req.evidence_dir) else str(arg) for arg in argv[1:]]])
        env = _build_subprocess_env(cfg)
        try:
            proc = subprocess.Popen(
                argv,
                cwd=str(workspace),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                start_new_session=True,
            )
        except FileNotFoundError:
            return self._empty_result(cfg, req, status="FAILED_EXECUTABLE_MISSING", structured_output_status="launch_failed", stderr_signature="executable not found", exit_code=127, argv_summary=argv_summary)
        except PermissionError:
            return self._empty_result(cfg, req, status="BLOCKED_PERMISSION", structured_output_status="launch_failed", stderr_signature="permission denied while launching external CLI", exit_code=126, argv_summary=argv_summary)
        except OSError as exc:
            return self._empty_result(cfg, req, status="FAILED_INTERNAL", structured_output_status="launch_failed", stderr_signature=f"launch failed: {type(exc).__name__}", argv_summary=argv_summary)

        stdout_capture = _StreamCapture()
        stderr_capture = _StreamCapture()
        io_errors: list[str] = []
        io_lock = threading.Lock()

        def _record_error(label: str, exc: BaseException) -> None:
            with io_lock:
                io_errors.append(f"{label}:{type(exc).__name__}")

        incremental_parser = strategy.create_incremental_parser()

        def _reader(stream, capture: _StreamCapture, label: str, incremental=None) -> None:
            if stream is None:
                _record_error(label, RuntimeError("missing stream"))
                return
            try:
                while True:
                    chunk = stream.read(65536)
                    if not chunk:
                        break
                    capture.feed(chunk)
                    if incremental is not None:
                        incremental.feed(chunk)
            except Exception as exc:  # normalized below
                _record_error(label, exc)
            finally:
                try:
                    stream.close()
                except Exception:
                    pass

        def _writer() -> None:
            if proc.stdin is None:
                _record_error("stdin", RuntimeError("missing stdin"))
                return
            try:
                proc.stdin.write(prompt_bytes)
                proc.stdin.flush()
            except Exception as exc:  # normalized below
                _record_error("stdin", exc)
            finally:
                try:
                    proc.stdin.close()
                except Exception:
                    pass

        stdout_thread = threading.Thread(
            target=_reader,
            args=(proc.stdout, stdout_capture, "stdout"),
            kwargs={"incremental": incremental_parser},
            daemon=True,
        )
        stderr_thread = threading.Thread(target=_reader, args=(proc.stderr, stderr_capture, "stderr"), daemon=True)
        stdin_thread = threading.Thread(target=_writer, daemon=True)
        started_at = time.monotonic()
        stdout_thread.start()
        stderr_thread.start()
        stdin_thread.start()

        timed_out = False
        cancelled = False
        cleanup_failed = False
        deadline = None
        if req.timeout_seconds and req.timeout_seconds > 0:
            deadline = started_at + float(req.timeout_seconds)

        active_external_cli_registry.activate()
        writer_failed = False
        try:
            while proc.poll() is None:
                if req.cancellation_requested and req.cancellation_requested():
                    cancelled = True
                    self._terminate_process_tree(proc)
                    break
                if active_external_cli_registry.cancellation_requested():
                    cancelled = True
                    self._terminate_process_tree(proc)
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    timed_out = True
                    self._terminate_process_tree(proc)
                    break
                with io_lock:
                    writer_failed = any(item.startswith("stdin:") for item in io_errors)
                if writer_failed:
                    self._terminate_process_tree(proc)
                    break
                time.sleep(0.05)

            # A blocked writer thread can be holding the stdin BufferedWriter's
            # internal lock inside write(); closing stdin from this thread
            # first would block on that same lock and never reach termination.
            # Terminating the process group breaks the pipe (EPIPE), which
            # unblocks the writer, so stdin is only closed below once the
            # child has exited or the wait has been bounded by a force-kill.
            try:
                exit_code = proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                if not cancelled and not writer_failed:
                    timed_out = True
                self._terminate_process_tree(proc, force=True)
                try:
                    exit_code = proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    cleanup_failed = True
                    exit_code = None
        finally:
            active_external_cli_registry.deactivate()
            self._close_stream(proc.stdin)
            stdin_thread.join(timeout=2)
            stdout_thread.join(timeout=2)
            stderr_thread.join(timeout=2)

        stdout_bytes = stdout_capture.content()
        stderr_bytes = stderr_capture.content()
        try:
            stdout_path, stdout_sha, stdout_size = _write_artifact(Path(req.evidence_dir), f"{req.task_id}-stdout", stdout_bytes)
            stderr_path, stderr_sha, stderr_size = _write_artifact(Path(req.evidence_dir), f"{req.task_id}-stderr", stderr_bytes)
        except OSError as exc:
            return self._empty_result(cfg, req, status="FAILED_INTERNAL", structured_output_status="artifact_write_failed", stderr_signature=f"artifact write failed: {type(exc).__name__}", exit_code=exit_code, timed_out=timed_out, cancelled=cancelled, argv_summary=argv_summary)

        stdout_text = stdout_bytes.decode("utf-8", errors="replace")
        stderr_text = stderr_bytes.decode("utf-8", errors="replace")
        if incremental_parser is not None:
            incremental_parser.finalize()
        can_parse = exit_code == 0 and not timed_out and not cancelled and not cleanup_failed
        if not can_parse:
            structured_payload = None
        elif incremental_parser is not None:
            # Streamed while the process ran, so a valid result survives even
            # when the raw stdout capture above was truncated at
            # MAX_CAPTURE_BYTES (that capture is now evidence-only).
            structured_payload = incremental_parser.result()
        else:
            structured_payload = strategy.parse_output(stdout_text)
        signal_number = -exit_code if isinstance(exit_code, int) and exit_code < 0 else None

        with io_lock:
            captured_io_errors = tuple(io_errors)
        if cancelled:
            status = "FAILED_CANCELLED"
            structured_status = "cancelled"
        elif timed_out:
            status = "FAILED_TIMEOUT"
            structured_status = "timeout"
        elif cleanup_failed or captured_io_errors:
            status = "FAILED_INTERNAL"
            structured_status = "io_failure" if captured_io_errors else "cleanup_failed"
        elif exit_code == 0 and structured_payload is not None:
            status = "COMPLETED"
            structured_status = "parsed"
        elif exit_code == 0:
            status = "FAILED_MALFORMED_OUTPUT"
            structured_status = "missing"
        else:
            terminal_failure = (
                incremental_parser.terminal_failure_text
                if incremental_parser is not None
                else None
            )
            failure_text = terminal_failure or f"{stdout_text}\n{stderr_text}"
            status = _classify_failure(
                failure_text,
                exit_code,
                timed_out=timed_out,
                cancelled=cancelled,
            )
            structured_status = (
                "terminal_failure" if terminal_failure else "missing"
            )

        return ExternalCliExecutionResult(
            status=status,
            executable=cfg.executable,
            argv_summary=argv_summary,
            cwd=str(workspace),
            structured_output_status=structured_status,
            stdout_artifact_path=stdout_path,
            stdout_sha256=stdout_sha,
            stdout_size=stdout_size,
            stderr_artifact_path=stderr_path,
            stderr_sha256=stderr_sha,
            stderr_size=stderr_size,
            stdout_summary=_sanitize_text(stdout_text),
            stderr_signature=_sanitize_text(";".join(captured_io_errors) or stderr_text),
            stdout_total_size=stdout_capture.total_size,
            stdout_truncated=stdout_capture.truncated,
            stderr_total_size=stderr_capture.total_size,
            stderr_truncated=stderr_capture.truncated,
            exit_code=exit_code,
            signal=signal_number,
            timed_out=timed_out,
            cancelled=cancelled,
            structured_payload=structured_payload,
        )

    @staticmethod
    def _close_stream(stream: Any) -> None:
        try:
            if stream is not None:
                stream.close()
        except Exception:
            pass

    @staticmethod
    def _terminate_process_tree(proc: subprocess.Popen, force: bool = False) -> None:
        try:
            if os.name == "posix":
                sig = signal.SIGKILL if force else signal.SIGTERM
                os.killpg(os.getpgid(proc.pid), sig)
            elif force:
                proc.kill()
            else:
                proc.terminate()
        except Exception:
            try:
                if force:
                    proc.kill()
                else:
                    proc.terminate()
            except Exception:
                pass
