"""Sandboxed, output-only adapter for the Antigravity ``agy`` CLI.

This module intentionally exposes no Hermes tools or project directory.  Each
turn runs in an empty private workspace and receives only the caller-supplied
normalized goal/context through stdin.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable, IO, Mapping

from agent.gemini_routing_contract import validate_antigravity_extra_args


DEFAULT_MODEL = "gemini-3.8-flash-low"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_INPUT_BYTES = 262_144
DEFAULT_MAX_OUTPUT_BYTES = 131_072
_SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)
_EXTRA_ARGS_UNSET = object()
_UNSUPPORTED_SCHEMA_KEYWORDS = frozenset(
    {
        "$ref",
        "$dynamicRef",
        "$recursiveRef",
        "$defs",
        "definitions",
        "if",
        "then",
        "else",
        "dependentRequired",
        "dependentSchemas",
        "patternProperties",
        "propertyNames",
        "contains",
        "minContains",
        "maxContains",
        "prefixItems",
        "unevaluatedItems",
        "unevaluatedProperties",
        "format",
    }
)

# Deliberately not inherited: HERMES_*, ANTIGRAVITY_*, provider keys, messaging
# tokens, and arbitrary parent variables.  HOME/config locations are required
# for the user's existing subscription login; AGY_CLI_* values are CLI runtime
# controls rather than credentials.
_BASE_ENV_ALLOWLIST = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "TMPDIR",
        "TMP",
        "TEMP",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TERM",
        "COLORTERM",
        "NO_COLOR",
        "FORCE_COLOR",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "XDG_CONFIG_HOME",
        "XDG_CACHE_HOME",
        "AGY_CLI_HIDE_LOGO",
        "AGY_CLI_DISABLE_AUTO_UPDATE",
        "AGY_CLI_DISABLE_ESCAPE_SEQUENCE_OPTIMIZATIONS",
    }
)


@dataclass(frozen=True)
class AntigravityResult:
    status: str
    response: str | None
    conversation_id: str | None
    usage: dict[str, Any]
    raw_envelope: dict[str, Any] | None
    exit_code: int | None
    duration_ms: int
    error_code: str | None
    error_message: str | None
    output_excerpt: str | None = None
    output_sha256: str | None = None
    output_bytes: int | None = None


class _BoundedPipeReader:
    """Drain a pipe while retaining at most ``limit + 1`` bytes."""

    def __init__(self, pipe: IO[bytes], limit: int) -> None:
        self._pipe = pipe
        self._limit = limit
        self._data = bytearray()
        self._digest = hashlib.sha256()
        self.total_bytes = 0
        self.too_large = False
        self._thread = threading.Thread(target=self._read, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float) -> None:
        if self._thread.ident is None:
            return
        self._thread.join(timeout)
        if self._thread.is_alive():
            try:
                self._pipe.close()
            except OSError:
                pass
            self._thread.join(0.2)

    def _read(self) -> None:
        try:
            while True:
                chunk = self._pipe.read(65_536)
                if not chunk:
                    return
                self._digest.update(chunk)
                self.total_bytes += len(chunk)
                remaining = self._limit + 1 - len(self._data)
                if remaining > 0:
                    self._data.extend(chunk[:remaining])
                if len(self._data) > self._limit or len(chunk) > remaining:
                    self.too_large = True
        except (OSError, ValueError):
            return

    @property
    def data(self) -> bytes:
        return bytes(self._data[: self._limit])

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()


class AntigravityWorker:
    """Run one output-only Antigravity turn at a time."""

    def __init__(
        self,
        *,
        command: str | os.PathLike[str] = "agy",
        model: str = DEFAULT_MODEL,
        effort: str = "low",
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        extra_args: object = _EXTRA_ARGS_UNSET,
    ) -> None:
        if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
            raise ValueError("timeout_seconds must be finite and positive")
        if max_input_bytes <= 0 or max_output_bytes <= 0:
            raise ValueError("byte limits must be positive")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(effort, str) or not effort.strip():
            raise ValueError("effort must be a non-empty string")
        self._command = os.fspath(command)
        self._model = model
        self._effort = effort
        self._extra_args = (
            []
            if extra_args is _EXTRA_ARGS_UNSET
            else validate_antigravity_extra_args(extra_args)
        )
        self._timeout_seconds = float(timeout_seconds)
        self._max_input_bytes = int(max_input_bytes)
        self._max_output_bytes = int(max_output_bytes)
        self._run_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._cleanup_lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._process: subprocess.Popen[bytes] | None = None
        self._stdout_reader: _BoundedPipeReader | None = None
        self._stderr_reader: _BoundedPipeReader | None = None
        self._workspace: Path | None = None
        self._closed = False

    def run(
        self,
        *,
        goal: str,
        context: str,
        output_schema: dict | None,
        on_process_started: Callable[[], None] | None = None,
    ) -> AntigravityResult:
        started = time.monotonic()
        if not isinstance(goal, str) or not isinstance(context, str):
            return self._failure(started, "invalid_input", "goal and context must be strings")
        if output_schema is not None and not isinstance(output_schema, dict):
            return self._failure(started, "invalid_schema", "output_schema must be an object")
        unsupported = _unsupported_schema_keyword(output_schema) if output_schema is not None else None
        if unsupported is not None:
            return self._failure(
                started,
                "invalid_schema",
                f"output_schema keyword {unsupported!r} is not supported by the parent validator",
            )

        try:
            schema_bytes = (
                json.dumps(output_schema, separators=(",", ":")).encode("utf-8")
                if output_schema is not None
                else None
            )
        except (TypeError, ValueError):
            return self._failure(started, "invalid_schema", "output_schema is not JSON serializable")

        prompt = self._build_prompt(goal, context)
        prompt_bytes = prompt.encode("utf-8")
        if len(prompt_bytes) + len(schema_bytes or b"") > self._max_input_bytes:
            return self._failure(started, "input_too_large", "worker input exceeds byte limit")

        if not self._run_lock.acquire(blocking=False):
            return self._failure(started, "already_running", "worker already has an active run")
        try:
            with self._state_lock:
                if self._closed:
                    return self._failure(started, "closed", "worker is closed")
                self._cancel_event.clear()
            try:
                result = self._run_locked(
                    started,
                    prompt_bytes,
                    output_schema,
                    schema_bytes,
                    on_process_started,
                )
            except Exception:
                result = self._failure(
                    started,
                    "worker_error",
                    "Antigravity worker failed",
                )
            with self._state_lock:
                teardown_unconfirmed = any(
                    resource is not None
                    for resource in (
                        self._process,
                        self._stdout_reader,
                        self._stderr_reader,
                        self._workspace,
                    )
                )
            if teardown_unconfirmed:
                return self._failure(
                    started,
                    "teardown_unconfirmed",
                    "Antigravity process teardown unconfirmed",
                )
            return result
        finally:
            self._run_lock.release()

    def cancel(self) -> None:
        """Request cancellation of the active run, if any."""
        self._cancel_event.set()
        with self._state_lock:
            process = self._process
        if process is None:
            return
        try:
            process_running = process.poll() is None
            if process_running:
                self._signal_process_group(process, signal.SIGTERM)
        except Exception:
            raise RuntimeError(
                "Antigravity process cancellation could not be confirmed"
            ) from None

    def close(self) -> None:
        """Permanently close this worker and cancel its active run."""
        with self._state_lock:
            self._closed = True
        try:
            self.cancel()
        except Exception:
            pass
        if not self._retry_cleanup_resources():
            raise RuntimeError("Antigravity process teardown unconfirmed") from None

    def _run_locked(
        self,
        started: float,
        prompt: bytes,
        output_schema: dict | None,
        schema_bytes: bytes | None,
        on_process_started: Callable[[], None] | None,
    ) -> AntigravityResult:
        workspace: Path | None = None
        process: subprocess.Popen[bytes] | None = None
        stdout_reader: _BoundedPipeReader | None = None
        stderr_reader: _BoundedPipeReader | None = None
        try:
            workspace = Path(tempfile.mkdtemp(prefix="hermes-antigravity-"))
            with self._state_lock:
                self._workspace = workspace
            os.chmod(workspace, 0o700)
            argv = [
                self._command,
                "--print",
                "--model",
                self._model,
                "--effort",
                self._effort,
                "--mode",
                "plan",
                "--sandbox",
                "--disable-slash-commands",
                "--output-format",
                "json",
                "--print-timeout",
                f"{self._timeout_seconds:g}s",
            ]
            argv.extend(self._extra_args)
            if output_schema is not None:
                schema_path = (workspace / "output-schema.json").resolve()
                assert schema_bytes is not None
                schema_path.write_bytes(schema_bytes)
                os.chmod(schema_path, 0o600)
                argv.extend(("--json-schema", str(schema_path)))

            with self._state_lock:
                if self._closed or self._cancel_event.is_set():
                    return self._failure(
                        started, "cancelled", "Antigravity run was cancelled"
                    )
                try:
                    process = subprocess.Popen(
                        argv,
                        cwd=workspace,
                        env=self._child_env(),
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        start_new_session=True,
                    )
                except (OSError, ValueError):
                    return self._failure(started, "spawn_error", "could not start agy")
                self._process = process
            if on_process_started is not None:
                try:
                    on_process_started()
                except Exception:
                    self._terminate_and_drain(process)
                    return self._failure(
                        started,
                        "process_start_callback_failed",
                        "Antigravity process start could not be recorded",
                        process.returncode,
                    )
            assert process.stdin is not None
            assert process.stdout is not None
            assert process.stderr is not None
            stdout_reader = _BoundedPipeReader(process.stdout, self._max_output_bytes)
            stderr_reader = _BoundedPipeReader(process.stderr, self._max_output_bytes)
            with self._state_lock:
                self._stdout_reader = stdout_reader
                self._stderr_reader = stderr_reader
            stdout_reader.start()
            stderr_reader.start()
            stdin_writer = threading.Thread(
                target=self._write_stdin, args=(process.stdin, prompt), daemon=True
            )
            stdin_writer.start()

            deadline = started + self._timeout_seconds
            failure_code: str | None = None
            while process.poll() is None:
                if self._cancel_event.is_set():
                    failure_code = "cancelled"
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    failure_code = "timeout"
                    break
                try:
                    process.wait(timeout=min(0.05, remaining))
                except subprocess.TimeoutExpired:
                    pass

            if failure_code is None and self._cancel_event.is_set():
                # cancel() may terminate the child between poll iterations; the
                # caller's cancellation intent still owns the terminal result.
                failure_code = "cancelled"
            if failure_code is not None:
                self._terminate_and_drain(process)
            else:
                process.wait()
            stdout_reader.join(1.0)
            stderr_reader.join(1.0)
            stdin_writer.join(0.2)

            if failure_code == "cancelled":
                return self._failure(
                    started, "cancelled", "Antigravity run was cancelled", process.returncode
                )
            if failure_code == "timeout":
                return self._failure(
                    started, "timeout", "Antigravity run exceeded its deadline", process.returncode
                )
            if stdout_reader.too_large or stderr_reader.too_large:
                oversized_reader = stdout_reader if stdout_reader.too_large else stderr_reader
                return self._failure(
                    started,
                    "output_too_large",
                    "Antigravity output exceeds byte limit",
                    process.returncode,
                    output_excerpt=oversized_reader.data.decode("utf-8", errors="ignore"),
                    output_sha256=oversized_reader.sha256,
                    output_bytes=oversized_reader.total_bytes,
                )
            if process.returncode != 0:
                return self._failure(
                    started,
                    "nonzero_exit",
                    "Antigravity exited unsuccessfully",
                    process.returncode,
                )
            return self._validate_envelope(
                started, stdout_reader.data, process.returncode, output_schema
            )
        finally:
            self._retry_cleanup_resources()

    def _retry_cleanup_resources(self) -> bool:
        """Release the complete retained run graph only after confirmed cleanup."""
        with self._cleanup_lock:
            with self._state_lock:
                process = self._process
                readers = (self._stdout_reader, self._stderr_reader)
                workspace = self._workspace

            process_exited = process is None
            if process is not None:
                try:
                    process_running = process.poll() is None
                except Exception:
                    process_running = True
                if process_running:
                    try:
                        self._terminate_and_drain(process)
                    except Exception:
                        pass
                try:
                    process_exited = process.poll() is not None
                except Exception:
                    process_exited = False

            readers_settled = True
            for reader in readers:
                if reader is None:
                    continue
                try:
                    reader.join(0.2)
                    if reader._thread.is_alive():
                        readers_settled = False
                except Exception:
                    readers_settled = False

            if not process_exited or not readers_settled:
                return False

            if workspace is not None:
                try:
                    shutil.rmtree(workspace)
                except FileNotFoundError:
                    pass
                except Exception:
                    return False
                if workspace.exists():
                    return False

            with self._state_lock:
                if self._process is process:
                    self._process = None
                if self._stdout_reader is readers[0]:
                    self._stdout_reader = None
                if self._stderr_reader is readers[1]:
                    self._stderr_reader = None
                if self._workspace is workspace:
                    self._workspace = None
                return (
                    self._process is None
                    and self._stdout_reader is None
                    and self._stderr_reader is None
                    and self._workspace is None
                )

    @staticmethod
    def _build_prompt(goal: str, context: str) -> str:
        contract = (
            "You are an output-only language worker. Do not use tools, inspect files, "
            "modify state, browse, or request permission. Use only the supplied goal "
            "and context. Return only the requested final answer."
        )
        payload = json.dumps(
            {"goal": goal, "context": context}, ensure_ascii=False, separators=(",", ":")
        )
        return f"{contract}\nINPUT_JSON:\n{payload}\n"

    @staticmethod
    def _child_env() -> dict[str, str]:
        env: dict[str, str] = {}
        for key in _BASE_ENV_ALLOWLIST:
            value = os.environ.get(key)
            if value is not None:
                env[key] = value
        env.setdefault("PATH", os.defpath)
        env.setdefault("LANG", "C.UTF-8")
        env["PYTHONUTF8"] = "1"
        return env

    @staticmethod
    def _write_stdin(pipe: IO[bytes], prompt: bytes) -> None:
        try:
            pipe.write(prompt)
            pipe.close()
        except (BrokenPipeError, OSError, ValueError):
            pass

    def _terminate_and_drain(self, process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        self._signal_process_group(process, signal.SIGTERM)
        try:
            process.wait(timeout=0.5)
            return
        except subprocess.TimeoutExpired:
            pass
        self._signal_process_group(process, _SIGKILL)
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            # Last-resort per-process kill for platforms without process groups.
            try:
                process.kill()
            except OSError:
                pass
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    "Antigravity process teardown unconfirmed"
                ) from None

    @staticmethod
    def _signal_process_group(process: subprocess.Popen[bytes], sig: signal.Signals) -> None:
        try:
            if process.poll() is not None:
                return
        except Exception:
            pass
        killpg = getattr(os, "killpg", None)
        if os.name == "posix" and killpg is not None:
            try:
                killpg(process.pid, sig)
                return
            except ProcessLookupError:
                return
            except OSError:
                pass
        try:
            if sig == _SIGKILL:
                process.kill()
            else:
                process.terminate()
        except OSError:
            pass

    def _validate_envelope(
        self,
        started: float,
        raw_stdout: bytes,
        exit_code: int,
        output_schema: dict | None,
    ) -> AntigravityResult:
        try:
            decoded = raw_stdout.decode("utf-8")
            envelope = json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self._failure(
                started, "malformed_envelope", "Antigravity returned malformed JSON", exit_code
            )
        if not isinstance(envelope, dict):
            return self._failure(
                started, "malformed_envelope", "Antigravity envelope must be an object", exit_code
            )
        if envelope.get("status") != "SUCCESS":
            return self._failure(
                started,
                "unsuccessful_status",
                "Antigravity did not report SUCCESS",
                exit_code,
                envelope,
            )
        if self._contains_blocked_tool_action(envelope):
            return self._failure(
                started,
                "tool_action_blocked",
                "Antigravity reported a denied or pending tool action",
                exit_code,
                envelope,
            )
        response = envelope.get("response")
        if not isinstance(response, str) or not response.strip():
            return self._failure(
                started,
                "invalid_response",
                "Antigravity response must be a nonempty string",
                exit_code,
                envelope,
            )
        if len(response.encode("utf-8")) > self._max_output_bytes:
            return self._failure(
                started,
                "output_too_large",
                "Antigravity response exceeds byte limit",
                exit_code,
                envelope,
            )
        if output_schema is not None:
            try:
                parsed_response = json.loads(response)
            except json.JSONDecodeError:
                return self._failure(
                    started,
                    "schema_validation_failed",
                    "Antigravity response is not valid JSON",
                    exit_code,
                    envelope,
                )
            schema_error = _schema_error(parsed_response, output_schema)
            if schema_error is not None:
                return self._failure(
                    started,
                    "schema_validation_failed",
                    f"Antigravity response failed schema validation: {schema_error}",
                    exit_code,
                    envelope,
                )
        conversation_id = envelope.get("conversation_id")
        if not isinstance(conversation_id, str):
            conversation_id = None
        usage = envelope.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        return AntigravityResult(
            status="success",
            response=response,
            conversation_id=conversation_id,
            usage=usage,
            raw_envelope=envelope,
            exit_code=exit_code,
            duration_ms=self._duration_ms(started),
            error_code=None,
            error_message=None,
        )

    @classmethod
    def _contains_blocked_tool_action(cls, value: Any, in_action: bool = False) -> bool:
        if isinstance(value, dict):
            for key, child in value.items():
                key_lower = str(key).lower()
                child_in_action = in_action or any(
                    marker in key_lower
                    for marker in ("tool", "action", "permission", "approval")
                )
                if (
                    child_in_action
                    and any(marker in key_lower for marker in ("status", "state", "decision"))
                    and isinstance(child, str)
                    and child.strip().upper() in {"DENIED", "PENDING"}
                ):
                    return True
                if cls._contains_blocked_tool_action(child, child_in_action):
                    return True
        elif isinstance(value, list):
            return any(cls._contains_blocked_tool_action(item, in_action) for item in value)
        return False

    @staticmethod
    def _duration_ms(started: float) -> int:
        return max(0, int((time.monotonic() - started) * 1000))

    @classmethod
    def _failure(
        cls,
        started: float,
        code: str,
        message: str,
        exit_code: int | None = None,
        envelope: dict[str, Any] | None = None,
        *,
        output_excerpt: str | None = None,
        output_sha256: str | None = None,
        output_bytes: int | None = None,
    ) -> AntigravityResult:
        return AntigravityResult(
            status="failed",
            response=None,
            conversation_id=None,
            usage={},
            raw_envelope=envelope,
            exit_code=exit_code,
            duration_ms=cls._duration_ms(started),
            error_code=code,
            error_message=message,
            output_excerpt=output_excerpt,
            output_sha256=output_sha256,
            output_bytes=output_bytes,
        )


def _unsupported_schema_keyword(schema: Mapping[str, Any]) -> str | None:
    for keyword in _UNSUPPORTED_SCHEMA_KEYWORDS:
        if keyword in schema:
            return keyword
    for key in ("not", "items", "additionalProperties"):
        child = schema.get(key)
        if isinstance(child, Mapping):
            found = _unsupported_schema_keyword(child)
            if found is not None:
                return found
    for key in ("allOf", "anyOf", "oneOf"):
        children = schema.get(key)
        if isinstance(children, list):
            for child in children:
                if isinstance(child, Mapping):
                    found = _unsupported_schema_keyword(child)
                    if found is not None:
                        return found
    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        for child in properties.values():
            if isinstance(child, Mapping):
                found = _unsupported_schema_keyword(child)
                if found is not None:
                    return found
    return None


def _schema_error(instance: Any, schema: Mapping[str, Any], path: str = "$") -> str | None:
    """Validate the JSON-Schema subset used by Hermes output contracts.

    Supported assertions cover typed nested objects/arrays, required and extra
    properties, enums/const, combinators, common string/number/array bounds,
    and schemas without references. Annotation and unsupported validation
    keywords are not evaluated; callers needing the full JSON Schema
    vocabulary should validate again at the parent boundary.
    """
    if not isinstance(schema, Mapping):
        return f"{path}: schema must be an object"

    if "allOf" in schema:
        for subschema in schema["allOf"]:
            error = _schema_error(instance, subschema, path)
            if error:
                return error
    if "anyOf" in schema:
        errors = [_schema_error(instance, sub, path) for sub in schema["anyOf"]]
        if all(errors):
            return f"{path}: does not match anyOf"
    if "oneOf" in schema:
        matches = sum(_schema_error(instance, sub, path) is None for sub in schema["oneOf"])
        if matches != 1:
            return f"{path}: must match exactly one oneOf schema"
    if "not" in schema and _schema_error(instance, schema["not"], path) is None:
        return f"{path}: matches forbidden schema"

    if "const" in schema and instance != schema["const"]:
        return f"{path}: does not equal const"
    if "enum" in schema and instance not in schema["enum"]:
        return f"{path}: is not in enum"

    expected = schema.get("type")
    if expected is not None:
        expected_types = [expected] if isinstance(expected, str) else expected
        if not isinstance(expected_types, list) or not any(
            _matches_json_type(instance, item) for item in expected_types
        ):
            return f"{path}: has wrong type"

    if isinstance(instance, dict):
        required = schema.get("required", [])
        if not isinstance(required, list):
            return f"{path}: required must be an array"
        for key in required:
            if key not in instance:
                return f"{path}: missing required property {key!r}"
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            return f"{path}: properties must be an object"
        additional = schema.get("additionalProperties", True)
        for key, value in instance.items():
            if key in properties:
                error = _schema_error(value, properties[key], f"{path}.{key}")
            elif additional is False:
                error = f"{path}: additional property {key!r} is not allowed"
            elif isinstance(additional, Mapping):
                error = _schema_error(value, additional, f"{path}.{key}")
            else:
                error = None
            if error:
                return error
        if "minProperties" in schema and len(instance) < schema["minProperties"]:
            return f"{path}: too few properties"
        if "maxProperties" in schema and len(instance) > schema["maxProperties"]:
            return f"{path}: too many properties"

    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            return f"{path}: too few items"
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            return f"{path}: too many items"
        if schema.get("uniqueItems") and len({json.dumps(x, sort_keys=True) for x in instance}) != len(instance):
            return f"{path}: items are not unique"
        items = schema.get("items")
        if isinstance(items, Mapping):
            for index, item in enumerate(instance):
                error = _schema_error(item, items, f"{path}[{index}]")
                if error:
                    return error

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            return f"{path}: string is too short"
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            return f"{path}: string is too long"
        if "pattern" in schema:
            try:
                if re.search(schema["pattern"], instance) is None:
                    return f"{path}: string does not match pattern"
            except (re.error, TypeError):
                return f"{path}: schema pattern is invalid"

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        for keyword, comparison in (
            ("minimum", lambda value, bound: value >= bound),
            ("maximum", lambda value, bound: value <= bound),
            ("exclusiveMinimum", lambda value, bound: value > bound),
            ("exclusiveMaximum", lambda value, bound: value < bound),
        ):
            if keyword in schema and not comparison(instance, schema[keyword]):
                return f"{path}: violates {keyword}"
        if "multipleOf" in schema:
            divisor = schema["multipleOf"]
            if not isinstance(divisor, (int, float)) or divisor <= 0:
                return f"{path}: schema multipleOf is invalid"
            quotient = instance / divisor
            if not math.isclose(quotient, round(quotient), abs_tol=1e-12):
                return f"{path}: violates multipleOf"
    return None


def _matches_json_type(value: Any, expected: Any) -> bool:
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return False
