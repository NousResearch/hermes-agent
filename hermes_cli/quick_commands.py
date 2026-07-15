"""Validation and bounded I/O helpers for deterministic quick commands."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any


QUICK_COMMAND_TIMEOUT_SECONDS = 30
QUICK_COMMAND_INPUT_MAX_BYTES = 8192
QUICK_COMMAND_OUTPUT_MAX_BYTES = 65536
QUICK_COMMAND_METADATA_MAX_BYTES = 256
QUICK_COMMAND_DESTINATION_ALIAS_MAX_BYTES = 64

_DESTINATION_ALIAS_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_TRUSTED_BASE_ENV_KEYS = (
    "HOME",
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "TMPDIR",
    # Windows process creation and executable lookup.
    "SYSTEMROOT",
    "PATHEXT",
)


class QuickCommandConfigError(ValueError):
    """Raised when a quick-command definition is unsafe or malformed."""


class QuickCommandOutputError(ValueError):
    """Raised when deterministic command output cannot be returned safely."""


def prepare_argv_command(qcmd: Any, argument_text: str) -> list[str]:
    """Validate an ``argv`` quick command and return its exact process argv."""
    if not isinstance(qcmd, Mapping):
        raise QuickCommandConfigError("configuration must be a mapping")

    command = qcmd.get("command")
    if not isinstance(command, list) or not command:
        raise QuickCommandConfigError("command must be a non-empty list of strings")
    if any(
        not isinstance(item, str) or not item.strip() or "\x00" in item
        for item in command
    ):
        raise QuickCommandConfigError("command must contain only non-empty strings")

    argument_mode = qcmd.get("argument_mode", "none")
    if argument_mode not in ("none", "text"):
        raise QuickCommandConfigError("argument_mode must be 'none' or 'text'")

    argv = list(command)
    if argument_mode == "text":
        if not isinstance(argument_text, str) or not argument_text.strip():
            raise QuickCommandConfigError("argument_mode 'text' requires text")
        input_bytes = len(argument_text.encode("utf-8"))
        if input_bytes > QUICK_COMMAND_INPUT_MAX_BYTES:
            raise QuickCommandConfigError(
                f"text exceeds {QUICK_COMMAND_INPUT_MAX_BYTES} UTF-8 bytes"
            )
        if "\x00" in argument_text:
            raise QuickCommandConfigError("text must not contain NUL bytes")
        # One exact argv item: metacharacters and whitespace are never parsed by
        # a shell and therefore cannot change the configured executable/flags.
        argv.append(argument_text)
    return argv


def build_argv_environment(extra: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Build a minimal child environment without Hermes-managed credentials."""
    child_env = {
        key: value
        for key in _TRUSTED_BASE_ENV_KEYS
        if (value := os.environ.get(key)) is not None and "\x00" not in value
    }
    for name, value in (extra or {}).items():
        child_env[name] = _bounded_metadata(value, name)
    return child_env


def build_gateway_argv_environment(
    qcmd: Any,
    *,
    platform: Any,
    message_id: Any,
    update_id: Any,
) -> dict[str, str]:
    """Build a secret-free gateway child environment with request provenance."""
    if not isinstance(qcmd, Mapping):
        raise QuickCommandConfigError("configuration must be a mapping")

    destination_alias = qcmd.get("destination_alias")
    if destination_alias is not None:
        if not isinstance(destination_alias, str) or not destination_alias:
            raise QuickCommandConfigError("destination_alias must be a non-empty string")
        if (
            len(destination_alias.encode("utf-8"))
            > QUICK_COMMAND_DESTINATION_ALIAS_MAX_BYTES
            or not _DESTINATION_ALIAS_RE.fullmatch(destination_alias)
        ):
            raise QuickCommandConfigError(
                "destination_alias must be 1-64 ASCII letters, digits, '.', '_', ':', or '-'"
            )

    provenance = {
            "HERMES_QUICK_COMMAND_PLATFORM": _bounded_metadata(
                platform, "platform", allow_empty=False
            ),
            "HERMES_QUICK_COMMAND_MESSAGE_ID": _bounded_metadata(
                message_id, "message_id"
            ),
            "HERMES_QUICK_COMMAND_UPDATE_ID": _bounded_metadata(
                update_id, "update_id"
            ),
        }
    if destination_alias is not None:
        provenance["HERMES_QUICK_COMMAND_DESTINATION_ALIAS"] = destination_alias
    return build_argv_environment(provenance)


def bounded_quick_command_output(stdout: Any, stderr: Any) -> str:
    """Return redacted stdout (or stderr fallback) after a combined byte cap."""
    stdout_text = _output_text(stdout)
    stderr_text = _output_text(stderr)
    output_bytes = len(stdout_text.encode("utf-8")) + len(stderr_text.encode("utf-8"))
    if output_bytes > QUICK_COMMAND_OUTPUT_MAX_BYTES:
        raise QuickCommandOutputError(
            f"output exceeds {QUICK_COMMAND_OUTPUT_MAX_BYTES} UTF-8 bytes"
        )

    output = stdout_text.strip() or stderr_text.strip()
    if not output:
        return ""
    try:
        from agent.redact import redact_sensitive_text

        # Deterministic command output crosses a user-visible gateway boundary;
        # force redaction even when the general display preference is disabled.
        return redact_sensitive_text(output, force=True)
    except Exception as exc:
        raise QuickCommandOutputError("output could not be safely redacted") from exc


def _bounded_metadata(
    value: Any, name: str, *, allow_empty: bool = True
) -> str:
    text = "" if value is None else str(value)
    if not allow_empty and not text:
        raise QuickCommandConfigError(f"{name} is required")
    if "\x00" in text:
        raise QuickCommandConfigError(f"{name} must not contain NUL bytes")
    if len(text.encode("utf-8")) > QUICK_COMMAND_METADATA_MAX_BYTES:
        raise QuickCommandConfigError(
            f"{name} exceeds {QUICK_COMMAND_METADATA_MAX_BYTES} UTF-8 bytes"
        )
    return text


def _output_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
