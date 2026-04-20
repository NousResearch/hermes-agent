from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from hermes_cli.command_templates import CommandInvocation, build_command_invocation
from hermes_cli.commands import resolve_command

WORK_COMMAND_ADAPTER_SCHEMA = "hermes/work-command-intake"
WORK_COMMAND_ADAPTER_VERSION = "2.0"


class WorkCommandContractError(ValueError):
    """User-facing failure for malformed work-command payloads."""

    def __init__(self, message: str, *, command_name: str, raw_args: str = "", cause: Exception | None = None):
        super().__init__(message)
        self.command_name = command_name
        self.raw_args = raw_args
        self.cause = cause


@dataclass(frozen=True)
class PreparedWorkCommand:
    invocation: CommandInvocation
    adapter_schema: str = WORK_COMMAND_ADAPTER_SCHEMA
    adapter_version: str = WORK_COMMAND_ADAPTER_VERSION

    @property
    def command_name(self) -> str:
        return self.invocation.command_name

    @property
    def raw_args(self) -> str:
        return self.invocation.raw_args

    @property
    def task_contract(self) -> dict[str, Any]:
        return self.invocation.task_contract

    @property
    def orchestration_hints(self) -> dict[str, Any]:
        return self.invocation.orchestration_hints

    @property
    def agent_message(self) -> str:
        return self.invocation.prompt_text

    @property
    def display_text(self) -> str:
        suffix = f" {self.raw_args}" if self.raw_args else ""
        return f"/{self.command_name}{suffix}"


def is_prepared_work_command(value: Any) -> bool:
    return isinstance(value, PreparedWorkCommand)


def _format_validation_error(exc: ValidationError) -> str:
    parts: list[str] = []
    for error in exc.errors(include_url=False):
        loc = ".".join(str(part) for part in error.get("loc") or ()) or "payload"
        msg = str(error.get("msg") or "invalid value")
        parts.append(f"{loc}: {msg}")
    return "; ".join(parts) or str(exc)


def _normalize_cwd(cwd: str | None) -> str | None:
    return str(Path(cwd).resolve()) if cwd else None


def _normalize_work_command_name(command_name: str) -> str:
    normalized_command = str(command_name or "").strip().lower().lstrip("/")
    resolved = resolve_command(normalized_command)
    if resolved is None:
        return normalized_command
    return resolved.name


def prepare_work_command(
    command_name: str,
    *,
    raw_args: str = "",
    session_id: str | None = None,
    cwd: str | None = None,
) -> PreparedWorkCommand:
    normalized_command = _normalize_work_command_name(command_name)
    normalized_args = str(raw_args or "").strip()
    try:
        invocation = build_command_invocation(
            normalized_command,
            raw_args=normalized_args,
            session_id=session_id,
            cwd=_normalize_cwd(cwd),
        )
    except json.JSONDecodeError as exc:
        raise WorkCommandContractError(
            f"Malformed /{normalized_command} work command JSON: expected a valid JSON object.",
            command_name=normalized_command,
            raw_args=normalized_args,
            cause=exc,
        ) from exc
    except ValidationError as exc:
        raise WorkCommandContractError(
            f"Invalid /{normalized_command} task contract: {_format_validation_error(exc)}",
            command_name=normalized_command,
            raw_args=normalized_args,
            cause=exc,
        ) from exc
    except (TypeError, ValueError) as exc:
        raise WorkCommandContractError(
            f"Invalid /{normalized_command} work command payload: {exc}",
            command_name=normalized_command,
            raw_args=normalized_args,
            cause=exc,
        ) from exc
    except KeyError as exc:
        raise WorkCommandContractError(
            f"Unknown work command: /{normalized_command}",
            command_name=normalized_command,
            raw_args=normalized_args,
            cause=exc,
        ) from exc
    return PreparedWorkCommand(invocation=invocation)


__all__ = [
    "PreparedWorkCommand",
    "WORK_COMMAND_ADAPTER_SCHEMA",
    "WORK_COMMAND_ADAPTER_VERSION",
    "WorkCommandContractError",
    "is_prepared_work_command",
    "prepare_work_command",
]
