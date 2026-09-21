from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .policy import AccountPolicy, PolicyError


class McpReadOnlyError(RuntimeError):
    """A request or response violated the mailbox reader's narrow contract."""


class ControlledMcpConfigError(ValueError):
    """The explicitly supplied operator config is not safe to execute."""


class McpSession(Protocol):
    def request(self, envelope: Mapping[str, object]) -> Mapping[str, object]: ...

    def close(self) -> None: ...


_METADATA_PAGE_SIZE = 25
_OUTLOOK_SELECT = "id,from,subject,receivedDateTime,bodyPreview"
_READ_CALLS: dict[str, dict[str, tuple[str, dict[str, object]]]] = {
    "gmail": {
        "list_metadata": (
            "search_gmail_messages",
            {"query": "is:unread", "page_size": _METADATA_PAGE_SIZE},
        )
    },
    "outlook": {
        "list_metadata": (
            "list-mail-messages",
            {
                "select": _OUTLOOK_SELECT,
                "filter": "isRead eq false",
                "top": _METADATA_PAGE_SIZE,
            },
        )
    },
}
_RESULT_LIST_KEYS = {"gmail": "messages", "outlook": "value"}


@dataclass(frozen=True, slots=True)
class ControlledStdioConfig:
    commands: Mapping[str, tuple[str, ...]]
    policies: tuple[AccountPolicy, ...]


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _address(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        email_address = value.get("emailAddress")
        if isinstance(email_address, Mapping):
            return _text(email_address.get("address"))
        return _text(value.get("address")) or _text(value.get("email"))
    return ""


def _normalise_message(row: Mapping[str, object]) -> dict[str, str]:
    return {
        "id": _text(row.get("id")) or _text(row.get("message_id")),
        "from": _address(row.get("from")) or _text(row.get("sender")),
        "subject": _text(row.get("subject")),
        "received_at": _text(row.get("received_at")) or _text(row.get("receivedDateTime")),
        "preview": _text(row.get("preview")) or _text(row.get("bodyPreview")) or _text(row.get("snippet")),
    }


def _rows_from_result(result: Mapping[str, object], provider: str) -> list[dict[str, str]]:
    content = result.get("content")
    if not isinstance(content, list):
        raise McpReadOnlyError("MCP result must contain text content")
    rows: list[dict[str, str]] = []
    for item in content:
        if not isinstance(item, Mapping) or item.get("type") != "text":
            raise McpReadOnlyError("MCP result contains non-text content")
        try:
            parsed = json.loads(_text(item.get("text")))
        except json.JSONDecodeError as exc:
            raise McpReadOnlyError("MCP text content must be JSON metadata") from exc
        list_key = _RESULT_LIST_KEYS.get(provider)
        if list_key is None or not isinstance(parsed, Mapping):
            raise McpReadOnlyError("MCP text content must use a documented provider JSON result shape")
        provider_rows = parsed.get(list_key)
        if not isinstance(provider_rows, list) or not all(isinstance(row, Mapping) for row in provider_rows):
            raise McpReadOnlyError("MCP text content must use a documented provider JSON result shape")
        rows.extend(_normalise_message(row) for row in provider_rows)
    return rows


class McpReadOnlyAdapter:
    """Allowlisted mailbox metadata calls over an injected MCP session."""

    def __init__(self, session: McpSession) -> None:
        self._session = session
        self._next_id = 1
        self._lock = threading.Lock()

    def __call__(self, operation: str, account: str, *, provider: str) -> list[dict[str, str]]:
        if not account.strip():
            raise McpReadOnlyError("every MCP read requires an explicit account")
        call = _READ_CALLS.get(provider, {}).get(operation)
        if call is None:
            if provider not in _READ_CALLS:
                raise McpReadOnlyError(f"unsupported MCP provider: {provider}")
            raise McpReadOnlyError(f"operation is not allowlisted: {operation}")
        tool, arguments = call
        arguments = dict(arguments)
        arguments["user_google_email" if provider == "gmail" else "account"] = account
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
            envelope: dict[str, object] = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {"name": tool, "arguments": arguments},
            }
            response = self._session.request(envelope)
        if response.get("id") not in (None, request_id):
            raise McpReadOnlyError("MCP response id did not match the request")
        error = response.get("error")
        if isinstance(error, Mapping):
            raise McpReadOnlyError(_text(error.get("message")) or "MCP tool call failed")
        result = response.get("result")
        if not isinstance(result, Mapping) or result.get("isError") is True:
            raise McpReadOnlyError("MCP tool call returned an invalid read result")
        return _rows_from_result(result, provider)

    def close(self) -> None:
        self._session.close()


class StdioJsonRpcSession:
    """Small serial JSON-RPC session for the controlled operator-only path."""

    def __init__(self, command: Sequence[str]) -> None:
        safe_environment = {
            name: os.environ[name]
            for name in ("HOME", "LANG", "LC_ALL", "PATH", "TERM", "TMPDIR", "USER")
            if name in os.environ
        }
        self._process = subprocess.Popen(
            list(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            env=safe_environment,
        )
        if self._process.stdin is None or self._process.stdout is None:
            self.close()
            raise McpReadOnlyError("failed to open MCP stdio")
        self._responses: queue.Queue[Mapping[str, object]] = queue.Queue()
        self._closed = False
        self._lock = threading.Lock()
        threading.Thread(target=self._read_stdout, daemon=True).start()
        self._send(
            {
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "mailbox-cleaner-readonly", "version": "1"},
                },
            }
        )
        self._wait_for(0)
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

    def _read_stdout(self) -> None:
        assert self._process.stdout is not None
        for line in self._process.stdout:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(message, Mapping):
                self._responses.put(message)

    def _send(self, envelope: Mapping[str, object]) -> None:
        if self._closed or self._process.stdin is None:
            raise McpReadOnlyError("MCP stdio session is closed")
        self._process.stdin.write(json.dumps(envelope, separators=(",", ":")) + "\n")
        self._process.stdin.flush()

    def _wait_for(self, request_id: int) -> Mapping[str, object]:
        while True:
            try:
                response = self._responses.get(timeout=30)
            except queue.Empty as exc:
                raise McpReadOnlyError("MCP stdio request timed out") from exc
            if response.get("id") == request_id:
                return response

    def request(self, envelope: Mapping[str, object]) -> Mapping[str, object]:
        request_id = envelope.get("id")
        if not isinstance(request_id, int):
            raise McpReadOnlyError("MCP request id must be an integer")
        with self._lock:
            self._send(envelope)
            return self._wait_for(request_id)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)


def load_controlled_stdio_config(path: Path) -> ControlledStdioConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ControlledMcpConfigError("controlled MCP config must be a readable JSON file") from exc
    if not isinstance(raw, Mapping) or set(raw) != {"providers", "accounts"}:
        raise ControlledMcpConfigError("controlled MCP config requires only providers and accounts")
    providers = raw.get("providers")
    accounts = raw.get("accounts")
    if not isinstance(providers, Mapping) or not providers:
        raise ControlledMcpConfigError("controlled MCP config requires provider-specific servers")
    commands: dict[str, tuple[str, ...]] = {}
    for provider, server in providers.items():
        if provider not in _READ_CALLS or not isinstance(server, Mapping) or set(server) - {"command", "args"}:
            raise ControlledMcpConfigError("controlled MCP servers must be documented provider-specific commands")
        command = server.get("command")
        args = server.get("args", [])
        if not isinstance(command, str) or not command.strip() or not isinstance(args, list):
            raise ControlledMcpConfigError("controlled MCP server requires command and optional string args")
        if not all(isinstance(arg, str) and arg for arg in args):
            raise ControlledMcpConfigError("controlled MCP args must be non-empty strings")
        commands[provider] = (command, *args)
    if not isinstance(accounts, list) or not accounts:
        raise ControlledMcpConfigError("controlled MCP config requires explicit accounts")
    policies: list[AccountPolicy] = []
    try:
        for account in accounts:
            if not isinstance(account, Mapping) or set(account) != {"account", "provider"}:
                raise ControlledMcpConfigError("each controlled MCP account needs only account and provider")
            policies.append(AccountPolicy(account=_text(account.get("account")), provider=_text(account.get("provider"))))
    except PolicyError as exc:
        raise ControlledMcpConfigError(str(exc)) from exc
    if len({policy.account for policy in policies}) != len(policies):
        raise ControlledMcpConfigError("controlled MCP accounts must be unique")
    if any(policy.provider not in commands for policy in policies):
        raise ControlledMcpConfigError("every controlled MCP account requires its provider-specific server")
    return ControlledStdioConfig(commands=commands, policies=tuple(policies))
