#!/usr/bin/env python3
"""stdio MCP server for Mithril fund reads and local record checks.

Local tools do not use the network. Remote tools are GET-only against
MITHRIL_API_ORIGIN and never call billing, sandbox launch, lake ingest,
research-scope writes, or device approval.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ALLOWED_ORIGINS = {"https://api.mithril.fund", "http://127.0.0.1:8787"}
REMOTE_GETS = {
    "domain_check": "/v1/check/domain",
    "account_me": "/v1/me",
    "list_models": "/v1/models",
    "billing_status": "/v1/billing/status",
    "list_cases": "/v1/cases",
    "sandbox_status": "/v1/sandbox/session",
}
AUTHENTICATED = {"account_me", "list_models", "billing_status", "list_cases", "sandbox_status"}


def _records():
    candidates = []
    override = os.environ.get("MITHRIL_RECORDS_SCRIPT", "").strip()
    if override:
        candidates.append(Path(override))
    here = Path(__file__).resolve()
    candidates.append(here.parents[2] / "skills" / "security" / "security-records" / "scripts" / "records.py")
    candidates.append(Path.home() / ".hermes" / "skills" / "security" / "security-records" / "scripts" / "records.py")
    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location("mithril_security_records", path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
    raise RuntimeError("security-records script was not found")


RECORDS = None


def records():
    global RECORDS
    if RECORDS is None:
        RECORDS = _records()
    return RECORDS


def _origin() -> str:
    origin = os.environ.get("MITHRIL_API_ORIGIN", "https://api.mithril.fund").rstrip("/")
    if origin not in ALLOWED_ORIGINS:
        raise ValueError("MITHRIL_API_ORIGIN is not an allowed Mithril API origin")
    return origin


def _get(path: str, query: dict[str, str] | None, authenticated: bool) -> dict[str, Any]:
    url = _origin() + path
    if query:
        url += "?" + urllib.parse.urlencode(query)
    headers = {"accept": "application/json"}
    if authenticated:
        token = os.environ.get("MITHRIL_API_TOKEN", "").strip()
        if not token:
            raise ValueError("MITHRIL_API_TOKEN is not set")
        headers["authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise ValueError(f"HTTP {exc.code}: {detail}") from exc
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        return {"result": parsed}
    return parsed


TOOLS = [
    {
        "name": "cmdb_check",
        "description": "Structural check of a caller-supplied asset and dependency list.",
        "inputSchema": {"type": "object", "properties": {"assets": {"type": "array"}, "dependencies": {"type": "array"}}, "required": ["assets", "dependencies"]},
    },
    {
        "name": "email_signals",
        "description": "Attachment-name signals. Does not read file bytes or message bodies.",
        "inputSchema": {"type": "object", "properties": {"messages": {"type": "array"}}, "required": ["messages"]},
    },
    {
        "name": "compliance_fit",
        "description": "Published framework-to-control mapping. Not an audit result.",
        "inputSchema": {"type": "object", "properties": {"framework": {"type": "string"}}, "required": ["framework"]},
    },
    {
        "name": "security_catalog",
        "description": "Published security-service catalog. Description only; it does not scan.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "spear_phishing_defence",
        "description": "Describes the spear-phishing defence page. It does not send mail.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "domain_check",
        "description": "GET the public phishing-domain projection for one domain name.",
        "inputSchema": {"type": "object", "properties": {"domain": {"type": "string"}}, "required": ["domain"]},
    },
    {
        "name": "account_me",
        "description": "GET /v1/me with the configured token.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_models",
        "description": "GET /v1/models with the configured token.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "billing_status",
        "description": "GET billing status. It does not start a checkout.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_cases",
        "description": "GET cases for the authenticated account.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "sandbox_status",
        "description": "GET the current sandbox session. It does not launch one.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def call_tool(name: str, arguments: dict[str, Any]) -> Any:
    module = records()
    if name == "cmdb_check":
        return module.check_cmdb(arguments)
    if name == "email_signals":
        return module.email_signals(arguments)
    if name == "compliance_fit":
        return module.compliance_fit(str(arguments.get("framework") or ""))
    if name == "security_catalog":
        return module.service_catalog()
    if name == "spear_phishing_defence":
        return module.spear_phishing_defence()
    if name == "domain_check":
        domain = str(arguments.get("domain") or "").strip().lower().rstrip(".")
        if not domain or "." not in domain:
            raise ValueError("a domain name is required")
        return _get(REMOTE_GETS[name], {"domain": domain}, False)
    if name in REMOTE_GETS:
        return _get(REMOTE_GETS[name], None, name in AUTHENTICATED)
    raise ValueError(f"unknown tool: {name}")


def handle(message: dict[str, Any]) -> dict[str, Any] | None:
    method = message.get("method")
    msg_id = message.get("id")
    if method == "notifications/initialized" or msg_id is None:
        return None
    try:
        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mithril-fund", "version": "0.1.0"},
            }
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            params = message.get("params") or {}
            payload = call_tool(str(params.get("name") or ""), params.get("arguments") or {})
            result = {"content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}], "isError": False}
        else:
            return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": "method not found"}}
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}
    except Exception as exc:  # noqa: BLE001 — tool errors are returned to the client
        if method == "tools/call":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"content": [{"type": "text", "text": json.dumps({"error": str(exc)})}], "isError": True},
            }
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32000, "message": str(exc)}}


def _read_message() -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = sys.stdin.buffer.readline()
        if line == b"":
            return None
        if line in (b"\r\n", b"\n"):
            break
        decoded = line.decode("ascii", errors="replace")
        if ":" not in decoded:
            continue
        key, value = decoded.split(":", 1)
        headers[key.strip().lower()] = value.strip()
    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return None
    body = sys.stdin.buffer.read(length)
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError("MCP message must be an object")
    return parsed


def _write_message(payload: dict[str, Any]) -> None:
    raw = json.dumps(payload).encode("utf-8")
    header = f"Content-Length: {len(raw)}\r\n\r\n".encode("ascii")
    sys.stdout.buffer.write(header)
    sys.stdout.buffer.write(raw)
    sys.stdout.buffer.flush()


def main() -> int:
    while True:
        message = _read_message()
        if message is None:
            return 0
        response = handle(message)
        if response is not None:
            _write_message(response)


if __name__ == "__main__":
    raise SystemExit(main())
