"""SYNTHETIC stdio MCP fixture — protocol proof only, never a real GitHub App server.

A pure-stdlib peer that speaks the stdio MCP protocol (newline-delimited JSON-RPC
2.0) so tests can exercise the scoped acceptance client against a conforming
server without network access, credentials or a real App. Behaviour is driven
entirely by env vars the profile's own scoped server env provides:

  FIXTURE_EVIDENCE  path to a JSON document with the interface-v1 evidence fields
                    (identity, pr_before, classic_required, ruleset_required,
                    check_runs, statuses, pr_after, complete, counts)
  FIXTURE_MODE      ok | tool_error | mismatch | extra_text | no_structured
  FIXTURE_TRACE     path; appends one JSON line per tools/call: the tool name, the
                    exact arguments received, and the env KEY NAMES present (never
                    values — secret material is never written or echoed)
  FIXTURE_GATE      optional path; the server blocks before answering tools/call
                    until the file exists (lets a test run a reclaim race while
                    the collector is inside the call)

This bundle is labelled synthetic: it is not wired into any live profile, never
touches GitHub, and exists only under the test tree.
"""
from __future__ import annotations

import json
import os
import sys
import time

APP_CREDENTIAL_KEYS = (
    "GITHUB_APP_ID",
    "GITHUB_APP_INSTALLATION_ID",
    "GITHUB_APP_PRIVATE_KEY",
)


def _trace(arguments, mode):
    path = os.environ.get("FIXTURE_TRACE")
    if not path:
        return
    record = {
        "tool": "get_pr_acceptance_evidence",
        "arguments": arguments,
        "mode": mode,
        "env_key_names": sorted(os.environ),
        "has_app_credentials": all(k in os.environ for k in APP_CREDENTIAL_KEYS),
    }
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _tool_result(content_blocks, structured=None, is_error=False):
    result = {"content": content_blocks, "isError": is_error}
    if structured is not None:
        result["structuredContent"] = structured
    return result


def _fixed_error(code):
    return _tool_result([{"type": "text", "text": code}], is_error=True)


def _validate_arguments(arguments):
    if not isinstance(arguments, dict) or set(arguments) != {"owner", "repo", "pullNumber"}:
        return False
    owner, repo, number = arguments["owner"], arguments["repo"], arguments["pullNumber"]
    if isinstance(number, bool) or not isinstance(number, int) or number < 1:
        return False
    for value in (owner, repo):
        if not isinstance(value, str) or not value or value != value.strip():
            return False
    return True


def _result_for(arguments):
    mode = os.environ.get("FIXTURE_MODE", "ok")
    if not _validate_arguments(arguments):
        return _fixed_error("invalid_arguments")
    if mode == "tool_error":
        return _fixed_error("fixture_tool_error")
    with open(os.environ["FIXTURE_EVIDENCE"], encoding="utf-8") as fh:
        evidence = json.load(fh)
    if mode == "mismatch":
        return _tool_result([{"type": "text", "text": json.dumps({"schema_version": 1, "tampered": True})}],
                            structured=evidence)
    if mode == "extra_text":
        return _tool_result([{"type": "text", "text": json.dumps(evidence)},
                             {"type": "text", "text": json.dumps(evidence)}], structured=evidence)
    if mode == "no_structured":
        return {"content": [{"type": "text", "text": json.dumps(evidence)}], "isError": False}
    return _tool_result([{"type": "text", "text": json.dumps(evidence)}], structured=evidence)


def _wait_for_gate():
    gate = os.environ.get("FIXTURE_GATE")
    if not gate:
        return
    deadline = time.monotonic() + 30
    while not os.path.exists(gate) and time.monotonic() < deadline:
        time.sleep(0.05)


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        method = request.get("method")
        request_id = request.get("id")
        if method == "initialize":
            params = request.get("params") or {}
            result = {
                "protocolVersion": params.get("protocolVersion", "2025-03-26"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "github-acceptance-fixture", "version": "0.0.0"},
            }
            reply, result = {"jsonrpc": "2.0", "id": request_id}, {"result": result}
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            reply, result = {"jsonrpc": "2.0", "id": request_id}, {"result": {"tools": [{
                "name": "get_pr_acceptance_evidence",
                "description": "SYNTHETIC fixture for protocol proof; not a real GitHub server",
                "inputSchema": {
                    "type": "object",
                    "properties": {"owner": {"type": "string"}, "repo": {"type": "string"},
                                   "pullNumber": {"type": "integer"}},
                    "required": ["owner", "repo", "pullNumber"],
                    "additionalProperties": False,
                },
                "outputSchema": {"type": "object", "additionalProperties": True},
            }]}}
        elif method == "tools/call":
            arguments = (request.get("params") or {}).get("arguments") or {}
            _trace(arguments, os.environ.get("FIXTURE_MODE", "ok"))
            _wait_for_gate()
            reply, result = {"jsonrpc": "2.0", "id": request_id}, {"result": _result_for(arguments)}
        else:
            reply = {"jsonrpc": "2.0", "id": request_id,
                     "error": {"code": -32601, "message": "method_not_found"}}
            result = {}
        if request_id is None:
            continue
        send = dict(reply)
        send.update(result)
        sys.stdout.write(json.dumps(send) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()