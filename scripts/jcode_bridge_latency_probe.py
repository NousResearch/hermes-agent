#!/usr/bin/env python3
"""Probe local Hermes/jcode bridge overhead without using model APIs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MCP_SERVER = ROOT / "bridges" / "hermes-mcp-server" / "hermes_mcp_server.py"


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _summary(samples_ms: list[float]) -> dict[str, Any]:
    return {
        "count": len(samples_ms),
        "min_ms": round(min(samples_ms), 3) if samples_ms else None,
        "p50_ms": round(_percentile(samples_ms, 0.50), 3),
        "p95_ms": round(_percentile(samples_ms, 0.95), 3),
        "max_ms": round(max(samples_ms), 3) if samples_ms else None,
    }


def _mcp_request(process: subprocess.Popen[str], request: dict[str, Any]) -> dict[str, Any]:
    assert process.stdin is not None
    assert process.stdout is not None
    process.stdin.write(json.dumps(request, ensure_ascii=True) + "\n")
    process.stdin.flush()
    line = process.stdout.readline()
    if not line:
        raise RuntimeError("MCP server closed stdout")
    return json.loads(line)


def probe_mcp_persistent(iterations: int) -> dict[str, Any]:
    process = subprocess.Popen(
        [sys.executable, str(MCP_SERVER), "--mock"],
        cwd=str(ROOT),
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    samples_ms: list[float] = []
    try:
        init_response = _mcp_request(process, {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "latency-probe", "version": "1"},
            },
        })
        list_response = _mcp_request(process, {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        })
        for index in range(iterations):
            request_id = index + 3
            started = time.perf_counter()
            response = _mcp_request(process, {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {
                    "name": "hermes_web_search",
                    "arguments": {
                        "query": "Hermes jcode bridge",
                        "limit": 2,
                    },
                },
            })
            samples_ms.append((time.perf_counter() - started) * 1000)
            if response.get("result", {}).get("isError") is not False:
                raise RuntimeError(f"MCP call returned error response: {response}")
        try:
            _mcp_request(process, {
                "jsonrpc": "2.0",
                "id": iterations + 3,
                "method": "shutdown",
                "params": {},
            })
        except Exception:
            pass
    finally:
        try:
            process.terminate()
            process.wait(timeout=2)
        except Exception:
            process.kill()
    stderr = ""
    if process.stderr is not None:
        stderr = process.stderr.read()
    tools = list_response.get("result", {}).get("tools", [])
    return {
        "success": True,
        "probe": "hermes_mcp_persistent_mock",
        "iterations": iterations,
        "summary": _summary(samples_ms),
        "server_info": init_response.get("result", {}).get("serverInfo"),
        "tool_count": len(tools) if isinstance(tools, list) else None,
        "stderr": stderr,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of persistent MCP tools/call samples.",
    )
    parser.add_argument(
        "--max-p95-ms",
        type=float,
        help="Optional failure threshold for persistent MCP p95 latency.",
    )
    ns = parser.parse_args(argv)
    iterations = max(1, ns.iterations)

    try:
        report = probe_mcp_persistent(iterations)
    except Exception as exc:
        report = {
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "probe": "hermes_mcp_persistent_mock",
        }
    if report.get("success") and ns.max_p95_ms is not None:
        p95 = report.get("summary", {}).get("p95_ms")
        if isinstance(p95, (int, float)) and p95 > ns.max_p95_ms:
            report["success"] = False
            report["error"] = f"p95_ms {p95} exceeded threshold {ns.max_p95_ms}"

    print(json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True))
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
