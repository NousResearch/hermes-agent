#!/usr/bin/env python3
"""Benchmark: UTCP direct SDK vs. UTCP through Hermes vs. MCP through Hermes.

All three paths execute an equivalent "echo" operation against a loopback
provider, so the measured cost is protocol + Hermes plumbing, not external API
latency.

    # Full matrix (slow; MCP baselines included)
    python scripts/benchmark_utcp.py --out utcp-bench.json

    # Skip MCP (no mcp SDK / faster)
    python scripts/benchmark_utcp.py --skip-mcp

Reported per configuration:
  * discovery_ms      — register manual + register tools (UTCP); connect + list
                        tools (MCP)
  * catalog_ms        — registry.get_definitions() for the whole catalog
  * exec p50/p95/p99  — single-call latency over N sequential calls
  * throughput_rps    — concurrent calls (--concurrency workers) divided by wall
  * cpu_ms            — process CPU consumed by the measured block
  * rss_mb            — peak RSS during the run

Methodology: warm-up calls are excluded; latency samples are wall-clock around
one call; percentiles use nearest-rank on the sorted sample. The mock providers
are stdlib HTTP servers on 127.0.0.1 (no TLS, no external network).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "tests" / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("HERMES_HOME", str(REPO_ROOT / ".bench-hermes-home"))

from tests.tools.utcp_mock_provider import MockUtcpProvider, utcp_tool as mock_tool


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((pct / 100.0) * len(ordered) + 0.5) - 1))
    return ordered[index]


def rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return float("nan")


def cpu_ms() -> float:
    times = os.times()
    return (times.user + times.system) * 1000.0


def make_tools(count: int) -> list[dict[str, Any]]:
    """N distinct tools that all hit the same echo endpoint."""
    tools = []
    for index in range(count):
        tools.append(
            mock_tool(
                f"op_{index:04d}",
                "/echo",
                description=f"Benchmark operation {index}",
                inputs={
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
            )
        )
    return tools


# ---------------------------------------------------------------------------
# UTCP configurations
# ---------------------------------------------------------------------------


def bench_utcp(
    *,
    providers: int,
    tools_per_provider: int,
    calls: int,
    concurrency: int,
    direct_sdk: bool,
) -> dict[str, Any]:
    from tools.registry import registry
    from tools import utcp_tool

    utcp_tool.shutdown_utcp_providers()
    utcp_tool.reset_utcp_state_for_tests()

    servers = [MockUtcpProvider(make_tools(tools_per_provider)).start() for _ in range(providers)]
    section = {
        "enabled": True,
        "discovery_timeout": 30,
        "providers": [
            {
                "name": f"bench{i}",
                "call_template_type": "http",
                "url": server.manual_url,
                "timeout": 30,
                "connect_timeout": 30,
            }
            for i, server in enumerate(servers)
        ],
    }
    result: dict[str, Any] = {"providers": providers, "tools_per_provider": tools_per_provider}
    try:
        original = utcp_tool._utcp_section
        utcp_tool._utcp_section = lambda: section
        try:
            mem_before = rss_mb()
            cpu_before = cpu_ms()
            started = time.perf_counter()
            registered = utcp_tool.discover_utcp_tools()
            discovery_ms = (time.perf_counter() - started) * 1000.0
            cpu_discovery = cpu_ms() - cpu_before
            mem_after_discovery = rss_mb()

            started = time.perf_counter()
            definitions = registry.get_definitions(set(registered))
            catalog_ms = (time.perf_counter() - started) * 1000.0

            target = registered[0]
            args = {"message": "bench"}
            if direct_sdk:
                state = utcp_tool._states["bench0"]
                spec = next(s for s in state.specs if s.registry_name == target)
                call = lambda: utcp_tool._run(
                    utcp_tool._invoke(state, spec, args), timeout=30
                )
            else:
                call = lambda: registry.dispatch(target, args)

            for _ in range(5):  # warm-up
                call()

            latencies: list[float] = []
            cpu_before = cpu_ms()
            started = time.perf_counter()
            for _ in range(calls):
                tick = time.perf_counter()
                call()
                latencies.append((time.perf_counter() - tick) * 1000.0)
            exec_wall = time.perf_counter() - started
            cpu_exec = cpu_ms() - cpu_before

            cpu_before = cpu_ms()
            started = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(call) for _ in range(calls)]
                for future in futures:
                    future.result()
            conc_wall = time.perf_counter() - started
            cpu_conc = cpu_ms() - cpu_before

            result.update(
                {
                    "mode": "utcp-direct-sdk" if direct_sdk else "utcp-via-hermes",
                    "tools": len(registered),
                    "schema_tools": len(definitions),
                    "discovery_ms": discovery_ms,
                    "discovery_cpu_ms": cpu_discovery,
                    "catalog_ms": catalog_ms,
                    "exec_p50_ms": statistics.median(latencies),
                    "exec_p95_ms": percentile(latencies, 95),
                    "exec_p99_ms": percentile(latencies, 99),
                    "exec_mean_ms": statistics.fmean(latencies),
                    "exec_rps": calls / exec_wall if exec_wall else float("nan"),
                    "exec_cpu_ms": cpu_exec,
                    "concurrency": concurrency,
                    "concurrent_rps": calls / conc_wall if conc_wall else float("nan"),
                    "concurrent_cpu_ms": cpu_conc,
                    "rss_mb_before": mem_before,
                    "rss_mb_after_discovery": mem_after_discovery,
                    "rss_mb_peak": rss_mb(),
                }
            )
        finally:
            utcp_tool._utcp_section = original
    finally:
        utcp_tool.shutdown_utcp_providers()
        utcp_tool.reset_utcp_state_for_tests()
        for server in servers:
            server.stop()
    return result


# ---------------------------------------------------------------------------
# MCP baseline
# ---------------------------------------------------------------------------


def bench_mcp(
    *,
    tools: int,
    calls: int,
    concurrency: int,
) -> Optional[dict[str, Any]]:
    try:
        import mcp  # noqa: F401
    except ImportError:
        return None
    try:  # current layout: discovery/lifecycle split out of mcp_tool
        from tools.mcp_tool_discovery import register_mcp_servers
        from tools.mcp_tool_lifecycle import shutdown_mcp_servers
    except ImportError:  # older single-module layout
        from tools.mcp_tool import register_mcp_servers, shutdown_mcp_servers

    server_script = REPO_ROOT / "tests" / "tools" / "mcp_bench_server.py"
    config = {
        "benchmcp": {
            "command": sys.executable,
            "args": [str(server_script)],
            "env": {"BENCH_MCP_TOOLS": str(tools)},
            "timeout": 60,
            "connect_timeout": 60,
        }
    }

    mem_before = rss_mb()
    cpu_before = cpu_ms()
    started = time.perf_counter()
    registered = register_mcp_servers(config)
    discovery_ms = (time.perf_counter() - started) * 1000.0
    cpu_discovery = cpu_ms() - cpu_before
    mem_after = rss_mb()
    if not registered:
        shutdown_mcp_servers()
        return None

    try:
        from tools.registry import registry

        started = time.perf_counter()
        definitions = registry.get_definitions(set(registered))
        catalog_ms = (time.perf_counter() - started) * 1000.0

        # Pick an echo-style tool so the work matches the UTCP path.
        target = next((name for name in registered if name.endswith("mcp_op_0000")), registered[0])
        call = lambda: registry.dispatch(target, {"message": "bench"})
        for _ in range(5):
            call()

        latencies: list[float] = []
        cpu_before = cpu_ms()
        started = time.perf_counter()
        for _ in range(calls):
            tick = time.perf_counter()
            call()
            latencies.append((time.perf_counter() - tick) * 1000.0)
        exec_wall = time.perf_counter() - started
        cpu_exec = cpu_ms() - cpu_before

        cpu_before = cpu_ms()
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(call) for _ in range(calls)]
            for future in futures:
                future.result()
        conc_wall = time.perf_counter() - started
        cpu_conc = cpu_ms() - cpu_before

        return {
            "mode": "mcp-via-hermes",
            "tools": len(registered),
            "schema_tools": len(definitions),
            "discovery_ms": discovery_ms,
            "discovery_cpu_ms": cpu_discovery,
            "catalog_ms": catalog_ms,
            "exec_p50_ms": statistics.median(latencies),
            "exec_p95_ms": percentile(latencies, 95),
            "exec_p99_ms": percentile(latencies, 99),
            "exec_mean_ms": statistics.fmean(latencies),
            "exec_rps": calls / exec_wall if exec_wall else float("nan"),
            "exec_cpu_ms": cpu_exec,
            "concurrency": concurrency,
            "concurrent_rps": calls / conc_wall if conc_wall else float("nan"),
            "concurrent_cpu_ms": cpu_conc,
            "rss_mb_before": mem_before,
            "rss_mb_after_discovery": mem_after,
            "rss_mb_peak": rss_mb(),
        }
    finally:
        shutdown_mcp_servers()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _run_single(args) -> dict[str, Any]:
    """Run one configuration in this process (called by the matrix driver)."""
    if args.single == "utcp":
        return bench_utcp(
            providers=args.providers,
            tools_per_provider=args.tools,
            calls=args.calls,
            concurrency=args.concurrency,
            direct_sdk=args.direct_sdk,
        )
    if args.single == "mcp":
        result = bench_mcp(tools=args.tools, calls=args.calls, concurrency=args.concurrency)
        if result is None:
            raise SystemExit("mcp baseline unavailable")
        return result
    raise SystemExit(f"unknown --single mode {args.single!r}")


def _spawn(args, *, mode: str, providers: int, tools: int, direct_sdk: bool) -> Optional[dict[str, Any]]:
    """Run one measurement point in a fresh interpreter (clean RSS + CPU)."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single", mode,
        "--providers", str(providers),
        "--tools", str(tools),
        "--calls", str(args.calls),
        "--concurrency", str(args.concurrency),
        "--json-stdout",
    ]
    if direct_sdk:
        cmd.append("--direct-sdk")
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, check=False)
    if completed.returncode != 0:
        tail = (completed.stderr or "").strip().splitlines()
        print(
            f"[skip] {mode} providers={providers} tools={tools}: {tail[-1] if tail else 'failed'}",
            file=sys.stderr,
        )
        return None
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="utcp-bench.json", help="JSON output path")
    parser.add_argument("--calls", type=int, default=200, help="measured calls per config")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--skip-mcp", action="store_true")
    parser.add_argument("--skip-direct", action="store_true", help="skip the raw-SDK baseline")
    # Single-point mode (used by the matrix driver; usually not called by hand).
    parser.add_argument("--single", choices=["utcp", "mcp"], help=argparse.SUPPRESS)
    parser.add_argument("--providers", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--tools", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--direct-sdk", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--json-stdout", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.single:
        print(json.dumps(_run_single(args)))
        return 0

    # Warm the SDK once so the first measured point isn't paying import cost
    # that every later point amortizes. The warm-up result is discarded and is
    # NOT reported as a measurement.
    _spawn(args, mode="utcp", providers=1, tools=1, direct_sdk=args.skip_direct)

    configs: list[tuple[int, int]] = [
        (1, 1),      # single provider, single tool
        (1, 100),
        (1, 500),
        (1, 1000),
        (5, 20),     # 5 providers / 100 tools
        (20, 25),    # 20 providers / 500 tools
    ]

    results: list[dict[str, Any]] = []
    for providers, tools_per_provider in configs:
        if not args.skip_direct:
            row = _spawn(args, mode="utcp", providers=providers, tools=tools_per_provider, direct_sdk=True)
            if row:
                results.append(row)
        row = _spawn(args, mode="utcp", providers=providers, tools=tools_per_provider, direct_sdk=False)
        if row:
            results.append(row)
        print(f"utcp done: providers={providers} tools/provider={tools_per_provider}", file=sys.stderr)

    if not args.skip_mcp:
        for tools in (100, 500):
            row = _spawn(args, mode="mcp", providers=1, tools=tools, direct_sdk=False)
            if row:
                results.append(row)
                print(f"mcp done: tools={tools}", file=sys.stderr)

    payload = {
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "utcp": _pkg_version("utcp"),
            "utcp_http": _pkg_version("utcp-http"),
            "mcp": _pkg_version("mcp"),
            "calls_per_point": args.calls,
            "concurrency": args.concurrency,
            "isolation": "one subprocess per measurement point",
        },
        "results": results,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    header = (
        f"{'mode':<18}{'prov':>5}{'tools':>7}{'disc ms':>9}{'cat ms':>8}"
        f"{'p50':>8}{'p95':>8}{'p99':>8}{'rps':>9}{'conc rps':>10}{'rss MB':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{row['mode']:<18}{row.get('providers', 1):>5}{row['tools']:>7}"
            f"{row['discovery_ms']:>9.1f}{row['catalog_ms']:>8.2f}"
            f"{row['exec_p50_ms']:>8.2f}{row['exec_p95_ms']:>8.2f}{row['exec_p99_ms']:>8.2f}"
            f"{row['exec_rps']:>9.1f}{row['concurrent_rps']:>10.1f}{row['rss_mb_peak']:>8.1f}"
        )
    print(f"\nWrote {args.out}")
    return 0


def _pkg_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
