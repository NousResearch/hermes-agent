#!/usr/bin/env python3
"""Benchmark Hermes runtime hot paths without making model API calls.

The startup benchmark answers "how fast does Hermes open?".  This benchmark
answers "how fast is Hermes while being used?": agent construction, subagent
construction, delegate_task scheduling, parallel tool-call execution, tool
dispatch overhead, and message persistence.

Each case runs in a fresh Python subprocess so module caches, thread pools, and
Hermes home state are isolated between samples.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


COMMON_PREFIX = r"""
import ctypes
import json
import os
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

os.environ["HERMES_HOME"] = tempfile.mkdtemp(prefix="hermes-runtime-bench-")
os.environ.setdefault("OPENAI_API_KEY", "dummy-key")

def rss_bytes():
    if os.name == "nt":
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return int(counters.WorkingSetSize)
        return None
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value * (1024 if sys.platform != "darwin" else 1))
    except Exception:
        return None

def run_measured(fn):
    rss0 = rss_bytes()
    tracemalloc.start()
    start = time.perf_counter()
    payload = fn() or {}
    case_wall = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss1 = rss_bytes()
    elapsed = payload.pop("_elapsed", case_wall)
    payload.update({
        "elapsed": elapsed,
        "case_wall": case_wall,
        "py_alloc_peak_mb": peak / (1024 * 1024),
        "rss_delta_mb": ((rss1 - rss0) / (1024 * 1024)) if (rss0 is not None and rss1 is not None) else None,
        "module_count": len(sys.modules),
    })
    print(json.dumps(payload))
"""


CASES: dict[str, str] = {
    "agent_init_file_terminal": COMMON_PREFIX
    + r"""
def case():
    from run_agent import AIAgent
    agent = AIAgent(
        base_url="http://127.0.0.1:9/v1",
        api_key="dummy-key",
        provider="custom",
        api_mode="chat_completions",
        model="bench-model",
        enabled_toolsets=["file", "terminal", "todo", "delegation"],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        max_iterations=1,
    )
    payload = {
        "tools": len(agent.tools),
        "valid_tools": len(agent.valid_tool_names),
    }
    if hasattr(agent, "close"):
        agent.close()
    return payload

run_measured(case)
""",
    "agent_init_default_tools": COMMON_PREFIX
    + r"""
def case():
    from run_agent import AIAgent
    agent = AIAgent(
        base_url="http://127.0.0.1:9/v1",
        api_key="dummy-key",
        provider="custom",
        api_mode="chat_completions",
        model="bench-model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        max_iterations=1,
    )
    payload = {
        "tools": len(agent.tools),
        "valid_tools": len(agent.valid_tool_names),
    }
    if hasattr(agent, "close"):
        agent.close()
    return payload

run_measured(case)
""",
    "delegate_child_build": COMMON_PREFIX
    + r"""
def case():
    import threading
    from types import SimpleNamespace
    from tools.delegate_tool import _build_child_agent

    parent = SimpleNamespace(
        _delegate_depth=0,
        _subagent_id=None,
        enabled_toolsets=["file", "terminal", "todo", "delegation"],
        valid_tool_names={"read_file", "write_file", "patch", "search_files", "terminal", "process", "todo", "delegate_task"},
        model="bench-model",
        base_url="http://127.0.0.1:9/v1",
        api_key="dummy-key",
        provider="custom",
        api_mode="chat_completions",
        acp_command=None,
        acp_args=[],
        platform="cli",
        reasoning_config=None,
        prefill_messages=None,
        _session_db=None,
        session_id="parent-bench",
        _active_children=[],
        _active_children_lock=threading.Lock(),
        tool_progress_callback=None,
        _fallback_chain=None,
        max_tokens=None,
        providers_allowed=None,
        providers_ignored=None,
        providers_order=None,
        provider_sort=None,
        _print_fn=None,
    )
    child = _build_child_agent(
        task_index=0,
        goal="benchmark child build",
        context="",
        toolsets=None,
        model=None,
        max_iterations=1,
        task_count=1,
        parent_agent=parent,
        role="leaf",
    )
    payload = {
        "tools": len(child.tools),
        "valid_tools": len(child.valid_tool_names),
        "active_children": len(parent._active_children),
    }
    if hasattr(child, "close"):
        child.close()
    return payload

run_measured(case)
""",
    "delegate_task_batch_scheduler": COMMON_PREFIX
    + r"""
def case():
    import threading
    from types import SimpleNamespace
    import tools.delegate_tool as dt

    parent = SimpleNamespace(
        _delegate_depth=0,
        _interrupt_requested=False,
        _active_children=[],
        _active_children_lock=threading.Lock(),
        _delegate_spinner=None,
        _memory_manager=None,
        session_id="parent-bench",
        session_estimated_cost_usd=0.0,
    )

    load_counter = {"count": 0}
    def fake_load_config():
        load_counter["count"] += 1
        return {
            "max_iterations": 1,
            "max_concurrent_children": 3,
            "max_spawn_depth": 2,
            "child_timeout_seconds": 45,
        }
    dt._load_config = fake_load_config
    dt._resolve_delegation_credentials = lambda cfg, parent_agent: {
        "model": None,
        "provider": None,
        "base_url": None,
        "api_key": None,
        "api_mode": None,
        "command": None,
        "args": [],
    }

    def fake_build(task_index, goal, context, toolsets, model, max_iterations, task_count, parent_agent, **kwargs):
        return SimpleNamespace(
            _delegate_role="leaf",
            _delegate_saved_tool_names=[],
            _subagent_id=f"sa-{task_index}",
            close=lambda: None,
        )

    def fake_run(task_index, goal, child=None, parent_agent=None, **kwargs):
        time.sleep(0.05)
        return {
            "task_index": task_index,
            "status": "completed",
            "summary": goal,
            "error": None,
            "api_calls": 0,
            "duration_seconds": 0.05,
            "_child_role": "leaf",
            "_child_cost_usd": 0.0,
        }

    dt._build_child_agent = fake_build
    dt._run_single_child = fake_run

    tasks = [{"goal": f"task {i}"} for i in range(3)]
    start = time.perf_counter()
    for i, task in enumerate(tasks):
        fake_run(i, task["goal"], None, parent)
    sequential = time.perf_counter() - start

    start = time.perf_counter()
    result = json.loads(dt.delegate_task(tasks=tasks, parent_agent=parent))
    delegate_elapsed = time.perf_counter() - start
    return {
        "_elapsed": delegate_elapsed,
        "tasks": len(tasks),
        "sequential_equivalent": sequential,
        "reported_total": result.get("total_duration_seconds"),
        "phase_timings": result.get("phase_timings"),
        "config_loads": load_counter["count"],
        "speedup": sequential / delegate_elapsed if delegate_elapsed else None,
    }

run_measured(case)
""",
    "parallel_tool_batch_sleep": COMMON_PREFIX
    + r"""
def case():
    import json as _json
    import threading
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    import run_agent as ra

    ra.maybe_persist_tool_result = lambda content, **kwargs: content
    ra.enforce_turn_budget = lambda *args, **kwargs: None
    ra.get_active_env = lambda task_id: None

    class Guardrails:
        def before_call(self, name, args):
            return SimpleNamespace(allows_execution=True)
        def after_call(self, name, args, result, failed=False):
            return SimpleNamespace(action="allow", should_halt=False)

    class Hints:
        def check_tool_call(self, name, args):
            return ""

    class FakeFunction:
        def __init__(self, name, arguments):
            self.name = name
            self.arguments = arguments

    class FakeToolCall:
        def __init__(self, idx):
            self.id = f"tc_{idx}"
            self.function = FakeFunction(
                "read_file",
                _json.dumps({"path": f"bench_{idx}.txt"}),
            )

    class FakeAssistantMessage:
        def __init__(self, tool_calls):
            self.tool_calls = tool_calls

    class Stub:
        _interrupt_requested = False
        quiet_mode = True
        verbose_logging = False
        log_prefix = ""
        log_prefix_chars = 200
        tool_progress_callback = None
        tool_start_callback = None
        tool_complete_callback = None
        valid_tool_names = set()
        _todo_store = MagicMock()
        _session_db = None
        session_id = "bench"
        _memory_manager = None
        _checkpoint_mgr = MagicMock(enabled=False)
        _subdirectory_hints = Hints()
        _tool_guardrails = Guardrails()
        _current_tool = None
        _print_fn = print

        def __init__(self):
            self._tool_worker_threads = set()
            self._tool_worker_threads_lock = threading.Lock()

        def _touch_activity(self, desc):
            pass
        def _vprint(self, msg, force=False):
            pass
        def _safe_print(self, msg):
            pass
        def _should_emit_quiet_tool_messages(self):
            return False
        def _should_start_quiet_spinner(self):
            return False
        def _apply_pending_steer_to_tool_results(self, *args, **kwargs):
            pass
        def _append_guardrail_observation(self, name, args, result, failed=False):
            return result
        def _guardrail_block_result(self, decision):
            return _json.dumps({"error": "blocked"})
        def _invoke_tool(self, function_name, function_args, effective_task_id, tool_call_id=None, messages=None, pre_tool_block_checked=False):
            time.sleep(0.05)
            return _json.dumps({"ok": True, "path": function_args.get("path")})

    stub = Stub()
    stub._execute_tool_calls_concurrent = ra.AIAgent._execute_tool_calls_concurrent.__get__(stub)

    calls = [FakeToolCall(i) for i in range(6)]
    msg = FakeAssistantMessage(calls)

    start = time.perf_counter()
    for call in calls:
        args = _json.loads(call.function.arguments)
        stub._invoke_tool(call.function.name, args, "bench", call.id)
    sequential = time.perf_counter() - start

    messages = []
    start = time.perf_counter()
    stub._execute_tool_calls_concurrent(msg, messages, "bench")
    concurrent = time.perf_counter() - start
    return {
        "_elapsed": concurrent,
        "tools": len(calls),
        "sequential_equivalent": sequential,
        "concurrent": concurrent,
        "messages_appended": len(messages),
        "speedup": sequential / concurrent if concurrent else None,
    }

run_measured(case)
""",
    "tool_dispatch_noop": COMMON_PREFIX
    + r"""
def case():
    import json as _json
    from model_tools import handle_function_call
    from tools.registry import registry

    registry.register(
        name="_bench_noop",
        toolset="_bench",
        schema={
            "name": "_bench_noop",
            "description": "Benchmark no-op.",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=lambda args, **kwargs: _json.dumps({"ok": True}),
    )
    iterations = 3000
    start = time.perf_counter()
    for _ in range(iterations):
        handle_function_call("_bench_noop", {}, task_id="bench")
    elapsed = time.perf_counter() - start
    return {
        "_elapsed": elapsed,
        "iterations": iterations,
        "per_call_ms": (elapsed / iterations) * 1000,
    }

run_measured(case)
""",
    "openrouter_metadata_disk_cache": COMMON_PREFIX
    + r"""
def case():
    os.environ["HERMES_OPENROUTER_METADATA_DISK_CACHE"] = "1"
    os.environ["HERMES_OPENROUTER_METADATA_CACHE_TTL"] = "3600"

    import agent.model_metadata as mm

    cache_path = Path(os.environ["HERMES_HOME"]) / "cache" / "openrouter_model_metadata.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    models = {
        f"provider/model-{idx}": {
            "context_length": 128000 + idx,
            "max_completion_tokens": 4096,
            "name": f"Model {idx}",
            "pricing": {},
        }
        for idx in range(500)
    }
    cache_path.write_text(
        json.dumps({
            "version": mm._MODEL_METADATA_DISK_CACHE_VERSION,
            "source_url": "bench",
            "fetched_at": time.time(),
            "models": models,
        }),
        encoding="utf-8",
    )
    mm.requests.get = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("network should not be used"))

    iterations = 100
    start = time.perf_counter()
    result = {}
    for _ in range(iterations):
        mm._model_metadata_cache = {}
        mm._model_metadata_cache_time = 0
        result = mm.fetch_model_metadata()
    elapsed = time.perf_counter() - start
    return {
        "_elapsed": elapsed,
        "iterations": iterations,
        "models": len(result),
        "per_lookup_ms": (elapsed / iterations) * 1000,
    }

run_measured(case)
""",
    "parallel_guard_read_files": COMMON_PREFIX
    + r"""
def case():
    import json as _json
    from run_agent import _should_parallelize_tool_batch

    class FakeFunction:
        def __init__(self, name, arguments):
            self.name = name
            self.arguments = arguments

    class FakeToolCall:
        def __init__(self, idx):
            self.function = FakeFunction(
                "read_file",
                _json.dumps({"path": f"bench_{idx}.txt"}),
            )

    calls = [FakeToolCall(i) for i in range(8)]
    iterations = 10000
    start = time.perf_counter()
    allowed = False
    for _ in range(iterations):
        allowed = _should_parallelize_tool_batch(calls)
    elapsed = time.perf_counter() - start
    return {
        "_elapsed": elapsed,
        "tools": len(calls),
        "iterations": iterations,
        "allowed": allowed,
        "per_batch_ms": (elapsed / iterations) * 1000,
    }

run_measured(case)
""",
    "session_append_messages_batch": COMMON_PREFIX
    + r"""
def case():
    from hermes_state import SessionDB

    tool_calls = [
        {"id": "c1", "function": {"name": "web_search", "arguments": "{}"}},
        {"id": "c2", "function": {"name": "read_file", "arguments": "{}"}},
    ]
    messages = []
    for i in range(80):
        messages.append({"role": "user", "content": f"prompt {i}"})
        messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
        messages.append({"role": "tool", "content": "ok", "tool_name": "web_search"})

    tmp = Path(tempfile.mkdtemp())
    db = SessionDB(db_path=tmp / "loop.db")
    db.create_session("s1", source="bench")
    start = time.perf_counter()
    for msg in messages:
        db.append_message("s1", **msg)
    loop = time.perf_counter() - start
    db.close()

    db = SessionDB(db_path=tmp / "batch.db")
    db.create_session("s1", source="bench")
    start = time.perf_counter()
    db.append_messages("s1", messages)
    batch = time.perf_counter() - start
    db.close()

    return {
        "_elapsed": batch,
        "messages": len(messages),
        "loop": loop,
        "batch": batch,
        "speedup": loop / batch if batch else None,
    }

run_measured(case)
""",
}


def _run_case(case: str, code: str) -> dict:
    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    wall = time.perf_counter() - start
    if proc.returncode != 0:
        return {
            "case": case,
            "ok": False,
            "elapsed": wall,
            "stderr": proc.stderr.strip()[-4000:],
            "stdout": proc.stdout.strip()[-2000:],
        }
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    try:
        payload = json.loads(lines[-1]) if lines else {}
    except json.JSONDecodeError:
        payload = {"stdout": proc.stdout.strip()[-4000:]}
    payload.update({"case": case, "ok": True, "wall": wall})
    return payload


def _summarize(samples: list[dict]) -> dict:
    elapsed = [float(s["elapsed"]) for s in samples if s.get("ok")]
    if not elapsed:
        return {"ok": False, "samples": samples}
    return {
        "ok": True,
        "min": min(elapsed),
        "median": statistics.median(elapsed),
        "mean": statistics.mean(elapsed),
        "max": max(elapsed),
        "samples": samples,
    }


def _fmt_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}s"


def _notes(sample: dict) -> str:
    notes = []
    for key in ("tools", "valid_tools", "tasks", "messages", "iterations", "models"):
        if key in sample:
            notes.append(f"{key}={sample[key]}")
    if sample.get("allowed") is not None:
        notes.append(f"allowed={sample['allowed']}")
    if "per_call_ms" in sample:
        notes.append(f"per_call={sample['per_call_ms']:.4f}ms")
    if "per_batch_ms" in sample:
        notes.append(f"per_batch={sample['per_batch_ms']:.4f}ms")
    if "per_lookup_ms" in sample:
        notes.append(f"per_lookup={sample['per_lookup_ms']:.4f}ms")
    if "sequential_equivalent" in sample:
        notes.append(f"seq={sample['sequential_equivalent']:.4f}s")
    if "concurrent" in sample:
        notes.append(f"concurrent={sample['concurrent']:.4f}s")
    if "loop" in sample and "batch" in sample:
        notes.append(f"loop={sample['loop']:.4f}s")
        notes.append(f"batch={sample['batch']:.4f}s")
    if "config_loads" in sample:
        notes.append(f"config_loads={sample['config_loads']}")
    if isinstance(sample.get("phase_timings"), dict):
        phases = sample["phase_timings"]
        short_phases = []
        for key in ("config_seconds", "child_build_seconds", "child_run_seconds"):
            if key in phases:
                short_phases.append(f"{key.replace('_seconds', '')}={phases[key]:.4f}s")
        if short_phases:
            notes.append("phases:" + "/".join(short_phases))
    if "case_wall" in sample:
        notes.append(f"case_wall={sample['case_wall']:.4f}s")
    if "speedup" in sample and sample["speedup"]:
        notes.append(f"speedup={sample['speedup']:.2f}x")
    if "py_alloc_peak_mb" in sample:
        notes.append(f"py_peak={sample['py_alloc_peak_mb']:.1f}MB")
    if sample.get("rss_delta_mb") is not None:
        notes.append(f"rss_delta={sample['rss_delta_mb']:.1f}MB")
    if "module_count" in sample:
        notes.append(f"modules={sample['module_count']}")
    return ", ".join(notes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--samples", type=int, default=3)
    parser.add_argument(
        "--case",
        action="append",
        choices=sorted(CASES),
        help="Run only the named case. Repeat to run multiple cases.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown.")
    args = parser.parse_args()

    selected_cases = {name: CASES[name] for name in (args.case or CASES.keys())}
    results = {
        name: _summarize([_run_case(name, code) for _ in range(args.samples)])
        for name, code in selected_cases.items()
    }

    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
        return 0

    print("| case | median | min | max | notes |")
    print("| --- | ---: | ---: | ---: | --- |")
    for name, summary in results.items():
        if not summary.get("ok"):
            print(f"| {name} | error | error | error | rerun with --json for stderr |")
            continue
        sample = next((s for s in summary["samples"] if s.get("ok")), {})
        print(
            f"| {name} | {_fmt_seconds(summary['median'])} | "
            f"{_fmt_seconds(summary['min'])} | {_fmt_seconds(summary['max'])} | "
            f"{_notes(sample)} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
