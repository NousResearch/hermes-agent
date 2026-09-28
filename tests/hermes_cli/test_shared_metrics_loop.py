"""Learning-loop, delegation and execution-backend shared metrics, read back from the real store."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from hermes_cli.observability import relay_shared_metrics
from hermes_cli.observability import shared_metrics_loop as loop
from hermes_cli.observability.shared_metrics import SharedMetricsStore
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tests.hermes_cli.test_relay_shared_metrics_runtime import direct_runtime  # noqa: F401

_LOOP_METRICS = {
    "hermes.memory.op.count", "hermes.curator.run.count", "hermes.delegation.run.count",
    "hermes.execution_backend.count",
}


def _rows(home: Path, metric: str) -> list[tuple[dict, int]]:
    relay_shared_metrics._reset_for_tests()  # drain the subscriber into the store
    root = home / "telemetry" / "shared_metrics"
    if not (root / "metrics.sqlite3").exists():
        return []
    store = SharedMetricsStore(root / "metrics.sqlite3", root / "outbox")
    return [(row["dimensions"], row["value"]) for row in store.counter_snapshot() if row["metric_name"] == metric]


@pytest.fixture
def home(direct_runtime, tmp_path):  # noqa: F811
    path = tmp_path / "hermes-home"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _memory_store():
    from tools.memory_tool import load_on_disk_store

    return load_on_disk_store()


def test_memory_tool_counts_each_operation_with_its_outcome_and_origin(home):
    from tools.memory_tool import memory_tool
    from tools.skill_provenance import BACKGROUND_REVIEW, reset_current_write_origin, set_current_write_origin

    store = _memory_store()
    assert json.loads(memory_tool("add", content="prefers tabs", store=store))["success"]
    memory_tool("replace", content="x", store=store)  # refused before the store: no old_text
    memory_tool("replace", content="x", old_text="no such entry", store=store)  # the store could not apply it
    memory_tool("explode", store=store)  # unknown action
    token = set_current_write_origin(BACKGROUND_REVIEW)
    try:
        memory_tool(operations=[{"action": "add", "content": "a"}, {"action": "add", "content": "b"}], store=store)
    finally:
        reset_current_write_origin(token)

    rows = {tuple(sorted(d.items())): v for d, v in _rows(home, "hermes.memory.op.count")}
    base = {"provider": "builtin", "origin": "foreground"}
    assert rows == {
        tuple(sorted({**base, "op": "add", "outcome": "success"}.items())): 1,
        tuple(sorted({**base, "op": "replace", "outcome": "rejected"}.items())): 1,
        tuple(sorted({**base, "op": "replace", "outcome": "failed"}.items())): 1,
        tuple(sorted({**base, "op": "other", "outcome": "rejected"}.items())): 1,
        tuple(sorted({**base, "origin": "background_review", "op": "add", "outcome": "success"}.items())): 2,
    }


def test_memory_provider_tools_report_bounded_provider_and_op(home):
    from agent.memory_manager import MemoryManager

    class _Provider:
        def __init__(self, name, tool, result=None, boom=False):
            self.name, self._tool, self._result, self._boom = name, tool, result, boom

        def get_tool_schemas(self):
            return [{"name": self._tool, "description": "", "parameters": {"type": "object", "properties": {}}}]

        def handle_tool_call(self, tool_name, args, **kw):
            if self._boom:
                raise RuntimeError("secret backend detail")
            return self._result

    manager = MemoryManager()
    manager._tool_to_provider = {
        "honcho_search": _Provider("honcho", "honcho_search", '{"results": []}'),
        "acme_private_remember": _Provider("acme-private-memory", "acme_private_remember", boom=True),
    }
    manager.handle_tool_call("honcho_search", {"query": "q"})
    manager.handle_tool_call("acme_private_remember", {"content": "c"})

    rows = sorted((d["provider"], d["op"], d["outcome"]) for d, _ in _rows(home, "hermes.memory.op.count"))
    assert rows == [("honcho", "search", "success"), ("plugin", "add", "failed")]


def test_disabled_shared_metrics_record_no_loop_rows(home, monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config_readonly", lambda: {"telemetry": {"shared_metrics": {"enabled": False}}},
    )
    from tools.memory_tool import memory_tool

    memory_tool("add", content="x", store=_memory_store())
    loop.record_execution_backend("terminal", "local", '{"exit_code": 0, "error": null}')
    loop.record_curator_run(trigger="manual", outcome="success")
    relay_shared_metrics._reset_for_tests()
    assert all(not _rows(home, metric) for metric in _LOOP_METRICS)


def test_async_curator_pass_records_in_the_profile_that_started_it(home, tmp_path):
    from agent import curator

    owner = tmp_path / "owner-profile"
    owner.mkdir()
    done = threading.Event()
    token = set_hermes_home_override(owner)  # the review thread does not inherit this binding
    try:
        curator.run_curator_review(consolidate=False, on_summary=lambda _msg: done.set())
    finally:
        reset_hermes_home_override(token)
    assert done.wait(30)

    assert _rows(home, "hermes.curator.run.count") == []
    zero = {f"{k}_bucket": "0" for k in ("archived", "created", "merged", "patched")}
    assert _rows(owner, "hermes.curator.run.count") == [({**zero, "outcome": "success", "trigger": "manual"}, 1)]


def test_scheduled_curator_pass_held_elsewhere_counts_as_skipped(home, monkeypatch):
    from agent import curator

    monkeypatch.setattr(curator, "should_run_now", lambda: True)
    monkeypatch.setattr(curator, "_claim_run", lambda: False)
    assert curator.maybe_run_curator() is None
    [(dims, value)] = _rows(home, "hermes.curator.run.count")
    assert (dims["trigger"], dims["outcome"], value) == ("scheduled", "skipped", 1)


def test_delegation_fanout_split_into_units_is_one_row_in_the_parent_profile(home, tmp_path):
    call = ["task-a", "task-b", "task-c"]
    loop.begin_delegation_run(call, subagents=3, depth=2)

    other = tmp_path / "unit-thread-profile"
    other.mkdir()

    def unit(results):
        token = set_hermes_home_override(other)  # async units finish on runner threads
        try:
            loop.finish_delegation_unit(call, results, background=True)
        finally:
            reset_hermes_home_override(token)

    for results in ([{"status": "completed"}], [{"status": "completed"}, {"status": "failed"}]):
        worker = threading.Thread(target=unit, args=(results,))
        worker.start()
        worker.join()

    assert _rows(other, "hermes.delegation.run.count") == []
    assert _rows(home, "hermes.delegation.run.count") == [(
        {"depth": "2", "mode": "background", "outcome": "partial", "subagent_count_bucket": "3_to_5"}, 1,
    )]


def test_delegate_task_call_is_one_row_however_many_children(home, monkeypatch):
    import types

    import tools.delegate_tool as dt

    statuses = iter(["completed", "failed", "completed"])
    monkeypatch.setattr(dt, "_run_single_child", lambda task_index, goal, child=None, parent_agent=None, **kw: {
        "task_index": task_index, "status": next(statuses), "summary": "ok", "error": None, "api_calls": 1,
        "duration_seconds": 1,
    })
    monkeypatch.setattr(dt, "_build_child_preserving_parent_tools", lambda **kw: types.SimpleNamespace(tool_progress_callback=None))
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **k: {
        "model": "m", "provider": "openrouter", "base_url": "https://x/v1", "api_key": "k", "api_mode": "chat_completions",
    })
    parent = types.SimpleNamespace(
        session_id="root", model="m", tool_progress_callback=None, _delegate_spinner=None, _safe_print=lambda _l: None,
    )

    dt.delegate_task(tasks=[{"goal": f"worker task number {i}"} for i in range(3)], parent_agent=parent)

    assert _rows(home, "hermes.delegation.run.count") == [(
        {"depth": "1", "mode": "foreground", "outcome": "partial", "subagent_count_bucket": "3_to_5"}, 1,
    )]


def test_terminal_calls_count_against_the_configured_backend(home):
    from tools.terminal_tool import terminal_tool

    assert json.loads(terminal_tool("echo hi"))["exit_code"] == 0
    assert json.loads(terminal_tool("exit 3"))["exit_code"] == 3  # the command failed, the backend did not
    terminal_tool("echo control-plane", _host_local=True)  # Hermes' own children are not user work

    assert _rows(home, "hermes.execution_backend.count") == [
        ({"backend": "local", "error_class": "none", "kind": "terminal", "outcome": "success"}, 2),
    ]


def test_browser_calls_name_the_backend_that_served_them(home):
    served = loop.record_browser_call(lambda legacy: legacy(lambda: '{"success": true}'), lambda: "browserbase")
    assert served == '{"success": true}'
    loop.record_browser_call(lambda legacy: '{"success": false, "error": "x"}', lambda: "browserbase")
    with pytest.raises(RuntimeError):
        loop.record_browser_call(lambda legacy: legacy(_raise), lambda: "some-private-provider")

    rows = sorted((d["backend"], d["outcome"], d["error_class"]) for d, _ in _rows(home, "hermes.execution_backend.count"))
    assert rows == [
        ("browserbase", "success", "none"), ("extension", "failed", "tool_error"), ("other", "failed", "exception"),
    ]


def test_browser_tool_backend_is_resolved_from_the_profile_at_call_time(home, monkeypatch):
    import tools.browser_tool as bt

    handler = bt._routed_handler("browser_back", lambda args, kw: '{"success": true}')
    handler({}, task_id="t")
    monkeypatch.setenv("BROWSER_CDP_URL", "ws://127.0.0.1:9/devtools/browser/x")  # `/browser connect` mid-session
    handler({}, task_id="t")

    rows = sorted((d["backend"], d["outcome"]) for d, _ in _rows(home, "hermes.execution_backend.count"))
    assert rows == [("cdp", "success"), ("local", "success")]


def _raise():
    raise RuntimeError("boom")
