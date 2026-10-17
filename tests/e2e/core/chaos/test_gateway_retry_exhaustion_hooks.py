"""Real-gateway lifecycle hooks for terminal provider retry failures."""

from __future__ import annotations

import json
from pathlib import Path

from tests.e2e.core.chaos._gateway_harness import GatewayProc
from tests.e2e.core.chaos._helpers import new_tag
from tests.fakes.fake_llm_provider import Error, FakeLLMServer, Text
from hermes_cli.config_defaults import DEFAULT_CONFIG


def _install_hook_trace_plugin(gateway: GatewayProc, trace_path: Path) -> None:
    plugin_init = gateway.hermes_home / "plugins" / "chaos-fake" / "__init__.py"
    plugin_init.write_text(
        "from tests.e2e.core.chaos._gateway_fake_platform import register as _register_platform\n"
        "import json\n"
        f"_TRACE = {str(trace_path)!r}\n"
        "def _record(name, **kwargs):\n"
        "    fields = ('session_id', 'task_id', 'turn_id', 'api_request_id', 'api_call_count',\n"
        "              'retry_count', 'status_code', 'completed', 'failed', 'interrupted')\n"
        "    row = {'event': name, **{key: kwargs.get(key) for key in fields}}\n"
        "    with open(_TRACE, 'a', encoding='utf-8') as stream:\n"
        "        stream.write(json.dumps(row) + '\\n')\n"
        "def register(ctx):\n"
        "    _register_platform(ctx)\n"
        "    _record('plugin_register')\n"
        "    for name in ('pre_llm_call', 'pre_api_request', 'api_request_error', 'on_session_end'):\n"
        "        ctx.register_hook(name, lambda _name=name, **kwargs: _record(_name, **kwargs))\n",
        encoding="utf-8",
    )


def _last_user_text(record: dict) -> str:
    for message in reversed(record["body"].get("messages") or []):
        if message.get("role") == "user":
            content = message.get("content")
            return content if isinstance(content, str) else str(content)
    return ""


def _run_gateway_turn(
    tmp_path: Path, server: FakeLLMServer
) -> tuple[GatewayProc, Path, str, int]:
    root = tmp_path / "gateway"
    trace = tmp_path / "hook-trace.jsonl"
    gateway = GatewayProc(
        root,
        server.base_url,
        new_tag(),
        cfg={"model": "fake-model", "api_max_retries": 2},
        extra_config=f"_config_version: {DEFAULT_CONFIG['_config_version']}\n",
    )
    _install_hook_trace_plugin(gateway, trace)
    gateway.start()
    warm_start = gateway.mark()
    gateway.send_user("are you there [[probe:retry-hooks:0]]")
    warm_answer = gateway.wait_for(
        lambda event: (
            event.ev == "send" and "alive retry-hooks 0" in event.data.get("text", "")
        ),
        90,
        "gateway warm-up answer",
        since=warm_start,
    )
    assert warm_answer is not None, (
        f"{gateway.log_tail()}\nmain_requests={server.main_requests()}"
    )
    idle = gateway.wait_for(
        lambda event: event.ev == "hb" and not event.data.get("busy"),
        90,
        "gateway warm-up session release",
        since=warm_answer.idx,
    )
    assert idle is not None, gateway.log_tail()
    target_trace_start = len(trace.read_text(encoding="utf-8").splitlines())
    marker = gateway.mark()
    message_id = gateway.send_user("please work [[terminal-hook-target]]")
    server.wait_for_requests(3, timeout=90)
    idle = gateway.wait_for(
        lambda event: event.ev == "hb" and not event.data.get("busy"),
        90,
        "gateway target session release",
        since=marker,
    )
    assert idle is not None, gateway.log_tail()
    return gateway, trace, message_id, target_trace_start


def _trace_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_gateway_retry_exhaustion_fires_one_terminal_hook(tmp_path):
    """An exhausted HTTP 500 turn reaches on_session_end exactly once with its identity."""

    def respond(record):
        return (
            Text("alive retry-hooks 0")
            if "[[probe:retry-hooks:0]]" in _last_user_text(record)
            else Error(500)
        )

    with FakeLLMServer(respond) as server:
        gateway = None
        try:
            gateway, trace, _message_id, target_trace_start = _run_gateway_turn(
                tmp_path, server
            )
            assert trace.exists(), (
                f"{gateway.log_tail()}\nmain_requests={len(server.main_requests())}"
            )
            rows = _trace_rows(trace)[target_trace_start:]
            names = [row["event"] for row in rows]
            llm_starts = [row for row in rows if row["event"] == "pre_llm_call"]
            errors = [row for row in rows if row["event"] == "api_request_error"]
            terminal = [row for row in rows if row["event"] == "on_session_end"]

            assert len(llm_starts) == 1, (
                rows,
                len(server.main_requests()),
                gateway.log_tail(),
            )
            assert names.count("pre_api_request") == len(errors) >= 1
            assert len(errors) >= 1
            assert all(row["status_code"] == 500 for row in errors)
            assert len(terminal) == 1, rows
            end = terminal[0]
            assert end["completed"] is False
            assert end["failed"] is True
            assert end["interrupted"] is False
            assert end["session_id"] == llm_starts[0]["session_id"]
            assert end["turn_id"] == llm_starts[0]["turn_id"]
            assert names[-1] == "on_session_end"
        finally:
            if gateway is not None:
                gateway.stop()


def test_gateway_retry_recovery_does_not_end_session_mid_retry(tmp_path):
    """A retry that succeeds has one successful terminal hook, after the final request."""
    calls = 0

    def respond(_record):
        nonlocal calls
        if "[[probe:retry-hooks:0]]" in _last_user_text(_record):
            return Text("alive retry-hooks 0")
        calls += 1
        return Error(500) if calls == 1 else Text("recovered")

    with FakeLLMServer(respond) as server:
        gateway = None
        try:
            gateway, trace, _message_id, target_trace_start = _run_gateway_turn(
                tmp_path, server
            )
            assert trace.exists(), (
                f"{gateway.log_tail()}\nmain_requests={len(server.main_requests())}"
            )
            rows = _trace_rows(trace)[target_trace_start:]
            names = [row["event"] for row in rows]
            llm_starts = [row for row in rows if row["event"] == "pre_llm_call"]
            terminal = [row for row in rows if row["event"] == "on_session_end"]

            assert len(server.main_requests()) == 3
            assert calls == 2, (
                "the target turn must encounter one provider error and then recover"
            )
            assert len(terminal) == 1
            assert terminal[0]["completed"] is True
            assert terminal[0]["failed"] is False
            assert terminal[0]["interrupted"] is False
            assert terminal[0]["turn_id"] == llm_starts[0]["turn_id"]
            assert names[-1] == "on_session_end"
        finally:
            if gateway is not None:
                gateway.stop()
