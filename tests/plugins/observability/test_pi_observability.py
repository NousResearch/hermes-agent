from __future__ import annotations

import importlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class FakeSender:
    def __init__(self):
        self.events = []

    def send(self, event):
        self.events.append(event)

    def close(self, *, drain=True):
        pass


def load_plugin(monkeypatch):
    monkeypatch.setenv("HERMES_PI_OBS_ENABLED", "true")
    mod = importlib.import_module("plugins.observability.pi_observability")
    mod.reset_for_tests()
    return mod


def runtime(mod):
    cfg = mod.PiObservabilityConfig(server_url="http://127.0.0.1:9", token="tok", pool="pool-a", tags=("hermes", "test"))
    sender = FakeSender()
    rt = mod.PiObservabilityRuntime(cfg, sender=sender)
    mod._RUNTIME = rt
    return rt, sender


def test_config_parsing_prefers_hermes_env_and_obs_aliases(monkeypatch):
    mod = load_plugin(monkeypatch)
    monkeypatch.setenv("OBS_SERVER_URL", "http://alias:43190")
    monkeypatch.setenv("HERMES_PI_OBS_SERVER_URL", "http://primary:43190/")
    monkeypatch.setenv("OBS_TOKEN", "alias-token")
    monkeypatch.setenv("HERMES_PI_OBS_POOL", "dev")
    monkeypatch.setenv("HERMES_PI_OBS_TAGS", "hermes,local,,mvp")
    monkeypatch.setenv("HERMES_PI_OBS_QUEUE_MAX", "7")

    cfg = mod.load_config_from_env()

    assert cfg.server_url == "http://primary:43190"
    assert cfg.token == "alias-token"
    assert cfg.pool == "dev"
    assert cfg.tags == ("hermes", "local", "mvp")
    assert cfg.queue_max == 7


def test_envelope_generation_has_pi_protocol_fields(monkeypatch):
    mod = load_plugin(monkeypatch)
    cfg = mod.PiObservabilityConfig(pool="pool-a", tags=("hermes",))
    state = mod.SessionState(session_id="s1", cwd="/tmp/project")

    event = mod._envelope("turn_start", {"turn_index": 0}, state=state, config=cfg, provider="openai", model="gpt")

    assert event["type"] == "turn_start"
    assert event["session_id"] == "s1"
    assert event["cwd"] == "/tmp/project"
    assert event["pool"] == "pool-a"
    assert event["tags"] == ["hermes"]
    assert event["provider"] == "openai"
    assert event["model"] == "gpt"
    assert event["seq"] == 0
    assert event["payload"] == {"turn_index": 0}
    assert event["event_id"]
    assert event["ts"].endswith("Z")


def test_usage_and_cost_mapping_from_hermes_summary(monkeypatch):
    mod = load_plugin(monkeypatch)

    usage = mod.normalize_usage(
        {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_tokens": 3,
            "cache_write_tokens": 2,
            "total_tokens": 20,
            "cost_total": 0.0123,
        }
    )

    assert usage == {
        "input": 10,
        "output": 5,
        "cache_read": 3,
        "cache_write": 2,
        "total_tokens": 20,
        "cost_total": 0.0123,
    }


def test_sender_posts_batched_events_and_auth_header(monkeypatch):
    mod = load_plugin(monkeypatch)
    captured = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            captured["path"] = self.path
            captured["auth"] = self.headers.get("Authorization")
            length = int(self.headers.get("Content-Length", "0"))
            captured["body"] = self.rfile.read(length)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ingested":1,"rejected":[]}')

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    try:
        cfg = mod.PiObservabilityConfig(
            server_url=f"http://127.0.0.1:{server.server_port}",
            token="secret",
            batch_size=1,
            batch_interval_s=0.05,
            timeout_s=1,
        )
        mod._post_events(cfg, [{"event_id": "e1"}])
    finally:
        server.server_close()
        thread.join(timeout=1)

    assert captured["path"] == "/events"
    assert captured["auth"] == "Bearer secret"
    assert json.loads(captured["body"].decode()) == [{"event_id": "e1"}]


def test_sender_fail_open_on_unreachable_server(monkeypatch):
    mod = load_plugin(monkeypatch)
    cfg = mod.PiObservabilityConfig(server_url="http://127.0.0.1:9", timeout_s=0.05, batch_size=1)
    sender = mod.EventSender(cfg)
    try:
        sender._post_batch([{"event_id": "e1"}])
    finally:
        sender.close(drain=False)

    assert sender.sent_batches == [[{"event_id": "e1"}]]


def test_registers_required_hooks(monkeypatch):
    mod = load_plugin(monkeypatch)
    hooks = []

    class Ctx:
        def register_hook(self, name, fn):
            hooks.append((name, fn.__name__))

    mod.register(Ctx())

    assert ("pre_api_request", "on_pre_api_request") in hooks
    assert ("post_api_request", "on_post_api_request") in hooks
    assert ("pre_llm_call", "on_pre_llm_call") in hooks
    assert ("post_llm_call", "on_post_llm_call") in hooks
    assert ("pre_tool_call", "on_pre_tool_call") in hooks
    assert ("post_tool_call", "on_post_tool_call") in hooks


def test_hooks_emit_session_turn_assistant_tool_and_error_events(monkeypatch):
    mod = load_plugin(monkeypatch)
    _rt, sender = runtime(mod)

    mod.on_pre_llm_call(session_id="s1", user_message="hello", provider="p", model="m", cwd="/repo")
    mod.on_post_api_request(
        session_id="s1",
        provider="p",
        model="m",
        assistant_response="hi",
        finish_reason="stop",
        usage={"input_tokens": 2, "output_tokens": 3, "cost_total": 0.1},
    )
    mod.on_pre_tool_call(session_id="s1", tool_call_id="tc1", tool_name="terminal", args={"cmd": "echo hi"})
    mod.on_post_tool_call(session_id="s1", tool_call_id="tc1", tool_name="terminal", result={"stdout": "hi", "exit_code": 0})
    mod.on_api_request_error(session_id="s1", error=RuntimeError("boom"))

    types = [event["type"] for event in sender.events]
    assert types == [
        "session_start",
        "agent_start",
        "turn_start",
        "assistant_message",
        "turn_end",
        "tool_call",
        "tool_result",
        "error",
    ]
    assert sender.events[3]["payload"]["usage"] == {
        "input": 2,
        "output": 3,
        "cache_read": 0,
        "cache_write": 0,
        "total_tokens": 5,
        "cost_total": 0.1,
    }
    assert sender.events[5]["payload"] == {
        "tool_call_id": "tc1",
        "tool_name": "terminal",
        "args": {"cmd": "echo hi"},
        "args_truncated": False,
    }
    assert sender.events[6]["payload"]["content_text"] == "hi"
    assert sender.events[6]["payload"]["is_error"] is False
    assert sender.events[7]["payload"]["where"] == "llm"


def test_tool_error_result_emits_error_event(monkeypatch):
    mod = load_plugin(monkeypatch)
    _rt, sender = runtime(mod)

    mod.on_pre_tool_call(session_id="s1", tool_call_id="tc1", tool_name="terminal", args={})
    mod.on_post_tool_call(session_id="s1", tool_call_id="tc1", tool_name="terminal", result={"stdout": "bad", "exit_code": 2})

    assert [event["type"] for event in sender.events][-2:] == ["tool_result", "error"]
    assert sender.events[-1]["payload"]["where"] == "tool:terminal"
