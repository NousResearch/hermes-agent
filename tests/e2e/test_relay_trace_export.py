"""The Relay trace a real turn exports: what reaches an OTLP collector is what operators analyse.

A real ``AIAgent`` runs a two-request tool turn against a recording fake provider with a
``relay-plugins.toml`` that exports OpenInference spans over OTLP/HTTP to an in-process collector.
Assertions read the decoded protobuf spans only, so they cover every layer between the provider
response and the collector: Hermes' stream accumulators, the Relay codec, OpenInference
projection and the exporter. Contracts (one test per provider wire):

* one LLM span per provider request, each with a distinct ``api_request_id`` and
  ``call_role=primary``;
* the span's token counts equal what the provider reported, cache read and cache write included —
  the counts cache-reuse analysis is computed from.
"""

from __future__ import annotations

import os
import threading
from importlib import metadata
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("nemo_relay")
from packaging.version import Version  # noqa: E402  (pinned beside nemo-relay on 3.14)

if Version(metadata.version("nemo-relay")) < Version("0.9"):  # the binding has no __version__
    pytest.skip("relay-plugins.toml needs nemo-relay 0.9 (pyproject's supported binding)", allow_module_level=True)
trace_pb = pytest.importorskip("opentelemetry.proto.collector.trace.v1.trace_service_pb2")

from tests.e2e.core.history._helpers import NO_BACKGROUND_REVIEW, OFFLINE_CONFIG  # noqa: E402

FLUSH_TIMEOUT = 20.0


class _Collector:
    """Loopback OTLP/HTTP trace receiver keeping each LLM span's attributes."""

    def __init__(self) -> None:
        self.spans: list[dict[str, Any]] = []
        spans = self.spans

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                request = trace_pb.ExportTraceServiceRequest.FromString(
                    self.rfile.read(int(self.headers["Content-Length"])))
                for resource in request.resource_spans:
                    for scope in resource.scope_spans:
                        for span in scope.spans:
                            spans.append({a.key: getattr(a.value, a.value.WhichOneof("value"))
                                          for a in span.attributes if a.value.WhichOneof("value")})
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}"

    def llm_spans(self, count: int) -> list[dict[str, Any]]:
        deadline, llm = time.monotonic() + FLUSH_TIMEOUT, []
        while time.monotonic() < deadline:
            llm = [s for s in self.spans if s.get("openinference.span.kind") == "LLM"]
            if len(llm) >= count:
                return sorted(llm, key=lambda s: s["openinference.metadata.api_request_id"])
            time.sleep(0.1)
        raise AssertionError(f"collector received {len(llm)} LLM spans, expected {count}")

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=10)


@pytest.fixture
def relay_home(monkeypatch):
    from agent import relay_runtime

    home = Path(os.environ["HERMES_HOME"])
    collector = _Collector()
    plugins = home / "relay-plugins.toml"
    plugins.write_text(
        'version = 1\n\n[[components]]\nkind = "observability"\nenabled = true\n\n'
        "[components.config]\nversion = 4\nenable_full_payloads = false\n\n"
        "[components.config.opentelemetry]\nenabled = true\n\n"
        "[[components.config.opentelemetry.endpoints]]\n"
        f'type = "openinference"\nendpoint = "{collector.endpoint}"\nscheduled_delay_millis = 100\n')
    monkeypatch.setenv("HERMES_NEMO_RELAY_PLUGINS_TOML", str(plugins))
    # The Anthropic credential pool also seeds from Claude Code's shared login file under HOME,
    # which the suite keeps real; point it inside the sandbox.
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(home / "claude"))
    relay_runtime._reset_for_tests()
    yield home, collector
    relay_runtime._reset_for_tests()
    collector.close()


def _run_turn(agent) -> None:
    from agent import relay_runtime

    try:
        agent.run_conversation("run echo", conversation_history=[], task_id="relay-trace")
    finally:
        agent.close()
        relay_runtime._reset_for_tests()  # shuts the host down, flushing the exporter


def _assert_trace(spans: list[dict[str, Any]], reported: list[dict[str, int]]) -> None:
    assert len(spans) == len(reported)
    assert len({s["openinference.metadata.api_request_id"] for s in spans}) == len(spans)
    assert {s["openinference.metadata.call_role"] for s in spans} == {"primary"}
    for span, usage in zip(spans, reported):
        counts = {key: span.get(f"llm.token_count.{key}") for key in usage}
        assert counts == usage, f"span token counts {counts} != provider-reported {usage}"


def test_chat_completions_trace_carries_provider_usage(relay_home):
    from run_agent import AIAgent
    from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall, write_hermes_home

    home, collector = relay_home
    with FakeLLMServer([ToolCall("terminal", {"command": "echo hi"}), Text("done", cached_tokens=80)]) as srv:
        write_hermes_home(home, srv.base_url, extra_config=OFFLINE_CONFIG + NO_BACKGROUND_REVIEW)
        _run_turn(AIAgent(provider="custom", base_url=srv.base_url, api_key="sk-fake-e2e", model="fake-model",
                          session_id="relay-trace-chat", quiet_mode=True, platform="cli", skip_memory=True, skip_context_files=True))
        assert len(srv.main_requests()) == 2

    _assert_trace(collector.llm_spans(2), [
        {"completion": 10, "prompt": 100},
        {"completion": 20, "prompt": 100, "prompt_details.cache_read": 80},
    ])


def test_anthropic_messages_trace_carries_provider_usage(relay_home, monkeypatch):
    import tests.fakes.providers.anthropic_messages as fake
    from run_agent import AIAgent

    build_message = fake.build_message

    def with_cache_usage(reply, tool_ids, content=True):  # message_start carries input + cache counts
        message = build_message(reply, tool_ids, content)
        message.usage.cache_read_input_tokens, message.usage.cache_creation_input_tokens = 70, 25
        return message

    monkeypatch.setattr(fake, "build_message", with_cache_usage)
    home, collector = relay_home
    script = [fake.Reply([fake.ToolUse("terminal", {"command": "echo hi"})]), fake.Reply([fake.Text("done")])]
    with fake.AnthropicMessagesServer(script) as srv:
        (home / "config.yaml").write_text("model:\n  provider: anthropic\n  default: claude-opus-5.5\n"
                                          + OFFLINE_CONFIG + NO_BACKGROUND_REVIEW)
        _run_turn(AIAgent(provider="anthropic", base_url=srv.base_url, api_key="sk-ant-api03-fake",
                          model="claude-opus-5.5", session_id="relay-trace-anthropic", quiet_mode=True,
                          platform="cli", skip_memory=True, skip_context_files=True))
        assert len(srv.main_requests()) == 2

    # OpenInference prompt = input + cache read + cache write.
    usage = {"completion": 20, "prompt": 195, "prompt_details.cache_read": 70, "prompt_details.cache_write": 25}
    _assert_trace(collector.llm_spans(2), [usage, usage])
