"""Regression for #87468: telemetry delivery must not hold up live sessions."""
from __future__ import annotations

import atexit
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    from hermes_cli.plugins import PluginManager

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "plugins:\n  enabled: [observability/langfuse]\n", encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    manager = PluginManager()
    manager.discover_and_load()
    loaded = manager._plugins["observability/langfuse"]
    assert loaded.enabled and not loaded.error
    yield manager, loaded.module
    atexit.unregister(loaded.module._finalize_all_traces)
    manager.unload()


@pytest.mark.parametrize("boundary", ["success", "error", "session", "empty_session"])
def test_live_boundary_returns_while_flush_is_blocked(plugin, monkeypatch, boundary):
    _, mod = plugin
    release = threading.Event()
    client = Mock()
    client.flush.side_effect = lambda: release.wait(10)
    monkeypatch.setattr(mod, "_get_langfuse", lambda: client)
    monkeypatch.setattr(mod, "_settled_client", lambda: client)
    root, generation, tool = Mock(), Mock(), Mock()
    key = mod._trace_key("", "live", turn_id="turn")
    if boundary != "empty_session":
        mod._TRACE_STATE[key] = mod.TraceState(
            trace_id="trace", root_ctx=None, root_span=root,
            generations={"1": generation}, tools={"tool": tool},
        )

    def finish():
        if boundary == "success":
            mod.on_post_llm_call(
                session_id="live", turn_id="turn", api_call_count=1,
                assistant_response="done",
            )
        elif boundary == "error":
            mod.on_api_request_error(
                session_id="live", turn_id="turn", api_call_count=1,
                retryable=False, error={"type": "APIError", "message": "unavailable"},
            )
        else:
            mod.on_session_finalize(session_id="live", reason="session_boundary")

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(finish)
        try:
            future.result(timeout=2)
            assert key not in mod._TRACE_STATE
            if boundary != "empty_session":
                for observation in (root, generation, tool):
                    observation.end.assert_called_once()
        finally:
            release.set()
            future.result(timeout=5)


@pytest.mark.parametrize("first_status", [200, 400])
def test_sdk_exports_after_stalled_or_rejected_request_and_drains_on_shutdown(
    plugin, monkeypatch, first_status,
):
    """Real discovery, SDK and local OTLP receiver; no external service or keys."""
    sdk = pytest.importorskip("langfuse")
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
    from opentelemetry.sdk.trace import TracerProvider

    manager, mod = plugin
    entered, release = threading.Event(), threading.Event()
    accepted = []

    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = self.rfile.read(int(self.headers["Content-Length"]))
            first = not entered.is_set()
            entered.set()
            release.wait(10)
            status = first_status if first else 200
            if status == 200:
                message = ExportTraceServiceRequest.FromString(payload)
                accepted.extend(
                    span for resource in message.resource_spans
                    for scope in resource.scope_spans for span in scope.spans
                )
            self.send_response(status)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    provider = TracerProvider()
    monkeypatch.setenv("HERMES_LANGFUSE_PUBLIC_KEY", f"pk-lf-{uuid4()}")
    monkeypatch.setenv("HERMES_LANGFUSE_SECRET_KEY", "sk-lf-local-test")
    monkeypatch.setenv("HERMES_LANGFUSE_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("HERMES_LANGFUSE_CAPTURE", "metadata")
    monkeypatch.setattr(
        mod, "Langfuse",
        partial(sdk.Langfuse, tracer_provider=provider, flush_at=1, flush_interval=0.05, timeout=10),
    )

    def turn(session, *, finalize=False):
        manager.invoke_hook(
            "pre_api_request", session_id=session, turn_id=session,
            api_call_count=1, request_messages=[{"role": "user", "content": "hello"}],
        )
        if finalize:
            manager.invoke_hook("on_session_finalize", session_id=session, reason="new_session")
        else:
            manager.invoke_hook(
                "post_api_request", session_id=session, turn_id=session,
                api_call_count=1, assistant_response="done",
            )

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            first = executor.submit(turn, "first")
            try:
                assert entered.wait(5), "SDK never reached the local receiver"
                first.result(timeout=2)
                executor.submit(turn, "rotated", finalize=True).result(timeout=2)
                assert not mod._TRACE_STATE
            finally:
                release.set()
                first.result(timeout=5)

        # Allow pending batches to settle, then prove a later short-lived
        # session still delivers its root and generation at shutdown.
        provider.force_flush()
        manager.invoke_hook(
            "pre_api_request", session_id="shutdown", turn_id="shutdown",
            api_call_count=1, request_messages=[],
        )
        manager.invoke_hook("on_session_finalize", reason="shutdown")
        assert not mod._TRACE_STATE
        roots = [span for span in accepted if span.name == "Hermes turn"]
        assert roots, "ended roots must still reach Langfuse"
        for session in ("rotated", "shutdown"):
            root = next(
                span for span in roots
                if any(attr.key == "session.id" and attr.value.string_value == session
                       for attr in span.attributes)
            )
            assert any(
                span.parent_span_id == root.span_id and span.name == "LLM call 1"
                for span in accepted
            ), f"{session} lost its generation"
    finally:
        release.set()
        manager.invoke_hook("on_session_finalize", reason="shutdown")
        provider.shutdown()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
