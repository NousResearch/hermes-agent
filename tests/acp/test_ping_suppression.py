"""Tests for acp_adapter.entry._BenignProbeMethodFilter.

Covers both the isolated filter logic and the full end-to-end path where a
client sends a bare JSON-RPC ``ping`` request over stdio and the acp runtime
surfaces the resulting ``RequestError`` via ``logging.exception("Background
task failed", ...)``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from io import StringIO

import pytest

from acp.exceptions import RequestError

from acp_adapter.entry import _BenignProbeMethodFilter


# -- Unit tests on the filter itself ----------------------------------------


def _make_record(msg: str, exc: BaseException | None) -> logging.LogRecord:
    record = logging.LogRecord(
        name="root",
        level=logging.ERROR,
        pathname=__file__,
        lineno=0,
        msg=msg,
        args=(),
        exc_info=(type(exc), exc, exc.__traceback__) if exc else None,
    )
    return record


def _bake_tb(exc: BaseException) -> BaseException:
    try:
        raise exc
    except BaseException as e:  # noqa: BLE001
        return e


@pytest.mark.parametrize("method", ["ping", "health", "healthcheck"])
def test_filter_suppresses_benign_probe(method: str) -> None:
    f = _BenignProbeMethodFilter()
    exc = _bake_tb(RequestError.method_not_found(method))
    record = _make_record("Background task failed", exc)
    assert f.filter(record) is False


def test_filter_allows_real_method_not_found() -> None:
    f = _BenignProbeMethodFilter()
    exc = _bake_tb(RequestError.method_not_found("session/custom"))
    record = _make_record("Background task failed", exc)
    assert f.filter(record) is True


def test_filter_allows_non_request_error() -> None:
    f = _BenignProbeMethodFilter()
    exc = _bake_tb(RuntimeError("boom"))
    record = _make_record("Background task failed", exc)
    assert f.filter(record) is True


def test_filter_allows_different_message_even_for_ping() -> None:
    """Only 'Background task failed' is muted — other messages pass through."""
    f = _BenignProbeMethodFilter()
    exc = _bake_tb(RequestError.method_not_found("ping"))
    record = _make_record("Some other context", exc)
    assert f.filter(record) is True


def test_filter_allows_request_error_with_different_code() -> None:
    f = _BenignProbeMethodFilter()
    exc = _bake_tb(RequestError.invalid_params({"method": "ping"}))
    record = _make_record("Background task failed", exc)
    assert f.filter(record) is True


def test_filter_allows_log_without_exc_info() -> None:
    f = _BenignProbeMethodFilter()
    record = _make_record("Background task failed", None)
    assert f.filter(record) is True


# -- End-to-end: drive a real JSON-RPC `ping` through acp.run_agent ---------


class _FakeAgent:
    """Minimal acp.Agent stub — we only need the router to build."""

    async def initialize(self, **kwargs):  # noqa: ANN003
        from acp.schema import AgentCapabilities, InitializeResponse

        return InitializeResponse(protocol_version=1, agent_capabilities=AgentCapabilities())

    async def new_session(self, cwd, mcp_servers=None, **kwargs):  # noqa: ANN001, ANN003
        from acp.schema import NewSessionResponse

        return NewSessionResponse(session_id="test")

    async def prompt(self, session_id, prompt, **kwargs):  # noqa: ANN001, ANN003
        from acp.schema import PromptResponse

        return PromptResponse(stop_reason="end_turn")

    async def cancel(self, session_id, **kwargs):  # noqa: ANN001, ANN003
        pass

    async def authenticate(self, **kwargs):  # noqa: ANN003
        pass

    def on_connect(self, conn):  # noqa: ANN001
        pass


@pytest.mark.asyncio
async def test_bare_ping_request_produces_proper_response_and_no_stderr_noise(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A bare ``ping`` must get a JSON-RPC -32601 back AND leave stderr clean
    when the filter is installed on the handler.
    """
    import acp

    # Attach the filter to a fresh stream handler that mirrors entry._setup_logging.
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(name)s|%(levelname)s|%(message)s"))
    handler.addFilter(_BenignProbeMethodFilter())
    root = logging.getLogger()
    prior_handlers = root.handlers[:]
    prior_level = root.level
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    # Also suppress propagation of caplog's default handler interfering with
    # our stream (caplog still captures via its own propagation hook).
    try:
        loop = asyncio.get_running_loop()

        # Pipe client -> agent
        client_to_agent_r, client_to_agent_w = os.pipe()
        # Pipe agent -> client
        agent_to_client_r, agent_to_client_w = os.pipe()

        in_read_file = os.fdopen(client_to_agent_r, "rb", buffering=0)
        in_write_file = os.fdopen(client_to_agent_w, "wb", buffering=0)
        out_read_file = os.fdopen(agent_to_client_r, "rb", buffering=0)
        out_write_file = os.fdopen(agent_to_client_w, "wb", buffering=0)

        # Agent reads its input from this StreamReader:
        agent_input = asyncio.StreamReader(limit=1024 * 1024, loop=loop)
        agent_input_proto = asyncio.StreamReaderProtocol(agent_input, loop=loop)
        await loop.connect_read_pipe(lambda: agent_input_proto, in_read_file)

        # Agent writes its output via this StreamWriter:
        out_transport, out_protocol = await loop.connect_write_pipe(
            asyncio.streams.FlowControlMixin, out_write_file
        )
        agent_output = asyncio.StreamWriter(out_transport, out_protocol, None, loop)

        # Test harness reads agent output via this StreamReader:
        client_input = asyncio.StreamReader(limit=1024 * 1024, loop=loop)
        client_input_proto = asyncio.StreamReaderProtocol(client_input, loop=loop)
        await loop.connect_read_pipe(lambda: client_input_proto, out_read_file)

        agent_task = asyncio.create_task(
            acp.run_agent(
                _FakeAgent(),
                input_stream=agent_output,
                output_stream=agent_input,
                use_unstable_protocol=True,
            )
        )

        # Send a bare `ping`
        request = {"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {}}
        in_write_file.write((json.dumps(request) + "\n").encode())
        in_write_file.flush()

        response_line = await asyncio.wait_for(client_input.readline(), timeout=5.0)
        # Give the supervisor task a tick to fire (filter should eat it)
        await asyncio.sleep(0.2)

        response = json.loads(response_line.decode())
        assert response["error"]["code"] == -32601, response
        assert response["error"]["data"] == {"method": "ping"}, response

        logs = stream.getvalue()
        assert "Background task failed" not in logs, (
            f"ping noise leaked to stderr:\n{logs}"
        )

        # Clean shutdown
        in_write_file.close()
        try:
            await asyncio.wait_for(agent_task, timeout=2.0)
        except (asyncio.TimeoutError, Exception):
            agent_task.cancel()
            try:
                await agent_task
            except BaseException:  # noqa: BLE001
                pass
    finally:
        root.handlers = prior_handlers
        root.setLevel(prior_level)
