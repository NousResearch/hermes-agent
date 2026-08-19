"""Tests for acp_adapter.entry startup wiring."""

import asyncio
import json
import sys

import acp
import pytest

from acp_adapter import entry


def test_main_enables_unstable_protocol(monkeypatch):
    calls = {}

    async def fake_run_agent(agent):
        calls["agent"] = agent

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)
    monkeypatch.setattr(entry, "_run_agent_with_initialize_compat", fake_run_agent)

    entry.main([])

    assert calls["agent"] is not None


def test_initialize_compat_runner_wires_stdio_and_unstable_protocol(monkeypatch):
    from acp.core import DEFAULT_STDIO_BUFFER_LIMIT_BYTES

    calls = {}
    agent = object()
    reader = asyncio.StreamReader()
    writer = object()

    async def fake_stdio_streams(*, limit):
        calls["limit"] = limit
        return reader, writer

    async def fake_run_agent(actual_agent, **kwargs):
        calls["agent"] = actual_agent
        calls["kwargs"] = kwargs

    monkeypatch.setattr("acp.stdio.stdio_streams", fake_stdio_streams)
    monkeypatch.setattr(acp, "run_agent", fake_run_agent)

    asyncio.run(entry._run_agent_with_initialize_compat(agent))

    assert calls["limit"] == DEFAULT_STDIO_BUFFER_LIMIT_BYTES
    assert calls["agent"] is agent
    assert calls["kwargs"]["input_stream"] is writer
    compat_reader = calls["kwargs"]["output_stream"]
    assert isinstance(compat_reader, entry._InitializeCompatReader)
    assert compat_reader._source is reader
    assert calls["kwargs"]["use_unstable_protocol"] is True


def test_initialize_compat_reader_normalizes_date_protocol_version():
    source = asyncio.StreamReader()
    source.feed_data(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "clientInfo": {"name": "\ud800"},
            "params": {"protocolVersion": "2025-11-25"},
        }).encode()
        + b"\n"
    )
    source.feed_eof()

    line = asyncio.run(entry._InitializeCompatReader(source, 1).readline())

    assert json.loads(line)["params"]["protocolVersion"] == 1


def test_initialize_compat_reader_only_inspects_first_frame(monkeypatch):
    initialize = (
        b'{"jsonrpc":"2.0","id":1,"method":"initialize",'
        b'"params":{"protocolVersion":1}}\n'
    )
    subsequent = b'{"jsonrpc":"2.0","method":"session/update","params":{"large":"payload"}}\n'
    source = asyncio.StreamReader()
    source.feed_data(initialize + subsequent)
    source.feed_eof()
    reader = entry._InitializeCompatReader(source, 1)

    assert asyncio.run(reader.readline()) == initialize
    monkeypatch.setattr(entry.json, "loads", lambda _line: pytest.fail("reparsed frame"))
    assert asyncio.run(reader.readline()) == subsequent


def test_initialize_compat_reader_preserves_valid_integer_version():
    line = (
        b'{"jsonrpc":"2.0","id":1,"method":"initialize",'
        b'"params":{"protocolVersion":1}}\n'
    )

    assert entry._normalize_initialize_frame(line, 7) == line


def test_initialize_compat_reader_preserves_non_initialize_frames():
    line = (
        b'{"jsonrpc":"2.0","id":2,"method":"session/new",'
        b'"params":{"protocolVersion":"2025-11-25"}}\n'
    )

    assert entry._normalize_initialize_frame(line, 1) == line


@pytest.mark.parametrize(
    "protocol_version",
    [None, True, {}, [], "not-a-date", "2025-99-99", "2025-02-30", 65536],
)
def test_initialize_compat_reader_preserves_other_invalid_versions(protocol_version):
    line = (
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": protocol_version},
            }
        ).encode()
        + b"\n"
    )

    assert entry._normalize_initialize_frame(line, 1) == line


def test_initialize_compat_reader_preserves_missing_protocol_version():
    line = (
        b'{"jsonrpc":"2.0","id":1,"method":"initialize",'
        b'"params":{"clientCapabilities":{}}}\n'
    )

    assert entry._normalize_initialize_frame(line, 1) == line


def test_main_skips_configured_mcp_discovery_when_requested(monkeypatch):
    discovery_calls = []

    async def fake_run_agent(agent):
        pass

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)
    monkeypatch.setenv("HERMES_ACP_SKIP_CONFIGURED_MCP", "1")
    monkeypatch.setattr(
        "tools.mcp_tool.discover_mcp_tools",
        lambda: discovery_calls.append(True),
    )
    monkeypatch.setattr(entry, "_run_agent_with_initialize_compat", fake_run_agent)

    entry.main([])

    assert discovery_calls == []










def test_main_setup_offers_browser_install_when_tty(monkeypatch):
    """When stdin is a TTY and the user answers yes, model setup is followed
    by a browser-tools bootstrap call."""
    monkeypatch.setattr("hermes_cli.main.main", lambda: None)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_args, **_kwargs: "y")

    bootstrap_calls = []
    monkeypatch.setattr(
        entry,
        "_run_setup_browser",
        lambda assume_yes=False: bootstrap_calls.append(assume_yes) or 0,
    )

    entry.main(["--setup"])

    assert bootstrap_calls == [False]










def test_main_setup_browser_propagates_browser_failure(monkeypatch):
    """If browser install fails, exit code is 1."""
    def fake_ensure(dep, interactive=True):
        return dep != "browser"  # browser fails

    monkeypatch.setattr("hermes_cli.dep_ensure.ensure_dependency", fake_ensure)

    with pytest.raises(SystemExit) as excinfo:
        entry.main(["--setup-browser"])
    assert excinfo.value.code == 1
