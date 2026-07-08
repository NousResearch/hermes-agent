"""Real-socket protocol tests for the ha_conversation handle server."""

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest
import pytest_asyncio
from wyoming.asr import Transcript
from wyoming.client import AsyncTcpClient
from wyoming.handle import Handled, NotHandled
from wyoming.info import Describe, Info

# handle_server.py is a pure-protocol module with no adapter dependency;
# load it by file path under a unique module name (same mechanism as
# tests/gateway/_plugin_adapter_loader — no sys.path mutation).
_HS_PATH = (
    Path(__file__).resolve().parents[2]
    / "plugins" / "platforms" / "ha_conversation" / "handle_server.py"
)
_spec = importlib.util.spec_from_file_location("ha_conversation_handle_server", _HS_PATH)
hs = importlib.util.module_from_spec(_spec)
sys.modules["ha_conversation_handle_server"] = hs
_spec.loader.exec_module(hs)


def make_server(on_transcript, **kwargs):
    return hs.HandleServer("127.0.0.1", 0, on_transcript=on_transcript, **kwargs)


async def _echo(text, context, respond):
    await respond(f"echo: {text}")


@pytest_asyncio.fixture
async def echo_server():
    server = make_server(_echo, supports_home_control=True)
    await server.start()
    yield server
    await server.stop()


@pytest.mark.asyncio
async def test_describe_returns_handle_program(echo_server):
    async with AsyncTcpClient("127.0.0.1", echo_server.port) as client:
        await client.write_event(Describe().event())
        event = await asyncio.wait_for(client.read_event(), 5)
    assert Info.is_type(event.type)
    info = Info.from_event(event)
    assert info.handle, "must advertise a HandleProgram"
    program = info.handle[0]
    assert program.installed is True
    assert program.supports_home_control is True
    # empty language list => HA treats it as MATCH_ALL
    assert program.models and program.models[0].languages == []


@pytest.mark.asyncio
async def test_transcript_round_trip(echo_server):
    async with AsyncTcpClient("127.0.0.1", echo_server.port) as client:
        await client.write_event(
            Transcript("what time is it", context={"satellite_id": "sat.kitchen"}).event()
        )
        event = await asyncio.wait_for(client.read_event(), 5)
    assert Handled.is_type(event.type)
    assert Handled.from_event(event).text == "echo: what time is it"


@pytest.mark.asyncio
async def test_respond_none_writes_not_handled():
    async def refuse(text, context, respond):
        await respond(None)

    server = make_server(refuse)
    await server.start()
    try:
        async with AsyncTcpClient("127.0.0.1", server.port) as client:
            await client.write_event(Transcript("").event())
            event = await asyncio.wait_for(client.read_event(), 5)
        assert NotHandled.is_type(event.type)
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_respond_is_single_use_and_reports_writer():
    outcomes = {}

    async def double(text, context, respond):
        outcomes["first"] = await respond("first")
        outcomes["second"] = await respond("second")

    server = make_server(double)
    await server.start()
    try:
        async with AsyncTcpClient("127.0.0.1", server.port) as client:
            await client.write_event(Transcript("hi").event())
            event = await asyncio.wait_for(client.read_event(), 5)
            assert Handled.from_event(event).text == "first"
            # no second frame arrives: connection closes after the turn
            assert await client.read_event() is None
    finally:
        await server.stop()
    assert outcomes == {"first": True, "second": False}


@pytest.mark.asyncio
async def test_unknown_events_ignored_and_context_passed():
    seen = {}

    async def record(text, context, respond):
        seen["text"], seen["context"] = text, context
        await respond("ok")

    server = make_server(record)
    await server.start()
    try:
        async with AsyncTcpClient("127.0.0.1", server.port) as client:
            # an unsupported event type must not kill the connection
            from wyoming.event import Event
            await client.write_event(Event(type="ping"))
            await client.write_event(
                Transcript("hello", context={"device_id": "d1"}, language="en").event()
            )
            event = await asyncio.wait_for(client.read_event(), 5)
        assert Handled.is_type(event.type)
    finally:
        await server.stop()
    assert seen["text"] == "hello"
    assert seen["context"] == {"device_id": "d1"}


@pytest.mark.asyncio
async def test_start_raises_when_port_taken():
    first = make_server(_echo)
    await first.start()
    try:
        second = hs.HandleServer("127.0.0.1", first.port, on_transcript=_echo)
        with pytest.raises(OSError):
            await second.start()
    finally:
        await first.stop()
