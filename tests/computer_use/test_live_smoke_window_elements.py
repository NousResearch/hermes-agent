"""The opt-in live smoke's two element readers must negotiate compact Cua responses too."""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from tests.computer_use import live_cua_0_9_smoke as smoke


@pytest.mark.parametrize(("properties", "structured_format"), [
    ({"tree_format": {"type": "string", "enum": ["markdown", "elements", "both"]}}, True),
    ({}, False),
])
def test_smoke_negotiates_both_element_reads(monkeypatch, properties, structured_format):
    calls = []
    editor = {("element_index" if structured_format else "index"): 7,
              "element_token": "s1234abcd:7", "role": "AXTextArea", "value": ""}

    class Client:
        async def initialize(self):
            pass

        async def list_tools(self):
            return SimpleNamespace(tools=[SimpleNamespace(name="get_window_state", inputSchema={
                "type": "object", "properties": properties})])

        async def call_tool(self, name, args):
            calls.append((name, dict(args)))
            data = {}
            if name == "get_window_state":
                assert args.get("tree_format") == ("elements" if structured_format else None)
                data = {"elements": [dict(editor)]}
            if name == "type_text":
                assert args["element_index"] == 7 and args["element_token"] == editor["element_token"]
                editor["value"] = args["text"]
            if name == "launch_app":
                data = {"pid": 101, "windows": [{"window_id": 202}]}
            return SimpleNamespace(content=[], structuredContent=data, isError=False)

    @asynccontextmanager
    async def stdio(_params):
        yield None, None

    @asynccontextmanager
    async def session(_read, _write):
        yield Client()

    monkeypatch.setattr(smoke, "stdio_client", stdio)
    monkeypatch.setattr(smoke, "ClientSession", session)
    monkeypatch.setattr(smoke, "textedit_process_contains", lambda pid, marker: False)
    result = asyncio.run(smoke.run_smoke("unused-test-socket"))
    assert result["foreground"]["classification"] == "pass"
    assert result["foreground"]["fresh_state_confirmed"] is True
    reads = [args for name, args in calls if name == "get_window_state"]
    assert len(reads) == 2 and reads[0] == reads[1]
    assert reads[0]["pid"] == 101 and reads[0]["window_id"] == 202
