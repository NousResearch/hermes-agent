"""Tests for the hermes-tools-as-MCP server module surface.

We don't run a live MCP session in unit tests — that requires the codex
subprocess + client + an event loop. These tests pin the static
contract: the module imports, the EXPOSED_TOOLS list is sane, and the
build helper assembles a server when the SDK is present.
"""

from __future__ import annotations

import base64
import inspect
import json
import sys

import pytest

from agent.transports.hermes_tools_mcp_server import (
    _project_tool_result,
    _signature_from_schema,
)


class TestSignatureFromSchema:
    """Test the JSON Schema -> Python signature conversion."""

    def test_simple_required_string_param(self):
        """A required string param becomes str with no default."""
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        sig, annots = _signature_from_schema(schema)

        assert len(sig.parameters) == 1
        param = sig.parameters["query"]
        assert param.name == "query"
        assert param.kind == inspect.Parameter.KEYWORD_ONLY
        assert annots["query"] == str
        assert param.default is inspect.Parameter.empty


    def test_skip_private_params(self):
        """Params starting with '_' are excluded from the signature."""
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "_internal": {"type": "string"},
            },
            "required": ["query", "_internal"],
        }
        sig, annots = _signature_from_schema(schema)

        assert "_internal" not in sig.parameters
        assert "_internal" not in annots
        assert "query" in sig.parameters

    def test_all_json_types(self):
        """All JSON schema types map to correct Python types."""
        schema = {
            "type": "object",
            "properties": {
                "s": {"type": "string"},
                "i": {"type": "integer"},
                "n": {"type": "number"},
                "b": {"type": "boolean"},
                "a": {"type": "array"},
                "o": {"type": "object"},
            },
            "required": ["s", "i", "n", "b", "a", "o"],
        }
        sig, annots = _signature_from_schema(schema)

        assert annots["s"] == str
        assert annots["i"] == int
        assert annots["n"] == float
        assert annots["b"] == bool
        assert annots["a"] == list
        assert annots["o"] == dict


class TestModuleSurface:

    def test_exposed_tools_are_safe_subset(self):
        """We MUST NOT expose tools codex already has, because codex'
        own builtins are better-integrated with its sandbox + approvals.
        Specifically: no terminal/shell, no read_file/write_file, no
        patch — those are codex's built-in tools."""
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
        forbidden = {
            "terminal", "shell", "read_file", "write_file", "patch",
            "search_files", "process",
        }
        leaked = forbidden & set(EXPOSED_TOOLS)
        assert not leaked, (
            f"these tools must NOT be exposed via the codex callback "
            f"because codex has built-in equivalents: {leaked}"
        )


class TestMain:
    def test_main_returns_2_when_mcp_unavailable(self, monkeypatch):
        """When the mcp package isn't installed, main() should exit
        cleanly with code 2 and an install hint, not crash."""
        import agent.transports.hermes_tools_mcp_server as m

        def boom_build(*a, **kw):
            raise ImportError("mcp not installed")

        monkeypatch.setattr(m, "_build_server", boom_build)
        rc = m.main(["--verbose"])
        assert rc == 2

    def test_main_handles_keyboard_interrupt(self, monkeypatch):
        import agent.transports.hermes_tools_mcp_server as m

        class FakeServer:
            def run(self):
                raise KeyboardInterrupt()

        monkeypatch.setattr(m, "_build_server", lambda: FakeServer())
        rc = m.main([])
        assert rc == 0

    def test_main_returns_1_on_runtime_error(self, monkeypatch):
        import agent.transports.hermes_tools_mcp_server as m

        class CrashingServer:
            def run(self):
                raise RuntimeError("boom")

        monkeypatch.setattr(m, "_build_server", lambda: CrashingServer())
        rc = m.main([])
        assert rc == 1


def test_multimodal_result_becomes_text_plus_image(tmp_path):
    """Screenshot-producing tools return a ``_multimodal`` envelope; MCP needs text plus an
    image block instead of the dict, which failed string validation and dropped the call."""
    shot = tmp_path / "shot.png"
    shot.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    result = _project_tool_result("browser_vision", {
        "_multimodal": True, "text_summary": "page", "meta": {"screenshot_path": str(shot)},
        "content": [{"type": "text", "text": "page"}]})
    assert isinstance(result, list) and result[0].startswith("page")
    assert type(result[1]).__name__ == "Image"
    assert _project_tool_result("t", "plain") == "plain"
    assert _project_tool_result("t", {"a": 1}) == '{"a": 1}'


def _call_through_sdk(monkeypatch, tool_name, tool_result):
    """Content blocks the installed SDK serializes when the real registered handler returns ``tool_result``."""
    import asyncio

    import model_tools
    from agent.transports.hermes_tools_mcp_server import _build_server

    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **_: [{"type": "function", "function": {
        "name": tool_name, "description": tool_name, "parameters": {"type": "object", "properties": {}}}}])
    monkeypatch.setattr(model_tools, "handle_function_call", lambda *_args, **_kwargs: tool_result)
    result = asyncio.run(_build_server().call_tool(tool_name, {}))
    assert not result.is_error
    return result.content


def test_vision_analyze_keeps_question_and_coordinate_mapping(monkeypatch):
    """vision_analyze carries its image only as a data URL in ``content`` and its question plus the
    crop/scale coordinate mapping only in the text part, so the client needs both, not the summary."""
    from tools.vision_tools import _build_native_vision_tool_result, _build_scale_note

    raw = b"\xff\xd8\xff\xe0" + bytes(range(256)) * 4
    note = _build_scale_note({"orig_width": 2000, "orig_height": 2000, "new_width": 500, "new_height": 500},
                             {"x": 100, "y": 200})
    envelope = _build_native_vision_tool_result(
        image_url="https://example.com/cat.jpg", question="where is the cat?",
        image_data_url="data:image/jpeg;base64," + base64.b64encode(raw).decode(),
        image_size_bytes=len(raw), scale_note=note)
    text, image = _call_through_sdk(monkeypatch, "vision_analyze", envelope)
    assert "where is the cat?" in text.text and note in text.text
    assert base64.b64decode(image.data) == raw and image.mime_type == "image/jpeg"


def test_prepared_inline_screenshot_wins_over_the_original_file(tmp_path, monkeypatch):
    """The browser producer embeds a copy resized to the payload budget and keeps the full-size
    original's path for sharing; MCP must deliver the prepared copy and must not need the file."""
    pil_image = pytest.importorskip("PIL.Image")
    from tools import vision_tools
    from tools.browser_use_cli import _native_screenshot_result

    shot = tmp_path / "shot.png"
    pil_image.new("RGB", (vision_tools._EMBED_MAX_DIMENSION * 2,) * 2, "navy").save(shot)
    monkeypatch.setattr(vision_tools, "_should_use_native_vision_fast_path", lambda: True)
    envelope = _native_screenshot_result({"success": True}, str(shot))
    prepared = next(p["image_url"]["url"] for p in envelope["content"] if p["type"] == "image_url")
    shot.unlink()
    _text, image = _call_through_sdk(monkeypatch, "browser_vision", envelope)
    assert base64.b64decode(image.data) == base64.b64decode(prepared.partition(",")[2])
    assert image.mime_type == prepared[len("data:"):].split(";")[0]


def test_only_the_canonical_envelope_takes_the_image_branch():
    """``_multimodal`` must be ``True`` with a ``content`` list (tool_dispatch_helpers' shape);
    a stray flag is ordinary data and is JSON-serialized, not rendered as an image-less summary."""
    for stray in ({"_multimodal": True, "text_summary": "x"}, {"_multimodal": "yes", "content": []}):
        assert json.loads(_project_tool_result("t", stray)) == stray


@pytest.mark.parametrize("url", [
    "https://example.com/a.png",
    "data:image/png;base64",
    "data:image/png;base64,",
    "data:image/png;base64,!!!!",
    "data:image/png;base64,é",
    "data:text/plain;base64,aGVsbG8=",
])
def test_unusable_images_degrade_to_text(url, tmp_path):
    """A remote URL is named rather than fetched; a malformed or non-image data URL, and a
    screenshot file that is gone, yield text instead of an empty or mistyped image block."""
    result = _project_tool_result("t", {
        "_multimodal": True, "text_summary": "seen", "meta": {"screenshot_path": str(tmp_path / "gone.png")},
        "content": [{"type": "image_url", "image_url": {"url": url}}]})
    assert isinstance(result, str) and result.startswith("seen")
    assert url.startswith("data:") or url in result


def test_sdk_without_an_image_helper_degrades_to_text(monkeypatch):
    """An SDK without an ``Image`` helper still returns the text instead of failing the call."""
    inline = {"_multimodal": True, "text_summary": "seen", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(b"\x89PNG").decode()}}]}
    assert isinstance(_project_tool_result("t", inline), list)
    for sdk_module in ("mcp.server.mcpserver.utilities.types", "mcp.server.fastmcp.utilities.types"):
        monkeypatch.setitem(sys.modules, sdk_module, None)  # None in sys.modules -> ImportError
    assert _project_tool_result("t", inline) == "seen"
