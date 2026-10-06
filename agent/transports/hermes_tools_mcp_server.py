"""Hermes-tools-as-MCP server for the codex_app_server runtime.

Codex owns the loop and tool list there, so a curated subset of Hermes tools is
exposed over stdio MCP; codex registers it via ``~/.codex/config.toml
[mcp_servers.hermes-tools]``. Run: ``python -m agent.transports.hermes_tools_mcp_server``.
"""

from __future__ import annotations

# First, like every entry point: stdio, import-path and environ-lifetime fixes (hermes_bootstrap).
# Only as ``python -m``: codex_runtime & co. import this module for a constant, and the
# bootstrap's scratch/TMPDIR exports must not fire in those library importers.
if __name__ == "__main__":
    try:
        import hermes_bootstrap  # noqa: F401
    except ModuleNotFoundError:
        pass  # a partial ``hermes update`` can leave the bootstrap unregistered

import base64
import inspect
import json
import logging
import mimetypes
import os
import sys
from pathlib import Path
from typing import Any, Optional

from agent.tool_dispatch_helpers import _is_multimodal_tool_result, _multimodal_text_summary

logger = logging.getLogger(__name__)

# The ``[mcp_servers.<name>]`` key under which the runtime migration registers this server. Every
# codex-side reference to it (worker ``-c mcp_servers.<name>.env.*`` overrides, elicitation
# auto-accept, display-name stripping) must use this constant: a drifted name materialises a
# second env-only entry that codex rejects at bootstrap ("invalid transport").
HERMES_TOOLS_MCP_SERVER_NAME = "hermes-tools"

# JSON Schema type -> Python type mapping for signature generation
_JSON_TO_PY = {"string": str, "integer": int, "number": float, "boolean": bool, "array": list, "object": dict}


def _signature_from_schema(schema: dict | None) -> tuple[inspect.Signature, dict[str, type]]:
    """KEYWORD_ONLY signature + annotations from a JSON schema (optional params default to None)."""
    props = (schema or {}).get("properties") or {}
    required = set((schema or {}).get("required") or [])
    params, annots = [], {}
    for pname, pspec in props.items():
        if pname.startswith("_"):
            continue
        py = _JSON_TO_PY.get((pspec or {}).get("type"), Any)
        ann, default = (py, inspect.Parameter.empty) if pname in required else (Optional[py], None)
        annots[pname] = ann
        params.append(inspect.Parameter(pname, inspect.Parameter.KEYWORD_ONLY, annotation=ann, default=default))
    return inspect.Signature(params, return_annotation=Any), annots


# Image types an MCP image block carries here (the SDK's own suffix table); anything else stays text.
_MCP_IMAGE_FORMATS = {"image/png": "png", "image/jpeg": "jpeg", "image/gif": "gif", "image/webp": "webp"}


def _decode_image_data_url(url: str) -> tuple[bytes, str] | None:
    """``(bytes, format)`` of a well-formed base64 image data URL, else None."""
    header, sep, payload = url.partition(",")
    if not (sep and payload and header.startswith("data:") and header.endswith(";base64")):
        return None
    fmt = _MCP_IMAGE_FORMATS.get(header[len("data:"):].split(";")[0].lower())
    try:
        data = base64.b64decode(payload, validate=True)
    except ValueError:  # binascii.Error, and non-ASCII input
        return None
    return (data, fmt) if fmt and data else None


def _read_image_file(path: str) -> tuple[bytes, str] | None:
    """``(bytes, format)`` of a readable image file, else None. Read here: the SDK's ``Image(path=)``
    reads lazily during serialization, where a vanished file fails the whole call."""
    fmt = _MCP_IMAGE_FORMATS.get(mimetypes.guess_type(path)[0] or "")
    try:
        data = Path(path).read_bytes() if fmt else b""
    except OSError:
        return None
    return (data, fmt) if data else None


def _project_tool_result(tool_name: str, result: Any) -> Any:
    """Project a Hermes tool result onto something MCP can actually deliver.

    Hermes tools return either a string or a ``_multimodal`` envelope used to attach a
    screenshot to vision-capable models::

        {"_multimodal": True, "text_summary": ..., "meta": {"screenshot_path": ...},
         "content": [{"type": "text", ...}, {"type": "image_url", ...}]}

    The MCP bridge only ever declared ``-> str`` and returned the result unchanged, so a
    screenshot-producing call (``browser_exec`` with ``capture_screenshot()``, ``vision_analyze``,
    ``computer_use``) died in pydantic validation — "Input should be a valid string" — instead of
    returning the image, and every MCP client lost the vision half of the tool surface.

    Text stays text; a multimodal envelope becomes its text blocks plus real MCP image blocks, as
    Hermes' own loop would send it to a vision model. Anything else is JSON-serialized so an
    unexpected shape degrades into text rather than a protocol error.
    """
    if isinstance(result, str):
        return result
    if not _is_multimodal_tool_result(result):
        return json.dumps(result, ensure_ascii=False, default=str)
    parts = [p for p in result["content"] if isinstance(p, dict)]
    urls = [p["image_url"].get("url") for p in parts
            if p.get("type") == "image_url" and isinstance(p.get("image_url"), dict)]
    urls = [u for u in urls if isinstance(u, str)]
    # The inline images are the producer's prepared copies (resized to the embed budget); the
    # meta path is the full-size original kept for sharing, so it is only the fallback.
    images = [image for image in map(_decode_image_data_url, urls) if image]
    meta = result.get("meta") or {}
    path = meta.get("screenshot_path") or meta.get("image_path") or ""
    if not images and path and (image := _read_image_file(path)):
        images.append(image)
    # A remote URL is named, not fetched: the bridge makes no network calls of its own.
    notes = [f"[image: {u}]" for u in urls if not u.startswith("data:")]
    if path:
        notes.append(f"[screenshot: {path}]")
    # The summary is what Hermes sends a model that cannot see the image.
    text_only = "\n\n".join([_multimodal_text_summary(result), *notes])
    if not images:
        logger.warning("%s returned a multimodal result with no deliverable image; sending text only",
                       tool_name)
        return text_only
    try:
        from mcp.server.mcpserver.utilities.types import Image  # mcp >= 2.0
    except ImportError:
        try:
            from mcp.server.fastmcp.utilities.types import Image  # mcp 1.x
        except ImportError:
            logger.warning("%s: SDK has no Image helper; sending text only", tool_name)
            return text_only
    # The text blocks carry what the summary drops: the question and any crop/scale coordinate mapping.
    text = "\n\n".join(str(p["text"]) for p in parts if p.get("type") == "text" and p.get("text"))
    return ["\n\n".join([text or _multimodal_text_summary(result), *notes]),
            *(Image(data=data, format=fmt) for data, fmt in images)]


# Each name MUST match a registered Hermes tool ``model_tools.handle_function_call()`` can dispatch.
# NOT exposed: terminal/file/search/process/clarify (codex built-ins + its own approval UI);
# delegate_task/memory/session_search/todo (need the running AIAgent context).
EXPOSED_TOOLS: tuple[str, ...] = (
    "web_search", "web_extract",
    "browser_navigate", "browser_click", "browser_type", "browser_press", "browser_snapshot", "browser_scroll",
    "browser_back", "browser_get_images", "browser_console", "browser_vision",
    "vision_analyze", "image_generate", "skill_view", "skills_list", "text_to_speech",
    # Kanban handoff tools: stateless (read HERMES_KANBAN_TASK, write kanban.db).
    # Without them a codex-runtime worker can't report completion and hangs.
    "kanban_complete", "kanban_block", "kanban_schedule", "kanban_request_review",
    "kanban_request_changes", "kanban_comment",
    "kanban_heartbeat", "kanban_show", "kanban_list",
    # Orchestrator-only (the kanban tool gates them on HERMES_KANBAN_TASK unset).
    "kanban_create", "kanban_unblock", "kanban_link",
)


def _build_server() -> Any:
    """Create the MCP server with Hermes tools attached (lazy imports: importable without ``mcp``)."""
    try:
        # mcp 2.0 renamed `mcp.server.fastmcp` to `mcp.server.MCPServer` (same surface).
        from mcp.server import MCPServer
    except ImportError as exc:  # pragma: no cover - install hint
        raise ImportError(f"hermes-tools MCP server requires the 'mcp' package: {exc}") from exc

    from model_tools import get_tool_definitions, handle_function_call

    mcp = MCPServer(
        HERMES_TOOLS_MCP_SERVER_NAME,
        instructions=(
            "Hermes Agent's tool surface, exposed for use inside a Codex "
            "session. Use these for capabilities Codex's built-in toolset "
            "doesn't cover: web search/extract, browser automation, "
            "subagent delegation, vision, image generation, persistent "
            "memory, skills, and cross-session search."
        ),
    )

    # Authoritative Hermes schemas so MCP clients see the same parameter docs the model does.
    all_defs = {
        td["function"]["name"]: td["function"]
        for td in (get_tool_definitions(quiet_mode=True) or [])
        if isinstance(td, dict) and td.get("type") == "function"
    }

    def _make_handler(tool_name: str, schema: dict | None, description: str):
        # The SDK derives the input schema from the callable's signature, so synthesize it from the JSON Schema.
        sig, annots = _signature_from_schema(schema)

        def _dispatch(**kwargs: Any) -> Any:
            try:
                # Drop None so unset optionals aren't forwarded to the handler.
                result = handle_function_call(tool_name, {k: v for k, v in kwargs.items() if v is not None})
                return _project_tool_result(tool_name, result)
            except Exception as exc:
                logger.exception("tool %s raised", tool_name)
                return json.dumps({"error": str(exc), "tool": tool_name})

        _dispatch.__name__ = tool_name
        _dispatch.__doc__ = description
        _dispatch.__signature__ = sig
        _dispatch.__annotations__ = {**annots, "return": Any}
        return _dispatch

    exposed_count = 0
    for name in EXPOSED_TOOLS:
        spec = all_defs.get(name)
        if spec is None:
            logger.debug("skipping %s — not registered in this Hermes process", name)
            continue
        description = spec.get("description") or f"Hermes {name} tool"
        params_schema = spec.get("parameters") or {"type": "object", "properties": {}}
        try:
            mcp.add_tool(_make_handler(name, params_schema, description), name=name, description=description)
        except TypeError:
            # Older mcp SDK: decorator-style registration; __signature__ still drives schema.
            mcp.tool(name=name, description=description)(_make_handler(name, params_schema, description))
        exposed_count += 1

    logger.info("hermes-tools MCP server registered %d/%d tools", exposed_count, len(EXPOSED_TOOLS))
    return mcp


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for `python -m agent.transports.hermes_tools_mcp_server`."""
    argv = argv or sys.argv[1:]
    verbose = "--verbose" in argv or "-v" in argv
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        stream=sys.stderr,  # MCP uses stdio for protocol — logs MUST go to stderr
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Keep Hermes' own banners off stdout (the MCP wire).
    os.environ.setdefault("HERMES_QUIET", "1")
    os.environ.setdefault("HERMES_REDACT_SECRETS", "true")

    try:
        server = _build_server()
    except ImportError as exc:
        sys.stderr.write(f"hermes-tools MCP server cannot start: {exc}\n")
        return 2
    try:
        server.run()  # defaults to stdio transport, which codex spawns us on
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logger.exception("hermes-tools MCP server crashed")
        sys.stderr.write(f"hermes-tools MCP server error: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
