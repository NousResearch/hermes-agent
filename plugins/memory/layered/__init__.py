"""Layered memory provider for Hermes.

This provider adds memory taxonomy/routing, optional MemPalace semantic recall,
and Caveman-style deterministic compression.  It does not auto-enable noisy
background jobs or cron tasks.  All writes remain explicit and routed to the
proper Hermes shelf.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from agent.memory_layers import CavemanCompressor, LayeredMemoryRouter, MemPalaceAdapter, decision_to_json
from agent.memory_provider import MemoryProvider
from hermes_cli.config import cfg_get
from tools.registry import tool_error


MEMORY_ROUTE_SCHEMA = {
    "name": "memory_route",
    "description": (
        "Classify a memory candidate and recommend the correct Hermes shelf: "
        "curated memory/user profile, skill, Obsidian/Drive/repo domain store, "
        "session_search, optional semantic recall, or skip. Use before saving "
        "ambiguous durable information so memory does not become a flat vector junk drawer."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The candidate memory/artifact/procedure text."},
            "domain": {"type": "string", "description": "Optional domain hint, e.g. obsidian, google_drive, repository."},
            "compress": {"type": "boolean", "description": "Whether to apply Caveman-style padding removal to the returned content."},
        },
        "required": ["content"],
    },
}

MEMORY_COMPRESS_SCHEMA = {
    "name": "memory_compress",
    "description": (
        "Compress memory/skill/context text by removing assistant padding and whitespace "
        "without summarizing or changing facts. Use for memory hygiene, not main response voice."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to compress."},
            "max_chars": {"type": "integer", "description": "Optional character cap; truncates at a sentence boundary."},
        },
        "required": ["text"],
    },
}


def _load_plugin_config() -> dict:
    try:
        from hermes_constants import get_hermes_home
        import yaml

        config_path = get_hermes_home() / "config.yaml"
        if not config_path.exists():
            return {}
        with open(config_path, encoding="utf-8-sig") as f:
            all_config = yaml.safe_load(f) or {}
        return cfg_get(all_config, "plugins", "layered-memory", default={}) or {}
    except Exception:
        return {}


class LayeredMemoryProvider(MemoryProvider):
    """Memory provider implementing routing + optional semantic recall."""

    def __init__(self, config: dict | None = None):
        self._config = config if config is not None else _load_plugin_config()
        self._router = LayeredMemoryRouter()
        self._compressor = CavemanCompressor()
        self._mempalace: MemPalaceAdapter | None = None
        self._session_id = ""
        self._scope = ""

    @property
    def name(self) -> str:
        return "layered"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id or ""
        profile = kwargs.get("agent_identity") or "default"
        platform = kwargs.get("platform") or "cli"
        chat = kwargs.get("chat_name") or kwargs.get("chat_id") or ""
        thread = kwargs.get("thread_id") or ""
        scope_parts = [f"profile:{profile}", f"platform:{platform}"]
        if chat:
            scope_parts.append(f"chat:{chat}")
        if thread:
            scope_parts.append(f"thread:{thread}")
        self._scope = "/".join(str(p).replace(" ", "_") for p in scope_parts)

        mp_enabled = _as_bool(self._config.get("mempalace_enabled", False))
        self._mempalace = MemPalaceAdapter(
            binary=str(self._config.get("mempalace_binary", "mempalace")),
            scope=str(self._config.get("mempalace_scope", self._scope)),
            enabled=mp_enabled,
            timeout=float(self._config.get("mempalace_timeout", 8.0)),
            cwd=kwargs.get("hermes_home"),
        )

    def system_prompt_block(self) -> str:
        mp = "enabled" if self._mempalace and self._mempalace.enabled else "disabled"
        return (
            "# Layered Memory\n"
            "Active. Route knowledge by shelf: preferences/facts → curated memory; "
            "procedures → skills; artifacts → Obsidian/Drive/repos; episodes → session_search; "
            f"semantic recall → MemPalace ({mp}). Use memory_route before saving ambiguous memory."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._mempalace:
            return ""
        limit = int(self._config.get("mempalace_limit", 5))
        return self._mempalace.prefetch(query, limit=limit)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [MEMORY_ROUTE_SCHEMA, MEMORY_COMPRESS_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "memory_route":
            return self._handle_memory_route(args)
        if tool_name == "memory_compress":
            return self._handle_memory_compress(args)
        return tool_error(f"Unknown layered memory tool: {tool_name}")

    def _handle_memory_route(self, args: Dict[str, Any]) -> str:
        content = str(args.get("content") or "").strip()
        if not content:
            return tool_error("content is required", success=False)
        metadata = {}
        if args.get("domain"):
            metadata["domain"] = args.get("domain")
        decision = self._router.route(content, metadata=metadata)
        do_compress = args.get("compress", True)
        compressed = self._compressor.compress(content) if do_compress else content
        return decision_to_json(decision, compressed_content=compressed)

    def _handle_memory_compress(self, args: Dict[str, Any]) -> str:
        text = str(args.get("text") or "")
        max_chars = args.get("max_chars")
        try:
            max_chars = int(max_chars) if max_chars is not None else None
        except (TypeError, ValueError):
            max_chars = None
        compressed = self._compressor.compress(text, max_chars=max_chars)
        return json.dumps(
            {
                "success": True,
                "compressed": compressed,
                "original_chars": len(text),
                "compressed_chars": len(compressed),
            },
            ensure_ascii=False,
        )

    def get_config_schema(self):
        return [
            {"key": "mempalace_enabled", "description": "Enable MemPalace CLI semantic recall", "default": "false", "choices": ["true", "false"]},
            {"key": "mempalace_binary", "description": "MemPalace CLI binary", "default": "mempalace"},
            {"key": "mempalace_scope", "description": "Optional MemPalace scope override", "default": ""},
            {"key": "mempalace_limit", "description": "Max MemPalace recall results", "default": "5"},
        ]

    def save_config(self, values, hermes_home):
        try:
            from pathlib import Path
            import yaml

            config_path = Path(hermes_home) / "config.yaml"
            existing = {}
            if config_path.exists():
                with open(config_path, encoding="utf-8-sig") as f:
                    existing = yaml.safe_load(f) or {}
            existing.setdefault("plugins", {})
            existing["plugins"]["layered-memory"] = dict(values or {})
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(existing, f, default_flow_style=False, sort_keys=False)
        except Exception:
            pass


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def register(ctx):
    ctx.register_memory_provider(LayeredMemoryProvider())
