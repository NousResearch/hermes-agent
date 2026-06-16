import os
import logging

from agent.memory_provider import MemoryProvider

from .loader import (
    load_l0,
    load_l1,
    load_for_query,
    read_file,
    file_exists,
    needs_l1,
    needs_l2,
    get_status,
    _cache,
)

logger = logging.getLogger(__name__)


class LayeredMemoryProvider(MemoryProvider):
    name = "layered"

    def __init__(self):
        self._memory_dir = ""
        self._hermes_home = ""
        self._session_id = ""
        self._l0_block = ""
        self._l1_block = ""
        self._l2_block = ""

    def is_available(self) -> bool:
        memory_dir = os.path.expanduser("~/.hermes/memory")
        return file_exists(os.path.join(memory_dir, "L0_core.md"))

    def get_config_schema(self):
        return [
            {
                "key": "memory_dir",
                "description": "Path to layered memory files. Defaults to $HERMES_HOME/memory",
                "required": False,
                "default": "",
            },
            {
                "key": "cache_ttl",
                "description": "Seconds to cache prefetch results (default 300)",
                "required": False,
                "default": 300,
            },
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = kwargs.get("hermes_home", os.path.expanduser("~/.hermes"))
        self._hermes_home = hermes_home
        self._memory_dir = os.path.join(hermes_home, "memory")
        self._session_id = session_id

        self._l0_block = load_l0(self._memory_dir)
        self._l1_block = load_l1(self._memory_dir)

        if not self._l0_block:
            logger.warning("LayeredMemoryProvider: L0_core.md not found in %s", self._memory_dir)

        logger.info(
            "LayeredMemoryProvider initialized: session=%s L0=%dB L1=%dB",
            session_id[:8] if session_id else "?",
            len(self._l0_block),
            len(self._l1_block),
        )

    def system_prompt_block(self) -> str:
        if not self._l0_block:
            return ""
        return (
            "# Layered Memory (L0)\n"
            "The following is your persistent memory — core information about the user, "
            "their environment, and workflows. Use it to inform your responses.\n\n"
            f"{self._l0_block}\n\n"
            "To load additional memory:\n"
            "- L1 (context: API keys, versions, skills) — loaded automatically when relevant\n"
            "- L2 (archive: decisions, environment, workflows) — loaded on demand"
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        return load_for_query(self._memory_dir, query, use_cache=True)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        pass

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages=None,
    ) -> None:
        pass

    def get_tool_schemas(self):
        return [
            {
                "name": "memory_read",
                "description": "Read memory from specific layered memory tiers. "
                               "L0 is always available; L1/L2 are loaded on demand.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to look up — keywords determine which tiers to load",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "memory_status",
                "description": "Show memory file status — which files exist and their sizes.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        import json

        if tool_name == "memory_read":
            query = args.get("query", "")
            result = load_for_query(self._memory_dir, query, use_cache=True)
            return json.dumps({"memory": result})

        if tool_name == "memory_status":
            status = get_status(self._memory_dir)
            return json.dumps(status, default=str)

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def shutdown(self) -> None:
        _cache.clear()
        logger.info("LayeredMemoryProvider shut down: cache cleared")
