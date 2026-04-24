"""Membase memory provider shim.

The provider implementation lives in the ``hermes-membase`` PyPI package.
This bundled shim keeps Hermes discovery light while allowing
``hermes memory setup`` to install the package on demand.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider

try:
    from membase_hermes.provider import MembaseMemoryProvider as _MembaseMemoryProvider
except ModuleNotFoundError as exc:
    if exc.name != "membase_hermes":
        raise
    _MembaseMemoryProvider = None


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class _MembaseSetupProvider(MemoryProvider):
    """Minimal setup-time provider used before hermes-membase is installed."""

    @property
    def name(self) -> str:
        return "membase"

    def is_available(self) -> bool:
        # Keep the provider visible in `hermes memory setup`; credentials are
        # handled by OAuth after setup via `hermes membase login`.
        return True

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "apiUrl",
                "description": "Membase API URL (press enter to accept default)",
                "required": False,
                "default": "https://api.membase.so",
                "secret": False,
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        home = Path(hermes_home).expanduser()
        config_path = home / "membase.json"
        token_path = home / "credentials" / "membase.json"
        mirror_index_path = home / "plugins" / "membase" / "mirror_index.json"

        config_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        mirror_index_path.parent.mkdir(parents=True, exist_ok=True)

        existing: Dict[str, Any] = {}
        if config_path.exists():
            try:
                raw = json.loads(config_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    existing = raw
            except Exception:
                existing = {}

        api_url = str(values.get("apiUrl", existing.get("apiUrl", "https://api.membase.so"))).strip()
        existing.update(
            {
                "apiUrl": api_url or "https://api.membase.so",
                "clientId": existing.get("clientId", ""),
                "tokenFile": str(token_path),
                "autoRecall": _as_bool(existing.get("autoRecall"), False),
                "autoWikiRecall": _as_bool(existing.get("autoWikiRecall"), False),
                "autoCapture": _as_bool(existing.get("autoCapture"), True),
                "mirrorBuiltin": _as_bool(existing.get("mirrorBuiltin"), True),
                "maxRecallChars": _as_int(existing.get("maxRecallChars"), 4000),
                "debug": _as_bool(existing.get("debug"), False),
            }
        )
        config_path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        if not token_path.exists():
            token_path.write_text('{"accessToken": "", "refreshToken": ""}\n', encoding="utf-8")
        if not mirror_index_path.exists():
            mirror_index_path.write_text("{}\n", encoding="utf-8")

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        return None

    def system_prompt_block(self) -> str:
        return (
            "<membase-notice>\n"
            "Membase requires the `hermes-membase` package. Run `hermes memory setup` "
            "or `uv pip install --python $(which python) \"hermes-membase>=0.1.5\"`, "
            "then run `hermes membase login`.\n"
            "</membase-notice>"
        )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        return "Membase requires hermes-membase>=0.1.5. Run `hermes memory setup` and `hermes membase login`."


MembaseMemoryProvider = _MembaseMemoryProvider or _MembaseSetupProvider


def register(ctx: Any) -> None:
    if hasattr(ctx, "register_memory_provider"):
        ctx.register_memory_provider(MembaseMemoryProvider())
