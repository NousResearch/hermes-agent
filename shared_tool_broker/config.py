from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class BrokerSettings:
    hermes_home: Path
    host: str = "127.0.0.1"
    port: int = 8767
    log_level: str = "INFO"
    notion_version: str = "2025-09-03"
    notion_api_key: str = ""
    grain_mcp_url: str = "https://api.grain.com/_/mcp"
    granola_mcp_url: str = "https://mcp.granola.ai/mcp"
    tailscale_base_url: str = "https://rj-spark.tailc13f7e.ts.net"
    zoom_account_id: str = ""
    zoom_client_id: str = ""
    zoom_client_secret: str = ""
    affinity_api_key: str = ""
    default_google_subject: str = "me"
    enable_debug_results: bool = False
    tool_broker_env: Path | None = None
    idempotency_store: Path = field(default_factory=lambda: Path.home() / ".hermes" / "tool-broker-idempotency.json")

    @classmethod
    def load(cls) -> "BrokerSettings":
        hermes_home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
        load_dotenv(hermes_home / ".env", override=False)
        load_dotenv(Path.home() / ".config" / "onecli" / "hermes-spark-proxy.env", override=False)
        extra_env = hermes_home / "tool-broker.env"
        if extra_env.exists():
            load_dotenv(extra_env, override=False)
        settings = cls(
            hermes_home=hermes_home,
            host=os.getenv("HERMES_TOOL_BROKER_HOST", "127.0.0.1"),
            port=int(os.getenv("HERMES_TOOL_BROKER_PORT", "8767")),
            log_level=os.getenv("HERMES_TOOL_BROKER_LOG_LEVEL", "INFO").upper(),
            notion_version=os.getenv("NOTION_VERSION", "2025-09-03"),
            notion_api_key=os.getenv("NOTION_API_KEY", "").strip(),
            grain_mcp_url=os.getenv("GRAIN_MCP_URL", "https://api.grain.com/_/mcp").strip(),
            granola_mcp_url=os.getenv("GRANOLA_MCP_URL", "https://mcp.granola.ai/mcp").strip(),
            tailscale_base_url=os.getenv("HERMES_SHARED_TOOLS_BASE_URL", "https://rj-spark.tailc13f7e.ts.net").rstrip("/"),
            zoom_account_id=os.getenv("ZOOM_ACCOUNT_ID", "").strip(),
            zoom_client_id=os.getenv("ZOOM_CLIENT_ID", "").strip(),
            zoom_client_secret=os.getenv("ZOOM_CLIENT_SECRET", "").strip(),
            affinity_api_key=os.getenv("AFFINITY_API_KEY", "").strip(),
            default_google_subject=os.getenv("GWS_DEFAULT_SUBJECT", "me").strip() or "me",
            enable_debug_results=_bool_env("HERMES_TOOL_BROKER_DEBUG_RESULTS", False),
            tool_broker_env=extra_env if extra_env.exists() else None,
            idempotency_store=Path(os.getenv("HERMES_TOOL_BROKER_IDEMPOTENCY_STORE", hermes_home / "tool-broker-idempotency.json")),
        )
        return settings
