"""Gateway runtime binding for Teams chat-context ingestion."""

from __future__ import annotations

import logging
from typing import Any

from gateway.config import Platform
from plugins.teams_context.graph import TeamsContextGraph
from plugins.teams_context.store import TeamsContextStore
from tools.microsoft_graph_auth import GraphCredentials, MicrosoftGraphTokenProvider
from tools.microsoft_graph_client import MicrosoftGraphClient

logger = logging.getLogger(__name__)


def build_runtime(gateway_config: Any) -> TeamsContextGraph:
    cfg = getattr(gateway_config, "raw_config", None)
    if not isinstance(cfg, dict):
        try:
            from hermes_cli.config import load_config

            cfg = load_config()
        except Exception:
            cfg = {}
    teams_context_cfg = dict((cfg or {}).get("teams_context") or {})
    store = TeamsContextStore(teams_context_cfg.get("store_path"))
    credentials = GraphCredentials.from_env()
    return TeamsContextGraph(
        client=MicrosoftGraphClient(MicrosoftGraphTokenProvider(credentials)),
        store=store,
        tenant_id=credentials.tenant_id,
    )


def _is_chat_notification(notification: dict[str, Any]) -> bool:
    resource = str(notification.get("resource") or "").strip().lower().strip("/")
    return resource.startswith("chats/") or resource.startswith("chats(")


def bind_gateway_runtime(gateway: Any) -> bool:
    adapter = gateway.adapters.get(Platform.MSGRAPH_WEBHOOK)
    if adapter is None:
        return False
    if getattr(gateway, "_teams_context_runtime", None) is not None:
        return True
    try:
        runtime = build_runtime(gateway.config)
    except Exception as exc:
        gateway._teams_context_runtime_error = str(exc)
        logger.warning("Teams context runtime unavailable: %s", exc)
        return False

    async def _schedule(notification: dict[str, Any], event: Any) -> None:
        del event
        if not _is_chat_notification(notification):
            return
        await runtime.ingest_notification(notification)

    if hasattr(adapter, "register_notification_scheduler"):
        adapter.register_notification_scheduler("teams_context", _schedule)
    else:
        adapter.set_notification_scheduler(_schedule)
    gateway._teams_context_runtime = runtime
    gateway._teams_context_runtime_error = None
    return True
