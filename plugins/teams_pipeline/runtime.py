"""Gateway runtime wiring for the Teams meeting pipeline plugin."""

from __future__ import annotations

import logging
from typing import Any

from gateway.config import Platform
from plugins.teams_pipeline.pipeline import TeamsMeetingPipeline
from plugins.teams_pipeline.store import TeamsPipelineStore, resolve_teams_pipeline_store_path
from plugins.teams_pipeline.subscriptions import build_graph_client

logger = logging.getLogger(__name__)

# Teams platform settings mirrored into the pipeline's teams_delivery config (pipeline keys win).
_DELIVERY_KEYS = ("incoming_webhook_url", "access_token", "team_id", "channel_id", "chat_id")


def _teams_delivery_is_configured(teams_extra: dict[str, Any], teams_delivery: dict[str, Any]) -> bool:
    def pick(key: str) -> Any:
        return teams_delivery.get(key) or teams_extra.get(key)
    delivery_mode = str(teams_delivery.get("mode") or pick("delivery_mode") or "").strip().lower()
    if delivery_mode == "incoming_webhook":
        return bool(pick("incoming_webhook_url"))
    if delivery_mode == "graph":
        return bool(pick("chat_id") or (pick("team_id") and pick("channel_id")))
    return False


def build_pipeline_runtime_config(gateway_config: Any) -> dict[str, Any]:
    """Pipeline knobs live under ``teams.extra.meeting_pipeline``; Teams delivery targets are
    mirrored from the Teams platform config so they are not configured twice."""
    teams_config = gateway_config.platforms.get(Platform("teams"))
    teams_extra = dict((teams_config.extra or {}) if teams_config else {})
    pipeline_config = dict(teams_extra.get("meeting_pipeline") or {})
    if teams_config and teams_config.enabled:
        teams_delivery = dict(pipeline_config.get("teams_delivery") or {})
        if delivery_mode := str(teams_extra.get("delivery_mode") or "").strip():
            teams_delivery["mode"] = delivery_mode
        for key in _DELIVERY_KEYS:
            value = teams_extra.get(key)
            if value not in {None, ""}:
                teams_delivery[key] = value
        if teams_delivery:
            teams_delivery["enabled"] = _teams_delivery_is_configured(teams_extra, teams_delivery)
            pipeline_config["teams_delivery"] = teams_delivery
    return pipeline_config


def build_pipeline_runtime(gateway: Any) -> TeamsMeetingPipeline:
    teams_sender = None
    teams_config = gateway.config.platforms.get(Platform("teams"))
    pipeline_config = build_pipeline_runtime_config(gateway.config)
    if teams_config and teams_config.enabled and (pipeline_config.get("teams_delivery") or {}).get("enabled"):
        try:
            from plugins.platforms.teams.summary_writer import TeamsSummaryWriter
        except ImportError:
            logger.debug("TeamsSummaryWriter unavailable; Teams outbound delivery remains disabled until the adapter layer is present.")
        else:
            teams_sender = TeamsSummaryWriter(platform_config=teams_config)
    return TeamsMeetingPipeline(
        graph_client=build_graph_client(), store=TeamsPipelineStore(resolve_teams_pipeline_store_path()),
        config=pipeline_config, teams_sender=teams_sender,
    )


def bind_gateway_runtime(gateway: Any) -> bool:
    """Attach the Teams pipeline runtime to the msgraph webhook adapter."""
    adapter = gateway.adapters.get(Platform.MSGRAPH_WEBHOOK)
    if adapter is None:
        return False
    if getattr(gateway, "_teams_pipeline_runtime", None) is not None:
        return True
    try:
        runtime = build_pipeline_runtime(gateway)
    except Exception as exc:
        # Without a runtime, notifications must still be acked (and dropped) so Graph does not retry-storm.
        gateway._teams_pipeline_runtime_error = str(exc)
        logger.warning(
            "Teams pipeline runtime unavailable: %s. Installing a drop-scheduler "
            "so Graph notifications ack cleanly without piling up unbound.", exc,
        )

        async def _drop(notification: dict[str, Any], event: Any) -> None:
            logger.debug("Dropping Graph notification because runtime is unavailable: id=%s resource=%s",
                         notification.get("id"), notification.get("resource"))
        adapter.set_notification_scheduler(_drop)
        return False

    async def _schedule(notification: dict[str, Any], event: Any) -> None:
        await runtime.run_notification(notification)
    adapter.set_notification_scheduler(_schedule)
    gateway._teams_pipeline_runtime = runtime
    gateway._teams_pipeline_runtime_error = None
    return True
