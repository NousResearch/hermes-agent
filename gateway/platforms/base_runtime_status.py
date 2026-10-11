"""An adapter's platform entry in ``gateway_state.json``, published for the runner's adapters only."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gateway.platforms.base import BasePlatformAdapter


def publish_adapter_runtime_status(adapter: BasePlatformAdapter, **kwargs: Any) -> None:
    """Publish ``adapter``'s platform status when the gateway runner wired it.

    The file is the running gateway's record. A sender that connects a throwaway adapter for one
    message (the Matrix fallback and BlueBubbles senders in ``tools/send_message_senders.py``, and
    the Feishu and WeCom ``standalone_sender_fn``s that out-of-process cron delivery uses) is not
    the gateway: from another process its write re-stamps the gateway's pid/argv/state and drops
    the other platforms' entries, and inside the gateway its disconnect marks the live platform
    ``disconnected``.
    """
    if not adapter._runtime_status_owned:
        return
    from gateway.status import publish_runtime_status
    # Multiplexed adapters share the status file; the runner stamps ``<profile>:<platform>``.
    platform_key = getattr(adapter, "_runtime_status_platform_key", None) or adapter.platform.value
    publish_runtime_status(platform=platform_key, **kwargs)
