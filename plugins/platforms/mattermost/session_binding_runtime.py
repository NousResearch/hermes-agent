"""Runtime wiring between the Mattermost plugin and Hermes' API server."""

from __future__ import annotations

from typing import Any

from .session_binding_api import MattermostSessionBindingAPI
from .session_bindings import normalize_mattermost_id


class MattermostSessionBindingRuntime:
    """Own live adapter references without adding Mattermost logic to Hermes core."""

    def __init__(self) -> None:
        self._mattermost_adapter: Any = None

    def wire_mattermost(self, _native: Any, adapter: Any) -> None:
        self._mattermost_adapter = adapter

    def mattermost_connected(self) -> bool:
        adapter = self._mattermost_adapter
        return bool(adapter is not None and getattr(adapter, "is_connected", False))

    async def normalize_target(self, channel_id: str, root_post_id: str) -> tuple[str, str]:
        channel = normalize_mattermost_id(channel_id, field="channel_id")
        requested_post = normalize_mattermost_id(root_post_id, field="root_post_id")
        adapter = self._mattermost_adapter
        if adapter is None or not getattr(adapter, "is_connected", False):
            raise RuntimeError("Mattermost adapter is not connected")
        post = await adapter._api_get(f"posts/{requested_post}")
        if not post or not post.get("id"):
            raise LookupError(f"Mattermost post not found: {requested_post}")
        if str(post.get("channel_id") or "") != channel:
            raise LookupError("Mattermost post does not belong to the requested channel")
        root = str(post.get("root_id") or post["id"])
        return channel, normalize_mattermost_id(root, field="root_post_id")

    def wire_api_server(self, app: Any, adapter: Any) -> None:
        MattermostSessionBindingAPI(
            adapter,
            target_normalizer=self.normalize_target,
            connected_probe=self.mattermost_connected,
        ).register_routes(app)
