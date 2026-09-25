"""Runtime wiring between the Mattermost plugin and Hermes' API server."""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
from collections.abc import Callable
from typing import Any

from .session_binding_api import MattermostSessionBindingAPI
from .session_bindings import (
    MattermostSessionBindingStore,
    SessionBinding,
    normalize_mattermost_id,
)

logger = logging.getLogger(__name__)


class MattermostSessionBindingRuntime:
    """Own live adapter references without adding Mattermost logic to Hermes core."""

    def __init__(
        self,
        *,
        store_factory: Callable[[], MattermostSessionBindingStore] = MattermostSessionBindingStore,
    ) -> None:
        self._mattermost_adapter: Any = None
        self._mattermost_loop: asyncio.AbstractEventLoop | None = None
        self._store_factory = store_factory
        self._pending_api_turns: dict[tuple[str, str], tuple[SessionBinding, str, str]] = {}
        self._pending_lock = threading.Lock()

    def wire_mattermost(self, _native: Any, adapter: Any) -> None:
        self._mattermost_adapter = adapter
        try:
            self._mattermost_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._mattermost_loop = None

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

    async def create_thread(
        self, session_id: str, channel_id: str, title: str
    ) -> tuple[str, str]:
        channel = normalize_mattermost_id(channel_id, field="channel_id")
        adapter = self._mattermost_adapter
        if adapter is None or not getattr(adapter, "is_connected", False):
            raise RuntimeError("Mattermost adapter is not connected")
        result = await adapter.create_session_thread(
            channel, title, session_id=session_id
        )
        if not result.success or not result.message_id:
            raise RuntimeError(result.error or "Mattermost returned no root post id")
        root = normalize_mattermost_id(result.message_id, field="root_post_id")
        return channel, root

    def wire_api_server(self, app: Any, adapter: Any) -> None:
        MattermostSessionBindingAPI(
            adapter,
            target_normalizer=self.normalize_target,
            thread_creator=self.create_thread,
            connected_probe=self.mattermost_connected,
            store_factory=self._store_factory,
        ).register_routes(app)

    def post_llm_call(
        self,
        *,
        session_id: str = "",
        turn_id: str = "",
        user_message: Any = "",
        assistant_response: Any = "",
        platform: str = "",
        **_: Any,
    ) -> None:
        """Capture API output until the turn's successful completion is confirmed."""
        if platform != "api_server" or not session_id or not turn_id:
            return
        user_text = str(user_message or "").strip()
        assistant_text = str(assistant_response or "").strip()
        if not user_text or not assistant_text:
            return
        try:
            binding = self._store_factory().get_by_session(session_id)
        except Exception:
            logger.exception("Mattermost binding lookup failed for API session %s", session_id)
            return
        if binding is None:
            return
        with self._pending_lock:
            self._pending_api_turns[(session_id, turn_id)] = (
                binding, user_text, assistant_text
            )

    def on_session_end(
        self,
        *,
        session_id: str = "",
        turn_id: str = "",
        completed: bool = False,
        failed: bool = False,
        interrupted: bool = False,
        **_: Any,
    ) -> None:
        """Mirror only turns the agent finalizer classified as successful."""
        with self._pending_lock:
            pending = self._pending_api_turns.pop((session_id, turn_id), None)
        if pending is None or not completed or failed or interrupted:
            return
        binding, user_text, assistant_text = pending
        adapter, loop = self._mattermost_adapter, self._mattermost_loop
        if adapter is None or loop is None or not loop.is_running():
            return

        async def _deliver() -> None:
            await self._mirror_api_turn(adapter, binding, turn_id, user_text, assistant_text)

        def _schedule() -> None:
            task = loop.create_task(_deliver())
            task.add_done_callback(self._log_delivery_result)

        loop.call_soon_threadsafe(_schedule)

    @staticmethod
    def _log_delivery_result(task: asyncio.Task) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Mattermost session mirror failed")

    async def _mirror_api_turn(
        self,
        adapter: Any,
        binding: SessionBinding,
        turn_id: str,
        user_text: str,
        assistant_text: str,
    ) -> None:
        store = self._store_factory()
        for role, content in (("user", user_text), ("assistant", assistant_text)):
            claimed = await asyncio.to_thread(
                store.claim_delivery, binding.session_id, turn_id, role
            )
            if not claimed:
                continue
            try:
                result = await adapter.send_session_mirror(
                    binding.channel_id,
                    binding.root_post_id,
                    content,
                    session_id=binding.session_id,
                    turn_id=turn_id,
                    role=role,
                )
                if not result.success or not result.message_id:
                    raise RuntimeError(result.error or "Mattermost returned no post id")
                await asyncio.to_thread(
                    store.complete_delivery,
                    binding.session_id,
                    turn_id,
                    role,
                    result.message_id,
                )
            except Exception:
                await asyncio.to_thread(
                    store.release_delivery, binding.session_id, turn_id, role
                )
                raise

    async def pre_gateway_dispatch(
        self,
        *,
        event: Any,
        gateway: Any,
        **_: Any,
    ) -> dict[str, str] | None:
        """Route authorized Mattermost thread replies into the bound Hermes session."""
        source = getattr(event, "source", None)
        platform = getattr(getattr(source, "platform", None), "value", "")
        if platform != "mattermost":
            return None
        post = getattr(event, "raw_message", None)
        if not isinstance(post, dict):
            return None
        props = post.get("props") if isinstance(post.get("props"), dict) else {}
        if props.get("hermes_origin") == "api_server":
            return {"action": "skip", "reason": "mattermost_bridge_echo"}
        channel_id = str(post.get("channel_id") or getattr(source, "chat_id", "") or "")
        root_post_id = str(post.get("root_id") or post.get("id") or "")
        if not channel_id or not root_post_id:
            return None
        try:
            binding = await asyncio.to_thread(
                self._store_factory().resolve, channel_id, root_post_id
            )
        except Exception:
            logger.exception("Mattermost reverse binding lookup failed")
            return None
        if binding is None:
            return None
        try:
            authorized = gateway._is_user_authorized_for_source(
                source, allow_adapter_delegation=False
            )
        except Exception:
            logger.exception("Mattermost binding authorization check failed")
            return None
        if not authorized:
            return None

        session_db = getattr(gateway, "_session_db", None)
        if session_db is None:
            return {"action": "skip", "reason": "bound_session_database_unavailable"}
        session = await self._maybe_await(session_db.get_session(binding.session_id))
        if not session:
            return {"action": "skip", "reason": "bound_session_missing"}
        session_key = gateway._session_key_for_source(source)
        current = await gateway.async_session_store.get_or_create_session(
            source, touch_activity=False
        )
        if current.session_id != binding.session_id:
            switched = await gateway.async_session_store.switch_session(
                session_key,
                binding.session_id,
                expected_session_id=current.session_id,
            )
            if switched is None:
                return {"action": "skip", "reason": "bound_session_route_changed"}
        metadata = getattr(event, "metadata", None)
        if isinstance(metadata, dict):
            metadata["mattermost_bound_session_id"] = binding.session_id
        return {"action": "allow"}

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        return await value if inspect.isawaitable(value) else value
