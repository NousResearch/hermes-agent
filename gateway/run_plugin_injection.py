"""Plugin-triggered gateway turns (``PluginContext.inject_message``): the process-wide injector this
runner publishes, the thread-safe scheduler, and the dispatch through the adapter message path."""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging

from gateway.platforms.event import MessageEvent, MessageType

logger = logging.getLogger("gateway.run")


class GatewayPluginInjectionMixin:
    def _install_plugin_message_injector(self) -> None:
        """Publish this live gateway's plugin message scheduler process-wide."""
        from hermes_cli.plugins import publish_gateway_message_host

        publish_gateway_message_host(self, self._schedule_plugin_message_injection)

    def _clear_plugin_message_injector(self) -> None:
        """Remove this runner's scheduler without clobbering a newer owner."""
        from hermes_cli.plugins import clear_published_gateway_message_host

        clear_published_gateway_message_host(self)

    def _schedule_plugin_message_injection(
        self, *, session_key: str, content: str, plugin_id: str
    ) -> bool:
        """Schedule a plugin-triggered turn on the live gateway loop (thread-safe)."""
        from gateway.run import safe_schedule_threadsafe
        loop = getattr(self, "_gateway_loop", None)
        if not getattr(self, "_running", False) or loop is None or loop.is_closed():
            return False

        coro = self._dispatch_plugin_message_injection(
            session_key=session_key, content=content, plugin_id=plugin_id,
        )
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if current_loop is loop:
            try:
                future = loop.create_task(coro)
            except Exception:
                coro.close()
                logger.warning("Plugin message injection scheduling failed", exc_info=True)
                return False
            self._background_tasks.add(future)
            future.add_done_callback(self._background_tasks.discard)
        else:
            future = safe_schedule_threadsafe(
                coro, loop, logger=logger, log_message="Plugin message injection scheduling failed",
                log_level=logging.WARNING,
            )
            if future is None:
                return False

        def _log_result(completed) -> None:
            try:
                if completed.result():
                    return
                what, exc = "was not routed", None
            except (asyncio.CancelledError, concurrent.futures.CancelledError):
                return
            except Exception as err:
                what, exc = "failed", err
            logger.warning(
                "Plugin message injection %s: plugin=%s session=%s", what, plugin_id, session_key, exc_info=exc,
            )

        future.add_done_callback(_log_result)
        return True

    async def _dispatch_plugin_message_injection(
        self, *, session_key: str, content: str, plugin_id: str
    ) -> bool:
        """Route a plugin-triggered turn through the session's live adapter."""
        def _accepting() -> bool:
            return getattr(self, "_running", False) and not getattr(self, "_draining", False)

        if not _accepting():
            return False
        entry = await self.async_session_store.lookup_by_session_key(session_key)
        if entry is None or entry.origin is None or not _accepting():
            return False

        from gateway.session_identity import replace_source
        source = replace_source(self._restored_source(entry))
        try:
            authorized = self._is_user_authorized_for_source(source, allow_adapter_delegation=False)
        except Exception:
            logger.warning(
                "Plugin message injection authorization check failed: plugin=%s session=%s",
                plugin_id, session_key, exc_info=True,
            )
            return False
        if not authorized:
            logger.warning(
                "Plugin message injection denied by current gateway authorization: "
                "plugin=%s session=%s", plugin_id, session_key,
            )
            return False

        adapter = self._delivery_adapter_for(source)
        if adapter is None:
            return False

        await adapter.handle_message(MessageEvent(
            text=content, message_type=MessageType.TEXT, source=source, internal=True,
            allow_gateway_control=False,
            metadata={
                "hermes_plugin_id": plugin_id, "hermes_plugin_injection": True,
                "gateway_session_key": session_key, "gateway_session_id": entry.session_id,
                "gateway_session_strict": True,
            },
        ))
        logger.info(
            "Plugin message injection dispatched: plugin=%s session=%s session_id=%s",
            plugin_id, session_key, entry.session_id,
        )
        return True
