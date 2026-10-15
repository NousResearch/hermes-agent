"""Paused-session recovery ownership and routing commit boundaries."""
from __future__ import annotations
import asyncio
import logging
from contextlib import suppress
from pathlib import Path
from typing import Optional

logger = logging.getLogger("gateway.run")


class GatewayTurnRecoveryMixin:
    @staticmethod
    async def _await_session_policy_commit(awaitable):
        """Keep admission owned until a storage worker and its publication tail settle."""
        task = asyncio.ensure_future(awaitable)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    logger.warning("Session policy task failed while settling cancellation", exc_info=True)
                    break
            try:
                task.result()
            except BaseException:
                logger.warning("Session policy write failed during cancellation", exc_info=True)
            raise

    async def _compression_exhaustion_action(self, source) -> Optional[str]:
        """Read only the explicit policy leaf; uncertainty is never permission to clear a pause."""
        def read():
            import yaml
            from gateway.run import _gateway_config_home
            from hermes_cli.config import _expand_env_vars
            from hermes_cli.managed_scope import get_managed_dir

            home = (self._resolve_profile_home_for_source(source)
                    if getattr(self.config, "multiplex_profiles", False) else _gateway_config_home())
            if home is None:
                raise RuntimeError("Source profile home is unavailable")
            managed = get_managed_dir()
            paths = [Path(home) / "config.yaml"]
            if managed is not None:
                paths.append(managed / "config.yaml")
            choice = None
            for path in paths:
                try:
                    raw = path.read_text(encoding="utf-8-sig")
                except FileNotFoundError:
                    continue
                layer = yaml.safe_load(raw)
                if layer is None:
                    continue
                if not isinstance(layer, dict):
                    raise ValueError("Configuration must be a mapping")
                if "compression" not in layer:
                    continue
                compression = layer["compression"]
                if not isinstance(compression, dict):
                    raise ValueError("compression must be a mapping")
                if "exhaustion_action" in compression:
                    choice = compression["exhaustion_action"]
            # The adapter installs the source profile scope; to_thread preserves it.
            # Expand only the winning leaf after both layers pass the strict guards.
            if isinstance(choice, str):
                choice = _expand_env_vars(choice)
            return choice if isinstance(choice, str) and choice in {"reset", "pause"} else None
        try:
            return await asyncio.to_thread(read)
        except Exception:  # health: allow BLE001 -- config errors may include credentials; fail closed without details
            logger.warning("Could not establish explicit compression.exhaustion_action; existing pause retained")
            return None

    @staticmethod
    def _compression_pause_notice(*, persistence_failed=False):
        if persistence_failed:
            return ("Compression recovery could not be persisted. Retry /compress or /new after fixing "
                    "storage, or /resume an authorized session. An initial pause held only in memory "
                    "cannot be guaranteed after a restart while storage is unavailable.")
        return ("Session paused: the conversation could not be compressed further. History is preserved. "
                "Use /compress, /new, or /resume another authorized session. Setting "
                "compression.exhaustion_action: reset explicitly resets this pause on your next message.")

    def _sync_compression_recovery_binding(self, source, entry, expected_id, *, reason):
        """Do not let a settled old tail rebind a route replaced by /new or /resume."""
        with self.session_store._lock:
            current = self.session_store._entries.get(entry.session_key)
            if current is entry and current.session_id == expected_id:
                self._sync_telegram_topic_binding(source, entry, reason=reason)

    async def _reset_session_after_compression_exhaustion(
        self, session_key, session_entry, source, *, require_primary=True, cleanup_for_manual_reset=False,
        event=None,
    ):
        expected_id = session_entry.session_id
        async def commit():
            new_entry = await self.async_session_store.reset_session(
                session_key, require_primary=require_primary, expected_session_id=expected_id,
            )
            if new_entry is None:
                raise RuntimeError("Session route changed before compression recovery")
            if not cleanup_for_manual_reset:
                from gateway.session_automation import move_route_wakes
                await move_route_wakes(self, session_key, expected_id)
            if cleanup_for_manual_reset:
                # /new is a full resource boundary. Keep its teardown inside the settled commit
                # so failure preserves the old resources and repeated cancellation retains admission.
                from gateway.slash_commands_session import _reset_process_scoped_tool_state
                await self._cleanup_old_agent_for_reset(session_key)
                with suppress(Exception):
                    from tools.async_delegation import interrupt_for_session
                    interrupt_for_session(session_key=session_key, reason="session_reset",
                                          parent_session_id=str(expected_id or ""))
                _reset_process_scoped_tool_state()
            self._evict_cached_agent(session_key)
            queued_events = None if cleanup_for_manual_reset else self._overflow_queue(session_key)
            self._clear_conversation_scope(session_key, reason="compression_exhausted_reset")
            if queued_events:
                self._session_state(session_key).conversation.queued_events = queued_events
            try:
                await asyncio.to_thread(
                    self._sync_compression_recovery_binding, source, new_entry, new_entry.session_id,
                    reason="compression-exhausted-reset",
                )
            except Exception:
                logger.warning("Compression recovery committed but topic binding sync failed", exc_info=True)
            if event is not None:
                event._gateway_preserve_pending = False
            return new_entry
        return await self._await_session_policy_commit(commit())
