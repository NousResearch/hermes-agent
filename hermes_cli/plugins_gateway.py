"""Host-owned typed continuation and deferred gateway task lifecycle for plugins."""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional

if TYPE_CHECKING:
    from hermes_cli.plugins import PluginRegistration

logger = logging.getLogger("hermes_cli.plugins")

class PluginGatewayContextMixin:
    def set_continuation_observer_ready(self, ready: bool, *, surface: str) -> None:
        """Publish whether the plugin's supervised observer has connected.

        Readiness never grants authority: the configured grant, live task and
        exact destination are independently checked on every admission.
        """
        if type(ready) is not bool or surface not in {"gateway", "desktop"}:
            raise ValueError("continuation readiness requires a bool and a supported surface")
        setattr(self, f"_{surface}_observer_ready", ready)

    def register_desktop_task(self, factory, *, name):
        """Register a worker owned only by the authenticated serve runtime."""
        if not callable(factory) or not isinstance(name, str) or not name.strip():
            raise ValueError("desktop task requires a factory and name")
        factories = getattr(self._manager, "_desktop_task_factories", None)
        if factories is None:
            factories = self._manager._desktop_task_factories = {}
        key = f"{self.plugin_id}:{name}"
        if key in factories:
            raise ValueError("desktop task already registered")
        factories[key] = factory
        contexts = getattr(self._manager, "_desktop_contexts", None)
        if contexts is None:
            contexts = self._manager._desktop_contexts = {}
        contexts[self.plugin_id] = self
        def release():
            factories.pop(key, None)
            if contexts.get(self.plugin_id) is self:
                contexts.pop(self.plugin_id, None)
            self._desktop_observer_ready = False
            task = getattr(self._manager, "_desktop_tasks", {}).pop(key, None)
            if task is not None and not task.done():
                task.get_loop().call_soon_threadsafe(task.cancel)
        return self._track("desktop_task", name, release)

    def current_desktop_destination(self):
        consumer = getattr(self._manager, "_desktop_consumer", None)
        return consumer.current_destination() if consumer is not None else None

    def desktop_continuation_readiness(self, destination):
        consumer = getattr(self._manager, "_desktop_consumer", None)
        tasks = getattr(self._manager, "_desktop_tasks", {})
        supervised = any(key.startswith(f"{self.plugin_id}:") and not task.done() for key, task in tasks.items())
        if consumer is None or not supervised or not self._gateway_injection_allowed() or getattr(self, "_desktop_observer_ready", False) is not True:
            return {"ready": False, "status": "unauthorized"}
        return consumer.readiness(destination)

    def current_gateway_destination(self):
        from gateway.session_context import current_plugin_gateway_destination
        registered = self._manager._gateway_message_injector
        return current_plugin_gateway_destination(registered[0]) if registered else None

    def gateway_continuation_readiness(self, destination):
        """Validate one native turn destination on its gateway owner's loop."""
        from gateway.internal_events import create_gateway_system_event
        registered = self._manager._gateway_message_injector
        tasks = self._manager._gateway_tasks
        supervised = any(key[0] == self.plugin_id and not task.done() for key, task in tasks.items())
        if (not registered or not supervised or not self._gateway_injection_allowed()
                or getattr(self, "_gateway_observer_ready", False) is not True
                or destination != self.current_gateway_destination()):
            return {"ready": False, "status": "unauthorized"}
        owner = registered[0]
        loop = getattr(owner, "_gateway_loop", None)
        if loop is None or loop.is_closed() or not loop.is_running():
            return {"ready": False, "status": "stopping"}
        try:
            if asyncio.get_running_loop() is loop:
                return {"ready": False, "status": "unsupported_call_context"}
        except RuntimeError:
            pass
        try:
            _, event = create_gateway_system_event(content="Readiness probe", session_key=destination["session_key"],
                expected_session_id=destination["session_id"], event_id="readiness", event_kind="external_tool_completed",
                plugin_id=self.plugin_id, expected_route={key: destination[key] for key in
                    ("profile_name", "platform", "user_id", "chat_id", "topic_id")}, eligibility_check=lambda: True)
            pending = asyncio.run_coroutine_threadsafe(owner._gateway_system_event_target(event), loop)
            status, _, _ = pending.result(timeout=5)
        except Exception:
            if "pending" in locals():
                pending.cancel()
            return {"ready": False, "status": "unavailable"}
        if self._manager._gateway_message_injector is not registered:
            return {"ready": False, "status": "stopping"}
        return {"ready": status is None, "status": status or "ready", "destination": dict(destination)}

    def inject_desktop_system_event(self, content, *, destination, event_id, event_kind, eligibility_check, completion_check):
        from gateway.internal_events import create_desktop_system_event
        if not self._gateway_injection_allowed():
            return None
        consumer = getattr(self._manager, "_desktop_consumer", None)
        if consumer is None:
            return None
        content, event = create_desktop_system_event(
            content=content, destination=destination, event_id=event_id, event_kind=event_kind,
            plugin_id=self.plugin_id,
            eligibility_check=lambda: self._gateway_injection_allowed() and eligibility_check(),
            completion_check=lambda: self._gateway_injection_allowed() and completion_check())
        return consumer.inject(content, event)

    def register_gateway_task(
        self,
        factory: Callable[[], Any],
        *,
        name: Optional[str] = None,
    ) -> PluginRegistration:
        """Register an async worker factory owned by the live gateway loop.

        Registration is safe during synchronous plugin discovery.  The host
        calls the factory after the gateway injector is live and cancels its
        task before removing that injector during shutdown.
        """
        if not callable(factory):
            raise TypeError("register_gateway_task expects a callable factory")
        task_name = name or f"plugin:{self.plugin_id}:gateway"
        if not isinstance(task_name, str) or not task_name.strip():
            raise ValueError("gateway task name must be a non-empty string")
        task_name = task_name.strip()[:128]
        owner_key = self.manifest.key or self.manifest.name
        key = (owner_key, task_name)
        if key in self._manager._gateway_task_factories:
            raise ValueError(f"gateway task {task_name!r} is already registered")
        self._manager._gateway_task_factories[key] = factory

        def _release() -> None:
            self._manager._remove_gateway_task_factory(key, factory)

        handle = self._track("gateway_task", task_name, _release)
        logger.debug(
            "Plugin %s registered gateway task: %s", self.manifest.name, task_name
        )
        return handle

    def inject_gateway_system_event(
        self,
        content: str,
        *,
        session_key: str,
        expected_session_id: str,
        event_id: str,
        event_kind: str,
        expected_route: Mapping[str, str],
        eligibility_check: Callable[[], bool],
    ):
        """Submit a typed internal event to one exact live gateway session.

        The returned ``concurrent.futures.Future`` resolves only after the
        gateway turn and platform delivery reach a terminal outcome.  Invalid
        request shapes return ``None``; authorization and runtime refusals are
        represented by the Future's bounded result mapping.  ``expected_route``
        is snapshotted, and ``eligibility_check`` must be a synchronous,
        process-local predicate that remains true through model admission.
        """
        from gateway.internal_events import (
            create_gateway_system_event,
            gateway_system_event_is_eligible,
            new_gateway_event_receipt,
            resolve_gateway_event_receipt,
        )

        plugin_id = self.manifest.key or self.manifest.name
        try:
            content, system_event = create_gateway_system_event(
                content=content,
                session_key=session_key,
                expected_session_id=expected_session_id,
                event_id=event_id,
                event_kind=event_kind,
                plugin_id=plugin_id,
                expected_route=expected_route,
                eligibility_check=eligibility_check,
            )
        except ValueError:
            logger.warning(
                "inject_gateway_system_event: invalid request from plugin %s",
                plugin_id,
            )
            return None

        receipt = new_gateway_event_receipt()
        if not self._gateway_injection_allowed():
            resolve_gateway_event_receipt(
                receipt, "unauthorized", event=system_event
            )
            return receipt
        if not gateway_system_event_is_eligible(system_event):
            resolve_gateway_event_receipt(
                receipt, "unauthorized", event=system_event
            )
            return receipt
        if not self._manager.has_gateway_message_injector:
            resolve_gateway_event_receipt(receipt, "stopping", event=system_event)
            return receipt

        try:
            scheduled = self._manager.inject_gateway_system_event(
                content=content,
                system_event=system_event,
                receipt=receipt,
            )
        except Exception:
            logger.warning(
                "inject_gateway_system_event: gateway scheduling failed for plugin %s",
                plugin_id,
                exc_info=True,
            )
            resolve_gateway_event_receipt(receipt, "agent_error", event=system_event)
            return receipt
        if scheduled is not receipt:
            if isinstance(scheduled, type(receipt)):
                return scheduled
            resolve_gateway_event_receipt(receipt, "agent_error", event=system_event)
        return receipt



class PluginGatewayManagerMixin:
    def gateway_message_injector_owned_by(self, owner: object) -> bool:
        """Return whether owner still owns the published gateway."""
        registered = self._gateway_message_injector
        return registered is not None and registered[0] is owner

    def inject_gateway_system_event(self, **kwargs: Any):
        """Submit a typed system event through the live gateway owner."""
        registered = self._gateway_message_injector
        if registered is None:
            return None
        return registered[1](**kwargs)

    def gateway_injection_allowed(self, plugin_id: str) -> bool:
        """Read this manager's current profile-scoped gateway grant."""
        from hermes_cli.plugins import _plugin_home_scope, load_config_readonly
        try:
            with _plugin_home_scope(self.home_path):
                cfg = load_config_readonly() or {}
        except Exception:
            return False
        return ((cfg.get("plugins") or {}).get("entries") or {}).get(
            plugin_id, {}
        ).get("allow_gateway_injection") is True

    def _start_gateway_tasks(self, loop) -> None:
        """Start every registered gateway worker on the host event loop."""
        if loop is None or loop.is_closed():
            return
        self._gateway_task_loop = loop
        for key, factory in list(self._gateway_task_factories.items()):
            current = self._gateway_tasks.get(key)
            if current is not None and not current.done():
                continue
            try:
                coro = factory()
                if not asyncio.iscoroutine(coro):
                    raise TypeError("gateway task factory must return a coroutine")
                task = loop.create_task(coro, name=key[1])
            except Exception:
                logger.warning(
                    "Plugin gateway task failed to start: plugin=%s task=%s",
                    key[0],
                    key[1],
                    exc_info=True,
                )
                continue
            self._gateway_tasks[key] = task

            def _finished(done, *, task_key=key) -> None:
                if self._gateway_tasks.get(task_key) is done:
                    self._gateway_tasks.pop(task_key, None)
                if done.cancelled():
                    return
                try:
                    done.result()
                except Exception:
                    logger.warning(
                        "Plugin gateway task failed: plugin=%s task=%s",
                        task_key[0],
                        task_key[1],
                        exc_info=True,
                    )

            task.add_done_callback(_finished)

    def _stop_gateway_tasks(self) -> None:
        """Cancel all workers owned by the current gateway lifecycle."""
        for task in list(self._gateway_tasks.values()):
            if not task.done():
                task.cancel()
        self._gateway_tasks.clear()
        self._gateway_task_loop = None

    def _remove_gateway_task_factory(
        self,
        key: tuple[str, str],
        factory: Callable[[], Any],
    ) -> None:
        """Remove one owned worker and cancel its live task, if any."""
        if self._gateway_task_factories.get(key) is not factory:
            return
        self._gateway_task_factories.pop(key, None)
        task = self._gateway_tasks.pop(key, None)
        if task is not None and not task.done():
            task.cancel()
