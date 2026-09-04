"""Durable model-selection authority for gateway commands.

The historical slash-command implementation owns provider resolution and all
user-facing picker/confirmation flows. This mixin wraps that implementation at
its commit boundary so a global model choice has one durable authority:
profile config first, then deletion of the redundant session override.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, TypeVar

from agent.i18n import t
from gateway.platforms.base import MessageEvent

_T = TypeVar("_T")


def _runner_lock(runner: Any, name: str) -> asyncio.Lock:
    lock = runner.__dict__.get(name)
    if lock is None:
        lock = asyncio.Lock()
        runner.__dict__[name] = lock
    return lock


def _keyed_lock(runner: Any, name: str, key: str) -> asyncio.Lock:
    locks = runner.__dict__.setdefault(name, {})
    lock = locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        locks[key] = lock
    return lock


@asynccontextmanager
async def _model_switch_guard(
    runner: Any,
    session_key: str,
    *,
    persist_global: bool,
    config_path: Path,
):
    """Serialize one session commit and shared profile config writes."""
    session_lock = _keyed_lock(
        runner, "_model_authority_session_locks", session_key
    )
    if persist_global:
        profile_key = str(config_path.resolve()).casefold()
        profile_lock = _keyed_lock(
            runner, "_model_authority_profile_locks", profile_key
        )
        async with profile_lock:
            async with session_lock:
                yield
        return
    async with session_lock:
        yield


def _instance_override(obj: Any, name: str, value: Any) -> tuple[bool, Any]:
    """Install an instance method override and return its restoration token."""
    had_value = name in getattr(obj, "__dict__", {})
    previous = obj.__dict__.get(name) if had_value else None
    setattr(obj, name, value)
    return had_value, previous


def _restore_instance_override(
    obj: Any, name: str, token: tuple[bool, Any]
) -> None:
    had_value, previous = token
    if had_value:
        setattr(obj, name, previous)
    else:
        try:
            delattr(obj, name)
        except AttributeError:
            pass


def _global_config_matches(config_path: Path, target: dict[str, Any]) -> bool:
    """Return whether config_path durably names the selected model route."""
    try:
        from gateway.run import _load_gateway_config

        cfg = _load_gateway_config(config_path=config_path)
    except Exception:
        return False
    if not isinstance(cfg, dict):
        return False
    model_cfg = cfg.get("model")
    if isinstance(model_cfg, str):
        configured_model = model_cfg.strip()
        configured_provider = ""
    elif isinstance(model_cfg, dict):
        configured_model = str(
            model_cfg.get("default") or model_cfg.get("model") or ""
        ).strip()
        configured_provider = str(model_cfg.get("provider") or "").strip()
    else:
        return False
    target_model = str(target.get("model") or "").strip()
    target_provider = str(target.get("provider") or "").strip()
    if not target_model or configured_model != target_model:
        return False
    return not target_provider or configured_provider == target_provider


def _replace_global_receipt(message: _T, warning: str) -> _T:
    """Replace the historical unconditional saved-global receipt."""
    if not isinstance(message, str):
        return message
    saved = t("gateway.model.saved_global").strip()
    lines = [line for line in message.splitlines() if line.strip() != saved]
    lines.append(warning)
    return "\n".join(lines)  # type: ignore[return-value]


class _OverrideCapture:
    """Capture the selected non-secret route at the session-store boundary."""

    def __init__(self, runner: Any, session_key: str) -> None:
        self.runner = runner
        self.session_key = session_key
        self.target: Optional[dict[str, Any]] = None
        self.store: Any = None
        self.original: Optional[Callable[..., Awaitable[Any]]] = None
        self.token: Optional[tuple[bool, Any]] = None

    async def __aenter__(self) -> "_OverrideCapture":
        store = self.runner.__dict__.get("_async_session_store")
        if store is None:
            try:
                store = self.runner.async_session_store
            except Exception:
                store = None
        setter = getattr(store, "set_model_override", None)
        if store is None or not callable(setter):
            return self
        self.store = store
        self.original = setter

        async def recording_setter(key: str, value: Optional[dict]) -> Any:
            if key == self.session_key and isinstance(value, dict):
                self.target = dict(value)
            assert self.original is not None
            return await self.original(key, value)

        self.token = _instance_override(
            store, "set_model_override", recording_setter
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self.store is not None and self.token is not None:
            _restore_instance_override(
                self.store, "set_model_override", self.token
            )

    async def write(self, value: Optional[dict]) -> None:
        if self.original is None:
            return
        await self.original(self.session_key, value)


class GatewayModelAuthorityMixin:
    """Wrap the legacy model command with serialized durable settlement."""

    async def _handle_model_command(self, event: MessageEvent) -> Optional[str]:
        from hermes_cli.model_switch import (
            parse_model_switch_args,
            resolve_persist_behavior,
        )
        from gateway.run import _hermes_home

        request = parse_model_switch_args(event.get_command_args().strip())
        persist_global = False
        if not request.errors:
            persist_global = resolve_persist_behavior(
                request.is_global,
                request.is_session,
                is_once=request.is_once,
                explicit_provider=request.explicit_provider,
            )

        source = event.source
        profile_home = None
        if getattr(getattr(self, "config", None), "multiplex_profiles", False):
            profile_home = self._resolve_profile_home_for_source(source)
        config_path = (profile_home or _hermes_home) / "config.yaml"

        normalized_source = await asyncio.to_thread(
            self._normalize_source_for_session_key, source
        )
        session_key = self._session_key_for_source(normalized_source)

        async def settle(call: Callable[[], Awaitable[_T]]) -> _T:
            # Confirmation implementations may execute their handler inline.
            # In that case the outer invocation already owns the commit locks
            # and store capture; recurse into the callback without deadlocking.
            current_task = asyncio.current_task()
            active_tasks = self.__dict__.setdefault(
                "_model_authority_active_tasks", set()
            )
            if current_task in active_tasks:
                return await call()
            async with _model_switch_guard(
                self,
                session_key,
                persist_global=persist_global,
                config_path=config_path,
            ):
                active_tasks.add(current_task)
                try:
                    if not persist_global:
                        return await call()
                    # The historical handler reaches the session store before the
                    # config write. Serialize that temporary method interception so
                    # another profile cannot have its write attributed to this turn.
                    async with _runner_lock(self, "_model_authority_store_patch_lock"):
                        async with _OverrideCapture(self, session_key) as capture:
                            result = await call()
                            target = capture.target
                            if target is None:
                                return result
                            if not _global_config_matches(config_path, target):
                                try:
                                    await capture.write(target)
                                except Exception:
                                    pass
                                return _replace_global_receipt(
                                    result,
                                    "⚠ Global config write failed; retained for this session.",
                                )
                            try:
                                await capture.write(None)
                            except Exception:
                                # The matching override remains the truthful live
                                # authority if its durable deletion cannot settle.
                                self._session_model_overrides[session_key] = target
                                return _replace_global_receipt(
                                    result,
                                    "⚠ Global config saved, but session override cleanup "
                                    "failed; retained for this session.",
                                )
                            self._session_model_overrides.pop(session_key, None)
                            return result
                finally:
                    active_tasks.discard(current_task)

        def wrap_callback(callback: Callable[..., Awaitable[_T]]):
            async def guarded_callback(*args: Any, **kwargs: Any) -> _T:
                return await settle(lambda: callback(*args, **kwargs))

            return guarded_callback

        async def invoke_legacy() -> Optional[str]:
            adapter = None
            try:
                adapter = self._adapter_for_source(normalized_source)
            except Exception:
                adapters = getattr(self, "adapters", None) or {}
                adapter = adapters.get(getattr(normalized_source, "platform", None))

            patched: list[tuple[Any, str, tuple[bool, Any]]] = []
            # Callback wiring mutates only long enough for the legacy handler to
            # hand our guarded callback to the adapter/confirmation registry.
            async with _runner_lock(self, "_model_authority_callback_patch_lock"):
                send_picker = getattr(adapter, "send_model_picker", None)
                if callable(send_picker):
                    async def guarded_send_picker(*args: Any, **kwargs: Any):
                        callback = kwargs.get("on_model_selected")
                        if callable(callback):
                            kwargs["on_model_selected"] = wrap_callback(callback)
                        return await send_picker(*args, **kwargs)

                    token = _instance_override(
                        adapter, "send_model_picker", guarded_send_picker
                    )
                    patched.append((adapter, "send_model_picker", token))

                request_confirm = getattr(self, "_request_slash_confirm", None)
                if callable(request_confirm):
                    async def guarded_request_confirm(*args: Any, **kwargs: Any):
                        handler = kwargs.get("handler")
                        if callable(handler):
                            kwargs["handler"] = wrap_callback(handler)
                        return await request_confirm(*args, **kwargs)

                    token = _instance_override(
                        self, "_request_slash_confirm", guarded_request_confirm
                    )
                    patched.append((self, "_request_slash_confirm", token))

                try:
                    return await super(GatewayModelAuthorityMixin, self)._handle_model_command(
                        event
                    )
                finally:
                    for obj, name, token in reversed(patched):
                        _restore_instance_override(obj, name, token)

        return await settle(invoke_legacy)
