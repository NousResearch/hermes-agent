"""Core-owned wrapper around a plugin-registered login backend.

Every plugin backend reaches the browser vault only through ``PluginLoginBackend``, so the isolation
rules hold however the plugin implements ``LoginBackend``:

- Its namespace is ``<registered name>:``, taken from the registry key, never from attributes the plugin
  can change later. Ownership is decided here, so a plugin's own ``owns()`` is never consulted and it
  is never handed a handle outside its namespace (built-in ``vault_``/``op:``/``bw:`` or another plugin's).
- It is handed the handle and nothing else: no page, CDP session, task id, other entries or master
  password. ``needs_unlock`` is always False, so Hermes' master-password prompt never runs for it.
- Its exceptions are replaced by a fixed-text ``PluginBackendError`` raised outside the ``except``, so a
  secret in a plugin's error text reaches no tool result, log line or exception chain. Returned values
  are validated and copied into core-owned objects inside the same guard: reading a field of a
  plugin-owned object (a property, ``__getattribute__``, a ``str`` subclass) can run plugin code too.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Optional

from agent.vault_backends.base import LoginBackend
from agent.vault_store import VAULT_KINDS, VaultItemMeta

_FAILED = object()
_INVALID = object()
_TEXT_FIELDS = ("id", "kind", "label", "created_at")
_OPTIONAL_TEXT_FIELDS = ("origin", "identifier", "identifier_type")


def _text(value: Any) -> Any:
    """An exact ``str`` copy (a subclass's methods never run outside the guard), or ``_INVALID``."""
    return str.__str__(value) if isinstance(value, str) else _INVALID


def _materialize(meta: Any) -> Any:
    """A core-owned ``VaultItemMeta`` copied from the plugin's, or ``_INVALID`` when a field has the wrong type."""
    if not isinstance(meta, VaultItemMeta):
        return _INVALID
    fields = {f: _text(getattr(meta, f)) for f in _TEXT_FIELDS}
    for f in _OPTIONAL_TEXT_FIELDS:
        value = getattr(meta, f)
        fields[f] = None if value is None else _text(value)
    has_otp, origins = meta.has_otp, meta.allowed_origins
    if not isinstance(origins, tuple) or type(has_otp) is not bool:
        return _INVALID
    fields["allowed_origins"] = tuple(_text(o) for o in origins)
    if _INVALID in fields.values() or _INVALID in fields["allowed_origins"] or fields["kind"] not in VAULT_KINDS:
        return _INVALID
    return VaultItemMeta(has_otp=has_otp, **fields)


class PluginBackendError(RuntimeError):
    """A plugin login backend failed or broke its contract (fixed text: never the plugin's own message)."""


class PluginLoginBackend(LoginBackend):
    needs_unlock = False

    def __init__(self, name: str, plugin: LoginBackend):
        self.name, self.prefix, self._plugin = name, f"{name}:", plugin
        self.display_name = f"{name} (plugin)"  # core-composed: a plugin cannot present itself as a built-in manager

    def owns(self, handle: str) -> bool:
        return isinstance(handle, str) and handle.startswith(self.prefix)

    def _call(self, fn: Callable[[], Any]) -> Any:
        try:
            result = fn()
        except Exception:  # health: allow BLE001 -- the plugin's error text may hold a credential; replaced below
            result = _FAILED
        if result is _FAILED:
            raise PluginBackendError(f"The {self.name} login backend failed.")
        return result

    def _require_own(self, handle: str) -> None:
        if not self.owns(handle):
            raise PluginBackendError(f"Handle is outside the {self.name} login backend's namespace.")

    def _require(self, ok: bool) -> None:
        if not ok:
            raise PluginBackendError(f"The {self.name} login backend returned an invalid value.")

    def list_items(self) -> list[VaultItemMeta]:
        items = self._call(lambda: [_materialize(m) for m in self._plugin.list_items()])
        self._require(all(m is not _INVALID for m in items))
        return [m for m in items if self.owns(m.id)]

    def get_meta(self, handle: str) -> Optional[VaultItemMeta]:
        if not self.owns(handle):
            return None

        def fetch() -> Any:
            meta = self._plugin.get_meta(handle)
            return None if meta is None else _materialize(meta)

        meta = self._call(fetch)
        self._require(meta is not _INVALID)
        return meta if meta is not None and meta.id == handle else None

    def resolve_password(self, handle: str) -> str:
        self._require_own(handle)
        value = self._call(lambda: _text(self._plugin.resolve_password(handle)))
        self._require(value is not _INVALID)
        return value

    def resolve_otp(self, handle: str) -> Optional[str]:
        self._require_own(handle)

        def fetch() -> Any:
            value = self._plugin.resolve_otp(handle)
            return None if value is None else _text(value)

        value = self._call(fetch)
        self._require(value is not _INVALID)
        return value

    def resolve_secret(self, handle: str) -> dict[str, str]:
        self._require_own(handle)

        def fetch() -> Any:
            value = self._plugin.resolve_secret(handle)
            if not isinstance(value, dict):
                return _INVALID
            copied = {_text(k): _text(v) for k, v in value.items()}
            return _INVALID if _INVALID in copied or _INVALID in copied.values() else copied

        value = self._call(fetch)
        self._require(value is not _INVALID)
        return value

    @contextmanager
    def fill_context(self):
        """The plugin's own context around one of ITS fills; it never sees the core's exception."""
        context = self._call(lambda: self._plugin.fill_context())
        self._call(lambda: context.__enter__())
        try:
            yield
        finally:
            self._call(lambda: context.__exit__(None, None, None))
