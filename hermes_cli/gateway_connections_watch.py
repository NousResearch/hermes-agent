"""The Webapp listener follows saved-gateway edits made by another writer (native Desktop).

The HTTP routes tear down what an entry holds when the Webapp itself removes or re-points it. Desktop
writes the same ``gateway-connections.json`` directly, so this server polls the file the way Electron
polls it (``electron/ui-preferences.ts::registerUiPreferencesIpc``): a cheap stat, then the revision,
then per-entry dial fingerprints. An entry that is gone or dials elsewhere loses its relayed sockets
and its SSH tunnel; a removed one is forgotten entirely (``gateway_connections_reach.forget``). An
entry whose fingerprint did not change is never touched.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

from hermes_cli import gateway_connections as store

_log = logging.getLogger(__name__)
POLL_INTERVAL_S = 1.5
#: What decides where and how an entry dials (``connection_dial_fields_changed``).
_DIAL_FIELDS = ("kind", "url", "authMode", "org", "host", "user", "port", "keyPath", "remoteHermesPath",
                "remoteProfile", "token", "headers")


def fingerprints(registry: dict) -> dict[str, str]:
    return {c["id"]: json.dumps([c.get(k) for k in _DIAL_FIELDS], sort_keys=True, default=str)
            for c in registry["connections"]}


def _stamp(path: Path) -> Optional[tuple[int, int, int]]:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size, st.st_ino


def _teardown(changed: list[str], removed: list[str]) -> None:
    from hermes_cli import gateway_connections_oauth as oauth
    from hermes_cli import gateway_connections_reach as reach
    from hermes_cli import gateway_connections_ssh as ssh
    from hermes_cli.web_routers.webapp_relay_proxy import close_gateway_sockets

    # Each entry on its own: a host that stopped answering (ssh teardown raising) must not leave
    # the rest open or stop the watcher.
    steps = ([(close_gateway_sockets, i) for i in [*changed, *removed]]
             + [(reach.forget, i) for i in removed]
             + [(ssh.disconnect, i) for i in changed] + [(oauth.cancel_sign_in, i) for i in changed])
    for step, connection_id in steps:
        try:
            step(connection_id)
        except Exception:  # health: allow BLE001 -- one entry's teardown failing must not skip the rest
            _log.warning("Saved gateway %s: %s failed after another writer's edit", connection_id,
                         step.__name__, exc_info=True)


class StoreWatch:
    """Per-entry change detection over the store file; :meth:`check` is one poll step."""

    def __init__(self, path: Path):
        self.path = path
        self.stamp = _stamp(path)
        self.revision, self.prints = self._read()

    def _read(self) -> tuple[int, dict[str, str]]:
        registry = store.read_registry(self.path)
        return registry["revision"], fingerprints(registry)

    def check(self) -> tuple[list[str], list[str]]:
        """``(changed, removed)`` ids since the last check; both empty when the file did not move."""
        stamp = _stamp(self.path)
        if stamp == self.stamp:
            return [], []
        self.stamp = stamp
        revision, prints = self._read()
        if revision == self.revision and prints == self.prints:
            return [], []
        before, self.revision, self.prints = self.prints, revision, prints
        removed = [i for i in before if i not in prints]
        changed = [i for i in before if i in prints and prints[i] != before[i]]
        return changed, removed


async def _poll(watch: StoreWatch, interval: float) -> None:
    while True:
        await asyncio.sleep(interval)
        try:
            changed, removed = await asyncio.to_thread(watch.check)
            if changed or removed:
                _log.info("Saved gateways changed by another writer: re-pointed %s, removed %s", changed, removed)
                await asyncio.to_thread(_teardown, changed, removed)
        except (store.ConnectionStoreBusy, OSError, ValueError) as exc:
            _log.debug("Saved gateways poll skipped: %s", exc)


@asynccontextmanager
async def watch_store(interval: float = POLL_INTERVAL_S) -> AsyncIterator[StoreWatch]:
    """Poll the machine's store for the listener's lifetime. The path is resolved once here: the
    store lives at the machine root whichever profile a later request is scoped to."""
    watch = await asyncio.to_thread(StoreWatch, store.store_path())
    task = asyncio.create_task(_poll(watch, interval), name="gateway-connections-watch")
    try:
        yield watch
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
