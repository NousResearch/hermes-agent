"""An in-memory secret store that keeps secrets off disk: the reference for the view hooks.

:class:`MemoryViewProvider` holds each profile home's env names and auth records in a dict (the
"remote" side) and serves the process a snapshot of it refreshed on :meth:`refresh`, the way a
store with a poller would. Its bootstrap set is :data:`BOOTSTRAP`. It is what the routing tests
select as ``HERMES_SECRET_STORE`` and what ``test_reference_view.py`` runs the view suite on.
"""

from __future__ import annotations

import contextlib
import copy
import threading
from pathlib import Path
from typing import Any, Optional

from hermes_secret_port import SecretStore, SecretStoreProvider, write_failure

BOOTSTRAP = frozenset({"VIEW_STORE_URL", "VIEW_STORE_TOKEN"})
BOOTSTRAP_AUTH = frozenset({"nous"})


class MemoryViewProvider(SecretStoreProvider):
    files_backed = False

    def __init__(self) -> None:
        self.remote: dict[str, dict[str, Any]] = {}   # home key -> {"env": {}, "auth": {}, "version": int}
        self.cells = {}
        self.seen: dict[str, dict[str, Any]] = {}     # home key -> the snapshot the process reads
        self.booted: list[str] = []
        self.writes: list[dict[str, Any]] = []        # every write's names (never values)
        self.failing = False
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "memview"

    def open(self, home: Any = None) -> SecretStore:
        raise NotImplementedError("the reference view store serves the process-wide hooks only")

    def assert_isolating(self) -> None:
        return

    def bootstrap_env_names(self) -> frozenset[str]:
        return BOOTSTRAP

    def bootstrap_auth_records(self) -> frozenset[str]:
        return BOOTSTRAP_AUTH

    def deployment_env_names(self) -> frozenset[str]:
        return frozenset({"VIEW_STORE_URL"})

    @staticmethod
    def _key(home: Any) -> str:
        from hermes_constants import hermes_home_key
        return hermes_home_key(Path(home))

    def _remote(self, home: Any) -> dict[str, Any]:
        return self.remote.setdefault(self._key(home), {"env": {}, "auth": {}, "version": 1})

    def refresh(self, home: Any = None) -> None:
        """Re-read the remote snapshot and install it in each home's snapshot cell."""
        from hermes_secret_port import SecretSnapshot, SecretSnapshotCell
        keys = [self._key(home)] if home is not None else list(self.remote)
        with self._lock:
            for key in keys:
                view = copy.deepcopy(self.remote.get(key) or {"env": {}, "auth": {}, "version": 1})
                self.seen[key] = view
                cell = self.cells.setdefault(key, SecretSnapshotCell())
                cell.install(SecretSnapshot(cell.current.version + 1, view["env"], view["auth"]))

    def current_snapshot(self, home):
        self._seen(home)
        return self.cells[self._key(home)].current

    def _seen(self, home: Any) -> dict[str, Any]:
        key = self._key(home)
        if key not in self.seen:
            self.refresh(home)
        return self.seen[key]

    def boot(self, home: Any) -> None:
        self.booted.append(self._key(home))
        self.refresh(home)

    def env_view(self, home: Any) -> dict[str, str]:
        return dict(self._seen(home)["env"])

    def auth_view(self, home: Any) -> dict[str, dict[str, Any]]:
        return copy.deepcopy(self._seen(home)["auth"])

    def view_version(self, home: Any) -> str:
        return str(self._seen(home)["version"])

    def write_view(self, home: Any, *, env_set: Optional[dict[str, str]] = None,
                   env_unset: tuple[str, ...] = (), auth: Optional[dict[str, Any]] = None) -> None:
        names = sorted({*(env_set or {}), *env_unset})
        refused = [n for n in names if n in BOOTSTRAP] or [r for r in (auth or {}) if r in BOOTSTRAP_AUTH]
        failing = self.failing
        if refused or failing:
            env_set = auth = None
            if refused:
                raise PermissionError(f"bootstrap entries never leave the local files: {refused}")
            raise write_failure("write", ",".join(names))
        with self._lock:
            remote = self._remote(home)
            remote["env"].update(env_set or {})
            for name in env_unset:
                remote["env"].pop(name, None)
            for rid, record in (auth or {}).items():
                if record is None:
                    remote["auth"].pop(rid, None)
                else:
                    remote["auth"][rid] = copy.deepcopy(record)
            remote["version"] += 1
            self.writes.append({"env": names, "auth": sorted(auth or {})})
        env_set = auth = None
        self.refresh(home)

    def seed_external(self, home: Any, env: dict[str, str]) -> None:
        with self._lock:
            remote = self._remote(home)
            remote["env"].update(env)
            remote["version"] += 1

    @contextlib.contextmanager
    def failing_writes(self):
        self.failing = True
        try:
            yield
        finally:
            self.failing = False
