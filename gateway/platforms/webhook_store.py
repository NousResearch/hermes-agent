"""Profile-aware, crash-safe persistence for webhook route configurations."""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Mapping

from gateway.platforms.webhook_models import WebhookRouteConfig, from_legacy_route

_FILENAME = "webhook_subscriptions.json"
_LOCK_FILENAME = ".webhook_subscriptions.lock"
_MODE = 0o600
_PROFILE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    fcntl = None  # type: ignore[assignment]
try:
    import msvcrt
except ImportError:  # pragma: no cover - exercised on POSIX
    msvcrt = None  # type: ignore[assignment]


class WebhookRouteStore:
    """Persist routes for one profile under a shared Hermes root.

    ``root`` is the profile root for the default profile (maintaining the
    existing ``~/.hermes/webhook_subscriptions.json`` location), or the
    common Hermes root when a named profile is supplied. All read-modify-write
    operations take the same sidecar lock, so CLI, REST, desktop, and reload
    writers cannot lose updates.
    """

    def __init__(self, root: str | os.PathLike[str], profile: str = "default"):
        self.root = Path(root)
        if not _PROFILE_RE.fullmatch(profile):
            raise ValueError(f"invalid webhook profile name: {profile!r}")
        self.profile = profile

    @classmethod
    def for_hermes_home(
        cls, home: str | os.PathLike[str], profile: str = "default"
    ) -> "WebhookRouteStore":
        """Build a store from an active profile home or shared Hermes root."""
        home_path = Path(home)
        if profile == "default":
            return cls(home_path, profile)
        root = home_path
        if home_path.parent.name == "profiles":
            root = home_path.parent.parent
        return cls(root, profile)

    @property
    def profile_root(self) -> Path:
        # The default profile has historically lived directly in HERMES_HOME;
        # named profiles live beneath the shared profiles directory.
        return self.root if self.profile == "default" else self.root / "profiles" / self.profile

    @property
    def path(self) -> Path:
        return self.profile_root / _FILENAME

    @property
    def lock_path(self) -> Path:
        return self.profile_root / _LOCK_FILENAME

    @contextmanager
    def _lock(self):
        self.profile_root.mkdir(parents=True, exist_ok=True)
        try:
            with self.lock_path.open("xb") as lock_file:
                lock_file.write(b"0")
        except FileExistsError:
            pass
        try:
            os.chmod(self.lock_path, _MODE)
        except OSError:
            pass
        with self.lock_path.open("r+b") as handle:
            handle.seek(0)
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            elif msvcrt is not None:
                # msvcrt.locking is mandatory on native Windows. Lock one
                # stable byte in the sidecar, blocking until the writer exits.
                deadline = time.monotonic() + 30.0
                while True:
                    try:
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except (OSError, PermissionError):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("timed out acquiring webhook route lock")
                        time.sleep(0.01)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                elif msvcrt is not None:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

    def _load_unlocked(self) -> dict[str, WebhookRouteConfig]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("route file must contain an object")
            routes: dict[str, WebhookRouteConfig] = {}
            for name, value in raw.items():
                if not isinstance(value, Mapping):
                    raise ValueError(f"route {name!r} must contain an object")
                if "secret_ref" in value and "secret" not in value:
                    route = WebhookRouteConfig.model_validate({"name": name, **value})
                else:
                    route = from_legacy_route(name, value, profile=self.profile)
                routes[route.name] = route
            return dict(sorted(routes.items()))
        except Exception:
            # Never replace a malformed file with {}. Preserve the exact bytes
            # under a unique quarantine name, then leave callers a safe empty
            # view until an operator or a later save repairs the file.
            quarantine = self.path.with_name(
                f"{self.path.name}.corrupt-{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"
            )
            try:
                os.replace(self.path, quarantine)
            except OSError:
                # A competing repair may have quarantined it already.
                pass
            return {}

    def load(self) -> dict[str, WebhookRouteConfig]:
        with self._lock():
            return self._load_unlocked()

    def save(self, routes: Mapping[str, WebhookRouteConfig | Mapping]) -> None:
        normalized: dict[str, WebhookRouteConfig] = {}
        for name, value in routes.items():
            if isinstance(value, WebhookRouteConfig):
                route = value
            elif isinstance(value, Mapping):
                route = (
                    WebhookRouteConfig.model_validate({"name": name, **value})
                    if "secret_ref" in value and "secret" not in value
                    else from_legacy_route(name, value, profile=self.profile)
                )
            else:
                raise TypeError(f"route {name!r} must be a WebhookRouteConfig or mapping")
            normalized[route.name] = route
        with self._lock():
            self._save_unlocked(dict(sorted(normalized.items())))

    def _save_unlocked(self, routes: Mapping[str, WebhookRouteConfig]) -> None:
        self.profile_root.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.profile_root
        )
        tmp = Path(tmp_name)
        try:
            payload = {
                name: route.model_dump(mode="json")
                for name, route in sorted(routes.items())
            }
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(tmp, _MODE)
            os.replace(tmp, self.path)
            os.chmod(self.path, _MODE)
            # Ensure the directory entry for the rename is durable where the
            # platform permits opening directories for fsync.
            try:
                dir_fd = os.open(self.profile_root, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                pass
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def update(
        self,
        mutator: Callable[[dict[str, WebhookRouteConfig]], Mapping[str, WebhookRouteConfig | Mapping]],
    ) -> dict[str, WebhookRouteConfig]:
        """Atomically apply a read-modify-write mutation under one lock."""
        with self._lock():
            current = self._load_unlocked()
            updated = dict(mutator(dict(current)))
            normalized: dict[str, WebhookRouteConfig] = {}
            for name, value in updated.items():
                normalized[name] = value if isinstance(value, WebhookRouteConfig) else from_legacy_route(name, value, profile=self.profile)
            normalized = dict(sorted(normalized.items()))
            self._save_unlocked(normalized)
            return normalized


__all__ = ["WebhookRouteStore"]
