"""The remote secret store: every profile secret lives on the connector's secrets plane.

Selected by ``HERMES_SECRET_STORE=remote``. Values apply at process start; a change restarts the
process. Per profile home the store keeps the plane's resolved view (``GET /v1/secrets/self``) in
memory only:

- **boot** fetches it once, before the first secret read, and refuses to start when the plane is
  unreachable or refuses (bounded retry for an outage, none for a refusal; no on-disk cache, no
  fallback to ``.env``). The view is then the profile ``.env`` layer: the dotenv loader and every
  scope builder read it in that file's place, the stored strings verbatim;
- **one-time import**: local ``.env`` / ``auth.json`` entries outside the bootstrap set are sent once
  with ``expectedProfileVersion: 0`` and the files are rewritten to the bootstrap set; when the plane
  already has the profile (409) it wins and the local entries move to ``.env.unimported`` /
  ``auth.unimported.json`` (``importer.py``);
- **change detection**: a per-process poller asks for every booted view with ``If-None-Match`` every
  :data:`POLL_SECONDS`, holding the version the process booted with or last wrote itself. Nothing is
  installed from a poll: a different version asks the process for a graceful restart
  (``hermes_secret_routing.request_restart``), so edits made elsewhere, revocations included, apply
  when it starts again. A failed poll keeps the process running;
- **writes** (``save_env_value``, ``remove_env_value``, the auth store's save) are one CAS
  ``PATCH /v1/secrets/self`` each. The reply's view is installed (the caller applies the write
  in-process exactly as a local save does) and its version becomes the process's own. A
  ``secret_version_conflict``, or a reply carrying a change this write did not make, restarts the
  process instead of merging into a view it did not boot with.

The bootstrap set (:meth:`RemoteSecretStoreProvider.bootstrap_env_names`, the ``nous`` auth record
the plane credential comes from) never leaves the local files. No value appears in any log line or
error this module builds: they name the route, the HTTP status, the plane's error code and, for a
change, the entry names.
"""
from __future__ import annotations

import contextlib
import logging
import os
import random
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional

from hermes_constants import hermes_home_key
from hermes_secret_port import (
    SECRET_STORE_ENV, SecretSnapshot, SecretSnapshotCell, SecretStore, SecretStoreProvider,
    SecretWriteFailed, thaw_secret,
)
from plugins.config_backends.remote import client as config_client
from plugins.config_backends.remote.credentials import IDP_ENV_PREFIX, PLANE_CREDENTIAL_ENV_NAMES, PlaneCredentialError

from . import client

logger = logging.getLogger(__name__)

POLL_SECONDS = 60.0
BOOT_RETRY_DELAYS = (10.0, 20.0)   # attempts at t = 0, ~10 s, ~30 s, as Remote Config
MAX_RETRY_WAIT = 20.0
#: The ``auth.json`` records the plane credential comes from: they never leave the local file.
BOOTSTRAP_AUTH_RECORDS = frozenset({"nous", "pool:nous"})
#: Names listed in a restart's log line before it says "and N more".
_NAMES_LOGGED = 20


class _FetchFailed(Exception):
    """A plane call that did not produce a usable view. ``detail`` names the route, status and
    error code only (never a value, never the plane's free-text message)."""

    def __init__(self, detail: str, *, retryable: bool, status: Optional[int] = None) -> None:
        super().__init__(detail)
        self.retryable = retryable
        self.status = status


@dataclass(frozen=True)
class View:
    """One resolved view of a profile's secrets, as ``GET /v1/secrets/self`` returns it."""

    env: dict[str, str]
    env_meta: dict[str, tuple[str, int]]        # name -> (level, entry version)
    auth: dict[str, dict[str, Any]]
    auth_meta: dict[str, tuple[str, int]]       # record id -> (level, entry version)
    etag: str
    profile_version: int
    version: int = 0

    def __post_init__(self):
        snapshot = SecretSnapshot(self.version, self.env, self.auth)
        object.__setattr__(self, "env", snapshot.env)
        object.__setattr__(self, "auth", snapshot.auth)
        object.__setattr__(self, "env_meta", MappingProxyType(dict(self.env_meta)))
        object.__setattr__(self, "auth_meta", MappingProxyType(dict(self.auth_meta)))

    def profile_entry_version(self, rid: str) -> int:
        """The profile-level version of auth record *rid* (0: the profile level has none)."""
        level, version = self.auth_meta.get(rid, ("", 0))
        return version if level == "profile" else 0


def parse_view(body: Any, etag: Optional[str]) -> View:
    """The view a ``/self`` body carries; ``ValueError`` (with no value in it) for a malformed one."""
    if not isinstance(body, dict) or not isinstance(body.get("env"), dict) or not isinstance(body.get("auth"), dict):
        raise ValueError("the secrets plane returned a malformed view")
    env: dict[str, str] = {}
    env_meta: dict[str, tuple[str, int]] = {}
    for name, entry in body["env"].items():
        if not isinstance(entry, dict) or not isinstance(entry.get("value"), str):
            raise ValueError("the secrets plane returned a malformed env entry")
        env[str(name)] = entry["value"]
        env_meta[str(name)] = (str(entry.get("level") or ""), int(entry.get("version") or 0))
    auth: dict[str, dict[str, Any]] = {}
    auth_meta: dict[str, tuple[str, int]] = {}
    for rid, entry in body["auth"].items():
        if not isinstance(entry, dict) or not isinstance(entry.get("record"), dict):
            raise ValueError("the secrets plane returned a malformed auth entry")
        auth[str(rid)] = entry["record"]
        auth_meta[str(rid)] = (str(entry.get("level") or ""), int(entry.get("version") or 0))
    version = etag or str(body.get("version") or "")
    profile_version = body.get("profileVersion")
    if not version or not isinstance(profile_version, int):
        raise ValueError("the secrets plane returned a view without its version")
    return View(env, env_meta, auth, auth_meta, version, profile_version)


def changed_names(old: View, new: View) -> list[str]:
    """The env names and auth record ids (``auth:<id>``) whose entry differs between two views
    (value, level or entry version), sorted: what a restart's log line names. Never a value."""
    names = {n for n in {*old.env, *new.env}
             if (old.env.get(n), old.env_meta.get(n)) != (new.env.get(n), new.env_meta.get(n))}
    names |= {f"auth:{r}" for r in {*old.auth, *new.auth}
              if (old.auth.get(r), old.auth_meta.get(r)) != (new.auth.get(r), new.auth_meta.get(r))}
    return sorted(names)


def _named(names: list[str]) -> str:
    shown = ", ".join(names[:_NAMES_LOGGED])
    return shown + (f" and {len(names) - _NAMES_LOGGED} more" if len(names) > _NAMES_LOGGED else "")


@dataclass
class _HomeState:
    home: Path
    profile: str
    view: View          # the view the process booted with, plus its own writes
    cell: SecretSnapshotCell = field(default_factory=SecretSnapshotCell)
    next_poll: float = 0.0
    fetched_at: float = 0.0
    last_error: Optional[str] = None

    def __post_init__(self):
        self.cell.install(replace(self.view, version=1))
        self.view = self.cell.current

    def install_own(self, view: View) -> None:
        """Install the view one of the process's own writes returned (its version is now ours)."""
        if self.cell.install(replace(view, version=self.cell.current.version + 1)):
            self.view = self.cell.current


def _failure(resp: client.Response) -> _FetchFailed:
    retryable = resp.status >= 500 or resp.status == 429
    return _FetchFailed(f"{client.ROUTE} answered HTTP {resp.status} {resp.error}", retryable=retryable,
                        status=resp.status)


class RemoteSecretStoreProvider(SecretStoreProvider):
    """Every profile's secrets on the secrets plane, one in-memory view per profile home."""

    files_backed = False
    literal_values = True
    levels = frozenset({"agent"})

    def __init__(self) -> None:
        self._states: dict[str, _HomeState] = {}
        self._home_locks: dict[str, threading.Lock] = {}   # home key -> its plane-call lock (``_serialized``)
        self._boot_events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._local = threading.local()   # .booting / .held: home keys this thread is loading / holds
        self._poller: Optional[threading.Thread] = None
        self._poller_pid: Optional[int] = None
        self._stop = threading.Event()
        self._restart_reason: Optional[str] = None

    @property
    def name(self) -> str:
        return "remote"

    # --- declarations -----------------------------------------------------------------------

    def bootstrap_env_names(self) -> frozenset[str]:
        return PLANE_CREDENTIAL_ENV_NAMES | {SECRET_STORE_ENV, client.URL_ENV}

    def keeps_local(self, name: str) -> bool:
        return name in self.bootstrap_env_names() or name.startswith(IDP_ENV_PREFIX)

    def bootstrap_auth_records(self) -> frozenset[str]:
        return BOOTSTRAP_AUTH_RECORDS

    def deployment_env_names(self) -> frozenset[str]:
        return frozenset({SECRET_STORE_ENV, client.URL_ENV, config_client.URL_ENV, config_client.INSTANCE_ENV})

    def supports_file_tooling(self) -> bool:
        return False

    def open(self, home: Any = None) -> SecretStore:
        """The port's store for *home*: the local backend's scoped store, whose reads and writes
        go through ``hermes_secret_routing`` and so reach this store's view (pin, administration
        and the no-value write boundary are the port's own)."""
        from hermes_secret_port_local import LocalSecretStore
        if home is None:
            from hermes_constants import get_hermes_home
            home = get_hermes_home()
        return LocalSecretStore(home)

    def assert_isolating(self) -> None:
        """This client stores nothing at rest: the plane encrypts every entry and serves only the
        agent its verified credential names. A hosted scope is refused by the store's pin."""
        return

    # --- homes ------------------------------------------------------------------------------

    @staticmethod
    def _key(home: Path) -> str:
        return hermes_home_key(Path(home))

    @staticmethod
    def _profile_for(home: Path) -> str:
        from hermes_constants import PROFILE_ID_RE, profile_name_for_home
        from hermes_secret_routing import SecretStoreUnavailable
        name = profile_name_for_home(home)
        if not name or not PROFILE_ID_RE.match(name):
            raise SecretStoreUnavailable(
                f"Remote secrets: {home} is not a Hermes profile home, so the secrets plane has no "
                "view for it (HERMES_SECRET_STORE=remote serves profile homes only).")
        return name

    def _booting(self) -> set[str]:
        keys = getattr(self._local, "booting", None)
        if keys is None:
            keys = self._local.booting = set()
        return keys

    @contextlib.contextmanager
    def _serialized(self, home: Path):
        """Hold *home*'s plane-call lock: a poll and a write never interleave, so a poll can never
        read the plane after a write committed but before the process recorded that write's
        version as its own. Network I/O and in-memory work only under it; resolve the bearer
        before entering (it may save ``auth.json``); a same-thread re-entry raises instead of
        waiting."""
        key = self._key(home)
        held = getattr(self._local, "held", None)
        if held is None:
            held = self._local.held = set()
        if key in held:
            raise RuntimeError("the remote secret store re-entered its own home lock")
        with self._lock:
            lock = self._home_locks.setdefault(key, threading.Lock())
        with lock:
            held.add(key)
            try:
                yield
            finally:
                held.discard(key)

    def _state(self, home: Path) -> Optional[_HomeState]:
        """*home*'s state, fetching it (fail closed) on first use; None while this thread is
        loading that very home (resolving its own credential reads only bootstrap entries)."""
        key = self._key(home)
        st = self._states.get(key)
        if st is not None:
            return st
        if key in self._booting():
            return None
        # Boot includes file cleanup and imports, so serialize its lifecycle with an event,
        # never a lock held across application calls. Other callers wait without holding locks.
        while True:
            with self._lock:
                st = self._states.get(key)
                event = self._boot_events.get(key)
                if st is None and event is None:
                    event = self._boot_events[key] = threading.Event()
                    break
            if st is not None:
                return st
            event.wait()
        booting = self._booting()
        booting.add(key)
        try:
            st = self._boot_fetch(Path(home), key)
        finally:
            booting.discard(key)
            with self._lock:
                self._boot_events.pop(key)
                event.set()
        self._ensure_poller()
        return st

    def _boot_fetch(self, home: Path, key: str) -> _HomeState:
        from hermes_secret_routing import SecretStoreUnavailable

        from . import importer
        profile = self._profile_for(home)
        if not client.instance_id():
            raise SecretStoreUnavailable(
                f"Remote secrets: {config_client.INSTANCE_ENV} is not set, so Hermes cannot identify this "
                "agent to the secrets plane. Hermes does not start without its secrets.")
        # The local entries to import are staged before the credential resolves: a token refresh
        # saves auth.json, and that save keeps only the bootstrap records.
        importer.stage(home, self)
        entries = importer.prepared_entries(home)
        delays = list(BOOT_RETRY_DELAYS)
        while True:
            failed: Optional[tuple[str, bool]] = None
            try:
                token = self._bearer(home)  # resolved holding no lock of this store
                with self._serialized(home):
                    st, imported = self._first_fetch(home, profile, token, entries)
                importer.finish(home, profile, st.view, self, entries, imported)
                with self._lock:
                    self._states[key] = st
            except _FetchFailed as exc:
                failed = (str(exc), exc.retryable)   # the failure's frames held values: dropped here
            if failed is None:
                return st
            detail, retryable = failed
            if retryable and delays:
                wait = min(delays.pop(0), MAX_RETRY_WAIT)
                logger.warning("Remote secrets: fetch for profile %r failed (%s); retrying in %.0fs",
                               profile, detail, wait)
                time.sleep(wait)
                continue
            raise SecretStoreUnavailable(
                f"Remote secrets: cannot load the secrets of profile {profile!r} from "
                f"{client.shown_url()}: {detail}. Hermes does not start without its secrets "
                "(HERMES_SECRET_STORE=remote has no local fallback).")

    def _first_fetch(self, home: Path, profile: str, token: str, entries) -> tuple[_HomeState, bool]:
        """Only network I/O and in-memory work under the home's plane-call lock."""
        view = self._get(home, profile, token=token)
        env, records = entries
        imported = False
        if (env or records) and view.profile_version == 0:
            body = {"expectedProfileVersion": 0, "set": env,
                    "auth": {rid: {"expectedVersion": 0, "record": r} for rid, r in records.items()}}
            try:
                resp = client.request("PATCH", home, profile, body=body, token=token)
            except client.TransportError:
                raise _FetchFailed(f"{client.ROUTE} unreachable during the import", retryable=True) from None
            finally:
                body = None
            if resp.status == 200:
                view = parse_view(resp.body, resp.etag)
                imported = True
            elif resp.status == 409 and resp.error == "secret_version_conflict":
                fresh = self._get(home, profile, token=token)
                view = fresh if fresh is not None else view
            else:
                raise _failure(resp)
        return (_HomeState(home=home, profile=profile, view=view, fetched_at=time.time(),
                           next_poll=time.monotonic() + _jittered()), imported)

    @staticmethod
    def _bearer(home: Path) -> str:
        try:
            return client.bearer(home)
        except PlaneCredentialError as exc:
            raise _FetchFailed(f"no plane credential: {exc}", retryable=exc.retryable) from None

    def _get(self, home: Path, profile: str, *, token: Optional[str] = None,
             etag: Optional[str] = None) -> Optional[View]:
        """``GET /self``: the view, or None for a 304. Raises :class:`_FetchFailed`."""
        try:
            resp = client.request("GET", home, profile, etag=etag, token=token)
        except PlaneCredentialError as exc:
            raise _FetchFailed(f"no plane credential: {exc}", retryable=exc.retryable) from None
        except client.TransportError as exc:
            raise _FetchFailed(f"{client.ROUTE} unreachable: {exc}", retryable=True) from None
        if resp.status == 304 and etag:
            return None
        if resp.status != 200:
            raise _failure(resp)
        try:
            return parse_view(resp.body, resp.etag)
        except ValueError as exc:
            raise _FetchFailed(str(exc), retryable=True) from None

    # --- the process-wide hooks -------------------------------------------------------------

    def boot(self, home: Any) -> None:
        self._state(Path(home))
        self._ensure_poller()   # a forked child re-arms its own poller

    def env_view(self, home: Any) -> dict[str, str]:
        st = self._state(Path(home))
        return {} if st is None else dict(st.view.env)

    def current_snapshot(self, home: Any):
        st = self._state(Path(home))
        return SecretSnapshot(0, {}, {}) if st is None else st.cell.current

    def auth_view(self, home: Any) -> dict[str, dict[str, Any]]:
        return thaw_secret(self.current_snapshot(home).auth)

    def view_version(self, home: Any) -> str:
        st = self._state(Path(home))
        return "" if st is None else st.view.etag

    def write_view(self, home: Any, *, env_set: Optional[dict[str, str]] = None,
                   env_unset: tuple[str, ...] = (), auth: Optional[dict[str, Any]] = None) -> None:
        from hermes_secret_routing import stays_local
        names = sorted({*(env_set or {}), *env_unset})
        rids = sorted(auth or {})
        refused = [n for n in names if stays_local(self, n)] + [r for r in rids if r in BOOTSTRAP_AUTH_RECORDS]
        code = None if refused else self._write(Path(home), env_set, env_unset, auth)
        # Nothing below raises with the value still bound in this frame (error reporters render
        # frame locals): the writer kept it in its own frame, which is gone.
        env_set = auth = None
        if refused:
            raise ValueError(f"the bootstrap entries never leave the local files: {', '.join(refused)}")
        if code is not None:
            raise SecretWriteFailed(f"the credential write failed: PATCH {client.ROUTE} ({code})")

    def _write(self, home: Path, env_set: Optional[dict[str, str]], env_unset: tuple[str, ...],
               auth: Optional[dict[str, Any]]) -> Optional[str]:
        """One CAS write; None on success, else a failure code. Never raises: every failure is
        caught here, in the only frame that holds the value."""
        try:
            st = self._state(home)
            if st is None:
                return "store_booting"
            try:
                token = self._bearer(home)  # resolved holding no lock of this store
            except _FetchFailed:
                return "credential_unavailable"
            with self._serialized(home):
                code, foreign = self._write_locked(st, env_set, env_unset, auth, token)
        except SystemExit:
            return "store_unavailable"
        except Exception:  # health: allow BLE001 -- a writer failure is never inspected, only reported fixed
            return "write_failed"
        if foreign is not None:
            self._restart(f"profile {st.profile!r}: {foreign}")
        return code

    def _write_locked(self, st: _HomeState, env_set: Optional[dict[str, str]], env_unset: tuple[str, ...],
                      auth: Optional[dict[str, Any]], token: str) -> tuple[Optional[str], Optional[str]]:
        """:meth:`_write`'s PATCH under the home's plane-call lock: ``(failure code or None, the
        restart reason when the plane holds a change this process did not make, or None)``."""
        body = _patch_body(st.view, env_set, env_unset, auth)
        try:
            resp = client.request("PATCH", st.home, st.profile, body=body, token=token)
        except (TypeError, ValueError):
            return "encode_failed", None
        except client.TransportError:
            return "plane_unreachable", None
        finally:
            body = None
        if resp.status == 409 and resp.error == "secret_version_conflict":
            return "secret_version_conflict", "a write conflicted with a change made elsewhere"
        if resp.status != 200:
            return f"HTTP {resp.status} {resp.error}", None
        before, reply = st.view, parse_view(resp.body, resp.etag)
        touched = {*(env_set or {}), *env_unset, *(f"auth:{r}" for r in auth or {})}
        st.install_own(_own_change(before, reply, touched))
        foreign = [n for n in changed_names(before, reply) if n not in touched]
        return None, (f"changed elsewhere: {_named(foreign)}" if foreign else None)

    # --- change detection -------------------------------------------------------------------

    def _restart(self, reason: str) -> None:
        """Ask for the process's graceful restart, once; the poller stops asking the plane."""
        from hermes_secret_routing import request_restart
        with self._lock:
            if self._restart_reason is not None:
                return
            self._restart_reason = reason
        request_restart(f"remote secrets, {reason}")

    def poll_one(self, st: _HomeState) -> bool:
        """One conditional GET against the version the process booted with or last wrote itself.
        True when the plane's view differs (a restart was asked for). Never raises: a failed poll
        keeps the process running on the view it has."""
        changed: Optional[list[str]] = None
        try:
            token = self._bearer(st.home)  # resolved holding no lock of this store
            with self._serialized(st.home):
                view = self._get(st.home, st.profile, etag=st.view.etag, token=token)
                if view is not None and view.etag != st.view.etag:
                    changed = changed_names(st.view, view)
            st.last_error = None
        except _FetchFailed as exc:
            st.last_error = str(exc)
            level = logging.ERROR if exc.status in (401, 403) else logging.WARNING
            logger.log(level, "Remote secrets: poll for profile %r failed (%s); keeping the loaded secrets",
                       st.profile, exc)
        except Exception:  # health: allow BLE001 -- a poll never stops the poller; its failure names no value
            st.last_error = "poll failed"
            logger.warning("Remote secrets: poll for profile %r failed; keeping the loaded secrets", st.profile)
        st.next_poll = time.monotonic() + _jittered()
        if changed is None:
            return False
        self._restart(f"profile {st.profile!r} changed elsewhere: {_named(changed) or 'view version'}")
        return True

    def poll_all(self) -> None:
        for st in self._roster():
            if self._restart_reason is not None:
                return
            self.poll_one(st)

    def _roster(self) -> list[_HomeState]:
        with self._lock:
            return list(self._states.values())

    def _ensure_poller(self) -> None:
        with self._lock:
            if self._poller is not None and self._poller.is_alive() and self._poller_pid == os.getpid():
                return
            self._poller_pid = os.getpid()  # a forked child re-arms its own thread
            self._poller = threading.Thread(target=self._poll_loop, name="remote-secrets-poll", daemon=True)
            self._poller.start()

    def _poll_loop(self) -> None:
        while not self._stop.is_set() and self._restart_reason is None:
            now = time.monotonic()
            for st in self._roster():
                if st.next_poll <= now and self._restart_reason is None:
                    self.poll_one(st)
            wake = min((st.next_poll for st in self._roster()), default=now + POLL_SECONDS)
            self._stop.wait(max(1.0, wake - time.monotonic()))

    # --- diagnostics ------------------------------------------------------------------------

    def doctor_warnings(self, home: Any) -> tuple:
        from . import importer
        left = [p for p in importer.unimported_paths(Path(home)) if p.exists()]
        if not left:
            return ()
        listed = ", ".join(str(p) for p in left)
        return ((f"Local secrets were not imported (the secrets plane already had this profile): {listed}",
                 "(review them, set what is still needed through Hermes or the editor, then delete the files)",
                 f"Review and delete {listed}: the secrets plane wins over these local entries"),)

    def describe(self, home: Any) -> str:
        st = self._states.get(self._key(Path(home)))
        head = f"Remote secrets {client.shown_url()} (instance {client.instance_id() or '<unset>'})"
        if st is None:
            return f"{head}: profile not fetched yet"
        age = int(time.time() - st.fetched_at) if st.fetched_at else -1
        tail = f", last poll failed: {st.last_error}" if st.last_error else ""
        if self._restart_reason is not None:
            tail += ", restart pending (changed elsewhere)"
        return (f"{head}: profile {st.profile!r}, profile level v{st.view.profile_version}, "
                f"{len(st.view.env)} env name(s), {len(st.view.auth)} auth record(s), loaded {age}s ago{tail}")


def _own_change(before: View, reply: View, touched: set[str]) -> View:
    """*before* with only the *touched* entries (env names, ``auth:<id>``) taken from a write's
    *reply*, and the reply's versions: what the process applies of its own write. Anything else
    the reply carries was changed elsewhere and waits for the restart."""
    def pick(old, new, old_meta, new_meta, key):
        out = {n: (old[n], old_meta[n]) for n in old if key(n) not in touched}
        out.update((n, (new[n], new_meta[n])) for n in new if key(n) in touched)
        return {n: v for n, (v, _m) in out.items()}, {n: m for n, (_v, m) in out.items()}
    env, env_meta = pick(before.env, reply.env, before.env_meta, reply.env_meta, lambda n: n)
    auth, auth_meta = pick(before.auth, reply.auth, before.auth_meta, reply.auth_meta, lambda r: f"auth:{r}")
    return View(env, env_meta, thaw_secret(auth), auth_meta, reply.etag, reply.profile_version)


def _jittered() -> float:
    return POLL_SECONDS * random.uniform(0.9, 1.1)


def _patch_body(view: View, env_set: Optional[dict[str, str]], env_unset: tuple[str, ...],
                auth: Optional[dict[str, Any]]) -> dict[str, Any]:
    """The ``PATCH /self`` body for one change against *view*'s versions: the profile level's
    version when env names change, and each auth record's own profile-level version."""
    body: dict[str, Any] = {}
    if env_set or env_unset:
        body["expectedProfileVersion"] = view.profile_version
        if env_set:
            body["set"] = dict(env_set)
        if env_unset:
            body["unset"] = list(env_unset)
    if auth:
        body["auth"] = {rid: {"expectedVersion": view.profile_entry_version(rid), "record": record}
                        for rid, record in auth.items()}
    return body
