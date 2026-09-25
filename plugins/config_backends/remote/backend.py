"""The remote config backend (config-config design §4, contract.md §11, §15).

The user layer of each profile home is the effective document the config plane resolves for
(this instance, that profile). It lives in memory only: fetched at boot (fail closed, D2), kept
fresh by a per-process ETag poller (D9), and written back as key-level changes on the profile
level with CAS (D10). Nothing is read from or written to a local ``config.yaml``.
"""
from __future__ import annotations

import copy
import logging
import os
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_cli.config_backend import (
    Changes, ConfigBackendUnavailable, ConfigLockedError, ConfigValueError, ConfigWriteError, UserLayer)

from . import client
from .credentials import PLANE_CREDENTIAL_ENV_NAMES, PlaneCredentialError
from .diff import diff, encode_changes, strip_locked, write_check
from .paths import Path as KeyPath
from .paths import PathError, decode, encode, from_dotted
from .values import secret_literal_path, to_wire

logger = logging.getLogger(__name__)

POLL_ENV = "HERMES_CONFIG_REMOTE_POLL_SECONDS"
DEFAULT_POLL_SECONDS = 300.0
MIN_POLL_SECONDS = 30.0            # contract §11.3: lower values are clamped
BOOT_RETRY_DELAYS = (10.0, 20.0)   # contract §11.4: attempts at t = 0, ~10 s, ~30 s
MAX_RETRY_WAIT = 20.0


def poll_interval() -> float:
    try:
        value = float(os.environ.get(POLL_ENV, "") or DEFAULT_POLL_SECONDS)
    except ValueError:
        value = DEFAULT_POLL_SECONDS
    return max(value, MIN_POLL_SECONDS)


def _latest_config_version() -> int:
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    version = DEFAULT_CONFIG.get("_config_version")
    return version if isinstance(version, int) else 1


class _FetchFailed(Exception):
    def __init__(self, detail: str, *, retryable: bool, retry_after: Optional[float] = None,
                 status: Optional[int] = None) -> None:
        super().__init__(detail)
        self.retryable = retryable
        self.retry_after = retry_after
        self.status = status


@dataclass
class _ProfileState:
    home: Path
    profile: str
    server_config: Dict[str, Any] = field(default_factory=dict)  # EffectiveResponse.config, as sent
    doc: Dict[str, Any] = field(default_factory=dict)            # what readers get (+ _config_version, migrated)
    etag: str = ""
    profile_version: int = 0
    locks: List[Tuple[KeyPath, str]] = field(default_factory=list)
    provenance: Dict[str, str] = field(default_factory=dict)
    changed_ns: int = 0
    gen: int = 0
    postprocessed: bool = False
    in_postprocess: bool = False
    fetched_at: float = 0.0
    last_error: Optional[str] = None
    next_poll: float = 0.0
    lock: threading.RLock = field(default_factory=threading.RLock)


def _effective_ok(body: Dict[str, Any]) -> bool:
    return (isinstance(body.get("config"), dict) and isinstance(body.get("profileVersion"), int)
            and isinstance(body.get("locks"), list))


class RemoteBackend:
    name = "remote"

    def __init__(self) -> None:
        self._states: Dict[str, _ProfileState] = {}
        self._fetch_locks: Dict[str, threading.Lock] = {}
        self._lock = threading.Lock()
        self._poller: Optional[threading.Thread] = None
        self._poller_pid: Optional[int] = None
        self._stop = threading.Event()
        self._managed_warned = False
        self._unknown_warned: set = set()

    # --- homes and state --------------------------------------------------------------------

    @staticmethod
    def _key(home: Path) -> str:
        from hermes_constants import hermes_home_key
        return hermes_home_key(home)

    @staticmethod
    def _profile_for(home: Path) -> str:
        from hermes_constants import PROFILE_ID_RE, profile_name_for_home
        name = profile_name_for_home(home)
        if not name or not PROFILE_ID_RE.match(name):
            raise ConfigBackendUnavailable(
                f"Remote Config: {home} is not a Hermes profile home, so it has no remote config "
                "(HERMES_CONFIG_BACKEND=remote serves profile homes only).")
        return name

    def _state(self, home: Path) -> _ProfileState:
        key = self._key(home)
        st = self._states.get(key)
        if st is not None:
            if self._poller_pid != os.getpid():  # a forked child has no poller thread
                self._ensure_poller()
            return st
        with self._lock:
            fetch_lock = self._fetch_locks.setdefault(key, threading.Lock())
        with fetch_lock:
            st = self._states.get(key)
            if st is None:
                st = self._initial_fetch(Path(home))
                self._states[key] = st
        self._ensure_poller()
        return st

    def _initial_fetch(self, home: Path) -> _ProfileState:
        """Boot fetch with bounded retry (contract §11.4); fails closed (D2)."""
        profile = self._profile_for(home)
        if not client.instance_id():
            raise ConfigBackendUnavailable(
                f"Remote Config: {client.INSTANCE_ENV} is not set, so Hermes cannot identify this agent "
                "to the config plane. Hermes does not start without its remote config.")
        st = _ProfileState(home=home, profile=profile)
        delays = list(BOOT_RETRY_DELAYS)
        while True:
            try:
                self._fetch_into(st, conditional=False)
                return st
            except _FetchFailed as exc:
                if exc.retryable and delays:
                    wait = min(max(exc.retry_after or 0.0, delays.pop(0)), MAX_RETRY_WAIT)
                    logger.warning("Remote Config: fetch for profile %r failed (%s); retrying in %.0fs",
                                   profile, exc, wait)
                    time.sleep(wait)
                    continue
                raise ConfigBackendUnavailable(
                    f"Remote Config: cannot load the config for profile {profile!r} from "
                    f"{client.base_url()}: {exc}. Hermes does not start without its remote config "
                    "(HERMES_CONFIG_BACKEND=remote has no local fallback).") from exc

    def _fetch_into(self, st: _ProfileState, *, conditional: bool) -> bool:
        """GET ``/self`` into ``st``; True when the document changed. Raises :class:`_FetchFailed`."""
        try:
            resp = client.request("GET", st.home, st.profile, etag=st.etag if conditional else None)
        except PlaneCredentialError as exc:
            raise _FetchFailed(str(exc), retryable=exc.retryable) from exc
        except client.TransportError as exc:
            raise _FetchFailed(str(exc), retryable=True) from exc
        if resp.status == 304 and conditional:
            st.fetched_at, st.last_error = time.time(), None
            return False
        if resp.status == 200 and _effective_ok(resp.body):
            with st.lock:
                self._install(st, resp.body, resp.etag)
            return True
        if resp.status == 200:
            raise _FetchFailed("the plane returned a malformed response", retryable=True)
        retryable = resp.status >= 500 or resp.status == 429
        raise _FetchFailed(f"HTTP {resp.status} {resp.error}: {resp.message}", retryable=retryable,
                           retry_after=resp.retry_after, status=resp.status)

    def _install(self, st: _ProfileState, body: Dict[str, Any], etag: Optional[str]) -> None:
        config = body["config"]
        doc = copy.deepcopy(config)
        doc.pop("_config_version", None)  # reserved on the wire (§3.6); never expected here
        # R6: the in-memory schema version is the oldest writer's; migrations run in memory (D12).
        writer_versions: List[int] = [
            lv["writerConfigVersion"] for lv in body.get("levels") or []
            if isinstance(lv, dict) and isinstance(lv.get("writerConfigVersion"), int)]
        doc["_config_version"] = min(writer_versions) if writer_versions else _latest_config_version()
        locks: List[Tuple[KeyPath, str]] = []
        for entry in body.get("locks") or []:
            try:
                locks.append((decode(entry["path"]), str(entry["level"])))
            except (KeyError, TypeError, PathError):
                logger.warning("Remote Config: ignoring malformed lock entry %r", entry)
        st.server_config = config
        st.doc = doc
        st.etag = etag or str(body.get("etag") or "")
        st.profile_version = int(body["profileVersion"])
        st.locks = locks
        st.provenance = dict(body.get("provenance") or {})
        st.changed_ns = time.time_ns()
        st.gen += 1
        st.postprocessed = False
        st.fetched_at, st.last_error = time.time(), None
        st.next_poll = time.monotonic() + poll_interval() * random.uniform(0.9, 1.1)

    def _postprocess(self, st: _ProfileState) -> None:
        """First read after a change: warn on unknown keys and migrate in memory (D12). Runs on a
        read, not at fetch, because both need ``hermes_cli.config``, which may still be importing
        when the boot fetch runs."""
        if st.postprocessed or st.in_postprocess:
            return
        # The flag, not st.lock, guards the work: a migration reads through hermes_cli.config (its
        # _CONFIG_LOCK), and holding st.lock across that could deadlock against a writer that holds
        # _CONFIG_LOCK and waits for st.lock. A concurrent reader meanwhile gets the unmigrated doc.
        with st.lock:
            if st.postprocessed or st.in_postprocess:
                return
            st.in_postprocess = True
        etag = st.etag
        try:
            try:
                from hermes_cli.config import _known_top_level_keys
                from hermes_cli.config_migrations import SUPPORT_FLOOR_VERSION, run_migrations
            except ImportError:
                return  # still importing; the next read retries
            for key in sorted(set(st.doc) - _known_top_level_keys() - {"_config_version"}):
                if key not in self._unknown_warned:
                    self._unknown_warned.add(key)
                    logger.warning("Remote Config: unknown config key %r is ignored by this Hermes version", key)
            current, latest = int(st.doc.get("_config_version") or 0), _latest_config_version()
            if SUPPORT_FLOOR_VERSION <= current < latest:
                self._migrate_in_memory(st, current, run_migrations)
            elif current < SUPPORT_FLOOR_VERSION:
                logger.warning("Remote Config: profile %r was written by config version %d, below the "
                               "migration floor %d; not migrated", st.profile, current, SUPPORT_FLOOR_VERSION)
            st.postprocessed = st.etag == etag  # a poll that landed meanwhile needs its own pass
        finally:
            st.in_postprocess = False

    def _migrate_in_memory(self, st: _ProfileState, current: int, run_migrations) -> None:
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        token = set_hermes_home_override(st.home)
        try:
            run_migrations(current, {"env_added": [], "config_added": [], "warnings": []}, True)
        finally:
            reset_hermes_home_override(token)
        doc = copy.deepcopy(st.doc)
        doc["_config_version"] = _latest_config_version()
        self.apply_in_memory(st.home, doc)
        logger.info("Remote Config: migrated profile %r in memory from config version %d", st.profile, current)

    # --- ConfigBackend: reads ---------------------------------------------------------------

    def read_user_layer(self, home: Path) -> UserLayer:
        st = self._state(home)
        self._postprocess(st)
        return UserLayer(doc=copy.deepcopy(st.doc), version=self._version_of(st),
                         locks={encode(p): level for p, level in st.locks},
                         provenance=f"remote:{client.base_url()} profile={st.profile}")

    def read_user_doc_readonly(self, home: Path) -> Any:
        st = self._state(home)
        self._postprocess(st)
        return st.doc

    @staticmethod
    def _version_of(st: _ProfileState) -> Tuple[Any, ...]:
        # Leads with a nanosecond timestamp like the file backend's st_mtime_ns (the TUI's config
        # watcher reads element 0 as a time); the etag and local generation make it exact.
        return (st.changed_ns, st.etag, st.gen)

    def version(self, home: Path) -> Tuple[Any, ...]:
        return self._version_of(self._state(home))

    def exists(self, home: Path) -> bool:
        return True  # the plane always resolves a document (an absent level is empty, contract R4)

    def locked(self, home: Path, dotted: str) -> Optional[str]:
        path = from_dotted(dotted)
        for lock, level in self._state(home).locks:
            if len(lock) <= len(path) and tuple(path[:len(lock)]) == lock:
                return level
        return None

    # --- ConfigBackend: writes --------------------------------------------------------------

    def write_changes(self, home: Path, changes: Changes) -> None:
        st = self._state(home)
        with st.lock:
            for attempt in (1, 2):
                built = self._build_patch(st, changes)
                if built is None:
                    return
                body, unsets = built
                try:
                    resp = client.request("PATCH", st.home, st.profile, body=body)
                except (PlaneCredentialError, client.TransportError) as exc:
                    raise ConfigWriteError(f"Remote Config write failed: {exc}", code="config_plane_unreachable") from exc
                if resp.status == 200 and _effective_ok(resp.body):
                    self._install(st, resp.body, resp.etag)
                    self._warn_inherited(st, unsets)
                    return
                if resp.status == 409 and attempt == 1:
                    try:  # contract §15.3: re-read, re-diff, retry once
                        self._fetch_into(st, conditional=False)
                    except _FetchFailed as exc:
                        raise ConfigWriteError(f"Remote Config write conflict and re-read failed: {exc}",
                                               code="config_version_conflict") from exc
                    continue
                raise self._write_error(resp)

    def _build_patch(self, st: _ProfileState, changes: Changes) -> Optional[Tuple[Dict[str, Any], List[KeyPath]]]:
        """The PATCH body for ``changes`` against the last read (§15.2), or None for no change."""
        base = copy.deepcopy(st.server_config)
        if changes.document is not None:
            new = to_wire(copy.deepcopy(changes.document))
            if not isinstance(new, dict):
                raise ConfigValueError("the config document must be a mapping")
        else:
            new = copy.deepcopy(base)
        for key, value in changes.set.items():
            path = from_dotted(key)
            _set_path(new, path, to_wire(value, path))
        for key in changes.unset:
            _pop_path(new, from_dotted(key))
        base.pop("_config_version", None)
        new.pop("_config_version", None)

        if changes.document is not None:
            base, new, dropped = strip_locked(base, new, st.locks)
            if dropped:
                print(f"Note: {len(dropped)} setting(s) locked by Remote Config were not saved: "
                      f"{', '.join(sorted(encode(p) for p in dropped))}", file=sys.stderr)
        sets, unsets = diff(base, new)
        refused = write_check(sets, unsets, st.locks)
        if refused is not None:
            raise ConfigLockedError(encode(refused[0]), refused[1],
                                    f"{encode(refused[0])} is locked by the {refused[1]} level of Remote Config")
        for p, v in sets.items():
            bad = secret_literal_path(p, v)
            if bad is not None:
                raise ConfigValueError(
                    f"{encode(bad)} is a secret-shaped key: Remote Config stores only a ${{VAR}} reference "
                    "there, never the secret itself. Put the value in .env and set a ${VAR} reference.",
                    code="config_secret_literal")
        if not sets and not unsets:
            return None
        set_body, unset_body = encode_changes(sets, unsets)
        body: Dict[str, Any] = {"expectedVersion": st.profile_version,
                                "writerConfigVersion": _latest_config_version()}
        if set_body:
            body["set"] = set_body
        if unset_body:
            body["unset"] = unset_body
        return body, unsets

    @staticmethod
    def _write_error(resp: client.Response) -> ConfigWriteError:
        if resp.status == 403 and resp.error == "config_key_locked":
            return ConfigLockedError(str(resp.body.get("path") or ""), str(resp.body.get("lockedBy") or "upper"),
                                     f"Remote Config refused the write: {resp.message}")
        if resp.status == 400 and resp.error == "config_secret_literal":
            return ConfigValueError(f"Remote Config refused the write: {resp.message}", code=resp.error)
        return ConfigWriteError(f"Remote Config refused the write ({resp.status} {resp.error}): {resp.message}",
                                code=resp.error)

    @staticmethod
    def _warn_inherited(st: _ProfileState, unsets: List[KeyPath]) -> None:
        """R5: removing a key at the profile level cannot remove a value an upper level supplies."""
        for p in unsets:
            node: Any = st.server_config
            for seg in p:
                if not isinstance(node, dict) or seg not in node:
                    break
                node = node[seg]
            else:
                logger.warning("Remote Config: %s is still set by an upper level; removing it from this "
                               "profile cannot remove an inherited value (set it to null for 'no value')",
                               encode(p))

    def apply_in_memory(self, home: Path, doc: dict) -> None:
        st = self._state(home)
        with st.lock:
            st.doc = copy.deepcopy(doc)
            st.changed_ns = time.time_ns()
            st.gen += 1

    # --- ConfigBackend: capabilities and boot -----------------------------------------------

    def supports_file_tooling(self) -> bool:
        return False

    def honors_managed_config(self) -> bool:
        return False

    def protected_env_names(self) -> frozenset:
        return PLANE_CREDENTIAL_ENV_NAMES

    def boot(self, home: Path) -> None:
        """Design §4.4 steps 2-4: fetch this home's layer (fail closed) and flag a managed file."""
        self._state(home)
        if not self._managed_warned:
            self._managed_warned = True
            path = managed_config_file()
            if path is not None:
                msg = (f"WARNING: {path} exists but is IGNORED: HERMES_CONFIG_BACKEND=remote takes config "
                       "and locks only from Remote Config (D18). Remove the file.")
                logger.warning(msg)
                print(msg, file=sys.stderr)

    def describe(self, home: Path) -> str:
        st = self._states.get(self._key(home))
        head = f"Remote Config {client.base_url()} (instance {client.instance_id() or '<unset>'})"
        if st is None:
            return f"{head}: profile not fetched yet"
        age = int(time.time() - st.fetched_at) if st.fetched_at else -1
        tail = f", last poll failed: {st.last_error}" if st.last_error else ""
        return (f"{head}: profile {st.profile!r}, profile level v{st.profile_version}, "
                f"{len(st.locks)} lock(s), fetched {age}s ago{tail}")

    # --- poller (D9) ------------------------------------------------------------------------

    def _ensure_poller(self) -> None:
        with self._lock:
            if self._poller is not None and self._poller.is_alive() and self._poller_pid == os.getpid():
                return
            self._poller_pid = os.getpid()  # a forked child re-arms its own thread
            self._poller = threading.Thread(target=self._poll_loop, name="remote-config-poll", daemon=True)
            self._poller.start()

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            now = time.monotonic()
            states = list(self._states.values())
            due = [st for st in states if st.next_poll <= now]
            for st in due:
                self.poll_one(st)
            wake = min((st.next_poll for st in self._states.values()), default=now + poll_interval())
            self._stop.wait(max(1.0, wake - time.monotonic()))

    def poll_one(self, st: _ProfileState) -> bool:
        """One conditional GET (contract §11.3). Never raises and never exits: an error keeps the
        in-memory document and is retried next tick."""
        try:
            changed = self._fetch_into(st, conditional=True)
        except _FetchFailed as exc:
            st.last_error = str(exc)
            # contract §11.3: a refused credential needs an operator; anything else is a blip.
            level = logging.ERROR if exc.status in (401, 403) else logging.WARNING
            logger.log(level, "Remote Config: poll for profile %r failed (%s); keeping the loaded config",
                       st.profile, exc)
            changed = False
        except Exception as exc:  # noqa: BLE001 — the poller thread must survive anything
            st.last_error = str(exc)
            logger.warning("Remote Config: poll for profile %r failed", st.profile, exc_info=True)
            changed = False
        st.next_poll = time.monotonic() + poll_interval() * random.uniform(0.9, 1.1)
        if changed:
            logger.info("Remote Config: profile %r changed (profile level v%d)", st.profile, st.profile_version)
        return changed

    def poll_all(self) -> None:
        for st in list(self._states.values()):
            self.poll_one(st)


def managed_config_file() -> Optional[Path]:
    from hermes_cli import managed_scope
    managed_dir = managed_scope.get_managed_dir()
    path = managed_dir / "config.yaml" if managed_dir else None
    if path is None:
        return None
    return path if path.exists() else None  # config-reader: ok — existence only, to warn it is ignored (D18)


def _set_path(doc: Dict[str, Any], path: KeyPath, value: Any) -> None:
    node = doc
    for seg in path[:-1]:
        if not isinstance(node.get(seg), dict):
            node[seg] = {}
        node = node[seg]
    node[path[-1]] = value


def _pop_path(doc: Dict[str, Any], path: KeyPath) -> None:
    node: Any = doc
    for seg in path[:-1]:
        node = node.get(seg) if isinstance(node, dict) else None
    if isinstance(node, dict):
        node.pop(path[-1], None)
