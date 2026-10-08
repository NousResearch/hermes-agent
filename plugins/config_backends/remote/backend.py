"""The remote config backend (config-config design §4, contract.md §11, §15).

The user layer of each profile home is the effective document the config plane resolves for
(this instance, that profile). It lives in memory only: fetched at boot (fail closed, D2), kept
fresh by a per-process ETag poller (D9), and written back as key-level changes on the profile
level with CAS (D10). Nothing is read from or written to a local ``config.yaml``.
"""
from __future__ import annotations

import contextvars
import copy
import itertools
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
    Changes, ConfigBackendUnavailable, ConfigLockedError, ConfigValueError, ConfigWriteError, UserDoc, UserLayer)

from . import client
from .credentials import PLANE_CREDENTIAL_ENV_NAMES, PlaneCredentialError, credential_kind
from .diff import _get_path, _pop_path, _set_path, apply_intent, diff, encode_changes, intent_diff, json_equal, strip_locked, write_check
from .paths import Path as KeyPath
from .paths import PathError, decode, encode
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


_READ_BASES = 8  # diff bases kept per profile for whole-document saves from an older read (_intent)


@dataclass
class _ProfileState:
    home: Path
    profile: str
    server_config: Dict[str, Any] = field(default_factory=dict)  # EffectiveResponse.config, as sent
    doc: Dict[str, Any] = field(default_factory=dict)            # what readers get (+ _config_version, migrated)
    etag: str = ""
    profile_version: int = 0
    # The profile level's stored writerConfigVersion: None = stored null, _UNKNOWN = not reported.
    profile_writer: Any = None
    other_writers: List[int] = field(default_factory=list)  # the other levels' int stamps (R6)
    locks: List[Tuple[KeyPath, str]] = field(default_factory=list)
    provenance: Dict[str, str] = field(default_factory=dict)
    changed_ns: int = 0
    gen: int = 0
    installed: int = 0         # bumped by every _install; a GET started before a newer install is stale
    base: Optional[Dict[str, Any]] = None  # the migrated doc writes diff against (None = server_config)
    postprocessed: bool = False   # st.doc is the completed migration output of generation `installed`
    fetched_at: float = 0.0
    last_error: Optional[str] = None
    next_poll: float = 0.0
    # The diff base of each recent generation readers were handed (UserDoc.read_version = gen), so a
    # whole-document save is diffed against the doc its caller read, not a newer one (see _intent).
    read_bases: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock)


_UNKNOWN = object()


@dataclass
class _Private:
    """An in-memory migration's working copy of one profile's document (D12).

    The migration steps read and persist through ``hermes_cli.config`` like any reader and writer;
    while one runs, THAT thread sees this copy instead of the published one (``_PRIVATE``), and
    everyone else keeps seeing the published doc with its matching diff base. The migrated doc and
    its base are published together, in one step, when the migration ends: a concurrent writer
    never diffs a half-migrated document against the unmigrated base, or the reverse."""
    st: "_ProfileState"
    doc: Dict[str, Any]
    serial: int
    gen: int = 0


_PRIVATE: "contextvars.ContextVar[Optional[_Private]]" = contextvars.ContextVar("remote_config_private", default=None)
_PRIVATE_SERIAL = itertools.count(1)


def _private_for(st: "_ProfileState") -> Optional[_Private]:
    m = _PRIVATE.get()
    return m if m is not None and m.st is st else None

# Plane refusals caused by the submitted value (contract §2.5, §11.1): surfaced to the caller as
# ConfigValueError with the plane's code. config_request_invalid is deliberately absent: a request
# the plane cannot parse is this client's bug, not the user's value, so it stays opaque.
_VALUE_REFUSALS = frozenset({
    (400, "config_secret_literal"), (400, "config_value_invalid"), (400, "config_path_invalid"),
    (400, "config_path_reserved"), (413, "config_level_too_large"), (413, "config_body_too_large"),
})


def _profile_writer(body: Dict[str, Any]) -> Any:
    """The profile level's stored ``writerConfigVersion`` (int or None), or ``_UNKNOWN``."""
    for lv in body.get("levels") or []:
        if isinstance(lv, dict) and lv.get("kind") == "profile":
            w = lv.get("writerConfigVersion")
            return w if w is None or isinstance(w, int) else _UNKNOWN
    return _UNKNOWN


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
            poller = self._poller
            # A forked child has no poller thread, and a poller that died must not leave every
            # cached profile without updates until the next new profile happens to re-arm it.
            if poller is None or self._poller_pid != os.getpid() or not poller.is_alive():
                self._ensure_poller()
            return st
        # The Cloud credential resolver (hermes_cli.auth) imports hermes_cli.config, whose import-time
        # reads come back here for this same home. Import it before taking the fetch lock: those
        # reads then fetch first, the resolver imports against a config module that is already
        # loading, and the fetch below finds the state (a fetch lock taken first would deadlock).
        if credential_kind() == "nous":
            import hermes_cli.config  # noqa: F401
        with self._lock:
            fetch_lock = self._fetch_locks.setdefault(key, threading.Lock())
        with fetch_lock:
            st = self._states.get(key)
            if st is None:
                st = self._initial_fetch(Path(home))
                with self._lock:  # the poller snapshots the roster under the same lock
                    self._states[key] = st
        self._ensure_poller()
        return st

    def _roster(self) -> List[_ProfileState]:
        with self._lock:
            return list(self._states.values())

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
        """GET ``/self`` into ``st``; True when the document changed. Raises :class:`_FetchFailed`.

        The GET runs outside ``st.lock`` (a slow plane must not stall writers), so a write can
        install a newer document while it is in flight. Its response is then stale, even when its
        ``profileVersion`` matches (an upper level may have changed in between): it is dropped, and
        the next poll fetches again."""
        started = st.installed
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
                if st.installed != started:
                    logger.debug("Remote Config: dropped a fetch for profile %r that a write overtook", st.profile)
                    return False
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
        st.profile_writer = _profile_writer(body)
        st.other_writers = [
            lv["writerConfigVersion"] for lv in body.get("levels") or []
            if isinstance(lv, dict) and lv.get("kind") != "profile" and isinstance(lv.get("writerConfigVersion"), int)]
        st.locks = locks
        st.provenance = dict(body.get("provenance") or {})
        st.changed_ns = time.time_ns()
        st.gen += 1
        st.installed += 1
        st.base = None
        st.postprocessed = False
        st.fetched_at, st.last_error = time.time(), None
        st.next_poll = time.monotonic() + poll_interval() * random.uniform(0.9, 1.1)
        _record_read_base(st)

    _MAX_POSTPROCESS = 8  # passes when polls keep installing newer docs mid-migration

    def _completed(self, st: _ProfileState) -> bool:
        """Bring the published doc to its completed postprocess output; False when that is not
        possible yet (``hermes_cli.config`` still importing during the boot fetch)."""
        if _private_for(st) is not None:
            return True  # a migration's own reads use its private copy (never another pass)
        for _ in range(self._MAX_POSTPROCESS):
            ready = self._postprocess(st)
            if not ready:
                return False
            with st.lock:
                if st.postprocessed:
                    return True
        raise ConfigWriteError("Remote Config: the profile's config kept changing while it was migrated in memory",
                               code="config_version_conflict")

    def _postprocess(self, st: _ProfileState) -> bool:
        """First read after a change: warn on unknown keys and migrate in memory (D12). Runs on a
        read, not at fetch, because both need ``hermes_cli.config``, which may still be importing
        when the boot fetch runs (then False: nothing could be done).

        No single flight: a reader or writer that finds the published doc unmigrated runs its own
        private pass, even while another thread's pass is in flight, and the first pass to finish
        publishes. Waiting for the other pass instead could deadlock (a migration reads through
        hermes_cli.config's _CONFIG_LOCK, which the waiting caller may hold), and returning the
        unmigrated doc meanwhile would hand that caller legacy settings the migration removes."""
        if _private_for(st) is not None:
            return True  # a migration's own reads: never start another one
        with st.lock:
            if st.postprocessed:
                return True
            # The doc, its schema version and its generation are captured together: a poll may
            # install a newer doc at any point after this, and migrating that doc from THIS version
            # would run steps its data never needed (and, D12, drop settings it holds on purpose).
            seen = st.installed
            current = int(st.doc.get("_config_version") or 0)
            doc = copy.deepcopy(st.doc)
        try:
            from hermes_cli.config import _known_top_level_keys
            from hermes_cli.config_migrations import SUPPORT_FLOOR_VERSION
        except ImportError:
            return False  # still importing; the next read retries
        for key in sorted(set(doc) - _known_top_level_keys() - {"_config_version"}):
            if key not in self._unknown_warned:
                self._unknown_warned.add(key)
                logger.warning("Remote Config: unknown config key %r is ignored by this Hermes version", key)
        latest = _latest_config_version()
        if SUPPORT_FLOOR_VERSION <= current < latest:
            self._migrate_in_memory(st, seen, current, doc)
            return True
        if current < SUPPORT_FLOOR_VERSION:
            logger.warning("Remote Config: profile %r was written by config version %d, below the "
                           "migration floor %d; not migrated", st.profile, current, SUPPORT_FLOOR_VERSION)
        with st.lock:
            if st.installed == seen:  # a doc installed meanwhile needs its own pass
                st.postprocessed = True
        return True

    def _run_private(self, st: _ProfileState, doc: Dict[str, Any], current: int) -> Dict[str, Any]:
        """Run the migrations from schema *current* over a private copy of *doc* and return it.
        Nothing is published: only this thread sees the copy while the steps run (:class:`_Private`)."""
        from hermes_cli.config_migrations import run_migrations
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        m = _Private(st=st, doc=copy.deepcopy(doc), serial=next(_PRIVATE_SERIAL))
        m.doc["_config_version"] = current
        private_token = _PRIVATE.set(m)
        home_token = set_hermes_home_override(st.home)
        try:
            # config_only: a read (or a refused write's check) must change no local file (D12).
            run_migrations(current, {"env_added": [], "config_added": [], "warnings": []}, True, config_only=True)
        finally:
            reset_hermes_home_override(home_token)
            _PRIVATE.reset(private_token)
        return m.doc

    def _migrate_in_memory(self, st: _ProfileState, started: int, current: int, doc: Dict[str, Any]) -> None:
        """Migrate doc generation *started* (*doc*, schema version *current*) in memory (D12).

        The steps work on a private copy; the result is published with its diff base in one step,
        and only while generation *started* is still the installed one and no concurrent pass over
        it has published first: once a poll or write has installed another doc, nothing of this
        migration lands on it (that doc gets its own pass).
        The migrated doc becomes the base later writes diff against: a reader was handed the
        migrated doc, so a write that leaves the migration's changes in place must not send them
        (D12: never write the migration back)."""
        with st.lock:
            if st.installed != started or st.postprocessed:
                return  # replaced, or another pass already published this generation
        migrated = self._run_private(st, doc, current)
        migrated["_config_version"] = _latest_config_version()
        base = copy.deepcopy(migrated)
        base.pop("_config_version", None)
        with st.lock:
            if st.installed != started or st.postprocessed:
                return  # a poll or write installed a newer doc, or a concurrent pass published
            self._publish_locked(st, migrated, base)
            st.postprocessed = True
        logger.info("Remote Config: migrated profile %r in memory from config version %d", st.profile, current)

    @staticmethod
    def _publish_locked(st: _ProfileState, doc: Dict[str, Any], base: Dict[str, Any]) -> None:
        st.doc = doc
        st.base = base
        st.changed_ns = time.time_ns()
        st.gen += 1
        _record_read_base(st)

    # --- ConfigBackend: reads ---------------------------------------------------------------

    def _snapshot(self, st: _ProfileState, take):
        """``take()`` under st.lock once the published doc is completed postprocess output (or
        this thread's own migration copy). A poll that installs a newer doc between completing
        and taking sends the caller round again, so no reader ever sees an unmigrated doc."""
        for _ in range(self._MAX_POSTPROCESS):
            ready = self._completed(st)
            with st.lock:
                if not ready or st.postprocessed or _private_for(st) is not None:
                    return take()
        raise ConfigWriteError("Remote Config: the profile's config kept changing while it was read",
                               code="config_version_conflict")

    def read_user_layer(self, home: Path) -> UserLayer:
        st = self._state(home)

        def take():
            m = _private_for(st)
            # A migration's private copy is never saved as a document; the published one is tagged
            # with its generation so a later whole-document save diffs against exactly this read.
            doc = copy.deepcopy(m.doc) if m is not None else UserDoc(copy.deepcopy(st.doc), read_version=st.gen)
            return doc, self._version_of(st), {encode(p): level for p, level in st.locks}

        doc, version, locks = self._snapshot(st, take)
        return UserLayer(doc=doc, version=version, locks=locks,
                         provenance=f"remote:{client.base_url()} profile={st.profile}")

    def read_user_doc_readonly(self, home: Path) -> Any:
        st = self._state(home)

        def take():
            m = _private_for(st)
            return m.doc if m is not None else st.doc

        return self._snapshot(st, take)

    @staticmethod
    def _version_of(st: _ProfileState) -> Tuple[Any, ...]:
        # Leads with a nanosecond timestamp like the file backend's st_mtime_ns (the TUI's config
        # watcher reads element 0 as a time); the etag and local generation make it exact. A
        # migration's own reads see its private copy under a version of their own, so no config
        # cache ever pairs the published version with a private doc, or the reverse. It must not
        # START with the published version either: config caches match a signature as a prefix.
        m = _private_for(st)
        if m is not None:
            return (-m.serial, m.gen, st.changed_ns, st.etag, st.gen)
        return (st.changed_ns, st.etag, st.gen)

    def version(self, home: Path) -> Tuple[Any, ...]:
        return self._version_of(self._state(home))

    def exists(self, home: Path) -> bool:
        return True  # the plane always resolves a document (an absent level is empty, contract R4)

    def locked(self, home: Path, dotted: str) -> Optional[str]:
        st = self._state(home)
        path = _key_path(st.doc, dotted)
        for lock, level in st.locks:
            if len(lock) <= len(path) and tuple(path[:len(lock)]) == lock:
                return level
        return None

    # --- ConfigBackend: writes --------------------------------------------------------------

    _MAX_PREPARE = 3  # re-preparations when a poll lands between preparing and sending

    def write_changes(self, home: Path, changes: Changes) -> None:
        """Send ``changes`` as one PATCH with CAS (contract §15.2), re-read and retry once on a
        conflict (§15.3).

        The edit is fixed once, against the doc the caller read (its *intent*: the key-level sets
        and unsets that turn that doc into ``changes``). A retry re-diffs that intent against the
        re-read doc, with the re-read locks; it never replays the caller's whole document, which
        would turn every key another writer changed meanwhile into an edit of ours."""
        st = self._state(home)
        intent: Optional[Tuple[Dict[KeyPath, Any], List[KeyPath]]] = None
        for attempt in (1, 2):
            for _ in range(self._MAX_PREPARE):
                # Diff against the doc readers get, i.e. migrated in memory (D12) — also after the
                # CAS re-read below or a poll landing mid-prepare. Outside st.lock: a migration
                # reads through hermes_cli.config.
                ready = self._completed(st)
                with st.lock:
                    if ready and not st.postprocessed:
                        continue  # a poll installed a newer doc since: complete that one first
                    seen = st.installed
                    if intent is None:
                        intent = self._intent(st, changes)
                    built = self._patch_body(st, *intent)
                    if built is None:
                        return
                    body, sets, unsets = built
                    check = self._migration_check_input(st, body, sets, unsets)
                if check is not None:  # outside st.lock, like any migration run
                    self._check_migration_keeps(st, sets, *check)
                with st.lock:
                    if st.installed != seen:
                        continue  # a poll landed meanwhile: prepare again against it
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
                        break
                    raise self._write_error(resp)
            else:
                raise ConfigWriteError("Remote Config write not sent: the profile's config kept changing "
                                       "while the write was prepared", code="config_version_conflict")

    def _intent(self, st: _ProfileState, changes: Changes) -> Tuple[Dict[KeyPath, Any], List[KeyPath]]:
        """The key-level edit ``changes`` makes to the last read doc (§15.2 steps 1-5)."""
        base = _diff_base(st)
        if changes.document is not None:
            new = to_wire(copy.deepcopy(changes.document))
            if not isinstance(new, dict):
                raise ConfigValueError("the config document must be a mapping")
        else:
            new = copy.deepcopy(base)
        for key, value in changes.set.items():
            path = _key_path(new, key)
            _set_path(new, path, to_wire(value, path))
        for key in changes.unset:
            _pop_path(new, _key_path(new, key))
        new.pop("_config_version", None)

        if changes.document is not None:
            # A whole document is the caller's read plus its edits, and a poll or another write may
            # have advanced the doc since. Diff it against the doc that read returned, so what
            # another writer changed meanwhile is not part of this edit (_patch_body applies only
            # the edit). A tagged read names its doc exactly; an untagged one (load_config() is a
            # merged copy, a caller may rebuild the dict) is matched to the recent doc it differs
            # from least, the current one on a tie.
            read = st.read_bases.get(changes.document.read_version) if isinstance(changes.document, UserDoc) else None
            if read is None:
                read = min([base, *reversed(st.read_bases.values())], key=lambda seen: _edit_size(seen, new))
            base = copy.deepcopy(read)
            base, new, dropped = strip_locked(base, new, st.locks)
            if dropped:
                print(f"Note: {len(dropped)} setting(s) locked by Remote Config were not saved: "
                      f"{', '.join(sorted(encode(p) for p in dropped))}", file=sys.stderr)
        # Leaf-level: a key added under a section the doc lacked stays a key-level edit, so a
        # retry after 409 keeps the siblings another writer put in that section meanwhile.
        return intent_diff(base, new)

    def _patch_body(self, st: _ProfileState, sets: Dict[KeyPath, Any], unsets: List[KeyPath]
                    ) -> Optional[Tuple[Dict[str, Any], Dict[KeyPath, Any], List[KeyPath]]]:
        """The PATCH body applying the intent to the current doc, or None when that is no change."""
        base = _diff_base(st)
        new = copy.deepcopy(base)
        apply_intent(new, sets, unsets)
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
        body: Dict[str, Any] = {"expectedVersion": st.profile_version}
        # The stamp describes the profile level's STORED data (contract §5.3: absent = unchanged).
        # Stamp only when that data is already at this agent's version, or unstamped (null: readers
        # already treat it as current, R6). An older stamp means the level still holds pre-migration
        # keys that D12 forbids writing back migrated; re-stamping would make later readers skip
        # those migration steps. A newer stamp must not be lowered either.
        latest = _latest_config_version()
        if st.profile_writer is None or st.profile_writer == latest:
            body["writerConfigVersion"] = latest
        if set_body:
            body["set"] = set_body
        if unset_body:
            body["unset"] = unset_body
        return body, sets, unsets

    def _migration_check_input(self, st: _ProfileState, body: Dict[str, Any], sets: Dict[KeyPath, Any],
                               unsets: List[KeyPath]) -> Optional[Tuple[Dict[str, Any], int]]:
        """``(stored doc after this write, its schema version)`` when readers will migrate it in
        memory after the write (R6: the oldest level's stamp), else None. Call under st.lock."""
        if not sets:
            return None
        try:
            from hermes_cli.config_migrations import SUPPORT_FLOOR_VERSION
        except ImportError:
            return None
        latest = _latest_config_version()
        profile = latest if "writerConfigVersion" in body else st.profile_writer
        stamps = [*st.other_writers, *([profile] if isinstance(profile, int) else [])]
        after = min(stamps) if stamps else latest
        if not SUPPORT_FLOOR_VERSION <= after < latest:
            return None
        stored = copy.deepcopy(st.server_config)
        stored.pop("_config_version", None)
        for p, v in sets.items():
            _set_path(stored, p, copy.deepcopy(v))
        for p in unsets:
            _pop_path(stored, p)
        return stored, after

    def _check_migration_keeps(self, st: _ProfileState, sets: Dict[KeyPath, Any],
                               stored: Dict[str, Any], after: int) -> None:
        """Refuse a write whose values every later read would migrate away.

        The plane stores the write as sent, but under a level stamp older than this agent's schema
        (the stamp keeps describing the level's untouched legacy data, contract §5.3), so every
        reader — this process and any later one — re-runs the migrations from that stamp over the
        stored doc. A value one of those steps rewrites (an old default the user set on purpose)
        would be acknowledged and then never read back. Refused instead, before anything is sent."""
        migrated = self._run_private(st, stored, after)
        latest = _latest_config_version()
        for p, v in sets.items():
            found, got = _get_path(migrated, p)
            changed = _first_change(p, v, found, got)
            if changed is not None:
                raise ConfigValueError(
                    f"Remote Config: {encode(changed)} cannot be saved with that value. This profile's remote "
                    f"settings are stored at config schema v{after}, and every read migrates them to "
                    f"v{latest} in memory, which rewrites that value (migrations are never written back, "
                    "D12), so it would never be read back. Nothing was sent; choose another value, or "
                    "have the profile level re-saved at the current schema on the config plane.",
                    code="config_migration_conflict")

    @staticmethod
    def _write_error(resp: client.Response) -> ConfigWriteError:
        if resp.status == 403 and resp.error == "config_key_locked":
            return ConfigLockedError(str(resp.body.get("path") or ""), str(resp.body.get("lockedBy") or "upper"),
                                     f"Remote Config refused the write: {resp.message}")
        if (resp.status, resp.error) in _VALUE_REFUSALS:
            # The submitted value is the problem (contract §2.5 table): the caller's to fix, so it
            # keeps its code and the plane's refusal text, which names a path and a reason, never a
            # value (config_secret_literal never echoes it).
            detail = resp.message or resp.error
            path, reason = resp.body.get("path"), resp.body.get("reason")
            extras = [f"{k} {v}" for k, v in (("path", path), ("reason", reason))
                      if isinstance(v, str) and v and v not in detail]
            if extras:
                detail = f"{detail} ({', '.join(extras)})"
            return ConfigValueError(f"Remote Config refused the write: {detail}", code=resp.error)
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
        m = _private_for(st)
        if m is not None:  # a migration step: lands in that migration's private copy only
            m.doc = copy.deepcopy(doc)
            m.gen += 1
            return
        with st.lock:  # outside a migration: the doc and the base writes diff against move together
            doc = copy.deepcopy(doc)
            base = copy.deepcopy(doc)
            base.pop("_config_version", None)
            self._publish_locked(st, doc, base)

    # --- ConfigBackend: capabilities and boot -----------------------------------------------

    def supports_file_tooling(self) -> bool:
        return False

    def honors_managed_config(self) -> bool:
        return False

    def protected_env_names(self) -> frozenset:
        return PLANE_CREDENTIAL_ENV_NAMES

    def deployment_env_names(self) -> frozenset:
        from hermes_cli.config_backend import BACKEND_ENV
        return frozenset({BACKEND_ENV, client.URL_ENV, client.INSTANCE_ENV, POLL_ENV})

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
            try:
                wake = self._poll_due()
            except Exception:  # noqa: BLE001 — the only poller thread must survive anything
                logger.warning("Remote Config: poll loop error; retrying", exc_info=True)
                wake = time.monotonic() + MIN_POLL_SECONDS
            self._stop.wait(max(1.0, wake - time.monotonic()))

    def _poll_due(self) -> float:
        """One poller tick: release deleted profiles, poll the due ones; returns the next wake time."""
        now = time.monotonic()
        for st in self._roster():  # a snapshot: a lazily added profile must not break iteration
            if not _home_is_live(st.home):
                self._release(st)
            elif st.next_poll <= now:
                self.poll_one(st)
        return min((st.next_poll for st in self._roster()), default=now + poll_interval())

    def _release(self, st: _ProfileState) -> None:
        """Forget a profile whose home was deleted (by this process or another): no more polls,
        no cached document. A later read of a recreated home fetches afresh."""
        key = self._key(st.home)
        with self._lock:
            if self._states.get(key) is st:
                del self._states[key]
                self._fetch_locks.pop(key, None)

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
        for st in self._roster():
            self.poll_one(st)


def _diff_base(st: _ProfileState) -> Dict[str, Any]:
    base = copy.deepcopy(st.base if st.base is not None else st.server_config)
    base.pop("_config_version", None)
    return base


def _edit_size(base: Dict[str, Any], new: Dict[str, Any]) -> int:
    sets, unsets = diff(base, new)
    return len(sets) + len(unsets)


def _record_read_base(st: _ProfileState) -> None:
    """Remember generation ``st.gen``'s diff base for saves made from a read of it (call under st.lock)."""
    st.read_bases[st.gen] = _diff_base(st)
    for gen in sorted(st.read_bases)[:-_READ_BASES]:
        del st.read_bases[gen]


def _home_is_live(home: Path) -> bool:
    """False once a named profile's home is deleted (tombstoned or gone); the default home always
    counts as live. Identity markers are not required: a remote profile has no local config.yaml."""
    from hermes_constants import named_profile_home, named_profile_is_deleted
    return named_profile_home(home) is None or (home.is_dir() and not named_profile_is_deleted(home))


def managed_config_file() -> Optional[Path]:
    from hermes_cli import managed_scope
    managed_dir = managed_scope.get_managed_dir()
    path = managed_dir / "config.yaml" if managed_dir else None
    if path is None:
        return None
    return path if path.exists() else None  # config-reader: ok — existence only, to warn it is ignored (D18)


def _first_change(path: KeyPath, sent: Any, found: bool, got: Any) -> Optional[KeyPath]:
    """The first path at or under *path* where *got* (as read back) differs from *sent*."""
    if not found:
        return path
    if isinstance(sent, dict) and isinstance(got, dict):
        for key, value in sent.items():
            changed = _first_change(path + (key,), value, key in got, got.get(key))
            if changed is not None:
                return changed
        return None
    try:
        return None if json_equal(to_wire(copy.deepcopy(got), path), sent) else path
    except ConfigValueError:
        return path


def _key_path(doc: Any, dotted: str) -> KeyPath:
    """``dotted`` as the file backend reads it (``utils.atomic_roundtrip_yaml_update``): ``\\.``
    escapes a dot, and an existing literal dotted key (``grok-4.6``) wins over splitting it."""
    from hermes_cli.config import _greedy_literal_match, _split_key_path
    parts, path, i = _split_key_path(dotted), [], 0
    while i < len(parts):
        seg, consumed = _greedy_literal_match(doc, parts[i:]) or (parts[i], 1)
        path.append(seg)
        doc = doc.get(seg) if isinstance(doc, dict) else None
        i += consumed
    return tuple(path)
