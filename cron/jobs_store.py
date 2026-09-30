"""Opt-in strict durability for the cron job store (``cron.store.strict_durability``).

A store that opts in is LATCHED: under its lock, before its first mutation (a strict read is
enough), ``jobs.json.strict-durability`` is published beside the resolved store. From then on
the store stays strict whatever happens to ``config.yaml`` — removal, ``null``, a dropped key, an
explicit ``false``, a parse error or a broken managed overlay all refuse loudly until an
operator follows the documented manual recovery. A profile that never opted in keeps the
historical default path, including its liveness when ``config.yaml`` cannot be parsed.

Strict sections hold the legacy logical ``<cron dir>/.jobs.lock`` (so default-mode and older
processes still exclude them) and then the physical ``.jobs.lock`` beside the resolved store.
Both lock files are opened without following symlinks, and their name/inode identity is proven
before root hands one to the store owner. The section pins the store's directory as an open
directory fd: every stage, read, rename, link, fsync and cleanup is relative to that fd, and
the directory's path identity is re-checked before AND after every commit, so a directory or
symlink swapped mid-publication can never receive a commit reported as clean.

Reads refuse a vanished primary once a primary was published, and refuse a noncanonical
primary (not strict JSON: duplicate keys at any level, NaN/Infinity, overflowing floats, a BOM,
invalid UTF-8, lone surrogate escapes, a noncanonical ``repeat.completed``) after preserving
its exact bytes as ``jobs.json.corrupt-<sha256>``. Publication stages beside the target,
verifies mode AND owner on the staged fd for every writer, fsyncs, replaces once, fsyncs the
directory, then refreshes ``jobs.json.last-good`` the same way. A last-good refresh failure
after a durable primary is a warning returned to the caller, not a failure. Nothing here
deletes or restores a backup, marker or snapshot, and nothing deletes a file it has not proven
to be its own. POSIX only.
"""

import codecs
import contextlib
import hashlib
import json
import logging
import math
import os
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

LAST_GOOD_SUFFIX = ".last-good"
CORRUPT_INFIX = ".corrupt-"
LATCH_SUFFIX = ".strict-durability"
# Set on a BaseException (KeyboardInterrupt, SystemExit) that escaped a strict publication, so a
# caller can tell a rollback from a possible commit without the shutdown being swallowed.
OUTCOME_ATTR = "cron_store_outcome"
OUTCOME_UNCHANGED = "unchanged"
OUTCOME_UNCERTAIN = "uncertain"
_STORE_KEYS = frozenset({"strict_durability"})
_PRIVATE_MODE = 0o600
_LATCH_ARMED = b"cron.store.strict_durability latched; no jobs.json published under it yet\n"
_LATCH_PUBLISHED = b"cron.store.strict_durability latched; jobs.json has been published\n"
_RECOVERY = "Follow the manual recovery steps in the cron docs (Strict store durability)."
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_STAGING_PREFIX = ".jobs_"
_STAGING_SUFFIX = ".tmp"


class CronStoreError(RuntimeError):
    """Strict store refused or could not publish a mutation; jobs.json is unchanged. A
    RuntimeError so callers' corrupt-store handling records it and the scheduler retries."""

    cron_store_outcome = OUTCOME_UNCHANGED


class CronStoreUncertainError(CronStoreError):
    """The primary rename may have happened but a later durability step failed: the new store
    may be visible and may or may not survive a crash. Never reported as a rollback. Callers
    must read the store back before retrying a create (``job_id`` names the attempted job when
    a create raised it)."""

    cron_store_outcome = OUTCOME_UNCERTAIN


class _ConfigUnreadable(Exception):
    """config.yaml or the managed overlay cannot be read as a mapping."""

    def __init__(self, message: str, user_opted_in: bool = False):
        super().__init__(message)
        self.user_opted_in = user_opted_in


def last_good_path(target: Path) -> Path:
    return target.with_name(target.name + LAST_GOOD_SUFFIX)


def latch_path(target: Path) -> Path:
    return target.with_name(target.name + LATCH_SUFFIX)


def forensic_path(target: Path, raw: bytes) -> Path:
    """Content-addressed (full SHA-256): the same damaged bytes always map to the same file."""
    return target.with_name(f"{target.name}{CORRUPT_INFIX}{hashlib.sha256(raw).hexdigest()}")


def lock_path(jobs_file: Path) -> Path:
    """Physical ``.jobs.lock`` beside the resolved store, shared by every alias of one file."""
    return Path(os.path.realpath(jobs_file)).parent / ".jobs.lock"


def open_lock_file(path: Path):
    """Legacy lock file for a strict section: never follows a planted symlink, never blocks on a
    FIFO, never leaks into children. Identity and owner are proven later, before any chown."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT | _O_NOFOLLOW | _O_NONBLOCK | _O_CLOEXEC, _PRIVATE_MODE)
    try:
        return os.fdopen(fd, "a+")
    except BaseException:
        os.close(fd)
        raise


def _not_held(jobs_file: Path) -> CronStoreError:
    return CronStoreError(
        f"Cron store locks for {jobs_file} are not held (another process held one past the "
        "timeout, flock is unsupported here, a lock file cannot be opened or is a symlink, or the "
        "section holds a different store's lock); strict durability refuses to use jobs.json "
        "without them. The scheduler retries on its next tick.")


def _mark_outcome(exc: BaseException, outcome: str, what: str) -> None:
    """Type a non-CronStoreError escape (KeyboardInterrupt, SystemExit, ...) without replacing
    or suppressing it; a later, wider stage (the whole store) overrides an inner file's outcome."""
    if isinstance(exc, CronStoreError):
        return
    setattr(exc, OUTCOME_ATTR, outcome)
    exc.add_note(f"cron store: {what} "
                 f"{'may have been replaced' if outcome == OUTCOME_UNCERTAIN else 'was not changed'}")
    if outcome == OUTCOME_UNCERTAIN:
        logger.error("Interrupted (%s) after %s may have been replaced; check it before relying on it",
                     type(exc).__name__, what)


# --- Strict JSON ---------------------------------------------------------------------------

class _Noncanonical(ValueError):
    pass


def _unique_pairs(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    obj: Dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise _Noncanonical("duplicate object key")
        obj[key] = value
    return obj


def _no_constant(_name: str) -> Any:
    raise _Noncanonical("NaN or Infinity")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise _Noncanonical("number overflows a finite float")
    return value


def parse_strict(raw: bytes) -> Any:
    """RFC 8259 JSON with unique keys in every object, finite numbers only, and text that
    round-trips as UTF-8 (a lone surrogate escape could be read but never saved again)."""
    if raw.startswith(codecs.BOM_UTF8):
        raise _Noncanonical("UTF-8 byte-order mark")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _Noncanonical("invalid UTF-8") from exc
    data = json.loads(text, object_pairs_hook=_unique_pairs,
                      parse_constant=_no_constant, parse_float=_finite_float)
    try:
        json.dumps(data, ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError as exc:
        raise _Noncanonical("lone surrogate escape") from exc
    return data


def _count(n: int, what: str) -> str:
    return f"{n} {what}{'' if n == 1 else 's'}"


def noncanonical_reason(raw: bytes) -> Optional[str]:
    """None for a canonical store, else a reason naming shapes and counts only, never content.
    The same validator the store uses; the manual recovery steps call it on candidate bytes."""
    try:
        data = parse_strict(raw)
    except _Noncanonical as exc:
        return f"not strict JSON: {exc}"
    except (ValueError, RecursionError):
        return "not strict UTF-8 JSON"
    if not isinstance(data, dict):
        return f"top level is {type(data).__name__}, not an object"
    jobs = data.get("jobs")
    if not isinstance(jobs, list):
        return f"'jobs' is {type(jobs).__name__}, not a list"
    junk = sum(not isinstance(job, dict) for job in jobs)
    if junk:
        return _count(junk, "non-object job entry").replace("entrys", "entries")
    ids = [job.get("id") for job in jobs]
    blank = sum(not isinstance(i, str) or not i.strip() for i in ids)
    if blank:
        return _count(blank, "job entry").replace("entrys", "entries") + " without a nonempty string 'id'"
    dupes = len(ids) - len(set(ids))
    if dupes:
        return _count(dupes, "duplicate job id")
    # A hand-edited count (null, "2", 1.0, -1) would otherwise be coerced on read — to 0 for
    # junk, which silently regrants every run of an exhausted job.
    counts = sum(
        isinstance(job.get("repeat"), dict) and "completed" in job["repeat"]
        and not (type(job["repeat"]["completed"]) is int and job["repeat"]["completed"] >= 0)
        for job in jobs)
    if counts:
        return _count(counts, "job") + " whose repeat.completed is not a non-negative integer"
    return None


def parse_jobs(raw: bytes) -> List[Dict[str, Any]]:
    """Jobs of an already validated canonical store."""
    return parse_strict(raw)["jobs"]


def serialize(jobs: List[Dict[str, Any]], updated_at: str) -> bytes:
    """Store payload bytes, strict JSON (``allow_nan=False``) and validated before staging."""
    try:
        payload = json.dumps({"jobs": jobs, "updated_at": updated_at}, indent=2,
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CronStoreError(
            f"Refusing to publish a jobs store that is not strict JSON ({type(exc).__name__})") from exc
    reason = noncanonical_reason(payload)
    if reason:
        raise CronStoreError(f"Refusing to publish a noncanonical jobs store ({reason})")
    return payload


# --- Policy ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class StorePolicy:
    strict: bool
    target: Path


# config path -> (file signatures, effective value)
_POLICY_CACHE: Dict[str, Tuple[Any, Optional[bool]]] = {}
_WARNED_UNREADABLE: set = set()


def _sig(path: Optional[Path]) -> Any:
    if path is None:
        return None
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        return ("unreadable", exc.errno)
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _yaml_mapping(path: Path) -> Optional[Dict[str, Any]]:
    """Top-level mapping of ``path``; None when absent. The loaders read a non-mapping as ``{}``,
    which for this policy would be a silent downgrade."""
    from utils import fast_safe_load

    failure: Optional[str] = None
    try:
        with open(path, encoding="utf-8-sig") as f:
            data = fast_safe_load(f)
    except FileNotFoundError:
        return None
    except Exception as exc:
        failure = type(exc).__name__
    if failure is not None:
        # Type only, raised outside the handler: a parser error quotes file content, and a
        # chained cause or context would print it in every traceback of this refusal.
        raise _ConfigUnreadable(f"{path} cannot be parsed ({failure})")
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise _ConfigUnreadable(f"{path} is not a YAML mapping at the top level (got {type(data).__name__})")
    return data


def _configured_value(cfg: Dict[str, Any], where: str) -> Optional[bool]:
    """``cron.store.strict_durability`` of one layer: None when absent or null; anything but a
    real bool (including an env-expanded "true" string), a non-mapping section or an unknown
    ``cron.store`` key raises."""
    node: Any = cfg
    for key in ("cron", "store"):
        node = node.get(key)
        if node is None:
            return None
        if not isinstance(node, dict):
            raise CronStoreError(
                f"{where}: '{key}' must be a mapping (got {type(node).__name__}); fix it before "
                "cron can use jobs.json.")
    unknown = sorted(map(repr, set(node) - _STORE_KEYS))
    if unknown:
        raise CronStoreError(
            f"{where}: cron.store has unknown key(s) {', '.join(unknown)}; the only supported key "
            "is 'strict_durability' (true or false). Fix the spelling before cron uses jobs.json.")
    value = node.get("strict_durability")
    if value is not None and not isinstance(value, bool):
        raise CronStoreError(
            f"{where}: cron.store.strict_durability must be true or false (got "
            f"{type(value).__name__}); refusing to guess the store policy.")
    return value


def _managed_config_path() -> Optional[Path]:
    from hermes_cli import managed_scope

    managed_dir = managed_scope.get_managed_dir()
    return managed_dir / "config.yaml" if managed_dir is not None else None


def _config_opt_in(home: Path) -> Optional[bool]:
    """Effective ``cron.store.strict_durability`` (user file + managed overlay through the
    canonical loader), read so both layers are the versions the loader used. Raises
    ``_ConfigUnreadable`` where the loaders would silently fall back or fail open — including
    an effective value that disagrees with a validated managed pin (the overlay was not
    applied), so a managed ``true`` can never be lost to a merge the loader swallowed."""
    from hermes_cli.config_effective import load_user_config_effective

    config_path = home / "config.yaml"
    for _attempt in range(3):
        managed_path = _managed_config_path()
        sigs = (_sig(config_path), managed_path, _sig(managed_path))
        cached = _POLICY_CACHE.get(str(config_path))
        if cached is not None and cached[0] == sigs:
            return cached[1]
        # The managed layer is read and validated first and independently: a managed ``true``
        # must survive a user config.yaml that cannot be parsed, even before the first latch.
        managed_value: Optional[bool] = None
        managed_unreadable: Optional[_ConfigUnreadable] = None
        try:
            managed = _yaml_mapping(managed_path) if managed_path is not None else None
        except _ConfigUnreadable as exc:
            managed_unreadable = exc
        else:
            if managed:
                managed_value = _configured_value(managed, f"managed {managed_path}")
        try:
            user_value = _configured_value(_yaml_mapping(config_path) or {}, str(config_path))
        except _ConfigUnreadable as exc:
            exc.user_opted_in = managed_value is True
            raise
        if managed_unreadable is not None:
            managed_unreadable.user_opted_in = user_value is True
            raise managed_unreadable
        failure: Optional[str] = None
        try:
            cfg = load_user_config_effective(config_path, fail_closed=True)
        except _ConfigUnreadable as exc:
            exc.user_opted_in = user_value is True or managed_value is True
            raise
        except CronStoreError:
            raise
        except Exception as exc:
            failure = type(exc).__name__
        if failure is not None:
            # Outside the handler, like _yaml_mapping: the loader's error may quote config content.
            raise _ConfigUnreadable(
                f"the effective config for {config_path} cannot be built ({failure})",
                user_opted_in=user_value is True or managed_value is True)
        value = _configured_value(cfg, f"effective config of {config_path}")
        if managed_value is not None and value is not managed_value:
            raise _ConfigUnreadable(
                f"the effective config for {config_path} does not carry the managed "
                f"cron.store.strict_durability: {str(managed_value).lower()} (the managed overlay "
                "was not applied)", user_opted_in=user_value is True or managed_value is True)
        if (_sig(config_path), managed_path, _sig(managed_path)) == sigs:
            _POLICY_CACHE[str(config_path)] = (sigs, value)
            return value
    raise _ConfigUnreadable(f"{config_path} (or the managed config) kept changing while cron read it")


def is_latched(target: Path) -> bool:
    """A store that has ever been strict: its latch marker or its last-good backup exists."""
    return os.path.lexists(latch_path(target)) or os.path.lexists(last_good_path(target))


def resolve_policy(jobs_file: Path, home: Path) -> StorePolicy:
    """Store policy for one lock section. Raises for a latched store whose config no longer
    affirms ``strict_durability: true`` and for an invalid ``cron.store`` value; a never-opted-in
    store with an unreadable config keeps the default path (the historical liveness)."""
    target = Path(os.path.realpath(jobs_file))
    latched = is_latched(target)
    try:
        value = _config_opt_in(home)
    except _ConfigUnreadable as exc:
        if latched or exc.user_opted_in:
            raise CronStoreError(
                f"{exc}; {target} is opted in to strict durability, so cron neither reads nor "
                "writes it until the config is fixed (`hermes config edit`, then `hermes config "
                "check`). The scheduler retries on its next tick.") from None
        key = (str(home), str(exc))
        if key not in _WARNED_UNREADABLE:
            _WARNED_UNREADABLE.add(key)
            logger.warning("%s; this profile's cron store never opted in to strict durability, so "
                           "it keeps the default store path", exc)
        return StorePolicy(False, target)
    if latched and value is not True:
        said = ("sets strict_durability: false" if value is False
                else "no longer sets cron.store.strict_durability: true")
        raise CronStoreError(
            f"{target} is latched to strict durability ({latch_path(target).name} or "
            f"{last_good_path(target).name} exists) but the effective config {said}. Cron will "
            "not silently downgrade an activated strict store: restore `strict_durability: true`, "
            f"or leave strict mode deliberately. {_RECOVERY}")
    return StorePolicy(value is True, target)


# --- Ownership and modes ----------------------------------------------------------------------

def _euid() -> Optional[int]:
    geteuid = getattr(os, "geteuid", None)
    return geteuid() if geteuid is not None else None


def _stage_mode_and_owner(fd: int, mode: int, owner: Tuple[int, int], what: Path) -> None:
    """Mode, then owner (root may chown; anyone may pick one of their groups), then verify BOTH on
    the staged fd. A writer that cannot produce the store owner's file refuses before publishing."""
    os.fchmod(fd, mode)
    st = os.fstat(fd)
    if (st.st_uid, st.st_gid) != owner:
        try:
            os.fchown(fd, owner[0] if st.st_uid != owner[0] else -1,
                      owner[1] if st.st_gid != owner[1] else -1)
            os.fchmod(fd, mode)
        except OSError as exc:
            raise CronStoreError(
                f"Could not give {what} owner uid={owner[0]} gid={owner[1]} "
                f"({exc.strerror or type(exc).__name__}); refusing to publish a file the store "
                "owner may not be able to use. Run cron as the store owner (or root).") from exc
    st = os.fstat(fd)
    if stat.S_IMODE(st.st_mode) != mode or (st.st_uid, st.st_gid) != owner:
        raise CronStoreError(
            f"{what} was staged as mode {stat.S_IMODE(st.st_mode):o} uid={st.st_uid} "
            f"gid={st.st_gid}, not mode {mode:o} uid={owner[0]} gid={owner[1]}; refusing to publish it.")


def prepare_dirs(cron_dir: Path, output_dir: Path, home: Path) -> None:
    """Root creating a strict store's missing cron dirs inside a gateway-owned profile home hands
    them to the home's owner first, or removes what it created and refuses — never leaving a
    root-only directory the gateway cannot enter. Other writers use ``ensure_dirs`` as before."""
    if _euid() != 0 or not home.is_dir():
        return
    home_st = os.stat(home)
    if home_st.st_uid == 0:
        return
    created: List[Path] = []
    try:
        for directory in (cron_dir, output_dir):
            try:
                os.mkdir(directory, 0o700)
            except FileExistsError:
                continue
            created.append(directory)
            fd = os.open(directory, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC)
            try:
                os.fchown(fd, home_st.st_uid, home_st.st_gid)
                st = os.fstat(fd)
            finally:
                os.close(fd)
            if (st.st_uid, st.st_gid) != (home_st.st_uid, home_st.st_gid):
                raise CronStoreError(f"{directory} could not be handed to uid={home_st.st_uid}")
    except BaseException as exc:
        for directory in reversed(created):
            with contextlib.suppress(OSError):
                os.rmdir(directory)
        if isinstance(exc, OSError):
            raise CronStoreError(
                f"Could not create {cron_dir} owned by the profile owner uid={home_st.st_uid} "
                f"({exc.strerror or type(exc).__name__}); nothing was left behind.") from exc
        raise


def _require_dir_owner(st: os.stat_result, directory: Path, owner: Tuple[int, int]) -> None:
    """Root only: files handed to the store owner inside a directory it does not own may be
    unreachable for it, so refuse before creating any."""
    if _euid() == 0 and st.st_uid != owner[0]:
        raise CronStoreError(
            f"{directory} is owned by uid={st.st_uid} but the cron store belongs to uid={owner[0]}; "
            "a root write would leave files the gateway user cannot reach. Fix the directory "
            "ownership, then retry.")


# --- Lock section ----------------------------------------------------------------------------

@dataclass
class _HeldLock:
    path: Path
    fd: int
    ident: Tuple[int, int]


@dataclass
class StrictSection:
    """What one strict lock section holds: the store's directory pinned as ``dir_fd`` (every file
    operation is relative to it), the resolved ``name`` inside it, and the held lock inodes."""
    jobs_file: Path
    target: Path
    owner: Tuple[int, int]
    parent_id: Tuple[int, int]
    dir_fd: int
    locks: Tuple[_HeldLock, ...]
    physical_lock: Any = None  # opened here; released by close_section before the legacy lock

    @property
    def name(self) -> str:
        return self.target.name

    @property
    def parent(self) -> Path:
        return self.target.parent


def _lstat_at(dir_fd: int, name: str) -> Optional[os.stat_result]:
    try:
        return os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _prove_lock(path: Path, fd: int, named: Optional[os.stat_result], owner: Tuple[int, int]) -> _HeldLock:
    """The open lock fd IS the regular, single-link file at ``path`` (never a symlink's target)
    — proven BEFORE root changes its owner. Root then only hands over an empty lock file that
    root or the store owner created; anything else is refused and left exactly as found."""
    st = os.fstat(fd)
    ident = (st.st_dev, st.st_ino)
    if named is None or stat.S_ISLNK(named.st_mode) or (named.st_dev, named.st_ino) != ident:
        raise CronStoreError(
            f"{path} is a symlink or was replaced; the lock held is not this store's lock. It is "
            "left untouched.")
    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
        raise CronStoreError(f"{path} is not a single-link regular lock file; it is left untouched")
    if _euid() == 0 and (st.st_uid, st.st_gid) != owner:
        if st.st_uid not in (0, owner[0]) or st.st_size != 0:
            raise CronStoreError(
                f"{path} belongs to uid={st.st_uid} (size {st.st_size}), not the store owner "
                f"uid={owner[0]}; root will not take it over. Fix its ownership, then retry.")
        os.fchown(fd, owner[0] if st.st_uid != owner[0] else -1,
                  owner[1] if st.st_gid != owner[1] else -1)
        st = os.fstat(fd)
        if (st.st_uid, st.st_gid) != owner:
            raise CronStoreError(f"{path} could not be handed to uid={owner[0]}")
    return _HeldLock(path, fd, ident)


def open_section(jobs_file: Path, cron_dir: Path, home: Path, legacy_lock: Any,
                 acquire: Callable[[Any, float], Optional[bool]], timeout: float,
                 release: Callable[[Any], None]) -> StrictSection:
    """Legacy logical lock (already held by the caller) first, then the physical one. Every
    process takes them in that order; crossed aliases (A's store inside B's cron dir and vice
    versa) can still contend, which the bounded ``timeout`` turns into a refusal, never a hang.
    On any failure everything opened here is closed before raising."""
    if legacy_lock is None:
        raise _not_held(jobs_file)
    target = Path(os.path.realpath(jobs_file))
    physical = None
    dir_fd: Optional[int] = None
    try:
        dir_fd = os.open(target.parent, os.O_RDONLY | _O_DIRECTORY | _O_CLOEXEC)
        pst = os.fstat(dir_fd)
        named_parent = os.stat(target.parent)
        if (named_parent.st_dev, named_parent.st_ino) != (pst.st_dev, pst.st_ino):
            raise CronStoreError(f"{target.parent} changed while cron opened it")
        existing = _lstat_at(dir_fd, target.name)
        owner_st = existing if existing is not None and stat.S_ISREG(existing.st_mode) else os.stat(home)
        owner = (owner_st.st_uid, owner_st.st_gid)
        _require_dir_owner(pst, target.parent, owner)
        real_cron = Path(os.path.realpath(cron_dir))
        _require_dir_owner(os.stat(real_cron), real_cron, owner)
        legacy_path = cron_dir / ".jobs.lock"
        locks = [_prove_lock(legacy_path, legacy_lock.fileno(), _lstat_or_none(legacy_path), owner)]
        if real_cron != target.parent:
            fd = os.open(".jobs.lock", os.O_RDWR | os.O_CREAT | _O_NOFOLLOW | _O_NONBLOCK | _O_CLOEXEC,
                         _PRIVATE_MODE, dir_fd=dir_fd)
            try:
                physical = os.fdopen(fd, "a+")
            except BaseException:
                os.close(fd)
                raise
            if acquire(physical, timeout) is not True:
                raise _not_held(jobs_file)
            locks.append(_prove_lock(target.parent / ".jobs.lock", physical.fileno(),
                                     _lstat_at(dir_fd, ".jobs.lock"), owner))
        section = StrictSection(jobs_file, target, owner, (pst.st_dev, pst.st_ino), dir_fd,
                                tuple(locks), physical)
        verify_section(section)
        return section
    except BaseException as exc:
        if physical is not None:
            release(physical)
        if dir_fd is not None:
            os.close(dir_fd)
        if isinstance(exc, OSError):
            raise CronStoreError(
                f"Cannot hold the strict cron store locks for {jobs_file} "
                f"({exc.strerror or type(exc).__name__}); the scheduler retries on its next tick.") from exc
        raise


def close_section(section: StrictSection, release: Callable[[Any], None]) -> None:
    """Release the physical lock and close the pinned directory; the caller then releases the
    legacy lock (reverse acquisition order). Runs on every exit path of the section."""
    try:
        if section.physical_lock is not None:
            release(section.physical_lock)
    finally:
        os.close(section.dir_fd)


def _lstat_or_none(path: Path) -> Optional[os.stat_result]:
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None


def verify_section(section: StrictSection) -> None:
    """The held lock inodes are still the lock paths, the store still resolves to the same
    target, and the pinned directory is still the one at that path: otherwise a commit could
    land outside the lock or where no reader looks. Called before and after every commit."""
    try:
        for lock in section.locks:
            held, named = os.fstat(lock.fd), os.lstat(lock.path)
            if (held.st_dev, held.st_ino) != lock.ident or (named.st_dev, named.st_ino) != lock.ident:
                raise CronStoreError(f"{lock.path} was replaced while held; refusing to commit outside the lock")
        if Path(os.path.realpath(section.jobs_file)) != section.target:
            raise CronStoreError(
                f"{section.jobs_file} was retargeted while its lock was held; refusing to commit")
        pinned, named_dir = os.fstat(section.dir_fd), os.stat(section.parent)
        if ((pinned.st_dev, pinned.st_ino) != section.parent_id
                or (named_dir.st_dev, named_dir.st_ino) != section.parent_id):
            raise CronStoreError(f"{section.parent} was replaced while its lock was held")
    except OSError as exc:
        raise CronStoreError(
            f"Cannot verify the strict cron store lock ({exc.strerror or type(exc).__name__})") from exc


# --- File primitives (all relative to the pinned directory) -------------------------------------

def _fsync_dir(dir_fd: int) -> None:
    os.fsync(dir_fd)


def _read_all(fd: int) -> bytes:
    chunks = []
    while True:
        chunk = os.read(fd, 1 << 20)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _open_no_follow(dir_fd: int, name: str) -> Optional[int]:
    """Read fd that never follows a final symlink and never blocks on a FIFO; None if absent."""
    try:
        return os.open(name, os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK | _O_CLOEXEC, dir_fd=dir_fd)
    except FileNotFoundError:
        return None


def _read_regular(section: StrictSection, name: str) -> Optional[bytes]:
    fd = _open_no_follow(section.dir_fd, name)
    if fd is None:
        return None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise CronStoreError(f"{section.parent / name} is not a regular file; strict durability will not read it")
        return _read_all(fd)
    finally:
        os.close(fd)


def _trust_problems(st: os.stat_result, owner: Tuple[int, int], mode: int, links: int = 1) -> List[str]:
    if not stat.S_ISREG(st.st_mode):
        return ["not a regular file"]
    problems = []
    if stat.S_IMODE(st.st_mode) != mode:
        problems.append(f"mode {stat.S_IMODE(st.st_mode):o}, expected {mode:o}")
    if (st.st_uid, st.st_gid) != owner:
        problems.append(f"owner {st.st_uid}:{st.st_gid}, expected {owner[0]}:{owner[1]}")
    if st.st_nlink != links:
        problems.append(f"{st.st_nlink} links, expected {links}")
    return problems


def _read_verified(section: StrictSection, name: str, mode: int, *, links: int = 1,
                   sync: bool = False) -> Optional[Tuple[bytes, Tuple[int, int]]]:
    """``(bytes, (dev, ino))`` of a file this module owns (latch, forensic copy or its staging):
    regular, ``links`` links, exactly ``mode`` and the store owner, else refused and left
    untouched. ``sync`` fsyncs it on reuse."""
    try:
        fd = _open_no_follow(section.dir_fd, name)
    except OSError as exc:
        raise CronStoreError(
            f"{name} exists but cannot be opened as a regular file "
            f"({exc.strerror or type(exc).__name__}); it is left untouched") from exc
    if fd is None:
        return None
    try:
        st = os.fstat(fd)
        problems = _trust_problems(st, section.owner, mode, links)
        if problems:
            raise CronStoreError(
                f"{name} exists but is not trusted ({'; '.join(problems)}); it is left untouched "
                "and never overwritten")
        data = _read_all(fd)
        if sync:
            os.fsync(fd)
        return data, (st.st_dev, st.st_ino)
    finally:
        os.close(fd)


def _unlink_own(dir_fd: int, name: str, ident: Optional[Tuple[int, int]]) -> None:
    """Remove ``name`` only while it is still exactly the inode this process created."""
    if ident is None:
        return
    with contextlib.suppress(OSError):
        st = _lstat_at(dir_fd, name)
        if st is not None and (st.st_dev, st.st_ino) == ident:
            os.unlink(name, dir_fd=dir_fd)


def _create_staging(dir_fd: int, name: Optional[str] = None) -> Tuple[int, str, Tuple[int, int]]:
    """A new owner-only file in the pinned directory (``O_EXCL``, no symlink follow)."""
    name = name or f"{_STAGING_PREFIX}{secrets.token_hex(8)}{_STAGING_SUFFIX}"
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW | _O_CLOEXEC,
                 _PRIVATE_MODE, dir_fd=dir_fd)
    try:
        st = os.fstat(fd)
    except BaseException:
        os.close(fd)
        with contextlib.suppress(OSError):
            os.unlink(name, dir_fd=dir_fd)
        raise
    return fd, name, (st.st_dev, st.st_ino)


def _fdopen_wb(fd: int):
    """``os.fdopen`` that closes the raw fd when the file object cannot be built."""
    try:
        return os.fdopen(fd, "wb")
    except BaseException:
        os.close(fd)
        raise


def _expect_unchanged(section: StrictSection, name: str,
                      expect: Tuple[bytes, Tuple[int, int]]) -> None:
    """Last check before replacing a file this module owns by name: it is still the trusted inode
    with the bytes verified earlier. Cooperating writers cannot interleave (they hold the section);
    this narrows a non-cooperating same-owner writer to the instant between check and rename."""
    if _read_verified(section, name, _PRIVATE_MODE) != expect:
        raise CronStoreError(f"{section.parent / name} changed before it could be replaced; it is left untouched")


def _publish_file(section: StrictSection, name: str, payload: bytes, mode: int,
                  expect: Optional[Tuple[bytes, Tuple[int, int]]] = None) -> None:
    """Stage beside ``name``, verify mode + owner, fsync, re-verify the section (and ``expect``,
    the verified file being replaced), replace, re-verify the section, fsync the directory. No
    copy or in-place fallback. Any escape is typed: CronStoreError (unchanged),
    CronStoreUncertainError, or a BaseException carrying ``cron_store_outcome``."""
    dir_fd = section.dir_fd
    tmp: Optional[str] = None
    ident: Optional[Tuple[int, int]] = None
    replaced = False
    try:
        try:
            fd, tmp, ident = _create_staging(dir_fd)
            with _fdopen_wb(fd) as f:
                f.write(payload)
                f.flush()
                _stage_mode_and_owner(f.fileno(), mode, section.owner, section.parent / name)
                os.fsync(f.fileno())
            verify_section(section)
            if expect is not None:
                _expect_unchanged(section, name, expect)
            os.replace(tmp, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        except OSError as exc:
            raise CronStoreError(
                f"{section.parent / name} was not replaced; the previous version is intact "
                f"({exc.strerror or type(exc).__name__})") from exc
        replaced = True
        verify_section(section)  # the directory we committed into is still the store's
        try:
            _fsync_dir(dir_fd)
        except OSError as exc:
            raise CronStoreUncertainError(
                f"{section.parent / name} was renamed into place but the directory fsync failed "
                f"({exc.strerror or type(exc).__name__}); the new version is visible but may not "
                "survive a crash. Check it after any power loss before relying on it.") from exc
    except BaseException as exc:
        # A signal can land after os.replace returned but before ``replaced`` was set: the
        # staged name no longer names our inode exactly when the rename happened.
        committed = replaced
        if not committed and ident is not None:
            with contextlib.suppress(OSError):
                st = _lstat_at(dir_fd, tmp)
                committed = st is None or (st.st_dev, st.st_ino) != ident
        if not committed:
            _unlink_own(dir_fd, tmp, ident)
            _mark_outcome(exc, OUTCOME_UNCHANGED, name)
            raise
        if isinstance(exc, Exception) and not isinstance(exc, CronStoreUncertainError):
            raise CronStoreUncertainError(
                f"{section.parent / name} was renamed into place, then publication failed "
                f"({exc}); the new version may not be where readers look and its durability is "
                "unknown. Read the store back before retrying.") from exc
        _mark_outcome(exc, OUTCOME_UNCERTAIN, name)
        raise


def _publish_new_file(section: StrictSection, name: str, payload: bytes) -> None:
    """No-clobber publication of an owner-only file: complete + fsynced under a staging name for
    this exact file, linked into place (never replaces), staging link removed, directory fsynced.
    A leftover staging link from an interrupted attempt is removed only after it is proven to be
    this module's file with exactly ``payload``; anything else is refused and left in place."""
    dir_fd = section.dir_fd
    staging = f".{name}.staging"
    leftover = _lstat_at(dir_fd, staging)
    if leftover is not None:
        done = _lstat_at(dir_fd, name)
        linked = done is not None and (done.st_dev, done.st_ino) == (leftover.st_dev, leftover.st_ino)
        found = _read_verified(section, staging, _PRIVATE_MODE, links=2 if linked else 1)
        if found is None or found[0] != payload:
            raise CronStoreError(f"{staging} holds other bytes; it is left untouched")
        _unlink_own(dir_fd, staging, found[1])
        _fsync_dir(dir_fd)
    if _lstat_at(dir_fd, name) is not None:
        return
    ident: Optional[Tuple[int, int]] = None
    try:
        fd, _staging, ident = _create_staging(dir_fd, staging)
        with _fdopen_wb(fd) as f:
            f.write(payload)
            f.flush()
            _stage_mode_and_owner(f.fileno(), _PRIVATE_MODE, section.owner, section.parent / name)
            os.fsync(f.fileno())
        verify_section(section)
        with contextlib.suppress(FileExistsError):
            os.link(staging, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd, follow_symlinks=False)
    finally:
        _unlink_own(dir_fd, staging, ident)
    _fsync_dir(dir_fd)


# --- Latch --------------------------------------------------------------------------------------

def _latch_found(section: StrictSection) -> Optional[Tuple[bytes, Tuple[int, int]]]:
    """``(content, (dev, ino))`` of a trusted latch, None when absent; anything else refuses."""
    found = _read_verified(section, section.name + LATCH_SUFFIX, _PRIVATE_MODE)
    if found is None or found[0] in (_LATCH_ARMED, _LATCH_PUBLISHED):
        return found
    raise CronStoreError(f"{latch_path(section.target)} has unexpected content; it is left untouched. {_RECOVERY}")


def _latch_state(section: StrictSection) -> Optional[bytes]:
    found = _latch_found(section)
    return None if found is None else found[0]


def ensure_latched(section: StrictSection) -> None:
    """Record the opt-in durably before this section can mutate anything."""
    if _latch_state(section) is not None:
        return
    published = (_lstat_at(section.dir_fd, section.name) is not None
                 or _lstat_at(section.dir_fd, section.name + LAST_GOOD_SUFFIX) is not None)
    _publish_file(section, section.name + LATCH_SUFFIX,
                  _LATCH_PUBLISHED if published else _LATCH_ARMED, _PRIVATE_MODE)
    logger.info("Latched %s to cron.store.strict_durability", section.target)


def _rearm_latch(section: StrictSection) -> None:
    """After a definitive failure of a store's FIRST publication nothing exists to protect, so
    the latch may again allow a first write — but only over the latch this module wrote, re-proven
    (same inode, same bytes) at the last instant before the rename. A missing, tampered or
    untrusted latch is never overwritten; the store then refuses until recovered."""
    if (_lstat_at(section.dir_fd, section.name) is not None
            or _lstat_at(section.dir_fd, section.name + LAST_GOOD_SUFFIX) is not None):
        return
    try:
        found = _latch_found(section)
        if found is None or found[0] == _LATCH_ARMED:
            return
        _publish_file(section, section.name + LATCH_SUFFIX, _LATCH_ARMED, _PRIVATE_MODE, expect=found)
    except (OSError, CronStoreError) as exc:
        logger.error("Could not re-arm %s after a failed first publication (%s); cron refuses "
                     "this store until it is recovered manually", latch_path(section.target), exc)


# --- Store operations -----------------------------------------------------------------------------

def _store_mode(section: StrictSection) -> int:
    """0600 normally. Managed/container installs keep an existing store's permission bits (an
    activation script may set 0640) — the same predicates ``hermes_cli.config._secure_file``
    skips on (``is_managed`` and ``hermes_constants._container_or_chmod_skipped``). A first
    write there is still 0600."""
    from hermes_cli import config
    from hermes_constants import _container_or_chmod_skipped

    if not (config.is_managed() or _container_or_chmod_skipped()):
        return _PRIVATE_MODE
    st = _lstat_at(section.dir_fd, section.name)
    return stat.S_IMODE(st.st_mode) & 0o777 if st is not None and stat.S_ISREG(st.st_mode) else _PRIVATE_MODE


def _preserve_forensic_copy(section: StrictSection, raw: bytes) -> Path:
    """Durable, owner-only, single-link exact copy of ``raw``; an existing snapshot is reused
    only when it passes the same checks and holds the same bytes (it is fsynced again)."""
    snapshot = forensic_path(section.target, raw)
    _publish_new_file(section, snapshot.name, raw)
    existing = _read_verified(section, snapshot.name, _PRIVATE_MODE, sync=True)
    if existing is None or existing[0] != raw:
        raise CronStoreError(f"{snapshot.name} exists with different bytes and is never overwritten")
    _fsync_dir(section.dir_fd)
    return snapshot


def _refuse_noncanonical(section: StrictSection, raw: bytes, reason: str):
    target = section.target
    try:
        where = f"its exact bytes are preserved in {_preserve_forensic_copy(section, raw).name}"
    except (OSError, CronStoreError) as exc:
        cause = exc if isinstance(exc, CronStoreError) else (exc.strerror or type(exc).__name__)
        where = (f"its exact bytes could NOT be preserved ({cause}); copy {target.name} aside "
                 "before changing anything")
    except BaseException as exc:
        _mark_outcome(exc, OUTCOME_UNCHANGED, target.name)
        raise
    logger.error("Refusing to use %s: %s; file left untouched and %s", target, reason, where)
    raise CronStoreError(
        f"{target} is corrupt or noncanonical ({reason}); strict durability leaves it untouched "
        f"and {where}. {_RECOVERY}")


def read_canonical(section: Optional[StrictSection], jobs_file: Path) -> Optional[bytes]:
    """Validated store bytes under the held section, or None for a store never published.
    Raises before any forgiving parse, repair or content log can happen."""
    if section is None:
        raise _not_held(jobs_file)
    verify_section(section)
    target = section.target
    try:
        raw = _read_regular(section, section.name)
    except OSError as exc:
        raise CronStoreError(f"Cannot read {target}: {exc.strerror or type(exc).__name__}") from exc
    if raw is None:
        backup = last_good_path(target)
        if _lstat_at(section.dir_fd, backup.name) is not None or _latch_state(section) == _LATCH_PUBLISHED:
            raise CronStoreError(
                f"{target} is missing but this store was published before ({backup.name} or "
                f"{latch_path(target).name} says so); refusing to read or start an empty store over "
                f"a lost one. Nothing is restored automatically. {_RECOVERY}")
        return None
    reason = noncanonical_reason(raw)
    if reason:
        _refuse_noncanonical(section, raw, reason)
    return raw


def check_before_mutation(section: Optional[StrictSection], jobs_file: Path) -> None:
    read_canonical(section, jobs_file)


def publish(section: Optional[StrictSection], jobs_file: Path, payload: bytes) -> Optional[str]:
    """Durably publish ``payload`` as the primary, then as its last-good copy (a separate file,
    never a hard link), refreshed only after the primary is durable. Returns None, or an
    actionable warning when the primary is durable but last-good could not be refreshed (it
    stays at the previous good version and the next successful save refreshes it). Raises only
    when the primary is unchanged (CronStoreError) or genuinely uncertain."""
    if section is None:
        raise _not_held(jobs_file)
    target = section.target
    mode = _store_mode(section)
    first = _lstat_at(section.dir_fd, section.name) is None
    try:
        latch = _latch_found(section)
        if latch is None or latch[0] != _LATCH_PUBLISHED:
            _publish_file(section, section.name + LATCH_SUFFIX, _LATCH_PUBLISHED, _PRIVATE_MODE,
                          expect=latch)
    except CronStoreError as exc:
        if first:
            _rearm_latch(section)
        raise CronStoreError(f"{target} was not changed: recording {latch_path(target).name} failed ({exc})") from exc
    except BaseException as exc:
        _mark_outcome(exc, OUTCOME_UNCHANGED, target.name)
        raise
    try:
        _publish_file(section, section.name, payload, mode)
    except CronStoreUncertainError:
        raise
    except CronStoreError:
        if first:
            _rearm_latch(section)
        raise
    backup = last_good_path(target)
    try:
        _publish_file(section, backup.name, payload, mode)
    except CronStoreError as exc:
        warning = (
            f"{target.name} was saved durably, but refreshing {backup.name} failed ({exc}); "
            f"{backup.name} still holds the previous good version"
            + (" (none: this was the store's first save, so there is no backup yet)" if first else "")
            + " and the next successful save refreshes it. Do not re-create the job.")
        logger.warning("%s", warning)
        return warning
    except BaseException as exc:
        _mark_outcome(exc, OUTCOME_UNCERTAIN, target.name)  # the primary is already committed
        raise
    return None
