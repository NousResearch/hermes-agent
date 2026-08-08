"""Choose which Claude Code login a Hermes job runs on.

Claude Code keeps one login per configuration directory. A person selects the
directory with ``CLAUDE_CONFIG_DIR`` and the matching secret store with
``CLAUDE_SECURESTORAGE_CONFIG_DIR``. This module reads a list of such
directories from ``config.yaml``, asks each account how much of its plan it
has used, and names the one a new job must run on.

Three properties keep this safe:

* Hermes never holds a Claude Code token. It sets two directory variables and
  lets the ``claude`` program read its own secret. The only token this module
  touches is the short-lived one it needs to ask for a usage number, and that
  token stays in memory.
* Hermes never spends paid usage past the plan. Only the plan windows count
  toward the decision.
* A conversation that already started on one account stays on that account.

The feature is off until a person configures two or more profiles.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import time
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Union

import httpx

logger = logging.getLogger(__name__)

CONFIG_SECTION = "claude_cli_profiles"
DEFAULT_STOP_AT_PERCENT = 95.0

# The plan-usage endpoint. It answers with numbers only. It starts no model.
USAGE_URL = "https://api.anthropic.com/api/oauth/usage"
USAGE_TIMEOUT_SECONDS = 15.0

# How long one profile's usage numbers stay good. A job that starts several
# child processes in a row then asks the endpoint once, not once per child.
USAGE_CACHE_SECONDS = 60.0

# Two clocks, on purpose.
#
# ``_clock`` is monotonic. It measures the age of the in-memory usage cache,
# which lives for one minute and must not be confused by a wall-clock jump.
#
# ``_wall_clock`` is the ordinary clock. It stamps a conversation's last use in
# the state file. That file outlives the process and the machine: a monotonic
# value written before a restart is meaningless after one, and every old pin
# would then look newer than every new one, so a trim would drop the live
# conversation and keep dead ones.
_clock = time.monotonic
_wall_clock = time.time

# name -> (read_at, ProfileUsage). It holds percentages and reset times only.
# It never holds a token: ``read_profile_usage`` caches the parsed record, and
# the token it used goes out of scope as soon as the call returns.
_usage_cache: dict = {}


def invalidate_usage_cache(name: Optional[str] = None) -> None:
    """Forget cached numbers, for one profile nickname or for every profile."""
    if name is None:
        _usage_cache.clear()
        return
    for key in [k for k in _usage_cache if k[0] == name]:
        _usage_cache.pop(key, None)

# Reasons a profile's usage could not be read. Two of them mean a person must
# act; the rest mean "ask again later".
PROBLEM_NO_LOGIN = "no_login"
PROBLEM_LOGIN_REJECTED = "login_rejected"
PROBLEM_STALE_LOGIN = "stale_login"
PROBLEM_UNREADABLE = "unreadable_usage"
PROBLEM_UNREACHABLE = "unreachable"

_NEEDS_A_PERSON = (PROBLEM_NO_LOGIN, PROBLEM_LOGIN_REJECTED)


@dataclass(frozen=True)
class ClaudeProfile:
    """One Claude Code login, named by a person and held in one directory."""

    name: str
    config_dir: Path
    securestorage_dir: Path


def _config_section(config: Optional[dict] = None) -> dict:
    if config is None:
        try:
            from hermes_cli.config import load_config_readonly

            config = load_config_readonly()
        except Exception:
            logger.debug("Could not read config.yaml for Claude profiles", exc_info=True)
            return {}
    section = (config or {}).get(CONFIG_SECTION)
    return section if isinstance(section, dict) else {}


def _as_dir(value: Any) -> Optional[Path]:
    """Turn a configured directory into one absolute path.

    A relative path in ``config.yaml`` would otherwise name a different
    directory for every working directory a child process starts in, and the
    child would then read a different account. Resolve it once, here.
    """
    text = str(value or "").strip()
    if not text:
        return None
    return Path(os.path.abspath(os.path.expanduser(text)))


def load_profiles(config: Optional[dict] = None) -> list[ClaudeProfile]:
    """Return the configured profiles, in the order a person wrote them.

    An entry with no name or no directory is dropped. A repeated name keeps
    the first entry, so one name always means one directory.
    """
    entries = _config_section(config).get("profiles")
    if not isinstance(entries, list):
        return []

    profiles: list[ClaudeProfile] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        config_dir = _as_dir(entry.get("config_dir"))
        if not name or config_dir is None or name in seen:
            continue
        seen.add(name)
        profiles.append(
            ClaudeProfile(
                name=name,
                config_dir=config_dir,
                securestorage_dir=_as_dir(entry.get("securestorage_dir")) or config_dir,
            )
        )
    return profiles


def switching_enabled(config: Optional[dict] = None) -> bool:
    """True only when a person configured two or more profiles."""
    return len(load_profiles(config)) >= 2


def stop_at_percent(config: Optional[dict] = None) -> float:
    """The used-percentage at which a usage window counts as full."""
    raw = _config_section(config).get("stop_at_percent", DEFAULT_STOP_AT_PERCENT)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_STOP_AT_PERCENT
    if value != value:  # not a number
        return DEFAULT_STOP_AT_PERCENT
    return min(100.0, max(1.0, value))


# ---------------------------------------------------------------------------
# Reading how much of a plan an account has used.
# ---------------------------------------------------------------------------


class ProfileUsageError(Exception):
    """The usage read failed. *kind* says which of the reasons above applies."""

    def __init__(self, kind: str, message: str = ""):
        super().__init__(message or kind)
        self.kind = kind


@dataclass(frozen=True)
class StoredLogin:
    """A token a profile holds now, and the moment it stops being valid.

    The token stays in memory. Nothing writes it to a file, a log line, or a
    command line. ``__repr__`` hides it, because a dataclass would otherwise
    print it inside any error message that shows the object.
    """

    token: str
    expires_at_ms: int = 0

    def __repr__(self) -> str:  # pragma: no cover — exercised through tests
        return f"StoredLogin(token=<hidden>, expires_at_ms={self.expires_at_ms})"

    def is_expired(self, *, buffer_seconds: int = 60) -> bool:
        if not self.expires_at_ms:
            return False
        return time.time() * 1000 >= (self.expires_at_ms - buffer_seconds * 1000)


@dataclass(frozen=True)
class ProfileUsage:
    """What one account has used, as percentages of its plan windows."""

    name: str
    five_hour_percent: Optional[float] = None
    weekly_percent: Optional[float] = None
    opus_weekly_percent: Optional[float] = None
    five_hour_reset: Optional[datetime] = None
    weekly_reset: Optional[datetime] = None
    opus_weekly_reset: Optional[datetime] = None
    problem: Optional[str] = None

    @property
    def needs_a_person(self) -> bool:
        return self.problem in _NEEDS_A_PERSON

    @property
    def windows(self) -> tuple:
        """Every plan window, as (label, percent, reopen time).

        The weekly Opus window fills on its own, ahead of the whole-week
        window, so it counts as an exhaustion window in its own right.
        """
        return (
            ("five-hour", self.five_hour_percent, self.five_hour_reset),
            ("weekly", self.weekly_percent, self.weekly_reset),
            ("weekly Opus", self.opus_weekly_percent, self.opus_weekly_reset),
        )

    @property
    def worst_percent(self) -> Optional[float]:
        known = [p for _label, p, _reset in self.windows if p is not None]
        return max(known) if known else None


class _Unreadable(Exception):
    """A usage field held something that is not a number."""


def _percent(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _Unreadable(repr(value))
    return float(value)


def _utilization_percent(value: Any) -> Optional[float]:
    """Read the older ``utilization`` field, which holds a fraction.

    That field runs from 0 to 1, so 0.87 there means 87 percent. The newer
    ``percent`` field is already a percentage and goes through
    :func:`_percent` untouched, so 0.87 there stays 0.87 percent. Keeping the
    two readers apart is what stops a barely-used account reading as nearly
    full, and a nearly-full one reading as barely used.
    """
    number = _percent(value)
    if number is None:
        return None
    return number * 100 if 0 <= number <= 1 else number


# A timestamp larger than this is milliseconds, not seconds. The boundary sits
# far past any date a plan window reopens on, and far below the millisecond
# spelling of the same moment.
_MILLISECOND_BOUNDARY = 1e11


def _reset_time(value: Any) -> Optional[datetime]:
    """Read a reopen time. Anything unreadable gives None, never an error.

    The endpoint has sent this field as an ISO 8601 string, as seconds since
    the epoch, and as milliseconds since the epoch. A wrong value must not
    stop the status report, so every failure returns None.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            if math.isnan(value) or math.isinf(value):
                return None
            seconds = float(value)
            if abs(seconds) > _MILLISECOND_BOUNDARY:
                seconds /= 1000.0
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (ValueError, OverflowError, OSError, TypeError):
            return None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (ValueError, OverflowError, OSError, TypeError):
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def _is_opus_scope(entry: dict) -> bool:
    """True when a scoped window belongs to the Opus model family."""
    scope = entry.get("scope")
    if not isinstance(scope, dict):
        return False
    model = scope.get("model")
    name = model.get("display_name") if isinstance(model, dict) else model
    return "opus" in str(name or "").lower()


def parse_usage_payload(name: str, payload: Any) -> ProfileUsage:
    """Turn one usage reply into a record of the plan windows.

    The endpoint answers in two shapes. The current shape holds a ``limits``
    list whose entries carry a ``kind`` and a ``percent``. The older shape
    holds ``five_hour``, ``seven_day``, and ``seven_day_opus`` objects with a
    ``utilization`` field. This reads whichever is present.

    A field that holds something other than a number makes the whole record
    unreadable. It never becomes a zero: a zero would read as a wide-open
    account and would send work to an account that is in fact full.
    """
    if not isinstance(payload, dict):
        return ProfileUsage(name=name, problem=PROBLEM_UNREADABLE)

    five_hour = weekly = opus = None
    five_hour_reset = weekly_reset = opus_reset = None
    try:
        limits = payload.get("limits")
        if limits is not None and not isinstance(limits, list):
            return ProfileUsage(name=name, problem=PROBLEM_UNREADABLE)
        for entry in limits or []:
            if not isinstance(entry, dict):
                continue
            kind = str(entry.get("kind") or "")
            if kind == "session" and five_hour is None:
                five_hour = _percent(entry.get("percent"))
                five_hour_reset = _reset_time(entry.get("resets_at"))
            elif kind == "weekly_all" and weekly is None:
                weekly = _percent(entry.get("percent"))
                weekly_reset = _reset_time(entry.get("resets_at"))
            elif kind == "weekly_scoped" and opus is None and _is_opus_scope(entry):
                opus = _percent(entry.get("percent"))
                opus_reset = _reset_time(entry.get("resets_at"))

        for key, current in (("five_hour", five_hour), ("seven_day", weekly),
                             ("seven_day_opus", opus)):
            if current is not None:
                continue
            window = payload.get(key)
            if not isinstance(window, dict):
                continue
            value = _utilization_percent(window.get("utilization"))
            reset = _reset_time(window.get("resets_at"))
            if key == "five_hour":
                five_hour, five_hour_reset = value, reset
            elif key == "seven_day":
                weekly, weekly_reset = value, reset
            else:
                opus, opus_reset = value, reset
    except _Unreadable as exc:
        logger.debug("Claude profile %s: unreadable usage field %s", name, exc)
        return ProfileUsage(name=name, problem=PROBLEM_UNREADABLE)

    if five_hour is None and weekly is None and opus is None:
        return ProfileUsage(name=name, problem=PROBLEM_UNREADABLE)

    return ProfileUsage(
        name=name,
        five_hour_percent=five_hour,
        weekly_percent=weekly,
        opus_weekly_percent=opus,
        five_hour_reset=five_hour_reset,
        weekly_reset=weekly_reset,
        opus_weekly_reset=opus_reset,
    )


def fetch_usage(token: str) -> dict:
    """Ask the usage endpoint for one account's numbers.

    This is one HTTP GET. It sends no prompt and it starts no model.
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "anthropic-beta": "oauth-2025-04-20",
        "User-Agent": "claude-code/2.1.0",
    }
    try:
        with httpx.Client(timeout=USAGE_TIMEOUT_SECONDS) as client:
            response = client.get(USAGE_URL, headers=headers)
    except Exception as exc:
        raise ProfileUsageError(PROBLEM_UNREACHABLE, str(exc)) from exc

    status = getattr(response, "status_code", 0)
    if status in (401, 403):
        raise ProfileUsageError(PROBLEM_LOGIN_REJECTED, f"HTTP {status}")
    try:
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise ProfileUsageError(PROBLEM_UNREACHABLE, str(exc)) from exc
    if not isinstance(payload, dict):
        raise ProfileUsageError(PROBLEM_UNREADABLE, "the reply is not an object")
    return payload


def read_profile_login(profile: ClaudeProfile) -> Optional[StoredLogin]:
    """Return the login one profile holds, or None when it holds none.

    Hermes reads this only to ask for a usage number. It does not copy the
    token anywhere, and the ``claude`` program still reads its own secret when
    it runs.
    """
    from agent.anthropic_adapter import (
        _read_claude_code_credentials_from_file,
        _read_claude_code_credentials_from_keychain,
    )

    creds = None
    try:
        creds = _read_claude_code_credentials_from_keychain(profile.config_dir)
    except Exception:
        logger.debug("Claude profile %s: keychain read failed", profile.name, exc_info=True)
    if not (creds and creds.get("accessToken")):
        try:
            creds = _read_claude_code_credentials_from_file(
                profile.config_dir, profile.securestorage_dir
            )
        except Exception:
            logger.debug("Claude profile %s: file read failed", profile.name, exc_info=True)
            creds = None
    if not (creds and creds.get("accessToken")):
        return None
    expires_at = creds.get("expiresAt") or 0
    if not isinstance(expires_at, (int, float)) or isinstance(expires_at, bool):
        expires_at = 0
    return StoredLogin(token=str(creds["accessToken"]), expires_at_ms=int(expires_at))


def read_profile_usage(
    profile: ClaudeProfile,
    *,
    token_reader: Optional[Callable[[ClaudeProfile], Any]] = None,
    usage_fetcher: Optional[Callable[[str], dict]] = None,
) -> ProfileUsage:
    """Read one profile's plan usage. No model runs and no prompt is sent.

    A good read stays cached for ``USAGE_CACHE_SECONDS``, so a job that starts
    several child processes asks the endpoint once. A failed read is never
    cached, so the next attempt tries again straight away.
    """
    cache_key = (profile.name, str(profile.config_dir))
    cached = _usage_cache.get(cache_key)
    if cached is not None:
        read_at, record = cached
        if (_clock() - read_at) < USAGE_CACHE_SECONDS:
            return record

    reader = token_reader or read_profile_login
    fetcher = usage_fetcher or fetch_usage

    login: Union[StoredLogin, str, None] = reader(profile)
    if isinstance(login, str):
        login = StoredLogin(token=login) if login.strip() else None
    if login is None or not login.token:
        return ProfileUsage(name=profile.name, problem=PROBLEM_NO_LOGIN)
    if login.is_expired():
        # Claude Code refreshes its own token when it starts, so a stale token
        # says "ask again later", not "this account is finished".
        return ProfileUsage(name=profile.name, problem=PROBLEM_STALE_LOGIN)

    try:
        # The read AND the parse both sit inside this guard. A surprise in
        # either one becomes a reported problem, never a stopped job.
        record = parse_usage_payload(profile.name, fetcher(login.token))
    except ProfileUsageError as exc:
        return ProfileUsage(name=profile.name, problem=exc.kind)
    except Exception as exc:
        logger.debug("Claude profile %s: usage read failed: %s", profile.name, exc)
        return ProfileUsage(name=profile.name, problem=PROBLEM_UNREACHABLE)

    if record.problem is None:
        # Only good numbers are cached, and the record holds no token.
        _usage_cache[cache_key] = (_clock(), record)
    return record


# ---------------------------------------------------------------------------
# Choosing a profile.
# ---------------------------------------------------------------------------

# How usable one account is right now.
#
# Only OPEN is selectable. UNKNOWN fails closed on purpose: an account whose
# usage Hermes could not read is an account whose identity Hermes could not
# confirm either, and running it would risk billing the wrong subscription
# without saying so. A wait a person can see beats a charge they cannot.
OPEN = "open"          # every window Hermes could read is below the threshold
UNKNOWN = "unknown"    # the read failed, so nothing about it is confirmed
FULL = "full"          # a window reached the threshold
BLOCKED = "blocked"    # a person must sign this profile in

# Why the selector returned what it returned.
REASON_KEPT_ACTIVE = "kept_active"
REASON_SWITCHED = "switched"
REASON_FIRST_RUN = "first_run"
REASON_PINNED = "pinned"
REASON_NONE_AVAILABLE = "none_available"


@dataclass(frozen=True)
class Selection:
    """The profile a job must run on, and why."""

    profile: Optional[ClaudeProfile]
    reason: str
    available: bool
    message: str = ""
    usage: dict = None  # type: ignore[assignment]  # name -> ProfileUsage

    def env(self) -> dict:
        """The two directory variables the child process needs."""
        return profile_env(self.profile) if self.profile else {}


def profile_env(profile: Optional[ClaudeProfile]) -> dict:
    """Return the environment entries that point Claude Code at *profile*."""
    if profile is None:
        return {}
    return {
        "CLAUDE_CONFIG_DIR": str(profile.config_dir),
        "CLAUDE_SECURESTORAGE_CONFIG_DIR": str(profile.securestorage_dir),
    }


def usability(usage: ProfileUsage, threshold: float) -> str:
    """Say how usable one account is, against the stop percentage."""
    if usage.needs_a_person:
        return BLOCKED
    if usage.problem:
        return UNKNOWN
    worst = usage.worst_percent
    if worst is None:
        return UNKNOWN
    return FULL if worst >= threshold else OPEN


def _format_time(when: Optional[datetime]) -> str:
    return when.strftime("%Y-%m-%d %H:%M UTC") if when else ""


def describe_wait(usage: ProfileUsage, threshold: float) -> str:
    """One sentence that says why this account is unavailable, and until when.

    It names the profile by the nickname a person chose. It never names an
    address, an account number, or an organisation.
    """
    if usage.problem in _NEEDS_A_PERSON:
        return f"{usage.name}: sign in with `claude auth login` on this profile."
    if usage.problem:
        return (
            f"{usage.name}: its usage could not be checked ({usage.problem}), "
            "so Hermes did not run on it."
        )
    parts = []
    for label, percent, reset in usage.windows:
        if percent is not None and percent >= threshold:
            when = _format_time(reset)
            parts.append(
                f"{label} window is full ({percent:.0f}%)"
                + (f", it reopens {when}" if when else ", the reopen time is unknown")
            )
    if not parts:
        return f"{usage.name}: unavailable."
    return f"{usage.name}: " + "; ".join(parts) + "."


def select_profile(
    profiles: list[ClaudeProfile],
    usages: dict,
    *,
    threshold: float,
    active_name: Optional[str] = None,
) -> Selection:
    """Name the profile a new job must run on.

    The order is:

    1. Keep the account the work is already on, while it is open. A lower
       number on another account is not a reason to move: a move costs the
       child process its whole warm context.
    2. Otherwise take the first configured account that is open.
    3. Otherwise report why, and return no profile. An account whose usage
       Hermes could not read is never selected: the read is also the check
       that the account answers as itself, so running it could bill a
       subscription the person did not choose.
    """
    by_name = {p.name: p for p in profiles}
    states = {
        name: usability(usages.get(name) or ProfileUsage(name=name, problem=PROBLEM_UNREADABLE), threshold)
        for name in by_name
    }

    # A recorded name that no longer names a configured profile still means
    # the work was running somewhere. Moving it is a switch, not a first run.
    had_active = bool(active_name)
    active = by_name.get(active_name or "")
    if active is not None and states[active.name] == OPEN:
        return Selection(
            profile=active,
            reason=REASON_KEPT_ACTIVE,
            available=True,
            usage=usages,
        )

    for candidate in profiles:
        if states[candidate.name] == OPEN:
            return Selection(
                profile=candidate,
                reason=REASON_SWITCHED if had_active else REASON_FIRST_RUN,
                available=True,
                usage=usages,
            )

    waits = [
        describe_wait(usages.get(p.name) or ProfileUsage(name=p.name, problem=PROBLEM_UNREADABLE), threshold)
        for p in profiles
    ]
    return Selection(
        profile=None,
        reason=REASON_NONE_AVAILABLE,
        available=False,
        message="No Claude Code profile is available. " + " ".join(waits),
        usage=usages,
    )


# ---------------------------------------------------------------------------
# The state file.
#
# It records the account the work is on and the account each conversation
# started on. It holds no token, no address, and no account number — only the
# nicknames a person chose and a timestamp.
# ---------------------------------------------------------------------------

STATE_FILE_NAME = "claude_cli_profiles.json"
STATE_LOCK_NAME = "claude_cli_profiles.json.lock"
STATE_VERSION = 1
MAX_PINNED_SESSIONS = 500

# A stored session key is the fingerprint below: 32 lowercase hexadecimal
# characters. An older file held the raw chat identifier. Those entries are
# dropped on read, so a chat name never survives an upgrade.
_FINGERPRINT_LENGTH = 32

try:  # POSIX
    import fcntl
except ImportError:  # pragma: no cover — Windows
    fcntl = None  # type: ignore[assignment]
try:  # Windows
    import msvcrt
except ImportError:  # pragma: no cover — POSIX
    msvcrt = None  # type: ignore[assignment]


def session_fingerprint(session_id: str) -> str:
    """Return a stable, non-reversible name for one conversation.

    Hermes stores this instead of the conversation identifier, because that
    identifier carries a platform, a chat, and often a person. The same
    conversation always gives the same fingerprint, so a resume still finds
    its own account.
    """
    key = str(session_id or "").strip()
    if not key:
        return ""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:_FINGERPRINT_LENGTH]


def _is_fingerprint(key: str) -> bool:
    return (
        len(key) == _FINGERPRINT_LENGTH
        and all(character in "0123456789abcdef" for character in key)
    )


def state_path() -> Path:
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home()) / STATE_FILE_NAME


def _lock_path() -> Path:
    """The lock file. It is created once and never deleted.

    Deleting it while a process holds it would unlink the inode every waiter
    is queued on, and the next writer would create a second, unrelated lock.
    Two writers would then edit the file at the same time.
    """
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home()) / STATE_LOCK_NAME


def _empty_state() -> dict:
    return {"version": STATE_VERSION, "active": None, "sessions": {}}


@contextlib.contextmanager
def _state_lock():
    """Serialize read-modify-write cycles on the state file across processes."""
    lock_path = _lock_path()
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        yield
        return
    if fcntl is None and msvcrt is None:  # pragma: no cover — no lock available
        yield
        return
    try:
        if msvcrt and (not lock_path.exists() or lock_path.stat().st_size == 0):
            lock_path.write_text(" ", encoding="utf-8")
        handle = open(lock_path, "r+" if msvcrt else "a+", encoding="utf-8")
    except OSError:
        # No lock file means no lock. One unserialized write beats a stopped
        # job, and every write is atomic on its own.
        logger.debug("Could not open the Claude profile lock file", exc_info=True)
        yield
        return
    locked = False
    try:
        try:
            if fcntl:
                fcntl.flock(handle, fcntl.LOCK_EX)
            else:  # pragma: no cover — Windows
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            locked = True
        except OSError:
            # Windows raises when another process holds the region, and it
            # gives up after a fixed number of tries. Go on without the lock
            # rather than stop the job.
            logger.debug("Could not take the Claude profile lock", exc_info=True)
        yield
    finally:
        if locked:
            try:
                if fcntl:
                    fcntl.flock(handle, fcntl.LOCK_UN)
                else:  # pragma: no cover — Windows
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        handle.close()


def read_state() -> dict:
    """Read the state file. A missing or damaged file reads as empty.

    A damaged file must never stop a job. The worst it costs is one forgotten
    pin, and the next write repairs the file.
    """
    path = state_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _empty_state()
    if not isinstance(raw, dict):
        return _empty_state()
    state = _empty_state()
    active = raw.get("active")
    if isinstance(active, str) and active:
        state["active"] = active
    sessions = raw.get("sessions")
    if isinstance(sessions, dict):
        state["sessions"] = {
            str(key): value
            for key, value in sessions.items()
            # An entry that is not a fingerprint came from an older file and
            # holds a readable chat identifier. Drop it.
            if _is_fingerprint(str(key))
            and isinstance(value, dict)
            and isinstance(value.get("profile"), str)
        }
    return state


def _write_state(state: dict) -> None:
    """Write the state file at mode 0o600, and replace it in one step."""
    from hermes_constants import secure_parent_dir

    path = state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        secure_parent_dir(path)
        tmp = path.with_suffix(f".json.tmp.{os.getpid()}")
        descriptor = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except OSError:
        logger.debug("Could not write the Claude profile state file", exc_info=True)


def _update_state(change: Callable[[dict], None]) -> None:
    with _state_lock():
        state = read_state()
        change(state)
        _write_state(state)


def record_active(name: Optional[str]) -> None:
    """Record the account new work runs on now.

    This is the fallback slot, for a job with no conversation bound to it —
    the command line, a cron job, a one-off script. A conversation carries its
    own account; see :func:`pin_session`.
    """
    _update_state(lambda state: state.__setitem__("active", name or None))


def active_profile_name() -> Optional[str]:
    return read_state().get("active")


def _recorded_at(entry: Any) -> float:
    """The moment one pin was last used. A damaged value counts as the oldest.

    The file can hold anything: a hand edit, a half-written record, an entry an
    older version wrote. Sorting on such a value would raise and stop the
    write. Zero is the safe reading — it makes the damaged entry the first one
    a trim drops, and a real conversation keeps its own real timestamp.
    """
    value = entry.get("at") if isinstance(entry, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    number = float(value)
    return 0.0 if number != number else number  # a not-a-number value is oldest


def _trim_sessions(sessions: dict) -> dict:
    if len(sessions) <= MAX_PINNED_SESSIONS:
        return sessions
    newest = sorted(sessions.items(), key=lambda item: _recorded_at(item[1]), reverse=True)
    return dict(newest[:MAX_PINNED_SESSIONS])


def pin_session(session_id: str, name: str) -> None:
    """Tie one conversation to the account that started it.

    A resume must never move to another account. Claude Code keeps its
    conversation record inside the profile directory, so the same identifier
    on another account either fails or starts fresh work and loses the first
    conversation.

    Calling this again for a live conversation refreshes its timestamp, so a
    busy conversation is never the one a trim drops.
    """
    fingerprint = session_fingerprint(session_id)
    if not fingerprint or not name:
        return

    def change(state: dict) -> None:
        sessions = state.get("sessions") or {}
        sessions[fingerprint] = {"profile": name, "at": _wall_clock()}
        state["sessions"] = _trim_sessions(sessions)

    _update_state(change)


def _session_entry(session_id: str) -> Optional[dict]:
    fingerprint = session_fingerprint(session_id)
    if not fingerprint:
        return None
    entry = read_state().get("sessions", {}).get(fingerprint)
    return entry if isinstance(entry, dict) else None


def pinned_profile_name(session_id: str) -> Optional[str]:
    entry = _session_entry(session_id)
    return entry.get("profile") if entry else None


def pin_recorded_at(session_id: str) -> float:
    """When this conversation last used its account. 0 when it has none."""
    return _recorded_at(_session_entry(session_id))


# The account chosen for the work running in this context. A ContextVar keeps
# two conversations apart inside one process, the same way the gateway keeps
# their session identities apart.
_SELECTED_PROFILE: ContextVar = ContextVar("HERMES_CLAUDE_PROFILE", default=None)


def bind_selected_profile(name: Optional[str]):
    """Name the account for the work in this context. Returns a reset token."""
    return _SELECTED_PROFILE.set(name or None)


def release_selected_profile(token) -> None:
    try:
        _SELECTED_PROFILE.reset(token)
    except (ValueError, RuntimeError):  # pragma: no cover — foreign context
        _SELECTED_PROFILE.set(None)


# The names that identify one conversation, most specific first.
#
# ``HERMES_SESSION_KEY`` is the gateway's per-chat key: one Telegram chat, one
# Discord channel. ``HERMES_SESSION_ID`` is the durable conversation id the
# command line, the terminal user interface, and the desktop application bind
# through ``set_current_session_id`` — it changes on ``/new`` and follows a
# ``/resume``, which is exactly the boundary a pin needs.
#
# Work with neither name has no conversation behind it: a cron job, a one-off
# script, a direct call. That work takes the shared slot and is not pinned.
# See ``select_for_job``.
_SESSION_KEY_NAMES = ("HERMES_SESSION_KEY", "HERMES_SESSION_ID")


def current_session_key() -> str:
    """The conversation this work belongs to, or "" when there is none."""
    for name in _SESSION_KEY_NAMES:
        try:
            from gateway.session_context import get_session_env

            value = get_session_env(name, "") or ""
        except Exception:
            value = os.environ.get(name, "") or ""
        if value.strip():
            return value.strip()
    return ""


def selected_profile_name() -> Optional[str]:
    """The account this work must use, most specific source first.

    1. The account this turn selected, held in a context variable. Two
       conversations running at once each read their own.
    2. The account this conversation started on.
    3. The process-wide slot, for work with no conversation behind it.
    """
    chosen = _SELECTED_PROFILE.get()
    if chosen:
        return chosen
    pinned = pinned_profile_name(current_session_key())
    if pinned:
        return pinned
    return active_profile_name()


def active_profile(config: Optional[dict] = None) -> Optional[ClaudeProfile]:
    """The profile a child process must use now, read from disk every time.

    This makes no network call. It reads ``config.yaml`` and the state file,
    so a person who edits either one changes the answer immediately. No
    service needs a restart.

    Returns None when the switcher is off, when nothing is recorded, or when
    the recorded nickname no longer names a configured profile.
    """
    profiles = load_profiles(config)
    if len(profiles) < 2:
        return None
    name = selected_profile_name()
    if not name:
        return None
    for candidate in profiles:
        if candidate.name == name:
            return candidate
    logger.debug("The recorded Claude Code profile %r is no longer configured", name)
    return None


def active_profile_env(config: Optional[dict] = None) -> dict:
    """The two directory variables for the profile in use now, or an empty map."""
    return profile_env(active_profile(config))


def clear_state() -> None:
    """Forget every selection and every pin. The next job starts fresh.

    The lock file stays. Deleting it while this process holds it would unlink
    the inode every waiter is queued on, and the next writer would create a
    second, unrelated lock. No credential is touched, because none is stored.
    """
    with _state_lock():
        try:
            state_path().unlink()
        except OSError:
            pass
    invalidate_usage_cache()


# ---------------------------------------------------------------------------
# The one call a job makes.
# ---------------------------------------------------------------------------

REASON_DISABLED = "disabled"


def _pinned_profile(
    session_id: Optional[str], profiles: list[ClaudeProfile]
) -> Optional[ClaudeProfile]:
    """Return the profile this conversation started on, when it still exists."""
    name = pinned_profile_name(session_id or "")
    if not name:
        return None
    for candidate in profiles:
        if candidate.name == name:
            return candidate
    logger.info(
        "Claude Code profile %r is pinned to a session but is no longer configured", name
    )
    return None


def _resume_on(
    pinned: ClaudeProfile,
    reader: Callable[[ClaudeProfile], ProfileUsage],
    threshold: float,
    session_id: Optional[str] = None,
) -> Selection:
    """Continue a conversation on the account that started it.

    A resume never moves to another account. Claude Code keeps the
    conversation record inside the profile directory, so the same identifier
    on another account either fails or starts fresh work and loses the first
    conversation. When the account is full, Hermes stops and reports the wait
    instead.
    """
    usage = reader(pinned)
    state = usability(usage, threshold)
    # Touch the pin either way. A conversation that is waiting for its account
    # to reopen is still live, and a trim must not drop it.
    pin_session(session_id or "", pinned.name)
    if state == OPEN:
        return Selection(
            profile=pinned,
            reason=REASON_PINNED,
            available=True,
            usage={pinned.name: usage},
        )
    invalidate_usage_cache(pinned.name)
    return Selection(
        profile=pinned,
        reason=REASON_PINNED,
        available=False,
        message=(
            f"This conversation started on the {pinned.name} profile and stays on it. "
            + describe_wait(usage, threshold)
        ),
        usage={pinned.name: usage},
    )


def select_for_job(
    session_id: Optional[str] = None,
    *,
    config: Optional[dict] = None,
    usage_reader: Optional[Callable[[ClaudeProfile], ProfileUsage]] = None,
) -> Selection:
    """Name the profile this job runs on, and remember the choice.

    With fewer than two profiles configured this changes nothing, so Hermes
    behaves exactly as it does today.
    """
    profiles = load_profiles(config)
    if len(profiles) < 2:
        return Selection(profile=None, reason=REASON_DISABLED, available=True, usage={})

    reader = usage_reader or read_profile_usage
    threshold = stop_at_percent(config)

    pinned = _pinned_profile(session_id, profiles)
    if pinned is not None:
        return _resume_on(pinned, reader, threshold, session_id)

    usages = {p.name: reader(p) for p in profiles}
    chosen = select_profile(
        profiles, usages, threshold=threshold, active_name=active_profile_name()
    )
    if chosen.profile is not None:
        record_active(chosen.profile.name)
        pin_session(session_id or "", chosen.profile.name)
        # The caller binds the context variable around its own child process
        # and releases it afterwards. Binding it here would leak this turn's
        # account into the next turn on the same thread.
        logger.info(
            "Claude Code profile %r selected (%s)", chosen.profile.name, chosen.reason
        )
    else:
        # Every account read as full. Drop the cached numbers so the next
        # attempt asks the endpoint again instead of repeating a stale "no".
        invalidate_usage_cache()
        logger.info("No Claude Code profile is available: %s", chosen.message)
    return chosen


# ---------------------------------------------------------------------------
# What a person reads before a job.
# ---------------------------------------------------------------------------

_STATE_WORDS = {
    OPEN: "open",
    UNKNOWN: "not checked",
    FULL: "full",
    BLOCKED: "sign in",
}


def _percent_cell(value: Optional[float]) -> str:
    return f"{value:.0f}%" if value is not None else "-"


def _reset_cell(usage: ProfileUsage, threshold: float) -> str:
    """The reopen time of the window that is full, or "-" when none is."""
    for _label, percent, reset in usage.windows:
        if percent is not None and percent >= threshold and reset is not None:
            return _format_time(reset)
    return "-"


def status_lines(
    *,
    config: Optional[dict] = None,
    usage_reader: Optional[Callable[[ClaudeProfile], ProfileUsage]] = None,
    usage_fetcher: Optional[Callable[[str], dict]] = None,
) -> list[str]:
    """Report every configured profile, its usage, and its reopen time.

    This reads local state and one usage endpoint per profile. It starts no
    model. It changes no selection: a person can read it at any moment without
    moving the work to another account.

    Every line names a profile by the nickname a person chose. No line holds a
    token, an address, an account number, or an organisation.
    """
    profiles = load_profiles(config)
    if len(profiles) < 2:
        found = len(profiles)
        return [
            "Claude Code profile switching: off "
            f"({found} profile{'s' if found != 1 else ''} configured; two are needed).",
            "Add profiles under `claude_cli_profiles.profiles` in config.yaml.",
        ]

    threshold = stop_at_percent(config)
    if usage_reader is None:
        def usage_reader(candidate: ClaudeProfile) -> ProfileUsage:  # noqa: E306
            return read_profile_usage(candidate, usage_fetcher=usage_fetcher)

    usages = {p.name: usage_reader(p) for p in profiles}
    in_use = active_profile_name()

    lines = [
        f"Claude Code profile switching: on. A window counts as full at {threshold:.0f}%.",
        f"{'NAME':<12}{'STATE':<10}{'5-HOUR':>8}{'WEEKLY':>8}{'OPUS':>8}  {'REOPENS':<22}",
    ]
    for candidate in profiles:
        usage = usages[candidate.name]
        state = usability(usage, threshold)
        row = (
            f"{candidate.name:<12}"
            f"{_STATE_WORDS.get(state, state):<10}"
            f"{_percent_cell(usage.five_hour_percent):>8}"
            f"{_percent_cell(usage.weekly_percent):>8}"
            f"{_percent_cell(usage.opus_weekly_percent):>8}"
            f"  {_reset_cell(usage, threshold):<22}"
        )
        if candidate.name == in_use:
            row += " in use"
        lines.append(row.rstrip())

    chosen = select_profile(profiles, usages, threshold=threshold, active_name=in_use)
    if not chosen.available:
        lines.append("")
        lines.append(chosen.message)
    return lines
