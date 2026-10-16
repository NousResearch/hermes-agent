"""Low-disk prompt for ``sessions.auto_prune`` (off by default).

Auto-prune deletes history, so it is opt-in. Users who never decided still need to hear about it
when the disk holding ``state.db`` runs low: every surface asks once the disk is low, and the
question stops forever as soon as ``sessions.auto_prune`` is written to the user's config.yaml
(either answer). "Later" snoozes for :data:`SNOOZE_SECONDS`.

The hot-path check is two syscalls (``os.stat`` + ``shutil.disk_usage``, microseconds), so it can
run at every session start. Only when it fires does :func:`advise` open the database read-only to
estimate what a prune would actually free, so the prompt can say plainly when pruning will not
fix the disk (nothing old enough to delete, or the store is a small share of the full disk).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

GIB = 1024 ** 3
# Low = under 2 GiB free, or under 10% free while also under 20 GiB (a 10%-free 4 TB disk is fine).
LOW_FREE_FLOOR = 2 * GIB
LOW_FREE_PERCENT = 10.0
LOW_FREE_HEADROOM = 20 * GIB
SNOOZE_SECONDS = 7 * 86400
_SNOOZE_FILE = ".auto_prune_prompt.json"


@dataclass(frozen=True)
class DiskSnapshot:
    db_bytes: int
    free_bytes: int
    total_bytes: int

    @property
    def low(self) -> bool:
        if self.total_bytes <= 0:
            return False
        free_pct = self.free_bytes * 100.0 / self.total_bytes
        return self.free_bytes < LOW_FREE_FLOOR or (
            free_pct < LOW_FREE_PERCENT and self.free_bytes < LOW_FREE_HEADROOM)


@dataclass(frozen=True)
class Advice:
    disk: DiskSnapshot
    retention_days: int
    prunable_sessions: int
    total_sessions: int
    reclaim_bytes: int  # estimate: the prunable share of stored message bytes times the file size
    estimate_ok: bool = True  # False when state.db could not be read (counts are unknown, not zero)
    vacuum_after_prune: bool = True

    @property
    def helps(self) -> bool:
        """Whether a prune would free a meaningful amount: at least 1 GiB, or enough to lift the
        disk back out of the low band."""
        if self.prunable_sessions == 0:
            return False
        lifted = DiskSnapshot(self.disk.db_bytes - self.reclaim_bytes,
                              self.disk.free_bytes + self.reclaim_bytes, self.disk.total_bytes)
        return self.reclaim_bytes >= GIB or not lifted.low


def _home(home: Optional[Path]) -> Path:
    if home is not None:
        return Path(home)
    from hermes_constants import get_hermes_home
    return get_hermes_home()


def disk_snapshot(home: Optional[Path] = None) -> Optional[DiskSnapshot]:
    """``state.db`` size plus free/total bytes of its filesystem; None when either is unreadable."""
    db = _home(home) / "state.db"
    try:
        size = os.stat(db).st_size
        usage = shutil.disk_usage(db.parent)
    except OSError:
        return None
    return DiskSnapshot(size, usage.free, usage.total)


def _effective_sessions(home: Optional[Path]) -> Optional[dict]:
    """*home*'s ``sessions:`` block as maintenance sees it (user file, ``${VAR}`` expansion, managed
    overlay; no defaults merged) or None when the file cannot be read. Read through *home*'s own
    config path, so gateway housekeeping gets each served profile's values, not the caller's."""
    from hermes_cli.config_effective import load_user_config_effective
    try:
        sessions = load_user_config_effective(_home(home) / "config.yaml", fail_closed=True).get("sessions")
    except Exception as exc:  # health: allow BLE001 -- any parse/read failure means "unreadable": never ask on it
        logger.debug("auto-prune advice: config unreadable: %s", exc, exc_info=True)
        return None
    return sessions if isinstance(sessions, dict) else {}


def _retention_days(sessions: dict) -> int:
    """Same coercion maintenance applies (``int(...)``); an unusable value reads as the default."""
    try:
        days = int(sessions.get("retention_days", 90))
    except (TypeError, ValueError):
        return 90
    return days if days >= 0 else 90


def undecided(home: Optional[Path] = None) -> bool:
    """True while neither the user's config nor an administrator's managed layer sets
    ``sessions.auto_prune``. A managed value is already decided: asking would promise a choice the
    user cannot make."""
    sessions = _effective_sessions(home)
    return sessions is not None and "auto_prune" not in sessions


def _snooze_path(home: Optional[Path]) -> Path:
    return _home(home) / _SNOOZE_FILE


def snoozed(home: Optional[Path] = None, *, now: Optional[float] = None) -> bool:
    try:
        until = float(json.loads(_snooze_path(home).read_text(encoding="utf-8-sig")).get("snoozed_until", 0))
    except (OSError, ValueError, TypeError, AttributeError):
        return False
    return (now or time.time()) < until


def snooze(home: Optional[Path] = None, seconds: float = SNOOZE_SECONDS) -> None:
    from utils import atomic_json_write
    atomic_json_write(_snooze_path(home), {"snoozed_until": time.time() + seconds})


def should_ask(home: Optional[Path] = None) -> Optional[DiskSnapshot]:
    """The cheap gate every surface runs at session start: the snapshot when the disk is low and
    the user has neither decided nor snoozed, else None. Two syscalls plus one small YAML read."""
    snap = disk_snapshot(home)
    if snap is None or not snap.low:
        return None
    if not undecided(home) or snoozed(home):
        return None
    return snap


def advise(home: Optional[Path] = None) -> Optional[Advice]:
    """:func:`should_ask` plus the reclaim estimate. Opens state.db read-only (a second or two on
    a multi-GB store), so callers run it only after the cheap gate fires or off the prompt thread."""
    snap = should_ask(home)
    if snap is None:
        return None
    # Read the settings from THIS home's config: gateway housekeeping calls this for every served
    # profile without entering its scope. Only an explicit key differs from the defaults here.
    sessions = _effective_sessions(home) or {}
    retention = _retention_days(sessions)
    prunable = total = prunable_bytes = total_bytes = 0
    estimate_ok = True
    try:
        from hermes_state import SessionDB
        db = SessionDB(db_path=_home(home) / "state.db", read_only=True)
        try:
            where, params = db._prune_where(retention, None, {}, whole_lineages=True)
            prunable, total = tuple(db._read_one(
                f"SELECT (SELECT COUNT(*) FROM sessions s WHERE {where}), (SELECT COUNT(*) FROM sessions)",
                params) or (0, 0))
            if prunable:
                # Weight by stored bytes, not message_count: old messages are far smaller than recent
                # ones, and message_count skips the soft-archived rows a prune also deletes.
                row_bytes = _row_bytes_sql(db)
                prunable_bytes = (db._read_one(
                    f"SELECT COALESCE(SUM({row_bytes}), 0) FROM messages m WHERE m.session_id IN "
                    f"(SELECT s.id FROM sessions s WHERE {where})", params) or (0,))[0]
                total_bytes = (db._read_one(
                    f"SELECT COALESCE(SUM({row_bytes}), 0) FROM messages m", []) or (0,))[0]
        finally:
            db.close()
    except (sqlite3.Error, OSError, RuntimeError, ValueError) as exc:  # a busy/damaged store: no estimate
        logger.debug("auto-prune advice: reclaim estimate unavailable: %s", exc)
        estimate_ok = False
    reclaim = int(snap.db_bytes * prunable_bytes / total_bytes) if total_bytes else 0
    return Advice(snap, retention, int(prunable), int(total), reclaim, estimate_ok,
                  bool(sessions.get("vacuum_after_prune", True)))


def _row_bytes_sql(db) -> str:
    """Stored size of one ``messages`` row: the byte length of every text/blob column (read from
    the live schema, so a column added later is counted without touching this)."""
    cols = [r[1] for r in db._read_all("PRAGMA table_info(messages)", [])
            if str(r[2]).upper() in {"TEXT", "BLOB", ""}]
    return " + ".join(f'COALESCE(LENGTH(CAST(m."{c}" AS BLOB)), 0)' for c in cols) or "0"


def _gb(n: int) -> str:
    """Human size: GB at or above 1 GiB, MB below (a 26 MB store must not read "0.0 GB")."""
    return f"{n / GIB:.1f} GB" if n >= GIB else f"{max(n, 0) / 1024 ** 2:.0f} MB"


def _sessions(n: int) -> str:
    return f"{n} session" + ("" if n == 1 else "s")


def advice_lines(advice: Advice, *, command_prefix: str = "/") -> list[str]:
    """Plain-text prompt shared by every surface (no markup); the last line names the commands."""
    d = advice.disk
    lines = [
        f"Low disk space: {_gb(d.free_bytes)} free of {_gb(d.total_bytes)}. "
        f"Your session history (state.db) uses {_gb(d.db_bytes)}.",
        f"Auto-prune is off. Turning it on deletes ended sessions inactive for more than "
        f"{advice.retention_days} days, once a day (pinned and open sessions are kept).",
    ]
    if not advice.estimate_ok:
        lines.append("Could not read state.db to estimate how much pruning would free.")
    elif advice.prunable_sessions == 0:
        held = (f"none of your {_sessions(advice.total_sessions)} is older than {advice.retention_days} days"
                if advice.total_sessions else "there is no session history to delete")
        lines.append(f"Pruning will not help right now: {held}. Something else is filling this disk.")
    elif not advice.helps:
        lines.append(f"Pruning would remove only about {_gb(advice.reclaim_bytes)} of history "
                     f"({_sessions(advice.prunable_sessions)}), which will not fix this disk on its own.")
    else:
        lines.append(f"Pruning would remove about {_gb(advice.reclaim_bytes)} of history "
                     f"({advice.prunable_sessions} of {_sessions(advice.total_sessions)}).")
    if advice.estimate_ok and advice.prunable_sessions:
        # Automatic compaction runs only past a free-page threshold most prunes never reach, so the
        # disk gets the space back reliably only from an explicit VACUUM.
        lines.append("Deleted history frees space inside state.db; to give it back to the disk, run "
                     "`hermes sessions optimize` with other Hermes windows closed.")
    p = command_prefix
    lines.append(f"{p}autoprune on to enable, {p}autoprune off to keep everything (never asked again), "
                 f"{p}autoprune later to be reminded in a week.")
    return lines


NOTICE_KEY = "sessions.auto_prune"


def notice_payload(advice: Advice) -> dict:
    """``notification.show`` payload for TUI/Desktop: a short status-line ``text`` plus the full
    question in ``detail`` (Desktop's expandable toast body; Ink prints it into the transcript)."""
    d = advice.disk
    return {"text": f"⚠ Low disk ({_gb(d.free_bytes)} free) · /autoprune on|off|later",
            "detail": "\n".join(advice_lines(advice)), "level": "warn", "kind": "sticky", "ttl_ms": None,
            "key": NOTICE_KEY, "id": NOTICE_KEY}


def answered(home: Optional[Path] = None) -> bool:
    """True once the question is settled for now (a saved on/off, or a live ``later`` snooze):
    the only state in which a shown notice may be withdrawn."""
    return not undecided(home) or snoozed(home)


def notice_due(home: Optional[Path], shown: dict) -> bool:
    """Per-process dedup that re-arms: a home already noticed is due again once the gate passes
    and a full snooze window has gone by since (so ``later`` really asks again in a week)."""
    if should_ask(home) is None:
        return False
    last = shown.get(str(_home(home)))
    return last is None or time.time() - last >= SNOOZE_SECONDS


def managed() -> bool:
    """True when an administrator pins ``sessions.auto_prune`` (the user cannot change it)."""
    from hermes_cli import managed_scope
    return managed_scope.is_key_managed("sessions.auto_prune")


def decide(enable: bool, home: Optional[Path] = None) -> bool:
    """Persist the user's answer as an explicit ``sessions.auto_prune``; ends the prompt for good.
    False when the save failed or did not take effect (an administrator-managed value wins)."""
    import hermes_yaml as yaml
    from hermes_constants import mkdir_under_hermes_home
    from utils import atomic_roundtrip_yaml_update
    if managed():
        return False
    path = _home(home) / "config.yaml"
    try:
        mkdir_under_hermes_home(path.parent)
        atomic_roundtrip_yaml_update(path, "sessions.auto_prune", bool(enable))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        logger.warning("Could not save sessions.auto_prune: %s", exc)
        return False
    try:
        _snooze_path(home).unlink(missing_ok=True)
    except OSError:
        pass
    return bool((_effective_sessions(home) or {}).get("auto_prune")) == bool(enable)


_SAVE_FAILED = "Could not save sessions.auto_prune; check disk space and config.yaml permissions."


def run_command(args: str, home: Optional[Path] = None) -> str:
    """``/autoprune [on|off|later|status]`` — the one handler behind the CLI, TUI, Desktop and
    messaging surfaces."""
    arg = (args or "").strip().lower()
    sessions = _effective_sessions(home) or {}
    state = "on" if sessions.get("auto_prune") else "off"
    if arg in {"on", "off", "enable", "disable", "yes", "no", "later", "snooze"} and managed():
        return f"Auto-prune is {state}, set by your administrator; it cannot be changed here."
    if arg in {"on", "enable", "yes"}:
        return ("Auto-prune is on: ended sessions inactive past retention_days are deleted once a day."
                if decide(True, home) else _SAVE_FAILED)
    if arg in {"off", "disable", "no"}:
        return ("Auto-prune is off and you will not be asked again. `hermes sessions prune` still works any time."
                if decide(False, home) else _SAVE_FAILED)
    if arg in {"later", "snooze"}:
        snooze(home)
        return "OK, I'll ask again in a week if the disk is still low."
    chosen = ("set by your administrator" if managed() else
              "default" if "auto_prune" not in sessions else "your setting")
    snap = disk_snapshot(home)
    disk = (f" state.db {_gb(snap.db_bytes)}; {_gb(snap.free_bytes)} free of {_gb(snap.total_bytes)}"
            f"{' (low)' if snap.low else ''}.") if snap else ""
    return (f"Auto-prune is {state} ({chosen}), retention {_retention_days(sessions)} days.{disk} "
            "Usage: /autoprune on | off | later")


def offer_interactive(*, cprint, ask=input) -> Optional[bool]:
    """Classic-CLI startup offer (TTY only). Returns the decision, or None when nothing was asked
    or the user chose later."""
    if should_ask() is None:
        return None
    # The byte estimate reads every message row: seconds on a cold multi-GB store. Say so.
    cprint("\n☤ Low disk space: checking how much session history pruning could free…")
    advice = advise()
    if advice is None:
        return None
    cprint("  " + advice_lines(advice)[0])
    for line in advice_lines(advice)[1:-1]:
        cprint(f"  {line}")
    try:
        answer = ask("  Enable auto-prune? [y = enable / n = never prune, don't ask again / Enter = remind me later]: ")
    except (KeyboardInterrupt, EOFError):
        print()
        return None
    answer = answer.strip().lower()
    if answer in {"y", "yes", "n", "no"}:
        enable = answer.startswith("y")
        if not decide(enable):
            cprint(f"  {_SAVE_FAILED}")
            return None
        cprint("  ✓ sessions.auto_prune: true" if enable else
               "  ✓ sessions.auto_prune: false. You won't be asked again.")
        return enable
    snooze()
    cprint("  Later. /autoprune on|off decides it any time.")
    return None
