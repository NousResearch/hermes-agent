"""Auto-Archivierung aelterer Kanban-Tasks (21-Tage-Regel).

Archiviert Tasks mit ``status in ('todo', 'ready')``, die aelter als
``ARCHIVE_AFTER_DAYS`` (Default 21) Tage sind, und schreibt je Task einen
Audit-Log-Eintrag (``task_events`` kind='auto_archived') mit trigger='auto-21d'.

Der Aufruf ist idempotent: bereits archivierte Tasks werden von der
Selektion ausgeschlossen, ein zweiter Lauf archiviert nichts doppelt.

Konfiguration (zentral, kein Hardcoding in der Business-Logik):
- ``auto_archive.archive_after_days`` (Default 21)
- ``auto_archive.stale_threshold_days`` (Default 3, Floor: juengere Tasks
  werden nie vorzeitig archiviert)

CLI: ``python -m hermes_cli.kanban_auto_archive`` — Exit 0 bei Erfolg,
1 bei Fehler. ``--dry-run`` listet Kandidaten ohne Aenderung.

Hinweis: Die ``tasks``-Tabelle hat kein ``updated_at``; das Task-Alter
wird ueber ``created_at`` (Erstellung) bestimmt, konsistent mit der
Diagnose in Parent-Task t_a20c7e64. Verweildauer pro Status ist nur ueber
``task_events`` rekonstruierbar.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional

_log = logging.getLogger("kanban.auto_archive")

# Zentrale Defaults — per Config (config.yaml kanban.auto_archive.*) oder
# ENV (KANBAN_AUTO_ARCHIVE_*) ueberschreibbar, nicht in der Logik verdrahtet.
DEFAULT_ARCHIVE_AFTER_DAYS = 21
DEFAULT_STALE_THRESHOLD_DAYS = 3

STATUS_CANDIDATES = ("todo", "ready")
TRIGGER = "auto-21d"
EVENT_KIND = "auto_archived"


@dataclass(frozen=True)
class AutoArchiveSettings:
    """Konfigurierbare Schwellwerte aus einer zentralen Quelle (config/ENV)."""

    archive_after_days: int = DEFAULT_ARCHIVE_AFTER_DAYS
    stale_threshold_days: int = DEFAULT_STALE_THRESHOLD_DAYS


def load_settings(env: Optional[Mapping[str, str]] = None) -> AutoArchiveSettings:
    """Liest die Schwellwerte aus der zentralen Hermes-Config (config.yaml,
    Abschnitt ``kanban.auto_archive``), mit ENV-Override und spec-Defaults.

    Reihenfolge: ENV > config.yaml > Default. Die Business-Logik selbst
    enthaelt keine Schwellwerte; hier ist die einzige Konfigurationsstelle.
    """
    import os

    env = env if env is not None else os.environ
    archive_days = DEFAULT_ARCHIVE_AFTER_DAYS
    stale_days = DEFAULT_STALE_THRESHOLD_DAYS

    config: Mapping[str, object] = _load_config_section()
    if config:
        if isinstance(config.get("archive_after_days"), (int, float)) and config["archive_after_days"] > 0:
            archive_days = int(config["archive_after_days"])
        if isinstance(config.get("stale_threshold_days"), (int, float)) and config["stale_threshold_days"] >= 0:
            stale_days = int(config["stale_threshold_days"])

    env_a = env.get("KANBAN_AUTO_ARCHIVE_AFTER_DAYS", "").strip()
    if env_a.isdigit():
        archive_days = int(env_a)
    env_s = env.get("KANBAN_STALE_THRESHOLD_DAYS", "").strip()
    if env_s.isdigit():
        stale_days = int(env_s)

    if stale_days > archive_days:
        # Floor ueber der Schwelle wuerde die Regel stilllegen; das ist
        # eine Konfigurationsfehlfreigabe, kein stiller Fixup.
        raise ValueError(
            f"stale_threshold_days ({stale_days}) darf archive_after_days ({archive_days}) nicht ueberschreiten"
        )
    return AutoArchiveSettings(archive_after_days=archive_days, stale_threshold_days=stale_days)


def _load_config_section() -> dict:
    """Kanban-Abschnitt der zentralen config.yaml (Operator-Workspace-Pfad
    respektiert HERMES_HOME analog zum Rest von Hermes)."""
    import os

    home = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
    cfg_path = home / "config.yaml"
    if not cfg_path.exists():
        return {}
    try:
        import yaml

        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        _log.warning("config.yaml nicht lesbar (%s); nutze Defaults", exc)
        return {}
    section = ((data or {}).get("kanban") or {}).get("auto_archive")
    return section if isinstance(section, dict) else {}


def find_stale_candidates(conn, settings: AutoArchiveSettings, now_ts: Optional[float] = None) -> list[dict]:
    """Selektiert Kandidaten: status in (todo, ready) UND created_at aelter
    als archive_after_days, mit stale_threshold_days als Floor (juengere
    Tasks werden nie vorzeitig archiviert)."""
    now_ts = now_ts if now_ts is not None else time.time()
    cutoff = now_ts - settings.archive_after_days * 86400
    # Floor-Precheck (spec): juenger als stale_threshold_days -> niemals
    # archivieren. load_settings garantiert stale_threshold_days <=
    # archive_after_days, daher ist jede Task, die den 21d-Cutoff passiert,
    # automatisch auch aelter als der Floor — der Floor ist hier ohne
    # zusaetzliche Bedingung erfuellt; ein widerspruechlicher Konfigwert
    # (Floor > Schwelle) wird in load_settings hart abgelehnt.
    rows = conn.execute(
        "SELECT id, title, status, created_at, assignee FROM tasks "
        "WHERE status IN (?, ?) AND created_at < ? "
        "ORDER BY created_at",
        (*STATUS_CANDIDATES, cutoff),
    ).fetchall()
    return [dict(row) for row in rows]


def auto_archive(conn, settings: Optional[AutoArchiveSettings] = None, *, dry_run: bool = False) -> list[dict]:
    """Archiviert alle Kandidaten und schreibt je Task einen Audit-Event.

    Idempotent: die Selektion schliesst 'archived' aus; ein zweiter Lauf
    findet dieselben Tasks nicht mehr. Liefert die archivierten Tasks.
    """
    from hermes_cli import kanban_db as kb

    settings = settings or load_settings()
    with kb.write_txn(conn):
        candidates = find_stale_candidates(conn, settings)
        if dry_run:
            return candidates
        archived: list[dict] = []
        for task in candidates:
            cur = conn.execute(
                "UPDATE tasks SET status = 'archived', "
                "    claim_lock = NULL, claim_expires = NULL, worker_pid = NULL, worker_started_at = NULL "
                "WHERE id = ? AND status IN (?, ?)",
                (task["id"], *STATUS_CANDIDATES),
            )
            if cur.rowcount != 1:
                continue  # concurrent change gewinnt; kein Doppel-Eintrag
            conn.execute(
                "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
                "VALUES (?, NULL, ?, ?, ?)",
                (
                    task["id"],
                    EVENT_KIND,
                    json.dumps(
                        {
                            "task_id": task["id"],
                            "old_status": task["status"],
                            "new_status": "archived",
                            "trigger": TRIGGER,
                            "age_days": round((time.time() - task["created_at"]) / 86400, 1),
                        }
                    ),
                    int(time.time()),
                ),
            )
            archived.append(task)
    return archived


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hermes_cli.kanban_auto_archive",
        description="Archiviert Kanban-Tasks aelter als ARCHIVE_AFTER_DAYS (Default 21 Tage).",
    )
    parser.add_argument("--dry-run", action="store_true", help="nur Kandidaten listen, nichts aendern")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    try:
        from hermes_cli import kanban_db_connect as kbc

        settings = load_settings()
        with kbc.connect_closing() as conn:
            if args.dry_run:
                candidates = auto_archive(conn, settings, dry_run=True)
                for task in candidates:
                    _log.info("DRY-RUN wuerde archivieren: %s (status=%s, created_at=%s)", task["id"], task["status"], task["created_at"])
                _log.info("Dry-Run: %d Kandidaten, keine Aenderung.", len(candidates))
                return 0
            archived = auto_archive(conn, settings)
            for task in archived:
                _log.info("archiviert: %s (was %s)", task["id"], task["status"])
            _log.info("Fertig: %d archiviert.", len(archived))
        return 0
    except Exception:
        _log.exception("auto_archive fehlgeschlagen")
        return 1


if __name__ == "__main__":
    sys.exit(main())