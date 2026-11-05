"""Crash points around a commit, at main and at the head, compared by what the next start leaves behind.

Each case kills a process with ``os._exit`` at one point: the first piece of work an operation runs after its write
commits (a hook, media collection, a file removal, a second write, a queue offer), named on each tree. Every run's
home is built by main's store from one recipe; the operation runs through its entry point in a subprocess importing
that side's tree, with a ``sitecustomize`` (``_fault_site.py``) that patches the named function; then the next start
runs in the same home (``_driver.py restart``). The case passes when the head leaves what main leaves: every database
file's ``sqlite_master`` and rows, the home's files, the ``on_session_delete`` calls the plugin logged, and what the
next start printed. Main runs twice, and the two runs must agree before main is the reference.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.persistence_sqlite._legacy_store import oracle_tree
from tests.persistence_sqlite.upgrade._oracle import snapshot

WORKTREE = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DRIVER = HERE / "_driver.py"
FAULT_EXIT = 86
_TIMEOUT_S = 240
# Files that differ between any two runs of one tree: logs, the first start's home maintenance (config backups,
# install records), locks, the WAL sidecars of a closed store and the harness's own files.
_VOLATILE = re.compile(
    r"^(logs|backups|installs)/|(^|/)__pycache__/|\.lock$|-(wal|shm|journal)$|^fault\.(marker|net)$")
_TIME = re.compile(r"\b1[5-9]\d{8}(\.\d+)?\b")
# Dates and ages a listing prints: two runs of one case can fall on either side of midnight or of a minute.
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\bjust now\b|\b\d+[smhd] ago\b")
_HEX = re.compile(
    r"(?<![0-9a-f])([0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})(?![0-9a-f])")
_PID = re.compile(r"(?<=:)\d{2,8}(?=:)")
# A retired medium's key hashes its absolute path, which names the run's home.
_MEDIA_KEY = re.compile(r"(?<=gateway\.media\.retired\.v1\.)[0-9a-f]{64}")
# The writing process's pid and start time, different in every run.
_PROCESS_COLUMNS = {"owner_pid", "owner_started_at"}


@dataclass(frozen=True)
class Point:
    """``module:qualname`` on one tree, killed before its body runs or once it returns, at the call number
    ``skip`` + 1 among the calls whose arguments mention ``match``."""

    target: str
    when: str = "before"
    match: str = ""
    skip: int = 0


@dataclass(frozen=True)
class Case:
    op: str
    main: Point
    head: Point


_PUT = Point("queue:Queue.put", match="async_delegation")
_EMIT = Point("hermes_state_sessions:emit_session_deleted")
_PRUNE_EMIT = Point("hermes_state_maintenance:emit_session_deleted")
_HEAD_EMIT = Point("session_retention:emit_session_deleted")

CASES = {
    # A delete's post-commit work: the hook, then retired media, then transcript files.
    "delete-before-hook": Case("delete", _EMIT, _HEAD_EMIT),
    "delete-before-transcript-files": Case(
        "delete", Point("hermes_state_media:collect_retired_media", when="after"),
        Point("session_retention:collect_retired_media", when="after")),
    # Releasing a retired medium: main unlinks inside its write, the head after its held-set read commits.
    "delete-media-release": Case(
        "delete", Point("gateway.session_ingress_media:_held_file_identities"),
        Point("gateway.session_ingress_media:_held_file_identities")),
    "prune-before-hook": Case("prune", _PRUNE_EMIT, _HEAD_EMIT),
    # The never-active prune's second write, the routing sweep after the delete.
    "never-active-before-routing-sweep": Case(
        "never_active", Point("hermes_state_gateway:SessionGatewayMixin._delete_routing_entries_for_sessions"),
        Point("persistence.sqlite.gateway.routing:SQLiteRoutingRepository._delete_routing_entries_for_sessions")),
    "archive-before-first-hook": Case("archive", _EMIT, _HEAD_EMIT),
    "dashboard-bulk-delete-before-hook": Case("bulk_delete", _EMIT, _HEAD_EMIT),
    "dashboard-delete-empty-before-hook": Case("delete_empty", _EMIT, _HEAD_EMIT),
    # The TUI archive: main fires the hook after its commit; the head's caller has nothing after the verb.
    "tui-archive-after-commit": Case(
        "tui_archive", _EMIT,
        Point("persistence.sqlite.conversation.sessions:SQLiteSessionRepository.set_archived", when="after")),
    "auto-prune-before-hook": Case("auto_prune", _PRUNE_EMIT, _HEAD_EMIT),
    # Delegation ledger: the recovery sweep's work between its read and its write, then each queue offer.
    "delegation-recovery-between-units": Case(
        "recover", Point("tools.async_delegation_recovery_hints:git_state_hint", skip=1),
        Point("tools.async_delegation_recovery_hints:git_state_hint", skip=1)),
    "delegation-restore-first-offer": Case("restore", _PUT, _PUT),
    "delegation-orphan-sweep-first-offer": Case("sweep", _PUT, _PUT),
    "delegation-requeue-offer": Case("requeue", _PUT, _PUT),
    # Hosted-room attachments: a blob written inside the put's write, and released blobs unlinked after a prune.
    "attachment-put-after-blob-write": Case(
        "attachment_put",
        Point("gateway.hosted_room_attachments:HostedRoomAttachmentStore._write_blob", when="after"),
        Point("persistence.sqlite.gateway.hosted_room_attachments:_write_blob", when="after")),
    "attachment-prune-before-unlink": Case(
        "attachment_expired_put", Point("gateway.hosted_room_attachments:HostedRoomAttachmentStore._blob_path"),
        Point("persistence.sqlite.gateway.hosted_room_attachments:_blob_path")),
    # The orphan sweep renames unreferenced blobs inside its write and unlinks them after it.
    "attachment-sweep-before-quarantine-unlink": Case(
        "attachment_first_use", Point("pathlib:Path.unlink", match=".prune-"),
        Point("pathlib:Path.unlink", match=".prune-")),
}

_main_outcomes: dict = {}


def _env(tree: Path, home: Path, site: Path, point: Point | None) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTEST_", "FAULT_", "HERMES_"))}
    env.update(HOME=str(home), HERMES_HOME=str(home), TMPDIR=str(home), PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join([str(site), str(tree)]), FAULT_NET_LOG=str(home / "fault.net"))
    if point is not None:
        env.update(FAULT_POINT=point.target, FAULT_WHEN=point.when, FAULT_MATCH=point.match,
                   FAULT_SKIP=str(point.skip), FAULT_MARKER=str(home / "fault.marker"))
    return env


def _run(tree: Path, home: Path, site: Path, step: str, *args, point: Point | None = None):
    return subprocess.run([sys.executable, str(DRIVER), step, str(home), *args], cwd=home, check=False,
                          env=_env(tree, home, site, point), capture_output=True, text=True, timeout=_TIMEOUT_S)


def _clean(text: str, home: Path, trees) -> str:
    for path in (home, *trees):
        text = text.replace(str(path), "<path>")
    text = _DATE.sub("<date>", _TIME.sub("<t>", _MEDIA_KEY.sub("<path-hash>", text)))
    return _PID.sub("<pid>", _HEX.sub("<hex>", text))


def _value(value, home, trees):
    if isinstance(value, bytes):
        return "<bytes>"
    if isinstance(value, str):
        return _clean(value, home, trees)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and 1.5e9 <= value < 2e9:
        return "<t>"
    if isinstance(value, (list, tuple)):
        return [_value(v, home, trees) for v in value]
    return value


def outcome(home: Path, trees, process) -> dict:
    """What a home holds once *process* ended: every database file's schema and rows, the files, the hook calls
    the plugin logged and what the process printed, with times, random ids and paths normalised."""
    dbs = {}
    for db in sorted(home.rglob("*.db")):
        shot = snapshot(db)
        tables = {}
        for name, rows in shot["tables"].items():
            per_run = [i for i, column in enumerate(shot["columns"][name]) if column in _PROCESS_COLUMNS]
            tables[name] = sorted(
                (_value(["<process>" if i in per_run else v for i, v in enumerate(row)], home, trees) for row in rows),
                key=repr)
        dbs[str(db.relative_to(home))] = {
            "sqlite_master": shot["sqlite_master"],
            "tables": tables,
            "fts": shot["fts"],
        }
    files = sorted(_clean(str(p.relative_to(home)), home, trees) for p in home.rglob("*")
                   if p.is_file() and not _VOLATILE.search(str(p.relative_to(home))))
    hook = home / "hook.log"
    return {"dbs": dbs, "files": files,
            "hook": [_clean(line, home, trees) for line in hook.read_text().splitlines()] if hook.exists() else [],
            "restart_exit": process.returncode,
            "restart_stdout": [line for line in _clean(process.stdout, home, trees).splitlines()
                               if not line.startswith("DRIVER-RESULT")]}


def _diff(a: dict, b: dict, labels=("main", "head")) -> list:
    out = []
    for key in sorted(set(a) | set(b)):
        left, right = a.get(key), b.get(key)
        if left == right:
            continue
        if isinstance(left, dict) and isinstance(right, dict):
            out += [f"{key}.{line}" for line in _diff(left, right, labels)]
        elif isinstance(left, list) and isinstance(right, list):
            out.append(f"{key}: only {labels[0]}={[x for x in left if x not in right]!r} "
                       f"only {labels[1]}={[x for x in right if x not in left]!r}")
        else:
            out.append(f"{key}: {labels[0]}={left!r} {labels[1]}={right!r}")
    return out


@pytest.fixture(scope="module")
def seeded(tmp_path_factory):
    """The directory the runs' homes go under, and the one holding the ``sitecustomize``."""
    root = tmp_path_factory.mktemp("crash-points")
    site = root / "site"
    site.mkdir()
    shutil.copy(HERE / "_fault_site.py", site / "sitecustomize.py")
    return root, site


def _side(seeded, case: str, side: str, run: int) -> dict:
    root, site = seeded
    tree, point = (oracle_tree(), CASES[case].main) if side == "main" else (WORKTREE, CASES[case].head)
    # Seeded in place: the store records absolute media paths, so a copied home would point at the original.
    home = root / f"{case}-{side}-{run}"
    shutil.rmtree(home, ignore_errors=True)  # a failed first try of main's reference run, retried by the next test
    home.mkdir()
    seed = _run(oracle_tree(), home, site, "seed")
    assert seed.returncode == 0, seed.stdout[-3000:] + seed.stderr[-6000:]
    drive = _run(tree, home, site, "drive", CASES[case].op, point=point)
    marker = home / "fault.marker"
    reached = marker.read_text().splitlines()[0] if marker.exists() else None
    assert drive.returncode == FAULT_EXIT and reached == point.target, (
        f"{side}: the operation did not stop at {point.target} (exit {drive.returncode})\n--- stdout\n"
        f"{drive.stdout[-3000:]}\n--- stderr\n{drive.stderr[-6000:]}")
    at_kill = outcome(home, (root, oracle_tree(), WORKTREE), drive)
    restart = _run(tree, home, site, "restart")
    result = outcome(home, (root, oracle_tree(), WORKTREE), restart)
    net = home / "fault.net"
    assert not net.exists(), f"{side}: a connection off this host was attempted: {net.read_text()}"
    assert restart.returncode == 0, f"{side}: the next start failed\n{restart.stdout[-3000:]}\n{restart.stderr[-6000:]}"
    # FAULTS_OUT names a directory for each run's state at the kill and after the next start.
    out = os.environ.get("FAULTS_OUT")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        Path(out, f"{case}-{side}-{run}.json").write_text(json.dumps(
            {"at_kill": {k: at_kill[k] for k in ("dbs", "files", "hook")}, "after_restart": result,
             "restart_stderr": _clean(restart.stderr, home, (root, oracle_tree(), WORKTREE))[-20000:]},
            indent=1, default=repr))
    return result


def _main_outcome(seeded, case: str) -> dict:
    if case not in _main_outcomes:
        _main_outcomes[case] = _side(seeded, case, "main", 1)
    return _main_outcomes[case]


@pytest.mark.parametrize("case", sorted(CASES))
def test_main_is_repeatable(seeded, case):
    """Main reaches its point and two runs leave the same state and the same start-up output."""
    first, second = _main_outcome(seeded, case), _side(seeded, case, "main", 2)
    assert first == second, "\n".join(_diff(first, second, ("run 1", "run 2")))


@pytest.mark.parametrize("case", sorted(CASES))
def test_head_leaves_what_main_leaves(seeded, case):
    """The head, killed at the same point, leaves what main leaves once the next start has run."""
    reference, head = _main_outcome(seeded, case), _side(seeded, case, "head", 1)
    assert head == reference, "\n".join(_diff(reference, head))
