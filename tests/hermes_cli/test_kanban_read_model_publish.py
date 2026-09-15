"""Tests for `hermes kanban publish-read-model` — H0.2b3b.

The publisher is the ONLY component allowed to open the live Kanban database,
and it may do so only under explicit owner authority. A ``delegate_task`` child
has no such authority: the plan requires the denial to happen *before* any DB
open and *before* any file write, so a delegated child can neither touch
``kanban.db`` (WAL/SHM side effects are a mutation) nor plant, truncate, or
replace an artifact on disk.

This file starts with that one boundary, then covers the authorized owner happy
path: the allowlisted query, the shared sanitization/bounds contract, and the
exact artifact-v1 document, the adversarial output-path matrix, and the
rule that a failed publication never changes the previous artifact, the
parent-swap rule, and the concurrent-reader invariant: while the publisher
commits, a reader observes only a complete previous or a complete next
artifact. The remaining symlink-race cases are later slices of H0.2b3b.
"""

from __future__ import annotations

import argparse
import builtins
import collections
import errno
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import tempfile
import threading
import time
import urllib.parse
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_read_model as krm
from hermes_cli import kanban_read_model_publish as krmp


def _parse_kanban_args(cli_args):
    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)
    return parser.parse_args(cli_args)


def test_publish_read_model_denies_delegated_child_before_db_open_and_any_write(
    tmp_path, monkeypatch, capsys
):
    """A delegated child must be refused before `sqlite3.connect` and before
    the first write-mode open — not merely before the rename of a finished
    artifact."""
    out = tmp_path / "published" / "read-model.json"

    db_connects: list[tuple] = []
    write_opens: list[tuple] = []

    def _tripwire_connect(*args, **kwargs):
        db_connects.append(args)
        raise AssertionError(
            "publish-read-model opened the live database in a delegated child"
        )

    monkeypatch.setattr(sqlite3, "connect", _tripwire_connect)

    real_open = builtins.open

    def _tripwire_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            write_opens.append((str(file), mode))
            raise AssertionError(
                "publish-read-model opened a file for writing in a delegated child"
            )
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _tripwire_open)

    real_os_open = os.open
    _WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC

    def _tripwire_os_open(path, flags, *args, **kwargs):
        if flags & _WRITE_FLAGS:
            write_opens.append((str(path), flags))
            raise AssertionError(
                "publish-read-model opened an fd for writing in a delegated child"
            )
        return real_os_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", _tripwire_os_open)

    # The established narrow signal, same one `_is_delegated_child_cli_mutation`
    # consults: a delegate_task child carries this marker across process
    # boundaries.
    monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "1")

    args = _parse_kanban_args(
        ["kanban", "publish-read-model", "--out", str(out), "--limit", "50"]
    )
    rc = kc.kanban_command(args)

    assert rc == 1
    assert db_connects == []
    assert write_opens == []
    assert not out.exists()
    assert not out.parent.exists()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "delegate_task child contexts cannot publish" in captured.err


def test_publish_read_model_fails_closed_when_the_authority_helper_raises(
    publish_sandbox, monkeypatch, capsys
):
    """An *available* authority helper that raises must deny, not fall back.

    The environment marker is only a lineage carrier for the case where the
    canonical helper cannot be imported at all (a trimmed install, an import
    cycle). It is NOT a substitute verdict: if the helper imports fine but
    fails operationally — a ContextVar lookup blowing up, a shadowed module
    raising on call — then authority is *unknown*, and unknown authority is
    not owner authority. Falling back to "no marker in os.environ, therefore
    owner" hands a delegated child exactly the publication right the marker
    exists to withhold, because a child that lost the marker is precisely the
    caller this branch is reached for.

    So the refusal has to land in the same place the delegated-child refusal
    lands: before `sqlite3.connect` and before the first write, leaving the
    board and the filesystem byte-identical.
    """
    root = publish_sandbox / "kanban-root"
    root.mkdir()
    board = "synthetic"
    db = root / "kanban" / "boards" / board / "kanban.db"
    # Seeded BEFORE the tripwires are armed: a real, publishable board, so a
    # leak here is a leak of live task data and not of an empty file.
    _seed_synthetic_board(db)
    out = publish_sandbox / "published" / "read-model.json"

    db_connects: list[tuple] = []
    writes: list[tuple] = []

    def _tripwire_connect(*args, **kwargs):
        db_connects.append(args)
        raise AssertionError(
            "publish-read-model opened the live database under an "
            "undetermined authority verdict"
        )

    def _tripwire_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            writes.append(("open", str(file), mode))
            raise AssertionError(
                "publish-read-model opened a file for writing under an "
                "undetermined authority verdict"
            )
        return real_open(file, mode, *args, **kwargs)

    real_open = builtins.open
    real_os_open = os.open
    _WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC

    def _tripwire_os_open(path, flags, *args, **kwargs):
        if flags & _WRITE_FLAGS:
            writes.append(("os.open", str(path), flags))
            raise AssertionError(
                "publish-read-model opened an fd for writing under an "
                "undetermined authority verdict"
            )
        return real_os_open(path, flags, *args, **kwargs)

    def _tripwire_named(name):
        def _fail(*args, **kwargs):
            writes.append((name,) + tuple(str(arg) for arg in args))
            raise AssertionError(
                f"publish-read-model called os.{name} under an undetermined "
                "authority verdict"
            )

        return _fail

    # The canonical helper imports cleanly and then fails operationally. The
    # marker text is distinctive so the stderr assertion below can prove the
    # CLI did not turn this command into an oracle for the failure reason.
    _OPERATIONAL = "OPERATIONAL-AUTHORITY-FAILURE-DETAIL"

    def _raising_helper():
        raise RuntimeError(_OPERATIONAL)

    args = _parse_kanban_args(
        [
            "kanban",
            "--kanban-root", str(root),
            "--board", board,
            "publish-read-model",
            "--db", str(db),
            "--out", str(out),
            "--limit", "3",
        ]
    )

    # Armed for the invocation and nothing else: `os.unlink`/`os.rmdir` stay
    # real for the sandbox fixture's own cleanup, which would otherwise trip a
    # wire meant for the publisher.
    with monkeypatch.context() as armed:
        armed.setattr(sqlite3, "connect", _tripwire_connect)
        armed.setattr(builtins, "open", _tripwire_open)
        armed.setattr(os, "open", _tripwire_os_open)
        # The rest of the mutating surface the publication path uses:
        # directory creation, the atomic swap, and the rollback's unlink.
        for _name in ("mkdir", "rename", "replace", "unlink", "chmod", "link"):
            armed.setattr(os, _name, _tripwire_named(_name))
        armed.setattr(
            "agent.delegation_context.is_delegated_child_process_context",
            _raising_helper,
        )
        # No environment marker: this is the exact shape in which the old
        # `except Exception` fallback returned False and authorized the write.
        armed.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)

        rc = kc.kanban_command(args)

    captured = capsys.readouterr()

    # Fail closed, and fail EARLY. `rc == 1` alone proves nothing here: a
    # tripwire that fires also surfaces as 1, so the tripwire ledgers are what
    # separate "refused" from "authorized, then caught in the act".
    assert db_connects == [], (
        "authority leak: the publisher opened the live Kanban database after "
        "an available authority helper raised"
    )
    assert writes == [], (
        f"authority leak: the publisher attempted writes {writes} after an "
        "available authority helper raised"
    )
    assert rc == 1
    assert not out.exists()
    assert not out.parent.exists()

    # A denial is not a result: nothing on stdout, and the reason stays
    # generic — the operational detail never reaches the caller.
    assert captured.out == ""
    assert _OPERATIONAL not in captured.err
    assert "RuntimeError" not in captured.err
    assert captured.err.strip() == (
        f"kanban publish-read-model: {krmp.AUTHORITY_UNDETERMINED_DENIAL}"
    )


# ---------------------------------------------------------------------------
# Cycle 2 — the authorized owner happy path: exact artifact-v1 allowlist
# ---------------------------------------------------------------------------

# A synthetic `tasks` table. It deliberately carries the SENSITIVE columns the
# PlanSpec excludes alongside the eight allowlisted ones, because that is the
# only way a test can tell "the publisher selected an allowlist" apart from
# "the publisher ran SELECT * against a table that happened to be narrow".
_SYNTHETIC_TASKS_DDL = """
CREATE TABLE tasks (
    id               TEXT PRIMARY KEY,
    title            TEXT NOT NULL,
    body             TEXT,
    assignee         TEXT,
    status           TEXT NOT NULL,
    priority         INTEGER DEFAULT 0,
    created_by       TEXT,
    created_at       INTEGER NOT NULL,
    started_at       INTEGER,
    completed_at     INTEGER,
    workspace_path   TEXT,
    branch_name      TEXT,
    claim_lock       TEXT,
    tenant           TEXT,
    result           TEXT,
    worker_pid       INTEGER,
    session_id       TEXT,
    skills           TEXT,
    model_override   TEXT,
    provider_override TEXT,
    reasoning_effort TEXT
)
"""

# Every excluded column below is seeded with this marker, so a single substring
# check over the RAW published bytes catches a leak through any of them —
# including one that slipped in under a key this test does not know to look at.
_SENSITIVE = "SENSITIVE-MUST-NOT-BE-PUBLISHED"


def _seed_synthetic_board(db_path):
    """Create a synthetic Kanban DB with six eligible rows and one archived.

    Priorities are chosen so the top three are NOT simply "the first three
    inserted" and so a three-way tie (p5) straddles the limit boundary: only a
    publisher that orders by priority DESC then id ASC can pick
    t_aaa/t_bbb/t_ccc rather than t_ddd.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(_SYNTHETIC_TASKS_DDL)
        rows = [
            # id,      title,          status,    prio, assignee, started, completed
            ("t_eee", "lowest",       "todo",       1, None,     None, None),
            ("t_ddd", "tie loser",    "ready",      5, "carol",  None, None),
            ("t_bbb", "tie winner",   "running",    5, "alice",  1700, None),
            ("t_ccc", "tie middle",   "review",     5, None,     1710, None),
            ("t_aaa", "highest",      "done",       9, "bob",    1720, 1730),
            ("t_fff", "also low",     "blocked",    2, None,     None, None),
            # Archived rows are excluded from the read model entirely; this one
            # outranks every eligible row, so its absence is unambiguous.
            ("t_zzz", "archived",     "archived", 100, "mallory", None, None),
        ]
        for task_id, title, status, priority, assignee, started, completed in rows:
            conn.execute(
                "INSERT INTO tasks ("
                " id, title, body, assignee, status, priority, created_by,"
                " created_at, started_at, completed_at, workspace_path,"
                " branch_name, claim_lock, tenant, result, worker_pid,"
                " session_id, skills, model_override, provider_override,"
                " reasoning_effort"
                ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    task_id, title, f"body {_SENSITIVE}", assignee, status,
                    priority, f"created_by {_SENSITIVE}", 1600, started,
                    completed, f"/ws/{_SENSITIVE}", f"branch-{_SENSITIVE}",
                    f"lock-{_SENSITIVE}", f"tenant-{_SENSITIVE}",
                    f"result {_SENSITIVE}", 4242, f"sess-{_SENSITIVE}",
                    f'["{_SENSITIVE}"]', f"model-{_SENSITIVE}",
                    f"provider-{_SENSITIVE}", f"effort-{_SENSITIVE}",
                ),
            )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def publish_sandbox():
    """A private directory a publication may actually land in.

    Deliberately NOT pytest's `tmp_path`. Under the canonical runner that
    lives inside the shared temp root — `PYTEST_DEBUG_TEMPROOT` defaults to
    `tempfile.gettempdir()`, i.e. `/tmp`, which pytest then resolves to
    `/private/tmp` on macOS — and the publisher refuses that whole subtree.
    Every destination built there would therefore be refused for a reason the
    test was not written to check: the happy path could never publish, and the
    symlink and group-/world-writable cases would pass without their own rule
    ever being consulted.

    A uniquely named directory under the invoking user's home is outside the
    temp subtree, outside any real Kanban root, and outside the real
    HERMES_HOME (the suite already redirects that to `tmp_path`). It is 0700
    so nothing built inside it trips the writability rule by inheritance, and
    it is removed whether the test passes or fails — pytest does not reclaim
    it the way it reclaims `tmp_path`.
    """
    sandbox = Path(
        tempfile.mkdtemp(prefix="hermes-h0-2b3b-sandbox-", dir=Path.home())
    )
    # Explicit chmod, not the mkdtemp default, for the same reason the cases
    # below chmod explicitly: the mode this sandbox grants is load-bearing, so
    # it is stated rather than inherited.
    os.chmod(sandbox, 0o700)
    try:
        yield sandbox
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


def test_publish_read_model_emits_the_bounded_artifact_v1_allowlist_for_an_owner(
    publish_sandbox, capsys
):
    """One explicit owner invocation publishes exactly the artifact-v1
    document: the six top-level keys, the eight task fields, nothing else, and
    a `truncated` flag that tells the truth about the applied limit."""
    root = publish_sandbox / "kanban-root"
    root.mkdir()
    board = "synthetic"
    # Non-default boards live at <root>/kanban/boards/<slug>/kanban.db — the
    # publisher is the only component allowed to name this path at all.
    db = root / "kanban" / "boards" / board / "kanban.db"
    _seed_synthetic_board(db)

    # Outside every Kanban-controlled root, as the artifact contract requires,
    # and outside the shared temp subtree, which the publisher refuses whole.
    out = publish_sandbox / "published" / "read-model.json"

    limit = 3
    before = int(time.time())
    args = _parse_kanban_args(
        [
            "kanban",
            "--kanban-root", str(root),
            "--board", board,
            "publish-read-model",
            "--db", str(db),
            "--out", str(out),
            "--limit", str(limit),
        ]
    )
    rc = kc.kanban_command(args)
    after = int(time.time())

    captured = capsys.readouterr()
    assert rc == 0, f"publish failed: {captured.err}"
    assert captured.err == ""

    raw = out.read_bytes()
    # The reader accepts nothing else, so a publisher that emits 0644 has
    # produced a file no reader will ever read.
    assert stat.S_IMODE(out.stat().st_mode) == krm.ARTIFACT_MODE
    assert len(raw) <= krm.MAX_ARTIFACT_BYTES

    # No excluded/sensitive column may appear anywhere in the published bytes.
    assert _SENSITIVE not in raw.decode("utf-8")

    document = json.loads(raw.decode("utf-8"))
    assert set(document) == set(krm.TOP_LEVEL_FIELDS)
    assert document["schema_version"] == krm.SCHEMA_VERSION
    assert document["board"] == board
    assert document["root_fingerprint"] == krm.root_fingerprint(root)
    assert isinstance(document["generated_at"], int)
    assert not isinstance(document["generated_at"], bool)
    assert before <= document["generated_at"] <= after

    # Six eligible rows, limit 3 → the flag must say so, as a real bool.
    assert document["truncated"] is True

    tasks = document["tasks"]
    assert isinstance(tasks, list)
    assert len(tasks) == limit
    # priority DESC, then id ASC. t_ddd ties t_bbb/t_ccc at p5 and loses on id;
    # t_zzz outranks everything but is archived and therefore not eligible.
    assert [task["id"] for task in tasks] == ["t_aaa", "t_bbb", "t_ccc"]

    for task in tasks:
        assert set(task) == set(krm.TASK_FIELDS)
        assert task["status"] in krm.VALID_STATUSES
        assert task["status"] != "archived"

    assert tasks[0] == {
        "id": "t_aaa",
        "title": "highest",
        "status": "done",
        "assignee": "bob",
        "priority": 9,
        "created_at": 1600,
        "started_at": 1720,
        "completed_at": 1730,
    }
    # Nullable columns survive as JSON null, not as "" or 0.
    assert tasks[2]["assignee"] is None
    assert tasks[2]["completed_at"] is None


# ---------------------------------------------------------------------------
# Cycle 3 — the output-location policy
# ---------------------------------------------------------------------------
#
# The PlanSpec refuses publication into a Kanban-controlled subtree (root,
# board, workspaces, attachments), into a shared system temp root (in both
# spellings macOS uses for it), through a symlinked path component or onto a
# symlinked target, and into a group- or world-writable directory.
#
# All of those are ONE behaviour — "an unsafe output location is refused
# before the database is opened and before anything is written" — so they are
# one parametrized test. The point is the ordering: the publisher must decide
# the destination is unsafe while it still has nothing open and nothing
# written, not after it has produced an artifact it then declines to move.
#
# Every path below is synthetic. The two shared-temp cases name a directory
# that does not exist and must not come into existence.


def _out_inside_kanban_root(tmp_path, root, board):
    return root / "published" / "read-model.json"


def _out_inside_board_dir(tmp_path, root, board):
    return root / "kanban" / "boards" / board / "read-model.json"


def _out_inside_workspaces(tmp_path, root, board):
    return root / "kanban" / "workspaces" / "scratch-1" / "read-model.json"


def _out_inside_attachments(tmp_path, root, board):
    return root / "kanban" / "boards" / board / "attachments" / "t_aaa" / "read-model.json"


def _out_in_canonical_tmp(tmp_path, root, board):
    # The shared temp root itself: world-writable, and every other user on
    # the host can create the artifact's name there before the publisher can.
    # The sticky bit protects existing entries, not a name nobody holds yet.
    return Path("/tmp") / f"hermes-h0-2b3b-{os.getpid()}-read-model.json"


def _out_in_macos_tmp_alias(tmp_path, root, board):
    # The same directory under its canonical macOS spelling — `/tmp` is a
    # symlink to `private/tmp`, so a policy that only knows the `/tmp`
    # spelling is bypassed by naming the target directly.
    return Path("/private/tmp") / f"hermes-h0-2b3b-{os.getpid()}-read-model.json"


def _out_under_symlinked_component(tmp_path, root, board):
    real = tmp_path / "real-published"
    real.mkdir()
    os.chmod(real, 0o700)
    link = tmp_path / "linked-published"
    link.symlink_to(real, target_is_directory=True)
    return link / "read-model.json"


def _out_is_a_symlinked_target(tmp_path, root, board):
    # The final component itself is a planted symlink. Following it would
    # publish into `decoy.json` instead of the named path.
    decoy = tmp_path / "decoy.json"
    decoy.write_text("decoy\n")
    parent = tmp_path / "symlink-target-parent"
    parent.mkdir()
    os.chmod(parent, 0o700)
    out = parent / "read-model.json"
    out.symlink_to(decoy)
    return out


def _out_in_group_writable_dir(tmp_path, root, board):
    parent = tmp_path / "group-writable"
    parent.mkdir()
    # Explicit chmod, not mkdir(mode=...): the umask masks the mkdir mode, so
    # the bit this case exists to test might never be set.
    os.chmod(parent, 0o770)
    return parent / "read-model.json"


def _out_in_world_writable_dir(tmp_path, root, board):
    parent = tmp_path / "world-writable"
    parent.mkdir()
    os.chmod(parent, 0o707)
    return parent / "read-model.json"


def _tree_manifest(base):
    """Every path under *base* with the metadata any write would change.

    `lstat`, never `stat`, and `followlinks=False`: the manifest must describe
    the planted symlinks themselves, not whatever they point at.
    """
    manifest = {}
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        for name in dirnames + filenames:
            path = os.path.join(dirpath, name)
            st = os.lstat(path)
            manifest[path] = (st.st_mode, st.st_ino, st.st_size, st.st_mtime_ns)
    return manifest


@pytest.mark.parametrize(
    "build_out",
    [
        pytest.param(_out_inside_kanban_root, id="inside-kanban-root"),
        pytest.param(_out_inside_board_dir, id="inside-board-dir"),
        pytest.param(_out_inside_workspaces, id="inside-workspaces"),
        pytest.param(_out_inside_attachments, id="inside-attachments"),
        pytest.param(_out_in_canonical_tmp, id="in-canonical-tmp"),
        pytest.param(_out_in_macos_tmp_alias, id="in-macos-tmp-alias"),
        pytest.param(_out_under_symlinked_component, id="symlinked-component"),
        pytest.param(_out_is_a_symlinked_target, id="symlinked-target"),
        pytest.param(_out_in_group_writable_dir, id="group-writable-dir"),
        pytest.param(_out_in_world_writable_dir, id="world-writable-dir"),
    ],
)
def test_publish_read_model_refuses_an_unsafe_output_location_before_db_open_and_any_write(
    tmp_path, monkeypatch, capsys, build_out
):
    """An unsafe destination is refused while nothing is open and nothing is
    written: no `sqlite3.connect`, no write-capable open, no directory
    created, and a filesystem that is metadata-identical afterwards."""
    root = tmp_path / "kanban-root"
    root.mkdir()
    board = "synthetic"
    # A real, valid board, so the ONLY thing wrong with this invocation is
    # where it was asked to publish: without the policy the run would succeed.
    db = root / "kanban" / "boards" / board / "kanban.db"
    _seed_synthetic_board(db)

    out = build_out(tmp_path, root, board)
    before = _tree_manifest(tmp_path)

    db_connects: list[tuple] = []
    side_effects: list[tuple] = []

    def _tripwire_connect(*args, **kwargs):
        db_connects.append(args)
        raise AssertionError("publish-read-model opened the live database")

    monkeypatch.setattr(sqlite3, "connect", _tripwire_connect)

    real_open = builtins.open

    def _tripwire_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            side_effects.append((str(file), mode))
            raise AssertionError("publish-read-model opened a file for writing")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _tripwire_open)

    real_os_open = os.open
    _WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC

    def _tripwire_os_open(path, flags, *args, **kwargs):
        if flags & _WRITE_FLAGS:
            side_effects.append((str(path), flags))
            raise AssertionError("publish-read-model opened an fd for writing")
        return real_os_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", _tripwire_os_open)

    real_mkdir = os.mkdir

    def _tripwire_mkdir(path, *args, **kwargs):
        # `os.makedirs` goes through `os.mkdir`, so this covers both: creating
        # the output directory is itself a side effect of a refused run.
        side_effects.append((str(path), "mkdir"))
        raise AssertionError("publish-read-model created a directory")

    monkeypatch.setattr(os, "mkdir", _tripwire_mkdir)

    args = _parse_kanban_args(
        [
            "kanban",
            "--kanban-root", str(root),
            "--board", board,
            "publish-read-model",
            "--db", str(db),
            "--out", str(out),
            "--limit", "3",
        ]
    )
    rc = kc.kanban_command(args)

    assert rc == 1
    assert db_connects == []
    assert side_effects == []

    captured = capsys.readouterr()
    assert captured.out == ""
    # One fixed generic string for every rejected location: the caller chose
    # the path, so a distinguishable reason would say whether it exists, how
    # it is spelled, and which rule caught it.
    assert captured.err == "kanban publish-read-model: unavailable\n"

    # Byte-for-byte unchanged sandbox: no artifact, no temp file, no created
    # directory, no WAL/SHM sidecar next to the synthetic database, and the
    # planted decoy untouched.
    assert _tree_manifest(tmp_path) == before

    if tmp_path not in out.parents:
        # The two shared-temp cases name a path outside the sandbox the
        # manifest covers, so assert their absence directly.
        assert not os.path.lexists(out)


# ---------------------------------------------------------------------------
# Cycle 4 — the shared temp root is refused as a SUBTREE, not as one directory
# ---------------------------------------------------------------------------


def _canonical_shared_temp_root():
    """The real directory `/tmp` names, spelled the way the kernel resolves it.

    macOS resolves `/tmp` to `/private/tmp`; Linux resolves it to itself.
    Resolving here is what makes the case below sharp: the path it builds has
    no symlinked component and a parent only its owner can write, so every
    other output-location rule passes it and the temp-subtree rule is the only
    thing left that can refuse it.
    """
    return Path(os.path.realpath("/tmp"))


def test_publish_read_model_refuses_a_private_subdirectory_of_the_shared_temp_root(
    tmp_path, monkeypatch, capsys
):
    """A private 0700 directory inside the shared temp root is still inside the
    shared temp root, and is refused before `sqlite3.connect` and before any
    write.

    The PlanSpec forbids the temp *subtree*, not the temp *directory*: a
    per-process 0700 subdirectory is the ordinary shape of a real destination
    there (`mkdtemp`, and pytest's own `tmp_path`), so this — not the bare
    `/tmp/name.json` case — is what decides whether the rule means anything.
    Nothing about the directory being private helps: the artifact still lands
    on a filesystem the host may clear between publication and the read that
    trusts it, which is the reason the subtree is out of bounds at all.
    """
    root = tmp_path / "kanban-root"
    root.mkdir()
    board = "synthetic"
    # A real, valid board: the ONLY thing wrong with this invocation is where
    # it was asked to publish. Without the policy the run would succeed.
    db = root / "kanban" / "boards" / board / "kanban.db"
    _seed_synthetic_board(db)

    # Uniquely named, so a leftover from an earlier run can never be what this
    # test observes, and removed in `finally` whether it passes or fails.
    parent = Path(
        tempfile.mkdtemp(
            prefix=f"hermes-h0-2b3b-private-{os.getpid()}-",
            dir=_canonical_shared_temp_root(),
        )
    )
    try:
        # Explicit chmod, not the mkdtemp default: the point of this case is
        # that the directory is owner-only writable and refused anyway, so the
        # mode it is refused under has to be stated, not inherited.
        os.chmod(parent, 0o700)
        out = parent / "read-model.json"
        before = _tree_manifest(tmp_path)

        db_connects: list[tuple] = []
        side_effects: list[tuple] = []

        def _tripwire_connect(*args, **kwargs):
            db_connects.append(args)
            raise AssertionError("publish-read-model opened the live database")

        monkeypatch.setattr(sqlite3, "connect", _tripwire_connect)

        real_open = builtins.open

        def _tripwire_open(file, mode="r", *args, **kwargs):
            if any(flag in mode for flag in ("w", "a", "x", "+")):
                side_effects.append((str(file), mode))
                raise AssertionError("publish-read-model opened a file for writing")
            return real_open(file, mode, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", _tripwire_open)

        real_os_open = os.open
        _WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC

        def _tripwire_os_open(path, flags, *args, **kwargs):
            if flags & _WRITE_FLAGS:
                side_effects.append((str(path), flags))
                raise AssertionError("publish-read-model opened an fd for writing")
            return real_os_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(os, "open", _tripwire_os_open)

        real_mkdir = os.mkdir

        def _tripwire_mkdir(path, *args, **kwargs):
            side_effects.append((str(path), "mkdir"))
            raise AssertionError("publish-read-model created a directory")

        monkeypatch.setattr(os, "mkdir", _tripwire_mkdir)

        args = _parse_kanban_args(
            [
                "kanban",
                "--kanban-root", str(root),
                "--board", board,
                "publish-read-model",
                "--db", str(db),
                "--out", str(out),
                "--limit", "3",
            ]
        )
        rc = kc.kanban_command(args)

        assert rc == 1
        assert db_connects == []
        assert side_effects == []

        captured = capsys.readouterr()
        assert captured.out == ""
        # The same generic string every other refused location gets: which
        # rule caught the path is not the caller's business.
        assert captured.err == "kanban publish-read-model: unavailable\n"

        # Nothing written here, and nothing written back in the sandbox either
        # — no artifact, no dotted temp file, no WAL/SHM beside the database.
        assert not os.path.lexists(out)
        assert list(os.scandir(parent)) == []
        assert _tree_manifest(tmp_path) == before
    finally:
        # Outside `tmp_path`, so pytest will not reclaim it for us.
        shutil.rmtree(parent, ignore_errors=True)


# ---------------------------------------------------------------------------
# Cycle 5 — a failed publication never changes the previous artifact
# ---------------------------------------------------------------------------
#
# Publication is a transaction: it either replaces the artifact with a
# complete new one or leaves the previous one exactly as it was. The four
# injection points below are the four places that transaction can break —
# serializing the document, flushing the temp file, swapping it into place,
# and making that swap durable — and they are ONE behaviour, so they are one
# parametrized test.
#
# The last of them is the one that makes this worth testing: the swap has
# already happened when the directory fsync fails, so "leave the previous
# artifact alone" is no longer something the publisher gets for free by not
# having written yet. It has to actively put the previous artifact back.
#
# Every case injects `OSError`, because that is what these calls raise in
# reality. A publisher that only survives a bespoke exception type has been
# fitted to the test rather than to the failure.

_INJECTED_FAILURE = "injected publication failure"

# The previous artifact is a REAL one — `_prior_artifact_bytes(root, board)`,
# built per test because it carries that root's fingerprint. Publication may
# only replace this board's own artifact (Cycle 12), so a placeholder here
# would be refused at the destination check and every injection point below
# would go unvisited while the test still passed.
#
# Its exact bytes, mode, and inode are the assertion: same content, still
# 0600, and still the SAME file — not a rewritten copy that compares equal.


def _fail_serialization(monkeypatch):
    """The document never becomes bytes."""

    class _RefusingJson:
        @staticmethod
        def dumps(*args, **kwargs):
            raise ValueError(_INJECTED_FAILURE)

    # The module attribute, not `json.dumps` globally: this must break the
    # publisher's serialization and nothing else in the process.
    monkeypatch.setattr(krmp, "json", _RefusingJson)


def _fail_temp_file_fsync(monkeypatch):
    """The temp file is written but never reaches the disk."""
    real_fsync = os.fsync

    def _fsync(fd):
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(errno.EIO, _INJECTED_FAILURE)
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", _fsync)


def _fail_rename(monkeypatch):
    """The finished temp file never becomes the artifact."""

    def _replace(*args, **kwargs):
        raise OSError(errno.EIO, _INJECTED_FAILURE)

    monkeypatch.setattr(os, "replace", _replace)


def _fail_post_rename_directory_fsync(monkeypatch):
    """The swap happened; making it durable did not.

    The only case where the previous artifact is already gone by the time the
    failure is observed, and therefore the only one that cannot be satisfied
    by refusing early.
    """
    real_fsync = os.fsync

    def _fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(errno.EIO, _INJECTED_FAILURE)
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", _fsync)


@pytest.mark.parametrize(
    "inject_failure",
    [
        pytest.param(_fail_serialization, id="serialization"),
        pytest.param(_fail_temp_file_fsync, id="temp-file-fsync"),
        pytest.param(_fail_rename, id="rename"),
        pytest.param(
            _fail_post_rename_directory_fsync, id="post-rename-directory-fsync"
        ),
    ],
)
def test_publish_read_model_leaves_the_previous_artifact_byte_identical_on_any_publication_failure(
    publish_sandbox, monkeypatch, capsys, inject_failure
):
    """A publication that fails anywhere leaves the previous artifact exactly
    as it was — same bytes, same 0600 mode, same inode — reports the one
    generic failure, and leaves no temp or backup file behind."""
    root = publish_sandbox / "kanban-root"
    root.mkdir()
    board = "synthetic"
    db = root / "kanban" / "boards" / board / "kanban.db"
    _seed_synthetic_board(db)

    # A previous publication, seeded exactly the way a real one would leave
    # the destination: a 0700 directory holding one 0600 artifact.
    previous = _prior_artifact_bytes(root, board)
    out = _plant_target(publish_sandbox, previous)
    before = out.lstat()

    # Proof that the run reached the transaction at all: every assertion below
    # is also satisfied by a publisher that refused before opening the board,
    # so the injected failure has to be the thing that stopped it.
    db_connects: list[tuple] = []
    real_connect = sqlite3.connect

    def _recording_connect(*args, **kwargs):
        db_connects.append(args)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", _recording_connect)

    # Any write-capable open of the artifact's own name is an in-place or
    # copy fallback: the only legitimate write target is the dotted temp file
    # the transaction renames into place.
    artifact_writes: list[tuple] = []

    real_open = builtins.open

    def _record_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            if os.path.basename(str(file)) == out.name:
                artifact_writes.append((str(file), mode))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _record_open)

    real_os_open = os.open
    _WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC

    def _record_os_open(path, flags, *args, **kwargs):
        if flags & _WRITE_FLAGS and os.path.basename(str(path)) == out.name:
            artifact_writes.append((str(path), flags))
        return real_os_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", _record_os_open)

    inject_failure(monkeypatch)

    args = _parse_kanban_args(
        [
            "kanban",
            "--kanban-root", str(root),
            "--board", board,
            "publish-read-model",
            "--db", str(db),
            "--out", str(out),
            "--limit", "3",
        ]
    )
    rc = kc.kanban_command(args)

    assert rc == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    # The same generic string every other failed publication gets — which
    # step broke, and whether the path exists, is not the caller's business.
    assert captured.err == "kanban publish-read-model: unavailable\n"

    assert db_connects != []
    assert out.read_bytes() == previous
    after = out.lstat()
    assert stat.S_IMODE(after.st_mode) == krm.ARTIFACT_MODE
    # Same inode: the previous artifact was preserved or put back, not
    # re-created from a copy the publisher happened to still hold.
    assert after.st_ino == before.st_ino
    assert after.st_mtime_ns == before.st_mtime_ns

    # No dotted temp file, no backup, nothing but the artifact itself.
    assert sorted(entry.name for entry in os.scandir(out.parent)) == [out.name]
    assert artifact_writes == []


# ---------------------------------------------------------------------------
# Cycle 6 — the destination parent cannot be swapped out after validation
# ---------------------------------------------------------------------------
#
# `_validated_output_location` proves the destination is safe by walking it
# one component at a time with `openat`/`O_NOFOLLOW`. That proof is about
# INODES. It stops meaning anything the moment the publisher throws the fds
# away and re-derives the directory from the pathname, because between the
# answer and the write an attacker who can create a name in any ancestor can
# move the validated directory aside and leave a symlink to their own
# directory in its place.
#
# `O_NOFOLLOW` does not save a path-based re-open here: it constrains only the
# LAST component, so a swapped grandparent is followed silently and the
# publication lands wherever the attacker pointed it — including next to, or
# on top of, a `kanban.db`.
#
# All three shapes below are ONE behaviour — "a destination parent swapped
# after validation cannot redirect publication" — so they are one parametrized
# test. Everything is synthetic and lives in the private 0700 sandbox outside
# the shared temp subtree, so no real board, root, or `/tmp` name is involved.

# Planted in the foreign directory the swap points at. A single byte-for-byte
# check over each sentinel catches a redirected write even if the surrounding
# manifest were ever loosened.
_FOREIGN_SENTINEL = b"FOREIGN-SENTINEL-MUST-NOT-BE-TOUCHED\n"

# The artifact a previous, legitimate publication already left in the
# destination the publisher validated is built per test by
# `_prior_artifact_bytes(root, board)`: publication may only replace this
# board's own artifact (Cycle 12), so a placeholder would be refused before
# the swap could ever be attempted. A swap must not cost the owner this file.


def _plant_validated_destination(sandbox, root, board):
    """`<sandbox>/pin/dest/read-model.json`, holding a previous artifact.

    Two directory levels, because one is not enough to express the bug: the
    publisher's own `O_NOFOLLOW` already refuses a symlink at the parent's own
    name, and only an ancestor above it shows what a path-based re-open
    actually resolves.
    """
    pin = sandbox / "pin"
    dest = pin / "dest"
    dest.mkdir(parents=True)
    # Explicit, not inherited: both levels are owner-only, so nothing here is
    # refused by the writability rule and the swap is the only thing at issue.
    os.chmod(pin, 0o700)
    os.chmod(dest, 0o700)

    out = dest / "read-model.json"
    prior = _prior_artifact_bytes(root, board)
    out.write_bytes(prior)
    os.chmod(out, krm.ARTIFACT_MODE)
    return pin, dest, out, prior


def _plant_sentinels(holder, target_name):
    """Seed *holder* with the two names a redirected publication would hit.

    `kanban.db` is the one the PlanSpec names outright — a publisher that can
    be steered into a directory it does not own can be steered onto the live
    database — and the requested target name is the one the transaction
    actually renames over.
    """
    sentinels = []
    for name in ("kanban.db", target_name):
        path = holder / name
        path.write_bytes(_FOREIGN_SENTINEL)
        os.chmod(path, 0o600)
        sentinels.append(path)
    return sentinels


def _swap_grandparent_onto_a_foreign_dir_holding_the_parent_name(sandbox, root, board):
    """The foreign directory already contains the validated parent's name.

    The sharpest shape: after the swap the requested pathname resolves, with
    no missing components, onto a directory full of someone else's files, so a
    path-based re-open succeeds and the transaction renames straight over the
    foreign artifact.
    """
    pin, dest, out, prior = _plant_validated_destination(sandbox, root, board)
    foreign = sandbox / "foreign"
    foreign.mkdir()
    os.chmod(foreign, 0o700)
    holder = foreign / dest.name
    holder.mkdir()
    os.chmod(holder, 0o700)
    sentinels = _plant_sentinels(holder, out.name)
    moved = sandbox / "pin-moved-aside"

    def swap():
        # Atomic: there is no instant where the pathname is absent, so a
        # publisher that re-derives the directory from it sees a directory
        # every time it looks and never notices the substitution.
        os.rename(pin, moved)
        os.symlink(foreign, pin)

    return {
        "out": out,
        "prior": prior,
        "swap": swap,
        "swapped": pin,
        "foreign": foreign,
        "sentinels": sentinels,
        "real_parent": moved / dest.name,
    }


def _swap_grandparent_onto_a_foreign_dir_missing_the_parent_name(sandbox, root, board):
    """The foreign directory does NOT contain the validated parent's name.

    A publisher that creates its output directory from the pathname makes the
    missing level itself, so this case is about creation rather than
    overwriting: the foreign directory must not gain a single entry.
    """
    pin, dest, out, prior = _plant_validated_destination(sandbox, root, board)
    foreign = sandbox / "foreign"
    foreign.mkdir()
    os.chmod(foreign, 0o700)
    sentinels = _plant_sentinels(foreign, out.name)
    moved = sandbox / "pin-moved-aside"

    def swap():
        os.rename(pin, moved)
        os.symlink(foreign, pin)

    return {
        "out": out,
        "prior": prior,
        "swap": swap,
        "swapped": pin,
        "foreign": foreign,
        "sentinels": sentinels,
        "real_parent": moved / dest.name,
    }


def _swap_the_validated_parent_itself_onto_a_foreign_dir(sandbox, root, board):
    """The validated parent's own name becomes the symlink.

    Held to the same outcome as the ancestor cases: the publication is refused
    and the previous artifact survives. A publisher may not "succeed" into a
    directory the caller's own `--out` no longer reaches.
    """
    pin, dest, out, prior = _plant_validated_destination(sandbox, root, board)
    foreign = sandbox / "foreign"
    foreign.mkdir()
    os.chmod(foreign, 0o700)
    sentinels = _plant_sentinels(foreign, out.name)
    moved = pin / "dest-moved-aside"

    def swap():
        os.rename(dest, moved)
        os.symlink(foreign, dest)

    return {
        "out": out,
        "prior": prior,
        "swap": swap,
        "swapped": dest,
        "foreign": foreign,
        "sentinels": sentinels,
        "real_parent": moved,
    }


@pytest.mark.parametrize(
    "plant_swap",
    [
        pytest.param(
            _swap_grandparent_onto_a_foreign_dir_holding_the_parent_name,
            id="grandparent-swapped-foreign-holds-parent-name",
        ),
        pytest.param(
            _swap_grandparent_onto_a_foreign_dir_missing_the_parent_name,
            id="grandparent-swapped-foreign-missing-parent-name",
        ),
        pytest.param(
            _swap_the_validated_parent_itself_onto_a_foreign_dir,
            id="validated-parent-itself-swapped",
        ),
    ],
)
def test_publish_read_model_cannot_be_redirected_by_a_destination_parent_swapped_after_validation(
    publish_sandbox, monkeypatch, capsys, plant_swap
):
    """A destination parent replaced by a symlink after the output-location
    check cannot redirect publication: nothing in the foreign directory is
    modified or created, the swapped pathname is never followed, the artifact
    already in the validated directory survives byte-for-byte, the run fails
    closed with the one generic message, and no temp or backup file is left
    behind."""
    root = publish_sandbox / "kanban-root"
    root.mkdir()
    board = "synthetic"
    # A real, valid board and a destination that passes every output-location
    # rule: without the pinning requirement this invocation would publish.
    db = root / "kanban" / "boards" / board / "kanban.db"
    _seed_synthetic_board(db)

    case = plant_swap(publish_sandbox, root, board)
    out = case["out"]
    foreign_before = _tree_manifest(case["foreign"])
    artifact_before = out.lstat()

    # Proof that the run reached the transaction the swap targets: a publisher
    # that refused before opening the board would satisfy every assertion
    # below without the pinning rule ever being consulted.
    db_connects: list[tuple] = []
    real_connect = sqlite3.connect

    def _recording_connect(*args, **kwargs):
        db_connects.append(args)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", _recording_connect)

    swaps: list[str] = []
    real_validate = krmp._validated_output_location

    def _validate_then_swap(out_path):
        """The PlanSpec's window, reproduced exactly once.

        The real policy runs first and whatever it returns is passed back
        untouched — this seam adds no leniency and asserts nothing about the
        answer's shape. Only the filesystem moves underneath, at the one
        instant that matters: after the destination has been judged safe and
        before the publication transaction creates its first file.
        """
        result = real_validate(out_path)
        case["swap"]()
        swaps.append(str(out_path))
        return result

    monkeypatch.setattr(krmp, "_validated_output_location", _validate_then_swap)

    args = _parse_kanban_args(
        [
            "kanban",
            "--kanban-root", str(root),
            "--board", board,
            "publish-read-model",
            "--db", str(db),
            "--out", str(out),
            "--limit", "3",
        ]
    )
    rc = kc.kanban_command(args)

    # The exploit has to have actually been staged, or every assertion below
    # would pass against a publication nobody attacked.
    assert swaps == [str(out)]
    assert db_connects != []

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    # The same generic string every other refused publication gets.
    assert captured.err == "kanban publish-read-model: unavailable\n"

    # The planted symlink is untouched: it was neither followed into nor
    # replaced by a directory or an artifact.
    swapped = case["swapped"]
    assert os.path.islink(swapped)
    assert os.readlink(swapped) == str(case["foreign"])

    # Nothing in the foreign directory changed, and nothing was created there
    # — no artifact, no temp file, no interposed parent directory.
    assert _tree_manifest(case["foreign"]) == foreign_before
    for sentinel in case["sentinels"]:
        assert sentinel.read_bytes() == _FOREIGN_SENTINEL

    # The owner's previous artifact, in the directory that was actually
    # validated, is the SAME inode with the same bytes and the same mode.
    survivor = case["real_parent"] / out.name
    artifact_after = survivor.lstat()
    assert survivor.read_bytes() == case["prior"]
    assert stat.S_IMODE(artifact_after.st_mode) == krm.ARTIFACT_MODE
    assert artifact_after.st_ino == artifact_before.st_ino
    assert artifact_after.st_mtime_ns == artifact_before.st_mtime_ns

    # No dotted temp file and no backup link: the validated directory holds
    # the previous artifact and nothing else.
    assert sorted(entry.name for entry in os.scandir(case["real_parent"])) == [
        out.name
    ]


# ---------------------------------------------------------------------------
# Cycle 7 — a concurrent reader sees only the previous or the next artifact
# ---------------------------------------------------------------------------
#
# The publication transaction is written to be atomic from a reader's point of
# view: the bytes go to a dotted `O_EXCL` temp file in the destination
# directory and reach the artifact's own name only through `rename`. Every
# claim in that sentence is about what a reader OBSERVES, and none of the
# cycles above observe anything — they assert on the directory after the
# publisher has finished, with nobody reading.
#
# This one runs the real H0 reader against the real publisher while the
# publisher is mid-transaction, and holds every observation to the contract:
#
#   * every successful read is byte-identical to one COMPLETE published
#     artifact — never a prefix, never one document's head with another's
#     tail, never an empty or half-flushed temp file;
#   * every one of those observations parses and validates through the real
#     reader path, so "complete" means "the reader accepts it", not merely
#     "the length matched";
#   * the reader is never sent somewhere else — no dotted temp file, no
#     backup link, no orphaned name; and
#   * the destination is left holding exactly the artifact, with no temp or
#     backup file surviving the run.
#
# The one failure a read may legitimately hit is spelled out rather than
# waved through. Between `os.link` and `os.replace` the previous artifact
# carries the publisher's rollback link, so its `st_nlink` is 2 for those two
# syscalls and the reader's link check refuses it. That is a fail-CLOSED
# availability blip — a generic `ReadModelUnavailable`, never a wrong, stale,
# or partial answer — and it is the cost of the rollback guarantee Cycle 5
# tests: the only way to keep the previous artifact's inode restorable without
# it is to rename that artifact aside first, which would expose the missing
# name this invariant forbids outright. Any OTHER failure means a reader
# reached a file the transaction should never have shown it, so the allowed
# set is exactly one string.
_ALLOWED_RACE_FAILURE = "artifact has more than one link"

# Enough back-to-back publications that reads land inside the transaction
# window rather than only before and after it, and still ~0.2s of publishing.
_STRESS_ROUNDS = 64
_STRESS_READERS = 4


def _republish_distinct(root, board, db, out, round_index):
    """Publish one artifact that differs from every other round's.

    The titles are rewritten first so the new document differs in its BODY,
    not only in `generated_at` — several rounds share a wall-clock second, and
    a mixed read that spliced two same-second documents together would be
    invisible if the stamp were the only thing that moved.
    """
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("UPDATE tasks SET title = ?", (f"round-{round_index:03d}",))
        conn.commit()
    finally:
        conn.close()
    krmp.publish_read_model(
        kanban_root=root, board=board, db=db, out=out, limit=3
    )
    return out.read_bytes()


def test_a_concurrent_reader_observes_only_a_complete_previous_or_next_artifact(
    publish_sandbox
):
    """While the publisher commits, a reader sees one whole artifact or the
    other — never partial or mixed JSON, never an orphaned or redirected
    target — and the only failure it may hit is the generic unavailable at the
    rollback-link window."""
    root = publish_sandbox / "kanban-root"
    root.mkdir()
    board = "synthetic"
    db = root / "kanban" / "boards" / board / "kanban.db"
    _seed_synthetic_board(db)

    out = publish_sandbox / "published" / "read-model.json"

    # The "previous" artifact: a real publication, not a hand-written file, so
    # what the readers start from is exactly what the publisher produces.
    krmp.publish_read_model(
        kanban_root=root, board=board, db=db, out=out, limit=3
    )
    previous = out.read_bytes()
    published = {previous}

    # The publisher is called directly rather than through `kanban_command`
    # for one reason: the CLI writes to the captured streams, and a stress
    # loop sharing `capsys` with four live threads would be measuring pytest's
    # capture plumbing as much as the publication transaction.

    lock = threading.Lock()
    observed: set[bytes] = set()
    failures: collections.Counter = collections.Counter()
    crashes: list[str] = []
    reads = 0

    stop = threading.Event()
    # Readers and publisher meet here, so publication starts only once every
    # reader is already in its loop — the reads straddle the commit points
    # instead of queueing up behind thread startup.
    start = threading.Barrier(_STRESS_READERS + 1)

    def _observe(raw):
        nonlocal reads
        with lock:
            reads += 1
            observed.add(raw)

    def _read_loop():
        try:
            # One read before the barrier, while nothing is publishing: this
            # makes "the previous artifact was observed" a fact rather than a
            # timing accident, so the assertion below can be exact.
            _observe(krm.read_artifact_bytes(out))
            start.wait(timeout=30)
            while not stop.is_set():
                try:
                    raw = krm.read_artifact_bytes(out)
                except krm.ReadModelUnavailable as exc:
                    with lock:
                        failures[str(exc)] += 1
                    continue
                _observe(raw)
        except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
            # A thread that dies takes its traceback with it, so anything the
            # reader did not raise as `ReadModelUnavailable` is recorded and
            # asserted on in the main thread instead of vanishing into a
            # silently short run.
            with lock:
                crashes.append(repr(exc))

    threads = [
        threading.Thread(target=_read_loop, name=f"h0-reader-{index}")
        for index in range(_STRESS_READERS)
    ]
    for thread in threads:
        thread.start()
    try:
        start.wait(timeout=30)
        for round_index in range(_STRESS_ROUNDS):
            published.add(
                _republish_distinct(root, board, db, out, round_index)
            )
        latest = out.read_bytes()

        # Wait for the last publication to actually be seen rather than
        # sleeping a guessed interval: that is what makes "a reader observed
        # the next artifact" an assertion instead of a hope.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            with lock:
                if latest in observed:
                    break
            time.sleep(0.005)
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=30)

    assert crashes == []
    assert not any(thread.is_alive() for thread in threads)

    # The run has to have been a real race, or every assertion below would
    # hold trivially against reads that never met a publication.
    assert reads >= _STRESS_ROUNDS, f"only {reads} reads across {_STRESS_ROUNDS} rounds"
    assert len(published) == _STRESS_ROUNDS + 1, "a round republished identical bytes"

    # Both ends of the contract were actually observed: the artifact that was
    # there before publication started, and the one left after it finished.
    assert previous in observed
    assert latest in observed
    assert len(observed) >= 2

    # The core invariant. Set membership is exact BYTE equality against the
    # complete documents the publisher emitted, so a prefix, a splice of two
    # rounds, or an empty read is caught by construction.
    unexpected = observed - published
    assert unexpected == set(), (
        f"{len(unexpected)} read(s) matched no complete published artifact: "
        f"{sorted(raw[:120] for raw in unexpected)}"
    )

    # ...and "complete" is decided by the reader, not by the comparison above:
    # every distinct observation is pushed through the real JSON-validation
    # path, which refuses truncated JSON, duplicate keys, an unknown schema,
    # and a row outside the allowlist.
    now = int(time.time())
    fingerprint = krm.root_fingerprint(root)
    for raw in observed:
        document = krm.parse_artifact_json(raw)
        validated = krm.validate_artifact(
            document,
            board=board,
            expected_fingerprint=fingerprint,
            now=now,
            max_age_seconds=krm.DEFAULT_MAX_AGE_SECONDS,
        )
        assert validated["schema_version"] == krm.SCHEMA_VERSION
        assert validated["board"] == board
        assert [task["id"] for task in validated["tasks"]] == [
            "t_aaa", "t_bbb", "t_ccc",
        ]
        # Nothing the transaction exposed mid-flight carried an excluded
        # column either: a reader that raced the publisher saw no more than a
        # reader that did not.
        assert _SENSITIVE not in raw.decode("utf-8")

    # Every failure was the one documented race boundary. Not asserted to be
    # non-empty: whether a read lands inside a two-syscall window is timing,
    # and a test that REQUIRES the blip would fail on a machine fast enough to
    # miss it.
    assert set(failures) <= {_ALLOWED_RACE_FAILURE}, dict(failures)

    # The destination holds the artifact and nothing else — no dotted temp
    # file, no leftover rollback link — and the artifact is still the only
    # thing the reader's own checks will accept.
    assert sorted(entry.name for entry in os.scandir(out.parent)) == [out.name]
    assert not os.path.islink(out)
    assert stat.S_IMODE(out.lstat().st_mode) == krm.ARTIFACT_MODE
    assert out.lstat().st_nlink == 1
    assert out.read_bytes() == latest


# ---------------------------------------------------------------------------
# Cycle 8 — end to end: the owner publishes, H0 reads, H0 opens no database
# ---------------------------------------------------------------------------
#
# Every cycle above tests one side of the boundary in isolation, and the H0
# reader's own suite proves it touches nothing — but it proves that against a
# HAND-WRITTEN artifact. That leaves the one claim the whole architecture
# rests on untested: that the artifact a real publication produces is the
# artifact the real reader accepts, and that consuming it costs H0 nothing.
#
# A reader suite alone cannot show this. It could pass against a publisher
# that emits a document no reader will take, and a publisher suite could pass
# against a reader that quietly falls back to the live database when the
# artifact disappoints it. Only running both in one process, against one real
# board, in that order, closes the gap.
#
# So this joins the two halves once:
#
#   1. an explicit owner publication from a real WAL-mode SQLite board with a
#      writer still connected — the exact shape that made the rejected
#      live-read design mutate `kanban.db-shm` just by reading;
#   2. then the real `read-model` CLI over what that publication produced.
#
# The instrumentation is split along the same seam, because the two phases are
# held to opposite standards. The publisher is SUPPOSED to connect to the
# database and stat it — that is its authority — so its events are recorded,
# asserted to be non-empty, and then cleared. Only afterwards is the tripwire
# armed, and from that point a database connection or any touch below the
# Kanban root is not merely recorded but REFUSED, so the boundary is enforced
# during the read rather than described after it.


def _live_wal_synthetic_board(db_path):
    """Seed the synthetic board and leave a WAL writer connected to it.

    Returns the still-open connection, which is what keeps `kanban.db-wal` and
    `kanban.db-shm` on disk: an idle WAL database has neither, so a reader that
    CREATED them would look identical to one that left the root alone.
    """
    _seed_synthetic_board(db_path)

    connection = sqlite3.connect(str(db_path))
    connection.execute("PRAGMA journal_mode=WAL")
    # A committed no-op write, so the WAL carries real frames rather than
    # being an empty file the reader could recreate byte-identically.
    connection.execute("UPDATE tasks SET priority = priority")
    connection.commit()

    for suffix in ("", "-wal", "-shm"):
        sidecar = Path(str(db_path) + suffix)
        assert sidecar.exists(), sidecar
    return connection


def _db_manifest(db_path):
    """Inode, mode, size, mtime_ns and SHA-256 of the database and both
    sidecars — enough to catch a mutation that preserves size or timestamps."""
    manifest = {}
    for suffix in ("", "-wal", "-shm"):
        path = Path(str(db_path) + suffix)
        st = path.lstat()
        manifest[suffix or "db"] = (
            st.st_ino,
            st.st_mode,
            st.st_size,
            st.st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    return manifest


def _traced_target(path):
    """The pathname an event names, normalized for comparison.

    `os.fspath`, because a tripwire that recognised only `str` would miss
    every `pathlib` call — which is most of the reader's.
    """
    if isinstance(path, int):
        return f"<fd:{path}>"
    try:
        target = os.fspath(path)
    except TypeError:
        return repr(path)
    return os.fsdecode(target) if isinstance(target, bytes) else target


class _PhaseTracer:
    """Records database and filesystem events for one phase at a time.

    While ``phase`` is ``"h0"`` this is a tripwire, not a log: a database
    connection or any touch BELOW the Kanban root raises rather than being
    noted, so the offending call never completes.

    The Kanban root ITSELF is deliberately not refused. `read_read_model`
    lstats the root's own path components to bind the artifact to one root,
    which is the documented, filesystem-free half of the contract; what it may
    never do is descend into the root, where `kanban.db` and its sidecars live.
    The assertions below therefore pin that single root lstat exactly rather
    than allowing "something under the root, but not much".
    """

    def __init__(self, root):
        self.root = str(root)
        self._below_root = str(root) + os.sep
        self.phase = "publisher"
        self.events = []

    def is_below_root(self, target):
        return target.startswith(self._below_root)

    def record(self, kind, target):
        self.events.append((kind, target))
        if self.phase == "h0" and (
            kind == "sqlite3.connect" or self.is_below_root(target)
        ):
            raise AssertionError(f"H0 read touched the live board: {kind} {target}")

    def drain(self):
        """Return this phase's events and reset the counters for the next."""
        drained, self.events = self.events, []
        return drained


def test_owner_publication_feeds_the_h0_reader_without_h0_touching_the_board(
    publish_sandbox, monkeypatch, capsys
):
    """One real publication, then one real `read-model`: the reader serves the
    published board exactly, while making zero database connections, opening
    and stat-ing nothing below the Kanban root, and leaving the live database,
    its WAL and its SHM byte-for-byte identical."""
    root = publish_sandbox / "kanban-root"
    root.mkdir()
    board = "synthetic"
    db = root / "kanban" / "boards" / board / "kanban.db"
    writer = _live_wal_synthetic_board(db)

    # Outside the Kanban root, as both halves of the contract require.
    out = publish_sandbox / "published" / "read-model.json"
    assert root not in out.parents

    # Owner authority is explicit: no delegated-child marker anywhere in the
    # environment, so the publication below is authorized rather than assumed.
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)

    tracer = _PhaseTracer(root)
    real_connect = sqlite3.connect
    real_os_open = os.open
    real_os_stat = os.stat
    real_os_lstat = os.lstat
    real_builtin_open = builtins.open

    def _traced_connect(*args, **kwargs):
        tracer.record("sqlite3.connect", _traced_target(args[0]) if args else "")
        return real_connect(*args, **kwargs)

    def _traced_os_open(path, flags, *args, **kwargs):
        tracer.record("os.open", _traced_target(path))
        return real_os_open(path, flags, *args, **kwargs)

    def _traced_os_stat(path, *args, **kwargs):
        tracer.record("os.stat", _traced_target(path))
        return real_os_stat(path, *args, **kwargs)

    def _traced_os_lstat(path, *args, **kwargs):
        tracer.record("os.lstat", _traced_target(path))
        return real_os_lstat(path, *args, **kwargs)

    def _traced_builtin_open(file, *args, **kwargs):
        tracer.record("open", _traced_target(file))
        return real_builtin_open(file, *args, **kwargs)

    try:
        # --- phase 1: the authorized owner publication -------------------
        monkeypatch.setattr(sqlite3, "connect", _traced_connect)
        monkeypatch.setattr(os, "open", _traced_os_open)
        monkeypatch.setattr(os, "stat", _traced_os_stat)
        monkeypatch.setattr(os, "lstat", _traced_os_lstat)
        monkeypatch.setattr(builtins, "open", _traced_builtin_open)

        limit = 3
        before = int(time.time())
        publish_rc = kc.kanban_command(
            _parse_kanban_args(
                [
                    "kanban",
                    "--kanban-root", str(root),
                    "--board", board,
                    "publish-read-model",
                    "--db", str(db),
                    "--out", str(out),
                    "--limit", str(limit),
                ]
            )
        )
        after = int(time.time())

        published = capsys.readouterr()
        assert publish_rc == 0, f"publish failed: {published.err}"
        assert published.err == ""

        # The publisher's own events, asserted before they are cleared. They
        # are what makes the reader's empty set meaningful: the same tripwire,
        # over the same board, DOES see a connection and a stat of the live
        # database when the authorized half of the boundary runs.
        publisher_events = tracer.drain()
        # Exactly one connection, in the URI spelling the publisher requires:
        # `mode=rw` opens this database and never creates one.
        assert [event for event in publisher_events
                if event[0] == "sqlite3.connect"] == [
            ("sqlite3.connect", f"{db.as_uri()}?mode=rw")
        ]
        # Nothing below the Kanban root is named by absolute pathname at all:
        # the walk to the board directory is `openat`-relative from the
        # filesystem anchor down, and so is the database open itself.
        assert [event for event in publisher_events
                if tracer.is_below_root(event[1])] == []
        # Which is where the publisher's reach into the board shows up instead
        # — the `O_NOFOLLOW` open of the final component against the pinned
        # board-directory fd, the one that binds the connection above to a
        # single inode. The same tripwire sees nothing of the sort from H0.
        assert ("os.open", db.name) in publisher_events

        # --- phase 2: the real H0 read, with the tripwire armed ----------
        db_before = _db_manifest(db)
        # That snapshot is the TEST's own read of the board, not the CLI's, so
        # it is dropped rather than counted: the armed phase has to start from
        # zero or the reader's empty event set would not be the reader's.
        tracer.drain()
        tracer.phase = "h0"

        read_rc = kc.kanban_command(
            _parse_kanban_args(
                [
                    "kanban",
                    "--kanban-root", str(root),
                    "--board", board,
                    "read-model",
                    "--artifact", str(out),
                    "--limit", "50",
                    "--json",
                ]
            )
        )
        tracer.phase = "done"
        served = capsys.readouterr()
        h0_events = tracer.drain()
        # Taken here, not after the teardown below: closing the last
        # connection checkpoints the WAL away, so a snapshot outside the
        # writer's lifetime would compare the H0 phase against a database
        # this test dismantled itself.
        db_after = _db_manifest(db)
    finally:
        writer.close()

    # The invariant, asserted before the outcome: a tripped wire surfaces as
    # the generic `unavailable` failure, so checking rc first would report
    # "the read failed" for what is actually a boundary violation.
    tripped = [
        event for event in h0_events
        if event[0] == "sqlite3.connect" or tracer.is_below_root(event[1])
    ]
    assert tripped == [], f"H0 connected to or descended into the board: {tripped}"
    # Caught separately, because a relative pathname opened against a
    # directory fd would name `kanban.db` without carrying the root prefix.
    assert [event for event in h0_events if "kanban.db" in event[1]] == []

    # The one Kanban-root-side touch H0 is allowed: lstat-ing the root's own
    # path to bind the artifact to it. Pinned exactly — not merely bounded —
    # so a future reader cannot start stat-ing the root repeatedly, or descend
    # one level, without this failing.
    assert [event for event in h0_events if event[1] == str(root)] == [
        ("os.lstat", str(root))
    ]

    # Positive control: the tripwire was installed and live during the read.
    # Without this, a monkeypatch that silently failed to apply would make
    # every assertion above pass while proving nothing.
    assert [event for event in h0_events if event[0] == "os.open"], h0_events

    assert read_rc == 0, f"H0 read failed: {served.err}"
    assert served.err == ""

    # The database, its WAL and its SHM are byte-for-byte what they were when
    # the publisher finished: same inode, mode, size, mtime and SHA-256.
    assert db_after == db_before

    document = json.loads(out.read_bytes().decode("utf-8"))
    # Binding and freshness, as the publisher stamped them.
    assert document["board"] == board
    assert document["root_fingerprint"] == krm.root_fingerprint(root)
    assert before <= document["generated_at"] <= after
    assert after - document["generated_at"] < krm.DEFAULT_MAX_AGE_SECONDS
    assert stat.S_IMODE(out.stat().st_mode) == krm.ARTIFACT_MODE

    # The served payload: the allowlist and nothing else. Six eligible rows
    # were published under `--limit 3`, so `truncated` stays true through the
    # reader even though the READ asked for 50 — it reports the publisher's
    # trim, not its own.
    assert json.loads(served.out) == {
        "schema_version": krm.SCHEMA_VERSION,
        "generated_at": document["generated_at"],
        "board": board,
        "truncated": True,
        "tasks": [
            {
                "id": "t_aaa", "title": "highest", "status": "done",
                "assignee": "bob", "priority": 9, "created_at": 1600,
                "started_at": 1720, "completed_at": 1730,
            },
            {
                "id": "t_bbb", "title": "tie winner", "status": "running",
                "assignee": "alice", "priority": 5, "created_at": 1600,
                "started_at": 1700, "completed_at": None,
            },
            {
                "id": "t_ccc", "title": "tie middle", "status": "review",
                "assignee": None, "priority": 5, "created_at": 1600,
                "started_at": 1710, "completed_at": None,
            },
        ],
    }
    # `root_fingerprint` binds the artifact; it is not output. Nothing from an
    # excluded column reaches the terminal either.
    assert "root_fingerprint" not in served.out
    assert _SENSITIVE not in served.out


# ---------------------------------------------------------------------------
# Cycle 9 — the publisher opens the explicit board's database and nothing else
# ---------------------------------------------------------------------------
#
# `--db` is an explicit input, but "explicit" is not the same as "bound". The
# artifact stamps `board` and `root_fingerprint(root)`, and a reader trusts
# both. So the database behind those two stamps has to be the one location the
# repository layout gives that board under that root — not merely *a* file the
# caller named that happens to sit somewhere below the root.
#
# Everything that can put a different database behind the same name is one
# case here: a sibling board's real database, a symlink standing in for the
# final component, a symlinked board directory (whose path spelling is
# lexically indistinguishable from the honest one), a board directory other
# users can write, a chain not owned by the caller at all, and the final
# component replaced *after* validation but before SQLite opens it.
#
# The evidence is deliberately not "an error was raised". Each case seeds a
# second, fully valid board whose rows carry a marker, and then pins three
# separate facts: no artifact exists at all (so the marker reached nothing —
# this is what actually fails today, with the foreign rows published under this
# board's name), the foreign database is byte-, mtime- and sidecar-identical
# afterwards, and no `sqlite3.connect` ever named anything but this board's own
# canonical path.
#
# Same-UID replacement is explicitly out of the threat model: a process running
# as the owner can always rename the board database. What is in scope is
# another user, and path confusion — and both are refused.

_FOREIGN_MARKER = "FOREIGN-BOARD-MUST-NOT-BE-PUBLISHED"


def _seed_foreign_board(db_path):
    """Seed a second, fully valid WAL board nothing may ever publish from.

    WAL on purpose, because that is the shape a real Hermes board has: a
    publisher redirected onto it would open it read-write and leave `-wal` and
    `-shm` behind on any failure short of a clean close.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(_SYNTHETIC_TASKS_DDL)
        conn.execute(
            "INSERT INTO tasks (id, title, status, priority, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            ("t_foreign", _FOREIGN_MARKER, "todo", 99, 1600),
        )
        conn.commit()
    finally:
        conn.close()


def _foreign_db_manifest(db_path):
    """Content, mtime and sidecars of a board database, as one comparable."""
    st = os.stat(db_path)
    return (
        hashlib.sha256(db_path.read_bytes()).hexdigest(),
        st.st_mtime_ns,
        st.st_size,
        sorted(p.name for p in db_path.parent.iterdir()),
    )


def _connect_target_path(target):
    """The filesystem path a `sqlite3.connect` target names, URI or plain."""
    if target.startswith("file:"):
        parsed = urllib.parse.urlparse(target)
        return Path(urllib.parse.unquote(parsed.path))
    return Path(target)


@pytest.mark.parametrize(
    "case",
    [
        "another_boards_regular_db",
        "final_component_symlink",
        "symlinked_board_directory",
        "group_writable_board_directory",
        "board_chain_not_owned_by_caller",
        "final_component_swapped_after_validation",
    ],
)
def test_publish_read_model_opens_only_the_explicit_boards_own_database(
    case, publish_sandbox, monkeypatch, capsys
):
    """No spelling, link, permission or swap may put another database behind
    the board the artifact claims to describe."""
    root = publish_sandbox / "kanban-root"
    board = "synthetic"
    board_dir = root / "kanban" / "boards" / board
    canonical_db = board_dir / "kanban.db"
    _seed_synthetic_board(canonical_db)
    # Stated, not inherited: every directory from the Kanban root down is the
    # owner's alone, so no case below passes or fails on the umask the suite
    # happened to run under.
    for directory in (root, root / "kanban", root / "kanban" / "boards", board_dir):
        os.chmod(directory, 0o700)

    out = publish_sandbox / "published" / "read-model.json"
    db_arg = canonical_db

    if case == "another_boards_regular_db":
        # A real, honest database — of the WRONG board, under the right root.
        foreign_db = root / "kanban" / "boards" / "other" / "kanban.db"
        _seed_foreign_board(foreign_db)
        db_arg = foreign_db
    elif case == "final_component_symlink":
        foreign_db = publish_sandbox / "foreign" / "kanban.db"
        _seed_foreign_board(foreign_db)
        canonical_db.unlink()
        os.symlink(foreign_db, canonical_db)
    elif case == "symlinked_board_directory":
        # The `--db` spelling stays character-for-character canonical; only
        # the board DIRECTORY is a link. Nothing lexical can see this.
        foreign_dir = publish_sandbox / "foreign-board"
        foreign_db = foreign_dir / "kanban.db"
        _seed_foreign_board(foreign_db)
        os.chmod(foreign_dir, 0o700)
        shutil.rmtree(board_dir)
        os.symlink(foreign_dir, board_dir)
    elif case == "group_writable_board_directory":
        # Any member of the group can swap `kanban.db` at will, so the file
        # found there is not evidence of anything.
        foreign_db = publish_sandbox / "foreign" / "kanban.db"
        _seed_foreign_board(foreign_db)
        os.chmod(board_dir, 0o770)
    elif case == "board_chain_not_owned_by_caller":
        foreign_db = publish_sandbox / "foreign" / "kanban.db"
        _seed_foreign_board(foreign_db)
        # Every component below the root now reads as another user's, which
        # is the one thing an ownership check exists to notice.
        monkeypatch.setattr(os, "getuid", lambda: os.stat(root).st_uid + 1)
    else:
        foreign_db = publish_sandbox / "foreign" / "kanban.db"
        _seed_foreign_board(foreign_db)

    foreign_before = _foreign_db_manifest(foreign_db)

    connect_targets = []
    real_connect = sqlite3.connect

    def _recording_connect(target, *args, **kwargs):
        if case == "final_component_swapped_after_validation":
            # Validation is over and SQLite has not opened anything yet: the
            # exact instant a path-based open can still be redirected.
            canonical_db.unlink()
            os.symlink(foreign_db, canonical_db)
        connect_targets.append(str(target))
        return real_connect(target, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", _recording_connect)

    args = _parse_kanban_args(
        [
            "kanban",
            "--kanban-root", str(root),
            "--board", board,
            "publish-read-model",
            "--db", str(db_arg),
            "--out", str(out),
            "--limit", "50",
        ]
    )
    rc = kc.kanban_command(args)

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    # One generic string: the caller controls these paths, so a distinguishable
    # reason would turn the refusal into a filesystem oracle.
    assert captured.err.strip() == "kanban publish-read-model: unavailable"

    # Nothing published, and no directory created on the way to not publishing.
    assert not out.exists()
    assert not out.parent.exists()

    # The foreign board is untouched: same bytes, same mtime, no `-wal` or
    # `-shm` left beside it.
    assert _foreign_db_manifest(foreign_db) == foreign_before

    # And no connection ever NAMED another board's database either.
    for target in connect_targets:
        assert _connect_target_path(target) == canonical_db


# ---------------------------------------------------------------------------
# Cycle 10 — the `tasks` source must BE the contracted table
# ---------------------------------------------------------------------------
#
# Cycle 9 bound the artifact to the right *file*. That still leaves the object
# inside it unbound: the allowlisted SELECT names `tasks`, and SQLite will
# happily resolve that name to a view, or to a table whose columns merely
# happen to be spelled the same while meaning something else. Neither is the
# table the read model is a projection of.
#
# Row validation cannot close this. `_sanitized_task` only ever sees rows that
# came back, so every case below is seeded EMPTY on purpose: a view over an
# empty base, a `created_at` that is really text, a `status` that may be NULL,
# an `id` that is not the primary key. Each one returns zero rows, sails
# through sanitization untouched, and publishes a perfectly well-formed
# artifact claiming this board has no tasks — a claim a reader then serves.
# An empty board and a board whose schema is not the contract must not be
# indistinguishable in the output.
#
# So the check has to happen against the schema, before the SELECT: the
# evidence pinned below is not just "no artifact", it is that
# `_TASK_SELECT_SQL` never ran at all, and that the read transaction the check
# runs in was closed rather than abandoned.

# The nine columns the contract needs to see on a real `tasks` table: the eight
# allowlisted ones plus `body`, which is here only so the view case is a
# faithful stand-in for the real table rather than a narrower one.
_COMPATIBLE_TASKS_COLUMNS = (
    "id           TEXT PRIMARY KEY",
    "title        TEXT NOT NULL",
    "body         TEXT",
    "assignee     TEXT",
    "status       TEXT NOT NULL",
    "priority     INTEGER DEFAULT 0",
    "created_at   INTEGER NOT NULL",
    "started_at   INTEGER",
    "completed_at INTEGER",
)


def _tasks_ddl(*, name="tasks", drop=(), replace=()):
    """The compatible DDL with named columns dropped or respelled."""
    replacements = dict(replace)
    columns = [
        replacements.get(column.split()[0], column)
        for column in _COMPATIBLE_TASKS_COLUMNS
        if column.split()[0] not in drop
    ]
    return f"CREATE TABLE {name} (\n    " + ",\n    ".join(columns) + "\n)"


# Each case is the DDL for an EMPTY source that `SELECT <allowlist> FROM tasks`
# either resolves to something that is not the contracted table, or cannot
# resolve at all — and in every query-compatible case the defect is invisible
# in the rows, because there are none.
_INCOMPATIBLE_TASKS_SCHEMAS = {
    # A view is not a table: it has no primary key, no NOT NULL and no declared
    # types of its own, and whatever it selects from can change underneath the
    # publisher without the name it queried ever changing.
    "tasks_is_a_view": (
        _tasks_ddl(name="tasks_store"),
        "CREATE VIEW tasks AS SELECT "
        + ", ".join(column.split()[0] for column in _COMPATIBLE_TASKS_COLUMNS)
        + " FROM tasks_store",
    ),
    # A required allowlisted column is simply absent.
    "missing_required_column": (_tasks_ddl(drop=("completed_at",)),),
    # `created_at` is declared TEXT: TEXT affinity where the contract requires
    # INTEGER, so this column stores timestamps that are not epoch ints.
    "wrong_declared_type": (
        _tasks_ddl(replace=(("created_at", "created_at TEXT NOT NULL"),)),
    ),
    # `status` may be NULL, so the table can hold rows with no status at all —
    # rows the eligibility filter silently drops rather than refuses.
    "required_column_is_nullable": (
        _tasks_ddl(replace=(("status", "status TEXT"),)),
    ),
    # Without `id` as the primary key, `priority DESC, id ASC` is no longer a
    # total order, so `--limit` stops meaning "the N most important".
    "id_is_not_primary_key": (
        _tasks_ddl(replace=(("id", "id TEXT NOT NULL"),)),
    ),
}


def _seed_incompatible_board(db_path, statements):
    """Create an empty board database whose `tasks` breaks the contract."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        for statement in statements:
            conn.execute(statement)
        conn.commit()
    finally:
        conn.close()


@pytest.mark.parametrize("case", sorted(_INCOMPATIBLE_TASKS_SCHEMAS))
def test_publish_read_model_rejects_a_schema_incompatible_tasks_source_before_selecting(
    case, publish_sandbox, monkeypatch, capsys
):
    """A `tasks` that is a view, or a table missing a required column, or
    declaring the wrong affinity, nullability or primary key, is refused before
    the allowlisted SELECT runs and before anything is written."""
    root = publish_sandbox / "kanban-root"
    board = "synthetic"
    board_dir = root / "kanban" / "boards" / board
    canonical_db = board_dir / "kanban.db"
    _seed_incompatible_board(canonical_db, _INCOMPATIBLE_TASKS_SCHEMAS[case])
    for directory in (root, root / "kanban", root / "kanban" / "boards", board_dir):
        os.chmod(directory, 0o700)

    out = publish_sandbox / "published" / "read-model.json"
    db_before = _foreign_db_manifest(canonical_db)

    # Every statement the publisher runs, in order, so the test can pin what
    # did NOT run as precisely as what did.
    statements: list[str] = []
    connections: list[sqlite3.Connection] = []
    real_connect = sqlite3.connect

    class _RecordingConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            statements.append(sql)
            return super().execute(sql, *args, **kwargs)

    def _recording_connect(target, *args, **kwargs):
        kwargs["factory"] = _RecordingConnection
        conn = real_connect(target, *args, **kwargs)
        connections.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", _recording_connect)

    args = _parse_kanban_args(
        [
            "kanban",
            "--kanban-root", str(root),
            "--board", board,
            "publish-read-model",
            "--db", str(canonical_db),
            "--out", str(out),
            "--limit", "50",
        ]
    )
    rc = kc.kanban_command(args)

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    # The same single generic string every other refusal uses: which part of
    # the schema disagreed is a description of the caller's own database, and
    # printing it would make this command a schema oracle.
    assert captured.err.strip() == "kanban publish-read-model: unavailable"

    # Nothing published, nothing replaced, and no directory created on the way
    # to not publishing.
    assert not out.exists()
    assert not out.parent.exists()

    # The decisive evidence: the allowlisted SELECT never executed. An artifact
    # refused only after the query would still have been refused for the wrong
    # reason — "the rows were bad" — and an empty incompatible source has no
    # bad rows to find.
    assert krmp._TASK_SELECT_SQL not in statements

    # The read transaction the schema check runs in is closed, not abandoned:
    # BEGIN/COMMIT-or-ROLLBACK strictly alternate, and the connection itself is
    # closed by the time the command returns.
    transaction = [
        sql for sql in statements
        if sql.startswith(("BEGIN", "COMMIT", "ROLLBACK", "END"))
    ]
    assert len(transaction) % 2 == 0, transaction
    assert all(sql.startswith("BEGIN") for sql in transaction[0::2]), transaction
    assert all(
        sql.startswith(("COMMIT", "ROLLBACK", "END")) for sql in transaction[1::2]
    ), transaction
    assert connections, "the publisher never opened the board database"
    for conn in connections:
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")

    # And the board itself is byte-, mtime- and sidecar-identical: a refused
    # publication leaves no `-wal`/`-shm` behind.
    assert _foreign_db_manifest(canonical_db) == db_before


# ---------------------------------------------------------------------------
# Cycle 11 — every ancestor of the board database, not only the ones below the
# Kanban root
# ---------------------------------------------------------------------------
#
# Cycle 9 pinned the board directory on an fd and re-checked the final
# component against that fd after `sqlite3.connect` returned. Both halves are
# real, and together they are still not enough, because of where the pinned
# walk starts CHECKING: ownership and mode are asserted only from the Kanban
# root downwards. Everything above the root is opened with `O_NOFOLLOW` and
# then trusted.
#
# `sqlite3.connect` does not use any of that. It is handed a pathname and
# re-resolves the whole chain in C, from `/` down, at the moment it opens the
# file. So one unchecked ancestor that another local user may write is enough:
# that user renames the Kanban root aside, drops an exact-looking tree of their
# own under the vacated name, lets the connect bind its descriptor to THEIR
# database, and puts the honest tree back. The post-connect identity check then
# stats the pinned board-directory fd — the honest one, restored, unchanged —
# and confirms it. SQLite is already holding the foreign file, and every row
# the artifact carries comes from it, stamped with this root's fingerprint and
# this board's name.
#
# The ancestor here is world-writable and NOT sticky, which is exactly the
# condition under which another local user may rename its entries. Nothing is
# monkeypatched to make that true: the mode bits say it.
#
# Same-UID replacement stays out of the threat model — a process running as the
# owner can rename the root at will and no permission bit distinguishes it.
# What is in scope is another local user, and this is the one path left where
# they could still decide which database the artifact describes.


def test_publish_read_model_refuses_a_board_under_an_ancestor_another_user_can_substitute(
    publish_sandbox, monkeypatch, capsys
):
    """A board database whose chain hangs off an ancestor another local user
    can rename must be refused BEFORE `sqlite3.connect`, not validated around
    it."""
    # The unchecked link in the chain: above the Kanban root, world-writable,
    # no sticky bit. Any local user may rename `kanban-root` out of it.
    shared = publish_sandbox / "shared"
    shared.mkdir()
    os.chmod(shared, 0o777)

    board = "synthetic"
    root = shared / "kanban-root"
    board_dir = root / "kanban" / "boards" / board
    canonical_db = board_dir / "kanban.db"
    _seed_synthetic_board(canonical_db)
    # From the root down everything is impeccable — stated, not inherited — so
    # the only thing this test can fail on is the ancestor above it.
    for directory in (root, root / "kanban", root / "kanban" / "boards", board_dir):
        os.chmod(directory, 0o700)

    # An exact-looking twin: same relative layout under the same final name, so
    # the identical pathname resolves into it once it is moved into place.
    foreign_holder = publish_sandbox / "foreign-holder"
    foreign_root = foreign_holder / "kanban-root"
    foreign_board_dir = foreign_root / "kanban" / "boards" / board
    foreign_db = foreign_board_dir / "kanban.db"
    _seed_foreign_board(foreign_db)
    for directory in (
        foreign_holder, foreign_root, foreign_root / "kanban",
        foreign_root / "kanban" / "boards", foreign_board_dir,
    ):
        os.chmod(directory, 0o700)

    out = publish_sandbox / "published" / "read-model.json"
    displaced = publish_sandbox / "displaced-kanban-root"

    foreign_before = _foreign_db_manifest(foreign_db)
    canonical_before = _foreign_db_manifest(canonical_db)

    connect_targets: list[str] = []
    swaps: list[str] = []
    real_connect = sqlite3.connect

    def _substituting_connect(target, *args, **kwargs):
        """The other user's whole move, compressed into the connect window."""
        connect_targets.append(str(target))
        # Validation is over; SQLite has not resolved the pathname yet. This
        # is the only instant that matters, and it is reachable from an
        # ancestor the walk never checked.
        os.rename(root, displaced)
        os.rename(foreign_root, root)
        swaps.append(str(target))
        try:
            # Binds its descriptor to the FOREIGN database. Nothing below can
            # unbind it: the fd outlives the name.
            conn = real_connect(target, *args, **kwargs)
        finally:
            # Restored before the post-connect identity check gets to look, so
            # that check stats the honest inode on the pinned fd and agrees.
            os.rename(root, foreign_root)
            os.rename(displaced, root)
        return conn

    monkeypatch.setattr(sqlite3, "connect", _substituting_connect)

    # Armed, not merely asserted afterwards: an artifact written and then
    # removed is still a write a reader could have seen.
    write_opens: list[tuple] = []
    real_builtin_open = builtins.open
    real_os_open = os.open
    _WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC

    def _recording_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            write_opens.append((str(file), mode))
        return real_builtin_open(file, mode, *args, **kwargs)

    def _recording_os_open(path, flags, *args, **kwargs):
        if flags & _WRITE_FLAGS:
            write_opens.append((str(path), flags))
        return real_os_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _recording_open)
    monkeypatch.setattr(os, "open", _recording_os_open)

    args = _parse_kanban_args(
        [
            "kanban",
            "--kanban-root", str(root),
            "--board", board,
            "publish-read-model",
            "--db", str(canonical_db),
            "--out", str(out),
            "--limit", "50",
        ]
    )
    rc = kc.kanban_command(args)

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    # One generic string: the caller controls these paths, so a distinguishable
    # reason would turn the refusal into a filesystem oracle.
    assert captured.err.strip() == "kanban publish-read-model: unavailable"

    # The decisive evidence. Not "the swap was detected afterwards" — the
    # connect that the swap needs must never happen at all, because a bound
    # descriptor cannot be taken back.
    assert connect_targets == []
    assert swaps == []

    # Nothing published, no directory created on the way to not publishing,
    # and no write anywhere.
    assert not out.exists()
    assert not out.parent.exists()
    assert write_opens == []

    # Neither database was touched: same bytes, same mtime, no `-wal`/`-shm`.
    assert _foreign_db_manifest(foreign_db) == foreign_before
    assert _foreign_db_manifest(canonical_db) == canonical_before


# ---------------------------------------------------------------------------
# Cycle 12 — publication may replace this board's own artifact and nothing else
# ---------------------------------------------------------------------------
#
# `--out` is a caller-supplied pathname, and publication ends in a `rename`
# over whatever currently answers to it. Every cycle above asks whether the
# artifact lands in a safe *place*; none of them asks what was already sitting
# there. So an owner who typos `--out` onto a private key, a password store, a
# half-finished document — or who is handed that path by a script — loses the
# file, atomically and unrecoverably, to a command that reports success.
#
# The rule is therefore the narrowest one that still lets the publisher do its
# job: an ABSENT target may be created, and an EXISTING target may be replaced
# only if it is this board's own prior read-model artifact — the exact schema,
# the reader's own structural and value bounds, 0600, one link, owned by the
# caller, and stamped with THIS explicit root's fingerprint and THIS explicit
# board. Anything else is somebody's file, not a previous publication, and the
# publisher leaves it exactly as it found it.
#
# Freshness is deliberately NOT part of that predicate: see
# `test_publish_read_model_replaces_this_boards_own_stale_prior_artifact`.
#
# All twelve shapes below are ONE behaviour — "an existing target that is not
# this board's own artifact is refused before the database is opened and before
# anything is written" — so they are one parametrized test. The ordering is the
# point again: the refusal has to happen while the publisher still has nothing
# open and nothing written, not after it has built an artifact it then declines
# to move.

# Planted inside every target whose payload can carry text. The refusal must
# not quote, echo, or summarise a single byte of the file it declined to
# overwrite: the caller named the path, but the CONTENT is not something this
# command may turn into output.
_TARGET_SECRET = "TARGET-SECRET-MUST-NOT-BE-DISCLOSED"

# The one task row every synthetic prior artifact below carries. Non-empty on
# purpose: a `tasks: []` artifact would satisfy the structural bounds without
# ever exercising them.
_PRIOR_TASK = {
    "id": "t_prior",
    "title": f"prior row {_TARGET_SECRET}",
    "status": "todo",
    "assignee": None,
    "priority": 1,
    "created_at": 1600,
    "started_at": None,
    "completed_at": None,
}

# Deliberately ancient. Freshness is the READER's rule, and refreshing a stale
# artifact is the entire reason this command exists.
_PRIOR_GENERATED_AT = 1_700_000_000


def _prior_artifact_bytes(root, board, **overrides):
    """The bytes a legitimate previous publication for *root*/*board* left.

    ``overrides`` replaces or adds top-level keys, which is how the foreign
    and malformed cases below stay one character away from a genuine artifact:
    everything except the named key is exactly what the publisher itself would
    have written.
    """
    document = {
        "schema_version": krm.SCHEMA_VERSION,
        "generated_at": _PRIOR_GENERATED_AT,
        "board": board,
        "root_fingerprint": krm.root_fingerprint(root),
        "truncated": False,
        "tasks": [dict(_PRIOR_TASK)],
    }
    document.update(overrides)
    return json.dumps(document, ensure_ascii=True).encode("utf-8")


def _plant_target(sandbox, payload, *, mode=krm.ARTIFACT_MODE):
    """Seed `<sandbox>/published/read-model.json` the way a publication would.

    A 0700 directory holding one file, so nothing about the DESTINATION is
    what refuses the run — the only thing under test is the file already
    answering to the target name.
    """
    parent = sandbox / "published"
    parent.mkdir()
    os.chmod(parent, 0o700)
    out = parent / "read-model.json"
    out.write_bytes(payload)
    os.chmod(out, mode)
    return out


def _target_is_arbitrary_owner_bytes(sandbox, root, board):
    """An ordinary private 0600 file the owner keeps at that name.

    Nothing about it is hostile and nothing about it is an artifact either.
    This is the case the whole rule exists for.
    """
    payload = (
        b"-----BEGIN OPENSSH PRIVATE KEY-----\n"
        + _TARGET_SECRET.encode("ascii")
        + b"\n-----END OPENSSH PRIVATE KEY-----\n"
    )
    return {"out": _plant_target(sandbox, payload), "payload": payload}


def _target_is_an_artifact_for_a_foreign_root(sandbox, root, board):
    """Structurally perfect, and bound to somebody else's Kanban root.

    The artifact stamps the root it was read for, and a reader trusts that
    stamp. Replacing this one would silently re-point another root's published
    view at this root's board.
    """
    payload = _prior_artifact_bytes(
        root, board, root_fingerprint=krm.root_fingerprint(sandbox / "other-root")
    )
    return {"out": _plant_target(sandbox, payload), "payload": payload}


def _target_is_an_artifact_for_a_foreign_board(sandbox, root, board):
    """Structurally perfect, same root, and published for another board."""
    payload = _prior_artifact_bytes(root, "otherboard")
    return {"out": _plant_target(sandbox, payload), "payload": payload}


def _target_is_an_artifact_with_the_wrong_mode(sandbox, root, board):
    """This board's own document, at a mode the publisher never produces.

    0644 means somebody else has already been able to read it, so it is not
    evidence of a previous publication by this command — and the reader
    refuses it anyway.
    """
    payload = _prior_artifact_bytes(root, board)
    return {"out": _plant_target(sandbox, payload, mode=0o644), "payload": payload}


def _target_is_an_artifact_owned_by_another_user(sandbox, root, board):
    """This board's own document, owned by somebody who is not the caller.

    Real foreign ownership cannot be created without privilege, so the
    *answer* is mocked rather than the file — and mocked for THIS INODE only.
    Moving `os.getuid` instead would make every ownership rule in the module
    fail at once, including the board-ancestor checks that run later, and the
    case would pass without the target ever being looked at.
    """
    payload = _prior_artifact_bytes(root, board)
    out = _plant_target(sandbox, payload)
    st = out.lstat()
    pinned = (st.st_dev, st.st_ino)

    def arm(monkeypatch):
        real_fstat = os.fstat

        def _foreign_owner_fstat(fd, *args, **kwargs):
            reported = real_fstat(fd, *args, **kwargs)
            if (reported.st_dev, reported.st_ino) != pinned:
                return reported
            fields = list(reported)
            # Index 4 is `st_uid` in the 10-field stat tuple.
            fields[4] = reported.st_uid + 1
            return os.stat_result(fields)

        monkeypatch.setattr(os, "fstat", _foreign_owner_fstat)

    return {"out": out, "payload": payload, "arm": arm}


def _target_is_a_hard_linked_artifact(sandbox, root, board):
    """This board's own document, reachable under a second name.

    A second link is a second writable name for the same inode, outside
    whatever protection the destination directory provides — so replacing
    "the artifact" would be replacing a file somebody else still holds open by
    another path. The reader refuses `st_nlink != 1` for the same reason.
    """
    payload = _prior_artifact_bytes(root, board)
    out = _plant_target(sandbox, payload)
    os.link(out, out.parent / "second-name.json")
    return {"out": out, "payload": payload, "siblings": ["second-name.json"]}


def _target_is_not_utf8(sandbox, root, board):
    """Bytes no artifact could be: a lone UTF-8 continuation sequence."""
    payload = b'{"schema_version": "\xff\xfe\x00 ' + _TARGET_SECRET.encode() + b'"}'
    return {"out": _plant_target(sandbox, payload), "payload": payload}


def _target_is_a_future_schema_artifact(sandbox, root, board):
    """A document from a publisher this build does not understand.

    Refused rather than overwritten: a schema this build cannot validate is
    also a schema it cannot conclude is its own previous output.
    """
    payload = _prior_artifact_bytes(
        root, board, schema_version="hermes.kanban.read-model.v2"
    )
    return {"out": _plant_target(sandbox, payload), "payload": payload}


def _target_carries_an_unknown_top_level_key(sandbox, root, board):
    """Everything the artifact requires, plus one key it does not allow."""
    payload = _prior_artifact_bytes(root, board, signature=_TARGET_SECRET)
    return {"out": _plant_target(sandbox, payload), "payload": payload}


def _target_exceeds_the_artifact_size_cap(sandbox, root, board):
    """Larger than any artifact the reader will ever accept."""
    payload = b"x" * (krm.MAX_ARTIFACT_BYTES + 1)
    return {"out": _plant_target(sandbox, payload), "payload": payload}


def _target_is_a_directory(sandbox, root, board):
    """A directory answering to the target name.

    `rename` onto a non-empty directory fails, but the publisher must not get
    there at all: deciding this at the destination check is what keeps the
    database closed.
    """
    parent = sandbox / "published"
    parent.mkdir()
    os.chmod(parent, 0o700)
    out = parent / "read-model.json"
    out.mkdir()
    os.chmod(out, 0o700)
    (out / "keep.txt").write_bytes(_TARGET_SECRET.encode("ascii"))
    return {"out": out}


def _target_is_a_fifo(sandbox, root, board):
    """A FIFO answering to the target name.

    Nothing may open it for reading without `O_NONBLOCK`: a blocking open
    waits for a writer that never comes, so a publisher that inspects the
    target carelessly hangs instead of refusing.
    """
    parent = sandbox / "published"
    parent.mkdir()
    os.chmod(parent, 0o700)
    out = parent / "read-model.json"
    os.mkfifo(out, 0o600)
    return {"out": out}


@pytest.mark.parametrize(
    "plant_target",
    [
        pytest.param(_target_is_arbitrary_owner_bytes, id="arbitrary-owner-bytes"),
        pytest.param(
            _target_is_an_artifact_for_a_foreign_root, id="foreign-root-fingerprint"
        ),
        pytest.param(
            _target_is_an_artifact_for_a_foreign_board, id="foreign-board"
        ),
        pytest.param(_target_is_an_artifact_with_the_wrong_mode, id="wrong-mode"),
        pytest.param(
            _target_is_an_artifact_owned_by_another_user, id="foreign-owner"
        ),
        pytest.param(_target_is_a_hard_linked_artifact, id="hard-linked"),
        pytest.param(_target_is_not_utf8, id="not-utf8"),
        pytest.param(
            _target_is_a_future_schema_artifact, id="unsupported-schema-version"
        ),
        pytest.param(
            _target_carries_an_unknown_top_level_key, id="unknown-top-level-key"
        ),
        pytest.param(
            _target_exceeds_the_artifact_size_cap, id="over-the-size-cap"
        ),
        pytest.param(_target_is_a_directory, id="directory-target"),
        pytest.param(_target_is_a_fifo, id="fifo-target"),
    ],
)
def test_publish_read_model_never_overwrites_a_target_that_is_not_this_boards_artifact(
    publish_sandbox, monkeypatch, capsys, plant_target
):
    """An existing target that is not this board's own prior artifact is
    refused before `sqlite3.connect` and before any write, is left
    byte-for-byte and inode-identical, and the refusal discloses nothing about
    it beyond the one generic string."""
    root = publish_sandbox / "kanban-root"
    root.mkdir()
    board = "synthetic"
    # A real, valid board and a destination directory that passes every
    # output-location rule: the ONLY thing wrong with this invocation is what
    # already answers to the target name. Without the rule it would publish.
    db = root / "kanban" / "boards" / board / "kanban.db"
    _seed_synthetic_board(db)

    case = plant_target(publish_sandbox, root, board)
    out = case["out"]
    before_stat = out.lstat()
    before = _tree_manifest(publish_sandbox)

    db_connects: list[tuple] = []
    side_effects: list[tuple] = []

    def _tripwire_connect(*args, **kwargs):
        db_connects.append(args)
        raise AssertionError("publish-read-model opened the live database")

    monkeypatch.setattr(sqlite3, "connect", _tripwire_connect)

    real_open = builtins.open

    def _tripwire_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            side_effects.append((str(file), mode))
            raise AssertionError("publish-read-model opened a file for writing")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _tripwire_open)

    real_os_open = os.open
    _WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC

    def _tripwire_os_open(path, flags, *args, **kwargs):
        if flags & _WRITE_FLAGS:
            side_effects.append((str(path), flags))
            raise AssertionError("publish-read-model opened an fd for writing")
        return real_os_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", _tripwire_os_open)

    # Every remaining way the destination directory could change: the
    # transaction's own `mkdir`/`link`/`replace`, and the plain unlink or
    # rename a "clear it first" shortcut would reach for.
    for name in ("mkdir", "link", "replace", "rename", "unlink", "truncate"):
        real = getattr(os, name)

        def _tripwire(*args, _name=name, _real=real, **kwargs):
            side_effects.append((_name, tuple(str(arg) for arg in args)))
            raise AssertionError(f"publish-read-model called os.{_name}")

        monkeypatch.setattr(os, name, _tripwire)

    if case.get("arm") is not None:
        case["arm"](monkeypatch)

    args = _parse_kanban_args(
        [
            "kanban",
            "--kanban-root", str(root),
            "--board", board,
            "publish-read-model",
            "--db", str(db),
            "--out", str(out),
            "--limit", "3",
        ]
    )
    rc = kc.kanban_command(args)
    # Disarmed as soon as the run is over: the tripwires above make `os.unlink`
    # and friends fatal, and the sandbox fixture has to be able to clean up.
    monkeypatch.undo()

    assert rc == 1
    # The decisive evidence: the refusal landed before the board was opened
    # and before anything on disk could move.
    assert db_connects == []
    assert side_effects == []

    captured = capsys.readouterr()
    assert captured.out == ""
    # One fixed generic string. Which rule caught the target — and therefore
    # what the file at that path IS — is not something this command reports.
    assert captured.err == "kanban publish-read-model: unavailable\n"
    assert _TARGET_SECRET not in captured.out + captured.err

    # The target survives as the SAME inode, with the same mode and the same
    # bytes: not restored from a copy, not re-created, not truncated.
    after_stat = out.lstat()
    assert after_stat.st_ino == before_stat.st_ino
    assert stat.S_IMODE(after_stat.st_mode) == stat.S_IMODE(before_stat.st_mode)
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns
    if case.get("payload") is not None:
        assert out.read_bytes() == case["payload"]

    # Nothing else moved either: no dotted temp file, no backup link, no
    # WAL/SHM sidecar beside the synthetic database.
    assert sorted(entry.name for entry in os.scandir(out.parent)) == sorted(
        [out.name, *case.get("siblings", [])]
    )
    assert _tree_manifest(publish_sandbox) == before


def test_publish_read_model_replaces_this_boards_own_stale_prior_artifact(
    publish_sandbox, capsys
):
    """The one existing target publication MAY replace, however old it is.

    Freshness is the reader's rule, and a stale artifact is precisely the one
    that needs republishing — a publisher that required its own previous
    output to be fresh could never refresh anything. So the predicate is about
    IDENTITY (this schema, this root, this board, 0600, one link, this owner)
    and not about age.
    """
    root = publish_sandbox / "kanban-root"
    root.mkdir()
    board = "synthetic"
    db = root / "kanban" / "boards" / board / "kanban.db"
    _seed_synthetic_board(db)

    prior = _prior_artifact_bytes(root, board)
    out = _plant_target(publish_sandbox, prior)
    # Old enough that no freshness window would accept it.
    assert int(time.time()) - _PRIOR_GENERATED_AT > krm.DEFAULT_MAX_AGE_SECONDS

    args = _parse_kanban_args(
        [
            "kanban",
            "--kanban-root", str(root),
            "--board", board,
            "publish-read-model",
            "--db", str(db),
            "--out", str(out),
            "--limit", "3",
        ]
    )
    rc = kc.kanban_command(args)

    captured = capsys.readouterr()
    assert rc == 0, f"publish failed: {captured.err}"
    assert captured.err == ""

    raw = out.read_bytes()
    assert raw != prior
    assert stat.S_IMODE(out.stat().st_mode) == krm.ARTIFACT_MODE

    document = json.loads(raw.decode("utf-8"))
    assert document["board"] == board
    assert document["root_fingerprint"] == krm.root_fingerprint(root)
    assert [task["id"] for task in document["tasks"]] == ["t_aaa", "t_bbb", "t_ccc"]

    # The destination holds the new artifact and nothing else.
    assert [entry.name for entry in os.scandir(out.parent)] == [out.name]
