"""Tests for the H0.2b3a reader-only JSON artifact contract.

H0.2b3a replaces the rejected live-SQLite read model with an owner-published,
sanitized, bounded JSON artifact. The reader module
(:mod:`hermes_cli.kanban_read_model`) must:

  * validate ONLY the explicit canonical Kanban root as a binding input —
    it never constructs, ``lstat``s, or opens ``kanban.db``;
  * open the artifact (which lives OUTSIDE the Kanban root) through an
    ``openat``/``O_NOFOLLOW`` component walk and validate the final ``fstat``;
  * bound the read to 256 KiB and reject anything larger;
  * reject duplicate JSON keys, ``NaN``/``Infinity``, unknown keys, wrong
    types, wrong schema, wrong board/root binding, future or stale
    timestamps, and over-cap row counts;
  * re-serialize only the PlanSpec v1 allowlist after control-character
    sanitization;
  * surface every post-usage failure as the single generic
    ``kanban read-model: unavailable`` on rc=1, with no path or detail.

The owner-side publisher is H0.2b3b and is deliberately NOT implemented here;
every artifact in these tests is written synthetically by the test itself.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import os
import signal
import sqlite3
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_read_model as krm


# ---------------------------------------------------------------------------
# Cycle 1 — canonical root-only validation
# ---------------------------------------------------------------------------


def test_validated_kanban_root_returns_the_canonical_root_directory(tmp_path):
    """The reader's only binding Kanban-root input is the explicit canonical
    root directory itself. It is returned unchanged as an absolute Path."""
    root = tmp_path / "explicit-root"
    root.mkdir()

    assert krm.validated_kanban_root(str(root)) == root


def test_validated_kanban_root_never_constructs_or_stats_kanban_db(tmp_path, monkeypatch):
    """Root validation must never derive the live DB path. With a real
    ``kanban.db`` plus WAL/SHM sidecars present, no ``lstat`` may name any of
    them — the H0 reader has no business knowing that path exists."""
    root = tmp_path / "explicit-root"
    (root / "kanban" / "boards" / "acme").mkdir(parents=True)
    for name in ("kanban.db", "kanban.db-wal", "kanban.db-shm"):
        (root / name).touch()
        (root / "kanban" / "boards" / "acme" / name).touch()

    seen: list[str] = []
    real_lstat = os.lstat

    def _spy(path, *args, **kwargs):
        seen.append(os.fspath(path))
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(krm.os, "lstat", _spy)

    assert krm.validated_kanban_root(str(root)) == root
    assert not [p for p in seen if "kanban.db" in p]


@pytest.mark.parametrize(
    "case",
    ["relative", "non-canonical-dotdot", "missing", "not-a-directory"],
)
def test_validated_kanban_root_rejects_unusable_roots(case, tmp_path):
    """A relative spelling, a ``..`` alias spelling, a missing root, and a
    non-directory root are all rejected as unavailable."""
    if case == "relative":
        candidate = "relative/root"
    elif case == "non-canonical-dotdot":
        real = tmp_path / "container" / "root"
        real.mkdir(parents=True)
        candidate = str(tmp_path / "container" / "root" / ".." / "root")
    elif case == "missing":
        candidate = str(tmp_path / "absent-90fe12")
    else:
        plain = tmp_path / "plain-file"
        plain.touch()
        candidate = str(plain)

    with pytest.raises(krm.ReadModelUnavailable):
        krm.validated_kanban_root(candidate)


@pytest.mark.parametrize(
    "_posix_lane",
    [
        pytest.param("linux", marks=pytest.mark.linux_only),
        pytest.param("macos", marks=pytest.mark.macos_only),
    ],
)
def test_validated_kanban_root_rejects_symlinked_root_component(tmp_path, _posix_lane):
    """A symlink anywhere in the root's component chain is rejected, even
    when it resolves inside the same tmp tree."""
    real = tmp_path / "real-root"
    real.mkdir()
    link = tmp_path / "root-link"
    link.symlink_to(real, target_is_directory=True)

    with pytest.raises(krm.ReadModelUnavailable):
        krm.validated_kanban_root(str(link))


# ---------------------------------------------------------------------------
# Cycle 2 — artifact fd acquisition (openat/O_NOFOLLOW component walk)
# ---------------------------------------------------------------------------


def _write_artifact(path: Path, payload: bytes, mode: int = 0o600) -> Path:
    """Write a synthetic artifact. Everything the reader consumes in these
    tests is produced here — the owner-side publisher is H0.2b3b and must
    not be imported, referenced, or executed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    os.chmod(path, mode)
    return path


@pytest.mark.parametrize(
    "_posix_lane",
    [
        pytest.param("linux", marks=pytest.mark.linux_only),
        pytest.param("macos", marks=pytest.mark.macos_only),
    ],
)
def test_read_artifact_bytes_component_walks_with_openat_nofollow(tmp_path, monkeypatch, _posix_lane):
    """Tracer: a valid external 0600 artifact is opened one path component at
    a time with ``openat`` (``dir_fd=``) and ``O_NOFOLLOW`` — never by handing
    the whole path to a single resolving ``open`` — and its bytes are read
    from exactly that validated fd."""
    artifact = _write_artifact(tmp_path / "outside" / "read-model.json", b'{"ok": true}')

    opens: list[tuple[str, int, object, int]] = []
    reads: list[int] = []
    real_open, real_read = os.open, os.read

    def _open_spy(path, flags, *a, **kw):
        fd = real_open(path, flags, *a, **kw)
        opens.append((os.fspath(path), flags, kw.get("dir_fd"), fd))
        return fd

    def _read_spy(fd, n):
        reads.append(fd)
        return real_read(fd, n)

    monkeypatch.setattr(krm.os, "open", _open_spy)
    monkeypatch.setattr(krm.os, "read", _read_spy)

    assert krm.read_artifact_bytes(artifact) == b'{"ok": true}'

    # The anchor is opened by absolute path; every later component is a single
    # bare name opened relative to the previous component's directory fd.
    assert opens, "no os.open call was made"
    assert opens[0][0] == "/" and opens[0][2] is None
    assert [name for name, _flags, _dfd, _fd in opens[1:]] == list(artifact.parts[1:])
    assert all(dir_fd is not None for _name, _flags, dir_fd, _fd in opens[1:])
    assert all(flags & os.O_NOFOLLOW for _name, flags, _dfd, _fd in opens)
    # Reads come only from the final validated fd, never from a re-opened path.
    assert reads and set(reads) == {opens[-1][3]}


# ---------------------------------------------------------------------------
# Cycle 3 — artifact path admission (absolute, canonical, outside the root)
# ---------------------------------------------------------------------------


def test_validated_artifact_path_accepts_canonical_path_outside_the_root(tmp_path):
    """The artifact lives outside every Kanban-controlled root and is named
    by an absolute, alias-free path, which is returned unchanged."""
    root = tmp_path / "kanban-root"
    root.mkdir()
    artifact = tmp_path / "published" / "read-model.json"

    assert krm.validated_artifact_path(str(artifact), root) == artifact


@pytest.mark.parametrize(
    "case",
    ["relative", "non-canonical-dotdot", "inside-root", "is-the-root"],
)
def test_validated_artifact_path_rejects_unusable_spellings(case, tmp_path):
    """A relative spelling, a ``..`` alias spelling, and any path at or under
    the Kanban root are rejected. The root containment check is lexical on
    purpose: a symlink that would smuggle the path back under the root cannot
    survive the ``O_NOFOLLOW`` component walk anyway, so no ``resolve()`` (and
    therefore no stat of anything under the root) is needed here."""
    root = tmp_path / "kanban-root"
    root.mkdir()

    if case == "relative":
        candidate = "published/read-model.json"
    elif case == "non-canonical-dotdot":
        candidate = str(tmp_path / "published" / ".." / "published" / "read-model.json")
    elif case == "inside-root":
        candidate = str(root / "kanban" / "read-model.json")
    else:
        candidate = str(root)

    with pytest.raises(krm.ReadModelUnavailable):
        krm.validated_artifact_path(candidate, root)


# ---------------------------------------------------------------------------
# Cycle 4 — the validated fd must be a regular file, checked before any read
# ---------------------------------------------------------------------------


class _WatchdogFired(BaseException):
    """Deliberately not an ``OSError``.

    ``TimeoutError`` *is* an ``OSError`` subclass, so a watchdog raising it
    would be swallowed by the reader's own ``except OSError`` and reported as
    a normal unavailable result — a hang would masquerade as a pass.
    """


@contextlib.contextmanager
def _watchdog(seconds: int = 5):
    """Fail instead of hanging. A FIFO opened ``O_RDONLY`` without
    ``O_NONBLOCK`` blocks until a writer appears, so an artifact reader that
    lacks ``O_NONBLOCK`` would wedge the test session forever rather than
    reporting a failure."""

    def _fire(_signum, _frame):
        raise _WatchdogFired("reader blocked on a non-regular artifact")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@pytest.mark.parametrize(
    "_posix_lane",
    [
        pytest.param("linux", marks=pytest.mark.linux_only),
        pytest.param("macos", marks=pytest.mark.macos_only),
    ],
)
@pytest.mark.parametrize("kind", ["directory", "fifo"])
def test_read_artifact_bytes_rejects_non_regular_file_before_reading(kind, tmp_path, monkeypatch, _posix_lane):
    """A directory or FIFO standing in for the artifact is rejected on the
    ``fstat`` of the validated fd — before a single byte is read, and without
    the open ever blocking."""
    target = tmp_path / "published" / "read-model.json"
    target.parent.mkdir(parents=True)
    if kind == "directory":
        target.mkdir()
    else:
        os.mkfifo(target, 0o600)

    reads: list[int] = []
    real_read = os.read
    monkeypatch.setattr(
        krm.os, "read", lambda fd, n: (reads.append(fd), real_read(fd, n))[1]
    )

    with _watchdog():
        with pytest.raises(krm.ReadModelUnavailable):
            krm.read_artifact_bytes(target)

    assert reads == []


# ---------------------------------------------------------------------------
# Cycle 5 — the validated fd must be single-link, owned by us, 0600 and bounded
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "_posix_lane",
    [
        pytest.param("linux", marks=pytest.mark.linux_only),
        pytest.param("macos", marks=pytest.mark.macos_only),
    ],
)
@pytest.mark.parametrize(
    "case",
    ["hardlinked", "group-readable", "owner-executable", "world-readable",
     "foreign-owner", "over-cap"],
)
def test_read_artifact_bytes_rejects_untrustworthy_fd(case, tmp_path, monkeypatch, _posix_lane):
    """The artifact must be a single-link, current-UID, exactly-0600 file no
    larger than the 256 KiB cap. Anything else — an extra hard link giving a
    second name write access, a loosened mode, a file owned by somebody else,
    or an oversized blob — is unavailable."""
    payload = b'{"ok": true}'
    mode = 0o600
    if case == "over-cap":
        payload = b"x" * (krm.MAX_ARTIFACT_BYTES + 1)
    elif case == "group-readable":
        mode = 0o640
    elif case == "owner-executable":
        mode = 0o700
    elif case == "world-readable":
        mode = 0o604

    artifact = _write_artifact(tmp_path / "published" / "read-model.json", payload, mode)

    if case == "hardlinked":
        os.link(artifact, tmp_path / "published" / "second-name.json")
        assert os.stat(artifact).st_nlink == 2
    elif case == "foreign-owner":
        monkeypatch.setattr(krm.os, "getuid", lambda: os.stat(artifact).st_uid + 1)

    with pytest.raises(krm.ReadModelUnavailable):
        krm.read_artifact_bytes(artifact)


@pytest.mark.parametrize(
    "_posix_lane",
    [
        pytest.param("linux", marks=pytest.mark.linux_only),
        pytest.param("macos", marks=pytest.mark.macos_only),
    ],
)
def test_read_artifact_bytes_accepts_a_single_link_owner_only_artifact(tmp_path, _posix_lane):
    """The exact-0600, single-link, current-UID, at-cap case still reads."""
    payload = b"y" * krm.MAX_ARTIFACT_BYTES
    artifact = _write_artifact(tmp_path / "published" / "read-model.json", payload)

    assert krm.read_artifact_bytes(artifact) == payload


# ---------------------------------------------------------------------------
# Cycle 6 — growth between fstat and read
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", ["growth-past-cap", "growth-within-cap"])
def test_read_artifact_bytes_rejects_growth_after_fstat(case, tmp_path, monkeypatch):
    """The size cap is enforced on the bytes actually read, not only on the
    ``fstat`` size: a file that grows after it was measured is rejected — past
    the cap because the cap is absolute, and below the cap because an artifact
    changing under the reader is not the immutable snapshot it claims to be."""
    grow_by = krm.MAX_ARTIFACT_BYTES if case == "growth-past-cap" else 16
    artifact = _write_artifact(tmp_path / "published" / "read-model.json", b'{"ok": true}')

    real_fstat = krm.os.fstat

    def _fstat_then_grow(fd):
        st = real_fstat(fd)
        with open(artifact, "ab") as handle:  # racing writer, same UID
            handle.write(b"x" * grow_by)
        return st

    monkeypatch.setattr(krm.os, "fstat", _fstat_then_grow)

    with pytest.raises(krm.ReadModelUnavailable):
        krm.read_artifact_bytes(artifact)


# ---------------------------------------------------------------------------
# Cycle 7 — symlink and parent-swap adversarial coverage
#
# These assert the guarantee the Cycle-2 openat/O_NOFOLLOW walk already
# provides; they are regression cover for it, not a new behavior, so they are
# expected to pass without any production change.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "_posix_lane",
    [
        pytest.param("linux", marks=pytest.mark.linux_only),
        pytest.param("macos", marks=pytest.mark.macos_only),
    ],
)
@pytest.mark.parametrize("case", ["artifact-is-symlink", "parent-is-symlink"])
def test_read_artifact_bytes_rejects_symlinked_components(case, tmp_path, monkeypatch, _posix_lane):
    """A symlink as the artifact itself or anywhere in its parent chain is
    refused, even when the target is a perfectly valid 0600 artifact."""
    real = _write_artifact(tmp_path / "elsewhere" / "real.json", b'{"ok": true}')

    if case == "artifact-is-symlink":
        artifact = tmp_path / "published" / "read-model.json"
        artifact.parent.mkdir(parents=True)
        artifact.symlink_to(real)
    else:
        (tmp_path / "published").symlink_to(real.parent, target_is_directory=True)
        artifact = tmp_path / "published" / "real.json"

    with pytest.raises(krm.ReadModelUnavailable):
        krm.read_artifact_bytes(artifact)


@pytest.mark.parametrize(
    "_posix_lane",
    [
        pytest.param("linux", marks=pytest.mark.linux_only),
        pytest.param("macos", marks=pytest.mark.macos_only),
    ],
)
def test_read_artifact_bytes_is_pinned_against_a_parent_swap_mid_walk(tmp_path, monkeypatch, _posix_lane):
    """Swapping a parent directory *after* the walk has opened it cannot
    redirect the read: each component is pinned to the inode that was opened,
    so the reader still sees the original artifact and never the attacker's."""
    artifact = _write_artifact(tmp_path / "published" / "read-model.json", b'{"real": true}')
    attacker = _write_artifact(tmp_path / "attacker" / "read-model.json", b'{"evil": true}')

    real_open = krm.os.open
    swapped = False

    def _swap_after_opening_parent(path, flags, *a, **kw):
        nonlocal swapped
        fd = real_open(path, flags, *a, **kw)
        if not swapped and os.fspath(path) == "published":
            swapped = True
            os.rename(tmp_path / "published", tmp_path / "published-moved-away")
            os.rename(attacker.parent, tmp_path / "published")
        return fd

    monkeypatch.setattr(krm.os, "open", _swap_after_opening_parent)

    assert swapped is False
    assert krm.read_artifact_bytes(artifact) == b'{"real": true}'
    assert swapped is True


# ---------------------------------------------------------------------------
# Cycle 8 — strict JSON parsing
# ---------------------------------------------------------------------------


def test_parse_artifact_json_returns_the_top_level_object(tmp_path):
    """A well-formed JSON object parses to a plain dict."""
    assert krm.parse_artifact_json(b'{"board": "acme", "tasks": []}') == {
        "board": "acme",
        "tasks": [],
    }


@pytest.mark.parametrize(
    "case, payload",
    [
        ("duplicate-top-level-key", b'{"board": "acme", "board": "evil"}'),
        ("duplicate-nested-key", b'{"tasks": [{"id": "a", "id": "b"}]}'),
        ("nan", b'{"priority": NaN}'),
        ("infinity", b'{"priority": Infinity}'),
        ("negative-infinity", b'{"priority": -Infinity}'),
        ("invalid-utf8", b'{"board": "\xff\xfe"}'),
        ("top-level-array", b'[{"board": "acme"}]'),
        ("top-level-string", b'"acme"'),
        ("trailing-garbage", b'{"board": "acme"} {"board": "evil"}'),
        ("empty", b""),
        ("truncated", b'{"board": "acme"'),
    ],
)
def test_parse_artifact_json_rejects_hostile_documents(case, payload):
    """Duplicate keys (at any depth), the non-standard ``NaN``/``Infinity``
    constants Python's json module accepts by default, invalid UTF-8, a
    non-object top level, and trailing garbage are all unavailable. Duplicate
    keys matter because ``json.loads`` keeps the *last* one by default, so a
    publisher-signed-looking field could be silently overridden."""
    with pytest.raises(krm.ReadModelUnavailable):
        krm.parse_artifact_json(payload)


# ---------------------------------------------------------------------------
# Cycle 9 — root fingerprint (the artifact's binding to one Kanban root)
# ---------------------------------------------------------------------------


def test_root_fingerprint_is_deterministic_per_root_and_touches_no_filesystem(tmp_path):
    """The fingerprint binds an artifact to exactly one canonical Kanban root.
    It is a pure function of that path: stable across calls, different for a
    different root, and computed without a single filesystem syscall — the
    reader must not stat anything under the Kanban root to derive it.

    The syscall stubs are installed and removed inside ``try``/``finally``
    rather than via ``monkeypatch``, because ``krm.os`` *is* the global ``os``
    module: leaving ``os.stat``/``os.open`` stubbed while pytest formats a
    failure report breaks pytest's own source lookup and hides the real error.
    """
    root = tmp_path / "absent-kanban-root"
    assert not root.exists()

    forbidden: list[str] = []
    saved = {name: getattr(os, name) for name in ("lstat", "stat", "open")}
    try:
        for name in saved:
            setattr(os, name, lambda *a, _name=name, **kw: forbidden.append(_name))
        first = krm.root_fingerprint(root)
        again = krm.root_fingerprint(root)
        other = krm.root_fingerprint(tmp_path / "other-root")
    finally:
        for name, original in saved.items():
            setattr(os, name, original)

    assert forbidden == []
    assert first == again
    assert first != other
    assert len(first) == 64 and set(first) <= set("0123456789abcdef")


# ---------------------------------------------------------------------------
# Cycle 10 — top-level artifact schema
# ---------------------------------------------------------------------------


SYNTHETIC_ROOT = "/synthetic/kanban-root"
GENERATED_AT = 1_700_000_000


def _document(**overrides):
    """A minimal well-formed artifact document (no tasks yet)."""
    document = {
        "schema_version": krm.SCHEMA_VERSION,
        "generated_at": GENERATED_AT,
        "board": "acme",
        "root_fingerprint": krm.root_fingerprint(SYNTHETIC_ROOT),
        "truncated": False,
        "tasks": [],
    }
    document.update(overrides)
    return document


def _validate(document, **overrides):
    kwargs = {
        "board": "acme",
        "expected_fingerprint": krm.root_fingerprint(SYNTHETIC_ROOT),
        "now": GENERATED_AT + 5,
        "max_age_seconds": 3600,
    }
    kwargs.update(overrides)
    return krm.validate_artifact(document, **kwargs)


def test_validate_artifact_accepts_a_well_formed_document():
    """A well-formed document yields the allowlisted payload. The binding
    ``root_fingerprint`` is deliberately NOT echoed back: it is an input
    token used to prove the artifact belongs to this root, not read-model
    data the caller asked for."""
    assert _validate(_document()) == {
        "schema_version": krm.SCHEMA_VERSION,
        "generated_at": GENERATED_AT,
        "board": "acme",
        "truncated": False,
        "tasks": [],
    }


@pytest.mark.parametrize(
    "case, document",
    [
        ("unknown-key", _document(extra="surprise")),
        ("missing-schema-version", {k: v for k, v in _document().items() if k != "schema_version"}),
        ("missing-tasks", {k: v for k, v in _document().items() if k != "tasks"}),
        ("wrong-schema-version", _document(schema_version="hermes.kanban.read-model.v2")),
        ("schema-version-not-a-string", _document(schema_version=1)),
        ("board-not-a-string", _document(board=["acme"])),
        ("fingerprint-not-a-string", _document(root_fingerprint=None)),
        ("truncated-not-a-bool", _document(truncated=1)),
        ("truncated-as-string", _document(truncated="false")),
        ("tasks-not-a-list", _document(tasks={})),
        ("generated-at-not-an-int", _document(generated_at="1700000000")),
        ("generated-at-float", _document(generated_at=float(GENERATED_AT))),
        ("generated-at-bool", _document(generated_at=True)),
        ("generated-at-negative", _document(generated_at=-1)),
    ],
)
def test_validate_artifact_rejects_bad_top_level_schema(case, document):
    """Unknown keys, missing keys, the wrong schema version, and wrong types
    are all unavailable. ``truncated=1`` and ``generated_at=True`` matter
    specifically: ``bool`` is a subclass of ``int`` in Python, so a sloppy
    ``isinstance`` check would accept both."""
    with pytest.raises(krm.ReadModelUnavailable):
        _validate(document)


# ---------------------------------------------------------------------------
# Cycle 11 — binding: the artifact must belong to THIS root and THIS board
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case, document, overrides",
    [
        ("other-board", _document(board="other"), {}),
        ("case-differing-board", _document(board="ACME"), {}),
        ("board-with-surrounding-space", _document(board=" acme "), {}),
        ("requested-other-board", _document(), {"board": "other"}),
        (
            "other-root-fingerprint",
            _document(root_fingerprint=krm.root_fingerprint("/synthetic/other-root")),
            {},
        ),
        ("empty-fingerprint", _document(root_fingerprint=""), {}),
    ],
)
def test_validate_artifact_rejects_a_mis_bound_artifact(case, document, overrides):
    """An artifact published for a different board or a different Kanban root
    is unavailable — never silently served for the board that was asked for.
    Matching is exact: no case folding, no whitespace trimming, no fallback."""
    with pytest.raises(krm.ReadModelUnavailable):
        _validate(document, **overrides)


# ---------------------------------------------------------------------------
# Cycle 12 — freshness: stale is unavailable, never "current"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case, overrides",
    [
        ("one-second-in-the-future", {"now": GENERATED_AT - 1}),
        ("far-in-the-future", {"now": GENERATED_AT - 86_400}),
        ("one-second-too-old", {"now": GENERATED_AT + 3601}),
        ("long-abandoned", {"now": GENERATED_AT + 31 * 86_400}),
        ("negative-max-age", {"max_age_seconds": -1}),
    ],
)
def test_validate_artifact_rejects_unfresh_documents(case, overrides):
    """An artifact stamped in the future (a clock lie, or a forgery reaching
    for permanent freshness) and an artifact older than the caller's explicit
    ``--max-age-seconds`` are both unavailable. Because publication is manual,
    staleness is the expected steady state — it must never be reported as
    healthy or current data."""
    with pytest.raises(krm.ReadModelUnavailable):
        _validate(_document(), **overrides)


@pytest.mark.parametrize(
    "case, overrides",
    [
        ("generated-exactly-now", {"now": GENERATED_AT}),
        ("exactly-at-the-age-limit", {"now": GENERATED_AT + 3600}),
        ("zero-max-age-generated-now", {"now": GENERATED_AT, "max_age_seconds": 0}),
    ],
)
def test_validate_artifact_accepts_documents_inside_the_freshness_window(case, overrides):
    """The freshness boundary is inclusive on both ends."""
    assert _validate(_document(), **overrides)["generated_at"] == GENERATED_AT


# ---------------------------------------------------------------------------
# Cycle 13 — task rows: exact field allowlist, types, sanitization, row cap
# ---------------------------------------------------------------------------


def _task(**overrides):
    task = {
        "id": "t-0001",
        "title": "Ship the read model",
        "status": "ready",
        "assignee": "thomas",
        "priority": 3,
        "created_at": GENERATED_AT - 900,
        "started_at": None,
        "completed_at": None,
    }
    task.update(overrides)
    return task


def test_validate_artifact_returns_task_rows_rebuilt_from_allowlisted_fields():
    """Task rows come back with exactly the PlanSpec v1 task allowlist, in a
    fixed key order, rebuilt from validated values rather than passed through."""
    result = _validate(_document(tasks=[_task(), _task(id="t-0002", assignee=None)]))

    assert [list(task) for task in result["tasks"]] == [list(krm.TASK_FIELDS)] * 2
    assert result["tasks"][0] == _task()
    assert result["tasks"][1]["assignee"] is None


@pytest.mark.parametrize(
    "case, tasks",
    [
        ("task-not-an-object", ["t-0001"]),
        ("unknown-task-field", [_task(body="secret worklog")]),
        ("missing-task-field", [{k: v for k, v in _task().items() if k != "status"}]),
        ("empty-id", [_task(id="")]),
        ("over-long-id", [_task(id="t" * 200)]),
        ("over-long-title", [_task(title="t" * (krm.MAX_FIELD_CHARS + 1))]),
        ("newline-in-title", [_task(title="ok\nkanban read-model: totally fine")]),
        ("nul-in-title", [_task(title="ok\x00hidden")]),
        ("ansi-escape-in-title", [_task(title="ok\x1b[31mred")]),
        ("carriage-return-in-status", [_task(status="done\rready")]),
        ("control-char-in-assignee", [_task(assignee="thomas\x07")]),
        ("status-not-a-string", [_task(status=None)]),
        ("empty-status", [_task(status="")]),
        ("assignee-wrong-type", [_task(assignee=7)]),
        ("priority-bool", [_task(priority=True)]),
        ("priority-float", [_task(priority=1.5)]),
        ("priority-out-of-range", [_task(priority=10**9)]),
        ("created-at-negative", [_task(created_at=-1)]),
        ("created-at-string", [_task(created_at="1700000000")]),
        ("started-at-wrong-type", [_task(started_at="1700000000")]),
        ("completed-at-bool", [_task(completed_at=False)]),
    ],
)
def test_validate_artifact_rejects_bad_task_rows(case, tasks):
    """Every task row must match the allowlist exactly. Control characters
    are refused outright rather than escaped: a title carrying ``\\n``,
    ``\\x00`` or an ANSI escape is how a row forges an extra line of CLI
    output, and there is no version of such a row worth showing."""
    with pytest.raises(krm.ReadModelUnavailable):
        _validate(_document(tasks=tasks))


def test_validate_artifact_rejects_more_rows_than_the_cap():
    """More rows than the publisher is allowed to emit is unavailable."""
    with pytest.raises(krm.ReadModelUnavailable):
        _validate(_document(tasks=[_task(id=f"t-{i}") for i in range(krm.MAX_TASKS + 1)]))


def test_validate_artifact_accepts_exactly_the_row_cap():
    """The row cap itself is inclusive."""
    tasks = [_task(id=f"t-{i}") for i in range(krm.MAX_TASKS)]

    assert len(_validate(_document(tasks=tasks))["tasks"]) == krm.MAX_TASKS


# ---------------------------------------------------------------------------
# Cycle 14 — board slug validation, without normalization or fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("board", ["default", "acme", "a", "a-b_1", "0", "x" * 64])
def test_validated_board_returns_the_slug_unchanged(board):
    """A valid slug is returned byte-for-byte: the reader never lowercases,
    trims, or otherwise rewrites the caller's explicit board."""
    assert krm.validated_board(board) == board


@pytest.mark.parametrize(
    "board",
    [
        "", "ACME", "Acme", " acme", "acme ", "acme\n", "acme\t", "-acme", "_acme",
        "acme/evil", "../acme", "acme.db", "acme;rm", "x" * 65, "acme\x00",
    ],
)
def test_validated_board_rejects_anything_outside_the_slug_grammar(board):
    """Uppercase, whitespace (including a trailing newline, which a ``match``
    based check would wrongly accept), path separators, and over-long slugs
    are rejected rather than repaired."""
    with pytest.raises(krm.ReadModelUnavailable):
        krm.validated_board(board)


def test_validated_board_rejects_a_non_string():
    with pytest.raises(krm.ReadModelUnavailable):
        krm.validated_board(None)


# ---------------------------------------------------------------------------
# Cycle 15 — read_read_model: the whole reader, end to end
# ---------------------------------------------------------------------------


def _publish(tmp_path, root, **document_overrides) -> Path:
    """Write a synthetic artifact bound to ``root``, outside ``root``."""
    document = _document(root_fingerprint=krm.root_fingerprint(root), **document_overrides)
    return _write_artifact(
        tmp_path / "published" / "read-model.json",
        json.dumps(document).encode("utf-8"),
    )


def _read(tmp_path, root, artifact, **overrides):
    kwargs = {
        "kanban_root": root,
        "board": "acme",
        "artifact": artifact,
        "max_age_seconds": 3600,
        "limit": 50,
        "now": GENERATED_AT + 5,
    }
    kwargs.update(overrides)
    return krm.read_read_model(**kwargs)


def _synthetic_root(tmp_path) -> Path:
    root = tmp_path / "kanban-root"
    root.mkdir()
    return root


def test_read_read_model_returns_the_sanitized_payload(tmp_path):
    """The happy path: an external, fresh, correctly bound 0600 artifact is
    read through the fd walk and returned as the allowlisted payload."""
    root = _synthetic_root(tmp_path)
    artifact = _publish(tmp_path, root, tasks=[_task()])

    assert _read(tmp_path, root, artifact) == {
        "schema_version": krm.SCHEMA_VERSION,
        "generated_at": GENERATED_AT,
        "board": "acme",
        "truncated": False,
        "tasks": [_task()],
    }


def test_read_read_model_applies_the_limit_and_reports_truncation(tmp_path):
    """``--limit`` trims the rows the caller sees and is reported honestly:
    a trimmed result is marked truncated rather than looking complete."""
    root = _synthetic_root(tmp_path)
    tasks = [_task(id=f"t-{i}") for i in range(10)]
    artifact = _publish(tmp_path, root, tasks=tasks)

    result = _read(tmp_path, root, artifact, limit=3)

    assert [task["id"] for task in result["tasks"]] == ["t-0", "t-1", "t-2"]
    assert result["truncated"] is True


def test_read_read_model_keeps_the_publisher_truncation_flag(tmp_path):
    """If the publisher already truncated, the reader cannot un-truncate it
    by staying under the limit."""
    root = _synthetic_root(tmp_path)
    artifact = _publish(tmp_path, root, tasks=[_task()], truncated=True)

    assert _read(tmp_path, root, artifact, limit=50)["truncated"] is True


@pytest.mark.parametrize("limit", [0, -1, krm.MAX_TASKS + 1, 1.5, True, None])
def test_read_read_model_rejects_an_unusable_limit(limit, tmp_path):
    """The limit must be a plain int between 1 and the row cap."""
    root = _synthetic_root(tmp_path)
    artifact = _publish(tmp_path, root, tasks=[_task()])

    with pytest.raises(krm.ReadModelUnavailable):
        _read(tmp_path, root, artifact, limit=limit)


def test_read_read_model_defaults_now_to_the_wall_clock(tmp_path):
    """With no explicit ``now``, freshness is judged against the real clock —
    so an artifact stamped in 2023 is stale, not silently accepted."""
    root = _synthetic_root(tmp_path)
    artifact = _publish(tmp_path, root, tasks=[_task()])

    with pytest.raises(krm.ReadModelUnavailable):
        _read(tmp_path, root, artifact, now=None)


@pytest.mark.parametrize(
    "case, overrides",
    [
        ("bad-board", {"board": "ACME"}),
        ("board-not-in-artifact", {"board": "other"}),
        ("relative-root", {"kanban_root": "kanban-root"}),
        ("missing-root", {"kanban_root": "/nonexistent-5a1c/kanban-root"}),
    ],
)
def test_read_read_model_rejects_bad_explicit_inputs(case, overrides, tmp_path):
    """Root, board, and artifact are all explicit required inputs; a bad one
    is unavailable rather than being resolved from somewhere else."""
    root = _synthetic_root(tmp_path)
    artifact = _publish(tmp_path, root, tasks=[_task()])

    with pytest.raises(krm.ReadModelUnavailable):
        _read(tmp_path, root, artifact, **overrides)


# ---------------------------------------------------------------------------
# Cycle 17 — runtime audit-hook proof of the nonmutation boundary
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).parents[2]

_AUDIT_DRIVER = r'''
"""Runs three reads under a CPython audit hook and reports what was touched.

Everything is synthetic and built BEFORE the hook is installed, so the setup's
own writes are not counted. The hook cannot be removed once installed, which is
why this runs in its own process instead of inside pytest.
"""
import json, os, sys, time

sys.path.insert(0, sys.argv[1])
from hermes_cli import kanban_read_model as krm

base = os.path.abspath(sys.argv[2])
root = os.path.join(base, "kanban-root")
os.makedirs(os.path.join(root, "kanban", "boards", "acme"), exist_ok=True)
now = int(time.time())

# A synthetic live database with WAL/SHM sidecars, exactly where the rejected
# design would have looked for it. If the reader derives that path at all, the
# audit hook below will see it.
for directory in (root, os.path.join(root, "kanban", "boards", "acme")):
    for name in ("kanban.db", "kanban.db-wal", "kanban.db-shm"):
        with open(os.path.join(directory, name), "wb") as handle:
            handle.write(b"synthetic sqlite payload")

published = os.path.join(base, "published")
os.makedirs(published, exist_ok=True)


def _document(**overrides):
    document = {
        "schema_version": krm.SCHEMA_VERSION,
        "generated_at": now,
        "board": "acme",
        "root_fingerprint": krm.root_fingerprint(root),
        "truncated": False,
        "tasks": [{
            "id": "t-0001", "title": "Ship the read model", "status": "ready",
            "assignee": "thomas", "priority": 3, "created_at": now - 10,
            "started_at": None, "completed_at": None,
        }],
    }
    document.update(overrides)
    return document


def _write(name, payload):
    path = os.path.join(published, name)
    with open(path, "wb") as handle:
        handle.write(payload)
    os.chmod(path, 0o600)
    return path


artifacts = {
    "success": _write("fresh.json", json.dumps(_document()).encode()),
    "stale": _write("stale.json", json.dumps(_document(generated_at=now - 86400)).encode()),
    "invalid": _write("invalid.json", b"{not json at all"),
}

events = []
sys.addaudithook(lambda event, args: events.append((event, [
    a if isinstance(a, str) else (a.decode("utf-8", "replace") if isinstance(a, bytes) else None)
    for a in args
])))

outcomes = {}
for label, artifact in artifacts.items():
    try:
        krm.read_read_model(
            kanban_root=root, board="acme", artifact=artifact,
            max_age_seconds=3600, limit=50,
        )
        outcomes[label] = "ok"
    except krm.ReadModelUnavailable:
        outcomes[label] = "unavailable"

sys.stdout.write(json.dumps({
    "root": root, "published": published, "outcomes": outcomes, "events": events,
}))
'''


def _run_audit_driver(tmp_path) -> dict:
    driver = tmp_path / "audit_driver.py"
    driver.write_text(_AUDIT_DRIVER, encoding="utf-8")
    home = tmp_path / "synthetic-home"
    home.mkdir()

    # Scrubbed environment: no HERMES_* variable, no real HOME, nothing that
    # could point the reader at the real Kanban installation.
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(home),
        "TZ": "UTC",
        "LANG": "C.UTF-8",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    result = subprocess.run(
        [sys.executable, str(driver), str(REPO_ROOT), str(tmp_path / "workspace")],
        env=env, cwd=str(tmp_path), capture_output=True, text=True, timeout=120, check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_reader_makes_no_database_connection_and_no_kanban_root_open(tmp_path):
    """Across a successful, a stale, and an invalid read: zero
    ``sqlite3.connect`` events and nothing under the synthetic Kanban root —
    which holds a `kanban.db` plus WAL and SHM sidecars — is ever opened."""
    report = _run_audit_driver(tmp_path)
    root = report["root"]

    assert report["outcomes"] == {
        "success": "ok",
        "stale": "unavailable",
        "invalid": "unavailable",
    }

    assert not [event for event, _args in report["events"] if "sqlite" in event]

    touched = [
        (event, arg)
        for event, args in report["events"]
        for arg in args
        if isinstance(arg, str) and (arg == root or arg.startswith(root + os.sep))
    ]
    assert touched == []

    named_db = [
        (event, arg)
        for event, args in report["events"]
        for arg in args
        if isinstance(arg, str) and "kanban.db" in arg
    ]
    assert named_db == []


def test_the_audit_hook_proof_is_not_vacuous(tmp_path):
    """Positive control: the same hook DOES observe the artifact's component
    walk. Without this, a hook that recorded nothing at all would make the
    test above pass while proving nothing."""
    report = _run_audit_driver(tmp_path)

    opens = [args for event, args in report["events"] if event == "open"]
    assert opens, "the audit hook recorded no open events at all"
    assert [args for args in opens if args and args[0] == "fresh.json"], opens


# ---------------------------------------------------------------------------
# Cycle 18 — the live-WAL Kanban root is byte-for-byte unchanged
# ---------------------------------------------------------------------------


def _manifest(root: Path) -> dict:
    """Path, inode, size, mtime_ns and content digest for everything under
    ``root`` — enough to catch a mutation that preserves size or timestamps."""
    manifest = {}
    for path in sorted(root.rglob("*")):
        st = path.lstat()
        digest = (
            hashlib.sha256(path.read_bytes()).hexdigest() if stat.S_ISREG(st.st_mode) else None
        )
        manifest[str(path.relative_to(root))] = (
            st.st_ino, st.st_mode, st.st_size, st.st_mtime_ns, digest
        )
    return manifest


def _live_wal_kanban_root(tmp_path) -> "tuple[Path, sqlite3.Connection]":
    """A synthetic Kanban root holding a real WAL-mode SQLite database with
    its `-wal` and `-shm` sidecars present and a writer still connected.

    This is the exact shape that made the earlier live-read design fail: on
    macOS/SQLite 3.53.3 a `mode=ro` + `query_only=ON` read of such a database
    still changed `kanban.db-shm`, and against an idle WAL database with no
    sidecars it created them.
    """
    root = tmp_path / "kanban-root"
    root.mkdir()
    connection = sqlite3.connect(root / "kanban.db")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT)")
    connection.execute("INSERT INTO tasks VALUES ('t-0001', 'Ship the read model')")
    connection.commit()

    for sidecar in ("kanban.db", "kanban.db-wal", "kanban.db-shm"):
        assert (root / sidecar).exists(), sidecar
    return root, connection


@pytest.mark.parametrize("case", ["success", "stale", "malformed", "missing-artifact"])
def test_reads_leave_a_live_wal_kanban_root_byte_for_byte_identical(case, tmp_path):
    """A successful read, a stale one, a malformed one, and a missing artifact
    all leave every file under a live-WAL Kanban root — database, WAL and SHM
    included — with the same inode, mode, size, mtime and SHA-256."""
    root, connection = _live_wal_kanban_root(tmp_path)
    try:
        now = GENERATED_AT
        document = _document(root_fingerprint=krm.root_fingerprint(root), tasks=[_task()])
        if case == "stale":
            document["generated_at"] = now - 86_400
        payload = (
            b"{not json at all"
            if case == "malformed"
            else json.dumps(document).encode("utf-8")
        )
        artifact = _write_artifact(tmp_path / "published" / "read-model.json", payload)
        if case == "missing-artifact":
            artifact.unlink()

        before = _manifest(root)

        try:
            result = _read(tmp_path, root, artifact, now=now)
            assert case == "success" and result["tasks"] == [_task()]
        except krm.ReadModelUnavailable:
            assert case != "success"

        assert _manifest(root) == before
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# Cycle 19 — the real CLI process, scrubbed environment, nothing mutated
# ---------------------------------------------------------------------------


def _run_read_model_cli(tmp_path, home: Path, *args: str):
    """Run the real `hermes kanban read-model` process with a scrubbed
    environment: a synthetic HERMES_HOME, no HERMES_KANBAN_* variable, and no
    inherited pointer to the owner's real Kanban installation."""
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(home),
        "HERMES_HOME": str(home),
        "PYTHONPATH": str(REPO_ROOT),
        "TZ": "UTC",
        "LANG": "C.UTF-8",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "kanban", *args],
        env=env, cwd=str(tmp_path), capture_output=True, text=True, timeout=120, check=False,
    )


def _warmed_up_home(tmp_path) -> Path:
    """A synthetic HERMES_HOME that has already absorbed the CLI's own
    first-run scaffolding.

    `hermes_cli.main` creates `SOUL.md` and `logs/` on startup for every
    command, before `read-model` is dispatched at all. That is pre-existing
    generic CLI behaviour, not something this slice may change — main.py is
    outside H0.2b3a's allowed files. Warming the home up with one throwaway
    invocation first (a usage error, which never reaches the reader) and
    snapshotting afterwards isolates what `read-model` itself writes, which
    must be nothing: a second invocation leaves this directory byte-identical.
    """
    home = tmp_path / "synthetic-home"
    home.mkdir()
    warm_up = _run_read_model_cli(tmp_path, home, "--board", "acme", "read-model")
    assert warm_up.returncode == 2, warm_up.stderr
    return home


def test_cli_subprocess_reads_the_artifact_and_mutates_nothing(tmp_path):
    """End to end through the real CLI process: a fresh, correctly bound
    artifact is served on stdout with rc=0, while the live-WAL Kanban root and
    the synthetic HERMES_HOME are both left byte-for-byte identical."""
    root, connection = _live_wal_kanban_root(tmp_path)
    try:
        home = _warmed_up_home(tmp_path)
        now = int(time.time())
        artifact = _write_artifact(
            tmp_path / "published" / "read-model.json",
            json.dumps(
                _document(
                    root_fingerprint=krm.root_fingerprint(root),
                    generated_at=now,
                    tasks=[_task(created_at=now - 60)],
                )
            ).encode("utf-8"),
        )

        root_before, home_before = _manifest(root), _manifest(home)

        result = _run_read_model_cli(
            tmp_path, home,
            "--kanban-root", str(root), "--board", "acme",
            "read-model", "--artifact", str(artifact), "--json",
        )

        assert result.returncode == 0, result.stderr
        assert result.stderr == ""
        assert json.loads(result.stdout) == {
            "schema_version": krm.SCHEMA_VERSION,
            "generated_at": now,
            "board": "acme",
            "truncated": False,
            "tasks": [_task(created_at=now - 60)],
        }
        assert _manifest(root) == root_before
        assert _manifest(home) == home_before
    finally:
        connection.close()


@pytest.mark.parametrize(
    "case",
    ["stale", "wrong-board-binding", "group-readable", "missing", "inside-the-root"],
)
def test_cli_subprocess_reports_one_generic_failure_and_mutates_nothing(case, tmp_path):
    """Every hostile or unusable artifact produces rc=1, exactly
    `kanban read-model: unavailable`, empty stdout, and an untouched Kanban
    root — and the messages are identical across causes, so the command cannot
    be used to probe the filesystem."""
    root, connection = _live_wal_kanban_root(tmp_path)
    try:
        home = _warmed_up_home(tmp_path)
        now = int(time.time())
        document = _document(
            root_fingerprint=krm.root_fingerprint(root),
            generated_at=now - (86_400 if case == "stale" else 0),
            board="other" if case == "wrong-board-binding" else "acme",
            tasks=[_task(created_at=now - 60)],
        )
        destination = (
            root / "published" / "read-model.json"
            if case == "inside-the-root"
            else tmp_path / "published" / "read-model.json"
        )
        artifact = _write_artifact(
            destination,
            json.dumps(document).encode("utf-8"),
            0o640 if case == "group-readable" else 0o600,
        )
        if case == "missing":
            artifact.unlink()

        root_before, home_before = _manifest(root), _manifest(home)

        result = _run_read_model_cli(
            tmp_path, home,
            "--kanban-root", str(root), "--board", "acme",
            "read-model", "--artifact", str(artifact), "--json",
        )

        assert result.returncode == 1
        assert result.stdout == ""
        assert result.stderr.strip() == "kanban read-model: unavailable"
        assert str(artifact) not in result.stderr
        assert _manifest(root) == root_before
        assert _manifest(home) == home_before
    finally:
        connection.close()


def test_cli_subprocess_rejects_a_missing_artifact_argument_as_usage(tmp_path):
    """A missing explicit input stays a usage error (rc=2) and is reported as
    one — usage errors are the one case that may say what is wrong, because
    they describe the caller's own command line, not the filesystem."""
    home = _warmed_up_home(tmp_path)

    result = _run_read_model_cli(
        tmp_path, home, "--kanban-root", str(tmp_path), "--board", "acme", "read-model"
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.strip() == (
        "kanban read-model: explicit --kanban-root, --board and --artifact are required"
    )


# ---------------------------------------------------------------------------
# Cycle 20 — platforms without openat/O_NOFOLLOW fail closed, not at import
# ---------------------------------------------------------------------------


def test_reader_module_imports_without_the_posix_open_constants(monkeypatch):
    """``O_NOFOLLOW``, ``O_DIRECTORY``, ``O_CLOEXEC`` and ``O_NONBLOCK`` do not
    exist on Windows. Referencing them at module scope would make importing
    ``hermes_cli.kanban`` — which imports this reader — fail outright there, so
    the whole Kanban CLI would die for a command nobody ran."""
    for flag in ("O_NOFOLLOW", "O_DIRECTORY", "O_CLOEXEC", "O_NONBLOCK"):
        monkeypatch.delattr(os, flag, raising=False)
    monkeypatch.setattr(os, "supports_dir_fd", set())

    try:
        importlib.reload(krm)
        assert krm.ReadModelUnavailable is not None
    finally:
        monkeypatch.undo()
        importlib.reload(krm)


def test_read_artifact_bytes_fails_closed_without_openat(tmp_path, monkeypatch):
    """Where the fd-based walk is unavailable, the reader refuses to read at
    all rather than falling back to a path-resolving open — the fallback is
    exactly the TOCTOU-prone read this design exists to avoid."""
    artifact = _write_artifact(tmp_path / "published" / "read-model.json", b'{"ok": true}')
    monkeypatch.setattr(krm, "_HAS_FD_WALK", False)

    with pytest.raises(krm.ReadModelUnavailable):
        krm.read_artifact_bytes(artifact)


# ---------------------------------------------------------------------------
# Cycle 20 — the roadmap status enum, with `archived` excluded
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status",
    ["archived", "closed", "in_progress", "READY", "ready ", " ready"],
)
def test_validate_artifact_rejects_statuses_outside_the_roadmap_enum(status):
    """`status` is an enum, not free text. `archived` is excluded outright
    (roadmap M2/H0: archived tasks never reach the read model), and an
    unknown, mis-cased, or space-padded value is a publisher this reader does
    not recognise rather than a status worth rendering."""
    with pytest.raises(krm.ReadModelUnavailable):
        _validate(_document(tasks=[_task(status=status)]))


# ---------------------------------------------------------------------------
# Cycle 21 — duplicate task IDs fail closed
# ---------------------------------------------------------------------------


def test_validate_artifact_rejects_duplicate_task_ids():
    """`id` is the unique final tie-breaker for the read model's deterministic
    order, so two rows sharing one are not a cosmetic wart: they make the order
    ambiguous and let one row's title be attributed to another task's id. There
    is no honest way to render that, so the whole artifact is unavailable."""
    tasks = [_task(id="t-0001", title="real"), _task(id="t-0001", title="impostor")]

    with pytest.raises(krm.ReadModelUnavailable):
        _validate(_document(tasks=tasks))


# ---------------------------------------------------------------------------
# Cycle 22 — deterministic `priority DESC, id ASC` order, applied before limit
# ---------------------------------------------------------------------------


def test_read_read_model_orders_by_priority_then_id_before_applying_the_limit(tmp_path):
    """The roadmap fixes one order: `priority DESC, id ASC`, with `id` as the
    unique final tie-breaker. The reader imposes it rather than trusting the
    artifact's own row sequence, and it does so BEFORE the limit — otherwise
    `--limit 2` returns whichever two rows the publisher happened to write
    first, and a low-priority task can hide the board's most urgent one."""
    root = _synthetic_root(tmp_path)
    artifact = _publish(
        tmp_path,
        root,
        tasks=[
            _task(id="t-c", priority=1),
            _task(id="t-a", priority=5),
            _task(id="t-d", priority=9),
            _task(id="t-b", priority=5),
        ],
    )

    ordered = _read(tmp_path, root, artifact, limit=50)
    top_two = _read(tmp_path, root, artifact, limit=2)

    assert [task["id"] for task in ordered["tasks"]] == ["t-d", "t-a", "t-b", "t-c"]
    assert [task["id"] for task in top_two["tasks"]] == ["t-d", "t-a"]


# ---------------------------------------------------------------------------
# Cycle 23 — Unicode bidi formatting controls are refused, not rendered
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, char",
    [
        ("ALM", "\u061c"),
        ("LRM", "\u200e"),
        ("RLM", "\u200f"),
        ("LRE", "\u202a"),
        ("RLE", "\u202b"),
        ("PDF", "\u202c"),
        ("LRO", "\u202d"),
        ("RLO", "\u202e"),
        ("LRI", "\u2066"),
        ("RLI", "\u2067"),
        ("FSI", "\u2068"),
        ("PDI", "\u2069"),
    ],
)
@pytest.mark.parametrize("field", ["title", "assignee"])
def test_validate_artifact_rejects_bidi_formatting_controls(field, name, char):
    """Bidi formatting controls are invisible but reorder everything after
    them, so a title carrying `RLO` renders as text the publisher never wrote —
    the Trojan Source trick, aimed at whoever reads the terminal. They survive
    the C0/C1 control check because they are ordinary printable code points, so
    they are refused explicitly."""
    tasks = [_task(**{field: f"ship{char}the read model"})]

    with pytest.raises(krm.ReadModelUnavailable):
        _validate(_document(tasks=tasks))


# ---------------------------------------------------------------------------
# Cycle 24 — task IDs follow a safe-character grammar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "task_id",
    ["t 0001", "t/0001", "../kanban-root", "t.0001", "t:0001", "t;0001",
     "t\u00fc0001", "-\u2013t", "t*", "t\u00a00001"],
)
def test_validate_artifact_rejects_task_ids_outside_the_safe_grammar(task_id):
    """A task id is an opaque handle, not free text: it is the sort key, it is
    what a caller feeds back into other tooling, and it is printed unquoted.
    Restricting it to `[A-Za-z0-9_-]` keeps path separators, whitespace,
    shell punctuation, and look-alike Unicode out of all three roles."""
    with pytest.raises(krm.ReadModelUnavailable):
        _validate(_document(tasks=[_task(id=task_id)]))


# ---------------------------------------------------------------------------
# Cycle 25 — textual fields are bounded in UTF-8 bytes, not just code points
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field, max_chars",
    [("title", "MAX_FIELD_CHARS"), ("assignee", "MAX_ID_CHARS")],
)
def test_validate_artifact_rejects_text_over_the_utf8_byte_cap(field, max_chars):
    """The code-point count is not the size. A string of astral characters
    stays under the character cap while weighing four bytes each, so it can
    quadruple the serialized artifact and the terminal line it produces. The
    roadmap requires a UTF-8 byte cap alongside the character cap, so a value
    that is short in code points and long in bytes is still refused."""
    oversized = "\U0001f600" * (getattr(krm, max_chars) - 1)
    assert len(oversized) < getattr(krm, max_chars)

    with pytest.raises(krm.ReadModelUnavailable):
        _validate(_document(tasks=[_task(**{field: oversized})]))
