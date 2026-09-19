"""H0 reader for the owner-published Kanban read-model JSON artifact.

This module is the ENTIRE H0 read path. It is deliberately isolated from the
rest of the Kanban stack: it must never import ``sqlite3``, ``kanban_db``,
``hermes_state``, ``sqlite_safe_read``, or the generic ``utils`` helpers, and
it must never open anything underneath the Kanban root. The live database is
read exclusively by the separately-approved owner-side publisher (H0.2b3b);
H0 consumes only the sanitized artifact the publisher emits outside every
Kanban-controlled root.

See ``.hermes/plans/2026-09-14_175055-h0-2b3-owner-published-json-read-model``
work unit H0.2b3a.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import time
from pathlib import Path


class ReadModelUnavailable(Exception):
    """Every reader failure. The message is for developers only.

    Callers must never surface it: the CLI reports one fixed generic string
    so a caller cannot probe the filesystem through error text.
    """


def validated_kanban_root(kanban_root: "os.PathLike[str] | str") -> Path:
    """Validate the explicit Kanban root and return it as a canonical Path.

    This is the ONLY Kanban-root-side check the reader performs. It exists
    purely to bind the artifact to one root (see :func:`root_fingerprint`) —
    it deliberately does not derive, ``lstat``, or open ``kanban.db`` or its
    WAL/SHM sidecars, because doing so is what made the rejected live-read
    design mutate the database.

    The root must be absolute, free of ``..`` alias segments (so the caller's
    spelling is already canonical), symlink-free in every component, and an
    existing directory. Anything else raises :class:`ReadModelUnavailable`.
    """
    root = Path(os.fspath(kanban_root))
    if not root.is_absolute():
        raise ReadModelUnavailable(f"kanban root is not absolute: {root}")
    if ".." in root.parts:
        raise ReadModelUnavailable(f"kanban root is not a canonical spelling: {root}")

    # Walk from the filesystem anchor down, lstat-ing (never stat-ing) each
    # component so a symlink is caught here instead of silently followed.
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current = current / part
        try:
            st = os.lstat(current)
        except OSError as exc:
            raise ReadModelUnavailable(f"kanban root component unusable: {current}") from exc
        if stat.S_ISLNK(st.st_mode):
            raise ReadModelUnavailable(f"kanban root component is a symlink: {current}")
        if not stat.S_ISDIR(st.st_mode):
            raise ReadModelUnavailable(f"kanban root component is not a directory: {current}")

    return root


# Board slugs the reader will accept. ``fullmatch`` (not ``match``) so a
# trailing newline cannot ride along on an otherwise valid slug.
_BOARD_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


def validated_board(board: object) -> str:
    """Return the explicit board slug unchanged, or refuse it.

    No lowercasing, trimming, or repair: a board the caller did not spell
    exactly is a board the caller did not ask for, and there is no ambient
    current-board fallback in this reader at all.
    """
    if not isinstance(board, str):
        raise ReadModelUnavailable("board is not a string")
    if not _BOARD_RE.fullmatch(board):
        raise ReadModelUnavailable("board is not a valid slug")
    return board


def validated_artifact_path(
    artifact: "os.PathLike[str] | str",
    kanban_root: "os.PathLike[str] | str",
) -> Path:
    """Validate the explicit artifact path against the explicit Kanban root.

    The artifact must be named absolutely, without ``..`` alias segments, and
    must live outside the Kanban root — the whole point of the published-
    artifact design is that H0 touches nothing the live database controls.

    Containment is checked lexically. That is sufficient because a symlink
    that pointed back under the root could never survive the ``O_NOFOLLOW``
    component walk in :func:`_open_artifact_fd`, and resolving the path here
    would mean stat-ing candidate paths under the Kanban root, which H0 must
    never do.
    """
    path = Path(os.fspath(artifact))
    if not path.is_absolute():
        raise ReadModelUnavailable(f"artifact path is not absolute: {path}")
    if ".." in path.parts:
        raise ReadModelUnavailable(f"artifact path is not a canonical spelling: {path}")

    root = Path(os.fspath(kanban_root))
    if path == root or root in path.parents:
        raise ReadModelUnavailable(f"artifact path is inside the kanban root: {path}")
    return path


# Hard cap on artifact size. The reader never buffers more than this plus one
# probe byte, so a hostile or runaway artifact cannot exhaust memory.
MAX_ARTIFACT_BYTES = 256 * 1024

# The only mode an owner-published artifact may carry.
ARTIFACT_MODE = 0o600

# Default freshness window. Publication is an explicit manual owner action, so
# this is deliberately short: an artifact nobody refreshed is unavailable
# rather than quietly ageing into a wrong answer.
DEFAULT_MAX_AGE_SECONDS = 300

# Flags shared by every component open in the walk below. O_NOFOLLOW makes a
# symlinked component fail (ELOOP) instead of being silently traversed and
# O_CLOEXEC keeps the fd out of any child process.
#
# They are read via getattr because none of them exist on Windows: naming
# them directly would make importing this module — and therefore the whole
# Kanban CLI, which imports it — fail on a platform where nobody even ran
# `read-model`. Missing support is handled at call time instead, by refusing
# to read rather than by falling back to a path-resolving open.
_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
_DIR_OPEN_FLAGS = _OPEN_FLAGS | getattr(os, "O_DIRECTORY", 0)
# The final component additionally gets O_NONBLOCK: opening a FIFO O_RDONLY
# blocks until a writer appears, so without it a planted FIFO would hang the
# reader indefinitely instead of failing closed. O_NONBLOCK has no effect on
# the regular file the artifact is required to be.
_FILE_OPEN_FLAGS = _OPEN_FLAGS | getattr(os, "O_NONBLOCK", 0)

# The fd-based walk needs both O_NOFOLLOW and openat support; without either,
# this reader has no safe way to open anything.
_HAS_FD_WALK = hasattr(os, "O_NOFOLLOW") and os.open in getattr(os, "supports_dir_fd", set())


def _open_artifact_fd(artifact: Path) -> int:
    """Open ``artifact`` one path component at a time and return its fd.

    Each component is opened relative to the previous component's directory
    fd (``openat``) with ``O_NOFOLLOW``, so no component — not the final file
    and not any parent — can be a symlink, and nothing that happens to the
    path *after* a component is opened can redirect the walk: every step is
    pinned to the inode it opened.
    """
    if not _HAS_FD_WALK:
        # No silent fallback to a plain open(): resolving the whole path in
        # one call is precisely the TOCTOU-prone read this design replaced.
        raise ReadModelUnavailable("platform lacks openat/O_NOFOLLOW")

    dir_fd = os.open(artifact.anchor, _DIR_OPEN_FLAGS)
    try:
        *parents, name = artifact.parts[1:]
        for part in parents:
            next_fd = os.open(part, _DIR_OPEN_FLAGS, dir_fd=dir_fd)
            os.close(dir_fd)
            dir_fd = next_fd
        return os.open(name, _FILE_OPEN_FLAGS, dir_fd=dir_fd)
    finally:
        os.close(dir_fd)


def _validated_artifact_fstat(fd: int) -> os.stat_result:
    """Validate the *opened fd itself*, never a path.

    Checking the fd rather than the path is what makes this race-free: the
    file these checks describe is exactly the file the subsequent reads come
    from, no matter what happens to the path afterwards.
    """
    try:
        st = os.fstat(fd)
    except OSError as exc:
        raise ReadModelUnavailable("artifact is not statable") from exc
    if not stat.S_ISREG(st.st_mode):
        raise ReadModelUnavailable("artifact is not a regular file")
    if st.st_nlink != 1:
        # A second hard link is a second writable name for the same inode,
        # outside whatever protection the published directory provides.
        raise ReadModelUnavailable("artifact has more than one link")
    if st.st_uid != os.getuid():
        raise ReadModelUnavailable("artifact is not owned by the current user")
    if stat.S_IMODE(st.st_mode) != ARTIFACT_MODE:
        # Exact match, not a "no group/world bits" subset check: 0600 is the
        # only mode the owner-side publisher is allowed to produce.
        raise ReadModelUnavailable("artifact mode is not 0600")
    if st.st_size > MAX_ARTIFACT_BYTES:
        raise ReadModelUnavailable("artifact exceeds the size cap")
    return st


def read_artifact_bytes(artifact: "os.PathLike[str] | str") -> bytes:
    """Return the artifact's bytes, read only from the validated fd.

    The path is never re-opened after validation: everything below operates
    on the single fd produced by :func:`_open_artifact_fd`.
    """
    path = Path(os.fspath(artifact))
    try:
        fd = _open_artifact_fd(path)
    except OSError as exc:
        raise ReadModelUnavailable("artifact is not openable") from exc
    try:
        st = _validated_artifact_fstat(fd)
        chunks: list[bytes] = []
        # One byte past the cap, so a file that grew after it was measured is
        # detected rather than silently truncated to a valid-looking prefix.
        remaining = MAX_ARTIFACT_BYTES + 1
        while remaining > 0:
            chunk = os.read(fd, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    except OSError as exc:
        raise ReadModelUnavailable("artifact is not readable") from exc
    finally:
        os.close(fd)

    payload = b"".join(chunks)
    if len(payload) > MAX_ARTIFACT_BYTES:
        raise ReadModelUnavailable("artifact exceeds the size cap")
    if len(payload) != st.st_size:
        # The artifact changed size between fstat and read: it is not the
        # immutable snapshot the contract requires, so there is no version of
        # it we can honestly report.
        raise ReadModelUnavailable("artifact changed while being read")
    return payload


def _no_duplicate_keys(pairs: "list[tuple[str, object]]") -> "dict[str, object]":
    """``object_pairs_hook`` that refuses repeated keys at any depth.

    ``json.loads`` keeps the *last* occurrence by default, so without this a
    document could carry a benign-looking ``"board"`` for anyone eyeballing it
    and a second one that actually wins.
    """
    seen: set[str] = set()
    for key, _value in pairs:
        if key in seen:
            raise ReadModelUnavailable("artifact contains a duplicate key")
        seen.add(key)
    return dict(pairs)


def _reject_json_constant(name: str) -> "None":
    """``parse_constant`` hook. Python's json module accepts the non-standard
    ``NaN``/``Infinity``/``-Infinity`` literals unless told otherwise."""
    raise ReadModelUnavailable(f"artifact contains the non-standard constant {name}")


def parse_artifact_json(payload: bytes) -> "dict[str, object]":
    """Parse the artifact bytes strictly and return the top-level object."""
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReadModelUnavailable("artifact is not valid UTF-8") from exc
    try:
        document = json.loads(
            text,
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (ValueError, RecursionError) as exc:
        raise ReadModelUnavailable("artifact is not valid JSON") from exc
    if not isinstance(document, dict):
        raise ReadModelUnavailable("artifact is not a JSON object")
    return document


# Domain separator so this digest can never collide with a hash of the same
# bytes computed for any other purpose.
_FINGERPRINT_DOMAIN = b"hermes.kanban.read-model.root.v1\x00"


def root_fingerprint(kanban_root: "os.PathLike[str] | str") -> str:
    """Return the binding fingerprint for one canonical Kanban root.

    Pure by design: it hashes the root's path bytes and nothing else. Deriving
    it from filesystem metadata (inode, device, a file under the root) would
    mean H0 stat-ing the Kanban tree, which is exactly what this architecture
    exists to avoid.
    """
    raw = os.fsencode(str(Path(os.fspath(kanban_root))))
    return hashlib.sha256(_FINGERPRINT_DOMAIN + raw).hexdigest()


# The one artifact schema this reader understands. A publisher emitting
# anything else is a stranger, not a newer friend.
SCHEMA_VERSION = "hermes.kanban.read-model.v1"

# Exact top-level key set. Unknown keys are refused rather than ignored: an
# artifact carrying fields this reader does not understand is not the artifact
# this reader was written against.
TOP_LEVEL_FIELDS = (
    "schema_version",
    "generated_at",
    "board",
    "root_fingerprint",
    "truncated",
    "tasks",
)


# Exact task-row key set, in the order the reader re-emits them.
TASK_FIELDS = (
    "id",
    "title",
    "status",
    "assignee",
    "priority",
    "created_at",
    "started_at",
    "completed_at",
)

# Most rows the reader will accept. The owner-side publisher is capped at the
# same number, so a larger artifact is a malformed one.
MAX_TASKS = 200

# Per-string-field length caps and a sanity bound on priority. These are
# defence-in-depth bounds on how much attacker-influenced text can reach the
# terminal, not Kanban domain rules.
MAX_FIELD_CHARS = 512
MAX_ID_CHARS = 128
MAX_PRIORITY = 10**6

# UTF-8 byte caps, enforced alongside the character caps above. A code-point
# count is not a size: 512 astral characters are 2 KiB, so a string that looks
# modest to Python can still quadruple the artifact and the terminal line it
# produces. Set to twice the character cap, which leaves ordinary accented and
# CJK text (1-3 bytes per character) comfortable room while refusing a payload
# padded out with the widest encodings.
MAX_FIELD_BYTES = 2 * MAX_FIELD_CHARS
MAX_ID_BYTES = 2 * MAX_ID_CHARS

# The exact status enum the roadmap defines for the read model. `archived` is
# deliberately absent: archived tasks are excluded from the read model, so an
# artifact carrying one is a publisher that did not apply the exclusion, not a
# row to render. Membership is exact — no case folding and no trimming, or an
# artifact could smuggle an unrecognised state past a forgiving comparison.
#
# Spelled out here rather than imported from ``kanban_db``: this module must
# stay free of every live-database import (see the module docstring).
VALID_STATUSES = frozenset(
    {"triage", "todo", "scheduled", "ready", "running", "blocked", "review", "done"}
)


# Unicode bidi formatting controls: the marks (ALM/LRM/RLM), the deprecated
# embedding/override block (LRE..RLO), and the isolates (LRI..PDI).
#
# They are printable code points, so the C0/C1 check below never sees them —
# but they reorder every character after them, which is how "Trojan Source"
# text renders as something the publisher never wrote. Like the control
# characters, they are refused rather than stripped: a row that needs them to
# display correctly is not a row this reader was asked to show.
_BIDI_CONTROLS = frozenset(
    "\u061c\u200e\u200f"              # ALM, LRM, RLM
    "\u202a\u202b\u202c\u202d\u202e"  # LRE, RLE, PDF, LRO, RLO
    "\u2066\u2067\u2068\u2069"        # LRI, RLI, FSI, PDI
)


def _sanitized_text(
    value: object,
    field: str,
    max_chars: int,
    max_bytes: int,
    *,
    allow_empty: bool = False,
) -> str:
    """Require a bounded, control-character-free string.

    Control characters are rejected, not escaped or stripped: a title holding
    a newline, a NUL, or an ANSI escape is how a row forges an extra line of
    CLI output or hides its own tail, and a row doing that has nothing worth
    salvaging.
    """
    if not isinstance(value, str):
        raise ReadModelUnavailable(f"task field {field} has the wrong type")
    if not value and not allow_empty:
        raise ReadModelUnavailable(f"task field {field} is empty")
    if len(value) > max_chars:
        raise ReadModelUnavailable(f"task field {field} is too long")
    if len(value.encode("utf-8")) > max_bytes:
        raise ReadModelUnavailable(f"task field {field} is too large")
    if any(ch < " " or ch == "\x7f" or "\x80" <= ch <= "\x9f" for ch in value):
        raise ReadModelUnavailable(f"task field {field} contains control characters")
    if not _BIDI_CONTROLS.isdisjoint(value):
        raise ReadModelUnavailable(f"task field {field} contains bidi controls")
    return value


# Task ids the reader will accept: the safe character set, bounded length.
# Hermes' own ids are `t_` plus hex, so this is deliberately wider than the
# generator needs and still narrow enough to keep path separators, whitespace,
# shell punctuation, and look-alike Unicode out of a value that is used as a
# sort key and printed unquoted. `fullmatch`, so no trailing newline rides in.
_TASK_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,%d}" % MAX_ID_CHARS)


def _validated_task_id(value: object) -> str:
    """Require an id inside the safe grammar."""
    if not isinstance(value, str):
        raise ReadModelUnavailable("task field id has the wrong type")
    if not _TASK_ID_RE.fullmatch(value):
        raise ReadModelUnavailable("task field id is not a safe id")
    return value


def _validated_status(value: object) -> str:
    """Require a status from :data:`VALID_STATUSES`."""
    if not isinstance(value, str):
        raise ReadModelUnavailable("task field status has the wrong type")
    if value not in VALID_STATUSES:
        raise ReadModelUnavailable("task field status is not a known status")
    return value


def _epoch_field(value: object, field: str) -> int:
    """Require a non-negative epoch-seconds int (never a bool)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReadModelUnavailable(f"task field {field} has the wrong type")
    if value < 0:
        raise ReadModelUnavailable(f"task field {field} is negative")
    return value


def _validated_task(row: object) -> "dict[str, object]":
    """Validate one task row and rebuild it from the allowlist."""
    if not isinstance(row, dict):
        raise ReadModelUnavailable("task row is not an object")
    if set(row) != set(TASK_FIELDS):
        # Both directions: an unknown key means the publisher emitted
        # something outside the allowlist, a missing one means the row is not
        # the shape this reader validated.
        raise ReadModelUnavailable("task row does not match the field allowlist")

    priority = row["priority"]
    if isinstance(priority, bool) or not isinstance(priority, int):
        raise ReadModelUnavailable("task field priority has the wrong type")
    if abs(priority) > MAX_PRIORITY:
        raise ReadModelUnavailable("task field priority is out of range")

    assignee = row["assignee"]
    if assignee is not None:
        assignee = _sanitized_text(assignee, "assignee", MAX_ID_CHARS, MAX_ID_BYTES)

    optional_stamps = {}
    for field in ("started_at", "completed_at"):
        value = row[field]
        optional_stamps[field] = None if value is None else _epoch_field(value, field)

    return {
        "id": _validated_task_id(row["id"]),
        "title": _sanitized_text(
            row["title"], "title", MAX_FIELD_CHARS, MAX_FIELD_BYTES, allow_empty=True,
        ),
        "status": _validated_status(row["status"]),
        "assignee": assignee,
        "priority": priority,
        "created_at": _epoch_field(row["created_at"], "created_at"),
        "started_at": optional_stamps["started_at"],
        "completed_at": optional_stamps["completed_at"],
    }


def _typed_field(document: "dict[str, object]", key: str, kind: type) -> object:
    """Fetch ``key`` and require exactly ``kind``.

    ``bool`` is a subclass of ``int`` in Python, so ``isinstance(True, int)``
    is true — an ``int`` field must therefore reject a ``bool`` explicitly, or
    ``"truncated": 1`` and ``"generated_at": true`` both slip through.
    """
    if key not in document:
        raise ReadModelUnavailable(f"artifact is missing {key}")
    value = document[key]
    if kind is int and isinstance(value, bool):
        raise ReadModelUnavailable(f"artifact field {key} is a bool, not an int")
    if not isinstance(value, kind):
        raise ReadModelUnavailable(f"artifact field {key} has the wrong type")
    return value


def validate_artifact(
    document: "dict[str, object]",
    *,
    board: str,
    expected_fingerprint: str,
    now: int,
    max_age_seconds: int,
) -> "dict[str, object]":
    """Validate one parsed artifact and return the allowlisted payload.

    The returned mapping is rebuilt field by field from validated values; no
    part of ``document`` is passed through by reference, so a key this reader
    does not know about cannot reach the output.
    """
    unknown = set(document) - set(TOP_LEVEL_FIELDS)
    if unknown:
        raise ReadModelUnavailable("artifact has unknown top-level keys")

    schema_version = _typed_field(document, "schema_version", str)
    if schema_version != SCHEMA_VERSION:
        raise ReadModelUnavailable("artifact schema version is not supported")

    if max_age_seconds < 0:
        raise ReadModelUnavailable("max age is negative")

    generated_at = _typed_field(document, "generated_at", int)
    if generated_at < 0:
        raise ReadModelUnavailable("artifact generated_at is negative")
    if generated_at > now:
        # No skew tolerance: the publisher stamps the artifact from the same
        # clock this reader uses, so a future stamp is a lie or a forgery
        # reaching for permanent freshness, not jitter.
        raise ReadModelUnavailable("artifact is stamped in the future")
    if now - generated_at > max_age_seconds:
        # Publication is manual, so staleness is the expected steady state.
        # It is reported as unavailable, never as current data.
        raise ReadModelUnavailable("artifact is stale")

    artifact_board = _typed_field(document, "board", str)
    if artifact_board != board:
        # Exact match only. Case folding or trimming here would let an
        # artifact published for one board answer a question about another.
        raise ReadModelUnavailable("artifact is bound to a different board")

    if _typed_field(document, "root_fingerprint", str) != expected_fingerprint:
        raise ReadModelUnavailable("artifact is bound to a different kanban root")

    truncated = _typed_field(document, "truncated", bool)
    tasks = _typed_field(document, "tasks", list)
    if len(tasks) > MAX_TASKS:
        raise ReadModelUnavailable("artifact has more task rows than the cap allows")
    validated_tasks = [_validated_task(row) for row in tasks]
    ids = {task["id"] for task in validated_tasks}
    if len(ids) != len(validated_tasks):
        # `id` is the unique final tie-breaker of the deterministic order, so a
        # repeat makes the order ambiguous and lets one row's text be shown
        # under another row's identity. Fail closed rather than pick a winner.
        raise ReadModelUnavailable("artifact contains duplicate task ids")

    # The reader imposes the order instead of trusting the artifact's own row
    # sequence, and it does so before any limit is applied, so `--limit n`
    # returns the n highest-priority rows rather than the n the publisher
    # happened to write first. `id` is unique (checked just above), so this is
    # a total order: the same artifact always renders identically.
    validated_tasks.sort(key=lambda task: (-task["priority"], task["id"]))

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "board": artifact_board,
        "truncated": truncated,
        "tasks": validated_tasks,
    }


def read_read_model(
    *,
    kanban_root: "os.PathLike[str] | str",
    board: object,
    artifact: "os.PathLike[str] | str",
    max_age_seconds: int,
    limit: object,
    now: "int | None" = None,
) -> "dict[str, object]":
    """Read one owner-published artifact and return the sanitized payload.

    Every input is explicit and required. The order matters: the cheap,
    filesystem-free checks (board slug, limit) run first, then the root and
    artifact path are validated, and only then is a single fd opened — the
    live database is never named, derived, stat-ed, or opened at any point.

    Raises :class:`ReadModelUnavailable` for every failure. Callers surface
    one fixed generic message; the exception text is for developers.
    """
    board = validated_board(board)
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ReadModelUnavailable("limit is not an int")
    if not 1 <= limit <= MAX_TASKS:
        raise ReadModelUnavailable("limit is out of range")

    root = validated_kanban_root(kanban_root)
    artifact_path = validated_artifact_path(artifact, root)

    payload = read_artifact_bytes(artifact_path)
    document = parse_artifact_json(payload)
    validated = validate_artifact(
        document,
        board=board,
        expected_fingerprint=root_fingerprint(root),
        now=int(time.time()) if now is None else now,
        max_age_seconds=max_age_seconds,
    )

    tasks = validated["tasks"]
    if len(tasks) > limit:
        # Report the trim honestly rather than handing back a short list that
        # looks like the whole board.
        validated["tasks"] = tasks[:limit]
        validated["truncated"] = True
    return validated
