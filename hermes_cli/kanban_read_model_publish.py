"""Owner-side publisher for the Kanban read-model JSON artifact (H0.2b3b).

This module is the mutation-capable counterpart of
:mod:`hermes_cli.kanban_read_model`. The reader never opens the live database;
the publisher deliberately does, with normal Hermes WAL/initialization
semantics, which is exactly why it runs only by explicit owner action.

The first thing it establishes is therefore *authority*, not data: a
``delegate_task`` child is refused before the database is opened and before any
file is written. Both side effects are mutations from H0's point of view — a DB
open touches WAL/SHM, and a write can plant or truncate an artifact a reader
trusts.

Every input is explicit. There is no ambient ``HERMES_HOME``, current-board, or
default-output fallback anywhere on this path: the root, the board, the live DB
path, and the output path are all named by the caller, and the artifact is bound
to the root and board it was actually read for.

Scope note — this slice implements the authorized owner happy path
(allowlisted query, the shared sanitization/bounds contract), the source-board
binding (``--db`` must BE the location the explicit root and board derive, and
the walk to it is fd-pinned and ``O_NOFOLLOW`` throughout, so neither a sibling
board, a symlinked component, nor a final component swapped after validation
can put another database behind the artifact's stamps), the
output-location policy (Kanban-root/board/workspace/attachment containment,
the shared temp root, symlinked components and targets, and group- or
world-writable destinations, all decided before the database is opened), the
existing-target rule (an absent target may be created, but an existing one may
be replaced only when it IS this board's own prior artifact, so a mistyped
``--out`` onto somebody's private file costs that file nothing), the
publication transaction (a failure at any step leaves the previous artifact
byte-identical and clears every intermediate), and the parent-swap rule: the
approved destination is pinned on a directory fd and never re-derived from
the pathname, so a directory moved aside and replaced by a symlink after
validation cannot redirect publication. The remaining symlink-race cases and
the concurrent-reader suite are later tracers of H0.2b3b and are NOT yet
enforced here.

See ``.hermes/plans/2026-09-14_175055-h0-2b3-owner-published-json-read-model``
work unit H0.2b3b.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import stat
import time
from pathlib import Path

from hermes_cli import kanban_read_model as krm


DELEGATED_CHILD_DENIAL = (
    "delegate_task child contexts cannot publish the Kanban read model"
)

# Deliberately says nothing about what failed. The caller controls no input
# that reaches this decision, so there is nothing here it needs explained —
# and the underlying exception text can name modules and paths.
AUTHORITY_UNDETERMINED_DENIAL = (
    "owner publish authority could not be established"
)


class PublishDenied(Exception):
    """The caller lacks owner authority to publish. Never a partial write."""


class PublishUnavailable(Exception):
    """The artifact could not be produced. Never a partial write.

    Like :class:`~hermes_cli.kanban_read_model.ReadModelUnavailable`, the
    message is for developers; the CLI collapses it into one generic string.
    """


def _is_delegated_child_context() -> bool:
    """The same narrow signal ``_is_delegated_child_cli_mutation`` consults.

    Re-asserted here rather than only in the CLI for the reason the Kanban DB
    layer already documents: the CLI check is a fast-fail for UX, but a child
    can import this module directly, so the durable boundary has to sit at the
    publisher itself.

    Three outcomes, not two. The canonical helper is the authority; the
    environment marker substitutes for it ONLY when there is no helper to ask,
    because the marker alone cannot distinguish "owner" from "child that lost
    its marker". So an available helper that fails — raising, or handing back
    something that is not a verdict — leaves authority *undetermined*, and
    undetermined authority is refused here rather than silently downgraded to
    the marker. Answering that case from ``os.environ`` is what would grant a
    delegated child the publication right the marker exists to withhold.
    """
    try:
        from agent.delegation_context import is_delegated_child_process_context
    except ImportError:
        # The one fallback: this install has no canonical helper at all (a
        # trimmed deployment, an import cycle), so the cross-process marker is
        # the entire signal that exists.
        return bool(os.environ.get("HERMES_DELEGATED_CHILD_CONTEXT"))

    try:
        verdict = is_delegated_child_process_context()
    except Exception as exc:  # noqa: BLE001 - any failure is "no verdict"
        raise PublishDenied(AUTHORITY_UNDETERMINED_DENIAL) from exc
    # The helper is contractually a bool. Anything else — a Mock, a shadowed
    # stub, a None from a half-initialised module — is not a verdict, and
    # truth-testing it would invent one.
    if not isinstance(verdict, bool):
        raise PublishDenied(AUTHORITY_UNDETERMINED_DENIAL)
    return verdict


def assert_owner_publish_authority() -> None:
    """Raise :class:`PublishDenied` unless this is an owner-side invocation.

    Call this before opening the database and before creating any file — it is
    the only ordering guarantee the H0 nonmutation invariant rests on here.
    """
    if _is_delegated_child_context():
        raise PublishDenied(DELEGATED_CHILD_DENIAL)


# ---------------------------------------------------------------------------
# Explicit input validation
# ---------------------------------------------------------------------------

def _validated_limit(limit: object) -> int:
    """Require ``1 <= limit <= MAX_TASKS`` as a real int.

    The upper bound is the reader's row cap: publishing more rows than the
    reader accepts produces an artifact that is unreadable by construction.
    """
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise PublishUnavailable("limit is not an int")
    if not 1 <= limit <= krm.MAX_TASKS:
        raise PublishUnavailable("limit is out of range")
    return limit


# ---------------------------------------------------------------------------
# Where the artifact may land
# ---------------------------------------------------------------------------

# Flags for every directory open below, and for the publish step further down.
# Read-only, must be a directory, must NOT be a symlink, never inherited by a
# child. Read via getattr because none of them exist on Windows, where naming
# them directly would break importing the whole Kanban CLI.
_DIR_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)

# The walk needs openat plus O_NOFOLLOW, exactly like the reader's. Without
# them there is no way to check a component without following it, so the
# publisher refuses rather than falling back to a path-resolving check.
# `os.lstat` is deliberately NOT named here: it is absent from
# `supports_dir_fd` even on platforms that support the call, so the final
# component is stat-ed as `os.stat(..., follow_symlinks=False)` instead.
_HAS_DIR_FD_WALK = (
    hasattr(os, "O_NOFOLLOW")
    and os.open in getattr(os, "supports_dir_fd", set())
    and os.stat in getattr(os, "supports_dir_fd", set())
    and os.stat in getattr(os, "supports_follow_symlinks", set())
)

# The shared system temp root, in both spellings macOS has for the one
# directory: `/tmp` is a symlink to `private/tmp`, so a policy that knows only
# the `/tmp` spelling is bypassed by naming the target directly.
#
# Refused as a whole SUBTREE, not merely as a destination directory. A private
# 0700 subdirectory (`mkdtemp`, and pytest's own `tmp_path`) closes the
# name-squatting hole and nothing else: the artifact still sits on a tree the
# host, the packaging, or a reboot may clear on its own schedule, and a reader
# that trusts a published artifact must not depend on that schedule. The
# PlanSpec therefore puts the subtree out of bounds outright rather than
# letting each destination under it argue its own case.
#
# Matched lexically, on both spellings, for the same reason the reader checks
# Kanban-root containment lexically: resolving the path would mean following
# exactly the symlinks this module refuses to follow, and would re-open the
# check to a swap between the answer and the publish. A path that reaches the
# temp subtree *through* a symlink is already refused by the O_NOFOLLOW walk
# below.
_SHARED_TEMP_ROOTS = (Path("/tmp"), Path("/private/tmp"))

# Mode bits that make a directory writable by someone other than its owner.
# Either one lets a third party plant the artifact's name before publication
# or replace the file after it.
_UNSAFE_DIR_MODE = stat.S_IWGRP | stat.S_IWOTH


def _assert_owner_only_writable(dir_fd: int, out_path: Path) -> None:
    """Require the directory *dir_fd* names to be writable only by its owner.

    Checked on the fd, never on the path: this is the inode the publish step
    walks into, whatever the name resolves to a moment later.
    """
    try:
        st = os.fstat(dir_fd)
    except OSError as exc:
        raise PublishUnavailable(f"output directory is not statable: {out_path}") from exc
    if stat.S_IMODE(st.st_mode) & _UNSAFE_DIR_MODE:
        raise PublishUnavailable(
            f"output directory is group- or world-writable: {out_path}"
        )


def _validated_output_location(out_path: Path) -> "tuple[int, tuple[str, ...]]":
    """Refuse an unsafe destination, and PIN the one it settles on.

    Containment inside the Kanban root — which also covers the board,
    workspace, and attachment subtrees, since they all live under it — is
    already settled by :func:`~hermes_cli.kanban_read_model.validated_artifact_path`.
    What is left is the destination itself: nowhere in the shared temp
    subtree, no symlinked component, no symlinked target, and a directory only
    its owner can write.

    The walk opens one component at a time relative to the previous
    component's fd (``openat``) with ``O_NOFOLLOW``. A symlinked component
    therefore fails in the kernel rather than being resolved here, and nothing
    that happens to the path afterwards can redirect a step that is already
    pinned to an inode.

    Returns ``(dir_fd, missing)``: an fd for the deepest directory on the path
    that exists, plus the directory components still to be created below it.
    The fd is the whole point of the return value. What this function proves
    is a fact about *inodes*, and re-deriving the directory from the pathname
    afterwards throws that proof away: ``O_NOFOLLOW`` constrains only the last
    component, so an attacker who can create a name in any ancestor can move
    the validated directory aside, leave a symlink to their own directory
    under its former name, and have the publication follow it. Every step of
    the transaction therefore runs relative to this fd, and the caller owns
    it — it is closed exactly once, on every outcome.
    """
    if any(
        out_path == temp_root or temp_root in out_path.parents
        for temp_root in _SHARED_TEMP_ROOTS
    ):
        raise PublishUnavailable(f"output path is in a shared temp root: {out_path}")

    if not _HAS_DIR_FD_WALK:
        raise PublishUnavailable("platform lacks openat/O_NOFOLLOW")

    try:
        dir_fd = os.open(out_path.anchor, _DIR_FLAGS)
    except OSError as exc:
        raise PublishUnavailable(f"output path anchor is unusable: {out_path}") from exc

    try:
        *parents, name = out_path.parts[1:]
        for index, part in enumerate(parents):
            try:
                next_fd = os.open(part, _DIR_FLAGS, dir_fd=dir_fd)
            except FileNotFoundError:
                # Nothing below this point exists yet; the publisher creates
                # it at 0700 itself, relative to this fd. The deepest
                # directory that DOES exist is the one that decides whether
                # that is safe, so it is checked here rather than skipped.
                _assert_owner_only_writable(dir_fd, out_path)
                return dir_fd, tuple(parents[index:])
            except OSError as exc:
                # ELOOP for a symlinked component (O_NOFOLLOW), ENOTDIR for a
                # file in the middle, EACCES for an unreadable one. Every one
                # of them fails closed, and none of them followed anything.
                raise PublishUnavailable(
                    f"output path component is unusable: {part}"
                ) from exc
            os.close(dir_fd)
            dir_fd = next_fd

        _assert_owner_only_writable(dir_fd, out_path)

        try:
            st = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        except FileNotFoundError:
            return dir_fd, ()
        except OSError as exc:
            raise PublishUnavailable(f"output path is unusable: {out_path}") from exc
        if stat.S_ISLNK(st.st_mode):
            # `rename` would replace the link rather than follow it, but a
            # caller who named a symlink did not name the file they would get,
            # and the next publication is not the place to discover that.
            raise PublishUnavailable(f"output path is a symlink: {out_path}")
        return dir_fd, ()
    except BaseException:
        # Only on the way out through a failure: on success the fd belongs to
        # the caller, which is what makes "closed exactly once" true on both
        # paths.
        os.close(dir_fd)
        raise


# Flags for an EXISTING target. Read-only, must NOT be a symlink, never
# inherited by a child, and O_NONBLOCK so a FIFO planted under the artifact's
# name fails immediately instead of blocking the publisher forever waiting for
# a writer to appear.
_TARGET_FILE_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NONBLOCK", 0)
)

# Every way an existing target can fail to be this board's own artifact, said
# in one voice. Deliberately shapeless: `--out` is the caller's own pathname,
# but what currently answers to it is not, and a distinguishable reason would
# turn publication into a read oracle for a file the caller is being refused
# access to. The CLI collapses all of it to `unavailable` anyway.
_UNREPUBLISHABLE_TARGET = "output path is not this board's read-model artifact"


def _assert_existing_target_is_republishable(
    dir_fd: int, out_path: Path, root: Path, board: str
) -> None:
    """Require an existing *out_path* to BE this board's own prior artifact.

    Publication ends in a ``rename`` over whatever currently holds the target
    name, and that is atomic and irreversible. An absent target is therefore
    the ordinary case and is allowed outright; an existing one has to earn the
    replacement, because ``--out`` is a caller-supplied pathname and a typo
    onto a private key, a password store, or a half-finished document must
    cost that file nothing.

    The predicate is the narrowest one that still lets the publisher do its
    job — the target must be indistinguishable from this command's own
    previous output:

    * opened ``openat``/``O_NOFOLLOW`` relative to the ALREADY PINNED
      destination fd, so the file inspected is the file the transaction will
      later rename over, whatever the pathname means in between;
    * a regular file with exactly one link, owned by this euid, at exactly
      0600, no larger than the artifact cap — the reader's own admission
      rules, asked here so the publisher cannot destroy something the reader
      would have refused to read;
    * strict UTF-8, strict JSON, schema v1, and the reader's full structural
      and value bounds; and
    * stamped with THIS explicit root's fingerprint and THIS explicit board,
      so another root's or another board's published view is somebody else's
      artifact and not a previous run of this one.

    Freshness is deliberately absent from that list. Staleness is the READER's
    rule, and an artifact nobody refreshed is exactly the one this command
    exists to replace — a publisher that required its own previous output to
    be current could never refresh anything. The age check is neutralised by
    measuring the document against its own stamp rather than the clock, which
    also keeps this decision free of wall-clock time.

    The reader's public parse/validate entry points do the structural work, so
    there is one definition of "an artifact" and not two that can drift. What
    is NOT reused is :func:`~hermes_cli.kanban_read_model.read_artifact_bytes`:
    it walks a pathname, and the whole point here is to stay on the fd the
    output-location check already pinned.
    """
    try:
        fd = os.open(out_path.name, _TARGET_FILE_FLAGS, dir_fd=dir_fd)
    except FileNotFoundError:
        # Nothing to overwrite. The first publication into this directory —
        # and the only case that needs no proof at all.
        return
    except OSError as exc:
        # ELOOP for a symlink (O_NOFOLLOW), ENXIO for a FIFO nobody is writing
        # (O_NONBLOCK), EACCES for a file this user may not read. Every one of
        # them fails closed, and none of them followed or blocked on anything.
        raise PublishUnavailable(_UNREPUBLISHABLE_TARGET) from exc

    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise PublishUnavailable(_UNREPUBLISHABLE_TARGET)
        if st.st_nlink != 1:
            # A second link is a second name for the same inode, outside
            # whatever protection this directory provides. Replacing it would
            # be replacing a file reachable somewhere this check cannot see.
            raise PublishUnavailable(_UNREPUBLISHABLE_TARGET)
        if st.st_uid != os.getuid():
            raise PublishUnavailable(_UNREPUBLISHABLE_TARGET)
        if stat.S_IMODE(st.st_mode) != krm.ARTIFACT_MODE:
            # Exact, not a subset: 0600 is the only mode this publisher ever
            # produces, so anything else was not produced by it.
            raise PublishUnavailable(_UNREPUBLISHABLE_TARGET)
        if st.st_size > krm.MAX_ARTIFACT_BYTES:
            raise PublishUnavailable(_UNREPUBLISHABLE_TARGET)

        chunks: "list[bytes]" = []
        # One byte past the cap, so a file that grew since it was measured is
        # detected rather than read as a valid-looking prefix of itself.
        remaining = krm.MAX_ARTIFACT_BYTES + 1
        try:
            while remaining > 0:
                chunk = os.read(fd, remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
        except OSError as exc:
            raise PublishUnavailable(_UNREPUBLISHABLE_TARGET) from exc
    finally:
        os.close(fd)

    payload = b"".join(chunks)
    if len(payload) > krm.MAX_ARTIFACT_BYTES or len(payload) != st.st_size:
        raise PublishUnavailable(_UNREPUBLISHABLE_TARGET)

    try:
        document = krm.parse_artifact_json(payload)
        stamp = document.get("generated_at")
        if isinstance(stamp, bool) or not isinstance(stamp, int):
            raise PublishUnavailable(_UNREPUBLISHABLE_TARGET)
        # `now=stamp` with a zero window: the document is measured against its
        # own stamp, so neither staleness nor a future stamp can decide this,
        # and every other rule the reader applies still does.
        krm.validate_artifact(
            document,
            board=board,
            expected_fingerprint=krm.root_fingerprint(root),
            now=stamp,
            max_age_seconds=0,
        )
    except krm.ReadModelUnavailable as exc:
        raise PublishUnavailable(_UNREPUBLISHABLE_TARGET) from exc


# ---------------------------------------------------------------------------
# Which database the artifact may be built from
# ---------------------------------------------------------------------------

# The one board whose database stayed at the pre-boards location. Spelled out
# here rather than imported from `kanban_db`, for the same reason this module
# keeps its own copy of the reader's sanitization rules: importing the live DB
# layer would pull the whole mutation-capable Kanban stack into the read-model
# feature, and this module needs the LAYOUT, not the resolution chain.
_DEFAULT_BOARD = "default"

# Flags for the final component. Read-only, must NOT be a symlink, never
# inherited by a child, and O_NONBLOCK so a planted FIFO fails instead of
# blocking the publisher forever waiting for a writer.
_DB_FILE_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NONBLOCK", 0)
)

# `sqlite3.connect` on a bare pathname re-resolves the whole path in C, after
# every check here has run, and CREATES whatever it does not find — so a board
# database that vanished a moment ago comes back as an empty one, written
# inside the Kanban root. The URI form is the only way to say "open exactly
# this, and never make it": `mode=rw`. URI filenames need SQLite 3.7.7, and
# below that this module has no safe way to open a board at all, so it refuses
# rather than falling back to the pathname.
#
# `nofollow=1` is deliberately NOT used. It is a flag of the C
# `sqlite3_open_v2` API, not a recognised URI query parameter, so SQLite
# ignores it silently — verified against the local 3.53.3 build, which opened a
# symlinked database with it set. Relying on it would be a check that is not
# there. The final component is pinned here instead: opened with `O_NOFOLLOW`
# relative to the board directory's fd before the connect, and re-verified
# against that same inode immediately after it, before a single statement runs.
_HAS_URI_DB_OPEN = sqlite3.sqlite_version_info >= (3, 7, 7)


def _canonical_board_db_path(root: Path, board: str) -> Path:
    """The ONE location *board* may keep its live database under *root*.

    Mirrors the layout :func:`hermes_cli.kanban_db.kanban_db_path` lays down,
    minus every ambient input that function honours — no ``HERMES_KANBAN_DB``,
    no ``<root>/kanban/current``, no ``expanduser``. The answer is a pure
    function of the two words the caller spelled out.
    """
    if board == _DEFAULT_BOARD:
        return root / "kanban.db"
    return root / "kanban" / "boards" / board / "kanban.db"


def _validated_db_path(
    db: "os.PathLike[str] | str", root: Path, board: str
) -> Path:
    """Require the explicit ``--db`` to BE this board's database.

    This is the ONE place in the whole read-model feature allowed to name
    ``kanban.db``. Containment inside the root is not enough: the artifact
    stamps ``board`` and ``root_fingerprint(root)``, and a reader trusts both,
    so a sibling board's perfectly valid database under the same root would
    make the pair a lie just as surely as a database from another machine
    would. The caller's spelling therefore has to match the derived location
    exactly — ``--db`` names the database, it does not choose it.
    """
    path = Path(os.fspath(db))
    if not path.is_absolute():
        raise PublishUnavailable(f"db path is not absolute: {path}")
    if ".." in path.parts:
        raise PublishUnavailable(f"db path is not a canonical spelling: {path}")
    expected = _canonical_board_db_path(root, board)
    if path != expected:
        raise PublishUnavailable(
            f"db path is not board {board!r}'s database: {path}"
        )
    return path


def _assert_db_component_protected(st: os.stat_result, label: str) -> None:
    """Require a component of the board chain to be the owner's alone.

    Both halves matter and neither implies the other. A component owned by
    somebody else is somebody else's to redefine; a component the caller owns
    but the group or the world can write is one any of them can rename aside
    and replace. Either way the database found at the end of the walk is not
    evidence of anything, so the walk stops here rather than reporting a board
    it cannot vouch for.
    """
    if st.st_uid != os.getuid():
        raise PublishUnavailable(f"{label} is not owned by the current user")
    if stat.S_IMODE(st.st_mode) & _UNSAFE_DIR_MODE:
        raise PublishUnavailable(f"{label} is group- or world-writable")


def _assert_component_is_not_substitutable(
    parent_st: os.stat_result, child_st: os.stat_result, label: str
) -> None:
    """Require that no OTHER local user can rename *label* out of its parent.

    Whether a name can be made to mean something else is a property of the
    directory that HOLDS the name, not of the thing named — so the question
    is about the pair, and it has to be asked about every pair on the chain,
    including the ones above the Kanban root. ``sqlite3.connect`` is handed a
    pathname and re-resolves it from ``/`` down in C, long after this walk is
    over; an ancestor a third party may write is therefore enough to decide
    which database SQLite binds its descriptor to, no matter how carefully
    the components below it were pinned.

    Two bits settle it:

    * **Owner.** A directory owned by another unprivileged user is theirs to
      rewrite wholesale. ``root`` is accepted alongside the caller because
      every real installation hangs off root-owned ancestors (``/``,
      ``/Users``, ``/private/var``) and a hostile root already owns this
      process — there is nothing to defend there.
    * **Write access, and the sticky exception.** A group- or world-writable
      directory lets a third party rename the entry away and plant their own
      under the vacated name. The one kernel-enforced exception is the sticky
      bit: in a sticky directory, removing or renaming an entry additionally
      requires owning the entry or the directory, so a ``/tmp``-shaped
      ancestor is safe exactly while the entry it holds belongs to this user
      or to root.

    Deliberately silent about a process running as THIS user: same-UID
    replacement is outside the documented threat model, and no permission bit
    can distinguish the owner from the owner.
    """
    uid = os.getuid()
    # A parent nobody but this user or root controls can still be sitting on
    # an inode the caller owns — ownership of the *parent* is what decides
    # who may re-point the name, so it is checked on the parent alone.
    if parent_st.st_uid not in (uid, 0):
        raise PublishUnavailable(f"{label} is held by another user's directory")
    parent_mode = stat.S_IMODE(parent_st.st_mode)
    if parent_mode & _UNSAFE_DIR_MODE:
        if not parent_mode & stat.S_ISVTX:
            raise PublishUnavailable(
                f"{label} is held by a group- or world-writable directory"
            )
        if child_st.st_uid not in (uid, 0):
            raise PublishUnavailable(
                f"{label} is renamable by its owner in a sticky directory"
            )


def _open_board_db_parent(root: Path, db_path: Path) -> int:
    """Walk to the board directory with ``openat``/``O_NOFOLLOW`` and PIN it.

    The lexical check above proves the caller spelled the canonical location.
    It cannot prove the location IS canonical: a symlinked board directory
    produces a path that is character-for-character identical to the honest
    one and resolves somewhere else entirely. Only opening each component
    without following it can tell the two apart, so that is what happens here
    — one component at a time, relative to the previous component's fd.

    Pinning each component is necessary and it is not sufficient, because the
    connect below does not use these descriptors: SQLite re-resolves the whole
    pathname from ``/`` in C. So every component gets a second check, from the
    filesystem root down — :func:`_assert_component_is_not_substitutable`,
    asked of each (parent, child) pair — establishing that no other local user
    can rename the next component out from under that pathname. Without it,
    one third-party-writable ancestor anywhere above the root is enough: the
    root is renamed aside, an exact-looking tree takes its place for the
    duration of the connect, and the honest tree is back before the
    post-connect identity check stats the pinned board fd and agrees.

    Every component at or below the Kanban root is required to be the owner's
    alone on top of that — the same rule, narrowed: at and below the root
    there is no system-owned component to make room for, so root ownership and
    the sticky exception both stop being reasons to accept anything.

    The returned fd is the board directory itself, and it is the caller's to
    close exactly once. It stays open across ``sqlite3.connect`` because it is
    what the post-connect identity check is asked on; the ancestors' fds are
    not held, because holding an fd does not hold a NAME on POSIX — what makes
    the pathname trustworthy across the connect is the ownership proof above,
    not a descriptor.
    """
    if not _HAS_DIR_FD_WALK:
        raise PublishUnavailable("platform lacks openat/O_NOFOLLOW")

    parent = db_path.parent
    try:
        dir_fd = os.open(parent.anchor, _DIR_FLAGS)
    except OSError as exc:
        raise PublishUnavailable(f"db path anchor is unusable: {db_path}") from exc
    try:
        current = Path(parent.anchor)
        # The anchor is the first parent in the chain; it is stat-ed here so
        # the loop below always has the level ABOVE the component it opens.
        parent_st = os.fstat(dir_fd)
        for part in parent.parts[1:]:
            try:
                next_fd = os.open(part, _DIR_FLAGS, dir_fd=dir_fd)
            except OSError as exc:
                # ELOOP for a symlinked component (O_NOFOLLOW), ENOTDIR for a
                # file in the middle, ENOENT for a board that does not exist.
                # Every one of them fails closed, and none followed anything.
                raise PublishUnavailable(
                    f"db path component is unusable: {current / part}"
                ) from exc
            os.close(dir_fd)
            dir_fd = next_fd
            current = current / part
            child_st = os.fstat(dir_fd)
            _assert_component_is_not_substitutable(
                parent_st, child_st, f"db path component {current}"
            )
            if current == root or root in current.parents:
                _assert_db_component_protected(
                    child_st, f"db path component {current}"
                )
            parent_st = child_st
        return dir_fd
    except BaseException:
        os.close(dir_fd)
        raise


def _pinned_board_db(dir_fd: int, db_path: Path) -> "tuple[int, int]":
    """Open the database itself without following it, and return its identity.

    ``O_NOFOLLOW`` on the final component is what a plain ``lstat`` cannot be:
    the file this reports on is the one that was opened, not the one the name
    happened to mean a moment earlier. The fd is closed again immediately — it
    exists to produce the ``(dev, ino)`` pair the connect below is checked
    against, because SQLite will open the file by name and there is no way to
    hand it this descriptor.
    """
    try:
        fd = os.open(db_path.name, _DB_FILE_FLAGS, dir_fd=dir_fd)
    except OSError as exc:
        raise PublishUnavailable(f"db path is unusable: {db_path}") from exc
    try:
        st = os.fstat(fd)
    finally:
        os.close(fd)
    if not stat.S_ISREG(st.st_mode):
        raise PublishUnavailable(f"db path is not a regular file: {db_path}")
    _assert_db_component_protected(st, f"db file {db_path}")
    return st.st_dev, st.st_ino


def _assert_board_db_is_still_pinned(
    dir_fd: int, db_path: Path, pinned: "tuple[int, int]"
) -> None:
    """Refuse if the board's name no longer means the inode that was pinned.

    Stat-ed on the pinned board-directory fd, never by pathname, and never
    following a link, so the question asked is exactly "is the entry in THIS
    directory still that file". Called after ``sqlite3.connect`` returns and
    before any statement runs: SQLite opens the main database lazily, so a
    swapped final component is caught while the connection is still a bare
    descriptor — nothing has been read from it and nothing written to it.
    """
    try:
        st = os.stat(db_path.name, dir_fd=dir_fd, follow_symlinks=False)
    except OSError as exc:
        raise PublishUnavailable(
            f"db path changed during publication: {db_path}"
        ) from exc
    if stat.S_ISLNK(st.st_mode) or (st.st_dev, st.st_ino) != pinned:
        raise PublishUnavailable(f"db path changed during publication: {db_path}")


# ---------------------------------------------------------------------------
# The allowlisted query
# ---------------------------------------------------------------------------

# The eight columns the artifact may carry, spelled out rather than derived.
# No `SELECT *`, no `_task_to_dict`, no `recompute_ready`: a task body,
# workspace path, session id, or claim lock must never enter this process, let
# alone the artifact — not being fetched is a stronger guarantee than being
# filtered out afterwards.
_SELECTED_COLUMNS = (
    "id",
    "title",
    "status",
    "assignee",
    "priority",
    "created_at",
    "started_at",
    "completed_at",
)

# Statuses eligible for the read model. `archived` is deliberately absent: the
# reader refuses it, so an archived row is excluded at the source.
_ELIGIBLE_STATUSES = tuple(sorted(krm.VALID_STATUSES))

# Ordering is part of the contract, not a convenience: `priority DESC, id ASC`
# is a total order (id is the primary key), so the same board always yields the
# same top-N and `--limit` means "the N most important", not "the N SQLite
# happened to reach first".
_TASK_SELECT_SQL = (
    "SELECT " + ", ".join(_SELECTED_COLUMNS) + " FROM tasks "
    "WHERE status IN (" + ", ".join("?" * len(_ELIGIBLE_STATUSES)) + ") "
    "ORDER BY priority DESC, id ASC LIMIT ?"
)

# How long to wait for a writer's lock before giving up. The read is a single
# short statement, so this only ever absorbs an in-flight commit.
_DB_TIMEOUT_SECONDS = 5.0


# ---------------------------------------------------------------------------
# The schema contract the query is a projection of
# ---------------------------------------------------------------------------

# `FROM tasks` names an object; it does not name a schema. SQLite resolves that
# name to whatever is there — a view, or a table whose columns are spelled the
# same and mean something else — and an EMPTY one of either returns zero rows,
# sanitizes cleanly, and publishes a well-formed artifact saying this board has
# no tasks. `_sanitized_task` cannot see that: it only ever inspects rows that
# came back. So the source is checked as a schema, before it is read as data,
# and "the board is empty" stays distinguishable from "the board is not the
# thing this query describes".
#
# Transcribed by hand from `kanban_db.SCHEMA_SQL`, and only for the eight
# allowlisted columns: what is pinned here is the contract the artifact's
# MEANING depends on, not a mirror of the whole table. Extra columns are
# expected and ignored — a real board carries some thirty of them, which is
# exactly why `_SELECTED_COLUMNS` exists — but every column named here must be
# present, must carry the affinity its value is read as, and the three the
# contract requires must be NOT NULL.
_REQUIRED_TASK_COLUMNS = {
    # lowercase name: (type affinity, must be declared NOT NULL)
    "id": ("TEXT", False),
    "title": ("TEXT", True),
    "status": ("TEXT", True),
    "assignee": ("TEXT", False),
    "priority": ("INTEGER", False),
    "created_at": ("INTEGER", True),
    "started_at": ("INTEGER", False),
    "completed_at": ("INTEGER", False),
}

# `id` carries no NOT NULL of its own, and does not need one: it is the primary
# key. That is also what makes `priority DESC, id ASC` a TOTAL order, and
# therefore what makes `--limit` mean "the N most important" rather than "N of
# them". A `tasks` whose `id` is not the primary key breaks the ordering
# contract, not merely a type check.
_TASKS_PRIMARY_KEY = ("id",)


def _declared_affinity(declared: object) -> str:
    """The SQLite type affinity of a declared column type.

    SQLite's own five rules in SQLite's own order (datatype3 §3.1), so
    `INTEGER`, `INT` and `BIGINT` are one thing and `TEXT`, `VARCHAR(40)` and
    `CLOB` are another. The contract is about what a column MEANS, not how the
    board's author spelled it: a respelling that stores the same values is not
    an incompatibility, and a `created_at` declared TEXT is.
    """
    if not isinstance(declared, str):
        return ""
    name = declared.upper()
    if "INT" in name:
        return "INTEGER"
    if "CHAR" in name or "CLOB" in name or "TEXT" in name:
        return "TEXT"
    if "BLOB" in name or not name:
        return "BLOB"
    if "REAL" in name or "FLOA" in name or "DOUB" in name:
        return "REAL"
    return "NUMERIC"


def _assert_tasks_matches_the_schema_contract(conn: sqlite3.Connection) -> None:
    """Refuse unless `tasks` is the real, compatible table the query projects.

    Runs on the same connection and inside the same read transaction as the
    SELECT, and before it: the schema this approves is therefore the schema the
    query actually runs against, and a refusal costs one `sqlite_master` row
    and one PRAGMA rather than an artifact.

    Every refusal is deliberately shapeless — which object, column or property
    disagreed is a description of the caller's own database, and the CLI
    collapses all of it to one generic `unavailable` anyway.
    """
    objects = conn.execute(
        "SELECT type FROM main.sqlite_master WHERE name = ? COLLATE NOCASE",
        ("tasks",),
    ).fetchall()
    if [row[0] for row in objects] != ["table"]:
        # A view, a virtual table, or nothing at all. It has to be the table
        # itself: a view has no primary key, no NOT NULL and no declared types
        # of its own, and what it selects from can be redefined without the
        # name this query names ever changing.
        raise PublishUnavailable("board tasks is not a compatible table")

    # Keyed lowercase because SQLite identifiers are case-insensitive, and two
    # columns cannot differ only in case.
    columns = {
        str(name).lower(): (_declared_affinity(declared), bool(notnull), int(pk))
        for _cid, name, declared, notnull, _default, pk in conn.execute(
            "PRAGMA main.table_info('tasks')"
        ).fetchall()
    }

    for name, (affinity, required_not_null) in _REQUIRED_TASK_COLUMNS.items():
        found = columns.get(name)
        if found is None:
            raise PublishUnavailable("board tasks is missing a required column")
        found_affinity, found_not_null, _pk = found
        if found_affinity != affinity:
            raise PublishUnavailable("board tasks column has the wrong type")
        if required_not_null and not found_not_null:
            # A board that may hold a NULL title/status/created_at is a board
            # whose rows the eligibility filter drops instead of refusing.
            raise PublishUnavailable("board tasks column is not required")

    primary_key = tuple(
        name
        for _rank, name in sorted(
            (pk, name) for name, (_a, _n, pk) in columns.items() if pk
        )
    )
    if primary_key != _TASKS_PRIMARY_KEY:
        raise PublishUnavailable("board tasks has the wrong primary key")


def _fetch_task_rows(
    db_path: Path, limit: int, *, dir_fd: int, pinned: "tuple[int, int]"
) -> "tuple[list[dict], bool]":
    """Return ``(rows, truncated)`` for one board.

    ``limit + 1`` rows are requested so truncation is *observed* rather than
    guessed: if the extra row comes back, more eligible tasks exist than were
    published, and ``truncated`` says so truthfully.

    *dir_fd* and *pinned* come straight from :func:`_open_board_db_parent` and
    :func:`_pinned_board_db`. They are the only reason the pathname handed to
    SQLite here is trustworthy: it is checked back against the pinned inode
    after the open and before the first read.
    """
    if not _HAS_URI_DB_OPEN:
        raise PublishUnavailable("platform lacks sqlite uri filenames")
    try:
        # `mode=rw`: open exactly this database, and never create one.
        conn = sqlite3.connect(
            f"{db_path.as_uri()}?mode=rw",
            uri=True,
            timeout=_DB_TIMEOUT_SECONDS,
            isolation_level=None,
        )
    except sqlite3.Error as exc:
        raise PublishUnavailable(f"board database is unusable: {exc}") from exc
    try:
        _assert_board_db_is_still_pinned(dir_fd, db_path, pinned)
        try:
            # One short, consistent read transaction. A single statement would
            # be atomic anyway; the explicit BEGIN documents that the snapshot
            # — and therefore the `truncated` flag derived from it — describes
            # one instant of the board, not two.
            conn.execute("BEGIN DEFERRED")
            try:
                # Inside the snapshot and ahead of the read: the source is
                # checked as a schema before it is read as data, so an empty
                # board and a board that is not this table cannot produce the
                # same artifact.
                _assert_tasks_matches_the_schema_contract(conn)
                fetched = conn.execute(
                    _TASK_SELECT_SQL, (*_ELIGIBLE_STATUSES, limit + 1)
                ).fetchall()
            finally:
                conn.execute("COMMIT")
        except sqlite3.Error as exc:
            raise PublishUnavailable(f"board query failed: {exc}") from exc
    finally:
        conn.close()

    truncated = len(fetched) > limit
    rows = [dict(zip(_SELECTED_COLUMNS, row)) for row in fetched[:limit]]
    return rows, truncated


# ---------------------------------------------------------------------------
# Sanitization and bounds
# ---------------------------------------------------------------------------

# Deliberately a second, independent copy of the reader's rules rather than a
# reach into its private helpers. The codebase already applies this pattern to
# read-model output (`_READ_MODEL_OUTPUT_FIELDS`): two independent allowlists
# mean one mistake on either side cannot put a body or a session id on screen.
# If the two ever diverge, the reader refuses the artifact — loudly, at the
# boundary, rather than by quietly publishing something it will not accept.
_BIDI_CONTROLS = frozenset(
    "؜‎‏"              # ALM, LRM, RLM
    "‪‫‬‭‮"  # LRE, RLE, PDF, LRO, RLO
    "⁦⁧⁨⁩"        # LRI, RLI, FSI, PDI
)

_TASK_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,%d}" % krm.MAX_ID_CHARS)


def _safe_text(
    value: object,
    field: str,
    max_chars: int,
    max_bytes: int,
    *,
    allow_empty: bool = False,
) -> str:
    """Require a bounded, control-character-free string."""
    if not isinstance(value, str):
        raise PublishUnavailable(f"task field {field} has the wrong type")
    if not value and not allow_empty:
        raise PublishUnavailable(f"task field {field} is empty")
    if len(value) > max_chars:
        raise PublishUnavailable(f"task field {field} is too long")
    if len(value.encode("utf-8")) > max_bytes:
        raise PublishUnavailable(f"task field {field} is too large")
    if any(ch < " " or ch == "\x7f" or "\x80" <= ch <= "\x9f" for ch in value):
        raise PublishUnavailable(f"task field {field} contains control characters")
    if not _BIDI_CONTROLS.isdisjoint(value):
        raise PublishUnavailable(f"task field {field} contains bidi controls")
    return value


def _epoch(value: object, field: str) -> int:
    """Require a non-negative epoch-seconds int (never a bool)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise PublishUnavailable(f"task field {field} has the wrong type")
    if value < 0:
        raise PublishUnavailable(f"task field {field} is negative")
    return value


def _sanitized_task(row: "dict[str, object]") -> "dict[str, object]":
    """Rebuild one row from the allowlist, or refuse the whole publication.

    A row that breaks the contract fails the run rather than being skipped:
    dropping it silently would leave an artifact that claims to be the top-N
    while being something else, and ``truncated`` would not describe it.
    """
    status = row["status"]
    if status not in krm.VALID_STATUSES:
        raise PublishUnavailable("task field status is not a known status")

    task_id = row["id"]
    if not isinstance(task_id, str) or not _TASK_ID_RE.fullmatch(task_id):
        raise PublishUnavailable("task field id is not a safe id")

    priority = row["priority"]
    if isinstance(priority, bool) or not isinstance(priority, int):
        # Includes a NULL priority: the ordering contract has no honest place
        # for a row whose sort key does not exist.
        raise PublishUnavailable("task field priority has the wrong type")
    if abs(priority) > krm.MAX_PRIORITY:
        raise PublishUnavailable("task field priority is out of range")

    assignee = row["assignee"]
    if assignee is not None:
        assignee = _safe_text(assignee, "assignee", krm.MAX_ID_CHARS, krm.MAX_ID_BYTES)

    optional = {}
    for field in ("started_at", "completed_at"):
        value = row[field]
        optional[field] = None if value is None else _epoch(value, field)

    return {
        "id": task_id,
        "title": _safe_text(
            row["title"], "title", krm.MAX_FIELD_CHARS, krm.MAX_FIELD_BYTES,
            allow_empty=True,
        ),
        "status": status,
        "assignee": assignee,
        "priority": priority,
        "created_at": _epoch(row["created_at"], "created_at"),
        "started_at": optional["started_at"],
        "completed_at": optional["completed_at"],
    }


# ---------------------------------------------------------------------------
# Atomic publication
# ---------------------------------------------------------------------------

_TMP_FLAGS = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)

_HAS_ATOMIC_PUBLISH = (
    hasattr(os, "O_NOFOLLOW")
    and os.open in getattr(os, "supports_dir_fd", set())
    and os.rename in getattr(os, "supports_dir_fd", set())
    # `os.link` is the rollback: without a second name for the previous
    # artifact there is nothing to put back once the rename has happened, so
    # a platform that lacks `linkat` cannot honour the transaction and is
    # refused rather than served a publication it cannot undo.
    and os.link in getattr(os, "supports_dir_fd", set())
    # `mkdirat` for the same reason: the missing levels below the pinned
    # directory have to be created relative to it, never by pathname.
    and os.mkdir in getattr(os, "supports_dir_fd", set())
)


def _assert_pinned_parent_is_still_named(dir_fd: int, pinned_path: Path) -> None:
    """Refuse if *pinned_path* no longer names the directory that was pinned.

    The fd is what every step of the transaction actually uses, so a swapped
    ancestor cannot redirect a single write. What a swap CAN still do is make
    the publication land somewhere the caller's own pathname no longer
    reaches: the owner would be told the artifact is at ``--out`` while a
    reader following that exact name arrives in the attacker's directory
    instead. That is not a successful publication, so it is refused rather
    than reported as one — before the first byte, so the previous artifact
    keeps its name and its inode.

    Resolving the pathname here is safe precisely because the answer is only
    ever *compared*. Nothing is opened through it, and a mismatch — or a name
    that has stopped resolving at all — fails closed.
    """
    try:
        named = os.stat(pinned_path)
    except OSError as exc:
        raise PublishUnavailable(
            f"output directory moved during publication: {pinned_path}"
        ) from exc
    pinned = os.fstat(dir_fd)
    if (named.st_dev, named.st_ino) != (pinned.st_dev, pinned.st_ino):
        raise PublishUnavailable(
            f"output directory moved during publication: {pinned_path}"
        )


def _descend_creating(dir_fd: int, missing: "tuple[str, ...]") -> "tuple[int, bool]":
    """Create *missing* below *dir_fd* and return ``(fd, owned)`` for the leaf.

    Each level is made with ``mkdirat`` and re-opened with
    ``openat``/``O_NOFOLLOW`` relative to the level above, so the chain stays
    pinned all the way down and a name planted between two steps is refused
    instead of followed. ``owned`` says whether the returned fd is a new one
    this function must have closed by its caller, or the caller's own fd
    handed straight back.
    """
    fd = dir_fd
    owned = False
    try:
        for part in missing:
            try:
                os.mkdir(part, 0o700, dir_fd=fd)
            except FileExistsError:
                # Created since the walk. The `openat`/`O_NOFOLLOW` below is
                # what decides whether that is acceptable, not this.
                pass
            next_fd = os.open(part, _DIR_FLAGS, dir_fd=fd)
            if owned:
                os.close(fd)
            fd = next_fd
            owned = True
        return fd, owned
    except BaseException:
        if owned:
            os.close(fd)
        raise


def _publish_bytes(
    out_path: Path, payload: bytes, dir_fd: int, missing: "tuple[str, ...]"
) -> None:
    """Write *payload* to *out_path* atomically, or leave the path untouched.

    A dedicated ``O_EXCL``/``O_NOFOLLOW`` temp file in the SAME directory is
    written, ``fsync``-ed, chmod-ed to exactly 0600, and then ``rename``-d over
    the target; the directory is ``fsync``-ed last. There is no copy or
    in-place-truncate fallback, so a concurrent reader sees either the previous
    artifact or the next one — never a half-written file.

    Publication is a transaction, and it is only committed once that final
    directory ``fsync`` returns. Up to the rename, failing is simply a matter
    of not having replaced anything yet. After it, it is not: the previous
    artifact's directory entry is already gone. So before the rename a second
    hard link to the previous artifact is made under a dotted name, and a
    failure after the rename renames that link back over the target. The
    previous artifact comes back as the SAME inode — identical bytes, identical
    mode — because it never stopped existing; only its name did.

    Either way the directory ends up holding exactly what it held before, plus
    or minus the artifact itself: the temp file and the backup link are removed
    on every path out of here.

    *dir_fd* and *missing* come straight from
    :func:`_validated_output_location` and are the only handle on the
    destination this function has. The parent is never re-opened by pathname,
    because that is exactly how a validated destination gets swapped out from
    under a publication.
    """
    if not _HAS_ATOMIC_PUBLISH:
        raise PublishUnavailable("platform lacks openat/O_NOFOLLOW/renameat")

    parent = out_path.parent
    pinned_path = parent
    for _ in missing:
        pinned_path = pinned_path.parent
    _assert_pinned_parent_is_still_named(dir_fd, pinned_path)

    try:
        dir_fd, owned_fd = _descend_creating(dir_fd, missing)
    except OSError as exc:
        raise PublishUnavailable(f"output directory is unusable: {parent}") from exc

    # Dotted and randomised so a partially-written artifact can never be
    # mistaken for the published one, and two runs cannot collide.
    stamp = f"{os.getpid()}.{os.urandom(6).hex()}"
    tmp_name = f".{out_path.name}.{stamp}.tmp"
    backup_name = f".{out_path.name}.{stamp}.prev"

    had_previous = False
    renamed = False
    try:
        fd = os.open(tmp_name, _TMP_FLAGS, krm.ARTIFACT_MODE, dir_fd=dir_fd)
        try:
            # Explicit: O_CREAT's mode is masked by the umask, and the reader
            # requires exactly 0600, not "0600 minus whatever umask says".
            os.fchmod(fd, krm.ARTIFACT_MODE)
            written = 0
            while written < len(payload):
                written += os.write(fd, payload[written:])
            os.fsync(fd)
        finally:
            os.close(fd)

        try:
            os.link(
                out_path.name, backup_name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd
            )
            had_previous = True
        except FileNotFoundError:
            # First publication: there is no previous artifact to preserve, so
            # "unchanged" means the target is absent again if this fails.
            had_previous = False

        os.replace(tmp_name, out_path.name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        renamed = True
        os.fsync(dir_fd)
    except OSError as exc:
        _rollback_publication(
            dir_fd,
            out_path,
            tmp_name,
            backup_name,
            had_previous=had_previous,
            renamed=renamed,
        )
        raise PublishUnavailable(f"could not publish artifact: {exc}") from exc
    else:
        if had_previous:
            try:
                os.unlink(backup_name, dir_fd=dir_fd)
            except OSError:
                # The publication is already committed and durable, so this is
                # cleanup, not the transaction: reporting it as a failure would
                # tell the caller nothing was published when something was. The
                # worst case is a stale dotted sibling of a correct artifact.
                pass
    finally:
        # Only the fd this function opened itself. The one validation handed
        # over stays the caller's, so it is closed exactly once, over there.
        if owned_fd:
            os.close(dir_fd)


def _rollback_publication(
    dir_fd: int,
    out_path: Path,
    tmp_name: str,
    backup_name: str,
    *,
    had_previous: bool,
    renamed: bool,
) -> None:
    """Undo a publication that failed, leaving the directory as it was found.

    Best effort by necessity — the filesystem is already failing when this
    runs — and deliberately silent: every caller re-raises the original failure
    afterwards, and a rollback error would only replace the real reason with a
    less useful one.
    """
    if renamed:
        # The swap happened, so the target currently holds the NEW artifact.
        if had_previous:
            # Atomically put the previous inode back under its own name; this
            # drops the new one in the same step.
            try:
                os.replace(
                    backup_name,
                    out_path.name,
                    src_dir_fd=dir_fd,
                    dst_dir_fd=dir_fd,
                )
            except OSError:
                pass
        else:
            # There was nothing here before, so a failed publication must not
            # leave a first artifact behind.
            try:
                os.unlink(out_path.name, dir_fd=dir_fd)
            except OSError:
                pass
        try:
            os.fsync(dir_fd)
        except OSError:
            pass
        return

    # The rename never happened: the previous artifact still holds its name,
    # and the only things to clear are the intermediates.
    for name in (tmp_name, backup_name):
        try:
            os.unlink(name, dir_fd=dir_fd)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def publish_read_model(
    *,
    kanban_root: "os.PathLike[str] | str",
    board: object,
    db: "os.PathLike[str] | str",
    out: "os.PathLike[str] | str",
    limit: int,
    now: "int | None" = None,
) -> str:
    """Publish the artifact at *out* and return its path.

    Order matters: authority first, then the cheap filesystem-free checks,
    then the explicit paths, and only then the database. A refusal at any of
    those points has opened nothing and written nothing.
    """
    assert_owner_publish_authority()

    limit = _validated_limit(limit)
    try:
        board = krm.validated_board(board)
        root = krm.validated_kanban_root(kanban_root)
        # The artifact must live outside the Kanban root — the same lexical
        # containment rule the reader enforces, checked from this side too.
        out_path = krm.validated_artifact_path(out, root)
    except krm.ReadModelUnavailable as exc:
        raise PublishUnavailable(str(exc)) from exc

    # Where the artifact may land is settled before the database path is even
    # validated: a refused destination must cost nothing on disk and nothing
    # in the board. It hands back an fd for the directory it approved, and
    # that fd — not the pathname — is what the publication below runs on.
    dir_fd, missing = _validated_output_location(out_path)
    try:
        if not missing:
            # A missing directory level means the target cannot exist yet, so
            # the question only arises when the whole path is already there.
            # Asked here, before the database is named: an existing target the
            # publisher may not replace costs nothing on disk and nothing in
            # the board.
            _assert_existing_target_is_republishable(
                dir_fd, out_path, root, board
            )
        return _publish_validated(
            root=root,
            board=board,
            db=db,
            out_path=out_path,
            limit=limit,
            now=now,
            dir_fd=dir_fd,
            missing=missing,
        )
    finally:
        # The one close, on every outcome: a refused DB path, a failed query,
        # a rejected row, an over-cap payload, and the successful publication
        # all leave through here.
        os.close(dir_fd)


def _publish_validated(
    *,
    root: Path,
    board: str,
    db: "os.PathLike[str] | str",
    out_path: Path,
    limit: int,
    now: "int | None",
    dir_fd: int,
    missing: "tuple[str, ...]",
) -> str:
    """Everything after the destination is pinned, with *dir_fd* borrowed.

    Split out only so :func:`publish_read_model` can own that fd in a single
    ``try``/``finally``; nothing here closes it.
    """
    db_path = _validated_db_path(db, root, board)
    db_dir_fd = _open_board_db_parent(root, db_path)
    try:
        pinned = _pinned_board_db(db_dir_fd, db_path)
        rows, truncated = _fetch_task_rows(
            db_path, limit, dir_fd=db_dir_fd, pinned=pinned
        )
    finally:
        # The board directory is pinned only for as long as the read needs it;
        # nothing below this line touches the Kanban root again.
        os.close(db_dir_fd)
    tasks = [_sanitized_task(row) for row in rows]
    if len({task["id"] for task in tasks}) != len(tasks):
        raise PublishUnavailable("board produced duplicate task ids")

    document = {
        "schema_version": krm.SCHEMA_VERSION,
        "generated_at": int(time.time()) if now is None else now,
        "board": board,
        "root_fingerprint": krm.root_fingerprint(root),
        "truncated": truncated,
        "tasks": [
            {field: task[field] for field in krm.TASK_FIELDS} for task in tasks
        ],
    }
    if set(document) != set(krm.TOP_LEVEL_FIELDS):
        raise PublishUnavailable("artifact is not the top-level allowlist")

    payload = json.dumps(document, ensure_ascii=True).encode("utf-8")
    if len(payload) > krm.MAX_ARTIFACT_BYTES:
        # Checked before the write, so an over-cap board leaves the previous
        # artifact in place instead of replacing it with one nobody can read.
        raise PublishUnavailable("artifact exceeds the size cap")

    _publish_bytes(out_path, payload, dir_fd, missing)
    return str(out_path)
