"""Spawn-seam substrate import guard for the kanban dispatcher (card t_a2f35c4d, ruling t_dbf5c876).

Split out of ``hermes_cli.kanban_db_dispatch`` (a facade already over its file cap):
the guard, its naming line, its classifier and the lane actor's filing live here.

The facade keeps the two seams callers patch -- ``_dispatching_tree_root`` (the one
root every helper takes as data) and ``_record_spawn_failure`` (the one entry point
both spawn ``except`` sites share). Everything in this module is reached through
them, so ``kanban_db_dispatch._dispatching_tree_root`` stays the patch point and
``kanban_db`` stays the seam for board writes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import sqlite3
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any
from typing import Optional


def _facade():
    """The dispatch facade, late-bound: it owns the tree root and the failure recorder."""
    from hermes_cli import kanban_db_dispatch as facade

    return facade


def _kb():
    """``hermes_cli.kanban_db``, late-bound so monkeypatching ``kanban_db.<name>`` works."""
    from hermes_cli import kanban_db as mod

    return mod


# ---------------------------------------------------------------------------
# Spawn-seam substrate import guard (card t_a2f35c4d, ruling t_dbf5c876)
# ---------------------------------------------------------------------------
#
# Measured incident (2026-10-10): every review-run worker spawn on the fleet died with
# ``cannot import name 'ADVISORY_SKILLS_ENV' from 'agent.skill_commands'`` -- an import no
# source file contained -- and the whole review LANE went silently dead. The stale
# ``cpython-311`` bytecode beside the tree was a smell, not the wound: CPython selects
# ``__pycache__/<mod>.<sys.implementation.cache_tag>.pyc``, so a 3.14 runtime cannot load a
# 3.11 pyc at all (measured: 733 inert foreign-tag files against 1594 live ones, 0 sourceless,
# 0 hash-based). The wound is a live tree whose OWN modules disagree -- an importer present,
# its provider absent -- so the durable half is to NAME that artifact at the spawn seam and to
# classify it as SUBSTRATE: the card spends nothing of its retry budget, and one signature-
# deduped lane actor is filed instead of the lane dying quietly.
#
# Boundary (deliberate, ruling D1): nothing here refuses a spawn on foreign-tag bytecode. A
# foreign-tag population is inert, and a refusal would be a false-positive spawn killer -- the
# exact failure mode this guard exists to remove. The sweep helper REPORTS the two layouts that
# can change behaviour (a sourceless ``module.pyc``, a non-zero header flags field) and measures
# the foreign-tag population as inert.

#: Greppable prefix of the one-line naming payload (D4). It leads, because
#: ``_record_task_failure`` truncates the stored error at 500 chars.
TREE_IMPORT_INCONSISTENCY = "tree-import-inconsistency:"

#: Classifier verdicts (D5). ``card`` is the default and never widens on its own.
SPAWN_FAILURE_CARD = "card"
SPAWN_FAILURE_SUBSTRATE_IMPORT = "substrate_import"
SPAWN_FAILURE_HOST_CAPACITY = "host_capacity"

#: ``substrate_import:<provider-relpath>:<symbol>`` -- the lane-level dedupe key (D6).
SUBSTRATE_SIGNATURE_PREFIX = "substrate_import:"

#: The LANE actor: one card per (signature, board, window), addressed to the domain's design
#: authority. The idempotency key is what makes N failures across N cards resolve to ONE card.
SUBSTRATE_ACTOR_IDEMPOTENCY_PREFIX = "substrate:"
SUBSTRATE_ACTOR_ASSIGNEE = "platform-stl"
SUBSTRATE_ACTOR_EVENT = "substrate_actor_filed"
SUBSTRATE_ACTOR_TITLE = "[guard] substrate import failure in the dispatching tree"

#: ``from x import y`` -- the shape whose ``y`` is the symbol the naming line quotes.
_CANNOT_IMPORT_NAME_RE = re.compile(r"cannot import name ['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]")

#: A ``def`` / ``class`` / assignment binding of a name: the ``defines-it`` HINT (D4 says a hint,
#: never authority -- the verdict that matters is the interpreter's, and that is the exception).
_SYMBOL_BINDING_RE_TEMPLATE = (
    r"(?m)^[ \t]*(?:async[ \t]+def|def|class)[ \t]+{name}\b"
    r"|^[ \t]*{name}[ \t]*(?::[^=\n]*)?=(?!=)"
    r"|^[ \t]*{name}[ \t]*(?:,|\)|\]|\}})"
)




def _pyc_tag(filename: str) -> Optional[str]:
    """The interpreter tag of a bytecode filename: ``mod.cpython-311.pyc`` -> ``cpython-311``.

    ``mod.pyc`` (untagged) -> None. The tag is the LAST dot-separated component before ``.pyc``,
    because a tag itself carries a hyphen -- splitting on ``-`` would mangle it.
    """
    name = str(filename)
    if not name.endswith(".pyc"):
        return None
    stem = name[: -len(".pyc")]
    if "." not in stem:
        return None
    return stem.rsplit(".", 1)[1] or None


def _pyc_header_flags(raw: bytes) -> int:
    """The 4-byte header flags field of a ``.pyc`` (0 when too short to carry one).

    Header layout: 4-byte magic, 4-byte flags, then either (mtime, size) or an 8-byte hash.
    """
    if len(raw) < 8:
        return 0
    return int.from_bytes(raw[4:8], "little")


def sweep_tree_bytecode_pycs(tree: Path) -> dict[str, Any]:
    """REPORT the bytecode layouts under ``tree`` that can change behaviour -- and those that cannot.

    ``sourceless``: an ``*.pyc`` OUTSIDE ``__pycache__`` with no ``*.py`` beside it -- the import
    system will happily load it, so it can shadow source that a checkout thinks it has.
    ``flags_nonzero`` / ``hash_based``: a header whose flags field is non-zero (a
    hash-invalidated pyc, whose staleness is judged by content rather than mtime).
    ``current_tag`` / ``foreign_tag`` / ``untagged``: the tag population. A FOREIGN tag is inert
    -- this interpreter can never select it -- which is why this function only ever reports and
    no spawn is gated on the result (ruling D1).
    """
    tree = Path(tree)
    cache_tag = sys.implementation.cache_tag
    report: dict[str, Any] = {
        "tree": str(tree),
        "cache_tag": cache_tag,
        "sourceless": [],
        "current_tag": 0,
        "foreign_tag": 0,
        "untagged": 0,
        "flags_nonzero": 0,
        "hash_based": 0,
        "pyc_total": 0,
    }
    try:
        paths = sorted(tree.rglob("*.pyc"))
    except OSError:
        return report
    for path in paths:
        report["pyc_total"] += 1
        tag = _pyc_tag(path.name)
        if tag == cache_tag:
            report["current_tag"] += 1
        elif tag is None:
            report["untagged"] += 1
        else:
            report["foreign_tag"] += 1
        if "__pycache__" not in path.parts and not path.with_suffix(".py").exists():
            try:
                rel = path.relative_to(tree).as_posix()
            except ValueError:
                rel = str(path)
            report["sourceless"].append(rel)
        try:
            flags = _pyc_header_flags(path.read_bytes())
        except OSError:
            continue
        if flags:
            report["flags_nonzero"] += 1
            if flags & 0b11:
                report["hash_based"] += 1
    return report


def _is_under(path: Any, tree: Path) -> bool:
    """Does ``path`` resolve inside ``tree``? The guard's whole "this tree's own" test."""
    try:
        Path(path).resolve().relative_to(Path(tree).resolve())
        return True
    except (OSError, TypeError, ValueError):
        return False


def _relative_to_tree(path: Any, tree: Path) -> str:
    """``path`` spelled relative to ``tree`` (posix), or its absolute spelling when it escapes."""
    try:
        return Path(path).resolve().relative_to(Path(tree).resolve()).as_posix()
    except (OSError, TypeError, ValueError):
        return str(path)


def _importer_frame(exc: BaseException, tree: Path) -> Optional[tuple[str, int]]:
    """Deepest traceback frame whose file lives under ``tree``: ``(relpath, lineno)`` or None.

    "Deepest" is the frame that RAISED -- the importer whose ``from ... import`` disagreed. The
    interpreter's own ``importlib`` frames are skipped by the under-tree test, and a failure whose
    only frames are outside the tree is by definition not the tree's (the card's, the test's, or
    a third-party module's).
    """
    frames = traceback.extract_tb(getattr(exc, "__traceback__", None))
    for frame in reversed(frames):
        if _is_under(frame.filename, tree):
            return _relative_to_tree(frame.filename, tree), int(frame.lineno or 0)
    return None


def _imported_symbol(exc: BaseException) -> Optional[str]:
    """The name the importer asked for: ``cannot import name 'X' from 'Y'`` -> ``X``.

    Only that shape carries a symbol, and a symbol is what the signature and the naming line are
    built from -- a bare ``No module named 'x'`` has none, so it cannot be deduped and stays the
    card's (deliberately never widened).
    """
    match = _CANNOT_IMPORT_NAME_RE.search(str(exc))
    return match.group(1) if match else None


def _provider_source(exc: BaseException) -> Optional[Path]:
    """The provider module's SOURCE FILE, resolved from its spec -- the spec is never imported.

    ``importlib.util.find_spec`` answers where the name would load from without executing it, so
    reporting a broken provider cannot itself run the broken module. A spec that is absent
    (a missing third-party dependency) or whose origin is not a ``.py`` file (a builtin, a frozen
    module, an extension, a namespace package) is not a source this guard speaks about -> None.
    """
    name = getattr(exc, "name", None)
    origin: Any = None
    if name:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, AttributeError, TypeError, ValueError):
            # No resolvable spec (a missing parent package, an odd name): not a source
            # this guard speaks about, so the failure stays the card's.
            spec = None
        origin = getattr(spec, "origin", None) if spec is not None else None
    if not origin:
        raw_path = getattr(exc, "path", None)
        origin = raw_path if isinstance(raw_path, (str, os.PathLike)) else None
    if not origin:
        return None
    path = Path(origin)
    return path if path.suffix == ".py" else None


def _source_defines_symbol(text: str, symbol: str) -> bool:
    """Does this provider source BIND ``symbol``? A hint for the naming line, never authority."""
    return re.search(
        _SYMBOL_BINDING_RE_TEMPLATE.format(name=re.escape(symbol)), text,
    ) is not None


def _head_verdict(tree: Path, relpath: str, on_disk_sha256: str) -> Optional[bool]:
    """Is the on-disk file different from ``HEAD:<relpath>``? ``None`` when git cannot answer.

    The exit status is checked FIRST: a failed ``git show`` piped into a hash yields the
    EMPTY-INPUT digest (``e3b0c442...``), which would report a difference that does not exist
    (ruling D4). A tree git cannot answer for reports unknown, never a verdict.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(tree), "show", f"HEAD:{relpath}"],
            capture_output=True, timeout=30, check=False,
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return hashlib.sha256(proc.stdout).hexdigest() != on_disk_sha256


def _foreign_tag_pyc_count(provider: Path) -> int:
    """Bytecode files beside the provider carrying a tag THIS interpreter cannot select."""
    cache = provider.parent / "__pycache__"
    try:
        entries = list(cache.glob("*.pyc"))
    except OSError:
        return 0
    cache_tag = sys.implementation.cache_tag
    return sum(1 for entry in entries if _pyc_tag(entry.name) not in (None, cache_tag))


def _tree_import_evidence(exc: BaseException, tree: Path) -> Optional[dict[str, Any]]:
    """What the naming line quotes about a tree-internal import disagreement, or None.

    None means "not this tree's own disagreement", and it is the ONLY place that decision is
    made: BOTH halves must hold -- an ``ImportError`` carrying a symbol, whose deepest under-tree
    frame raised it, whose provider resolves to a source file under the same tree. A missing
    third-party dependency (no resolvable spec, or one outside the tree) therefore stays on the
    card. This runs on the FAILURE path only; the happy path walks no import closure (D4).
    """
    if not isinstance(exc, ImportError):
        return None
    importer = _importer_frame(exc, tree)
    if importer is None:
        return None
    provider = _provider_source(exc)
    if provider is None or not _is_under(provider, tree):
        return None
    symbol = _imported_symbol(exc)
    if not symbol:
        return None
    try:
        raw = provider.read_bytes()
        stat = provider.stat()
    except OSError:
        return None
    sha256 = hashlib.sha256(raw).hexdigest()
    provider_rel = _relative_to_tree(provider, tree)
    defines_it = _source_defines_symbol(raw.decode("utf-8", "replace"), symbol)
    return {
        "importer_rel": importer[0],
        "importer_line": importer[1],
        "symbol": symbol,
        "provider_rel": provider_rel,
        "provider_sha256": sha256,
        "provider_mtime": int(stat.st_mtime),
        "provider_size": int(stat.st_size),
        "defines_it": defines_it,
        "differs_from_head": _head_verdict(tree, provider_rel, sha256),
        "foreign_tag_pyc": _foreign_tag_pyc_count(provider),
    }


def describe_tree_import_failure(
    exc: BaseException, *, tree: Optional[Path] = None, task_id: Optional[str] = None,
) -> Optional[str]:
    """ONE line naming the artifact, front-loaded for the 500-char truncation (D4), else None.

    Shape::

        tree-import-inconsistency: importer <relpath>:<lineno> imports <symbol>; provider
        <relpath> sha256=<64> mtime=<epoch> size=<n> defines-it=no differs-from-HEAD=yes;
        tree-self-consistent=no; foreign-tag-pyc-in-provider-dir=0; nothing of card <id> ran

    ``tree-self-consistent`` is the tree's own verdict: this closure has exactly one witness --
    the provider's current source -- so it reads the same fact as ``defines-it``, and it is
    carried separately because it is what the reader greps for ("are this tree's modules
    agreeing?"). ``differs-from-HEAD`` says whether the broken provider is the committed bytes,
    which is what separates "someone edited this tree" from "the tree shipped broken".
    """
    tree = Path(tree) if tree is not None else _facade()._dispatching_tree_root()
    evidence = _tree_import_evidence(exc, tree)
    if evidence is None:
        return None
    head = evidence["differs_from_head"]
    head_text = "unknown" if head is None else ("yes" if head else "no")
    self_consistent = "yes" if evidence["defines_it"] else "no"
    return (
        f"{TREE_IMPORT_INCONSISTENCY} importer {evidence['importer_rel']}:{evidence['importer_line']}"
        f" imports {evidence['symbol']}; provider {evidence['provider_rel']}"
        f" sha256={evidence['provider_sha256']} mtime={evidence['provider_mtime']}"
        f" size={evidence['provider_size']} defines-it={'yes' if evidence['defines_it'] else 'no'}"
        f" differs-from-HEAD={head_text}; tree-self-consistent={self_consistent};"
        f" foreign-tag-pyc-in-provider-dir={evidence['foreign_tag_pyc']};"
        f" nothing of card {task_id or 'unknown'} ran"
    )


def substrate_import_signature(
    exc: BaseException, *, tree: Optional[Path] = None,
) -> Optional[str]:
    """``substrate_import:<provider-relpath>:<symbol>`` -- the lane-level dedupe key (D6), or None."""
    tree = Path(tree) if tree is not None else _facade()._dispatching_tree_root()
    evidence = _tree_import_evidence(exc, tree)
    if evidence is None:
        return None
    return f"{SUBSTRATE_SIGNATURE_PREFIX}{evidence['provider_rel']}:{evidence['symbol']}"


def classify_spawn_failure(exc: BaseException, *, tree: Optional[Path] = None) -> str:
    """The ONE decider for who owns a failed spawn: ``card`` | ``substrate_import`` | ``host_capacity``.

    * ``host_capacity`` -- the host refused to place the worker (``RestartSafeScopeUnavailable``):
      neither the card's failure nor the tree's, retried spaced and never charged.
    * ``substrate_import`` -- THIS tree's own modules disagree: the importer is source under the
      tree and its provider is source under the tree that does not supply the name. Nothing of
      the card ran, so the failure must not spend the card's retry budget, and it must be counted
      for the LANE actor instead of parking the card.
    * ``card`` -- everything else, deliberately including a missing third-party dependency and a
      failure whose frames are all outside the tree. The classifier never widens its own reach:
      an import failure it cannot attribute to THIS tree stays the card's.

    Callers must not scatter ``isinstance`` checks beside this: one verdict per failure.
    """
    try:
        from tools.process_registry import RestartSafeScopeUnavailable
    except ImportError:  # pragma: no cover - a host without the module has no such refusal
        RestartSafeScopeUnavailable = ()  # type: ignore[assignment]
    if RestartSafeScopeUnavailable and isinstance(exc, RestartSafeScopeUnavailable):
        return SPAWN_FAILURE_HOST_CAPACITY
    tree = Path(tree) if tree is not None else _facade()._dispatching_tree_root()
    if _tree_import_evidence(exc, tree) is not None:
        return SPAWN_FAILURE_SUBSTRATE_IMPORT
    return SPAWN_FAILURE_CARD


def _spawn_failure_report(
    exc: BaseException, task_id: str, *, tree: Optional[Path] = None,
) -> tuple[str, str, Optional[dict[str, Any]]]:
    """``(error_text, kind, metadata_extra)`` for one failed spawn -- the seam's single classify call.

    ``error_text`` is the naming line when this is the tree's own import disagreement (it LEADS,
    because the stored error is cut at 500 chars), else the exception's own text. ``metadata_extra``
    rides the run and the event so the lane actor can dedupe on the signature without re-deriving
    it from prose later.
    """
    tree = Path(tree) if tree is not None else _facade()._dispatching_tree_root()
    kind = classify_spawn_failure(exc, tree=tree)
    if kind != SPAWN_FAILURE_SUBSTRATE_IMPORT:
        return str(exc), kind, None
    metadata_extra: dict[str, Any] = {"substrate_import": True}
    signature = substrate_import_signature(exc, tree=tree)
    if signature:
        metadata_extra["substrate_signature"] = signature
    named = describe_tree_import_failure(exc, tree=tree, task_id=task_id)
    return (named or str(exc)), kind, metadata_extra


def _substrate_failure_threshold() -> int:
    """``kanban.failure_threshold`` (legacy alias ``spawn_failure_threshold``) else the default.

    NO new config key (D5): the lane actor's bar is the same one the per-card breaker uses, so an
    operator who already tuned ``failure_threshold`` tunes this too. A non-positive or
    unparseable value falls back to the dispatcher's own ``DEFAULT_FAILURE_LIMIT``.
    """
    kanban_cfg: Any = {}
    log = _kb()._log
    try:
        from hermes_cli.config import load_config_readonly

        kanban_cfg = (load_config_readonly() or {}).get("kanban") or {}
    except Exception:
        # A config that cannot be read must not take the failure path down; the traceback is
        # logged (BLE001's deliberate-catch-all shape) and the empty mapping falls through to
        # the same default the per-card breaker uses.
        log.debug("kanban dispatcher: substrate threshold unreadable", exc_info=True)
    if not isinstance(kanban_cfg, dict):
        return _facade().DEFAULT_FAILURE_LIMIT
    raw = kanban_cfg.get("failure_threshold", kanban_cfg.get("spawn_failure_threshold"))
    if raw is None:
        return _facade().DEFAULT_FAILURE_LIMIT
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return _facade().DEFAULT_FAILURE_LIMIT
    return value if value >= 1 else _facade().DEFAULT_FAILURE_LIMIT


def _substrate_window(now: int) -> tuple[int, int]:
    """``(window_start, window_seconds)`` for the lane actor's idempotency key.

    The window is the SAME cooldown the retired card is spaced by
    (``_resolve_rate_limit_cooldown_seconds``), bucketed so every failure inside one window
    resolves to one key -- that bucketing is what makes the actor idempotent per tick. A disabled
    cooldown (0, the tests' switch) still needs a bucket, so it falls back to the default.
    """
    seconds = _kb()._resolve_rate_limit_cooldown_seconds()
    if seconds <= 0:
        seconds = _facade().DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS
    now = int(now)
    return (now // seconds) * seconds, seconds


def _substrate_affected_cards(
    conn: sqlite3.Connection, signature: str, *, window_start: int,
) -> list[str]:
    """DISTINCT cards whose recorded substrate failure carries ``signature`` in this window.

    Read from the board's own events: the ``spawn_failed`` row the seam writes carries
    ``substrate_signature`` (``_record_task_failure(..., metadata_extra=...)``), so the count is
    per BOARD with no new table and no new config key. Distinct CARDS, not failures: the actor
    exists because a LANE is dying, and one hot card retrying is the per-card ``gave_up``'s story.
    """
    affected: list[str] = []
    for row in conn.execute(
        "SELECT task_id, payload FROM task_events "
        "WHERE kind = 'spawn_failed' AND created_at >= ? ORDER BY created_at",
        (int(window_start),),
    ).fetchall():
        payload = _kb()._json_dict(row["payload"])
        if payload.get("substrate_signature") != signature:
            continue
        task_id = row["task_id"]
        if task_id not in affected:
            affected.append(task_id)
    return affected


def _substrate_provider_path(tree: Path, signature: str) -> Optional[Path]:
    """The provider file a signature names: ``substrate_import:<relpath>:<symbol>`` -> path."""
    body = signature[len(SUBSTRATE_SIGNATURE_PREFIX):]
    if ":" not in body:
        return None
    relpath = body.rsplit(":", 1)[0]
    if not relpath:
        return None
    return Path(tree) / relpath


def _recorded_naming_line(conn: sqlite3.Connection, affected: list[str]) -> Optional[str]:
    """The naming line as RECORDED on one affected card -- quoted verbatim in the actor's body."""
    for task_id in affected:
        row = conn.execute(
            "SELECT last_failure_error FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        text = _kb()._lossy_text(row["last_failure_error"]) if row is not None else None
        if text and TREE_IMPORT_INCONSISTENCY in text:
            return text
    return None


def _substrate_actor_body(
    conn: sqlite3.Connection, signature: str, affected: list[str], tree: Path,
) -> str:
    """The actor card's body: the naming line, the affected cards, and the tree evidence (D6).

    Deterministic -- no model, no prose a human has to interpret. What a reader needs is: which
    module disagreeing with which symbol, which cards died on it, and the tree's own state.
    """
    window_start, window_seconds = _substrate_window(int(time.time()))
    lines = [
        "## Observed",
        "",
        f"`{TREE_IMPORT_INCONSISTENCY}` the dispatching tree's own modules disagree, so worker",
        "spawns are dying at the seam. Filed by the dispatcher's spawn-seam substrate guard --",
        "deterministic, no model was consulted.",
        "",
        f"- signature: `{signature}`",
        f"- affected cards ({len(affected)}): " + ", ".join(f"`{tid}`" for tid in affected),
        f"- window: {window_start} (+{window_seconds}s)",
        "",
        "## Recorded failure line",
        "",
        "```",
        _recorded_naming_line(conn, affected) or f"{TREE_IMPORT_INCONSISTENCY} {signature}",
        "```",
        "",
        "## Tree evidence",
        "",
        f"- tree: `{tree}`",
    ]
    provider = _substrate_provider_path(tree, signature)
    if provider is not None and provider.is_file():
        try:
            raw = provider.read_bytes()
            stat = provider.stat()
        except OSError:
            raw, stat = b"", None
        if stat is not None:
            lines.append(
                f"- provider: `{_relative_to_tree(provider, tree)}`"
                f" sha256={hashlib.sha256(raw).hexdigest()}"
                f" mtime={int(stat.st_mtime)} size={int(stat.st_size)}"
            )
    else:
        lines.append(f"- provider: `{_substrate_provider_path(tree, signature)}` (not readable)")
    lines += ["", "```", *_git_status_lines(tree), "```"]
    return "\n".join(lines)


def _git_status_lines(tree: Path) -> list[str]:
    """``git status --porcelain`` for the tree, or the reason it could not be read."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(tree), "status", "--porcelain"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return [f"(git unavailable: {exc})"]
    if proc.returncode != 0:
        return [f"(git status failed rc={proc.returncode}: {proc.stderr.strip()})"]
    return proc.stdout.splitlines() or ["(clean)"]


def _board_key(conn: sqlite3.Connection, board: Optional[str]) -> str:
    """The board slug the actor's idempotency key is scoped to.

    The CONNECTION is the board: ``PRAGMA database_list`` names the DB file it writes, and the
    directory holding it IS the slug (``<kanban_home>/boards/<slug>/kanban.db``). An explicit
    ``board`` argument wins when the caller carries one, and the current-board helper is the last
    fallback. Only upstream helpers are used: an override-local one would make the key's spelling
    depend on whatever else the tree happens to carry.
    """
    if board:
        try:
            explicit = _kb()._normalize_board_slug(board)
        except (AttributeError, TypeError, ValueError):
            explicit = None
        if explicit:
            return explicit
    try:
        for row in conn.execute("PRAGMA database_list").fetchall():
            if str(row[1]) == "main" and row[2]:
                db_file = Path(str(row[2]))
                if db_file.parent == Path(_kb().kanban_home()):
                    return "default"      # the default board's DB sits at <kanban_home>/kanban.db
                slug = db_file.parent.name
                if slug:
                    return slug
    except sqlite3.Error:
        # A connection that cannot name its file still gets a key, just not a board-scoped
        # one; log the reason so a broken DB handle is not mistaken for the default board.
        _kb()._log.debug("kanban dispatcher: board slug unreadable from the connection", exc_info=True)
    try:
        return _kb()._slug_or_default(board)
    except (AttributeError, TypeError, ValueError):
        return "default"


def _file_substrate_actor(
    conn: sqlite3.Connection,
    signature: str,
    *,
    affected: list[str],
    tree: Path,
    board: Optional[str] = None,
) -> Optional[str]:
    """File EXACTLY ONE acting card for a lane-level signature, idempotently (D6).

    The idempotency key (``substrate:<signature>:<board>:<window-start>``) is the whole mechanism:
    N cards failing the same way in one window resolve to one card, and a tree that stays broken
    into the NEXT window can file its own. Best-effort by design -- a board that refuses the write
    must not take the dispatcher tick down with it, and the per-card naming line is already
    recorded either way -- but never silent: the refusal is logged with the key it could not file.
    """
    key = (
        f"{SUBSTRATE_ACTOR_IDEMPOTENCY_PREFIX}{signature}"
        f":{_board_key(conn, board)}:{_substrate_window(int(time.time()))[0]}"
    )
    try:
        existing = conn.execute(
            "SELECT id FROM tasks WHERE idempotency_key = ? AND status != 'archived' "
            "ORDER BY created_at DESC LIMIT 1",
            (key,),
        ).fetchone()
        if existing is not None:
            return existing["id"]
        actor_id = _kb().create_task(
            conn,
            title=SUBSTRATE_ACTOR_TITLE,
            body=_substrate_actor_body(conn, signature, affected, tree),
            assignee=SUBSTRATE_ACTOR_ASSIGNEE,
            created_by="kanban-dispatcher",
            idempotency_key=key,
        )
        with _kb().write_txn(conn):
            _kb()._append_event(
                conn, actor_id, SUBSTRATE_ACTOR_EVENT,
                {"signature": signature, "affected": list(affected),
                 "board": _board_key(conn, board)},
            )
        return actor_id
    except Exception as exc:  # health: allow BLE001 -- one tick must survive any board refusal; logged with the traceback
        _kb()._log.warning(
            "kanban dispatcher: could not file the substrate actor for %s (key=%s): %s",
            signature, key, exc, exc_info=True,
        )
        return None


def _maybe_file_substrate_actor(
    conn: sqlite3.Connection,
    signature: Optional[str],
    *,
    tree: Path,
    board: Optional[str] = None,
) -> Optional[str]:
    """File the lane actor once ``kanban.failure_threshold`` cards share one signature (D6).

    A per-card ``blocked`` + ``gave_up`` names a card, never the lane -- which is how the review
    lane went silently dead. This is the lane-level half: count DISTINCT cards carrying the
    signature inside the cooldown window, across the board, and at the threshold file one acting
    card for the design authority. The existing per-card ``gave_up`` is left exactly as it was.
    """
    if not signature:
        return None
    window_start, _ = _substrate_window(int(time.time()))
    affected = _substrate_affected_cards(conn, signature, window_start=window_start)
    if len(affected) < _substrate_failure_threshold():
        return None
    return _file_substrate_actor(
        conn, signature, affected=affected, tree=tree, board=board,
    )


def record_spawn_failure(
    conn: sqlite3.Connection,
    task_id: str,
    exc: BaseException,
    *,
    failure_limit: int,
    board: Optional[str] = None,
    prefix: str = "",
    tree: Path,
) -> bool:
    """Record one failed spawn through the guard: classify, name, charge-or-not, then escalate.

    Returns True when the card was auto-blocked. One helper for BOTH spawn-seam ``except`` sites
    (workspace resolution and the spawn call), so the verdict is reached exactly once, by
    :func:`classify_spawn_failure`, and no caller re-implements the infrastructure rule beside it
    (D5: one named classifier, no scattered ``isinstance`` checks).
    """
    error_text, kind, metadata_extra = _spawn_failure_report(exc, task_id, tree=tree)
    infrastructure = kind in (SPAWN_FAILURE_SUBSTRATE_IMPORT, SPAWN_FAILURE_HOST_CAPACITY)
    if kind == SPAWN_FAILURE_HOST_CAPACITY:
        _kb()._log.warning(
            "kanban dispatcher: spawn of %s deferred, host cannot place the worker: %s",
            task_id, exc,
        )
    elif kind == SPAWN_FAILURE_SUBSTRATE_IMPORT:
        _kb()._log.warning(
            "kanban dispatcher: spawn of %s refused, the dispatching tree's own modules "
            "disagree: %s", task_id, error_text,
        )
    # The naming line leads for a substrate failure; an ordinary failure keeps its caller's
    # context label ("workspace: ...").
    text = error_text if kind == SPAWN_FAILURE_SUBSTRATE_IMPORT else f"{prefix}{error_text}"
    blocked = _facade()._record_task_failure(
        conn, task_id, text,
        outcome="spawn_failed", failure_limit=failure_limit, release_claim=True, end_run=True,
        infrastructure=infrastructure, metadata_extra=metadata_extra,
    )
    if kind == SPAWN_FAILURE_SUBSTRATE_IMPORT:
        _maybe_file_substrate_actor(
            conn, (metadata_extra or {}).get("substrate_signature"), tree=tree, board=board,
        )
    return blocked
