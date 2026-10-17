"""Bounded CONTENT fingerprint of the source a long-lived process imported.

The incident this answers (platform-stl ruling ``t_8fed34c8``, R2; measured live 2026-10-10):
an interrupted ``git merge`` moved the WORKING TREE while ``.git/HEAD`` stayed at the pre-merge
commit, so the gateway's ref-only fingerprint (:func:`gateway.code_skew.detect_code_skew`, which
reads ``.git/HEAD`` -> ref -> sha and never the working tree) reported NO skew while the live
gateway served a mixed revision -- a patched ``hermes_cli/*`` beside an unpatched ``agent/*`` --
and every dispatcher spawn died with ``cannot import name 'ADVISORY_SKILLS_ENV' from
'agent.skill_commands'`` for ~2h56m after the disk was repaired. A ref-only guard is blind to
exactly that window, so the consumer's skew signal must be a fingerprint of the CONTENT it
imported.

What is fingerprinted, and why it is bounded
--------------------------------------------
One digest over the ``.py`` set of the package trees a Hermes process actually loads
(:data:`SOURCE_PACKAGE_TREES`), each file contributing ``relpath`` + ``size`` + ``sha256(bytes)`` in
a stable (sorted) order. ``__pycache__`` is excluded: bytecode is a cache of these files, not
source. Bounded by construction -- a fixed set of directories, one filename suffix, ~1.6k files /
~32MB on this host, measured 0.79s cold -- so it is cheap enough to re-check once per dispatcher
tick.

``mtime_ns`` is NOT in the digest, and that is deliberate: the contract is "the content this process
imported still matches the content on disk", so a rewrite that restores byte-identical files must
read CLEAN. Modification time appears only in the memo key below -- a re-read hint, never evidence.

It is deliberately cheap on the happy path: the digest is memoized on a stat-only signature
(``relpath`` + ``size`` + ``mtime_ns``, measured 0.048s), so a repeat check on an UNCHANGED tree is
one stat walk and file bytes are re-read only when something moved. The known, accepted gap in that
memo is a write that changes content while leaving both size and mtime_ns identical to the byte --
which the incident's shape (and any real rewrite) does not do, and which the pre-fix signal could
not see at all.

The boot record (:func:`record_boot`) is the ONE record; the two carriers that already exist in
the live tree write through it -- ``gateway.code_skew.record_boot_fingerprint()`` in the gateway
process, and ``hermes_cli.tree_identity.record_dispatching_tree()`` (which stamps
:func:`content_fingerprint` into ``dispatcher_tree.json`` beside tree + pid + process_start).
Nothing here invents a second boot record.

Fail-open, always
-----------------
Every uncertain read returns ``None`` -- an absent tree, a non-git install, an IO error, no boot
record yet -- and ``None`` NEVER means skew. A guard that refuses on an unreadable read would
stop the fleet on a transient filesystem hiccup; the ref-only guard it backstops fails open the
same way.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

#: Package directories whose source a Hermes process loads at import time. The fingerprint covers
#: these and nothing else: a bounded, stable set is what keeps the check cheap enough for a tick.
SOURCE_PACKAGE_TREES: tuple[str, ...] = ("hermes_cli", "agent", "gateway", "tools")

#: Stable, greppable prefix of every fingerprint. A reader must be able to tell a content
#: fingerprint from a ``git:<ref>:<sha>`` one (the ref-only fingerprint this backstops).
FINGERPRINT_PREFIX = "tree-content:"

#: How many ``(root, stat-signature) -> digest`` entries to keep. One live tree per process is the
#: norm; the cap is a leak guard, not a tuning knob (no config key).
_MEMO_MAX_ENTRIES = 8


def default_root() -> Path:
    """The tree THIS module executes from -- the closure the boot record reasons about.

    Derived from this file's own location (never a hardcoded path), and deliberately a function so
    a caller -- or a test that must not touch the real checkout -- can point the fingerprint at
    another root instead of breaking the live one.
    """
    return Path(__file__).resolve().parent.parent


def _source_files(root: Path) -> list[tuple[str, Path]]:
    """``(relative posix path, absolute path)`` for every source file under :data:`SOURCE_PACKAGE_TREES`.

    Sorted by relative path so the digest is order-stable across filesystems, and tolerant of a
    missing package directory (a tree that carries only some of the four is still fingerprinted).
    """
    found: list[tuple[str, Path]] = []
    for package in SOURCE_PACKAGE_TREES:
        base = root / package
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
            for filename in sorted(filenames):
                if not filename.endswith(".py"):
                    continue
                path = Path(dirpath) / filename
                try:
                    relpath = path.relative_to(root).as_posix()
                except ValueError:  # pragma: no cover - os.walk under root cannot escape it
                    relpath = str(path)
                found.append((relpath, path))
    found.sort(key=lambda item: item[0])
    return found


def _stat_signature_root(root: Path) -> Optional[tuple[str, list[tuple[str, Path]]]]:
    """``(stat signature, source files)`` -- one walk, cheap enough to run every tick, or None."""
    try:
        files = _source_files(root)
    except OSError:
        return None
    digest = hashlib.sha256()
    for relpath, path in files:
        try:
            stat = path.stat()
        except OSError:
            return None
        digest.update(f"{relpath}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest(), files


def _digest_files(files: list[tuple[str, Path]], root: Path) -> Optional[str]:
    """The content digest over ``files``; ``None`` when any file cannot be read (fail open).

    ``mtime_ns`` is deliberately NOT part of this digest. The contract is "the content this process
    imported still matches the content on disk", so a rewrite that restores byte-identical files
    (a `git checkout`, a re-applied override) must read CLEAN -- mtime is a re-read hint for the
    memo, not evidence of drift. Path + size + content hash is the whole signal.
    """
    digest = hashlib.sha256()
    for relpath, path in files:
        try:
            with open(path, "rb") as handle:
                content = hashlib.sha256(handle.read()).digest()
            size = path.stat().st_size
        except OSError:
            return None
        digest.update(f"{relpath}|{size}|{content.hex()}\n".encode())
    return digest.hexdigest()


#: ``str(root) -> (stat signature, content digest)``. Bounded and process-local by design: the
#: memo is a cost optimization for repeated ticks, never the source of truth (a changed stat
#: signature always recomputes the bytes).
_content_memo: dict[str, tuple[str, str]] = {}


def content_fingerprint(root: Optional[Path] = None) -> Optional[str]:
    """``tree-content:<sha256>`` for the source under ``root`` (default: this tree), else ``None``.

    Two-tier on purpose: a stat-only signature is computed first and short-circuits a repeat call
    on an unchanged tree, and the file bytes are re-read ONLY when that signature moved. So the
    steady-state cost of a per-tick check is one stat walk, while the thing recorded and compared
    is always a CONTENT digest -- which is the load-bearing property (a working-tree edit with
    ``HEAD`` unchanged must be visible, and is).
    """
    resolved = Path(root) if root is not None else default_root()
    try:
        resolved = resolved.resolve()
    except OSError:  # pragma: no cover - resolve only fails on a broken link chain
        return None
    probed = _stat_signature_root(resolved)
    if probed is None:
        return None
    stat_signature, files = probed
    key = str(resolved)
    memoized = _content_memo.get(key)
    if memoized is not None and memoized[0] == stat_signature:
        return f"{FINGERPRINT_PREFIX}{memoized[1]}"
    digest = _digest_files(files, resolved)
    if digest is None:
        return None
    if len(_content_memo) >= _MEMO_MAX_ENTRIES and key not in _content_memo:
        _content_memo.clear()
    _content_memo[key] = (stat_signature, digest)
    return f"{FINGERPRINT_PREFIX}{digest}"


def short(fingerprint: Optional[str]) -> str:
    """Compact label for a fingerprint -- the first 12 hex chars, or ``"unresolved"``."""
    if not fingerprint:
        return "unresolved"
    body = fingerprint.removeprefix(FINGERPRINT_PREFIX)
    body = body.split(":", 1)[-1]
    return body[:12] if body else "unresolved"


# --- the ONE boot record ----------------------------------------------------------------

#: ``{"fingerprint", "pid", "tree", "recorded_at"}`` -- set once per process by :func:`record_boot`
#: (through ``gateway.code_skew.record_boot_fingerprint``) and read by the consumer gate.
_boot_record: Optional[dict] = None


def record_boot(root: Optional[Path] = None, *, now: Optional[int] = None) -> Optional[dict]:
    """Snapshot this process's tree content as the boot record (idempotent).

    Idempotent by design: the FIRST recording wins, because the whole point is to remember the
    revision this process IMPORTED. A second call in the same process must not overwrite it with
    a tree that has already moved -- that is the skew we exist to detect.
    """
    global _boot_record
    if _boot_record is not None:
        return _boot_record
    resolved = Path(root) if root is not None else default_root()
    fingerprint = content_fingerprint(resolved)
    if fingerprint is None:
        return None
    _boot_record = {
        "fingerprint": fingerprint,
        "pid": os.getpid(),
        "tree": str(resolved),
        "recorded_at": int(time.time() if now is None else now),
    }
    return _boot_record


def boot_record() -> Optional[dict]:
    """The recorded boot identity, or ``None`` when this process never recorded one."""
    return _boot_record


def reset_boot() -> None:
    """Drop the boot record (test helper; no production caller needs this)."""
    global _boot_record
    _boot_record = None


def detect_skew(root: Optional[Path] = None) -> Optional[tuple[str, str]]:
    """``(boot_fingerprint, disk_fingerprint)`` when the CONTENT moved since boot, else ``None``.

    ``None`` covers every fail-open case: no boot record (a process that never adopted the guard),
    an unreadable tree, and a tree whose git ref moved without its content moving (the ref half of
    the pair lives in ``gateway.code_skew`` and is consulted separately, as a fast path only).
    """
    record = _boot_record
    if record is None or not record.get("fingerprint"):
        return None
    current = content_fingerprint(root)
    if current is None or current == record["fingerprint"]:
        return None
    return str(record["fingerprint"]), current
