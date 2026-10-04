"""Scratch-dir retention: idle detection plus the reaping an idle tree needs before it goes.

Deleting an idle ``cache/scratch/<entry>`` is not enough on its own: a lane's e2e run leaves
headless browsers whose cwd was inside the tree (they survived for days with a ``(deleted)``
cwd), and a repo whose linked worktree lived in the tree keeps a dangling registration until
someone runs ``git worktree prune``. Both are reaped here, right before ``rmtree``.

Every departure leaves a record: each pruned entry logs its name, its newest mtime, and its
size where cheap, each reaped process its pid and cmdline, and the boot caller logs the
summary count instead of dropping it. A delete with no trace was the #132401 lesson: the
prune itself is policy, the silence was the defect.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# How long a TERMed process gets before KILL; browsers exit well within this.
_REAP_GRACE_SECONDS = 3.0
# ``.git`` files (linked worktrees) are looked for this deep; lanes nest repo/tree/subtree.
_GIT_FILE_MAX_DEPTH = 4
# argv flags whose FOLLOWING token is a credential value (``--password`` / ``-p`` VALUE).
# Command lines routinely carry secrets as separate arguments; the audit record must
# never persist them raw (PR review: split-form credentials reach the durable log).
_SECRET_ARGV_FLAGS = (
    "--password", "-p", "--pass", "--passwd", "--secret", "--api-key", "--apikey",
    "--token", "--auth-token", "--access-token", "--client-secret", "--private-key",
    "--aws-secret-access-key", "--secret-access-key", "-k", "--key",
)
_SECRET_ARGV_KEYS = ("password", "passwd", "secret", "token", "api-key", "apikey", "api_key")


def _redact_cmdline(argv: list[str]) -> str:
    """Render an argv for the audit record with credential-shaped arguments masked.

    Three shapes are masked, at capture, before ANY sink: ``--password VALUE``
    (value as the following token), ``--password=VALUE`` (inline form), and
    env-style ``KEY=VALUE`` tokens whose key carries a secret-bearing word.
    Mirrors ``agent.redact``'s mask-shape (head+tail) without importing it —
    the reap path must stay stdlib-only and import-safe before ``setup_logging``.
    """
    out: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if "=" in tok:
            key, _, value = tok.partition("=")
            base = key.lower().lstrip("-").replace("_", "-")
            if key in _SECRET_ARGV_FLAGS or any(w in base for w in _SECRET_ARGV_KEYS):
                out.append(f"{key}={mask_secret(value)}")
            else:
                out.append(tok)
            i += 1
            continue
        if tok in _SECRET_ARGV_FLAGS:
            if i + 1 < len(argv):
                out.append(f"{tok} {mask_secret(argv[i + 1])}")
                i += 2
                continue
            out.append(f"{tok}=<missing>")
            i += 1
            continue
        base = tok.lower().lstrip("-").replace("_", "-")
        if any(w in base for w in _SECRET_ARGV_KEYS) and i + 1 < len(argv):
            out.append(f"{tok} {mask_secret(argv[i + 1])}")
            i += 2
            continue
        out.append(tok)
        i += 1
    return " ".join(out)[:200]


def _mask(value: str) -> str:
    """Head+tail mask, ``agent.redact.mask_secret``-shaped, stdlib-local."""
    if len(value) <= 8:
        return "****"
    return f"{value[:4]}...{value[-4:]}" if len(value) <= 64 else f"{value[:4]}...****"


def mask_secret(value: str) -> str:
    """Public mask used by ``_redact_cmdline`` (kept stdlib: import-safe pre-logging)."""
    return _mask(value)


def _audit_log_path() -> Path:
    """Durable audit sink: ``<home>/logs/scratch-prune.log`` — the platform's real
    hermes home (same resolver as every other hermes path), not a guess."""
    try:
        from hermes_constants import get_process_hermes_home  # lazy: avoids import cycle

        return get_process_hermes_home() / "logs" / "scratch-prune.log"
    except Exception:  # noqa: BLE001 — resolver unavailable: last-resort literal
        home = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
        return home / "logs" / "scratch-prune.log"


def audit_info(log: logging.Logger, msg: str, *args: object) -> None:
    """Audit record at INFO with a durable fallback sink (#132401).

    teknium1's ask was that the records "land in ``~/.hermes/logs/``" — but the
    boot-time prune can fire before ``setup_logging`` installs the file handler,
    so a bare ``logger.info`` is delivery-by-hope on exactly the path that
    matters most. This emits through the logger as asked, and when nothing at
    the root would capture INFO (the early-boot case) it appends the record to
    ``<home>/logs/scratch-prune.log`` directly: an audit line either reaches
    ``agent.log`` or the audit file, never nowhere. Never raises — the prune
    must not fail for want of a log line. The fallback text passes through
    ``redact_sensitive_text`` when importable (the fallback file has no
    formatter); ``_redact_cmdline`` has already masked argv-shaped secrets
    regardless, so the stdlib-only early-boot path is never raw.
    """
    try:
        log.info(msg, *args)
        # Delivery check: a handler at INFO level counts only if INFO would actually
        # be PROCESSED — Python's logger-level filter runs BEFORE handlers, so a root
        # logger above INFO swallows records even when a low-level handler waits.
        root = logging.getLogger()
        root_will_process = logging.INFO >= root.getEffectiveLevel()
        sink_live = root_will_process and any(
            h.level <= logging.INFO for h in root.handlers
        )
        if sink_live:
            return  # the log file sink is live; no fallback duplication
        text = (msg % args) if args else msg
        text = _redact_for_disk(text)
        path = _audit_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Appends are one short line per rare event; POSIX O_APPEND makes each
        # atomic. Windows MSVCRT appends are seek-then-write (weaker under
        # concurrent prunes) — accepted for now given the event rarity.
        with open(path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write("%s\t%s\n" % (time.strftime("%Y-%m-%dT%H:%M:%S"), text))
    except Exception:  # noqa: BLE001 — audit must never break the prune
        pass


def _redact_for_disk(text: str) -> str:
    """Redact before the fallback sink: argv secrets were masked at capture
    (``_redact_cmdline``); this adds the repo's own redactor when importable so
    formatter-level coverage extends to the file the formatter never sees."""
    try:
        from agent.redact import redact_sensitive_text  # lazy: post-boot import is safe

        return redact_sensitive_text(text)
    except Exception:  # noqa: BLE001 — stdlib-only environments (early boot)
        return text


def subtree_touched_since(path: Path, cutoff: float) -> bool:
    """True when *path* or anything beneath it has an mtime at or after *cutoff*.

    Thin alias for the first element of :func:`_scan_entry_idleness` — one walk is the
    primitive, the audit fields ride the same stat calls. See there for the exact
    semantics (early stop, no symlink following, unreadable = kept).
    """
    return _scan_entry_idleness(path, cutoff)[0]


def _scan_entry_idleness(path: Path, cutoff: float) -> tuple[bool, float | None, int | None]:
    """One walk, three answers: (touched_since_cutoff, newest_mtime, bytes).

    Early-stops at the first recent mtime — a live tree costs one hit (bytes stay
    ``None``: untouched entries are not audited). A walk that completes returns the
    newest mtime seen and the byte total collected from the *same* stat calls — the
    audit record never costs a second walk of a doomed tree, the #132401 review's
    boot-cost objection answered by construction. Symlinks are never followed: a link
    into the repo would make the target's activity keep the entry alive. An unreadable
    entry reports ``(True, None, None)``: an incomplete scan cannot establish that it
    is idle, so it is kept (the pinned scan-failure semantics). The bytes figure is
    approximate by design: file bytes plus the root entry's own inode size.
    """
    try:
        st = os.lstat(path)
    except OSError:
        return True, None, None
    if st.st_mtime >= cutoff:
        return True, None, None
    if not path.is_dir() or path.is_symlink():
        return False, st.st_mtime, st.st_size
    newest = st.st_mtime
    total = st.st_size
    stack = [str(path)]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for child in it:
                    try:
                        cst = child.stat(follow_symlinks=False)
                    except OSError:
                        return True, None, None
                    if cst.st_mtime >= cutoff:
                        return True, None, None
                    if cst.st_mtime > newest:
                        newest = cst.st_mtime
                    if child.is_dir(follow_symlinks=False):
                        stack.append(child.path)
                    else:
                        total += cst.st_size
        except OSError:
            return True, None, None
    return False, newest, total


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _own_lineage() -> set[int]:
    """This process and its ancestors: never reap the shell that is running the prune."""
    import psutil

    pids: set[int] = set()
    try:
        proc = psutil.Process()
        while proc is not None and proc.pid not in pids:
            pids.add(proc.pid)
            proc = proc.parent()
    except (psutil.Error, OSError):
        pass
    return pids


def reap_processes_rooted_in(scratch_root: Path, doomed: list[Path]) -> int:
    """TERM (then KILL) same-user processes whose cwd is inside an entry about to be
    pruned, or is a path under *scratch_root* that no longer exists. Returns the count.

    Only the cwd is consulted: a process merely holding a file open inside scratch (an
    editor, a log tail) is not ours to kill, but one *living* in a directory we are
    about to delete, or in one already gone, has nothing left to run for.
    """
    import psutil

    root = os.path.realpath(str(scratch_root))
    targets = [os.path.realpath(str(p)) for p in doomed]
    skip = _own_lineage()
    uid = os.getuid() if hasattr(os, "getuid") else None
    victims: list[psutil.Process] = []
    for proc in psutil.process_iter(["pid"]):
        if proc.pid in skip:
            continue
        try:
            if uid is not None and proc.uids().real != uid:
                continue
            cwd = proc.cwd()
        except (psutil.Error, OSError):
            continue
        if not cwd:
            continue
        deleted = cwd.endswith(" (deleted)")
        cwd_path = cwd[: -len(" (deleted)")] if deleted else cwd
        if not _under(cwd_path, root):
            continue
        if deleted or not os.path.exists(cwd_path) or any(_under(cwd_path, t) for t in targets):
            victims.append(proc)
    if not victims:
        return 0
    # Evidence is captured BEFORE the kill: a terminated process's cmdline is gone,
    # so the audit record reads from this snapshot, not from post-mortem queries.
    # argv is redacted at capture: split-form credentials (``--password`` VALUE)
    # never reach any sink raw (PR review round 3).
    evidence: list[tuple[int, str]] = []
    for proc in victims:
        try:
            evidence.append((proc.pid, _redact_cmdline(list(proc.cmdline() or []))))
        except (psutil.Error, OSError):
            evidence.append((proc.pid, "<unavailable>"))
    for proc in victims:
        try:
            proc.terminate()
        except (psutil.Error, OSError):
            continue
    _, alive = psutil.wait_procs(victims, timeout=_REAP_GRACE_SECONDS)
    for proc in alive:
        try:
            proc.kill()
        except (psutil.Error, OSError):
            continue
    # Confirmed exits only: the record distinguishes what the reap actually achieved.
    # A victim that resisted TERM and KILL is reported as failed, never as reaped —
    # an audit receipt must not claim a kill that did not happen (PR review round 3).
    confirmed = 0
    for pid, cmdline in evidence:
        proc = next((p for p in victims if p.pid == pid), None)
        exited = True
        if proc is not None:
            try:
                exited = not (psutil.pid_exists(pid) and psutil.Process(pid).is_running())
            except (psutil.Error, OSError):
                exited = False  # cannot establish the exit: do not claim it
        if exited:
            confirmed += 1
            audit_info(logger, "scratch prune: reaped pid=%d cmdline=%s", pid, cmdline)
        else:
            audit_info(logger, "scratch prune: reap failed pid=%d cmdline=%s", pid, cmdline)
    audit_info(
        logger, "scratch prune: reaped %d process(es) rooted in pruned entries", confirmed
    )
    return confirmed


def _linked_worktree_repos(entry: Path) -> set[str]:
    """Repos whose linked worktrees live inside *entry* (``.git`` FILES, ``gitdir: <repo>/.git/worktrees/<n>``)."""
    repos: set[str] = set()
    stack = [(str(entry), 0)]
    while stack:
        current, depth = stack.pop()
        try:
            with os.scandir(current) as it:
                for child in it:
                    if child.name == ".git" and child.is_file(follow_symlinks=False):
                        try:
                            line = Path(child.path).read_text(encoding="utf-8", errors="replace").strip()
                        except OSError:
                            continue
                        if line.startswith("gitdir:"):
                            gitdir = Path(line[len("gitdir:"):].strip())
                            # <repo>/.git/worktrees/<name> -> <repo>
                            if gitdir.parent.name == "worktrees" and gitdir.parent.parent.name == ".git":
                                repos.add(str(gitdir.parent.parent.parent))
                    elif child.is_dir(follow_symlinks=False) and depth < _GIT_FILE_MAX_DEPTH \
                            and child.name not in ("node_modules", ".venv", "venv"):
                        stack.append((child.path, depth + 1))
        except OSError:
            continue
    return repos


def release_git_worktrees(repos: set[str]) -> None:
    """``git worktree prune`` in each repo: drops registrations whose tree we just deleted."""
    for repo in sorted(repos):
        if not os.path.isdir(repo):
            continue
        try:
            subprocess.run(
                ["git", "-C", repo, "worktree", "prune"],
                stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=15, check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("git worktree prune in %s failed: %s", repo, exc)


def prune_idle_entries(root: Path, max_idle_hours: float, skip_names: frozenset[str]) -> int:
    """Delete top-level entries of *root* with no write anywhere in their subtree for
    *max_idle_hours*, reaping processes and worktree registrations rooted in them first.
    Returns the count removed.

    Every departure leaves a per-entry record at INFO — name, newest mtime, and size —
    so a delete is never silent (#132401): the count alone hid five multi-day work
    products being destroyed with no log, no quarantine, no trace. The reap detail
    (pid + cmdline) and the boot caller's summary log alongside.
    """
    cutoff = time.time() - max_idle_hours * 3600
    try:
        entries = [e for e in root.iterdir() if e.name not in skip_names]
    except OSError:
        return 0
    doomed: list[tuple[Path, float | None, int | None]] = []
    for e in entries:
        touched, newest, bytes_ = _scan_entry_idleness(e, cutoff)
        if not touched:
            doomed.append((e, newest, bytes_))
    # Runs even with nothing to delete: orphans whose cwd was removed by an earlier pass
    # (or by hand) are found by the deleted-cwd rule, detail via the reap's own record.
    try:
        reap_processes_rooted_in(root, [e for e, _, _ in doomed])
    except Exception as exc:  # psutil missing or restricted host: the deletion still proceeds
        logger.debug("scratch prune: process reap skipped: %s", exc)
    if not doomed:
        return 0
    repos: set[str] = set()
    removed = 0
    for entry, newest, bytes_ in doomed:
        kind = "dir" if (entry.is_dir() and not entry.is_symlink()) else "file"
        if kind == "dir":
            repos |= _linked_worktree_repos(entry)
        # Last-moment re-validation (#132401 C1/F1): the doomed list was snapshotted
        # before the reap ran, and a resumed writer (cwd outside scratch) can land
        # fresh work in that window. Anything that became touched since selection is
        # rescued here, not deleted — the selection snapshot is a candidate list,
        # never a verdict. Runs after the worktree scan so the re-check sits as
        # close to the delete as the loop allows.
        touched, newest, bytes_ = _scan_entry_idleness(entry, cutoff)
        if touched:
            if not entry.exists():
                audit_info(
                    logger,
                    "scratch prune: entry=%r vanished since selection",
                    entry.name,
                )
            else:
                audit_info(
                    logger,
                    "scratch prune: rescued entry=%r — touched since selection, kept",
                    entry.name,
                )
            continue
        try:
            if kind == "dir":
                repos |= _linked_worktree_repos(entry)
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink()
        except OSError as exc:
            # Attempted, failed outright (permissions, vanished mid-run): the record
            # is the audit; the count only ever claims confirmed removals.
            audit_info(
                logger,
                "scratch prune: removal failed entry=%r kind=%s error=%s",
                entry.name, kind, exc,
            )
            continue
        if entry.exists():
            # ``rmtree(ignore_errors=True)`` can leave residue — a partial removal is
            # not a removal. Recorded, never counted (#132401 review round 2).
            audit_info(
                logger,
                "scratch prune: removal left residue entry=%r kind=%s bytes=%d newest_mtime=%.0f",
                entry.name, kind, bytes_ or 0, newest or 0.0,
            )
            continue
        audit_info(
            logger,
            "scratch prune: removed entry=%r kind=%s bytes=%d newest_mtime=%.0f",
            entry.name, kind, bytes_ or 0, newest or 0.0,
        )
        removed += 1
    release_git_worktrees(repos)
    return removed
