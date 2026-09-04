"""Boot-time post-update bootstrap.

Every install kind (git checkout, desktop bundled payload, docker, nix)
compares two per-install facts at boot:

* current identity — the commit this install IS: ``install-stamp.json``
  for sealed trees, git HEAD for checkouts. Reading it is a couple of
  file reads (plus one rev-parse for checkouts).
* last-known identity — the commit this install last bootstrapped,
  recorded under ``installs/<sha16>/bootstrap/`` keyed by the canonical
  install root.

Equal → nothing happens (the fast path, ~2 ms). Different → run the
idempotent post-update steps from ``hermes_cli.post_update`` under a
single-flight lock, then record the new identity.

Two records, one per step scope:

* home record — ``bootstrap/<profile>.json``. Gates home-scoped steps.
  HERMES_HOME moves per profile, so each profile bootstraps its own
  state once per code change.
* machine record — ``bootstrap/machine.json``. Every profile resolves
  the same file, so machine-global steps run once per machine per code
  change and the record's lock serializes concurrent profile boots.

The records are an optimization, never the correctness layer: every step
is idempotent and self-gating, so a deleted record costs one redundant
slow path, nothing more.

Ported from the restack branch's boot_bootstrap, rewritten onto this
branch's vocabulary: stamps via ``hermes_cli.steward``, managed tools via
``pm`` (facts.json ledger) instead of the retired ``installation`` package.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

RECORD_SCHEMA_VERSION = 1
LOCK_STALE_SECONDS = 600


def default_project_root() -> Path:
    """The tree this code runs from: HERMES_INSTALL_ROOT for sealed
    artifacts whose stamp lives outside the package dir (the override
    hermes_cli.version_info honours), the code root otherwise."""
    env = os.environ.get("HERMES_INSTALL_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# current identity
# ---------------------------------------------------------------------------

def read_git_head(root: Path) -> str | None:
    """The commit SHA of the checkout at ``root``.

    Asks git, rather than reimplementing it: parsing ``.git`` by hand
    (worktree gitfiles, symbolic HEAD, packed-refs, commondir) is a
    reimplementation of ``git rev-parse HEAD`` that reftable breaks
    wholesale. The managed git comes first, a PATH git second; with
    neither, the answer is None and boot carries on — fail-open.

    Cost: one ~10ms subprocess. Only checkouts pay it — a sealed tree
    reads its stamp and never gets here — and it happens once per boot.
    """
    git = _git_binary()
    if git is None:
        return None
    try:
        out = subprocess.run(
            [git, "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    sha = out.stdout.strip()
    return sha if len(sha) >= 7 else None


def _git_binary() -> str | None:
    """The git to run, or None when this machine has no usable one.

    pm's pinned git first (the win32 Git-for-Windows package — where it
    is installed, it is the one whose bash contract the rest of the tree
    already trusts), then a PATH git. The imports are local: this module
    is loaded early enough that a module-level import would widen the
    boot import graph for a lookup most platforms answer from PATH.
    """
    try:
        from pm.ensure import _facts, _store
        from pm.registry import get_package
        from pm.store import current_target

        fact = _facts().get("git")
        if fact is not None:
            binary = get_package("git").binary(
                _store().entry(fact["entry"]), current_target()
            )
            if binary is not None and binary.is_file():
                return str(binary)
    except Exception as exc:  # noqa: BLE001 — boot must not die on a lookup
        logger.debug("pm git lookup failed: %s", exc)
    import shutil

    return shutil.which("git")


def current_install_identity(project_root: Path) -> str | None:
    """What code this install is: stamp commit for sealed trees, git HEAD
    for checkouts, None for broken trees (never bootstrap, never write)."""
    from hermes_cli.steward import read_install_stamp

    root = Path(project_root)
    if (root / ".git").exists():
        return read_git_head(root)
    stamp = read_install_stamp(root)
    commit = stamp.get("commit")
    if isinstance(commit, str) and len(commit) >= 7:
        return commit
    # A tagless/commitless stamp is a broken artifact; the tag alone is
    # accepted as a weaker identity (bundled artifacts always carry one).
    tag = stamp.get("tag")
    return tag if isinstance(tag, str) and tag else None


# ---------------------------------------------------------------------------
# the per-install state folder: installs/<SHA16>/ under the DEFAULT home
# ---------------------------------------------------------------------------

def _install_key(project_root: Path) -> str:
    try:
        canonical = str(Path(project_root).resolve())
    except OSError:
        canonical = str(project_root)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


INSTALLS_DIR_NAME = "installs"


def installs_root() -> Path:
    """The parent of every per-install state folder.

    Anchored to the DEFAULT home, not the active profile home: profiles
    share one folder per install, with per-profile bootstrap records as
    files INSIDE it (bootstrap/<profile>.json) rather than per-profile
    folders. That keeps profile semantics while collapsing the anchor
    count to one.
    """
    from hermes_cli.profiles import _get_default_hermes_home

    return _get_default_hermes_home() / INSTALLS_DIR_NAME


def install_state_dir(project_root: Path) -> Path:
    """``installs/<SHA16>/`` for this install. Derivation only — no I/O."""
    return installs_root() / _install_key(project_root)


def ensure_install_dir(project_root: Path) -> Path:
    """The state folder, created with its identity record on first touch.

    install.json is the REVERSE map (sha16 → canonical root) that makes
    orphan GC possible: `hermes doctor` enumerates installs/*/install.json
    and flags entries whose recorded root no longer exists. Written once,
    under the same single-flight lock the records use; the steward comes
    from hermes_cli.steward so the record says who owns the tree, not who
    touched it first.
    """
    state = install_state_dir(project_root)
    marker = state / "install.json"
    if marker.is_file():
        return state
    state.mkdir(parents=True, exist_ok=True)
    lock = _RecordLock(state / ".install-json.lock")
    if not lock.acquire():
        return state  # someone else is writing it right now — theirs wins
    try:
        if not marker.is_file():
            from datetime import datetime, timezone

            from hermes_cli.steward import sealed_steward

            steward = sealed_steward(Path(project_root))
            payload = {
                "root": str(Path(project_root).resolve()),
                "steward": steward if steward is not None else "checkout",
                "firstSeen": datetime.now(timezone.utc).isoformat(),
            }
            tmp = marker.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, marker)
    finally:
        lock.release()
    return state


def orphaned_installs() -> list[tuple[Path, str]]:
    """State folders whose recorded root no longer exists.

    ``(folder, recorded_root)`` pairs for `hermes doctor`'s sweep. A
    folder without a readable install.json is orphaned by definition —
    nothing can ever claim it again, because claiming goes through
    ensure_install_dir which writes the record first.
    """
    root = installs_root()
    if not root.is_dir():
        return []
    orphans: list[tuple[Path, str]] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        try:
            recorded = json.loads(
                (entry / "install.json").read_text(encoding="utf-8-sig")
            ).get("root", "")
        except (OSError, ValueError):
            orphans.append((entry, "<unreadable install.json>"))
            continue
        if not recorded or not Path(recorded).exists():
            orphans.append((entry, recorded or "<empty>"))
    return orphans


def orphaned_store_entries() -> list[tuple[Path, int]]:
    """Tool-store entries the installed-state ledger no longer references.

    ``(entry_dir, size_bytes)`` pairs for `hermes doctor`'s sweep. pm's
    facts.json is the only authority consulted — the same ledger
    ``pm.ensure`` resolves by, so this can never flag an entry pm would
    still hand out. An entry is REFERENCED when facts records it (tool
    entries) or when it is a ``fetch-*`` archive cache entry backing a
    referenced install. Everything else is bytes no lookup can ever
    return: superseded versions left behind by pin bumps.

    Doubt errs toward KEEP: a facts file that exists but cannot be read
    aborts the whole sweep (empty result), because its references are
    unknowable and any entry might be one of them. No facts file at all
    means pm never installed anything — nothing is referenced, but there
    is also nothing to GC against, so the sweep reports nothing.
    """
    from pm import paths as pm_paths
    from pm.lock import Facts

    store = pm_paths.store_root()
    if not store.is_dir():
        return []
    facts_file = pm_paths.facts_path()
    if not facts_file.is_file():
        return []
    try:
        raw = json.loads(facts_file.read_text(encoding="utf-8-sig"))
        if not isinstance(raw, dict):
            return []
    except (OSError, ValueError):
        # Unreadable ledger: references unknowable — keep everything.
        return []
    referenced = Facts(facts_file).entries_in_use()

    orphans: list[tuple[Path, int]] = []
    for entry in sorted(store.iterdir()):
        # Scratch dirs (.staging-*), the ledger itself, and stray files
        # are pm's own cleanup problem, never GC candidates. fetch-*
        # archive cache entries are content-addressed and cheap to keep;
        # skip them too (re-fetch avoidance is their whole point).
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        if entry.name.startswith("fetch-"):
            continue
        if entry.name in referenced:
            continue
        size = 0
        for f in entry.rglob("*"):
            try:
                if f.is_file() and not f.is_symlink():
                    size += f.stat().st_size
            except OSError:
                continue
        orphans.append((entry, size))
    return orphans


def record_path(project_root: Path, scope: str) -> Path:
    """Where the last-known record for ``project_root`` lives.

    Both scopes live INSIDE the per-install state folder:
    ``bootstrap/machine.json`` for machine scope, ``bootstrap/<profile>.json``
    for home scope — the per-profile semantics ride the FILENAME, not a
    per-profile anchor directory.
    """
    if scope == "home":
        from hermes_cli.profiles import get_active_profile_name

        name = get_active_profile_name() or "default"
        filename = f"{name}.json"
    elif scope == "machine":
        filename = "machine.json"
    else:
        raise ValueError(f"unknown record scope: {scope!r}")
    return install_state_dir(project_root) / "bootstrap" / filename


def read_last_known(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_record(path: Path, identity: str, results: dict) -> None:
    payload = {
        "schemaVersion": RECORD_SCHEMA_VERSION,
        "identity": identity,
        "bootstrappedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_record(project_root: Path, scope: str, identity: str, results: dict | None = None) -> None:
    """Record ``identity`` as bootstrapped. Also used by the update phase
    after it runs the steps itself, so the next boot skips."""
    _write_record(record_path(project_root, scope), identity, results or {})


def needs_bootstrap(project_root: Path, scope: str) -> str | None:
    """The new identity when this install changed since its last bootstrap,
    else None. None identity (broken tree) never bootstraps."""
    identity = current_install_identity(project_root)
    if not identity:
        return None
    known = read_last_known(record_path(project_root, scope))
    if known.get("identity") == identity:
        return None
    return identity


# ---------------------------------------------------------------------------
# single-flight lock
# ---------------------------------------------------------------------------

class _RecordLock:
    """O_CREAT|O_EXCL existence-as-mutex next to a record file.

    Losers skip (boot never waits on another process's bootstrap; the steps
    are idempotent, so a botched winner only costs redundant work later).
    A stale lock — older than LOCK_STALE_SECONDS — is broken and re-tried
    once: a crashed winner died before its record write, so re-running is
    correct.
    """

    def __init__(self, record: Path):
        self.path = record.with_name(record.name + ".lock")
        self.acquired = False

    def _try_create(self) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        except OSError:
            return False
        try:
            os.write(fd, json.dumps({"pid": os.getpid(), "startedAt": time.time()}).encode("utf-8"))
        finally:
            os.close(fd)
        return True

    def _is_stale(self) -> bool:
        try:
            body = json.loads(self.path.read_text(encoding="utf-8-sig"))
            started = float(body.get("startedAt", 0))
        except (OSError, ValueError):
            # Unreadable lock: age it by mtime instead.
            try:
                started = self.path.stat().st_mtime
            except OSError:
                return False
        return (time.time() - started) > LOCK_STALE_SECONDS

    def acquire(self) -> bool:
        if self._try_create():
            self.acquired = True
            return True
        if self._is_stale():
            try:
                self.path.unlink()
            except OSError:
                return False
            if self._try_create():
                self.acquired = True
                return True
        return False

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            self.path.unlink()
        except OSError:
            pass
        self.acquired = False


# ---------------------------------------------------------------------------
# the boot entry point
# ---------------------------------------------------------------------------

def _report_sealed_runtime_drift(project_root: Path) -> str | None:
    """Check a SEALED tree's managed tools against pm's lockfile, loudly.

    pm's artifact-time gates (bundle staging, docker build, nix check)
    are the wall; this is the boot-time backstop for artifacts assembled
    around them: every boot of a drifted sealed tree prints the problem
    list to stderr, so the drift is impossible to not-know about.

    Report, not refusal: this runs inside the never-raises boot path, and
    a sealed gateway that boots on stale tools is degraded — but a
    gateway that refuses to boot over a tool version is DOWN, remotely,
    with the fix (rebuild the artifact) out of the machine's own reach.

    Returns the message when drift was found (for the boot summary), None
    otherwise. Checkouts return None without reading anything — they
    provision on demand and drift is their normal, self-healing state.
    """
    root = Path(project_root)
    if (root / ".git").exists():
        return None
    try:
        from hermes_cli.steward import sealed_steward

        steward = sealed_steward(root)
        if steward is None:
            return None
        import pm

        problems = pm.check()
    except Exception as exc:  # noqa: BLE001 — a backstop must not become a gate
        logger.debug("sealed runtime drift check failed: %s", exc)
        return None
    if not problems:
        return None
    message = (
        f"this {steward}-managed install's tools drifted from its pin table: "
        + "; ".join(problems)
        + " — rebuild the artifact to fix"
    )
    print(f"\n✗ {message}\n", file=sys.stderr)
    return message


def run_boot_bootstrap(project_root: Path) -> dict:
    """Run due home- and machine-scoped steps for this install. Returns a
    summary dict (for tests/logs); use maybe_run_boot_bootstrap at call
    sites."""
    from hermes_cli import post_update

    summary: dict = {"home": "skipped", "machine": "skipped"}

    drift_message = _report_sealed_runtime_drift(Path(project_root))
    if drift_message:
        summary["sealed_runtime_drift"] = drift_message

    for scope, steps, deferred in (
        ("home", post_update.HOME_STEPS, False),
        ("machine", post_update.MACHINE_STEPS, True),
    ):
        identity = needs_bootstrap(project_root, scope)
        if not identity:
            continue
        record = record_path(project_root, scope)
        lock = _RecordLock(record)
        if not lock.acquire():
            summary[scope] = "lost-race"
            continue
        try:
            # Double-check under the lock: the previous holder may have
            # finished between our read and our acquire.
            if read_last_known(record).get("identity") == identity:
                summary[scope] = "done-by-other"
                continue
            logger.info(
                "post-update bootstrap (%s scope): code changed to %s, running steps",
                scope, identity[:12],
            )
            if deferred:
                # Slow machine steps (network installers) must not block
                # boot readiness: record first, then run detached. A crash
                # mid-step leaves the record written — intended: the record
                # gates "did we trigger for this identity", and the steps
                # re-gate themselves (pm.check stamp comparisons) next change.
                _write_record(record, identity, {"deferred": True})
                import threading

                threading.Thread(
                    target=post_update.run_steps,
                    args=(steps,),
                    name=f"hermes-bootstrap-{scope}",
                    daemon=True,
                ).start()
                summary[scope] = "deferred"
            else:
                results = post_update.run_steps(steps)
                _write_record(record, identity, results)
                summary[scope] = results
        finally:
            lock.release()
    return summary


def maybe_run_boot_bootstrap(project_root: Path) -> None:
    """The one call boot paths use. Never raises: a bootstrap problem must
    not stop the gateway/serve/CLI from starting."""
    try:
        run_boot_bootstrap(Path(project_root))
    except Exception as exc:
        logger.warning("boot bootstrap failed (continuing boot): %s", exc)
