"""Hermes update pipeline — extracted from ``hermes_cli/main.py``.

Mechanical move (main.py decomposition): ``_cmd_update_impl``, ``_cmd_update_check``
and every module-level helper used only by the update path, plus the update-only
constants they read. Function bodies are lifted verbatim; the only mechanical
change is that references to helpers/constants that STAY in ``hermes_cli.main``
(and to moved-but-test-patched siblings) are routed through ``_m()`` — a lazy
``hermes_cli.main`` reference — so existing call sites and test monkeypatches
that target ``hermes_cli.main.<name>`` (``PROJECT_ROOT``, ``_is_windows``,
``_run_pre_update_backup``, ...) keep working unchanged. ``main.py`` re-imports
every public-ish name from here (``# noqa: F401``) so the argparse wiring and
the test-patch surface still resolve on ``hermes_cli.main``.

Three self-contained closures nested inside ``_cmd_update_impl``
(``_print_items``, ``_wait_for_service_active``, ``_service_restart_sec``) were
hoisted to module level; they capture no enclosing state (verified via
``symtable``). ``_restart_one_systemd_gateway_unit``, ``_resolve_manage_cmd``
and ``_on_unit_timeout`` DO capture enclosing locals and stay nested,
byte-identical.

Imports are one-way: ``hermes_cli.main`` imports this module, never the reverse
at import time (``_m()`` resolves lazily at call time, when main.py is fully
loaded, so there is no import cycle).
"""

import hashlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import time as _time
from datetime import datetime
from pathlib import Path
from typing import Optional

from hermes_cli.config import get_hermes_home
from hermes_constants import venv_python_path

logger = logging.getLogger(__name__)


def _m():
    """Lazy ``hermes_cli.main`` reference.

    Lets callers keep patching ``hermes_cli.main.<helper>`` (the historical
    test surface) and have those patches reach this code path, and defers the
    import so ``hermes_cli.main`` -> ``hermes_cli.update_cmd`` stays one-way
    at import time.
    """
    from hermes_cli import main

    return main


_UPDATE_RUNTIME_RELOAD_MODULES = (
    "hermes_constants",
    "tools.environments.local",
    "tools.lazy_deps",
)

#: Package prefixes whose cached modules become stale the moment the checkout
#: changes under this process. Purged (not reloaded) by
#: ``_purge_stale_hermes_modules`` so any LATER import chain resolves against
#: fresh on-disk source only.
_STALE_PURGE_PREFIXES = (
    "hermes_cli",
    "gateway",
    "tools",
    "tui_gateway",
    "agent",
)

#: Modules that must survive the purge: they are (or are referenced by) the
#: code currently EXECUTING the update, so evicting them buys nothing — the
#: running frames keep their module objects alive regardless — and reloading
#: them mid-flight is the one genuinely unsafe move.
_STALE_PURGE_PROTECTED = frozenset(
    {
        "hermes_cli",
        "hermes_cli.main",
        "hermes_cli.update_cmd",
        "hermes_cli.hermes_logging",
    }
)


def _purge_stale_hermes_modules() -> None:
    """Evict every cached Hermes module after the checkout changed in-place.

    ``hermes update`` keeps running in the pre-pull Python process. The
    gateway auto-restart phase that follows does function-level
    ``from hermes_cli.gateway import ...`` — executing NEW source inside an
    OLD ``sys.modules`` world. The moment new source references a symbol
    that was added to an already-cached module, the import dies (2026-08-20
    field failure: freshly-pulled ``hermes_cli.gateway`` does
    ``from hermes_cli.cli_output import line_input``, but ``cli_output`` was
    cached from before d0132b582 which introduced ``line_input`` → the whole
    restart phase aborted and the gateway kept serving pre-update code).

    ``_UPDATE_RUNTIME_RELOAD_MODULES`` handled this per-symptom — three
    hardcoded module names, re-fixed every time a new module grew a new
    export. This is the class fix: drop EVERY cached module under the Hermes
    package prefixes so subsequent lazy imports rebuild a self-consistent,
    all-new module graph from the updated checkout. Old module objects
    referenced by the running updater frames stay alive and functional (a
    purge only removes the ``sys.modules`` cache entry); only genuinely
    executing modules are exempted, because reloading-in-place — not purging
    — is the operation that can pull code out from under a running frame.

    Best-effort: never raises.
    """
    try:
        import importlib

        importlib.invalidate_caches()
        purged = []
        for name in list(_m().sys.modules):
            if name in _STALE_PURGE_PROTECTED:
                continue
            if not name.startswith(_STALE_PURGE_PREFIXES):
                continue
            root = name.split(".", 1)[0]
            if root not in _STALE_PURGE_PREFIXES:
                # Prefix-string match caught an unrelated package
                # (e.g. ``gateway_foo``) — leave it alone.
                continue
            if _m().sys.modules.pop(name, None) is not None:
                purged.append(name)
        if purged:
            logger.debug(
                "Purged %d stale Hermes module(s) after checkout update", len(purged)
            )
    except Exception as exc:
        logger.debug("Could not purge stale Hermes modules: %s", exc)


def _reload_updated_runtime_modules() -> None:
    """Reload update-sensitive modules after the checkout changes in-place.

    ``hermes update`` keeps running in the pre-pull Python process. After a
    large update, modules already present in ``sys.modules`` can still expose
    old symbols even though their source files on disk are new. Refresh the
    small module set used by lazy-backend refresh before that step imports
    newly-updated code paths.
    """
    try:
        import importlib

        importlib.invalidate_caches()
        for module_name in _UPDATE_RUNTIME_RELOAD_MODULES:
            module = _m().sys.modules.get(module_name)
            if module is None:
                continue
            try:
                importlib.reload(module)
            except Exception as exc:
                logger.debug("Could not reload updated module %s: %s", module_name, exc)
    except Exception as exc:
        logger.debug("Could not refresh update runtime modules: %s", exc)


def _reload_config_modules() -> None:
    """Force-reload modules from disk after git pull.

    ``hermes update`` runs in the PRE-pull Python process. After ``git pull``
    updates the source files on disk, modules already in ``sys.modules``
    still hold the OLD code. Function-level imports return the cached module,
    so ``DEFAULT_CONFIG["_config_version"]`` is the OLD value and
    ``check_config_version()`` reports ``(33, 33)`` — "up to date" — even
    though the freshly-pulled code has v34 with a migration to run.

    This function force-reloads ``hermes_cli.config_defaults``,
    ``hermes_cli.config``, and ``hermes_cli.config_migrations`` from disk
    so subsequent imports read the UPDATED code.

    It also reloads ``hermes_cli._subprocess_compat`` and
    ``hermes_cli.dashboard_procs`` so that post-update dashboard cleanup
    (``_finish_dashboard_update_cleanup`` → ``_scan_dashboard_processes``)
    uses the freshly-pulled code. Without this, a new symbol added to
    ``_subprocess_compat`` (e.g. ``bounded_probe_run``) is invisible to the
    cached module object, causing ``ImportError`` during the cleanup step
    that runs later in the same process.
    """
    import importlib

    importlib.invalidate_caches()
    for mod_name in (
        "hermes_cli.config_defaults",
        "hermes_cli.config",
        "hermes_cli.config_migrations",
        "hermes_cli._subprocess_compat",
        "hermes_cli.dashboard_procs",
    ):
        mod = sys.modules.get(mod_name)
        if mod is not None:
            try:
                importlib.reload(mod)
            except Exception as exc:
                logger.debug("Could not reload %s for fresh post-update code: %s", mod_name, exc)


def _run_config_check_fresh() -> tuple:
    """Check config version using freshly-reloaded modules.

    See ``_reload_config_modules`` for why this is necessary.
    Returns ``(current_ver, latest_ver)``.
    """
    _reload_config_modules()
    from hermes_cli.config import check_config_version

    return check_config_version()


def _run_migrate_config_fresh(*, interactive: bool = False, quiet: bool = False) -> dict:
    """Run config migration using freshly-reloaded modules.

    See ``_reload_config_modules`` for why this is necessary.
    Returns the migration results dict.
    """
    _reload_config_modules()
    from hermes_cli.config import migrate_config

    return migrate_config(interactive=interactive, quiet=quiet)


def _migrate_sibling_profile_configs() -> list[tuple[str, int, int]]:
    """Migrate every SIBLING profile's config.yaml to the current version.

    #91277 Phase 2 (fleet-wide config migration; #20438/#54926/#79048): the
    shared checkout serves every profile, but ``hermes update`` historically
    migrated only the active profile's config — siblings drifted versions
    until their gateway hit a config the new code couldn't read.

    Per profile home (skipping the active one, already migrated by the
    caller): scope config reads/writes via the context-local HERMES_HOME
    override (thread-safe — never ``os.environ``), check the version, and
    run the NON-INTERACTIVE, quiet migration. Prompt-requiring settings are
    left for the profile's own next interactive session, identical to the
    gateway-mode contract for the active profile.

    Returns ``[(profile_name, from_version, to_version), ...]`` for profiles
    actually migrated. Never raises; a failing profile is skipped (its own
    startup migration remains the fallback).
    """
    migrated: list[tuple[str, int, int]] = []
    try:
        from hermes_constants import (
            get_process_hermes_home,
            reset_hermes_home_override,
            set_hermes_home_override,
        )
        from hermes_cli.profiles import _get_profiles_root, _PROFILE_ID_RE

        active_home = get_process_hermes_home()
        root = _get_profiles_root()
        if not root.is_dir():
            return migrated
        for entry in sorted(root.iterdir()):
            if not entry.is_dir() or not _PROFILE_ID_RE.match(entry.name):
                continue
            try:
                if entry.resolve() == Path(active_home).resolve():
                    continue
            except OSError:
                continue
            if not (entry / "config.yaml").is_file():
                continue  # profile never configured — nothing to migrate
            token = set_hermes_home_override(entry)
            try:
                current_ver, latest_ver = _run_config_check_fresh()
                if current_ver >= latest_ver:
                    continue
                _run_migrate_config_fresh(interactive=False, quiet=True)
                after_ver, _ = _run_config_check_fresh()
                if after_ver > current_ver:
                    migrated.append((entry.name, current_ver, after_ver))
            except Exception as exc:
                logger.debug(
                    "Config migration for profile %s failed: %s", entry.name, exc
                )
            finally:
                reset_hermes_home_override(token)
    except Exception as exc:
        logger.debug("Sibling profile enumeration failed: %s", exc)
    return migrated


# Critical files that Hermes must be able to import immediately after an
# update/install. Most are imported on every CLI startup; ``web_server.py``
# is the desktop/dashboard backend path that a fresh Windows install launches
# right away. If any of these fail to parse after a pull, the user can be
# left with a bricked CLI or desktop backend. The post-pull syntax guard
# validates these and auto-rolls-back on failure.
_UPDATE_CRITICAL_FILES = (
    "hermes_cli/main.py",
    "hermes_cli/config.py",
    "hermes_cli/__init__.py",
    "hermes_cli/web_server.py",
    "cli.py",
    "run_agent.py",
    "model_tools.py",
    "toolsets.py",
    "hermes_constants.py",
)

def _capture_head_sha(git_cmd, cwd) -> str | None:
    """Return the current HEAD SHA, or None if it can't be resolved."""
    try:
        result = subprocess.run(
            git_cmd + ["rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            check=True,
        )
        return result.stdout.strip() or None
    except (subprocess.CalledProcessError, OSError):
        return None

# Files that define the editable install. A pull that touches none of them
# cannot have invalidated it.
_INSTALL_DEFINING_FILES = (
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "MANIFEST.in",
    "uv.lock",
)

def _editable_install_is_current(git_cmd, cwd, pre_pull_sha: str | None) -> bool:
    """True when the pulled commits cannot have invalidated the editable install.

    ``uv pip install -e .`` never audits an editable target — it reinstalls on
    every invocation, and every reinstall rewrites the console-script shims.
    On Windows that rewrite is the only reason the running ``hermes.exe`` has
    to be quarantined, and a quarantine that loses its race is the whole
    ``os error 32`` family. Not reinstalling when the reinstall provably
    cannot change anything removes that risk outright for the common update,
    rather than trying to make the rename win more often.

    Skipping is safe because Hermes pins its editable finder to a *static*
    module list (``[tool.setuptools] py-modules`` plus
    ``packages.find.include``). The one source-only change that would stale
    that finder is a new top-level module or package, and it cannot land
    without a ``pyproject.toml`` diff. Dependencies and ``[project.scripts]``
    live there too. New submodules inside an already-mapped package resolve
    through the real package directory and need no reinstall.

    Fails closed: an unresolvable pre-pull SHA (shallow checkout, ZIP swap)
    or a failed ``git diff`` returns False and the install runs as before.
    """
    if not pre_pull_sha:
        return False
    try:
        result = subprocess.run(
            git_cmd
            + ["diff", "--name-only", f"{pre_pull_sha}..HEAD", "--"]
            + list(_INSTALL_DEFINING_FILES),
            cwd=cwd,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
    except OSError:
        return False
    if result.returncode != 0:
        return False
    return not result.stdout.strip()

def _validate_critical_files_syntax(root) -> tuple[bool, str | None, str | None]:
    """Compile each file in ``_UPDATE_CRITICAL_FILES`` to catch SyntaxErrors.

    These are the files imported on every ``hermes`` startup; if any of them
    has a syntax error (orphan merge-conflict markers, bad ref to a name
    that no longer exists, etc.) the CLI can't bootstrap at all. We validate
    them after a successful ``git pull`` so we can auto-roll-back instead of
    leaving the user with a bricked install.

    The compiled ``.pyc`` is written to a temp directory rather than the
    source tree's ``__pycache__/`` so we don't race with concurrent test
    workers that walk the same dir, and so we don't leave a stale pyc
    behind in production if the next interpreter run picks a different
    Python version. The pyc is discarded on function return either way —
    we only care about the compile-or-not signal.

    Returns ``(ok, failing_path, error_message)``. ``ok=True`` means every
    file parsed cleanly.
    """
    import py_compile
    import tempfile

    root = Path(root)
    with tempfile.TemporaryDirectory(prefix="hermes-syntax-check-") as tmpdir:
        for relpath in _UPDATE_CRITICAL_FILES:
            path = root / relpath
            if not path.exists():
                # Missing file is suspicious but not necessarily fatal — a future
                # refactor may legitimately remove one of these. Skip and move on.
                continue
            # Mirror the relative path under the tmpdir so two different
            # files with the same basename don't collide on the cfile name.
            cfile = Path(tmpdir) / (relpath.replace("/", "__") + "c")
            try:
                py_compile.compile(str(path), cfile=str(cfile), doraise=True)
            except py_compile.PyCompileError as exc:
                return False, str(path), str(exc)
            except OSError as exc:
                return False, str(path), f"could not read: {exc}"
    return True, None, None


# Modules imported on every agent startup. Unlike _UPDATE_CRITICAL_FILES (which
# is only parsed), these are actually *imported* so that cross-module breakage
# is caught — a file can be syntactically perfect and still fail to import
# because a name it pulls from a sibling module no longer exists.
_UPDATE_CRITICAL_MODULES = (
    "hermes_cli.main",
    "run_agent",
    "model_tools",
    "toolsets",
)


def _validate_critical_modules_import(root) -> tuple[bool, str | None, str | None]:
    """Import each module in ``_UPDATE_CRITICAL_MODULES`` in a subprocess.

    ``_validate_critical_files_syntax`` only *parses* files, so it cannot see
    cross-module breakage: a partially-updated tree where ``agent/`` is new but
    ``tools/`` is old parses perfectly and still dies at startup with
    ``ImportError: cannot import name 'TODO_INJECTION_HEADER' from
    'tools.todo_tool'``. Every file is valid Python; the *combination* is not.

    That skew is reachable on the Windows ZIP-update path, whose copy loop
    walks top-level entries in ``os.listdir`` order and replaces each one
    independently — ``agent/`` lands long before ``tools/``, so a failure or
    interruption between them leaves exactly that mismatch on disk.

    Runs in a subprocess because importing these modules into the running
    updater would pollute ``sys.modules`` and execute import-time side effects
    against the half-updated tree. Costs ~0.4s.

    Uses the project venv's interpreter when there is one (matching
    ``_venv_core_imports_healthy``): ``hermes update`` can be driven by a
    different Python than the install's own, and probing the wrong
    interpreter would test a tree the user never runs.

    Returns ``(ok, failing_module, error_message)``.
    """
    from hermes_constants import FIRST_PARTY_MODULE_ROOTS

    probe = (
        "import importlib, sys\n"
        "for name in %r:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except ModuleNotFoundError as exc:\n"
        # A missing *third-party* module means dependencies aren't installed
        # yet, not a skewed checkout. Only our own packages count as breakage.
        # The root set is injected from hermes_constants so this can't drift
        # from the hint the user is shown (they disagreed once already).
        "        missing = (getattr(exc, 'name', '') or '').split('.')[0]\n"
        "        if missing in %r or missing.startswith('hermes_'):\n"
        "            sys.stdout.write(name + '\\n' + str(exc))\n"
        "            raise SystemExit(3)\n"
        "    except ImportError as exc:\n"
        "        sys.stdout.write(name + '\\n' + str(exc))\n"
        "        raise SystemExit(3)\n"
        "    except Exception:\n"
        "        pass\n"  # non-import errors (config/env) aren't update breakage
        "raise SystemExit(0)\n"
        % (_UPDATE_CRITICAL_MODULES, tuple(sorted(FIRST_PARTY_MODULE_ROOTS)))
    )
    try:
        interpreter = sys.executable
        try:
            venv_python = venv_python_path(
                Path(root) / "venv", windows=_m()._is_windows()
            )
            if venv_python.exists():
                interpreter = str(venv_python)
        except Exception:
            pass  # fall back to the running interpreter
        result = subprocess.run(
            [interpreter, "-c", probe],
            cwd=str(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        # Can't run the probe — don't block the update on our own tooling.
        return True, None, None
    if result.returncode == 3:
        parts = (result.stdout or "").split("\n", 1)
        module = parts[0].strip() or "unknown"
        detail = parts[1].strip() if len(parts) > 1 else ""
        return False, module, detail
    return True, None, None

def _gateway_prompt(prompt_text: str, default: str = "", timeout: float = 300.0) -> str:
    """File-based IPC prompt for gateway mode.

    Writes a prompt marker file so the gateway can forward the question to the
    user, then polls for a response file.  Falls back to *default* on timeout.

    Used by ``hermes update --gateway`` so interactive prompts (stash restore,
    config migration) are forwarded to the messenger instead of being silently
    skipped.
    """
    import json as _json
    import uuid as _uuid
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    prompt_path = home / ".update_prompt.json"
    response_path = home / ".update_response"

    # Clean any stale response file
    response_path.unlink(missing_ok=True)

    payload = {
        "prompt": prompt_text,
        "default": default,
        "id": str(_uuid.uuid4()),
    }
    tmp = prompt_path.with_suffix(".tmp")
    tmp.write_text(_json.dumps(payload), encoding="utf-8")
    tmp.replace(prompt_path)

    # Poll for response
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        if response_path.exists():
            try:
                answer = response_path.read_text(encoding="utf-8").strip()
                response_path.unlink(missing_ok=True)
                prompt_path.unlink(missing_ok=True)
                return answer if answer else default
            except (OSError, ValueError):
                pass
        _time.sleep(0.5)

    # Timeout — clean up and use default
    prompt_path.unlink(missing_ok=True)
    response_path.unlink(missing_ok=True)
    print(f"  (no response after {int(timeout)}s, using default: {default!r})")
    return default

def _npm_bin_exists(bin_dir: Path, name: str) -> bool:
    """True when an npm bin shim for *name* exists (POSIX or Windows)."""
    return any(
        (bin_dir / candidate).exists()
        for candidate in (name, f"{name}.cmd", f"{name}.ps1", f"{name}.exe")
    )

def _web_build_toolchain_ready(*roots: Path) -> bool:
    """True when ``tsc`` and ``vite`` shims are reachable from any of *roots*.

    Callers must pass every root the build would search; checking only one
    reports a healthy tree as broken.
    """
    bin_dirs = [
        bin_dir
        for bin_dir in (root / "node_modules" / ".bin" for root in roots)
        if bin_dir.is_dir()
    ]
    return bool(bin_dirs) and all(
        any(_npm_bin_exists(bin_dir, tool) for bin_dir in bin_dirs)
        for tool in ("tsc", "vite")
    )

def _web_toolchain_roots(web_dir: Path) -> tuple[Path, ...]:
    """Roots whose ``node_modules/.bin`` can satisfy the web build.

    ``npm run build`` prepends ``node_modules/.bin`` for the package and each
    of its ancestors, so shims hoisted to the workspace root and shims nested
    under a package that owns its lockfile (#42973) are equally valid.
    """
    return (web_dir, web_dir.parent)

def _print_curator_first_run_notice() -> None:
    """Print a short heads-up about the skill curator after `hermes update`.

    Only fires when the curator is enabled AND has no recorded run yet, which
    is exactly the window where the gateway ticker used to fire Curator
    against a fresh skill library immediately after an update. We defer the
    first real pass by one ``interval_hours``; this notice tells the user how
    to preview or disable before then. Silent on steady state.
    """
    try:
        from agent import curator
    except Exception:
        return
    try:
        if not curator.is_enabled():
            return
        state = curator.load_state()
    except Exception:
        return
    if state.get("last_run_at"):
        # Curator has run before (real or already seeded) — no notice needed.
        return
    try:
        hours = curator.get_interval_hours()
    except Exception:
        hours = 24 * 7
    days = max(1, hours // 24)
    print()
    print("ℹ Skill curator")
    print(
        f"  Background skill maintenance is enabled. First pass is deferred "
        f"~{days}d after installation; only agent-created skills are in "
        f"scope and nothing is ever auto-deleted (archive is recoverable)."
    )
    print("  Preview now:  hermes curator run --dry-run")
    print("  Pause it:     hermes curator pause")
    print(
        "  Docs:         https://hermes-agent.nousresearch.com/docs/user-guide/features/curator"
    )

def _print_fts_optimize_available_notice() -> None:
    """Advertise the opt-in v23 search-index optimization after `hermes update`.

    Only fires when the current profile's state.db is still on the legacy
    (pre-v23) inline FTS layout. Leads with the reclaimable-space figure and
    points at the exact command. Honors ``sessions.fts_optimize_notice``:
    ``advise`` (default) prints an advisory notice, ``require`` prints a
    firmer required-upgrade notice, ``off`` suppresses it. Silent for
    fresh/already-optimized installs.
    """
    mode = "advise"
    try:
        from hermes_cli.config import load_config

        mode = str(
            ((load_config() or {}).get("sessions") or {}).get(
                "fts_optimize_notice", "advise"
            )
        ).strip().lower()
    except Exception:
        mode = "advise"
    if mode == "off":
        return

    try:
        from hermes_constants import get_hermes_home
        from hermes_state import SessionDB
    except Exception:
        return
    db_path = get_hermes_home() / "state.db"
    if not db_path.exists():
        return
    try:
        size_gb = db_path.stat().st_size / (1024 ** 3)
    except OSError:
        return
    # Skip the notice for trivially small DBs — the win isn't worth the nag.
    if size_gb < 0.5:
        return
    db = None
    interrupted = False
    try:
        db = SessionDB(db_path=db_path, read_only=True)
        # read_only opens skip schema init, so probe the layout directly.
        row = db._conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'table' AND name = 'messages_fts'"
        ).fetchone()
        # An interrupted `optimize-storage` run: the table is already the
        # v23 shape, but backfill markers / demoted trash tables remain.
        # Offer the command again — re-running resumes and finishes it.
        interrupted = bool(
            db._conn.execute(
                "SELECT 1 FROM state_meta "
                "WHERE key = 'fts_rebuild_high_water' LIMIT 1"
            ).fetchone()
            or db._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                "AND name LIKE 'fts\\_v22\\_trash\\_%' ESCAPE '\\' LIMIT 1"
            ).fetchone()
            or db._conn.execute(
                "SELECT 1 FROM state_meta WHERE key IN "
                "('fts_cjk_rebuild_high_water', 'fts_cjk_stale') LIMIT 1"
            ).fetchone()
        )
    except Exception:
        return
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass
    sql = (row[0] if row else "") or ""
    if not sql or ("tool_name" in sql and not interrupted):
        # v23 layout already present (fresh/optimized) — nothing to offer.
        return

    if interrupted:
        print()
        print("◆ Session database optimization incomplete")
        print(
            "  A previous `hermes sessions optimize-storage` run was "
            "interrupted. Search still works; re-run the command to resume "
            "and finish reclaiming disk:"
        )
        print("    hermes sessions optimize-storage")
        return

    # Concrete size framing — lead with the savings the user cares about.
    est_reclaim = size_gb * 0.6
    print()
    if mode == "require":
        print("◆ Session database upgrade required")
        print(
            f"  Your search index uses the OLD storage layout and should be "
            f"upgraded. The new layout typically frees ~60% of state.db "
            f"(≈{est_reclaim:.1f} GB of your current {size_gb:.1f} GB) and is "
            f"required for continued optimal operation."
        )
    else:
        print("◆ Reclaim ~60% of your session database disk")
        print(
            f"  Your search index uses the old storage layout. Upgrading it "
            f"typically frees ~60% of state.db — about {est_reclaim:.1f} GB "
            f"of your current {size_gb:.1f} GB."
        )
    print("  Run when convenient:  hermes sessions optimize-storage")
    print(
        "  It runs in the foreground with a progress bar, is safe to "
        "interrupt/re-run, and never changes your conversations."
    )

def _print_curator_recent_run_notice() -> None:
    """Print the most recent curator run summary, exactly once.

    The curator runs in the background (gateway tick + CLI session start),
    so users learn about skill consolidations only by stumbling into a
    rename. ``hermes update`` is a high-attention surface — surface the
    most recent run's rename map here, once.

    Show-once: state stamps ``last_run_summary_shown_at`` after printing.
    Subsequent ``hermes update`` invocations skip the block until a newer
    curator run lands. Silent when the curator has never run, when the
    most recent summary has already been shown, or when the summary has
    no rename information to display (no archives).
    """
    try:
        from agent import curator
    except Exception:
        return
    try:
        state = curator.load_state()
    except Exception:
        return

    last_run_at = state.get("last_run_at")
    if not last_run_at:
        return  # no curator run yet — first-run notice handles this case

    if state.get("last_run_summary_shown_at") == last_run_at:
        return  # already shown for this run

    summary = state.get("last_run_summary") or ""
    if not summary:
        return

    # Only print when there's something interesting to show — i.e. the
    # rename map block was appended (multi-line summary). A bare "auto:
    # no changes; llm: no change" doesn't warrant interrupting the
    # update flow.
    if "\n" not in summary:
        # Still stamp it shown so we don't reconsider it on every update.
        try:
            state["last_run_summary_shown_at"] = last_run_at
            curator.save_state(state)
        except Exception:
            pass
        return

    # Format the timestamp as "Xh ago" for readability.
    when = _format_time_ago(last_run_at)
    print()
    print(f"ℹ Skill curator — last run {when}")
    for line in summary.splitlines():
        print(f"  {line}")
    print(
        "  (This message shows once per curator run. "
        "View anytime: hermes curator status)"
    )

    # Stamp shown so we don't repeat on the next update.
    try:
        state["last_run_summary_shown_at"] = last_run_at
        curator.save_state(state)
    except Exception:
        pass

def _format_time_ago(iso_ts: str) -> str:
    """Render an ISO timestamp as `Xh ago` / `Xd ago` / `Xm ago`. Best effort."""
    try:
        from datetime import datetime, timezone
        ts = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - ts
        secs = int(delta.total_seconds())
        if secs < 60:
            return "just now"
        if secs < 3600:
            return f"{secs // 60}m ago"
        if secs < 86400:
            return f"{secs // 3600}h ago"
        return f"{secs // 86400}d ago"
    except Exception:
        return "recently"

def _reload_process_scan_modules() -> None:
    """Force-reload the process-scan modules from disk after an update.

    ``_finish_dashboard_update_cleanup`` runs in the PRE-update Python
    process, but ``_scan_dashboard_processes`` does a function-level
    ``from hermes_cli._subprocess_compat import bounded_probe_run``. If the
    update added a new symbol to ``_subprocess_compat`` (as #87134 did with
    ``bounded_probe_run``), the cached OLD module object doesn't have it and
    the cleanup step crashes with ImportError — after the code update itself
    already succeeded. Reload dependency-first so ``dashboard_procs`` binds
    against the fresh ``_subprocess_compat``.

    Lives here (called from the cleanup entry point) rather than only in
    ``_reload_config_modules`` so EVERY caller — the git-update path, the
    Windows ZIP fallback path, and any future one — is covered.
    """
    import importlib

    importlib.invalidate_caches()
    for mod_name in (
        "hermes_cli._subprocess_compat",
        "hermes_cli.dashboard_procs",
    ):
        mod = sys.modules.get(mod_name)
        if mod is not None:
            try:
                importlib.reload(mod)
            except Exception as exc:
                # warning, not debug: a failed reload here surfaces seconds
                # later as an ImportError in the same process — leave a trail.
                logger.warning(
                    "Could not reload %s for post-update cleanup: %s",
                    mod_name,
                    exc,
                )


def _finish_dashboard_update_cleanup(
    node_failures: list[str], already_restarted_units: "set[str] | None" = None
) -> None:
    """Refresh managed dashboards or stop stale manual ones after an update.

    *already_restarted_units* forwards the systemd unit names (no
    ``.service`` suffix) that the fleet-restart loop already restarted
    directly, so a Serve-only install's freshly restarted process isn't
    found and restarted a second time here (review on #83595).
    """
    if node_failures:
        print()
        print("  ℹ Leaving running dashboard process(es) untouched because the")
        print("    Node.js dependency refresh did not complete.")
        return

    # The scan path lazy-imports symbols from _subprocess_compat; make sure
    # both modules reflect the freshly-updated source before touching them.
    _reload_process_scan_modules()

    stop_result = _m()._kill_stale_dashboard_processes(
        restart_managed=True, already_restarted_units=already_restarted_units
    )
    if not stop_result.get("unrecovered"):
        return

    print()
    print(
        "⚠ A web dashboard/serve process was stopped during update and could "
        "not be auto-restarted."
    )
    print("  Re-launch it when you want the web UI back:")
    print("    hermes dashboard --port <port>")

def _atomic_replace_dir(src: str, dst: str) -> None:
    """Replace directory *dst* with *src* without leaving *dst* half-deleted.

    The naive ``rmtree(dst); copytree(src, dst)`` has a destructive window: if
    the copy fails partway (common on the Windows ZIP-update path, which only
    runs because file I/O is already flaky on that machine), the old directory
    is already gone and nothing replaced it — the install is left with a
    deleted tree (issue #49145, where ``ui-tui/`` vanished and broke the TUI).

    Now a thin single-entry alias over the two-phase helpers below, which
    generalise the same stage-then-swap discipline across every entry the ZIP
    update touches (#76104). Retained because it is part of the mechanical
    ``hermes_cli.main`` re-export surface and guards the #49145 regression.
    """
    _commit_staged_replacements([(_stage_replacement(src, dst), dst)])


def _stage_replacement(src: str, dst: str) -> str:
    """Copy *src* to a sibling staging path for *dst*; return the staging path.

    Phase 1 of the two-phase replace. Handles both directories and plain
    files. Touches nothing live, so a failure here leaves the whole install
    untouched.
    """
    staging = f"{dst}.hermes-update-staging"
    backup = f"{dst}.hermes-update-old"
    # A previous run may have died between "move dst aside" and "move staging
    # in" — leaving dst missing and the backup as the ONLY copy of that entry.
    # Restore it before clearing leftovers: deleting the backup first and then
    # failing to stage (disk exhaustion is likely right after writing a full
    # staging copy) would leave a hole in the install with nothing to roll
    # back to. The restore is a same-filesystem rename — instant and safe.
    if not os.path.exists(dst) and os.path.exists(backup):
        os.rename(backup, dst)
    for leftover in (staging, backup):
        if os.path.isdir(leftover):
            shutil.rmtree(leftover, ignore_errors=True)
        elif os.path.exists(leftover):
            os.remove(leftover)
    if os.path.isdir(src):
        shutil.copytree(src, staging)
    else:
        shutil.copy2(src, staging)
    return staging


def _discard_staged(staged) -> None:
    """Remove staging paths for entries that were never committed.

    Without this a phase-1 failure (typically disk exhaustion) orphans one
    staging copy per entry already processed — up to a full second copy of
    the tree. The user then follows the "re-run `hermes update`" advice with
    *less* free space than before and the retry fails harder than the
    original attempt.
    """
    for staging, _dst in staged:
        try:
            if os.path.isdir(staging):
                shutil.rmtree(staging, ignore_errors=True)
            elif os.path.exists(staging):
                os.remove(staging)
        except OSError as exc:  # best-effort cleanup, never fatal
            logger.warning("could not remove staging path %s: %s", staging, exc)


def _commit_staged_replacements(staged) -> None:
    """Phase 2: swap every staged entry into place, rolling back all on failure.

    ``_atomic_replace_dir`` makes each *individual* directory swap safe, but
    the ZIP update replaces ~90 top-level entries in a loop, and nothing made
    the loop atomic *as a whole*. A failure partway left some entries at the
    new version and the rest at the old one — every file valid Python, the
    combination unbootable (issue #76104; the ``ImportError`` in #76091 and
    the field report in #63717 are both this).

    This covers plain files as well as directories: the repo root holds 20
    first-party modules (``run_agent.py``, ``cli.py``, ``hermes_constants.py``
    …), so a files-only failure reproduces exactly the bug class we are
    closing. Every swap is an ``os.rename`` onto a path that was just moved
    aside — a same-filesystem rename is atomic on POSIX and NTFS alike, so a
    file swap can never leave a half-written module the way ``copy2`` onto a
    live path can.

    Splitting stage-all-then-swap-all shrinks the failure window from "the
    duration of a full tree copy" to "the duration of N renames", and makes
    the remaining window recoverable: if a swap fails we restore every entry
    already swapped, so the tree lands wholly new or wholly old.
    """
    swapped: list[tuple[str, str]] = []  # (dst, backup) in swap order; "" = absent
    try:
        for staging, dst in staged:
            backup = f"{dst}.hermes-update-old"
            if os.path.exists(dst):
                os.rename(dst, backup)
                swapped.append((dst, backup))
            else:
                swapped.append((dst, ""))
            os.rename(staging, dst)
    except OSError:
        # Undo every swap already made so the install stays self-consistent.
        for dst, backup in reversed(swapped):
            try:
                if os.path.isdir(dst):
                    shutil.rmtree(dst, ignore_errors=True)
                elif os.path.exists(dst):
                    os.remove(dst)
                if backup and os.path.exists(backup):
                    os.rename(backup, dst)
            except OSError as exc:
                # Keep restoring the rest — a silent failure here is the one
                # thing that turns a recoverable rollback into a mixed tree,
                # so say so rather than swallowing it.
                logger.warning("rollback failed for %s: %s", dst, exc)
        raise
    # All swaps succeeded — drop the backups (best-effort, never fatal).
    for _dst, backup in swapped:
        if backup and os.path.isdir(backup):
            shutil.rmtree(backup, ignore_errors=True)
        elif backup and os.path.exists(backup):
            try:
                os.remove(backup)
            except OSError:
                pass


def _branch_head_label(git_cmd=None, cwd=None) -> str | None:
    """``"<branch> @ <short-sha>"`` for the checkout, or None when unknown.

    Appended to the update summary lines so branch drift is visible at a
    glance (live incident 2026-08-17: a checkout parked on a stale feature
    branch got "✓ Update complete!" with nothing on the line saying WHERE
    the checkout actually sat). Never raises — summary decoration must not
    break an update.
    """
    try:
        cmd = list(git_cmd) if git_cmd else ["git"]
        root = cwd if cwd is not None else _m().PROJECT_ROOT
        branch = subprocess.run(
            cmd + ["rev-parse", "--abbrev-ref", "HEAD"],
            cwd=root, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        sha = subprocess.run(
            cmd + ["rev-parse", "--short", "HEAD"],
            cwd=root, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        branch_name = branch.stdout.strip()
        sha_text = sha.stdout.strip()
        if branch.returncode != 0 or sha.returncode != 0 or not sha_text:
            return None
        if not branch_name:
            return None
        label = "detached" if branch_name == "HEAD" else branch_name
        return f"{label} @ {sha_text}"
    except Exception:
        return None


def _branch_head_suffix(git_cmd=None, cwd=None) -> str:
    """`` [<branch> @ <sha>]`` suffix for summary lines ("" when unknown)."""
    label = _branch_head_label(git_cmd, cwd)
    return f" [{label}]" if label else ""


def _assess_parked_branch_switch(
    git_cmd: list[str], cwd: Path, current_branch: str, target_branch: str
) -> tuple[bool, str]:
    """Decide whether it is safe to auto-switch a parked feature branch back
    to the update target.

    Live incident (2026-08-17, Teknium's box): the source checkout sat on a
    stale feature branch left behind by earlier tooling; ``hermes update``
    autostashed, ran its post-update steps and printed "✓ Code updated!"
    while the running code stayed days behind main. The guard's contract:

    - (True, "") when the working tree + index are clean AND every commit on
      the parked branch is already contained in ``origin/<target_branch>``
      (``git cherry`` reports no ``+`` lines).
    - (True, "unmerged:<count>") when the tree is clean but the branch has
      commits not yet in the target. Switching is safe — ``git checkout``
      never discards committed work and the branch keeps the commits — but
      the caller must print a LOUD notice naming the branch and count so the
      work is not forgotten. This is what non-interactive callers (desktop
      update button, gateway /update, cron) rely on: they have no way to
      resolve a skip, so a clean checkout must always reach the target.
    - (False, <reason>) — dirty tree, git errors, or the
      ``updates.auto_switch_parked_branch: false`` config opt-out — and the
      caller must NOT touch the branch. A dirty tree is the one genuinely
      unsafe case: uncommitted work would have to ride an autostash across
      branches, which is how the 2026-08-17 incident started.

    Block reasons: "disabled", "dirty", "unverifiable".
    """
    try:
        from hermes_cli.config import load_config

        _update_cfg = (load_config() or {}).get("updates", {})
        if isinstance(_update_cfg, dict) and not bool(
            _update_cfg.get("auto_switch_parked_branch", True)
        ):
            return False, "disabled"
    except Exception as exc:
        # A config read failure must not disable the guard's safety checks —
        # fall through to them with the default (auto-switch allowed).
        logger.debug("Could not read updates.auto_switch_parked_branch: %s", exc)

    status = subprocess.run(
        git_cmd + ["status", "--porcelain"],
        cwd=cwd, capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )
    if status.returncode != 0:
        return False, "unverifiable"
    if status.stdout.strip():
        return False, "dirty"

    cherry = subprocess.run(
        git_cmd + ["cherry", f"origin/{target_branch}"],
        cwd=cwd, capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )
    if cherry.returncode != 0:
        return False, "unverifiable"
    unmerged = [
        line for line in cherry.stdout.splitlines() if line.startswith("+")
    ]
    if unmerged:
        # Clean tree: switching is safe (checkout keeps the commits on the
        # branch). The reason string tells the caller to print the loud
        # "branch kept with N unmerged commit(s)" notice.
        return True, f"unmerged:{len(unmerged)}"
    return True, ""


def _print_parked_branch_skip_warning(
    git_cmd: list[str],
    cwd: Path,
    current_branch: str,
    target_branch: str,
    reason: str,
) -> None:
    """LOUD block explaining why the code update was skipped on a parked
    branch, with the behind-count and the exact commands to resolve."""
    behind = None
    try:
        behind_result = subprocess.run(
            git_cmd + ["rev-list", f"HEAD..origin/{target_branch}", "--count"],
            cwd=cwd, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        if behind_result.returncode == 0 and behind_result.stdout.strip():
            behind = int(behind_result.stdout.strip())
    except Exception:
        behind = None

    if reason == "dirty":
        why = "the working tree has uncommitted changes"
    elif reason == "disabled":
        why = "updates.auto_switch_parked_branch is set to false in config.yaml"
    else:
        why = (
            f"the branch state could not be verified against "
            f"origin/{target_branch}"
        )

    bar = "=" * 68
    print()
    print(bar)
    print(f"⚠ CODE UPDATE SKIPPED — checkout is parked on '{current_branch}'")
    print(f"  Not auto-switching to {target_branch}: {why}.")
    if behind is not None and behind > 0:
        print(
            f"  This checkout is {behind} commit(s) BEHIND "
            f"origin/{target_branch} — the code you are running is stale."
        )
    print()
    print("  To resolve, inspect the branch and switch back yourself:")
    print(f"    git -C {cwd} status")
    print(f"    git -C {cwd} checkout {target_branch} && hermes update")
    print(
        "  (commit or stash your work on the branch first if you want to "
        "keep it)"
    )
    print(bar)


def _print_parked_branch_kept_notice(
    current_branch: str, target_branch: str, unmerged_count: str
) -> None:
    """LOUD notice printed when a clean parked branch with unmerged commits
    is auto-switched back to the update target.

    Non-interactive callers (desktop update button, gateway /update, cron)
    cannot resolve a skip, so a clean checkout always proceeds to the
    target — but the unmerged work must be impossible to miss.  The commits
    are untouched: ``git checkout`` never discards committed work; the
    branch keeps them until the user returns.
    """
    bar = "=" * 68
    print()
    print(bar)
    print(
        f"⚠ Checkout was parked on '{current_branch}' with "
        f"{unmerged_count} commit(s) not merged into origin/{target_branch}."
    )
    print(
        f"  Switching to {target_branch} so the update can proceed — your "
        f"commit(s) are safe on '{current_branch}'."
    )
    print()
    print("  To pick the work back up later:")
    print(f"    git checkout {current_branch}")
    print(bar)


def _print_update_completion(message: str) -> None:
    """Print an update outcome plus, when the dashboard launched this run
    with an action id, a terminal receipt line the Desktop can match after
    the dashboard restarts (see #47359 / #58764).

    The outcome line carries the checkout's actual branch + HEAD short-sha
    so branch drift is visible at a glance (2026-08-17 parked-branch
    incident)."""
    print(f"{message}{_branch_head_suffix()}")
    action_id = os.environ.get("HERMES_ACTION_ID", "")
    if len(action_id) == 32 and all(char in "0123456789abcdef" for char in action_id):
        print(f"=== hermes-update completed {action_id} ===")


def _called_process_error_cmd_parts(exc: subprocess.CalledProcessError) -> list[str]:
    """Normalize ``CalledProcessError.cmd`` into argv-style tokens."""
    cmd = exc.cmd
    if cmd is None:
        return []
    if isinstance(cmd, (str, bytes)):
        text = cmd.decode("utf-8", "replace") if isinstance(cmd, bytes) else cmd
        try:
            return shlex.split(text, posix=os.name != "nt")
        except ValueError:
            return text.split()
    return [str(part) for part in cmd]


def _called_process_error_is_git(exc: subprocess.CalledProcessError) -> bool:
    """True when the failed subprocess was git itself."""
    parts = _called_process_error_cmd_parts(exc)
    if not parts:
        return False
    # Windows argv may use backslashes; basename() on POSIX would otherwise
    # keep the whole path. Normalize separators before taking the name.
    name = os.path.basename(parts[0].replace("\\", "/")).lower()
    return name in {"git", "git.exe"}


def _called_process_error_is_python_dep_install(
    exc: subprocess.CalledProcessError,
) -> bool:
    """True when the failed subprocess was a uv/pip (or ensurepip) install."""
    parts = [part.lower() for part in _called_process_error_cmd_parts(exc)]
    if not parts:
        return False
    exe = os.path.basename(parts[0].replace("\\", "/"))
    if "ensurepip" in parts:
        return True
    if "install" in parts and (
        "pip" in parts or exe in {"pip", "pip.exe", "pip3", "pip3.exe", "uv", "uv.exe"}
    ):
        return True
    return False


def _format_update_failure_stage(exc: subprocess.CalledProcessError) -> str:
    """Name the update stage that actually failed.

    The git pull and the Python-dependency install share one ``try`` in
    ``_cmd_update_impl``. Calling every ``CalledProcessError`` a git failure
    (the historical Windows message) sent users hunting in the wrong place
    and, worse, keyed the ZIP overlay on exception *type* rather than on git
    actually having failed (#87304, #85840).
    """
    if _called_process_error_is_python_dep_install(exc):
        return "Python dependency install failed"
    if _called_process_error_is_git(exc):
        return "Git update failed"
    return "Update step failed"


def _shim_quarantine_error_type() -> "type[BaseException]":
    """The strict-quarantine refusal type, resolved lazily through ``_m()``.

    Falls back to a never-raised private type when main.py lacks it (torn
    mid-update tree), so the ``except`` clause stays valid.
    """
    cls = getattr(_m(), "ShimQuarantineError", None)
    if isinstance(cls, type) and issubclass(cls, BaseException):
        return cls

    class _Never(Exception):
        pass

    return _Never


def _refuse_update_for_contended_shims(exc: BaseException) -> None:
    """Refuse the dependency sync when live shims could not be quarantined.

    #87331 fail-closed half: a shim rename that failed every retry proves a
    process holds the venv without FILE_SHARE_DELETE — running the installer
    anyway is exactly how the venv ends up stranded between versions. The
    code swap (when one happened) is already committed; only the dependency
    install is deferred, via the update-incomplete marker, to the next fresh
    launch after the holder exits. Exits 2 (refused) so the command-boundary
    receipt net records it as a refusal, not a failure.
    """
    print("✗ Cannot continue the update: live Hermes launcher(s) could not be")
    print("  moved aside:")
    for name in getattr(exc, "failed_shims", []) or ["hermes.exe"]:
        print(f"    {name}")
    print("  Another process is holding this install's venv — typically Hermes")
    print("  Desktop, a gateway, or another hermes REPL — and mutating the venv")
    print("  now would strand it half-updated.")
    print("  The dependency install has been deferred: close the process(es)")
    print("  above, then run any `hermes` command to finish it automatically.")
    # Idempotent: the git path already dropped the marker before the sync;
    # this covers the ZIP/repair paths so the deferral is never silent.
    _write_update_incomplete_marker()
    sys.exit(2)


def _should_zip_fallback_on_update_error(exc: BaseException) -> bool:
    """ZIP fallback is for Windows git file-I/O breakage, not later stages.

    A dependency-install failure (locked ``hermes.exe`` / ``uv pip install``
    exit 2) is not a git failure. The pull has already succeeded by then, so
    re-downloading the source ZIP cannot fix the install and would replace
    every top-level entry except ``venv`` / ``node_modules`` / ``.git`` /
    ``.env`` — permanently deleting uncommitted edits and untracked files.
    """
    return (
        isinstance(exc, subprocess.CalledProcessError)
        and _m()._is_windows()
        and _called_process_error_is_git(exc)
    )


def _print_called_process_error_tail(
    exc: subprocess.CalledProcessError, *, limit: int = 12
) -> None:
    """Print a captured stderr/stdout tail when the failing call recorded one."""
    blob = exc.stderr or exc.stdout or ""
    if isinstance(blob, bytes):
        blob = blob.decode("utf-8", "replace")
    lines = [line for line in str(blob).splitlines() if line.strip()]
    if not lines:
        return
    print("  Last output:")
    for line in lines[-limit:]:
        print(f"    {line}")


def _zip_overlay_block_reason(
    root: Path, *, ignore_staging_artifacts: bool = False
) -> Optional[str]:
    """Why overlaying a ZIP onto ``root`` would destroy work, or None if safe.

    The ZIP path swaps every top-level entry (except a tiny preserve set) and
    then deletes the backups, so uncommitted edits and untracked files under
    a replaced directory are gone. Fail closed when git status cannot run:
    unknown dirtiness is not a license to clobber the tree (#87304).

    ``ignore_staging_artifacts`` is for the pre-swap re-check: phase 1 of the
    two-phase replace creates ``*.hermes-update-staging`` siblings inside the
    checkout, which git reports as untracked. Those are our own artifacts,
    not user work — without the filter the re-check would always refuse.
    """
    if not (root / ".git").exists():
        return None
    git_cmd = ["git"]
    if sys.platform == "win32":
        git_cmd = ["git", "-c", "windows.appendAtomically=false"]
    result = subprocess.run(
        # -uall: a user-level ``status.showUntrackedFiles = no`` git config
        # would otherwise hide untracked files and silently blind this guard.
        git_cmd + ["status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        suffix = f" ({detail[0]})" if detail else ""
        return f"could not check the working tree{suffix}"
    lines = [line for line in (result.stdout or "").splitlines() if line.strip()]
    if ignore_staging_artifacts:
        lines = [
            line for line in lines if not _is_zip_staging_artifact_status_line(line)
        ]
    if lines:
        return "the working tree has uncommitted changes or untracked files"
    return None


_ZIP_STAGING_ARTIFACT_SUFFIXES = (".hermes-update-staging", ".hermes-update-old")


def _is_zip_staging_artifact_status_line(line: str) -> bool:
    """True when a porcelain status line is our own two-phase-swap artifact."""
    payload = line[3:] if len(line) >= 3 else line
    top_level = (
        payload.strip().strip('"').replace("\\", "/").rstrip("/").split("/", 1)[0]
    )
    return top_level.endswith(_ZIP_STAGING_ARTIFACT_SUFFIXES)


def _abort_zip_update_if_dirty_tree() -> None:
    """Refuse to overlay a ZIP onto a dirty git checkout (#87304)."""
    reason = _zip_overlay_block_reason(_m().PROJECT_ROOT)
    if reason is None:
        return
    print(f"✗ ZIP fallback refused: {reason}.")
    print(
        "  Overlaying the ZIP would overwrite uncommitted edits and permanently "
        "delete untracked files."
    )
    print("  Stash or commit your changes, then rerun `hermes update`.")
    print("  To inspect: git status --porcelain")
    _m().sys.exit(1)


def _read_project_version() -> str | None:
    """Read the ``version`` field from the checkout's pyproject.toml.

    Reads the on-disk file (not importlib.metadata) because after a git
    pull the installed distribution metadata still describes the OLD
    version; the file is the only source that reflects what was just
    pulled. Returns None on any failure — version reporting is cosmetic
    and must never break an update.
    """
    try:
        import tomllib

        with open(_m().PROJECT_ROOT / "pyproject.toml", "rb") as fh:  # windows-footgun: ok — binary mode, tomllib requires bytes
            version = tomllib.load(fh).get("project", {}).get("version")
        return str(version) if version else None
    except Exception:
        return None


def _update_complete_message(pre_version: str | None) -> str:
    """Completion line with the version transition when it is known.

    Ported from PrimeIntellect-ai/prime-agent#630: after a successful
    self-update, show both versions (``v0.19.4 → v0.20.0``) so the user
    can see what they actually got. Falls back to the plain message when
    either side is unknown or the version did not change (e.g. several
    commits landed within one release).
    """
    post_version = _read_project_version()
    if pre_version and post_version and pre_version != post_version:
        return f"✓ Update complete! (v{pre_version} → v{post_version})"
    if post_version:
        return f"✓ Update complete! (v{post_version})"
    return "✓ Update complete!"
 
 
def _clear_stale_sqlite_sidecars(db_path: Path) -> None:
    """Delete the WAL / shared-memory / rollback-journal files next to *db_path*.

    Call this immediately before overwriting a database file with a snapshot
    image. Quick snapshots are produced by ``backup._safe_copy_db`` through
    ``sqlite3.backup()``, so the image is already checkpointed and owns no WAL —
    which is exactly why ``backup._EXCLUDED_SUFFIXES`` refuses to ship sidecars
    inside a snapshot. Copying the image over the destination replaces only the
    main database file, so any ``-wal`` / ``-shm`` left behind by the *old*
    database (a crashed writer, or a second Hermes process the updater's drain
    did not stop) survives and is replayed over the fresh image on the next
    open. The result passes ``PRAGMA integrity_check`` while serving the old
    database's contents, and the first checkpoint folds it in permanently.

    Removing them is safe here specifically: they belong to a database the
    caller has already declared corrupt and is about to discard.
    """
    for suffix in ("-wal", "-shm", "-journal"):
        db_path.with_name(db_path.name + suffix).unlink(missing_ok=True)


def _print_update_summary(
    *,
    node_failures: list,
    desktop_build_ok: bool,
    pre_update_version: str | None,
) -> None:
    """Final update banner. A failed Desktop rebuild is non-fatal for the
    Python side, but must not print ``✓ Update complete!`` (#88251)."""
    print()
    if node_failures or not desktop_build_ok:
        parts = []
        if node_failures:
            parts.append(
                f"Node.js dependencies for {', '.join(node_failures)} did not refresh"
            )
        if not desktop_build_ok:
            parts.append(
                "the desktop app was not rebuilt and is still on the previous build"
            )
        print("⚠ Update partially complete — " + "; ".join(parts) + ".")
        if node_failures:
            print("  Code and Python deps are updated, but the dashboard/TUI may")
            print("  be in a mixed state until the Node deps are rebuilt.")
        if not desktop_build_ok:
            print("  Run `hermes desktop` to retry the desktop rebuild.")
    else:
        _print_update_completion(_update_complete_message(pre_update_version))


def _write_gateway_update_exit_code(ok: bool) -> None:
    path = get_hermes_home() / ".update_exit_code"
    try:
        path.write_text("0" if ok else "1", encoding="utf-8")
    except OSError:
        pass


def _restore_state_db_from_snapshot(state_path: Path, snap_state: Path) -> bool:
    """Replace *state_path* with the snapshot image at *snap_state*.

    Shared by both post-update auto-restore paths (the ZIP update and the git
    pull). The destination's stale sidecars are cleared before the copy, so the
    restored image cannot be silently overwritten by the corrupt database's WAL
    replay — see :func:`_clear_stale_sqlite_sidecars`.

    Refuses (returns ``False``) while another process still holds the database
    or its sidecars open: copying a snapshot over a live writer's inode makes
    the writer's page cache and WAL index disagree with the file bytes, and
    its next checkpoint writes pages at offsets that no longer mean what it
    thinks — the #90950 page-1 clobber. ``None`` (scan unavailable) proceeds:
    the updater has already drained gateways, and refusing on "unknown" would
    disable auto-restore on every non-Linux host.

    Returns ``True`` when the restored file passes an integrity check. Raises
    ``OSError`` if the copy itself fails, which callers already report.
    """
    from hermes_cli.backup import _foreign_db_holder_pids, verify_sqlite_integrity

    holders = _foreign_db_holder_pids(state_path)
    if holders:
        print(
            f"  ✗ Auto-restore refused: process(es) {holders} still hold "
            "state.db or its WAL open. Stop them (hermes gateway stop), "
            "then restore manually with /snapshot restore."
        )
        return False
    _clear_stale_sqlite_sidecars(state_path)
    shutil.copy2(snap_state, state_path)
    restored = verify_sqlite_integrity(
        state_path, check_header=True, run_pragma=True
    )
    return bool(restored.get("valid"))


def _update_via_zip(args, *, had_desktop_app_before_update: bool = False) -> bool:
    """Update Hermes Agent by downloading a ZIP archive.

    Used on Windows when git file I/O is broken (antivirus, NTFS filter
    drivers causing 'Invalid argument' errors on file creation).

    Returns ``False`` when a Desktop rebuild ran and failed; ``True`` otherwise.
    """
    active_tool_dependencies = _m()._capture_active_tool_dependencies()

    import tempfile
    import zipfile
    from urllib.request import urlretrieve

    # Snapshot the pre-update version before files are replaced so the
    # completion line can report the transition (prime-agent#630 port).
    pre_update_version = _read_project_version()

    # The ZIP fallback exists for Windows git-file-I/O breakage. It pulls a
    # static archive from GitHub, which is fine for the default "main"
    # channel but would silently ignore --branch and update from main even
    # if the user asked for something else — exactly the silent-divergence
    # bug --branch was added to prevent. Refuse to proceed in that case
    # rather than lie.
    branch = _m()._resolve_update_branch(args)
    if branch != "main":
        print(
            f"✗ --branch={branch} is not supported on the Windows ZIP-fallback "
            "update path."
        )
        print(
            "  This path runs when git file I/O is broken on the system. "
            "Either resolve the git-side breakage (typically an antivirus "
            "or NTFS filter holding files open) and rerun `hermes update "
            f"--branch {branch}`, or update against main with `hermes update`."
        )
        _m().sys.exit(1)
    _abort_zip_update_if_dirty_tree()
    zip_url = (
        f"https://github.com/NousResearch/hermes-agent/archive/refs/heads/{branch}.zip"
    )

    print("→ Downloading latest version...")
    tmp_dir = tempfile.mkdtemp(prefix="hermes-update-")
    try:
        zip_path = os.path.join(tmp_dir, f"hermes-agent-{branch}.zip")
        urlretrieve(zip_url, zip_path)

        print("→ Extracting...")
        import stat as _stat
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Validate paths to prevent zip-slip (path traversal) AND reject
            # symlink members. A GitHub source ZIP for hermes-agent itself
            # should never contain symlinks — they'd point outside the
            # extracted tree and let an attacker who can compromise the
            # update mirror plant arbitrary files via the update path.
            tmp_dir_real = os.path.realpath(tmp_dir)
            for member in zf.infolist():
                member_path = os.path.realpath(os.path.join(tmp_dir, member.filename))
                if (
                    not member_path.startswith(tmp_dir_real + os.sep)
                    and member_path != tmp_dir_real
                ):
                    raise ValueError(
                        f"Zip-slip detected: {member.filename} escapes extraction directory"
                    )
                # Unix mode lives in the upper 16 bits of external_attr;
                # mask to the file-type bits.
                mode = (member.external_attr >> 16) & 0o170000
                if _stat.S_ISLNK(mode):
                    raise ValueError(
                        f"ZIP contains unsupported symlink member: {member.filename}"
                    )
            zf.extractall(tmp_dir)

        # GitHub ZIPs extract to hermes-agent-<branch>/
        extracted = os.path.join(tmp_dir, f"hermes-agent-{branch}")
        if not os.path.isdir(extracted):
            # Try to find it
            for d in os.listdir(tmp_dir):
                candidate = os.path.join(tmp_dir, d)
                if os.path.isdir(candidate) and d != "__MACOSX":
                    extracted = candidate
                    break

        # Copy updated files over existing installation, preserving venv/node_modules/.git
        preserve = {"venv", "node_modules", ".git", ".env"}
        entries = [i for i in os.listdir(extracted) if i not in preserve]

        # Two-phase replace (#76104). Phase 1 copies every entry — directories
        # AND top-level files — to a sibling staging path without touching
        # anything live; phase 2 swaps them all in with same-filesystem
        # renames and rolls back every swap if any one fails. Replacing
        # entries one-at-a-time (the previous shape) meant an interruption
        # partway left `agent/` new and `tools/` stale — all files valid, the
        # tree unbootable. Files matter as much as directories here: the repo
        # root holds 20 first-party modules (run_agent.py, cli.py,
        # hermes_constants.py, ...).
        #
        # Staging costs one extra copy of the tree on disk. Check up front so
        # we fail with a clear message instead of running out mid-copy.
        need = sum(
            os.path.getsize(os.path.join(dirpath, f))
            for entry in entries
            for dirpath, _dirs, files in os.walk(os.path.join(extracted, entry))
            for f in files
        ) + sum(
            os.path.getsize(os.path.join(extracted, e))
            for e in entries
            if os.path.isfile(os.path.join(extracted, e))
        )
        # Only the staging copy is new — the live tree already occupies its
        # space and the swaps are renames, not copies. Ask for the staging
        # copy plus 20% headroom rather than a full 2x, which would block
        # updates that would have succeeded on exactly the space-constrained
        # machines most likely to hit this path.
        required = int(need * 1.2)
        free = shutil.disk_usage(str(_m().PROJECT_ROOT)).free
        if free < required:
            raise RuntimeError(
                f"not enough free disk space to stage the update safely "
                f"(need ~{required // (1024 * 1024)} MB, have "
                f"{free // (1024 * 1024)} MB)"
            )

        staged: list[tuple[str, str]] = []
        try:
            for item in entries:
                src = os.path.join(extracted, item)
                dst = os.path.join(str(_m().PROJECT_ROOT), item)
                staged.append((_stage_replacement(src, dst), dst))
                # #70337/#87331: the GitHub source ZIP contains only source —
                # apps/desktop/release/ (the BUILT desktop app, win-unpacked/
                # Hermes.exe) exists only in the LIVE tree. Swapping `apps`
                # without it deletes the desktop build and breaks the
                # shortcut. Graft the live release dir into the staged copy
                # BEFORE the swap so the commit preserves it atomically.
                if item == "apps":
                    live_release = os.path.join(dst, "desktop", "release")
                    staged_release = os.path.join(
                        staged[-1][0], "desktop", "release"
                    )
                    if os.path.isdir(live_release) and not os.path.exists(
                        staged_release
                    ):
                        os.makedirs(os.path.dirname(staged_release), exist_ok=True)
                        shutil.copytree(live_release, staged_release)
        except Exception:
            # Nothing is live yet; drop the partial staging copies so a retry
            # starts from the same free space this attempt did.
            _discard_staged(staged)
            raise

        try:
            # Re-check the tree right before the swap (#87304 TOCTOU): the
            # download + extract + staging window above can take minutes, and
            # work created in it would be destroyed by the commit below. Our
            # own phase-1 staging siblings are filtered out — they are the
            # expected artifacts of getting here, not user work.
            recheck_reason = _zip_overlay_block_reason(
                _m().PROJECT_ROOT, ignore_staging_artifacts=True
            )
            if recheck_reason is not None:
                _discard_staged(staged)
                print(f"✗ ZIP fallback aborted before the swap: {recheck_reason}.")
                print(
                    "  Files appeared in the checkout while the update was "
                    "downloading; committing the swap would delete them."
                )
                print("  Stash or commit your changes, then rerun `hermes update`.")
                _m().sys.exit(1)
            _commit_staged_replacements(staged)
        except Exception:
            # The rollback already restored every swapped entry, but staging
            # copies for the not-yet-swapped entries (potentially most of a
            # full tree) are still on disk. Drop them, or the retry's
            # up-front free-space check — which runs BEFORE the lazy
            # per-entry leftover cleanup — fails on litter this attempt
            # left behind: the exact "retry fails harder" failure mode
            # _discard_staged exists to prevent. Safe post-rollback: swapped
            # entries' staging paths were renamed away, and _discard_staged
            # skips paths that no longer exist.
            _discard_staged(staged)
            raise
        update_count = len(staged)

        print(f"✓ Updated {update_count} items from ZIP")

    except Exception as e:
        print(f"✗ ZIP update failed: {e}")
        # The two-phase replace either commits every entry or rolls them all
        # back, so a failure here does not leave a mixed-version tree — don't
        # scare the user toward a reinstall they don't need.
        print("  Your existing install was left in place.")
        print(
            "  Re-run `hermes update` to retry; if the agent won't start, "
            "reinstall from https://hermes-agent.nousresearch.com"
        )
        _m().sys.exit(1)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Clear stale bytecode after ZIP extraction
    removed = _m()._clear_bytecode_cache(_m().PROJECT_ROOT)
    if removed:
        print(
            f"  ✓ Cleared {removed} stale __pycache__ director{'y' if removed == 1 else 'ies'}"
        )
    _m()._record_bytecode_fingerprint()
    _m()._refresh_bootstrap_cache_scripts(branch)

    # Reinstall Python dependencies. Prefer .[all], but if one optional extra
    # breaks on this machine, keep base deps and reinstall the remaining extras
    # individually so update does not silently strip working capabilities.
    #
    # Self-lock deferral (relocated preflight — #86735): the ZIP code swap
    # above is already committed; defer only the dependency sync when this
    # process holds a native extension the sync must rewrite.
    _m()._abort_dependency_sync_if_self_locked()
    print("→ Updating Python dependencies...")

    from hermes_cli.managed_uv import ensure_uv, update_managed_uv

    # Keep managed uv current — runs `uv self update` if we already have one.
    update_managed_uv()

    uv_bin = ensure_uv()

    pip_cmd = [_m().sys.executable, "-m", "pip"]
    if not uv_bin:
        uv_bin = _ensure_uv_for_termux(pip_cmd)
    if uv_bin:
        # Same third-party UV-env isolation as the main update path (#83914):
        # a user-level UV_PYTHON_INSTALL_DIR / UV_PYTHON from unrelated
        # software must not steer which interpreter uv resolves here.
        from hermes_cli.managed_uv import managed_python_env

        uv_env = managed_python_env()
        uv_env["VIRTUAL_ENV"] = str(_m().PROJECT_ROOT / "venv")
        if _m()._is_termux_env(uv_env):
            uv_env.pop("PYTHONPATH", None)
            uv_env.pop("PYTHONHOME", None)
        try:
            _m()._install_python_dependencies_with_optional_fallback([uv_bin, "pip"], env=uv_env)
        except _shim_quarantine_error_type() as _sqe:
            # #87331: this runs inside the ZIP-fallback error handler, so the
            # boundary except clause in cmd_update cannot catch it — refuse
            # here with the same defer-via-marker contract.
            _refuse_update_for_contended_shims(_sqe)
    else:
        # Use sys.executable to explicitly call the venv's pip module,
        # avoiding PEP 668 'externally-managed-environment' errors on Debian/Ubuntu.
        # Some environments lose pip inside the venv; bootstrap it back with
        # ensurepip before trying the editable install.
        try:
            subprocess.run(
                pip_cmd + ["--version"],
                cwd=_m().PROJECT_ROOT,
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError:
            subprocess.run(
                [_m().sys.executable, "-m", "ensurepip", "--upgrade", "--default-pip"],
                cwd=_m().PROJECT_ROOT,
                check=True,
            )
        _m()._install_python_dependencies_with_optional_fallback(pip_cmd)

    install_prefix = [uv_bin, "pip"] if uv_bin else pip_cmd
    install_env = uv_env if uv_bin else None
    _m()._restore_active_tool_dependencies(
        active_tool_dependencies,
        install_prefix,
        env=install_env,
    )

    # ZIP path parity: heal the active memory provider's bridge packages
    # after the dependency reinstall, same as the git-pull path (#53272,
    # #70636).
    _m()._refresh_active_memory_provider_dependencies()

    # Now that dependencies are installed, verify the tree actually imports.
    # The copy loop above replaces top-level entries one at a time in
    # os.listdir order, so an interruption between (say) `agent/` and `tools/`
    # leaves a tree whose files all parse but cannot be imported together —
    # the ImportError-on-startup class this guard exists to catch. Deliberately
    # placed *after* the dependency reinstall so a genuinely-new third-party
    # requirement isn't misreported as a partial copy. There is no SHA to roll
    # back to here, so surface it with a concrete recovery step rather than
    # reporting a successful update over a bricked install.
    import_ok, failing_module, import_error = _validate_critical_modules_import(
        _m().PROJECT_ROOT
    )
    if not import_ok:
        print()
        print("✗ Update left the install in an unimportable state:")
        print(f"  {failing_module}: {import_error}")
        print()
        print("  This usually means the copy was interrupted partway through.")
        print("  Re-run `hermes update` to complete it.")
        _m().sys.exit(1)

    node_failures = _update_node_dependencies()
    _m()._build_web_ui(_m().PROJECT_ROOT / "web")
    desktop_build_ok = _rebuild_desktop_after_update(
        _m().PROJECT_ROOT / "apps" / "desktop",
        had_desktop_app_before_update=had_desktop_app_before_update,
    )

    # Sync skills
    try:
        from tools.skills_sync import sync_skills

        print("→ Syncing bundled skills...")
        result = sync_skills(quiet=True)
        if result["copied"]:
            print(f"  + {len(result['copied'])} new: {', '.join(result['copied'])}")
        if result.get("updated"):
            print(
                f"  ↑ {len(result['updated'])} updated: {', '.join(result['updated'])}"
            )
        if result.get("user_modified"):
            print(f"  ~ {len(result['user_modified'])} user-modified (kept)")
            print(
                "    → see them: hermes skills list-modified  "
                "(diff/reset to resume updates)"
            )
        if result.get("cleaned"):
            print(f"  − {len(result['cleaned'])} removed from manifest")
        if result.get("relocated"):
            print(
                f"  → {len(result['relocated'])} moved to new upstream paths: "
                f"{', '.join(result['relocated'])}"
            )
        if not result["copied"] and not result.get("updated"):
            print("  ✓ Skills are up to date")
    except Exception:
        pass

    # Seed the model-catalog disk cache from the freshly-unpacked checkout
    # (same rationale as the git-pull path in _cmd_update_impl). Non-fatal.
    try:
        from hermes_cli.model_catalog import seed_cache_from_checkout

        if seed_cache_from_checkout(_m().PROJECT_ROOT):
            print("  ✓ Model catalog cache refreshed from checkout")
    except Exception as e:
        logger.debug("Model catalog seed during zip update failed: %s", e)

    # ── Post-update state.db integrity guard (#68474) ─────────────────
    # Same as the git-pull path: verify state.db survived the ZIP update
    # and auto-restore from the most recent pre-update snapshot if needed.
    try:
        from hermes_cli.backup import _quick_snapshot_root, verify_sqlite_integrity

        _state_path = get_hermes_home() / "state.db"
        if _state_path.exists():
            _state_ok = verify_sqlite_integrity(
                _state_path, check_header=True, run_pragma=True
            )
            if not _state_ok.get("valid"):
                print()
                print(
                    "⚠ state.db is corrupted after update: "
                    + _state_ok.get("message", "unknown error")
                )
                _snap_root = _quick_snapshot_root(get_hermes_home())
                if _snap_root.exists():
                    _snap_dirs = sorted(
                        (d for d in _snap_root.iterdir() if d.is_dir()),
                        reverse=True,
                    )
                    for _snap_dir in _snap_dirs:
                        _snap_state = _snap_dir / "state.db"
                        if _snap_state.exists():
                            _snap_ok = verify_sqlite_integrity(
                                _snap_state, check_header=True, run_pragma=True
                            )
                            if _snap_ok.get("valid"):
                                try:
                                    if _restore_state_db_from_snapshot(
                                        _state_path, _snap_state
                                    ):
                                        print(
                                            "  ✓ Auto-restored from snapshot "
                                            f"{_snap_dir.name}"
                                        )
                                    else:
                                        print(
                                            "  ✗ Auto-restore FAILED — restored "
                                            "copy also failed integrity"
                                        )
                                    break
                                except OSError as _exc:
                                    print(
                                        f"  ✗ Auto-restore file copy failed: {_exc}"
                                    )
                                    break
    except Exception as exc:
        logger.debug(
            "Post-update state.db integrity check (zip path) failed: %s", exc
        )

    _print_update_summary(
        node_failures=node_failures,
        desktop_build_ok=desktop_build_ok,
        pre_update_version=pre_update_version,
    )
    try:
        _print_curator_first_run_notice()
    except Exception as e:
        logger.debug("Curator first-run notice failed: %s", e)
    try:
        _print_curator_recent_run_notice()
    except Exception as e:
        logger.debug("Curator recent-run notice failed: %s", e)
    # Don't stop a working dashboard when the Node refresh failed — see the
    # git-update path for rationale (#30271).
    _finish_dashboard_update_cleanup(node_failures)
    try:
        from hermes_cli.update_receipt import finalize_update_receipt

        finalize_update_receipt(
            "success" if (desktop_build_ok and not node_failures) else "partial"
        )
    except Exception as _receipt_exc:
        logger.debug("Update receipt finalize (zip path) failed: %s", _receipt_exc)
    return desktop_build_ok

def _stash_local_changes_if_needed(git_cmd: list[str], cwd: Path) -> Optional[str]:
    status = subprocess.run(
        git_cmd + ["status", "--porcelain"],
        cwd=cwd,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        check=True,
    )
    if not status.stdout.strip():
        return None

    # If the index has unmerged entries (e.g. from an interrupted merge/rebase),
    # git stash will fail with "needs merge / could not write index".  Clear the
    # conflict state with `git reset` so the stash can proceed.  Working-tree
    # changes are preserved; only the index conflict markers are dropped.
    unmerged = subprocess.run(
        git_cmd + ["ls-files", "--unmerged"],
        cwd=cwd,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )
    if unmerged.stdout.strip():
        print("→ Clearing unmerged index entries from a previous conflict...")
        subprocess.run(git_cmd + ["reset"], cwd=cwd, capture_output=True)

    from datetime import datetime, timezone

    stash_name = datetime.now(timezone.utc).strftime(
        "hermes-update-autostash-%Y%m%d-%H%M%S"
    )
    print("→ Local changes detected — stashing before update...")
    prev_stash = subprocess.run(
        git_cmd + ["rev-parse", "--verify", "refs/stash"],
        cwd=cwd,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    ).stdout.strip()
    push = subprocess.run(
        git_cmd + ["stash", "push", "--include-untracked", "-m", stash_name],
        cwd=cwd,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )
    if push.stdout.strip():
        print(push.stdout.strip())
    stash_probe = subprocess.run(
        git_cmd + ["rev-parse", "--verify", "refs/stash"],
        cwd=cwd,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )
    stash_ref = stash_probe.stdout.strip()
    stash_created = (
        stash_probe.returncode == 0 and bool(stash_ref) and stash_ref != prev_stash
    )

    if push.returncode != 0:
        if stash_created:
            # git stash push exits non-zero when it saved everything but could
            # not delete some swept untracked files from the working tree
            # (e.g. a root-owned directory: "warning: failed to remove ...:
            # Permission denied").  The stash entry is complete — the changes
            # are safe — so this is not a failure.  Leave the undeletable
            # files in place and continue the update.
            if push.stderr.strip():
                print(push.stderr.strip())
            print(
                "  ⚠ Some untracked files could not be removed from the "
                "working tree (permission denied)."
            )
            print(
                "    They were still saved to the stash and were left in "
                "place — the update will continue."
            )
            # A partially-failed stash push also aborts its working-tree
            # cleanup for TRACKED modifications — they are saved in the stash
            # but still dirty the tree, which would break the checkout/pull
            # that follows. Safe to reset: everything is in the stash entry.
            subprocess.run(
                git_cmd + ["reset", "--hard", "HEAD"],
                cwd=cwd,
                capture_output=True,
            )
        else:
            # No stash entry was created: the changes were NOT saved.  This
            # is a real failure — bail out before the update touches HEAD.
            print("✗ Could not stash local changes — update aborted.")
            if push.stderr.strip():
                print(f"  {push.stderr.strip().splitlines()[0]}")
            print(
                "  Commit, stash, or clean up your local changes manually, "
                "then re-run `hermes update`."
            )
            raise subprocess.CalledProcessError(
                push.returncode, push.args, output=push.stdout, stderr=push.stderr
            )

    return stash_ref

def _resolve_stash_selector(
    git_cmd: list[str], cwd: Path, stash_ref: str
) -> Optional[str]:
    stash_list = subprocess.run(
        git_cmd + ["stash", "list", "--format=%gd %H"],
        cwd=cwd,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        check=True,
    )
    for line in stash_list.stdout.splitlines():
        selector, _, commit = line.partition(" ")
        if commit.strip() == stash_ref:
            return selector.strip()
    return None

def _print_stash_cleanup_guidance(
    stash_ref: str, stash_selector: Optional[str] = None
) -> None:
    print(
        "  Check `git status` first so you don't accidentally reapply the same change twice."
    )
    print("  Find the saved entry with: git stash list --format='%gd %H %s'")
    if stash_selector:
        print(f"  Remove it with: git stash drop {stash_selector}")
    else:
        print(
            f"  Look for commit {stash_ref}, then drop its selector with: git stash drop stash@{{N}}"
        )

def _stash_apply_failed_only_on_existing_untracked(stderr: str) -> bool:
    """True when a ``git stash apply`` failure is ONLY about untracked files
    that already exist in the working tree.

    This is the tail end of the permission-denied autostash class: ``git stash
    push --include-untracked`` swept undeletable files (e.g. a root-owned
    ``packaging/`` directory) into the stash but could not remove them from
    disk.  On restore, git applies all tracked changes, then refuses to
    overwrite those still-present files (``already exists, no checkout`` /
    ``could not restore untracked files from stash``) and exits non-zero even
    though nothing was lost.  Any other error line (e.g. ``would be
    overwritten by merge`` / ``Aborting``) means the tracked apply itself
    failed and this returns False.
    """
    lines = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
    if not lines:
        return False
    saw_untracked_error = False
    for ln in lines:
        if "already exists, no checkout" in ln:
            saw_untracked_error = True
        elif "could not restore untracked files from stash" in ln:
            saw_untracked_error = True
        elif ln.startswith(("warning:", "hint:")):
            continue
        else:
            return False
    return saw_untracked_error

def _park_stashed_changes(stash_ref: str) -> None:
    """Leave a pre-update autostash parked instead of re-applying it.

    Used by ``hermes update --keep-stash`` (the desktop updater's mode): the
    stash made the update possible on a dirty tree, but local source edits
    must never be silently re-applied onto the updated code. Nothing is
    lost — the entry stays in ``git stash`` with printed recovery guidance.
    """
    print()
    print("ℹ️  Local changes were stashed before updating and were NOT re-applied (--keep-stash).")
    print(f"  Stash ref: {stash_ref}")
    print(f"  Restore manually with: git stash apply {stash_ref}")


def _restore_stashed_changes(
    git_cmd: list[str],
    cwd: Path,
    stash_ref: str,
    prompt_user: bool = False,
    input_fn=None,
) -> bool:
    if prompt_user:
        print()
        print("⚠ Local changes were stashed before updating.")
        print(
            "  Restoring them may reapply local customizations onto the updated codebase."
        )
        print("  Review the result afterward if Hermes behaves unexpectedly.")
        print("Restore local changes now? [Y/n]")
        if input_fn is not None:
            response = input_fn("Restore local changes now? [Y/n]", "y")
        else:
            try:
                response = input().strip().lower()
            except (EOFError, UnicodeDecodeError):
                # Mirror the config-migration prompt's fix: don't let a
                # terminal-encoding issue or a closed stdin crash the
                # update mid-restore. Falls through to the existing
                # skip-restore path below, which already explains how to
                # restore manually from git stash.
                response = "n"
        if response not in {"", "y", "yes"}:
            print("Skipped restoring local changes.")
            print("Your changes are still preserved in git stash.")
            print(f"Restore manually with: git stash apply {stash_ref}")
            return False

    print("→ Restoring local changes...")
    restore = subprocess.run(
        git_cmd + ["stash", "apply", stash_ref],
        cwd=cwd,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )

    # Check for unmerged (conflicted) files — can happen even when returncode is 0
    unmerged = subprocess.run(
        git_cmd + ["diff", "--name-only", "--diff-filter=U"],
        cwd=cwd,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )
    has_conflicts = bool(unmerged.stdout.strip())

    if restore.returncode != 0 and not has_conflicts and (
        _stash_apply_failed_only_on_existing_untracked(restore.stderr)
    ):
        # Permission-denied autostash tail end: the tracked changes applied
        # cleanly; the only "failure" is untracked files that never left the
        # working tree (git could not delete them at stash time, so it now
        # refuses to overwrite them). Their content was never touched —
        # nothing is lost. Treat as restored.
        print(
            "  ⚠ Some stashed untracked files already exist in the working "
            "tree and were kept as-is."
        )
    elif restore.returncode != 0 or has_conflicts:
        print("✗ Update pulled new code, but restoring local changes hit conflicts.")
        if restore.stdout.strip():
            print(restore.stdout.strip())
        if restore.stderr.strip():
            print(restore.stderr.strip())

        # Show which files conflicted
        conflicted_files = unmerged.stdout.strip()
        if conflicted_files:
            print("\nConflicted files:")
            for f in conflicted_files.splitlines():
                print(f"  • {f}")

        print("\nYour stashed changes are preserved — nothing is lost.")
        print(f"  Stash ref: {stash_ref}")

        # Always reset to clean state — leaving conflict markers in source
        # files makes hermes completely unrunnable (SyntaxError on import).
        # The user's changes are safe in the stash for manual recovery.
        subprocess.run(
            git_cmd + ["reset", "--hard", "HEAD"],
            cwd=cwd,
            capture_output=True,
        )
        print("Working tree reset to clean state.")
        print(f"Restore your changes later with: git stash apply {stash_ref}")
        # Don't sys.exit — the code update itself succeeded, only the stash
        # restore had conflicts.  Let cmd_update continue with pip install,
        # skill sync, and gateway restart.
        return False

    stash_selector = _resolve_stash_selector(git_cmd, cwd, stash_ref)
    if stash_selector is None:
        print(
            "⚠ Local changes were restored, but Hermes couldn't find the stash entry to drop."
        )
        print(
            "  The stash was left in place. You can remove it manually after checking the result."
        )
        _print_stash_cleanup_guidance(stash_ref)
    else:
        drop = subprocess.run(
            git_cmd + ["stash", "drop", stash_selector],
            cwd=cwd,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        if drop.returncode != 0:
            print(
                "⚠ Local changes were restored, but Hermes couldn't drop the saved stash entry."
            )
            if drop.stdout.strip():
                print(drop.stdout.strip())
            if drop.stderr.strip():
                print(drop.stderr.strip())
            print(
                "  The stash was left in place. You can remove it manually after checking the result."
            )
            _print_stash_cleanup_guidance(stash_ref, stash_selector)

    print("⚠ Local changes were restored on top of the updated codebase.")
    print("  Review `git diff` / `git status` if Hermes behaves unexpectedly.")
    return True

def _discard_stashed_changes(
    git_cmd: list[str],
    cwd: Path,
    stash_ref: str,
) -> bool:
    """Throw away a stash created before an update, without applying it.

    Used only on a NON-interactive update when the user has set
    ``updates.non_interactive_local_changes: discard`` — i.e. they've opted out
    of keeping local source edits on this machine. Drops the stash entry
    instead of re-applying it, so the working tree stays clean at the freshly
    pulled HEAD. Unlike ``git reset --hard`` + ``git clean -fd``, this only
    affects what was stashed (tracked changes + the untracked files we
    explicitly captured) — ignored paths like node_modules/venv/build outputs
    are never touched, since they were never stashed.

    Returns True if the stash was dropped, False on a git failure (in which
    case the stash is left in place for safety).
    """
    stash_selector = _resolve_stash_selector(git_cmd, cwd, stash_ref)
    if stash_selector is None:
        print(
            "⚠ Configured to discard local changes on non-interactive update, "
            "but Hermes couldn't find the stash entry to drop."
        )
        _print_stash_cleanup_guidance(stash_ref)
        return False

    drop = subprocess.run(
        git_cmd + ["stash", "drop", stash_selector],
        cwd=cwd,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )
    if drop.returncode != 0:
        print(
            "⚠ Configured to discard local changes, but Hermes couldn't drop "
            "the saved stash entry."
        )
        if drop.stderr.strip():
            print(f"  {drop.stderr.strip().splitlines()[0]}")
        _print_stash_cleanup_guidance(stash_ref, stash_selector)
        return False

    print("→ Discarded local source changes (updates.non_interactive_local_changes=discard).")
    return True

OFFICIAL_REPO_URLS = {
    "https://github.com/NousResearch/hermes-agent.git",
    "git@github.com:NousResearch/hermes-agent.git",
    "https://github.com/NousResearch/hermes-agent",
    "git@github.com:NousResearch/hermes-agent",
}

OFFICIAL_REPO_URL = "https://github.com/NousResearch/hermes-agent.git"

SKIP_UPSTREAM_PROMPT_FILE = ".skip_upstream_prompt"

def _get_origin_url(git_cmd: list[str], cwd: Path) -> Optional[str]:
    """Get the URL of the origin remote, or None if not set."""
    try:
        result = subprocess.run(
            git_cmd + ["remote", "get-url", "origin"],
            cwd=cwd,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None

def _is_fork(origin_url: Optional[str]) -> bool:
    """Check if the origin remote points to a fork (not the official repo)."""
    if not origin_url:
        return False
    # Normalize URL for comparison (strip trailing .git if present)
    normalized = origin_url.rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    for official in OFFICIAL_REPO_URLS:
        official_normalized = official.rstrip("/")
        if official_normalized.endswith(".git"):
            official_normalized = official_normalized[:-4]
        if normalized == official_normalized:
            return False
    return True

def _has_upstream_remote(git_cmd: list[str], cwd: Path) -> bool:
    """Check if an 'upstream' remote already exists."""
    try:
        result = subprocess.run(
            git_cmd + ["remote", "get-url", "upstream"],
            cwd=cwd,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        return result.returncode == 0
    except Exception:
        return False

def _add_upstream_remote(git_cmd: list[str], cwd: Path) -> bool:
    """Add the official repo as the 'upstream' remote. Returns True on success."""
    try:
        result = subprocess.run(
            git_cmd + ["remote", "add", "upstream", OFFICIAL_REPO_URL],
            cwd=cwd,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        return result.returncode == 0
    except Exception:
        return False

def _count_commits_between(git_cmd: list[str], cwd: Path, base: str, head: str) -> int:
    """Count commits on `head` that are not on `base`. Returns -1 on error."""
    try:
        result = subprocess.run(
            git_cmd + ["rev-list", "--count", f"{base}..{head}"],
            cwd=cwd,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception:
        pass
    return -1

def _should_skip_upstream_prompt() -> bool:
    """Check if user previously declined to add upstream."""
    from hermes_constants import get_hermes_home

    return (get_hermes_home() / SKIP_UPSTREAM_PROMPT_FILE).exists()

def _mark_skip_upstream_prompt():
    """Create marker file to skip future upstream prompts."""
    try:
        from hermes_constants import get_hermes_home

        (get_hermes_home() / SKIP_UPSTREAM_PROMPT_FILE).touch()
    except Exception:
        pass

def _sync_fork_with_upstream(git_cmd: list[str], cwd: Path) -> bool:
    """Attempt to push updated main to origin (sync fork).

    Returns True if push succeeded, False otherwise.
    """
    try:
        result = subprocess.run(
            git_cmd + ["push", "origin", "main", "--force-with-lease"],
            cwd=cwd,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        return result.returncode == 0
    except Exception:
        return False

def _sync_with_upstream_if_needed(git_cmd: list[str], cwd: Path) -> None:
    """Check if fork is behind upstream and sync if safe.

    This implements the fork upstream sync logic:
    - If upstream remote doesn't exist, ask user if they want to add it
    - Compare origin/main with upstream/main
    - If origin/main is strictly behind upstream/main, pull from upstream
    - Try to sync fork back to origin if possible
    """
    has_upstream = _has_upstream_remote(git_cmd, cwd)

    if not has_upstream:
        # Check if user previously declined
        if _should_skip_upstream_prompt():
            return

        # Ask user if they want to add upstream
        print()
        print("ℹ Your fork is not tracking the official Hermes repository.")
        print("  This means you may miss updates from NousResearch/hermes-agent.")
        print()
        try:
            response = (
                input("Add official repo as 'upstream' remote? [Y/n]: ").strip().lower()
            )
        except (EOFError, KeyboardInterrupt, UnicodeDecodeError):
            print()
            response = "n"

        if response in {"", "y", "yes"}:
            print("→ Adding upstream remote...")
            if _add_upstream_remote(git_cmd, cwd):
                print(
                    "  ✓ Added upstream: https://github.com/NousResearch/hermes-agent.git"
                )
                has_upstream = True
            else:
                print("  ✗ Failed to add upstream remote. Skipping upstream sync.")
                return
        else:
            print(
                "  Skipped. Run 'git remote add upstream https://github.com/NousResearch/hermes-agent.git' to add later."
            )
            _mark_skip_upstream_prompt()
            return

    # Fetch upstream main only. This sync compares upstream/main with
    # origin/main, so there's no reason to pull every upstream ref — and a bare
    # fetch drags in thousands of auto-generated branches.
    print()
    print("→ Fetching upstream...")
    try:
        subprocess.run(
            git_cmd + ["fetch", "upstream", "main", "--quiet"],
            cwd=cwd,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        print("  ✗ Failed to fetch upstream. Skipping upstream sync.")
        return

    # Compare origin/main with upstream/main
    origin_ahead = _count_commits_between(git_cmd, cwd, "upstream/main", "origin/main")
    upstream_ahead = _count_commits_between(
        git_cmd, cwd, "origin/main", "upstream/main"
    )

    if origin_ahead < 0 or upstream_ahead < 0:
        print("  ✗ Could not compare branches. Skipping upstream sync.")
        return

    # If origin/main has commits not on upstream, don't trample
    if origin_ahead > 0:
        print()
        print(f"ℹ Your fork has {origin_ahead} commit(s) not on upstream.")
        print("  Skipping upstream sync to preserve your changes.")
        print("  If you want to merge upstream changes, run:")
        print("    git pull upstream main")
        return

    # If upstream is not ahead, fork is up to date
    if upstream_ahead == 0:
        print("  ✓ Fork is up to date with upstream")
        return

    # origin/main is strictly behind upstream/main (can fast-forward)
    print()
    print(f"→ Fork is {upstream_ahead} commit(s) behind upstream")
    print("→ Pulling from upstream...")

    try:
        subprocess.run(
            git_cmd + ["pull", "--ff-only", "upstream", "main"],
            cwd=cwd,
            check=True,
        )
    except subprocess.CalledProcessError:
        print(
            "  ✗ Failed to pull from upstream. You may need to resolve conflicts manually."
        )
        return

    print("  ✓ Updated from upstream")

    # Try to sync fork back to origin
    print("→ Syncing fork...")
    if _sync_fork_with_upstream(git_cmd, cwd):
        print("  ✓ Fork synced with upstream")
    else:
        print(
            "  ℹ Got updates from upstream but couldn't push to fork (no write access?)"
        )
        print("    Your local repo is updated, but your fork on GitHub may be behind.")

def _invalidate_update_cache():
    """Delete the update-check cache for ALL profiles so no banner
    reports a stale "commits behind" count after a successful update.

    The git repo is shared across profiles — when one profile runs
    ``hermes update``, every profile is now current.
    """
    homes = []
    # Default profile home (Docker-aware — uses /opt/data in Docker)
    from hermes_constants import get_default_hermes_root

    default_home = get_default_hermes_root()
    homes.append(default_home)
    # Named profiles under <root>/profiles/
    profiles_root = default_home / "profiles"
    if profiles_root.is_dir():
        for entry in profiles_root.iterdir():
            if entry.is_dir():
                homes.append(entry)
    for home in homes:
        try:
            cache_file = home / ".update_check"
            if cache_file.exists():
                cache_file.unlink()
        except Exception:
            pass

def _write_marker_file(path: Path, *, label: str) -> None:
    """Drop an update-recovery breadcrumb. Never raises."""
    if _m()._pytest_owns_live_checkout(path.parent):
        logger.debug("Skipping %s marker under pytest (live checkout)", label)
        return
    try:
        path.write_text(
            f"started={_time.time()}\npid={os.getpid()}\n", encoding="utf-8"
        )
    except OSError as exc:
        logger.debug("Could not write %s marker: %s", label, exc)

def _write_update_incomplete_marker() -> None:
    """Drop the interrupted core-install breadcrumb. Never raises."""
    _write_marker_file(_m()._update_marker_path(), label="update-incomplete")

def _write_lazy_refresh_incomplete_marker() -> None:
    """Drop the interrupted lazy-refresh breadcrumb. Never raises."""
    _write_marker_file(_m()._lazy_refresh_marker_path(), label="lazy-refresh-incomplete")

def _format_concurrent_instances_message(
    matches: list[tuple[int, str]], scripts_dir: Path
) -> str:
    """Build a human-readable explanation + remediation hint for the user."""
    shim = scripts_dir / "hermes.exe"
    lines = ["✗ Another hermes.exe is running:"]
    for pid, name in matches:
        lines.append(f"    PID {pid}  {name}")
    lines.append("")
    lines.append(f"  Updating now would fail to overwrite {shim} because")
    lines.append("  Windows blocks REPLACE on a running executable.")
    lines.append("")
    lines.append("  Close Hermes Desktop, exit any open `hermes` REPLs, and")
    lines.append("  stop the gateway (`hermes gateway stop`) before retrying.")
    lines.append("")
    if matches:
        pid_args = " ".join(f"/PID {pid}" for pid, _ in matches)
        lines.append("  If you've already closed everything and these PIDs are")
        lines.append("  stale, terminate them directly, then retry the update:")
        lines.append(f"      taskkill {pid_args} /F")
        lines.append("")
    lines.append("  Override with `hermes update --force` if you've already")
    lines.append("  confirmed those processes will not write to the venv.")
    return "\n".join(lines)


def _classify_concurrent_instance(pid: int) -> str:
    """Return ``"gateway"`` when ``pid``'s command line is a gateway runtime.

    Delegates to ``_is_pausable_gateway`` — the same canonical
    ``gateway run`` matcher (``gateway.status.looks_like_gateway_command_line``,
    shlex-tokenized, profile-selector aware) used by the Desktop preflight
    exemption and the venv-holder guard fallback — so a PID classified as
    ``"gateway"`` here is exactly the set the pause/kill+restart machinery
    downstream will stop. That symmetry is what lets the pre-update
    concurrent gate skip the abort for gateway-only matches: the gateway is
    going to be stopped by ``_pause_windows_gateways_for_update()`` moments
    later anyway, so refusing the update just to make the user kill it
    manually is friction without benefit.

    Returns ``"non-gateway"`` when the cmdline doesn't match, and
    ``"unknown"`` when psutil can't read it (process gone, access denied,
    psutil missing). The gate treats ``"unknown"`` as non-gateway — we'd
    rather block an update we could have completed than proceed against a
    process we couldn't positively identify as a gateway.
    """
    try:
        import psutil  # noqa: PLC0415
    except Exception:
        return "unknown"

    try:
        proc = psutil.Process(int(pid))
        cmdline_list = proc.cmdline()
    except Exception:
        return "unknown"

    from hermes_cli._scan_venv_blockers import _is_pausable_gateway  # noqa: PLC0415

    cmdline = " ".join(cmdline_list or [])
    if _is_pausable_gateway(cmdline):
        return "gateway"
    return "non-gateway"


def _filter_non_gateway_concurrent_instances(
    matches: list[tuple[int, str]],
) -> list[tuple[int, str]]:
    """Return only the concurrent-instance matches that are NOT the gateway.

    Used by the pre-update concurrent gate to decide whether to abort
    ``hermes update``. If every concurrent instance is a gateway, the pause
    machinery (``_pause_windows_gateways_for_update``) and the post-update
    kill+restart block handle it — the update proceeds. If anything else (a
    TUI shell, a Hermes Desktop backend child, an unrelated ``hermes`` REPL)
    is in the list, the gate still aborts with the existing message, since
    those have no pause machinery downstream.
    """
    non_gateway: list[tuple[int, str]] = []
    for pid, name in matches:
        if _classify_concurrent_instance(pid) != "gateway":
            non_gateway.append((pid, name))
    return non_gateway

def _upgrade_pip_before_lazy_refresh(
    install_cmd_prefix: list[str],
    *,
    env: dict[str, str] | None = None,
) -> None:
    """Upgrade pip before lazy-backend refreshes.

    Older pip (e.g. 24.0 on Python 3.11) can fail setuptools-backed source
    builds during lazy installs and leave a partially-written venv (#57828).
    Never raises.
    """
    try:
        _m()._run_package_only_install(
            install_cmd_prefix + ["install", "--upgrade", "pip"],
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        logger.debug("pip upgrade before lazy refresh failed: %s", exc)


def _capture_active_lazy_features() -> list[str]:
    """Snapshot active lazy backends before a managed runtime is replaced."""
    try:
        from tools import lazy_deps

        return lazy_deps.active_features()
    except Exception as exc:
        logger.debug("Could not snapshot active lazy features: %s", exc)
        return []


def _capture_active_tool_dependencies() -> list[str]:
    """Snapshot Python dependencies installed explicitly through ``hermes tools``."""
    try:
        from hermes_cli import tools_config

        return tools_config.active_restorable_python_tool_dependencies()
    except Exception as exc:
        logger.debug("Could not snapshot active Hermes Tools dependencies: %s", exc)
        return []


def _restore_active_tool_dependencies(
    dependencies: list[str],
    install_cmd_prefix: list[str],
    *,
    env: dict[str, str] | None = None,
) -> None:
    """Restore allowlisted ``hermes tools`` dependencies into a rebuilt venv.

    The dependency names came from a pre-rebuild import probe and are resolved
    through a static package allowlist. Never raises: a failed optional tool
    must not block the core update, but the user must be told what stayed
    unavailable.
    """
    if not dependencies:
        return

    try:
        from hermes_cli import tools_config
    except Exception as exc:
        logger.debug("Hermes Tools dependency restore skipped (import failed): %s", exc)
        return

    target_python = _m()._resolve_install_target_python(install_cmd_prefix, env)
    missing: list[tuple[str, tuple[str, ...]]] = []
    for name in dependencies:
        spec = tools_config.restorable_python_tool_dependency(name)
        if spec is None:
            continue
        module_name, install_args = spec
        if target_python is not None:
            try:
                probe = subprocess.run(
                    [
                        str(target_python),
                        "-c",
                        "import importlib.util,sys; "
                        "raise SystemExit(0 if importlib.util.find_spec(sys.argv[1]) else 1)",
                        module_name,
                    ],
                    capture_output=True,
                    env=env,
                    check=False,
                )
                if probe.returncode == 0:
                    continue
            except (subprocess.SubprocessError, OSError):
                # An indeterminate probe is safer to repair than to treat as
                # proof that a pre-rebuild dependency survived.
                pass
        missing.append((name, install_args))

    if not missing:
        return

    print()
    print(f"→ Restoring {len(missing)} Hermes Tools dependency set(s)...")
    restored: list[str] = []
    failed: list[tuple[str, str]] = []
    for name, install_args in missing:
        try:
            _m()._run_package_only_install(
                install_cmd_prefix + ["install", *install_args, "--quiet"],
                env=env,
            )
            restored.append(name)
        except Exception as exc:
            # This is best-effort recovery for optional tooling. Unexpected
            # installer failures must be surfaced without aborting the core
            # runtime update.
            failed.append((name, str(exc)))

    if restored:
        print(f"  ✓ {len(restored)} restored: {', '.join(restored)}")
    for name, reason in failed:
        if len(reason) > 200:
            reason = reason[:200] + "..."
        print(f"  ⚠ {name} failed to restore: {reason}")


def _refresh_active_lazy_features(
    install_cmd_prefix: list[str] | None = None,
    *,
    env: dict[str, str] | None = None,
    features: list[str] | None = None,
) -> bool:
    """Refresh lazy-installed backends after a code update.

    When pyproject.toml's ``[all]`` extra was slimmed down (May 2026), most
    optional backends moved to ``tools/lazy_deps.py`` and only install on
    first use. ``hermes update`` runs ``uv pip install -e .[all]`` which
    leaves those packages untouched — so if we bump a pin in
    :data:`LAZY_DEPS` (CVE response, transitive bug fix), users who already
    activated the backend keep the stale version forever.

    This function asks lazy_deps which features the user has previously
    activated and reinstalls them under the current pins. Features the
    user never enabled stay quiet — no churn for cold backends.

    Returns True when the venv is safe to use (refresh succeeded, or no
    active lazy backends, or post-failure import repair succeeded). Returns
    False when a failed lazy install left broken core imports that automatic
    repair could not fix (#57828).

    Never raises. A failure here must not block the rest of the update.
    """
    try:
        from tools import lazy_deps
    except Exception as exc:
        logger.debug("Lazy refresh skipped (import failed): %s", exc)
        return True

    if features is None:
        try:
            active = lazy_deps.active_features()
        except Exception as exc:
            logger.debug("Lazy refresh skipped (active_features failed): %s", exc)
            return True
    else:
        active = features

    if not active:
        return True

    print()
    print(f"→ Refreshing {len(active)} active lazy backend(s)...")

    unexpected_failure = False
    try:
        if features is None:
            results = lazy_deps.refresh_active_features(prompt=False)
        else:
            results = lazy_deps.restore_features(active)
    except Exception as exc:
        # refresh_active_features is documented as never-raise, but defend
        # the update flow against future regressions.
        print(f"  ⚠ Lazy refresh failed unexpectedly: {exc}")
        results = {}
        unexpected_failure = True

    refreshed = [f for f, s in results.items() if s in {"refreshed", "restored"}]
    current = [f for f, s in results.items() if s == "current"]
    failed = [(f, s) for f, s in results.items() if s.startswith("failed:")]
    skipped = [(f, s) for f, s in results.items() if s.startswith("skipped:")]

    if refreshed:
        print(f"  ↑ {len(refreshed)} refreshed: {', '.join(refreshed)}")
    if current:
        print(f"  ✓ {len(current)} already current")
    if skipped:
        # Most common reason: security.allow_lazy_installs=false. Show one
        # line so the user knows why; not an error.
        names = ", ".join(f for f, _ in skipped)
        reason = skipped[0][1].split(": ", 1)[-1]
        print(f"  · {len(skipped)} skipped ({reason}): {names}")

    if not failed and not unexpected_failure:
        return True

    for feature, status in failed:
        reason = status.split(": ", 1)[-1]
        # Clip noisy pip stderr to keep update output legible.
        if len(reason) > 200:
            reason = reason[:200] + "..."
        print(f"  ⚠ {feature} failed to refresh: {reason}")

    if install_cmd_prefix is None:
        print("  ⚠ Lazy refresh failed; rerun `hermes update` once resolved.")
        return False

    # Immediate import-based recovery — metadata-only verifiers miss the case
    # where DISTRIBUTION-INFO remains but import files were wiped (#57828).
    # Unavailable probes are indeterminate, not healthy — keep the lazy marker.
    status = _m()._repair_venv_via_import_probes(install_cmd_prefix, env=env)
    if status == "repaired":
        print(
            "  Lazy backend(s) keep their previous version until refresh succeeds."
        )
        return True
    if status == "healthy":
        print(
            "  Lazy backend(s) keep their previous version; probed packages look intact."
        )
        print("  Rerun `hermes update` once the upstream issue is resolved.")
        return True
    if status == "indeterminate":
        print(
            "  ⚠ Leaving `.lazy-refresh-incomplete` until import probes can confirm health."
        )
    return False

def _refresh_active_memory_provider_dependencies() -> None:
    """Refresh pip dependencies for the configured external memory provider.

    Memory-provider bridge packages are declared in each provider's
    ``plugin.yaml`` (plus mode-dependent extras like Hindsight's
    ``hindsight-all``), NOT in Hermes' editable-install extras or
    ``LAZY_DEPS`` alone — so the core dependency reinstall above can strip
    or downgrade them (#53272 mem0ai, #70636 hindsight-embed). Re-run the
    provider's declared install for the ACTIVE provider only, after the
    core install and lazy refresh, so the last write to any shared package
    is the one the active provider needs.

    Never raises. A failure here must not block the rest of the update.
    """
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
    except Exception as exc:
        logger.debug("Memory provider refresh skipped (config load failed): %s", exc)
        return

    provider = ""
    if isinstance(cfg, dict):
        memory_cfg = cfg.get("memory")
        if isinstance(memory_cfg, dict):
            if memory_cfg.get("enabled") is False:
                return
            provider = str(memory_cfg.get("provider") or "").strip()

    # "default" / empty is the built-in file-backed store — no pip deps.
    if not provider or provider in {"default", "builtin", "none"}:
        return

    try:
        from hermes_cli.memory_setup import _install_dependencies
    except Exception as exc:
        logger.debug("Memory provider refresh skipped (import failed): %s", exc)
        return

    print()
    print(f"→ Refreshing active memory provider dependencies ({provider})...")

    try:
        _install_dependencies(provider, force=True)
    except Exception as exc:
        print(f"  ⚠ {provider} dependencies failed to refresh: {exc}")

def _is_android_python() -> bool:
    return _m().sys.platform == "android"

def _install_psutil_android_compat(
    install_cmd_prefix: list[str],
    *,
    env: dict[str, str] | None = None,
) -> None:
    """Install psutil on Android by patching upstream platform detection.

    psutil's setup currently gates Linux sources behind
    ``sys.platform.startswith('linux')``. On Termux Python reports
    ``sys.platform == 'android'``, so setup aborts with
    "platform android is not supported" despite compiling fine when using the
    Linux source path.

    We patch only the extracted build tree used for this install attempt;
    nothing is persisted in the repository.

    Stopgap: remove this once https://github.com/giampaolo/psutil/pull/2762
    merges and ships in a release. The standalone installer script uses the
    same shared helper and should be removed together.
    """
    import tempfile
    import urllib.request
    from hermes_cli.psutil_android import PSUTIL_URL, prepare_patched_psutil_sdist

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "psutil.tar.gz"
        urllib.request.urlretrieve(PSUTIL_URL, archive)
        src_root = prepare_patched_psutil_sdist(archive, tmp_path)

        _m()._run_install_with_heartbeat(
            install_cmd_prefix + ["install", "--no-build-isolation", str(src_root)],
            env=env,
        )

def _ensure_uv_for_termux(pip_cmd: list[str]) -> str | None:
    """Best-effort uv bootstrap on Termux for faster update installs.

    The normal path (``ensure_uv()`` in managed_uv) installs the managed
    standalone uv into ``$HERMES_HOME/bin/uv``, but on Termux the official
    installer may not work (glibc vs bionic).  Prefer a uv already on PATH
    (e.g. ``pkg install uv``); only if there is none do we fall back to a
    wheel-only ``pip install uv`` so we never source-build the Rust crate.
    """
    from hermes_cli.managed_uv import resolve_uv

    existing = resolve_uv()
    if existing:
        return existing
    if not _m()._is_termux_env():
        return None
    # A Termux-packaged uv lands on PATH but not in the managed bin dir, so
    # resolve_uv() misses it. Use it before pip, which has no Android wheel and
    # would otherwise build uv from source on a low-memory device.
    system_uv = shutil.which("uv")
    if system_uv:
        return system_uv
    try:
        print("  → Termux detected: trying to install uv for faster dependency updates...")
        result = subprocess.run(
            pip_cmd + ["install", "uv", "--only-binary", ":all:"],
            cwd=_m().PROJECT_ROOT,
            check=False,
        )
        if result.returncode != 0:
            return None
    except Exception:
        pass
    # After pip install, check managed path first, then PATH
    return resolve_uv() or shutil.which("uv")

def _npm_manifest_paths() -> tuple[Path, ...]:
    """Manifests whose changes must defeat the update-skip.

    The lockfile alone is NOT a sufficient key: on a local checkout a dev
    can edit package.json (root or a workspace) without running npm — the
    lockfile is then unchanged but `hermes update` is exactly the step
    expected to sync node_modules (via the `npm install` fallback in
    _run_npm_install_deterministic).

    The workspace list is pulled from the root package.json's `workspaces`
    globs (npm's own source of truth) rather than hardcoded, so adding a
    workspace can never silently escape the skip key. Every workspace
    manifest belongs in the key — desktop included, even though the
    install only names ui-tui and web — because the single lockfile spans
    the whole workspace graph, so any manifest edit can put the lockfile
    out of sync and change what the install must do. Falls back to hashing
    just root manifests if package.json is unreadable (never skips more
    than main would have installed).
    """
    root_pkg = _m().PROJECT_ROOT / "package.json"
    paths = [_m().PROJECT_ROOT / "package-lock.json", root_pkg]
    try:
        workspaces = json.loads(root_pkg.read_text(encoding="utf-8")).get(
            "workspaces", []
        )
        if isinstance(workspaces, dict):  # legacy {"packages": [...]} form
            workspaces = workspaces.get("packages", [])
        for pattern in workspaces:
            for match in sorted(_m().PROJECT_ROOT.glob(str(pattern))):
                manifest = match / "package.json"
                if manifest.is_file():
                    paths.append(manifest)
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return tuple(paths)

def _npm_manifests_digest() -> str | None:
    """Combined sha256 over the lockfile + all workspace package.json files.

    Returns None when the lockfile is missing (never skip then).
    """
    if not (_m().PROJECT_ROOT / "package-lock.json").exists():
        return None
    h = hashlib.sha256()
    for p in _npm_manifest_paths():
        h.update(str(p.relative_to(_m().PROJECT_ROOT)).encode())
        try:
            h.update(p.read_bytes())
        except OSError:
            h.update(b"<missing>")
    return h.hexdigest()

def _npm_lockfile_changed(hermes_root: Path) -> bool:
    current = _npm_manifests_digest()
    if current is None:
        return True
    # Also check that node_modules exists; a matching hash with missing
    # node_modules means the cache was recorded by another checkout.
    if not (_m().PROJECT_ROOT / "node_modules").is_dir():
        return True
    # A matching lockfile hash over a tree whose web build toolchain never
    # landed must NOT skip the reinstall — otherwise every later `hermes
    # update` keeps rebuilding against a half-installed tree and serving a
    # stale dist.
    web_dir = _m().PROJECT_ROOT / "web"
    if (web_dir / "package.json").is_file() and not _web_build_toolchain_ready(
        *_web_toolchain_roots(web_dir)
    ):
        return True
    try:
        # Key the cache by PROJECT_ROOT so parallel worktrees don't collide.
        cache_key = hashlib.sha256(str(_m().PROJECT_ROOT).encode()).hexdigest()[:12]
        cache_file = hermes_root / f".npm_lock_hash_{cache_key}"
        if not cache_file.exists():
            return True
        return cache_file.read_text(encoding="utf-8").strip() != current
    except OSError:
        return True

def _record_npm_lockfile_hash(hermes_root: Path) -> None:
    digest = _npm_manifests_digest()
    if digest is None:
        return
    try:
        cache_key = hashlib.sha256(str(_m().PROJECT_ROOT).encode()).hexdigest()[:12]
        cache_file = hermes_root / f".npm_lock_hash_{cache_key}"
        cache_file.write_text(digest, encoding="utf-8")
    except OSError:
        logger.debug("Could not write npm lockfile hash cache")

def _repair_node_deps_on_current_checkout(print_completion) -> None:
    """Repair Node deps on the ``commit_count == 0`` path (#77211).

    A current checkout does not imply healthy Node deps: a previous npm
    install may have failed (EBADENGINE from a node/npm mismatch, network
    timeout, interrupted install) and its error message says to "re-run
    hermes update" — but the early return never reached the Node refresh,
    so that repair advice could never work. ``_update_node_dependencies``
    self-gates on the lockfile hash, which is only recorded after a
    SUCCESSFUL npm install (and re-trips when node_modules is missing or
    the web toolchain never landed), so this is a cheap no-op on healthy
    installs and a real repair after a failed one.
    """
    node_failures = _update_node_dependencies()
    if node_failures:
        print(f"  ⚠ Node.js refresh failed for: {', '.join(node_failures)}")
        print("    Fix npm and re-run `hermes update`.")
        print_completion(
            "⚠ Checkout is current, but Node.js dependencies could not be repaired."
        )
        return
    # Pair the refresh with the web build like every other
    # _update_node_dependencies call site; it staleness-checks internally,
    # so this is a no-op when nothing changed.
    _m()._build_web_ui(_m().PROJECT_ROOT / "web")
    print_completion("✓ Already up to date!")


def _update_node_dependencies() -> list[str]:
    """Refresh Node deps for the ui-tui and web workspaces.

    Returns the list of labels whose npm install failed (empty on success),
    so the caller can treat a Node refresh failure as a partial update rather
    than silently reporting ``Update complete!`` (#30271).
    """
    if not (_m().PROJECT_ROOT / "package.json").exists():
        return []

    npm = _m()._resolve_node_runtime_npm()
    if not npm:
        # If the only npm reachable inside this WSL shell is the Windows one,
        # flag it loudly: silently skipping leaves ui-tui deps stale while the
        # rest of the update proceeds, and running it would corrupt the tree.
        from hermes_constants import is_wsl

        path_npm = shutil.which("npm")
        if is_wsl() and path_npm and _m()._is_windows_npm_path(path_npm):
            print("→ Updating Node.js dependencies...")
            print("  ⚠ Skipped: only a Windows npm is reachable from this WSL shell.")
            print("    Install Node.js inside the WSL distro (nvm, or your distro's")
            print("    package manager), then re-run `hermes update`.")
            failed = []
            if any(
                (_m().PROJECT_ROOT / workspace / "package.json").exists()
                for workspace in ("ui-tui", "web")
            ):
                failed.append("ui-tui, web workspaces")
            return failed
        return []

    from hermes_constants import get_default_hermes_root

    # This cache describes PROJECT_ROOT/node_modules, which is shared by every
    # Hermes profile using this checkout. Keep one per-checkout cache under the
    # shared Hermes root rather than rerunning npm once per named profile.
    shared_hermes_root = get_default_hermes_root()

    # Best-effort: warm npx's cache for agent-browser (#43564). Runs before
    # the lockfile-unchanged early return below since that's the common
    # `hermes update` case. Synchronous and can block ~11s on a true cold
    # cache (~0.4s once warm) — print first so that doesn't look like a hang.
    print("→ Warming npx cache for agent-browser...")
    try:
        from tools.browser_tool import warm_agent_browser_npx_cache
        warm_agent_browser_npx_cache()
    except Exception:
        pass

    if not _m()._npm_lockfile_changed(shared_hermes_root):
        logger.info("npm lockfile unchanged, skipping npm install")
        return []

    # Root package.json has no dependencies of its own (agent-browser and
    # @streamdown/math were moved out — see #43564): agent-browser resolves
    # at runtime via `npx agent-browser` (tools/browser_tool.py), and
    # @streamdown/math is a desktop-only import now declared in
    # apps/desktop/package.json. That means a plain workspace-scoped install
    # can never prune anything root-only, so we only need to name the
    # workspaces the CLI/TUI/web build actually requires. apps/desktop pulls
    # in Electron as a devDependency with a ~200MB postinstall download, so
    # it's deliberately never named here — desktop deps install on demand
    # (see _desktop_build_needed).
    print("→ Updating Node.js dependencies...")

    def _partial_update_failure(*labels: str) -> list[str]:
        print()
        print("  ⚠ Node.js dependency refresh did not complete cleanly; the")
        print("    installation may be in a mixed state (updated code, stale Node")
        print("    deps). Fix npm and re-run `hermes update`.")
        return list(labels)

    install_args = [
        "--no-fund", "--no-audit", "--prefer-offline", "--progress=false",
        "--workspace", "ui-tui", "--workspace", "web",
        # Root package.json's own devDependencies (the shared ESLint flat
        # config every workspace's eslint.config.mjs imports) are otherwise
        # pruned by this scoped install, same as agent-browser/@streamdown
        # math used to be before they moved out of root entirely (#43564).
        # Unlike those, root's devDependencies have nowhere else to live —
        # this flag still excludes apps/desktop, which is never named above.
        "--include-workspace-root",
    ]

    from hermes_constants import with_hermes_node_path

    nixos_env = with_hermes_node_path(_m()._nixos_build_env())

    # NOTE: capture_output=False here is deliberate (#18840) — optional
    # postinstall scripts print download progress, and capturing it makes a
    # long download look hung. The chatty npm-deprecation noise during
    # `hermes update` comes from the *desktop* build, not this step; that
    # one is captured to update.log.
    result = _m()._run_npm_install_deterministic(
        npm,
        _m().PROJECT_ROOT,
        extra_args=tuple(install_args),
        capture_output=False,
        env=nixos_env,
    )
    if result.returncode == 0:
        _record_npm_lockfile_hash(shared_hermes_root)
        print("  ✓ ui-tui, web workspaces installed (desktop skipped)")
        failures: list[str] = []
    else:
        print("  ⚠ npm install failed")
        stderr = (result.stderr or "").strip() if result.stderr else ""
        if stderr:
            print(f"    {stderr.splitlines()[-1]}")
        failures = _partial_update_failure("ui-tui, web workspaces")

    return failures

def _log_only_write(text: str) -> None:
    """Write ``text`` to ``~/.hermes/logs/update.log`` only, never the terminal.

    During ``hermes update`` ``sys.stdout`` is an ``_UpdateOutputStream`` that
    mirrors to both the terminal and ``update.log``. Loud, low-signal
    subprocess output (npm installs, the Electron/vite build, the cua-driver
    installer's "Next steps" wall) should be captured and tucked into the log
    so failures stay debuggable, without flooding the user's terminal. This
    reaches past the mirroring stream straight to the underlying log handle.
    """
    if not text:
        return
    stream = _m().sys.stdout
    log_file = getattr(stream, "_log", None)
    if log_file is None:
        return
    try:
        log_file.write(text if text.endswith("\n") else text + "\n")
        log_file.flush()
    except Exception:
        pass

def _run_logged_subprocess(cmd, *, cwd=None, env=None):
    """Run ``cmd`` capturing combined output into update.log (not the terminal).

    Returns the ``CompletedProcess`` (with ``stdout`` populated) so the caller
    can decide whether to surface the captured output on failure.
    """
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    _log_only_write(result.stdout or "")
    return result

def _classify_fetch_failure(stderr: str) -> str:
    """Map git-fetch stderr to a one-line, user-facing diagnosis.

    Order matters: curl surfaces HTTP failures as
    ``fatal: unable to access '<url>': The requested URL returned error: 429``,
    so the rate-limit/outage checks must run BEFORE the generic
    "unable to access" network check or a GitHub 429/5xx gets misreported as a
    local network problem. The caller always prints the first raw stderr line
    alongside this diagnosis — the friendly message adds guidance, it never
    replaces the wire error.
    """

    def _has_http_code(*codes: str) -> bool:
        return any(
            f"HTTP {code}" in stderr or f"returned error: {code}" in stderr
            for code in codes
        )

    if _has_http_code("429") or "rate limit" in stderr.lower():
        return (
            "✗ GitHub is rate limiting requests or having an outage (HTTP 429)"
            " — try again in 5 minutes."
        )
    if _has_http_code("500", "502", "503", "504"):
        return (
            "✗ GitHub appears to be having an outage — try again in a few"
            " minutes (https://www.githubstatus.com)."
        )
    if "Could not resolve host" in stderr or "unable to access" in stderr:
        return "✗ Network error — cannot reach the remote repository."
    if "Authentication failed" in stderr or "could not read Username" in stderr:
        return "✗ Authentication failed — check your git credentials or SSH key."
    return "✗ Failed to fetch updates from origin."


def _print_fetch_failure(stderr: str) -> None:
    """Print the classified diagnosis plus the first raw stderr line."""
    stderr = (stderr or "").strip()
    print(_classify_fetch_failure(stderr))
    if stderr:
        print(f"  {stderr.splitlines()[0]}")


def _cmd_update_check(branch: str = "main", *, branch_explicit: bool = False):
    """Implement ``hermes update --check``: fetch and report without installing.

    ``branch`` selects which branch the check compares against. Default is
    "main"; callers can pass another branch to ask "are there new commits
    on origin/<branch>?" without performing the update.

    ``branch_explicit`` is True iff the caller passed --branch on the CLI.
    Installs that can't honor non-default branches (e.g. Docker) surface a
    one-line notice instead of silently dropping the flag.
    """
    from hermes_cli.config import (
        detect_install_method,
        is_nix_install_method,
        recommended_update_command_for_method,
    )
    method = detect_install_method(_m().PROJECT_ROOT)
    if method == "docker":
        # Docker can't ``git fetch`` from within the container.  Surface the
        # same long-form ``docker pull`` guidance ``hermes update`` (apply
        # path) uses — telling the user to "reinstall via curl" or that
        # ".git is missing" would point them at the wrong remediation.
        from hermes_cli.config import format_docker_update_message
        print(format_docker_update_message())
        sys.exit(1)

    if is_nix_install_method(method) or method == "apt":
        print(recommended_update_command_for_method(method))
        sys.exit(1)

    git_dir = _m().PROJECT_ROOT / ".git"
    if not git_dir.exists():
        print("✗ Not a git repository — cannot check for updates.")
        sys.exit(1)

    git_cmd = ["git"]
    if sys.platform == "win32":
        git_cmd = ["git", "-c", "windows.appendAtomically=false"]

    # A crashed/interrupted fetch can leave .git/shallow.lock (or another git
    # lock file) behind; every later fetch then fails with "File exists" and
    # the check reports a hard failure (or, in the banner path, silently
    # compares stale refs). Self-heal abandoned locks before fetching.
    from hermes_cli.gitlock import clear_stale_git_locks

    cleared = clear_stale_git_locks(_m().PROJECT_ROOT)
    for lock_path in cleared:
        print(f"  (removed stale git lock: {lock_path})")

    # Fetch only the branch we compare against; prefer upstream as the canonical
    # reference. A bare `git fetch <remote>` pulls every ref, and this repo has
    # thousands of auto-generated branches, so scope the fetch to <branch>.
    # Note: upstream/<branch> may not exist for non-main branches (a fork's
    # bb/gui has no upstream counterpart), so when the caller picks a
    # non-default branch we skip the upstream probe and use origin directly.
    # Installer checkouts are shallow (`git clone --depth 1`). A plain
    # `git fetch` would unshallow the repo (dragging in the whole history —
    # the exact cost the shallow clone avoided) and the rev-list count below
    # would then report a huge bogus "behind" number. Detect shallow up front:
    # fetch with --depth 1 to preserve the boundary and report presence-only.
    is_shallow = (
        subprocess.run(
            git_cmd + ["rev-parse", "--is-shallow-repository"],
            cwd=_m().PROJECT_ROOT,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        ).stdout.strip()
        == "true"
    )
    depth_args = ["--depth", "1"] if is_shallow else []

    if branch == "main":
        # Probe locally (~6 ms) whether an 'upstream' remote exists at all
        # before spending a network fetch on it. Non-fork installs have no
        # 'upstream' remote, and the old flow burned a failed network attempt
        # (~0.3-1 s) on every --check before falling back to origin.
        has_upstream_remote = (
            subprocess.run(
                git_cmd + ["remote", "get-url", "upstream"],
                cwd=_m().PROJECT_ROOT,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            ).returncode
            == 0
        )
        fetch_result = None
        if has_upstream_remote:
            print("→ Fetching from upstream...")
            fetch_result = subprocess.run(
                git_cmd + ["fetch"] + depth_args + ["upstream", branch],
                cwd=_m().PROJECT_ROOT,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            )
        if fetch_result is not None and fetch_result.returncode == 0:
            upstream_exists = True
            compare_branch = f"upstream/{branch}"
        else:
            # No upstream remote, or the upstream fetch failed — use origin.
            print("→ Fetching from origin...")
            fetch_result = subprocess.run(
                git_cmd + ["fetch"] + depth_args + ["origin", branch],
                cwd=_m().PROJECT_ROOT,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            )
            upstream_exists = False
            compare_branch = f"origin/{branch}"
    else:
        # Non-default branch: compare against origin/<branch> directly.
        print("→ Fetching from origin...")
        fetch_result = subprocess.run(
            git_cmd + ["fetch"] + depth_args + ["origin", branch],
            cwd=_m().PROJECT_ROOT,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        upstream_exists = False
        compare_branch = f"origin/{branch}"

    if fetch_result.returncode != 0:
        _print_fetch_failure(fetch_result.stderr)
        sys.exit(1)

    # Verify the compare ref actually exists before asking rev-list about it.
    # Without this, `git rev-list HEAD..origin/<bogus> --count` exits 128 and
    # (with check=True) raises CalledProcessError, surfacing a Python
    # traceback. Friendlier to detect-and-report.
    verify_result = subprocess.run(
        git_cmd + ["rev-parse", "--verify", "--quiet", compare_branch],
        cwd=_m().PROJECT_ROOT,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )
    if verify_result.returncode != 0:
        print(f"✗ Branch '{branch}' not found on {compare_branch.split('/', 1)[0]}.")
        sys.exit(1)

    if is_shallow:
        # No history to count across the shallow boundary. Compare tip SHAs
        # (mirrors the banner's _check_via_local_git), then try to recover the
        # exact count via the GitHub compare API — the remote graph is complete
        # even when the local one is truncated.
        head_sha = subprocess.run(
            git_cmd + ["rev-parse", "HEAD"],
            cwd=_m().PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
        ).stdout.strip()
        target_sha = subprocess.run(
            git_cmd + ["rev-parse", compare_branch],
            cwd=_m().PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
        ).stdout.strip()
        if head_sha and target_sha and head_sha == target_sha:
            print("✓ Already up to date.")
        else:
            from hermes_cli.banner import _github_compare_behind
            from hermes_cli.config import recommended_update_command

            counted = _github_compare_behind(head_sha, target_sha)
            if counted == 0:
                # Local commits on top of the remote tip — not behind.
                print("✓ Already up to date.")
                return
            if counted is not None:
                commits_word = "commit" if counted == 1 else "commits"
                print(f"⚕ Update available: {counted} {commits_word} behind {compare_branch}.")
            else:
                print(f"⚕ Update available (behind {compare_branch}).")
            print(f"  Run '{recommended_update_command()}' to install.")
        return

    rev_result = subprocess.run(
        git_cmd + ["rev-list", f"HEAD..{compare_branch}", "--count"],
        cwd=_m().PROJECT_ROOT,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        check=True,
    )
    behind = int(rev_result.stdout.strip())

    if behind == 0:
        print("✓ Already up to date.")
    else:
        commits_word = "commit" if behind == 1 else "commits"
        print(f"⚕ Update available: {behind} {commits_word} behind {compare_branch}.")
        from hermes_cli.config import recommended_update_command

        print(f"  Run '{recommended_update_command()}' to install.")

def _ensure_fhs_path_guard() -> None:
    """Ensure /usr/local/bin is on PATH for RHEL-family root non-login shells.

    Mirrors the post-symlink probe added to ``scripts/install.sh`` so that
    existing FHS-layout root installs on RHEL/CentOS/Rocky/Alma 8+ get
    repaired on ``hermes update`` without requiring a reinstall.  The
    installer's assumption that ``/usr/local/bin`` is on PATH for every
    standard shell breaks on those distros in non-login interactive shells
    (su, sudo -s, tmux panes, some web terminals): /etc/bashrc doesn't
    add /usr/local/bin and /root/.bash_profile doesn't either.  Symptom:
    ``hermes`` prints ``command not found`` even though the symlink lives
    at /usr/local/bin/hermes.

    Silent no-op on: non-Linux, non-root, non-FHS installs, and any system
    where ``bash -i -c 'command -v hermes'`` already resolves.  Idempotent.
    """
    if _m().sys.platform != "linux":
        return
    try:
        if os.geteuid() != 0:  # windows-footgun: ok — Linux FHS helper, guarded by sys.platform == "linux" above + AttributeError catch
            return
    except AttributeError:
        return
    # Only act when this is actually an FHS-layout install (command link at
    # /usr/local/bin/hermes, code at /usr/local/lib/hermes-agent).
    fhs_link = Path("/usr/local/bin/hermes")
    if not fhs_link.is_symlink() and not fhs_link.exists():
        return

    # Probe a fresh non-login interactive bash the way the user will use it.
    # ``bash -i -c`` sources ~/.bashrc but NOT ~/.bash_profile or /etc/profile,
    # which is the exact scenario where RHEL root loses /usr/local/bin.
    home = os.environ.get("HOME") or "/root"
    try:
        probe = subprocess.run(
            [
                "env",
                "-i",
                f"HOME={home}",
                f"TERM={os.environ.get('TERM', 'dumb')}",
                "bash",
                "-i",
                "-c",
                "command -v hermes",
            ],
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return  # no bash or probe hung — don't block update on this
    if probe.returncode == 0:
        return  # already on PATH, nothing to do

    path_line = 'export PATH="/usr/local/bin:$PATH"'
    path_comment = (
        "# Hermes Agent — ensure /usr/local/bin is on PATH " "(RHEL non-login shells)"
    )
    wrote_any = False
    for candidate in (".bashrc", ".bash_profile"):
        cfg = Path(home) / candidate
        if not cfg.is_file():
            continue
        try:
            existing = cfg.read_text(errors="replace", encoding="utf-8")
        except OSError:
            continue
        # Idempotency: skip if any uncommented PATH= line already references
        # /usr/local/bin.  Mirrors the grep pattern used by install.sh.
        already_guarded = any(
            "/usr/local/bin" in line
            and "PATH" in line
            and not line.lstrip().startswith("#")
            for line in existing.splitlines()
        )
        if already_guarded:
            continue
        try:
            with cfg.open("a", encoding="utf-8") as f:
                f.write("\n" + path_comment + "\n" + path_line + "\n")
        except OSError as e:
            print(f"  ⚠ Could not update {cfg}: {e}")
            continue
        print(f"  ✓ Added /usr/local/bin to PATH in {cfg}")
        wrote_any = True
    if wrote_any:
        print("    (reload your shell or run 'source ~/.bashrc' to pick it up)")

def _ensure_acp_launcher() -> None:
    r"""Self-heal: install a ``hermes-acp`` launcher next to the ``hermes`` one.

    Mirrors the launcher block in ``scripts/install.sh`` so existing installs
    gain the ACP command on ``hermes update`` without a reinstall.  ACP hosts
    (Zed, JetBrains, Buzz Desktop) spawn the agent by resolving the
    ``hermes-acp`` command name against the login-shell PATH; the console
    script of that name lives inside the install's venv, which is not on that
    PATH, so those hosts report Hermes as not installed even when it is.

    The shim simply delegates to the sibling ``hermes`` launcher with the
    ``acp`` subcommand, which makes it correct for every install layout
    (venv wrapper, FHS symlink, pipx/pip console script) without having to
    reconstruct interpreter/entrypoint paths.

    No-op on Windows (install.ps1 stages the ``hermes`` / ``hermes-acp``
    launchers into the managed binary dir ``$HermesHome\bin`` and puts THAT
    on the user PATH — never the whole ``venv\Scripts`` dir, which would
    shadow the user's ``python`` (#83797); when those launchers go missing,
    ``hermes_cli._install_repair.ensure_windows_bin_launchers`` re-stages
    them) and wherever a ``hermes-acp`` is already present next to the
    ``hermes`` command.  Unwritable directories (e.g. ``/usr/local/bin`` as
    non-root) are skipped silently.  Idempotent.
    """
    if _m().sys.platform == "win32":
        # Windows launcher staging/repair lives in _install_repair
        # (ensure_windows_bin_launchers at process start,
        # migrate_windows_bin_path in this command's tail) — not here.
        return
    for bin_dir in (Path.home() / ".local" / "bin", Path("/usr/local/bin")):
        hermes_cmd = bin_dir / "hermes"
        acp_cmd = bin_dir / "hermes-acp"
        try:
            if not (hermes_cmd.is_file() or hermes_cmd.is_symlink()):
                continue
            # Already present — a console script (pip/pipx install), an
            # earlier shim, or a symlink. is_symlink() catches broken
            # symlinks that exists() would miss; never follow-and-overwrite
            # (the #21454 failure mode).
            if acp_cmd.exists() or acp_cmd.is_symlink():
                continue
            shim = (
                "#!/usr/bin/env bash\n"
                "# Hermes Agent — ACP launcher (written by `hermes update`).\n"
                "# ACP hosts (Zed, JetBrains, Buzz) resolve the agent by this\n"
                "# command name on the login-shell PATH.\n"
                f'exec "{hermes_cmd}" acp "$@"\n'
            )
            acp_cmd.write_text(shim, encoding="utf-8")
            acp_cmd.chmod(acp_cmd.stat().st_mode | 0o755)
        except OSError:
            continue
        print(f"  ✓ Installed hermes-acp launcher → {acp_cmd}")

_PRE_UPDATE_SNAPSHOT_KEEP = 1
# Sibling-profile snapshot ids from the current run's pre-update backup
# ({profile: snapshot_id}) — consumed by the post-update per-profile
# cron-jobs safety net (#66140). Module-level because the snapshot and the
# restore run in the same process but far apart in _cmd_update_impl.
_LAST_SIBLING_SNAPSHOTS: dict = {}

# Per-file size cap for the pre-update quick snapshot. Anything larger is
# skipped with a warning: the snapshot exists to protect small, hard-to-
# regenerate state (pairing JSONs, cron jobs, config, auth) — not to copy a
# multi-GB state.db on every update (observed: a 24 GB state.db added ~60s
# of wall time and silently ate 24 GB of disk per update).
_PRE_UPDATE_SNAPSHOT_MAX_FILE_SIZE = 1 << 30  # 1 GiB

def _resolve_pre_update_backup_mode(args) -> str:
    """Resolve the pre-update backup mode: ``"off"``, ``"quick"``, or ``"full"``.

    CLI flags win over config; ``--no-backup`` beats ``--backup`` when both
    are set. Config accepts the mode strings plus legacy booleans:
    ``true`` → ``full`` (the old zip behavior), ``false`` → ``off``
    (an explicit opt-out now disables the quick snapshot too — previously
    it ran unconditionally, ignoring the user's setting). A missing key
    defaults to ``quick``.
    """
    if getattr(args, "no_backup", False):
        return "off"
    if getattr(args, "backup", False):
        return "full"

    try:
        from hermes_cli.config import load_config

        cfg = load_config()
    except Exception as exc:
        logging.getLogger(__name__).debug(
            "Could not load config for pre-update backup: %s", exc
        )
        cfg = {}

    updates_cfg = cfg.get("updates", {}) if isinstance(cfg, dict) else {}
    raw = updates_cfg.get("pre_update_backup", "quick")

    if raw is True:
        return "full"
    if raw is False:
        return "off"
    mode = str(raw).strip().lower()
    if mode in ("off", "false", "none", "disabled"):
        return "off"
    if mode in ("full", "zip", "true"):
        return "full"
    if mode == "quick":
        return "quick"
    logging.getLogger(__name__).warning(
        "Unknown updates.pre_update_backup value %r — using 'quick'", raw
    )
    return "quick"

def _run_pre_update_backup(args) -> Optional[str]:
    """Run the pre-update safety backup and return the quick-snapshot id.

    Single consolidated mechanism gated on ``updates.pre_update_backup``:

    - ``off``   — nothing runs. Explicit user opt-out is honored fully.
    - ``quick`` (default) — a state snapshot of critical small files
      (pairing JSONs, cron jobs, config, auth; see ``_QUICK_STATE_FILES``)
      under ``state-snapshots/``. Files over 1 GiB are skipped with a
      warning so a bloated state.db can never stall the update
      (issues #15733, #34600 are the reason this safety net exists).
    - ``full``  — the quick snapshot PLUS a full zip of HERMES_HOME under
      ``backups/`` (restorable via ``hermes import``; the #48200 wrong-path
      wipe is the reason this level exists).

    ``--backup`` forces ``full`` for one run; ``--no-backup`` forces ``off``.
    Never raises — a backup failure should not block the update itself.

    Returns the quick-snapshot id (used by the post-update cron-jobs
    restore safety net), or ``None`` when mode is ``off`` or the snapshot
    failed.
    """
    mode = _resolve_pre_update_backup_mode(args)

    if mode == "off":
        if getattr(args, "no_backup", False):
            print("◆ Pre-update backup: skipped (--no-backup)")
            print()
        # Config-level off is silent — the user opted out; don't spam them
        # on every update.
        return None

    snapshot_id = None
    try:
        from hermes_cli.backup import (
            _quick_snapshot_root,
            create_quick_snapshot,
            verify_sqlite_integrity,
        )

        # NOTE: this function later does `from hermes_constants import
        # get_hermes_home`, which makes the name function-local — the
        # module-level import is shadowed and unbound here. Alias explicitly.
        from hermes_cli.config import get_hermes_home as _get_home

        snapshot_id = create_quick_snapshot(
            label="pre-update",
            keep=_PRE_UPDATE_SNAPSHOT_KEEP,
            max_file_size=_PRE_UPDATE_SNAPSHOT_MAX_FILE_SIZE,
        )

        # After the snapshot, verify the source state.db is still intact.
        # The snapshot was taken via _safe_copy_db (read-only SQLite backup
        # API), but a concurrent process (antivirus, force-killed gateway
        # releasing file handles, Windows filter driver) can corrupt the live
        # file at any point. A silent zeroing at this point would proceed with
        # the update and exit code 0 — exactly the #68474 symptom.
        if snapshot_id:
            _src_path = _get_home() / "state.db"
            if _src_path.exists():
                _integrity = verify_sqlite_integrity(
                    _src_path,
                    check_header=True,
                    run_pragma=True,
                    max_bytes=_PRE_UPDATE_SNAPSHOT_MAX_FILE_SIZE,
                )
                if not _integrity.get("valid"):
                    _msg = _integrity.get("message", "unknown error")
                    print(
                        f"  ⚠ state.db integrity check FAILED after snapshot: {_msg}"
                    )
                    # Check if the snapshot itself is valid.
                    _snap_root = _quick_snapshot_root(_get_home())
                    _snap_state = _snap_root / snapshot_id / "state.db"
                    if _snap_state.exists():
                        _snap_ok = verify_sqlite_integrity(
                            _snap_state, check_header=True, run_pragma=True
                        )
                        if _snap_ok.get("valid"):
                            print(
                                "  ✓ Snapshot copy is valid — continuing update."
                            )
                            print(
                                "    If state.db is lost after update it will be auto-restored."
                            )
                        else:
                            print(
                                "  ✗ Snapshot copy ALSO failed integrity — "
                                "the source was already corrupted before the backup."
                            )
                    else:
                        print(
                            "  ⚠ Snapshot does not contain state.db (was skipped or too large)."
                        )
                    print()
        if snapshot_id:
            print(f"◆ Pre-update snapshot: {snapshot_id}")

        # #66140: the code swap + fleet restart touch EVERY profile, so
        # every profile gets the same snapshot (same set, same 1GiB cap,
        # keep=1) under its own state-snapshots/. Best-effort per profile.
        try:
            from hermes_cli.backup import create_pre_update_snapshots_all_profiles

            _sibling_snaps = create_pre_update_snapshots_all_profiles(
                keep=_PRE_UPDATE_SNAPSHOT_KEEP,
                max_file_size=_PRE_UPDATE_SNAPSHOT_MAX_FILE_SIZE,
            )
            if _sibling_snaps:
                print(
                    f"◆ Sibling profile snapshot(s): "
                    + ", ".join(sorted(_sibling_snaps))
                )
                try:
                    from hermes_cli.update_receipt import record_step

                    record_step(
                        "sibling_profile_snapshots",
                        True,
                        ", ".join(
                            f"{k}={v}" for k, v in sorted(_sibling_snaps.items())
                        ),
                    )
                except Exception:
                    pass
                global _LAST_SIBLING_SNAPSHOTS
                _LAST_SIBLING_SNAPSHOTS = _sibling_snaps
        except Exception as _sib_exc:
            logging.getLogger(__name__).debug(
                "Sibling profile snapshots failed: %s", _sib_exc
            )
    except Exception as exc:
        # Never let a snapshot failure block an update.
        logging.getLogger(__name__).debug("Pre-update snapshot failed: %s", exc)

    if mode != "full":
        if snapshot_id:
            print()
        return snapshot_id

    try:
        from hermes_cli.backup import create_pre_update_backup
    except Exception as exc:
        print(
            f"⚠ Pre-update backup: could not load backup module ({exc}); continuing update."
        )
        print()
        return snapshot_id

    try:
        from hermes_cli.config import load_config

        _keep = (load_config() or {}).get("updates", {}).get("backup_keep", 5)
    except Exception:
        _keep = 5

    print("◆ Creating pre-update backup...")
    t0 = _time.monotonic()
    try:
        out_path = create_pre_update_backup(keep=int(_keep))
    except Exception as exc:  # defensive — helper already swallows, but just in case
        print(f"  ⚠ Backup failed: {exc}")
        print("  Continuing with update.")
        print()
        return snapshot_id

    elapsed = _time.monotonic() - t0

    if out_path is None:
        print("  ⚠ Backup skipped (no files found or write failed); continuing update.")
        print()
        return snapshot_id

    try:
        size_bytes = out_path.stat().st_size
    except OSError:
        size_bytes = 0

    # Human-readable size
    from hermes_cli.sizefmt import format_bytes

    size_str = format_bytes(size_bytes)

    # Render path using display_hermes_home so the user sees ~/.hermes/...
    try:
        from hermes_constants import get_hermes_home, display_hermes_home

        home = get_hermes_home()
        try:
            display_path = f"{display_hermes_home()}/{out_path.relative_to(home)}"
        except ValueError:
            display_path = str(out_path)
    except Exception:
        display_path = str(out_path)

    print(f"  Saved:    {display_path} ({size_str}, {elapsed:.1f}s)")
    print(f"  Restore:  hermes import {out_path}")
    print("  Disable:  set updates.pre_update_backup: quick (or off) in config.yaml")
    print()
    return snapshot_id

def _write_update_planned_stop_marker(profile_path: Path, pid: int) -> bool:
    """Write a planned-stop marker into a specific profile home."""
    try:
        from datetime import timezone

        from gateway.status import _get_process_start_time
        from utils import atomic_json_write

        record = {
            "target_pid": pid,
            "target_start_time": _get_process_start_time(pid),
            "stopper_pid": os.getpid(),
            "written_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json_write(
            Path(profile_path) / ".gateway-planned-stop.json",
            record,
            indent=None,
            separators=(",", ":"),
        )
        return True
    except (OSError, PermissionError):
        return False

def _wait_for_windows_update_gateway_exit(
    pids: list[int], *, timeout: float
) -> set[int]:
    """Wait for the given gateway PIDs to exit, returning survivors."""
    if not pids:
        return set()

    from gateway.status import _pid_exists

    remaining = set(pids)
    deadline = _time.monotonic() + max(timeout, 0.0)
    while remaining and _time.monotonic() < deadline:
        for pid in list(remaining):
            try:
                if not _pid_exists(pid):
                    remaining.discard(pid)
            except Exception:
                remaining.discard(pid)
        if remaining:
            _time.sleep(0.25)

    survivors: set[int] = set()
    for pid in remaining:
        try:
            if _pid_exists(pid):
                survivors.add(pid)
        except Exception:
            pass
    return survivors

def _venv_core_imports_healthy() -> tuple[bool, str]:
    """Probe the project venv for the core imports the backend needs to boot.

    Runs a tiny import check inside the venv interpreter (NOT this process —
    ``hermes update`` may be driven by a different Python). Catches the
    half-updated-venv state: git checkout current but a dependency sync that
    failed or was killed partway (e.g. Windows access-denied on a loaded
    .pyd), leaving imports like ``fastapi``'s new transitive deps missing.
    Without this probe, ``hermes update`` on a current checkout prints
    "Already up to date!" and returns without ever re-syncing dependencies —
    the user's install stays broken no matter how many times they update
    (ryanc's incident, July 2026).

    Returns ``(healthy, detail)``. Never raises; unknown states report
    healthy so a probe failure can't force needless reinstalls.
    """
    venv_dir = _m().PROJECT_ROOT / "venv"
    venv_python = venv_python_path(venv_dir, windows=_m()._is_windows())
    if not venv_python.exists():
        # No venv interpreter at all. In a dev checkout that's normal (the
        # dev may run hermes from any interpreter), so report healthy to
        # avoid forcing reinstalls. But on a MANAGED install (the Windows
        # installer / desktop bootstrap stamps `.hermes-bootstrap-complete`,
        # and an interrupted update leaves `.update-incomplete`), the venv
        # IS the install — its absence means a repair got interrupted after
        # the old venv was moved aside, and "Already up to date!" would
        # gaslight the user while nothing can run.
        managed_markers = (
            _m().PROJECT_ROOT / ".hermes-bootstrap-complete",
            _m()._update_marker_path(),
        )
        if any(m.exists() for m in managed_markers):
            return False, f"venv python missing ({venv_python})"
        return True, ""

    # Core web/serve imports plus their newest transitive deps. Import (not
    # just metadata) — a package can have intact dist-info but a missing
    # module after an interrupted uninstall/install cycle.
    check = (
        "import importlib\n"
        "mods = ['fastapi', 'uvicorn', 'pydantic', 'openai', 'yaml']\n"
        "missing = []\n"
        "for m in mods:\n"
        "    try: importlib.import_module(m)\n"
        "    except Exception as e: missing.append(f'{m}: {e}')\n"
        "print('\\n'.join(missing))\n"
    )
    try:
        result = subprocess.run(
            [str(venv_python), "-c", check],
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=60,
            cwd=_m().PROJECT_ROOT,
        )
    except Exception as exc:
        logger.debug("venv health probe failed to run: %s", exc)
        return True, ""

    missing = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    if result.returncode != 0 and not missing:
        # Interpreter itself is broken (e.g. deleted stdlib) — that IS unhealthy.
        detail = (result.stderr or "").strip().splitlines()
        return False, detail[0] if detail else "venv python failed to run"
    if missing:
        return False, "; ".join(missing[:4])
    return True, ""

def _detect_venv_python_processes(
    *, exclude_pids: set[int] | None = None
) -> list[tuple[int, str, str]]:
    """Find live processes running from the project venv's interpreter.

    The hermes.exe shim guard misses the biggest lock-holder class on
    Windows: the Desktop app's backend (``python.exe -m hermes_cli.main
    serve``) and anything else running straight off ``venv\\Scripts\\python
    (w).exe``. Those processes keep native ``.pyd`` extensions mapped, so a
    dependency sync mid-update dies with access-denied and strands the venv
    half-updated (ryanc's brotlicffi/_sodium.pyd incidents, July 2026).

    Killing them from here is pointless — the Desktop app supervises its
    backend and respawns it within seconds — so the caller should refuse and
    tell the user to close the app instead. Returns ``(pid, name, cmdline)``
    tuples; empty off-Windows / without psutil / when nothing matches. The
    calling process and its ancestors are always excluded (a CLI ``hermes
    update`` itself runs from the venv python). Never raises.
    """
    if not _m()._is_windows():
        return []
    try:
        import psutil
    except Exception:
        return []

    venv_dir = _m().PROJECT_ROOT / "venv"
    try:
        venv_prefix = str(venv_dir.resolve()).lower().rstrip(os.sep) + os.sep
    except OSError:
        venv_prefix = str(venv_dir).lower().rstrip(os.sep) + os.sep
    try:
        root_prefix = str(_m().PROJECT_ROOT.resolve()).lower().rstrip(os.sep) + os.sep
    except OSError:
        root_prefix = str(_m().PROJECT_ROOT).lower().rstrip(os.sep) + os.sep

    skip: set[int] = set(exclude_pids or set())
    skip.add(os.getpid())
    try:
        from gateway.status import looks_like_gateway_command_line as _is_gw
    except Exception:
        _is_gw = None
    try:
        for anc in psutil.Process().parents():
            # #87594: do NOT blanket-exclude ancestors. When `/update` runs
            # from a messaging platform the updater is a CHILD of the gateway
            # — excluding all ancestors hides the gateway from the scan, so
            # the pause machinery downstream never sees the one process it
            # exists to stop, and the update dead-ends on `venv-blocked`.
            # A GATEWAY ancestor stays visible (the pause path stops it
            # gracefully; a detached child updater survives its parent's
            # stop on Windows). Every other ancestor (shells, terminals,
            # this CLI's own venv python chain) stays excluded — an updater
            # must never nominate its own interactive ancestry as blockers.
            try:
                anc_cmdline = " ".join(anc.cmdline() or [])
            except Exception:
                anc_cmdline = ""
            if _is_gw is not None and anc_cmdline and _is_gw(anc_cmdline):
                continue
            skip.add(int(anc.pid))
    except Exception:
        pass

    matches: list[tuple[int, str, str]] = []
    try:
        proc_iter = psutil.process_iter(["pid", "exe", "name", "cmdline", "cwd"])
    except Exception:
        return []
    for proc in proc_iter:
        try:
            info = proc.info
        except Exception:
            continue
        pid = info.get("pid")
        exe = info.get("exe")
        if not exe or pid is None or int(pid) in skip:
            continue
        try:
            exe_norm = str(Path(exe).resolve()).lower()
        except (OSError, ValueError):
            exe_norm = str(exe).lower()
        cmdline_raw = " ".join(info.get("cmdline") or [])
        cmdline_low = cmdline_raw.lower()
        cwd_low = str(info.get("cwd") or "").lower().rstrip(os.sep) + os.sep

        # Primary match: the executable itself lives under this venv
        # (venv\Scripts\python(w).exe — the desktop backend / gateway case).
        is_holder = exe_norm.startswith(venv_prefix)
        # Fallback: uv/base-interpreter trampolines run a python whose exe is
        # OUTSIDE the venv but which still imports from it and holds its .pyd
        # files. Catch those by what they're running: a cmdline that references
        # this venv's path, or a `-m hermes_cli.main ...` invocation tied to
        # this install (install root in the cmdline or as the working dir).
        if not is_holder and venv_prefix in cmdline_low:
            is_holder = True
        if not is_holder and "hermes_cli.main" in cmdline_low:
            if root_prefix in cmdline_low or cwd_low.startswith(root_prefix):
                is_holder = True
        if not is_holder:
            continue
        name = info.get("name") or Path(exe).name
        # Return the FULL cmdline: callers match against it (the Desktop
        # preflight's pausable-gateway exemption parses for `gateway run`).
        # Truncating here cut long managed-runtime interpreter paths before
        # the `-m hermes_cli.main gateway run` argv, so autostarted gateways
        # were misreported as blockers and the update dead-ended. Truncate
        # only at display time.
        matches.append((int(pid), str(name), cmdline_raw))
    return matches

# Native-extension modules that pin files inside the venv once imported.  If
# the updater process itself has any of these loaded, the dependency sync
# below cannot rewrite the backing ``.pyd``/``.dll`` — Windows blocks REPLACE
# on a mapped image — and the update dies with ``os error 5`` between
# uninstall and reinstall, stranding the venv half-updated (#83569).
# ``cryptography`` is the canonical case: ``hermes_cli.main`` used to import
# it at startup while resolving external secret sources; ``PyYAML``'s
# ``_yaml`` C extension is loaded by every CLI process (config parsing).
# Keep this guard as defence-in-depth against future eager imports (new
# secret sources, plugins absorbed into core, refactors of the startup
# order) — but the guard must be HONEST (#86735/#86780/#86781: a preflight
# that fired on every run, before the fetch, re-bricked the exact flow it
# was meant to protect).  Two honesty gates:
#
# 1. It only fires when the dependency sync would actually REWRITE the
#    loaded distribution (``_dependency_sync_would_rewrite``): if the
#    installed version already satisfies the on-disk pyproject pins, uv/pip
#    will not touch the mapped ``.pyd``, so there is no lock to trip.
# 2. It runs AFTER the code swap (git pull / ZIP commit), immediately
#    before the venv rewrite — so the on-disk pyproject is the NEW one
#    (gate 1 compares against the right target) and a deferral no longer
#    strands the user on the old checkout: the next launch's marker
#    recovery completes the dependency install against the already-updated
#    pyproject.
#
# Keys are module prefixes in ``sys.modules``; values are
# ``(display name, PyPI distribution name)``.
_SELF_LOCKING_NATIVE_MODULES: dict[str, tuple[str, str]] = {
    "cryptography.hazmat.bindings._rust": ("cryptography (_rust.pyd)", "cryptography"),
    "yaml._yaml": ("PyYAML (_yaml.pyd)", "pyyaml"),
}


def _dependency_sync_would_rewrite(dist_name: str) -> bool | None:
    """Whether ``uv pip install -e .[all]`` would replace *dist_name*'s files.

    Compares the installed distribution version against every applicable
    requirement for it in the on-disk ``pyproject.toml`` (base dependencies
    plus all optional extras).  Returns:

    - ``False`` — installed version satisfies every pin: the resolver will
      leave the wheel alone, so a mapped extension is NOT at risk.
    - ``True``  — some pin is not satisfied (or the distribution is
      missing): the sync will rewrite it.
    - ``None``  — could not determine (parse failure, unparseable pins).

    Never raises.  Callers treat ``None`` as fail-OPEN (no deferral): a
    module in the registry can be loaded by every process (PyYAML), so
    deferring on uncertainty would recreate the #86735 always-firing loop.
    """
    try:
        from importlib import metadata as _ilmd

        installed = _ilmd.version(dist_name)
    except Exception:
        return True  # not installed → the sync will definitely install it
    try:
        import tomllib

        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name
        from packaging.version import Version

        pyproject = _m().PROJECT_ROOT / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        project = data.get("project") or {}
        req_strings: list[str] = list(project.get("dependencies") or [])
        for extra_reqs in (project.get("optional-dependencies") or {}).values():
            req_strings.extend(extra_reqs or [])

        target = canonicalize_name(dist_name)
        installed_v = Version(installed)
        saw_pin = False
        for req_str in req_strings:
            try:
                req = Requirement(req_str)
            except Exception:
                continue
            if canonicalize_name(req.name) != target:
                continue
            if req.marker is not None and not req.marker.evaluate():
                continue
            saw_pin = True
            if installed_v not in req.specifier:
                return True
        if saw_pin:
            return False
        # Not pinned anywhere in pyproject: the resolver may still move it
        # as a transitive — we cannot cheaply predict that, so stay honest
        # about the uncertainty.
        return None
    except Exception:
        return None


def _detect_self_loaded_native_modules() -> list[str]:
    """Native venv extensions loaded into THIS process that the sync would rewrite.

    Returns display names (empty off Windows — POSIX lets a running process
    keep using an unlinked inode, so self-locking is a Windows-only hazard).
    A loaded module whose installed version already satisfies the on-disk
    pyproject pins is NOT reported: the dependency sync will not touch its
    files, so there is no swap at risk (#86735 — the always-firing variant
    of this preflight bricked every Windows update).  Never raises.
    """
    if not _m()._is_windows():
        return []
    found = []
    for prefix, (display, dist) in _SELF_LOCKING_NATIVE_MODULES.items():
        if prefix not in sys.modules:
            continue
        # Defer ONLY on a CONFIRMED pending rewrite. An "unknown" result
        # (unreadable/unparseable pyproject, no pin found) must fail OPEN:
        # PyYAML is loaded in every CLI process, so treating unknown as
        # at-risk would re-create the exact always-firing loop this guard's
        # first version caused (#86735). The downside of a missed deferral
        # is the pre-existing failure mode — a mid-sync os error 5 that the
        # marker recovery already handles — which is strictly less harmful
        # than an update that can never run.
        if _m()._dependency_sync_would_rewrite(dist) is not True:
            continue
        found.append(display)
    return sorted(set(found))


def _abort_dependency_sync_if_self_locked(gateway_resume=None) -> None:
    """Defer the venv rewrite when THIS process holds something it must replace.

    Runs at the last moment before the venv rewrite — after the code swap —
    so the on-disk pyproject reflects the update target and a deferral
    leaves the user on NEW code with only the dependency install pending.
    No-op when nothing at-risk is held.

    Two hazards, both "this process holds a file the sync must replace", and
    they end differently because their recoveries differ:

    - A mapped native extension (``.pyd``).  Exit 2 and let the next launch's
      marker recovery finish the install: that launch runs the install before
      importing anything heavy, so it maps nothing and the swap succeeds.

    - The ``hermes.exe`` console shim we were launched from (#88838, #89599).
      The marker cannot help here — every future ``hermes`` launch is also the
      shim, so deferring to the next launch defers forever.  Hand the install
      to a child under the venv interpreter and exit, releasing the shim.
    """
    locked = _m()._detect_self_loaded_native_modules()
    if locked:
        _m()._defer_update_for_self_lock(locked)
        if gateway_resume is not None:
            _m()._resume_windows_gateways_after_update(gateway_resume)
        sys.exit(2)

    if _m()._reexec_dependency_sync_off_windows_shim():
        if gateway_resume is not None:
            _m()._resume_windows_gateways_after_update(gateway_resume)
        sys.exit(0)


def _defer_update_for_self_lock(loaded: list[str]) -> None:
    """Bail out before the dependency sync when the updater holds a lock.

    The install cannot win this race from inside the locked process — even
    killing threads would not unmap the image — so defer it: drop the
    update-incomplete marker (next launch's fresh process completes the
    install before importing anything heavy), explain, and exit 2 like the
    other preflight refusals.
    """
    print("✗ This updater process has already loaded native venv modules that")
    print("  the dependency sync must replace:")
    for name in loaded:
        print(f"    {name}")
    print()
    print("  On Windows a mapped extension cannot be replaced by the process")
    print("  holding it. The code update has been applied; only the dependency")
    print("  sync has been deferred: the next `hermes` launch will complete it")
    print("  in a fresh process before anything imports these modules.")
    _m()._write_update_incomplete_marker()


_HOLDER_VALUE_FLAGS_FALLBACK = frozenset(
    {
        "--profile", "-p", "--config",
        "--model", "-m", "--provider", "--reasoning",
        "--toolsets", "-t", "--skills", "-s",
        "--continue", "-c", "--resume", "-r",
        "--oneshot", "-z", "--in", "--usage-file",
    }
)
_holder_value_flags_cache: frozenset | None = None


def _holder_value_flags() -> frozenset:
    """Top-level CLI flags that consume a value — derived from the REAL parser.

    Introspects ``build_top_level_parser()`` (every option with nargs != 0)
    so the holder classifier can never drift from the argparse surface
    (#91869 review: a handwritten subset misparsed ``--reasoning high
    serve`` as subcommand ``high`` and ``-m dashboard serve`` as
    ``dashboard`` — recreating the wrong-hint class). The pre-argparse
    profile selectors (``--profile``/``-p``, ``--config``) are added
    explicitly since they are stripped before argparse sees argv. Falls
    back to a static snapshot when the parser cannot be imported (the
    updater must classify holders even mid-upgrade on a broken tree).
    Cached per process.
    """
    global _holder_value_flags_cache
    if _holder_value_flags_cache is not None:
        return _holder_value_flags_cache
    flags: set[str] = {"--profile", "-p", "--config"}
    try:
        from hermes_cli._parser import build_top_level_parser

        parser = build_top_level_parser()[0]
        for action in parser._actions:
            if action.option_strings and action.nargs != 0:
                flags.update(action.option_strings)
        _holder_value_flags_cache = frozenset(flags)
    except Exception:
        _holder_value_flags_cache = _HOLDER_VALUE_FLAGS_FALLBACK
    return _holder_value_flags_cache


def _hermes_holder_subcommand(cmdline: str) -> str | None:
    """The actual Hermes SUBCOMMAND a venv-holder argv runs, or None.

    Token-based, never substring (#90778: ``kanban --preserve-cache``
    contained \"serve\" and got labeled as the Desktop backend). Finds the
    ``hermes_cli.main`` / ``hermes(.exe)`` entry token, then returns the
    first following token that is not a flag or a flag's value. Profile
    selectors (``--profile X``, ``-p X``) are skipped like the canonical
    gateway matcher does. Returns None when no subcommand can be
    determined — callers must NOT guess a label in that case.
    """
    try:
        import shlex

        tokens = shlex.split(cmdline, posix=False)
    except Exception:
        tokens = cmdline.split()

    entry_idx: int | None = None
    for i, token in enumerate(tokens):
        low = token.lower().strip('"')
        if low.endswith("hermes_cli.main") and i > 0 and tokens[i - 1] == "-m":
            entry_idx = i
            break
        base = low.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
        if base in ("hermes", "hermes.exe"):
            entry_idx = i
            break
    if entry_idx is None:
        return None

    value_flags = _holder_value_flags()
    i = entry_idx + 1
    while i < len(tokens):
        token = tokens[i]
        if token in value_flags or token.split("=", 1)[0] in value_flags:
            # --flag value consumes two tokens; --flag=value consumes one.
            i += 1 if "=" in token else 2
            continue
        if token.startswith("-"):
            i += 1
            continue
        return token.lower()
    return None


def _format_venv_python_holders_message(matches: list[tuple[int, str, str]]) -> str:
    """Explain which venv processes block the update and how to clear them.

    Holder labels come from the parsed SUBCOMMAND, never substring matching
    (#90778): a standalone ``hermes dashboard`` must not be labeled as the
    Desktop backend (advice to close an app that isn't running), and flags
    like ``--preserve-cache`` must not match \"serve\". Unknown argv gets no
    hint rather than a wrong one.
    """
    lines = [
        "✗ Other Hermes processes are running from this install's venv:",
    ]
    hint_by_subcommand = {
        "serve": "  ← Hermes backend (if the Desktop app is open, close it)",
        "dashboard": "  ← hermes dashboard (stop it: hermes dashboard stop, or close that terminal)",
        "gateway": "  ← gateway",
    }
    for pid, name, cmdline in matches[:6]:
        sub = _hermes_holder_subcommand(cmdline)
        hint = hint_by_subcommand.get(sub or "", "")
        lines.append(f"  PID {pid}  {name}  {cmdline[:120]}{hint}")
    if len(matches) > 6:
        lines.append(f"  ... and {len(matches) - 6} more")
    lines.append("")
    lines.append(
        "  On Windows these keep native extension files (.pyd) locked, so the"
    )
    lines.append(
        "  dependency update would fail partway and leave a broken install."
    )
    lines.append(
        "  Close the Hermes desktop app / other Hermes terminals, then re-run:"
    )
    lines.append("    hermes update")
    lines.append("  (or use `hermes update --force-venv` to proceed anyway at your own risk)")
    return "\n".join(lines)

def _venv_launcher_ancestors(pids: list[int]) -> list[int]:
    """Return venv-interpreter ancestors of *pids* that hold the install open.

    On Windows a gateway started through the venv shim is a **two-process
    chain**: ``venv\\Scripts\\python.exe`` (the launcher, which keeps native
    ``.pyd`` files from the venv mapped) spawns the actual interpreter from
    uv's managed CPython directory (``AppData\\Roaming\\uv\\python\\...``).
    The gateway writes its PID file from the *child*, so
    ``find_gateway_pids()`` — and therefore this module's pause set — only
    ever sees the uv-side worker.

    ``_detect_venv_python_processes()`` matches on the venv path prefix, so
    the guard downstream of the pause sees the *launcher* instead. The two
    sets are disjoint, which meant a paused gateway still tripped the
    venv-holder guard and aborted the update every time (the Desktop
    "venv-blocked: N process(es) hold the install" dead-end, where the
    reported holder is a gateway the updater believes it already stopped).

    Walking one hop up from each mapped gateway PID and keeping ancestors
    that live under the project venv closes the gap. Only the venv-side
    parent is returned — unrelated ancestors (the Scheduled Task's
    ``cmd.exe``, an operator's shell) are ignored so we never widen the
    blast radius beyond the gateway's own launcher. Never raises.
    """
    if not _m()._is_windows() or not pids:
        return []
    try:
        import psutil
    except Exception:
        return []

    venv_dir = _m().PROJECT_ROOT / "venv"
    try:
        venv_prefix = str(venv_dir.resolve()).lower().rstrip(os.sep) + os.sep
    except OSError:
        venv_prefix = str(venv_dir).lower().rstrip(os.sep) + os.sep

    # Never return ourselves or our own ancestry: a CLI ``hermes update``
    # runs from the venv python and would otherwise nominate itself.
    # Same #87594 carve-out as _detect_venv_python_processes: a GATEWAY
    # ancestor is not "our own ancestry" in the interactive sense — it is
    # the process the pause machinery must see (the /update-from-gateway
    # topology makes the updater the gateway's child).
    try:
        from gateway.status import looks_like_gateway_command_line as _is_gw
    except Exception:
        _is_gw = None
    skip: set[int] = {os.getpid()}
    try:
        for anc in psutil.Process().parents():
            try:
                anc_cmdline = " ".join(anc.cmdline() or [])
            except Exception:
                anc_cmdline = ""
            if _is_gw is not None and anc_cmdline and _is_gw(anc_cmdline):
                continue
            skip.add(int(anc.pid))
    except Exception:
        pass

    found: list[int] = []
    for pid in pids:
        try:
            parent = psutil.Process(int(pid)).parent()
        except Exception:
            continue
        if parent is None:
            continue
        ppid = int(parent.pid)
        if ppid in skip or ppid in found or ppid in set(pids):
            continue
        try:
            exe = (parent.exe() or "").lower()
        except Exception:
            continue
        if exe.startswith(venv_prefix):
            found.append(ppid)
    return found


def _leftover_pausable_gateway_pids(
    matches: list[tuple[int, str, str]],
) -> list[int] | None:
    """PIDs from *matches* when every remaining venv holder is a pausable gateway.

    ``_pause_windows_gateways_for_update()`` stops every gateway its discovery
    finds, but the venv-holder guard downstream sees the process table as it
    is *now*: a gateway respawned by its supervisor (Scheduled Task, login
    watchdog) inside the pause→guard window, or one started through a spawn
    path the discovery does not map, still holds venv ``.pyd`` files and
    would dead-end the update — an abort pointed at exactly the kind of
    process the pause machinery exists to stop.

    Holders are classified with the same matcher the Desktop preflight uses
    to exempt them (``_is_pausable_gateway``), so the preflight's exemption
    and this guard's tolerance cannot drift apart — matcher drift between
    two views of the same process table is what produced the launcher/worker
    dead-end fixed above. The scan captures only a 120-char cmdline prefix,
    so the live argv is re-read where psutil allows; an unreadable argv
    falls back to the captured prefix.

    Returns ``None`` when any holder is not a pausable gateway — an operator
    REPL, a stray script, or the Desktop backend has no pause machinery
    downstream, and the guard must keep refusing exactly as before.
    """
    from hermes_cli._scan_venv_blockers import _is_pausable_gateway

    try:
        import psutil  # type: ignore
    except Exception:
        psutil = None

    pids: list[int] = []
    for pid, _name, cmdline in matches:
        argv = cmdline
        if psutil is not None:
            try:
                argv = " ".join(psutil.Process(int(pid)).cmdline()) or cmdline
            except Exception:
                pass
        if not _is_pausable_gateway(argv):
            return None
        pids.append(int(pid))
    return pids


def _orphaned_desktop_backend_pids(
    matches: list[tuple[int, str, str]],
) -> list[int] | None:
    """PIDs from *matches* when every remaining holder is an ORPHANED backend.

    The venv-holder guard refuses on the Desktop app's ``serve`` backend by
    design: while the Desktop is open, killing its backend is futile (the app
    supervises and respawns it within seconds), so the user must close the
    app. But in the GUI-updater handoff path the Desktop has *already
    exited* — by contract it tree-kills its backends and waits for the venv
    shim before spawning hermes-setup, and the update-in-progress marker
    parks any relaunched Desktop from spawning a fresh backend (#50238). A
    ``serve`` backend still holding the venv at that point is a straggler
    whose supervisor is gone: SIGTERM raced its spawn, or it belongs to a
    crashed window. Nothing will respawn it, and refusing on it dead-ends
    the update with "Hermes is still running" while the user stares at zero
    open windows (ryanc's 2026-08-09 01:59/02:17 failures).

    A holder qualifies only when BOTH hold:

    - its cmdline is a Hermes backend (``hermes_cli.main`` + ``serve`` /
      ``dashboard``), and
    - its supervising parent is demonstrably gone: the parent PID no longer
      exists, or the PID was reused (parent created *after* the child).

    Tree-aware: the scanner can return an orphaned backend AND one of its
    managed-runtime descendants (the ``.hermes-runtime`` interpreter child)
    in the same holder set. That descendant has a live parent — the orphaned
    backend itself — and isn't a ``serve`` cmdline, so per-process rules
    would refuse a set that is entirely safe to reap. Holders that sit
    inside an accepted orphan root's tree are therefore folded into that
    root (only roots are returned; ``taskkill /T`` reaps the descendants).

    Any other live-parent backend (the Desktop is still open), non-backend
    holder outside an orphan tree, or unprovable case disqualifies the whole
    set — the guard must keep refusing exactly as before. Returns ``None``
    in that case, or when psutil is unavailable (can't prove orphanhood →
    refuse). Never raises.
    """
    try:
        import psutil  # type: ignore
    except Exception:
        return None

    def _is_backend(argv_low: str) -> bool:
        return "hermes_cli.main" in argv_low and (
            " serve" in argv_low or " dashboard" in argv_low
        )

    # Pass 1: find orphaned backend ROOTS among the holders.
    roots: list[int] = []
    remaining: list[tuple[int, str]] = []  # (pid, argv_low) still to justify
    for pid, _name, cmdline in matches:
        argv = cmdline
        try:
            argv = " ".join(psutil.Process(int(pid)).cmdline()) or cmdline
        except psutil.NoSuchProcess:
            # Holder exited between scan and classification — nothing to
            # reap, nothing blocking. Skip it.
            continue
        except Exception:
            pass
        low = argv.lower()
        if not _is_backend(low):
            remaining.append((int(pid), low))
            continue
        try:
            proc = psutil.Process(int(pid))
            ppid = proc.ppid()
            parent = psutil.Process(ppid) if ppid else None
            if parent is not None and parent.is_running():
                # PID-reuse check: a "parent" created after its child is a
                # recycled PID, not the real (dead) supervisor.
                if parent.create_time() <= proc.create_time():
                    # Live parent — NOT a root. But it may still be a
                    # descendant of an orphan root: the venv python.exe is
                    # a trampoline that re-execs the uv-managed interpreter
                    # with the SAME backend argv, so the worker half of the
                    # two-process chain lands here. Defer to pass 2 instead
                    # of refusing outright.
                    remaining.append((int(pid), low))
                    continue
        except psutil.NoSuchProcess:
            pass  # parent gone → orphan
        except Exception:
            return None
        roots.append(int(pid))

    # Pass 2: every non-backend holder must be a descendant of an accepted
    # orphan root — then it dies with the root's tree reap. Anything else
    # (operator REPL, stray script) keeps the refusal.
    root_set = set(roots)
    for pid, _low in remaining:
        if not root_set:
            return None
        try:
            ancestors = {int(a.pid) for a in psutil.Process(pid).parents()}
        except psutil.NoSuchProcess:
            continue  # exited already
        except Exception:
            return None
        if not (ancestors & root_set):
            return None
    return roots


def _ledger_reapable_backend_pids(
    matches: list[tuple[int, str, str]],
) -> list[int]:
    """PIDs positively identified by the spawn ledger as orphaned backends.

    The strongest rung: instead of inferring lineage from PPIDs or cmdline
    shape, look each venv holder up in the machine spawn ledger
    (``hermes_cli.process_identity``). A holder qualifies when ALL of:

    - its ``(pid, create_time)`` matches a live ledger entry (PID reuse
      cannot forge this pair);
    - the entry's purpose is a reapable backend kind (serve/dashboard/
      gateway — never interactive processes);
    - the entry's recorded SPAWNER is provably dead (``spawner_is_dead``).

    Unlike the heuristic rungs, this is safe in ANY update context — no
    hand-off contract needed — because the ownership claim is explicit: the
    process itself declared who supervises it, and that supervisor is gone.
    Holders not in the ledger are simply not returned (they fall through to
    the later rungs); they never disqualify the identified ones. Never raises.
    """
    try:
        from hermes_cli.process_identity import (
            REAPABLE_PURPOSES,
            ledger_entries,
            spawner_is_dead,
        )

        entries = ledger_entries()
    except Exception:
        return []
    by_pid = {e.get("pid"): e for e in entries if isinstance(e.get("pid"), int)}
    roots: list[int] = []
    for pid, _name, _cmdline in matches:
        entry = by_pid.get(int(pid))
        if not entry:
            continue
        if entry.get("purpose") not in REAPABLE_PURPOSES:
            continue
        if spawner_is_dead(entry) is True:
            roots.append(int(pid))
    return roots


def _handoff_reapable_backend_pids(
    matches: list[tuple[int, str, str]],
) -> list[int] | None:
    """PIDs of Hermes ``serve``/``dashboard`` backends safe to reap during a
    GUI-updater hand-off, INCLUDING ones with a still-live parent.

    Complements ``_orphaned_desktop_backend_pids``, which only reaps backends
    whose supervisor is provably dead. That check returns ``None`` (keep
    refusing) the moment ANY holder still has a live parent — which is exactly
    the case that produced the field incident this fixes: a Windows Desktop
    update hand-off (``update --yes --gateway --force``) left a *swarm* of
    per-profile ``serve`` backends (mr-tester, probe-inherit, turqoise, …)
    holding ``cryptography\\_rust.pyd``. Several still had a lingering
    parent (the tearing-down Electron process, or the two-hop venv
    launcher→worker chain mid-exit), so the orphan check disqualified the
    WHOLE set and the update dead-ended — the user saw a 12-minute hang, then
    force-closed, and the half-done state stranded bot sessions.

    The hand-off is the safe signal: when the update-incomplete marker is
    present (the GUI updater claimed it) AND this is a ``--gateway`` hand-off
    run AND no live Desktop shim (``hermes.exe``) is open, NOTHING legitimate
    is supervising or respawning a ``serve`` backend from this venv — by the
    hand-off contract the Desktop tree-kills its backends and parks any
    relaunch behind the marker (#50238). Any ``serve`` backend still holding
    the venv here is therefore a leak, live parent or not, and reaping its
    tree is correct rather than a race.

    Guarded conservatively:

    - Only Hermes backends (``hermes_cli.main`` + ``serve``/``dashboard``)
      from THIS install's venv qualify; a non-backend holder (operator REPL,
      stray script) disqualifies the whole set → ``None`` (keep refusing), so
      we never widen the blast radius during a hand-off.
    - Only runs when the CALLER has confirmed the hand-off context
      (``args.gateway`` AND a claimed update-incomplete marker AND no live
      ``hermes.exe`` shim) — outside that gate this function is never called
      and the stricter orphan-only path stands.
    - psutil unavailable → ``None`` (can't re-read argv to classify → refuse).

    Returns the backend root PIDs to tree-reap, or ``None`` to leave the
    decision to the caller's existing rungs. Never raises.
    """
    try:
        import psutil  # type: ignore
    except Exception:
        return None

    def _is_backend(argv_low: str) -> bool:
        return "hermes_cli.main" in argv_low and (
            " serve" in argv_low or " dashboard" in argv_low
        )

    roots: list[int] = []
    for pid, _name, cmdline in matches:
        argv = cmdline
        try:
            argv = " ".join(psutil.Process(int(pid)).cmdline()) or cmdline
        except psutil.NoSuchProcess:
            # Exited between scan and classification — nothing to reap.
            continue
        except Exception:
            pass
        if not _is_backend(argv.lower()):
            # A non-backend holder during a hand-off is unexpected; refuse the
            # whole set rather than reap something we cannot justify.
            return None
        roots.append(int(pid))

    return roots or None


def _stop_process_trees(pids: list[int]) -> None:
    """Force-stop each PID with its full child tree (Windows).

    ``taskkill /T /F`` mirrors the Desktop's ``forceKillProcessTree`` and
    install.ps1's venv sweep: stopping only the parent can leave a managed
    ``.hermes-runtime`` interpreter child alive and holding the install open
    (#70026). Best effort; never raises.
    """
    for pid in pids:
        try:
            subprocess.run(
                ["taskkill", "/PID", str(int(pid)), "/T", "/F"],
                check=False,
                capture_output=True,
            )
        except Exception as exc:
            logger.debug("Could not stop process tree %s: %s", pid, exc)


def _looks_like_desktop_control_plane(cmdline: str) -> bool:
    """True for this-install ``hermes serve`` / ``hermes dashboard`` argv.

    That is the Desktop control plane, not the messaging gateway. Serve and
    dashboard do not host platform adapters (#92091); do not feed this into
    ``looks_like_gateway_command_line``.

    Token-based via the parser-derived subcommand classifier — never
    substring (#90778/#91869: ``kanban --preserve-cache`` contains "serve",
    ``-m dashboard chat`` contains " dashboard"; both are NOT control
    planes). A cmdline whose subcommand cannot be determined is NOT a
    control plane — callers must not guess ownership.
    """
    if "hermes_cli.main" not in (cmdline or "").lower():
        return False
    return _hermes_holder_subcommand(cmdline) in ("serve", "dashboard")


def _desktop_owns_gateway_lifecycle() -> bool:
    """True when Desktop currently supervises this install's control plane.

    The updater must not steal gateway start in that case: Desktop owns
    start/stop via ``/api/gateway/*``. This is *not* proof messaging is
    already served — a live serve process is the control plane, and the
    gateway is a detached sibling (#76129 / #92091).

    Prefer the spawn ledger (owned identity). Fall back to the install-scoped
    venv-holder scan already used by the lock guard; an orphaned control-plane
    process (supervisor gone) does not count.
    """
    try:
        from hermes_cli.process_identity import ledger_entries, spawner_is_dead

        for entry in ledger_entries():
            if entry.get("purpose") not in ("serve", "dashboard"):
                continue
            if spawner_is_dead(entry) is False:
                return True
    except Exception as exc:
        logger.debug("Desktop-lifecycle ledger probe failed: %s", exc)

    try:
        import psutil
    except Exception:
        psutil = None

    try:
        holders = _m()._detect_venv_python_processes()
    except Exception as exc:
        logger.debug("Desktop-lifecycle holder scan failed: %s", exc)
        return False

    for pid, _name, cmdline in holders:
        if not _looks_like_desktop_control_plane(cmdline):
            continue
        if psutil is None:
            # Cannot prove orphanhood; a live this-install control plane is
            # enough to refuse stealing gateway start.
            return True
        try:
            proc = psutil.Process(int(pid))
            parent = proc.parent()
            if parent is None or not parent.is_running():
                continue
            if parent.create_time() > proc.create_time():
                continue
            return True
        except Exception:
            continue
    return False


def _pause_windows_gateways_for_update() -> dict | None:
    """Stop running Windows gateways before mutating the checkout or venv.

    Windows scheduled/startup gateways run through pythonw.exe, so the generic
    hermes.exe concurrent-instance guard does not see them. They still import
    from the checkout and can keep files locked while ``git`` or ``uv`` updates
    the install. Stop only PIDs that the gateway discovery code identifies.
    """
    if not _m()._is_windows():
        return None

    try:
        from gateway.status import terminate_pid
        from hermes_cli.gateway import (
            _capture_gateway_argv,
            _get_restart_drain_timeout,
            find_gateway_pids,
            find_profile_gateway_processes,
        )
    except Exception as exc:
        logger.debug("Could not prepare Windows gateway pause for update: %s", exc)
        return None

    try:
        running_pids = list(dict.fromkeys(find_gateway_pids(all_profiles=True)))
    except Exception as exc:
        logger.debug("Could not discover Windows gateway PIDs before update: %s", exc)
        return None
    if not running_pids:
        # No gateway is running right now, but the user may have installed an
        # autostart entry (Scheduled Task or Startup-folder login item) — that
        # is an explicit "I want a gateway" signal. A gateway that died between
        # updates (e.g. the spawning terminal/TUI closed, taking its child with
        # it) would otherwise never come back: the autostart entry only fires on
        # the next login, and the update flow's resume path only relaunched
        # gateways that were running when the update began. Cold-start one after
        # the update so an installed gateway is actually up post-update. Users
        # who run gateway-less (no autostart entry) get nothing forced on them.
        #
        # Exception: Desktop currently owns this install's gateway lifecycle
        # (live supervised serve/dashboard). A vestigial Startup/Scheduled
        # Task is not the owner — spawning ``gateway run`` beside Desktop
        # races ports/state (#76129). Serve is the control plane, not proof
        # messaging is served; the skip is ownership, not liveness (#92091).
        try:
            if _desktop_owns_gateway_lifecycle():
                logger.debug(
                    "Skipping Windows gateway cold-start plan: "
                    "Desktop owns gateway lifecycle"
                )
                return None
        except Exception as exc:
            logger.debug(
                "Could not check Desktop gateway-lifecycle ownership before update: %s",
                exc,
            )
        try:
            from hermes_cli import gateway_windows

            if gateway_windows.is_installed():
                return {
                    "resume_needed": True,
                    "profiles": {},
                    "unmapped_pids": [],
                    "unmapped": [],
                    "cold_start_if_installed": True,
                }
        except Exception as exc:
            logger.debug(
                "Could not check Windows gateway autostart state before update: %s",
                exc,
            )
        return None

    profile_processes = {}
    try:
        profile_processes = {
            proc.pid: proc for proc in find_profile_gateway_processes()
        }
    except Exception as exc:
        logger.debug("Could not map Windows gateway PIDs to profiles: %s", exc)

    profiles: dict[str, int] = {}
    mapped_pids = []
    for pid in running_pids:
        proc = profile_processes.get(pid)
        if proc is None:
            continue
        profiles[str(proc.profile)] = int(pid)
        mapped_pids.append(int(pid))
        _write_update_planned_stop_marker(Path(proc.path), int(pid))

    # Resolve each mapped worker's venv-side launcher BEFORE draining: the
    # drain stops tracking a PID exactly when it dies, so a gracefully
    # drained worker is gone by the time the wait returns — and a dead pid's
    # parent cannot be recovered (psutil raises NoSuchProcess). The snapshot
    # is stopped after the drain alongside the survivors.
    #
    # Why launchers matter: the drain targets the PID that wrote the PID
    # file (the uv-side worker). On Windows that worker's parent is usually
    # the venv-side ``python.exe`` launcher, which keeps venv ``.pyd`` files
    # mapped and is what ``_detect_venv_python_processes()`` reports
    # downstream. Left alive, it trips the venv-holder guard and aborts the
    # update even though the gateway itself is stopped.
    launcher_pids = _m()._venv_launcher_ancestors(mapped_pids)

    print("→ Stopping Windows gateway process(es) before updating Hermes...")
    try:
        drain_timeout = max(float(_get_restart_drain_timeout()), 1.0)
    except Exception:
        drain_timeout = 10.0
    survivors = _m()._wait_for_windows_update_gateway_exit(
        mapped_pids,
        timeout=drain_timeout,
    )
    unmapped_pids = [pid for pid in running_pids if pid not in profile_processes]

    # Snapshot each unmapped gateway's command line *before* we force-kill it,
    # so ``_resume_windows_gateways_after_update`` can respawn it by replaying
    # its own argv. Unmapped gateways are ones with no profile→PID-file mapping
    # — e.g. a Windows Scheduled Task running ``pythonw.exe -m hermes_cli.main
    # gateway run``. Without this snapshot they were force-killed and never
    # restarted (the "Restart manually after update" dead-end from #50090).
    unmapped: list[dict] = []
    for pid in unmapped_pids:
        argv = None
        try:
            argv = _capture_gateway_argv(int(pid))
        except Exception as exc:
            logger.debug("Could not capture argv for unmapped gateway %s: %s", pid, exc)
        unmapped.append({"pid": int(pid), "argv": argv})

    # Stop drain survivors, unmapped gateways, and the pre-drain launcher
    # snapshot. ``terminate_pid(force=True)`` is a tree kill, so a launcher
    # that outlived its worker takes any stragglers with it; a launcher that
    # already exited with its drained worker raises ProcessLookupError below
    # and is skipped.
    force_killed = []
    for pid in sorted(set(survivors).union(unmapped_pids).union(launcher_pids)):
        try:
            terminate_pid(int(pid), force=True)
            force_killed.append(int(pid))
        except (ProcessLookupError, PermissionError, OSError):
            pass

    if profiles:
        print(f"  ✓ Paused gateway profile(s): {', '.join(sorted(profiles))}")
    if force_killed:
        print(f"  → Force-stopped {len(force_killed)} gateway process(es)")

    if unmapped_pids:
        respawnable = sum(1 for u in unmapped if u.get("argv"))
        print(
            f"  → Stopped {len(unmapped_pids)} gateway process(es) without profile mapping"
        )
        if respawnable < len(unmapped_pids):
            # Some had no recoverable command line (psutil missing, access
            # denied, already gone): those still need a manual restart.
            print("    Restart manually after update: hermes gateway run")

    return {
        "resume_needed": True,
        "profiles": profiles,
        "unmapped_pids": unmapped_pids,
        "unmapped": unmapped,
    }

def _cold_start_windows_gateway_after_update() -> None:
    """Start a fresh detached gateway after update when one is installed but down.

    Invoked from ``_resume_windows_gateways_after_update`` for the
    ``cold_start_if_installed`` case: no gateway was running when the update
    began, but an autostart entry (Scheduled Task / Startup-folder login item)
    is installed, signalling the user wants a gateway. Unlike the relaunch
    paths — which watch an old PID and respawn once it exits — this is a direct
    fresh spawn via the same hidden-console + breakaway path that
    ``hermes gateway start`` uses (``gateway_windows._spawn_detached``).

    Best-effort and idempotent: re-checks that nothing is running first so a
    concurrent start (e.g. the autostart entry firing) can't produce a
    duplicate gateway.

    A successful ``Popen`` only proves the process was created, not that it
    survived (e.g. a Windows job object denying breakaway kills it before it
    logs anything — #84185). So the success line is gated on the same
    post-spawn liveness poll every other ``_spawn_detached`` caller uses
    (``gateway_windows._report_gateway_start``), instead of being printed
    unconditionally from the returned PID.
    """
    if not _m()._is_windows():
        return
    try:
        from hermes_cli import gateway_windows
        from hermes_cli.gateway import find_gateway_pids
    except Exception as exc:
        logger.debug("Could not load Windows gateway cold-start helpers: %s", exc)
        return

    # Re-check liveness right before spawning — between pause and resume the
    # autostart entry may have already brought a gateway up, or a leftover
    # process may have re-registered. Don't double-start.
    try:
        if list(find_gateway_pids(all_profiles=True)):
            return
    except Exception as exc:
        logger.debug("Could not re-check gateway liveness before cold-start: %s", exc)
        return

    try:
        if _desktop_owns_gateway_lifecycle():
            logger.debug(
                "Skipping Windows gateway cold-start: Desktop owns gateway lifecycle"
            )
            return
    except Exception as exc:
        logger.debug(
            "Could not re-check Desktop gateway-lifecycle ownership before cold-start: %s",
            exc,
        )

    try:
        pid = gateway_windows._spawn_detached()
    except Exception as exc:
        logger.debug("Could not cold-start Windows gateway after update: %s", exc)
        return

    if pid:
        print()
        gateway_windows._report_gateway_start(f"cold-start after update (PID {pid})")

def _for_each_systemd_gateway_unit(
    list_units_stdout: str,
    *,
    process_unit,
    on_unit_timeout,
) -> None:
    """Process each ``hermes-gateway*.service``/``hermes-serve*.service`` unit
    from ``systemctl list-units``.

    ``subprocess.TimeoutExpired`` raised by ``process_unit`` is isolated to
    that unit via ``on_unit_timeout`` so one wedged systemctl call cannot
    abort the rest of the fleet (#68523).
    """
    for line in (list_units_stdout or "").strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        unit = parts[0]
        if not unit.endswith(".service"):
            continue
        # list-units is already pattern-filtered, but keep the name gate so a
        # stray non-gateway/serve line cannot enter the restart path.
        # ``unit.startswith("hermes-serve")`` alone would also accept the
        # unrelated ``hermes-server.service`` — require the exact base unit
        # or the hyphenated profile family instead (review on #83595).
        if not (
            unit == "hermes-gateway.service"
            or unit.startswith("hermes-gateway-")
            or unit == "hermes-serve.service"
            or unit.startswith("hermes-serve-")
        ):
            continue
        svc_name = unit.removesuffix(".service")
        try:
            process_unit(svc_name)
        except subprocess.TimeoutExpired as exc:
            on_unit_timeout(svc_name, exc)

def _service_unit_supports_graceful_sigusr1_restart(svc_name: str) -> bool:
    """Whether *svc_name* wires SIGUSR1 to a graceful drain-then-restart.

    Only ``hermes-gateway*`` units run ``gateway/run.py``, which installs the
    SIGUSR1 handler. ``hermes-serve*`` units (#83438) don't, so sending them
    SIGUSR1 would just invoke the default terminate action and burn the full
    drain budget waiting for an exit that was never graceful — go straight to
    the blunt ``systemctl restart`` path for those instead.

    Uses the same strict exact/hyphenated shape as the unit-name gate in
    ``_for_each_systemd_gateway_unit`` so a hypothetical near-prefix unit
    (``hermes-gateway-helper`` is fine — profile units are
    ``hermes-gateway-<profile>`` — but ``hermes-gatewayd``-style names are
    not) can't be sent a SIGUSR1 it doesn't handle.
    """
    return svc_name == "hermes-gateway" or svc_name.startswith("hermes-gateway-")


def _warn_incomplete_gateway_fleet_restart(failed_units: list) -> None:
    """Print an explicit incomplete-update warning for unrestarted units."""
    from hermes_cli.gateway import is_macos

    if not failed_units:
        return
    # Preserve discovery order while de-duplicating.
    seen = set()
    ordered = []
    for name in failed_units:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    print()
    print("⚠ Update incomplete — some units were not restarted:")
    for name in ordered:
        print(f"    - {name}")
    if is_macos():
        # A launchd label reaches this list when launchd was not supervising a
        # live process after the restart (#88848), so the unit is not merely
        # stale — it is very likely deregistered, and `launchctl kickstart`
        # cannot revive a job launchd no longer knows about.
        print("  Listed services may be deregistered from launchd, or still")
        print("  running pre-update code (mixed sys.modules). Recover with:")
        print("    hermes gateway status")
        print("    launchctl list | grep <label>")
        print("    launchctl bootstrap gui/$(id -u) "
              "~/Library/LaunchAgents/<label>.plist")
        return
    print("  Skipped units may still be running pre-update code (mixed")
    print("  sys.modules). Restart them manually, then verify:")
    print("    hermes gateway status")
    if any(not name.startswith("ai.hermes.") for name in ordered):
        print("    systemctl --user restart <unit>   # user-scope")
        print("    sudo systemctl restart <unit>     # system-scope")
    if any(name.startswith("ai.hermes.") for name in ordered):
        print("    launchctl kickstart -k gui/$UID/<label>   # macOS (or user/$UID)")


def _restart_launchd_gateway_after_update(
    *, supervision_verify: bool = True
) -> tuple[list, list]:
    """Restart the invoking profile's launchd gateway after an update.

    #74973 (salvage #75021 by @jeff-mettel): the restart used to be gated on
    ``launchctl list <label>`` exiting 0. A *booted-out* job — plist present,
    definition deregistered from launchd (crashed helper, manual bootout,
    failed prior update) — fails that check, so the whole branch silently
    skipped: no restart, no message, ``KeepAlive`` unable to revive a
    definition launchd no longer knows, and the update still printed
    "Update complete!". ``launchctl list`` is also session-scoped and can
    exit non-zero while the job is alive in its gui/user domain, so it is
    not a reliable classifier at all.

    The fix performs NO list-based classification: when the plist exists,
    ``launchd_restart()`` always runs — it drains a live PID, kickstarts
    with ``-k``, and owns the bootout/bootstrap/kickstart ladder for the
    genuinely unloaded state. Every failure path is loud and names the
    manual recovery command.

    Returns ``(restarted_labels, failed_labels)``. With
    ``supervision_verify`` (the update path), success additionally requires
    launchd reporting a fresh supervised PID (#88848 — "the call returned"
    is not "the gateway is supervised").
    """
    from hermes_cli.gateway import (
        get_launchd_label,
        get_launchd_plist_path,
        launchd_restart,
        wait_for_launchd_gateway_supervision,
    )

    current_label = get_launchd_label()
    try:
        if not get_launchd_plist_path().exists():
            return [], []  # not a launchd install — nothing to do or warn
        try:
            launchd_restart()
        except subprocess.CalledProcessError as e:
            stderr = (getattr(e, "stderr", "") or "").strip()
            print(
                f"  ⚠ Gateway restart failed: {stderr}\n"
                "    The gateway may be DOWN on pre-update code. "
                "Recover manually: hermes gateway restart"
            )
            return [], [current_label]
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        # A plist exists, so a gateway is SUPPOSED to be supervised here —
        # a broken/missing/wedged launchctl is not proof nothing needs
        # restarting. The old code `pass`ed here (#74973's second silent
        # variant); count it and tell the operator.
        print(
            "  ⚠ Could not restart the gateway "
            f"({e.__class__.__name__}: {e}).\n"
            "    Recover manually: hermes gateway restart"
        )
        return [], [current_label]

    if not supervision_verify:
        return [current_label], []

    # launchd_restart() returning is only "restart REQUESTED" — the
    # self-restart branch hands work to the running gateway, a plist reload
    # to a detached helper; both asynchronous. A helper that dies before its
    # first bootstrap (#88848), or a bootstrap that exits 0 without
    # registering (measured on macOS 26.6.1), otherwise reaches "Update
    # complete!" with nothing supervising the gateway. Verified
    # domain-agnostically (a domain locate fails on macOS-26 hosts whose
    # per-user domains reject service management).
    if wait_for_launchd_gateway_supervision(label=current_label):
        return [current_label], []
    print(
        f"  ✗ {current_label} restarted but launchd is not supervising it.\n"
        "    Check logs, then: hermes gateway restart"
    )
    return [], [current_label]


def _restart_macos_launchd_gateways(
    restarted_services: list,
    failed_or_stale_units: list,
    drain_budget: float,
) -> None:
    """Restart every launchd-managed gateway after an update (macOS).

    The code update (git pull) is shared across all profiles, so every
    ``ai.hermes.gateway*`` LaunchAgent must reload it — restarting only the
    invoking profile's service leaves siblings on pre-update ``sys.modules``
    until their next agent turn imports a symbol the old module generation
    doesn't have (#41403).  Parity with the systemd fleet path.

    The invoking profile keeps the existing ``launchd_restart()`` treatment
    (self-restart request → graceful drain → kickstart).  Siblings get the
    same drain-first sequence, with their launchd domain resolved per label:
    a sibling bootstrapped in the other supported domain (``gui/<uid>`` vs
    ``user/<uid>``) must not be kickstarted in the current profile's domain.
    ``subprocess.TimeoutExpired`` is isolated per label so one wedged
    launchctl call cannot leave the rest of the fleet on old code (#68523).
    """
    from hermes_cli.gateway import (
        get_launchd_label,
        get_launchd_plist_path,
        launchd_restart,
        launchd_gateway_labels_for_install,
        _graceful_restart_via_sigusr1,
        _launchd_kickstart,
        _launchd_service_registered,
        _locate_launchd_gateway_service,
        _wait_for_launchd_service_pid,
        wait_for_launchd_gateway_supervision,
    )

    # --- Current profile: unchanged single-service path ---------------------
    _restarted, _failed = _restart_launchd_gateway_after_update(
        supervision_verify=True
    )
    restarted_services.extend(_restarted)
    failed_or_stale_units.extend(_failed)
    current_label = get_launchd_label()

    # --- Sibling profiles ---------------------------------------------------
    for label in launchd_gateway_labels_for_install():
        if label == current_label:
            continue
        try:
            # Locate = liveness + domain in one domain-explicit probe; the
            # kickstart and fresh-PID verification below reuse the located
            # domain, so a sibling in the other gui/user domain can never be
            # probed in one domain and restarted in another.
            domain, old_pid = _locate_launchd_gateway_service(label)
            if domain is None:
                # Installed but not bootstrapped (stopped/uninstalled
                # mid-way) — nothing is running old code here.
                continue
            graceful_ok = False
            if old_pid is not None and old_pid > 0:
                print(f"  → {label}: draining (up to {int(drain_budget)}s)...")
                graceful_ok = _graceful_restart_via_sigusr1(
                    old_pid, drain_timeout=drain_budget
                )
            if graceful_ok and _wait_for_launchd_service_pid(
                label, old_pid=old_pid, timeout=10.0, domain=domain
            ):
                # Unconditional KeepAlive already respawned it on the new
                # code — a hard kickstart now would kill the fresh process.
                restarted_services.append(label)
                continue
            try:
                _launchd_kickstart(label, domain)
            except subprocess.CalledProcessError as e:
                stderr = (getattr(e, "stderr", "") or "").strip()
                failed_or_stale_units.append(label)
                print(
                    f"  ⚠ Failed to restart {label}: {stderr}\n"
                    f"    Recover manually: launchctl kickstart -k {domain}/{label}"
                )
                continue
            if _wait_for_launchd_service_pid(
                label, old_pid=old_pid, timeout=15.0, domain=domain
            ):
                restarted_services.append(label)
            else:
                failed_or_stale_units.append(label)
                print(
                    f"  ✗ {label} failed to come back after restart.\n"
                    f"    Check logs, then: launchctl kickstart -k {domain}/{label}"
                )
        except subprocess.TimeoutExpired:
            failed_or_stale_units.append(label)
            print(
                f"  ⚠ launchctl timed out restarting {label}; "
                "continuing with remaining gateways"
            )


def _surviving_gateway_pids_after_failed_restart():
    """Best-effort PIDs of gateways still running after the restart phase died.

    Returns ``None`` when the answer cannot be determined — most importantly
    when ``hermes_cli.gateway`` itself no longer imports, which is one of the
    ways the restart phase aborts in the first place (the update replaced the
    checkout under a process that already loaded the old modules). ``None`` and
    a non-empty list are both treated as "assume stale" by the caller; only a
    positive empty result is proof that nothing needs restarting.
    """
    try:
        from hermes_cli.gateway import find_gateway_pids

        return list(find_gateway_pids(all_profiles=True))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not probe for surviving gateways after update: %s", exc)
        return None

def _warn_gateway_restart_phase_aborted(exc: BaseException, pids) -> None:
    """Print a recovery warning when the whole restart phase raised.

    Issue #78574: the gateway auto-restart phase was wrapped in a blanket
    ``except Exception`` that only logged at debug level, so an early failure
    (e.g. importing ``hermes_cli.gateway`` from the freshly pulled checkout)
    erased every drain/restart line from the update output. The update still
    printed "Update complete!" and exited 0 while the running gateway kept
    serving pre-update modules against replaced source files — the next turn
    died with an ImportError.
    """
    print()
    print(f"⚠ Update incomplete — gateway auto-restart failed: {exc}")
    if pids:
        listed = ", ".join(str(pid) for pid in pids)
        print(f"  Gateway process(es) still running pre-update code: {listed}")
    else:
        print("  Any gateway still running is serving pre-update code")
        print("  (mixed sys.modules) against the updated checkout.")
    print("  Restart it manually, then verify:")
    print("    hermes gateway restart")
    print("    hermes gateway status")

def _refresh_windows_gateway_launchers() -> None:
    """Regenerate installed Windows gateway launcher scripts after update.

    The Scheduled Task / Startup-folder launchers (``gateway.cmd`` +
    ``gateway.vbs``) are persistence artifacts written once at install time —
    ``hermes update`` never touched them, so installs created before the
    hidden-console rework (aa2ae36c3f) kept launching the gateway through
    ``pythonw.exe`` forever: every descendant spawn flashed a conhost
    (#54220/#56747) and, since #70344, the console-less gateway died at
    startup with ``RuntimeError: sys.stderr is None`` (#71671).

    The task's /TR points at a stable script path, so rewriting the files in
    place retargets the task without any schtasks call (no UAC needed).
    ``_write_task_script`` is idempotent and renders from current code, so
    this is a no-op for modern installs. Best-effort: a failed refresh must
    never fail the update.
    """
    if not _m()._is_windows():
        return
    try:
        from hermes_cli import gateway_windows

        if not gateway_windows.is_installed():
            return
        gateway_windows._write_task_script()
        print("  ✓ Refreshed Windows gateway launcher scripts")
    except Exception as exc:
        logger.debug("Could not refresh Windows gateway launchers after update: %s", exc)

def _refresh_bootstrap_cache_scripts(branch: str = "main") -> None:
    """Sync the installer's bootstrap-cache scripts from the fresh checkout.

    The Desktop GUI updater (``hermes-setup.exe``) executes
    ``$HERMES_HOME/bootstrap-cache/install-<ref>.ps1`` (or ``.sh``) for its
    repair/bootstrap stages. Installer binaries built before the #67193
    cache-refresh fix (June 2026 and earlier) NEVER re-download a cached
    branch-ref script — ``install-main.ps1`` cached at install time is
    reused forever, executing months-stale code with long-fixed bugs (the
    2026-08-09 incident: a June 4 cached script's venv stage lacked the
    #81327 process-tree sweep and died on ``Access denied``). The binary
    has no self-update path, so the poisoned cache outlives every
    ``hermes update``.

    Overwriting the cached script for *branch* with the freshly pulled
    ``scripts/install.ps1`` / ``scripts/install.sh`` on every update turns
    the stale binary's unconditional reuse into a feature: it "reuses" a
    file this function keeps permanently current. Post-#67193 installers
    re-download on each run anyway, so for them this is a harmless
    pre-seed of the same bytes.

    Scope guards, mirroring ``install_script.rs``:

    - Only the cache key for the update-target *branch* is rewritten
      (``sanitize_ref``: non ``[A-Za-z0-9._-]`` chars become ``_``, so
      ``bb/gui`` → ``install-bb_gui.ps1``). Sibling mutable refs cache
      DIFFERENT branches' scripts — updating main must not clobber
      ``install-bb_gui.ps1`` with main's script.
    - Commit-SHA pins are immutable by design and never touched. The
      installer's ``is_valid_commit()`` accepts **7–40** hex chars, so an
      abbreviated pin like ``install-4ce1994.ps1`` is just as immutable as
      a full 40-hex one; the sanitized *branch* is additionally required
      to not itself look like a commit pin (defense in depth against a
      caller passing a SHA as the branch).

    The .ps1 copy gets a UTF-8 BOM to match the installer's cache format
    (#67193 encoding fix). Best-effort: a failed refresh must never fail
    the update.
    """
    try:
        import re as _re

        cache_dir = Path(_m().get_hermes_home()) / "bootstrap-cache"
        if not cache_dir.is_dir():
            return
        # Mirror install_script.rs::sanitize_ref().
        safe_ref = _re.sub(r"[^A-Za-z0-9._-]", "_", str(branch or "main"))
        # Mirror install_script.rs::is_valid_commit(): 7-40 hex chars is an
        # immutable commit pin — abbreviated SHAs included. Never rewrite.
        if _re.fullmatch(r"[0-9a-fA-F]{7,40}", safe_ref):
            return
        refreshed = []
        for kind, src_name in (("ps1", "install.ps1"), ("sh", "install.sh")):
            src = _m().PROJECT_ROOT / "scripts" / src_name
            if not src.is_file():
                continue
            cached = cache_dir / f"install-{safe_ref}.{kind}"
            if not cached.is_file():
                continue  # this ref was never bootstrap-cached — nothing to heal
            data = src.read_bytes()
            if kind == "ps1" and not data.startswith(b"\xef\xbb\xbf"):
                # Match the installer's cache format: PowerShell needs the
                # UTF-8 BOM or localized/em-dash text mis-decodes (#67193).
                data = b"\xef\xbb\xbf" + data
            if cached.read_bytes() == data:
                continue  # already current
            tmp = cached.with_suffix(cached.suffix + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, cached)
            refreshed.append(cached.name)
        if refreshed:
            print(
                "  ✓ Refreshed installer bootstrap-cache script(s): "
                + ", ".join(sorted(refreshed))
            )
    except Exception as exc:
        logger.debug("Could not refresh bootstrap-cache scripts after update: %s", exc)

def _resume_windows_gateways_after_update(token: dict | None) -> None:
    """Restart Windows profile gateways previously paused for update."""
    if not token or not token.get("resume_needed"):
        return
    token["resume_needed"] = False
    if not _m()._is_windows():
        return

    # Regenerate the persisted launcher scripts before respawning anything,
    # so a legacy pythonw-era Scheduled Task / Startup entry comes back on
    # the current hidden-console design at the next login too.
    _m()._refresh_windows_gateway_launchers()

    profiles = token.get("profiles") or {}
    unmapped = token.get("unmapped") or []
    cold_start = bool(token.get("cold_start_if_installed"))
    if not profiles and not any(u.get("argv") for u in unmapped):
        if cold_start:
            _m()._cold_start_windows_gateway_after_update()
        return

    try:
        from hermes_cli.gateway import (
            launch_detached_gateway_restart_by_cmdline,
            launch_detached_profile_gateway_restart,
        )
    except Exception as exc:
        logger.debug("Could not load Windows gateway restart helper: %s", exc)
        return

    relaunched = []
    for profile, old_pid in sorted(profiles.items()):
        try:
            if launch_detached_profile_gateway_restart(str(profile), int(old_pid)):
                relaunched.append(str(profile))
        except Exception as exc:
            logger.debug(
                "Could not restart Windows gateway profile %s after update: %s",
                profile,
                exc,
            )

    # Respawn unmapped gateways (no profile→PID-file mapping, e.g. a Scheduled
    # Task) by replaying the argv we snapshotted before force-killing them.
    unmapped_relaunched = 0
    for entry in unmapped:
        argv = entry.get("argv")
        old_pid = entry.get("pid")
        if not argv or not old_pid:
            continue
        try:
            if launch_detached_gateway_restart_by_cmdline(int(old_pid), list(argv)):
                unmapped_relaunched += 1
        except Exception as exc:
            logger.debug(
                "Could not restart unmapped Windows gateway (pid %s) after update: %s",
                old_pid,
                exc,
            )

    if relaunched:
        print()
        print(f"  ✓ Restarting Windows gateway profile(s): {', '.join(relaunched)}")
    if unmapped_relaunched:
        if not relaunched:
            print()
        print(
            f"  ✓ Restarting {unmapped_relaunched} unmapped Windows gateway process(es)"
        )

def _discard_lockfile_churn(git_cmd, repo_root):
    """Restore tracked ``package-lock.json`` files that npm dirtied locally.

    npm rewrites lockfiles non-deterministically at install/build time. On a
    managed install those diffs are never intentional, so we discard them so
    ``hermes update`` sees a clean tree instead of autostashing every run.
    Best-effort; only ever touches files named ``package-lock.json``.
    """
    try:
        diff = subprocess.run(
            git_cmd + ["diff", "--name-only"],
            cwd=repo_root,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        if diff.returncode != 0:
            return
        dirty_package_dirs = {
            Path(line.strip()).parent
            for line in diff.stdout.splitlines()
            if line.strip().endswith("package.json")
        }
        dirty = [
            line.strip()
            for line in diff.stdout.splitlines()
            if line.strip().endswith("package-lock.json")
            and Path(line.strip()).parent not in dirty_package_dirs
        ]
        if not dirty:
            return
        subprocess.run(
            git_cmd + ["checkout", "--", *dirty],
            cwd=repo_root,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            check=False,
        )
        print(f"→ Discarded npm lockfile churn ({len(dirty)} file(s))")
    except Exception:
        # Never let lockfile cleanup block an update.
        pass

def _normalize_managed_eol(git_cmd, repo_root):
    """Take a managed checkout off ``core.autocrlf=true`` without leaving it dirty.

    Git for Windows ships ``core.autocrlf=true`` in its system config, which
    renormalizes this repo's LF text files to CRLF in the working tree. That
    breaks ``git checkout`` on update with "Your local changes would be
    overwritten", so ``install.ps1`` pins ``core.autocrlf=false`` on the managed
    clone (#67730). Checkouts created before that landed never got the pin and
    cannot receive it — the bootstrap installer reuses its build-pinned
    ``install.ps1`` forever — so ``hermes update``, which ships with the checkout
    itself, is the only path left that can fix them.

    The pin and the cleanup are one operation. Under ``autocrlf=true`` git
    compares normalized content, so a CRLF working tree reads clean; pinning
    alone would expose every text file as modified and hand the update an
    autostash of the whole tree. So the pin is written only after the tree is
    verified clean under it, and a checkout we cannot fully normalize is left
    exactly as it was. Best-effort: never blocks an update.
    """
    # -c, not config: evaluate the tree as it WOULD look pinned, without
    # persisting anything we might not be able to follow through on.
    probe = git_cmd + ["-c", "core.autocrlf=false"]

    def _dirty(*extra):
        out = subprocess.run(
            probe + ["diff", "-z", "--name-only", *extra],
            cwd=repo_root,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        if out.returncode != 0:
            return None
        return {p for p in out.stdout.split("\0") if p}

    def _real_dirty():
        # Files with a *content* change once CRLF differences are ignored.
        # NOTE: ``diff --name-only --ignore-cr-at-eol`` still LISTS CR-only
        # files (the name list is computed from blob/stat differences before
        # the CR filter is applied), so it cannot be used to isolate real
        # edits. ``--numstat`` does honor the filter: a CR-only file produces
        # no numstat record, while a genuinely-edited file does. Parse the
        # paths out of numstat instead.
        out = subprocess.run(
            probe + ["-c", "core.quotepath=false",
                     "diff", "--numstat", "--ignore-cr-at-eol"],
            cwd=repo_root,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        if out.returncode != 0:
            return None
        paths = set()
        for line in out.stdout.splitlines():
            if not line.strip():
                continue
            # Format: "<added>\t<deleted>\t<path>". Rename detection is off in
            # plain diff, so there is exactly one path field per record.
            parts = line.split("\t", 2)
            if len(parts) == 3 and parts[2]:
                paths.add(parts[2])
        return paths

    def _eol_only():
        all_dirty, real_dirty = _dirty(), _real_dirty()
        if all_dirty is None or real_dirty is None:
            return None
        return all_dirty - real_dirty

    try:
        effective = subprocess.run(
            git_cmd + ["config", "--get", "core.autocrlf"],
            cwd=repo_root,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        # Only "true" rewrites LF to CRLF on checkout. Unset, false, and input
        # all leave the working tree alone, so there is nothing to repair.
        if effective.stdout.strip().lower() != "true":
            return

        eol_only = _eol_only()
        if eol_only is None:
            return
        if eol_only:
            # Pathspec over stdin, not argv: a fully renormalized checkout is
            # thousands of paths, well past the Windows command-line limit.
            subprocess.run(
                probe
                + ["checkout", "--pathspec-from-file=-", "--pathspec-file-nul", "--"],
                cwd=repo_root,
                input="\0".join(sorted(eol_only)),
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                check=False,
            )
            if _eol_only():
                # Still dirty — persisting the pin here would only surface churn
                # we failed to clear. Leave the checkout as we found it.
                return
            print(f"→ Normalized line-ending churn ({len(eol_only)} file(s))")

        subprocess.run(
            git_cmd + ["config", "core.autocrlf", "false"],
            cwd=repo_root,
            capture_output=True,
            check=False,
        )
    except Exception:
        # Never let line-ending cleanup block an update.
        pass


def _desktop_app_present(desktop_dir: Path) -> bool:
    """Return whether a packaged or source Desktop build exists."""
    return (
        _m()._desktop_packaged_executable(desktop_dir) is not None
        or _m()._desktop_dist_exists(desktop_dir)
    )


def _rebuild_desktop_after_update(
    desktop_dir: Path, *, had_desktop_app_before_update: bool
) -> bool:
    """Rebuild an installed Desktop app when its source or artifact changed.

    Returns ``False`` only when a rebuild was attempted and failed, so the
    caller can withhold ``✓ Update complete!`` and (in gateway mode) write
    a failing ``.update_exit_code`` (#88251). Every other outcome — nothing
    to rebuild, up to date, build succeeded, Desktop never installed —
    returns ``True``.
    """
    # The release tree is ignored by git and can disappear during an update.
    # Its pre-update presence is enough to restore it; do not make people who
    # have never used Desktop pay for an Electron build.
    has_desktop_app = had_desktop_app_before_update or _desktop_app_present(desktop_dir)
    if not (
        (desktop_dir / "package.json").exists()
        and _m()._resolve_node_runtime_npm()
        and has_desktop_app
    ):
        return True

    print("→ Checking if desktop app needs rebuilding...")
    # Consult the content-hash stamp IN-PROCESS first. The spawned
    # `hermes desktop --build-only` subprocess re-imports the whole CLI stack
    # (~1-3 s) just to reach the same _m()._desktop_build_needed check; when
    # the stamp already says "up to date" we can skip the spawn entirely. The
    # update path never passes --source, so the subprocess would run with
    # source_mode=False — mirror that here. Any error in the pre-check falls
    # through to the subprocess.
    skip_desktop_build = False
    try:
        skip_desktop_build = not _m()._desktop_build_needed(
            desktop_dir, _m().PROJECT_ROOT, source_mode=False
        )
    except Exception:
        skip_desktop_build = False
    if skip_desktop_build:
        print("  ✓ Desktop app up to date")
        return True

    desktop_build_cmd = [sys.executable, "-m", "hermes_cli.main", "desktop", "--build-only"]
    # Capture the (very loud) Electron/vite build output into update.log
    # instead of streaming it to the terminal. On the rare nonzero exit,
    # retry once after waiting again for the venv — this covers a
    # still-settling rebuild window the first wait didn't fully catch — then
    # surface the captured tail so the failure is debuggable.
    #
    # Start the build subprocess with the Hermes-managed Node on PATH: when
    # `hermes update` runs inside the desktop updater chain (Desktop →
    # hermes-setup → hermes update), the shell PATH customizations are lost,
    # so a bare-PATH child would fail with `node: not found` before cmd_gui can
    # self-heal.
    from hermes_constants import with_hermes_node_path

    build_env = with_hermes_node_path()
    build_result = _m()._run_logged_subprocess(
        desktop_build_cmd, cwd=_m().PROJECT_ROOT, env=build_env
    )
    if build_result.returncode != 0:
        build_result = _m()._run_logged_subprocess(
            desktop_build_cmd, cwd=_m().PROJECT_ROOT, env=build_env
        )
    if build_result.returncode != 0:
        print("  ⚠ Desktop build failed (run `hermes desktop` to retry)")
        tail = "\n".join((build_result.stdout or "").strip().splitlines()[-15:])
        if tail:
            print(tail)
        from hermes_constants import display_hermes_home as _dhh

        print(f"  Full build log: {_dhh()}/logs/update.log")
        return False
    print("  ✓ Desktop app up to date")
    return True


def _cmd_update_impl(args, gateway_mode: bool):
    """Body of ``cmd_update`` — kept separate so the wrapper can always
    restore stdio even on ``sys.exit``."""
    # A managed-runtime refresh can replace site-packages before the normal
    # ``.[all]`` install runs. Snapshot while the old environment can still
    # prove which optional backends the user had activated.
    active_lazy_features = _m()._capture_active_lazy_features()
    active_tool_dependencies = _m()._capture_active_tool_dependencies()

    # Snapshot the pre-update version before any code is pulled so the
    # completion line can report the transition (prime-agent#630 port).
    pre_update_version = _read_project_version()
    # In gateway mode, use file-based IPC for prompts instead of stdin
    gw_input_fn = (
        (lambda prompt, default="": _gateway_prompt(prompt, default))
        if gateway_mode
        else None
    )
    assume_yes = bool(getattr(args, "yes", False))
    # --keep-stash (desktop updater): stash local changes so the update can
    # proceed, but never re-apply them afterward — they stay parked in git
    # stash. Only applies when an update actually landed; abort/no-op paths
    # still restore, since the tree they restore onto is unchanged.
    keep_stash = bool(getattr(args, "keep_stash", False))
    # --switch-branch: on a branch carrying unmerged commits, prefer switching
    # to the update target over an in-place merge, so the branch's history is
    # never written to by an update (#89507 review feedback). Only meaningful
    # when updates.parked_branch_strategy is "update_in_place".
    switch_branch = bool(getattr(args, "switch_branch", False))

    # Whether this update is running without a human at the keyboard.
    # Interactive terminal updates always stash-and-ask (unchanged behavior);
    # only non-interactive updates (desktop/chat app, gateway, `--yes`) consult
    # the `updates.non_interactive_local_changes` config setting to decide
    # whether to auto-restore stashed local source changes or throw them away.
    _non_interactive_update = (
        gateway_mode
        or assume_yes
        or not (sys.stdin.isatty() and sys.stdout.isatty())
    )
    discard_local_changes = False
    if _non_interactive_update:
        try:
            from hermes_cli.config import load_config

            _update_cfg = (load_config() or {}).get("updates", {})
            if isinstance(_update_cfg, dict):
                _mode = str(_update_cfg.get("non_interactive_local_changes", "stash")).lower()
                discard_local_changes = _mode == "discard"
        except Exception as exc:
            # Never let a config read failure change the safe default.
            logger.debug("Could not read updates.non_interactive_local_changes: %s", exc)
            discard_local_changes = False

    print("⚕ Updating Hermes Agent...")
    print()

    # Phase 1 (#91277): structured update receipt — record what this run
    # discovers, does, and skips, so silent-failure classes (#88848,
    # #74973, #85753, #81193) become diagnosable from disk.
    try:
        from hermes_cli.update_receipt import begin_update_receipt

        begin_update_receipt()
    except Exception as _receipt_exc:
        logger.debug("Update receipt unavailable: %s", _receipt_exc)

    # Plan phase (#91277 Phase 2): snapshot the pre-update fleet — every
    # running Hermes runtime, its supervisor, and its running code version —
    # into the receipt, so a post-mortem can compare what the update SAW
    # against what it did. Read-only; a probe failure records nothing.
    # ``_pre_update_plan`` is read again AFTER the restart phase to reconcile
    # every planned runtime against the phase's bookkeeping (restart via
    # declared mechanism — the plan is the worklist, not just a printout).
    _pre_update_plan = None
    try:
        from hermes_cli.update_inventory import (
            collect_runtime_inventory,
            record_plan_in_receipt,
        )

        _pre_update_plan = collect_runtime_inventory()
        record_plan_in_receipt(_pre_update_plan)
        if _pre_update_plan.runtimes:
            _n = len(_pre_update_plan.runtimes)
            _profiles = ", ".join(
                sorted({r.profile for r in _pre_update_plan.runtimes})
            )
            print(f"→ Fleet: {_n} running service(s) across profiles: {_profiles}")
    except Exception as _plan_exc:
        logger.debug("Update plan phase failed: %s", _plan_exc)

    # On Windows, abort early if another hermes.exe is holding the venv shim
    # open. Continuing would result in a string of WinError 32 warnings and
    # then either a deferred-rename leftover or a failed git-pull fast path
    # that silently falls back to the slower ZIP route. See issue #26670.
    #
    # Exception (#37039): when every concurrent instance is a gateway
    # runtime, the pause machinery a few lines below
    # (``_pause_windows_gateways_for_update``) stops it before any file
    # mutation, and the post-update restart phase brings it back. Aborting
    # just to make the user run the same kill manually is friction without
    # benefit. Anything not positively identified as a gateway (TUI shell,
    # Desktop backend child, unreadable cmdline) still aborts exactly as
    # before.
    if _m()._is_windows() and not getattr(args, "force", False):
        scripts_dir = _m()._venv_scripts_dir()
        if scripts_dir is not None:
            concurrent = _m()._detect_concurrent_hermes_instances(scripts_dir)
            if concurrent:
                non_gateway = _m()._filter_non_gateway_concurrent_instances(
                    concurrent
                )
                if non_gateway:
                    print(
                        _format_concurrent_instances_message(
                            non_gateway, scripts_dir
                        )
                    )
                    sys.exit(2)

    # Pre-update backup — runs before any git/file mutation so users can
    # always roll back to the exact state they had before this update.
    # Returns the quick-snapshot id (or None when disabled/failed); the
    # post-update cron-jobs safety net uses it to detect job loss.
    pre_update_snapshot_id = _m()._run_pre_update_backup(args)
    try:
        from hermes_cli.update_receipt import record_step

        record_step(
            "pre_update_backup",
            pre_update_snapshot_id is not None,
            f"snapshot={pre_update_snapshot_id}" if pre_update_snapshot_id else "disabled or failed",
        )
    except Exception:
        pass

    _windows_gateway_resume = _m()._pause_windows_gateways_for_update()
    if _windows_gateway_resume:
        import atexit as _atexit

        _atexit.register(
            _m()._resume_windows_gateways_after_update,
            _windows_gateway_resume,
        )

    # With gateways paused, anything still running from the venv interpreter
    # (most commonly the Desktop app's `hermes serve` backend) will keep .pyd
    # files locked and corrupt the dependency sync below. Refuse rather than
    # race: killing the desktop backend is futile (the app supervises and
    # respawns it), so the user must close the app. Deliberately NOT bypassed
    # by plain --force: the desktop bootstrap updater passes --force to skip
    # the hermes.exe shim guard above, but its lock probe only checks the shim
    # and app.asar — a non-desktop venv python holding a .pyd would sail
    # through and corrupt the sync (the exact failure this guard exists for).
    # --force-venv is the explicit escape hatch.
    if _m()._is_windows() and not getattr(args, "force_venv", False):
        _venv_holders = _m()._detect_venv_python_processes()
        if _venv_holders:
            _gateway_holders = _m()._leftover_pausable_gateway_pids(_venv_holders)
            if _gateway_holders is not None:
                # Every remaining holder is a gateway the pause machinery
                # already owns — respawned by its supervisor inside the
                # pause→guard window, or up through a spawn path discovery
                # does not map. Stop them and re-check instead of
                # dead-ending; the post-update resume (and the supervisor
                # that respawned them) brings gateways back afterwards.
                from gateway.status import terminate_pid

                print(
                    f"  ⚠ {len(_gateway_holders)} gateway process(es) still "
                    "hold the venv after the pause; stopping them"
                )
                for _pid in _gateway_holders:
                    try:
                        terminate_pid(int(_pid), force=True)
                    except Exception as exc:
                        logger.debug(
                            "Could not stop leftover gateway %s: %s", _pid, exc
                        )
                _time.sleep(1.0)
                _venv_holders = _m()._detect_venv_python_processes()
        if _venv_holders:
            # Positive-identity rung (runs FIRST, any update context): holders
            # the spawn ledger proves are orphaned Hermes backends — the
            # process self-registered (pid, create_time, purpose, spawner) at
            # startup and its recorded spawner is provably dead. No PPID
            # archaeology, no hand-off contract required.
            _ledger_backends = _m()._ledger_reapable_backend_pids(_venv_holders)
            if _ledger_backends:
                print(
                    f"  ⚠ {len(_ledger_backends)} ledger-identified orphaned "
                    "Hermes backend process(es) hold the venv; stopping their trees"
                )
                _m()._stop_process_trees(_ledger_backends)
                _time.sleep(1.0)
                _venv_holders = _m()._detect_venv_python_processes()
        if _venv_holders:
            _orphan_backends = _m()._orphaned_desktop_backend_pids(_venv_holders)
            if _orphan_backends:
                # Every remaining holder is a Desktop `serve` backend whose
                # supervising app is GONE — the GUI-updater handoff race:
                # Electron's teardown lost the SIGTERM race, exited, and left
                # its backend (and any .hermes-runtime child) holding the
                # venv. Nothing will respawn an orphan, so reap the tree and
                # re-check instead of dead-ending with "Hermes is still
                # running" while no window is open. Backends whose Desktop
                # is still alive never reach here (_orphaned_desktop_
                # backend_pids returns None for them) — that path keeps the
                # refusal, because the app would just respawn what we kill.
                print(
                    f"  ⚠ {len(_orphan_backends)} orphaned Desktop backend "
                    "process(es) still hold the venv; stopping their trees"
                )
                _m()._stop_process_trees(_orphan_backends)
                _time.sleep(1.0)
                _venv_holders = _m()._detect_venv_python_processes()
        if _venv_holders:
            # Final rung before the dead-end: a GUI-updater hand-off
            # (`update --gateway --force` with the update-incomplete marker
            # claimed) means the Desktop is contractually gone and nothing
            # legitimate will respawn a `serve` backend from this venv. The
            # orphan-only reap above bails the instant ANY holder still has a
            # live parent — which stranded a whole swarm of per-profile
            # backends (the tearing-down Electron parent / the venv
            # launcher→worker chain still mid-exit) and hung the update. In
            # the hand-off context those surviving Hermes backends are leaks,
            # live parent or not — reap them by cmdline instead of dead-ending.
            _handoff = False
            try:
                _handoff = bool(getattr(args, "gateway", False)) and _m()._update_marker_path().exists()
            except Exception:
                _handoff = False
            # Fail closed: if we cannot positively verify the shim state
            # (scripts dir unresolvable, detection raised), assume a live
            # shim exists and keep refusing rather than reap.
            _no_live_shim = False
            try:
                _scripts_dir = _m()._venv_scripts_dir()
                if _scripts_dir is not None:
                    _no_live_shim = not _m()._detect_concurrent_hermes_instances(_scripts_dir)
            except Exception:
                _no_live_shim = False
            if _handoff and _no_live_shim:
                _handoff_backends = _m()._handoff_reapable_backend_pids(_venv_holders)
                if _handoff_backends:
                    print(
                        f"  ⚠ {len(_handoff_backends)} Hermes backend process(es) "
                        "still hold the venv after the Desktop hand-off; "
                        "stopping their trees"
                    )
                    _m()._stop_process_trees(_handoff_backends)
                    _time.sleep(1.0)
                    _venv_holders = _m()._detect_venv_python_processes()
        if _venv_holders:
            print(_format_venv_python_holders_message(_venv_holders))
            _m()._resume_windows_gateways_after_update(_windows_gateway_resume)
            sys.exit(2)

    # Self-lock deferral moved: the venv-holder sweep above excludes this
    # process by design (a CLI `hermes update` IS the venv python), and an
    # updater that has imported a native venv extension cannot rewrite its
    # own mapped .pyd (#83569). That check used to run HERE — before the
    # fetch — but firing pre-fetch meant a deferral stranded the user on the
    # OLD checkout, and any startup path that eagerly loaded cryptography
    # turned every Windows update into an exit-2 loop (#86735/#86780/#86781).
    # It now runs via _abort_dependency_sync_if_self_locked() after the code
    # swap, immediately before the dependency sync — the only phase the lock
    # can actually break — and only when the sync would truly rewrite the
    # loaded distribution.

    # Capture this after every fail-closed venv guard, but before either
    # update path can remove the ignored release tree.
    desktop_dir = _m().PROJECT_ROOT / "apps" / "desktop"
    had_desktop_app_before_update = _desktop_app_present(desktop_dir)

    # Try git-based update first, fall back to ZIP download on Windows
    # when git file I/O is broken (antivirus, NTFS filter drivers, etc.)
    use_zip_update = False
    git_dir = _m().PROJECT_ROOT / ".git"

    if not git_dir.exists():
        if sys.platform == "win32":
            use_zip_update = True
        else:
            print("✗ Not a git repository. Please reinstall:")
            print(
                "  curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash"
            )
            sys.exit(1)

    # On Windows, git can fail with "unable to write loose object file: Invalid argument"
    # due to filesystem atomicity issues. Set the recommended workaround.
    if sys.platform == "win32" and git_dir.exists():
        subprocess.run(
            [
                "git",
                "-c",
                "windows.appendAtomically=false",
                "config",
                "windows.appendAtomically",
                "false",
            ],
            cwd=_m().PROJECT_ROOT,
            check=False,
            capture_output=True,
        )

    # Build git command once — reused for fork detection and the update itself.
    git_cmd = ["git"]
    if sys.platform == "win32":
        git_cmd = ["git", "-c", "windows.appendAtomically=false"]

    # Discard npm lockfile churn before any stash/branch logic. npm rewrites
    # tracked package-lock.json files non-deterministically at install/build
    # time (platform-specific optional deps, ideallyInert annotations, etc.),
    # which is never an intentional edit on a managed install but leaves the
    # tree dirty — forcing an autostash on every update and making branch
    # switches fragile. Restoring them first lets the common case (only
    # lockfile churn) update with a clean tree.
    _discard_lockfile_churn(git_cmd, _m().PROJECT_ROOT)
    # Same rationale, different generator: line-ending churn is machine-made
    # dirt on a managed checkout, so clear it (and stop generating it) before
    # the stash/branch logic rather than autostashing the entire tree.
    _normalize_managed_eol(git_cmd, _m().PROJECT_ROOT)

    # Detect if we're updating from a fork (before any branch logic)
    origin_url = _m()._get_origin_url(git_cmd, _m().PROJECT_ROOT)
    is_fork = _is_fork(origin_url)

    if is_fork:
        print("⚠ Updating from fork:")
        print(f"  {origin_url}")
        print()

    if use_zip_update:
        # ZIP-based update for Windows when git is broken
        try:
            desktop_build_ok = _update_via_zip(
                args,
                had_desktop_app_before_update=had_desktop_app_before_update,
            )
        finally:
            _m()._resume_windows_gateways_after_update(_windows_gateway_resume)
        if gateway_mode:
            _write_gateway_update_exit_code(desktop_build_ok)
        return

    # Fetch and pull
    try:

        # Resolve the target branch up front so the fetch can be scoped to it.
        # A bare `git fetch origin` pulls every ref, and this repo carries
        # thousands of auto-generated branches — an unscoped fetch can stall for
        # minutes on a non-single-branch checkout. Fetch only what we update
        # against.
        branch = _m()._resolve_update_branch(args)

        # Self-heal abandoned git lock files (e.g. .git/shallow.lock left by a
        # crashed fetch) before the fetch — otherwise the update fails with
        # "Unable to create .../shallow.lock: File exists" and never reaches
        # the network.
        from hermes_cli.gitlock import clear_stale_git_locks

        cleared = clear_stale_git_locks(_m().PROJECT_ROOT)
        if cleared:
            print("  (removed stale git lock(s): %s)" % ", ".join(cleared))

        print("→ Fetching updates...")
        fetch_result = subprocess.run(
            git_cmd + ["fetch", "origin", branch],
            cwd=_m().PROJECT_ROOT,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        if fetch_result.returncode != 0:
            _print_fetch_failure(fetch_result.stderr)
            sys.exit(1)

        # Get current branch (returns literal "HEAD" when detached)
        result = subprocess.run(
            git_cmd + ["rev-parse", "--abbrev-ref", "HEAD"],
            cwd=_m().PROJECT_ROOT,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            check=True,
        )
        current_branch = result.stdout.strip()

        # Parked-branch guard (2026-08-17 live incident): the checkout can be
        # left parked on a stale feature branch by earlier tooling. Blindly
        # stash-switch-pull-switch-back "updates" main while the running code
        # stays days behind, then prints "✓ Code updated!".
        #
        # What happens next is routed by what the branch carries (which is
        # exactly what the guard measures) plus updates.parked_branch_strategy:
        #
        #   fully merged  -> a stale leftover with nothing to lose: switch
        #                    back to the target.
        #   unmerged: N   -> strategy "switch" (default): switch to the
        #                    target anyway — committed work is safe on the
        #                    branch (git checkout never discards commits) and
        #                    a loud "kept" notice names the branch + count.
        #                    Deterministic, so non-interactive callers
        #                    (desktop update button, gateway /update, cron)
        #                    always reach the target.
        #                    strategy "update_in_place": a maintained custom
        #                    branch (local patches on top of main) is updated
        #                    IN PLACE from origin/<target> — the checkout
        #                    never moves, local commits survive, the running
        #                    code advances. --switch-branch overrides back to
        #                    the switch path for one run.
        #   anything else -> dirty / unverifiable / opted out: touch nothing,
        #                    warn loudly, mark the code update SKIPPED, and
        #                    stop before the post-update steps reinforce the
        #                    stale tree.
        parked_branch_switched = False
        in_place_update = False
        if current_branch != branch and current_branch != "HEAD":
            switch_safe, switch_block_reason = _m()._assess_parked_branch_switch(
                git_cmd, _m().PROJECT_ROOT, current_branch, branch
            )
            if not switch_safe:
                _m()._print_parked_branch_skip_warning(
                    git_cmd,
                    _m().PROJECT_ROOT,
                    current_branch,
                    branch,
                    switch_block_reason,
                )
                print()
                print(
                    "⚠ Update finished — code update SKIPPED"
                    f"{_branch_head_suffix(git_cmd, _m().PROJECT_ROOT)}"
                )
                _m()._resume_windows_gateways_after_update(
                    _windows_gateway_resume
                )
                sys.exit(1)
            if switch_block_reason.startswith("unmerged:"):
                _in_place_configured = False
                try:
                    from hermes_cli.config import load_config as _load_cfg

                    _upd_cfg = (_load_cfg() or {}).get("updates", {})
                    _in_place_configured = (
                        isinstance(_upd_cfg, dict)
                        and _upd_cfg.get("parked_branch_strategy", "switch")
                        == "update_in_place"
                    )
                except Exception as exc:
                    logger.debug(
                        "Could not read updates.parked_branch_strategy: %s", exc
                    )
                if _in_place_configured and not switch_branch:
                    # The merge source must exist upstream; --branch typos
                    # previously surfaced through the checkout failing, which
                    # does not run on this path.
                    verify_ref = subprocess.run(
                        git_cmd + ["rev-parse", "--verify", "--quiet", f"origin/{branch}"],
                        cwd=_m().PROJECT_ROOT,
                        capture_output=True,
                        text=True, encoding="utf-8", errors="replace",
                    )
                    if verify_ref.returncode != 0:
                        print(f"✗ Branch '{branch}' does not exist locally or on origin.")
                        sys.exit(1)
                    in_place_update = True
                    print(
                        f"  ℹ On branch '{current_branch}' — updating it in place from "
                        f"origin/{branch} (no branch switch; local commits preserved)."
                    )
                else:
                    parked_branch_switched = True
                    _m()._print_parked_branch_kept_notice(
                        current_branch,
                        branch,
                        switch_block_reason.split(":", 1)[1],
                    )
            else:
                parked_branch_switched = True
                print(
                    f"  ⚠ Checkout was parked on '{current_branch}' "
                    f"(fully merged) — switching back to {branch}..."
                )

        if not in_place_update and current_branch != branch:
            if current_branch == "HEAD":
                print(
                    f"  ⚠ Currently on detached HEAD — switching to {branch} "
                    "for update..."
                )
            # Stash before checkout so uncommitted work isn't lost
            auto_stash_ref = _m()._stash_local_changes_if_needed(git_cmd, _m().PROJECT_ROOT)
            checkout_result = subprocess.run(
                git_cmd + ["checkout", branch],
                cwd=_m().PROJECT_ROOT,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            )
            if checkout_result.returncode != 0:
                # Local checkout doesn't have this branch yet. Try to set
                # it up as a tracking branch of origin/<branch>. This is
                # the common case when the requested branch exists upstream
                # but was never checked out locally.
                track_result = subprocess.run(
                    git_cmd + ["checkout", "-B", branch, f"origin/{branch}"],
                    cwd=_m().PROJECT_ROOT,
                    capture_output=True,
                    text=True, encoding="utf-8", errors="replace",
                )
                if track_result.returncode != 0:
                    # Restore the user's prior stash before bailing
                    # so we don't leave them stranded in a weird state.
                    if auto_stash_ref is not None:
                        _m()._restore_stashed_changes(
                            git_cmd,
                            _m().PROJECT_ROOT,
                            auto_stash_ref,
                            prompt_user=False,
                            input_fn=gw_input_fn,
                        )
                    print(f"✗ Branch '{branch}' does not exist locally or on origin.")
                    if track_result.stderr.strip():
                        print(f"  {track_result.stderr.strip().splitlines()[0]}")
                    sys.exit(1)
        else:
            auto_stash_ref = _m()._stash_local_changes_if_needed(git_cmd, _m().PROJECT_ROOT)

        prompt_for_restore = (
            auto_stash_ref is not None
            and not assume_yes
            and (gateway_mode or (sys.stdin.isatty() and sys.stdout.isatty()))
        )

        # Check if there are updates. On shallow checkouts `rev-list --count`
        # walks the truncated graph and can report the entire remote ancestry
        # (e.g. "Found 9980 new commit(s)" on a depth-1 install — #53479).
        # The zero/nonzero gate is still sound (HEAD == origin/<branch> counts
        # 0), so keep it, but treat the shallow NUMBER as unknown and recover
        # the real one via the GitHub compare API when possible.
        result = subprocess.run(
            git_cmd + ["rev-list", f"HEAD..origin/{branch}", "--count"],
            cwd=_m().PROJECT_ROOT,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            check=True,
        )
        commit_count = int(result.stdout.strip())

        apply_is_shallow = (
            subprocess.run(
                git_cmd + ["rev-parse", "--is-shallow-repository"],
                cwd=_m().PROJECT_ROOT,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            ).stdout.strip()
            == "true"
        )
        if commit_count > 0 and apply_is_shallow:
            from hermes_cli.banner import _github_compare_behind

            head_sha = subprocess.run(
                git_cmd + ["rev-parse", "HEAD"],
                cwd=_m().PROJECT_ROOT, capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            ).stdout.strip()
            target_sha = subprocess.run(
                git_cmd + ["rev-parse", f"origin/{branch}"],
                cwd=_m().PROJECT_ROOT, capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            ).stdout.strip()
            counted = _github_compare_behind(head_sha, target_sha)
            # counted == 0 means local-ahead (remote tip reachable from HEAD):
            # not behind, fall through to the up-to-date path.
            commit_count = counted if counted is not None else -1

        # A fork can match origin while still trailing upstream. The sync can
        # therefore advance HEAD even though the origin comparison found no
        # commits. Detect that BEFORE taking the no-update return so dependency
        # refreshes, gateway restarts, AND the fleet version matrix still run
        # for the pulled code (#73108 — previously the sync lived inside the
        # commit_count == 0 branch, which returns immediately after: an update
        # that pulled hundreds of upstream commits printed "Already up to
        # date!" and verified nothing).
        if commit_count == 0 and is_fork and branch == "main":
            pre_sync_sha = _capture_head_sha(git_cmd, _m().PROJECT_ROOT)
            _m()._sync_with_upstream_if_needed(git_cmd, _m().PROJECT_ROOT)
            post_sync_sha = _capture_head_sha(git_cmd, _m().PROJECT_ROOT)
            if pre_sync_sha and post_sync_sha and pre_sync_sha != post_sync_sha:
                synced_count = _count_commits_between(
                    git_cmd,
                    _m().PROJECT_ROOT,
                    pre_sync_sha,
                    post_sync_sha,
                )
                # HEAD moving is itself proof of an update. Keep the update
                # path active even if the informational count cannot be read.
                commit_count = max(1, synced_count)

        if commit_count == 0:
            _invalidate_update_cache()

            # Restore stash and switch back to original branch if we moved.
            # EXCEPTION: a parked feature branch we verified clean + fully
            # merged stays on the target — re-parking the checkout on the
            # stale branch is the 2026-08-17 incident all over again.
            if auto_stash_ref is not None:
                _m()._restore_stashed_changes(
                    git_cmd,
                    _m().PROJECT_ROOT,
                    auto_stash_ref,
                    prompt_user=prompt_for_restore,
                    input_fn=gw_input_fn,
                )
            if parked_branch_switched:
                if switch_block_reason.startswith("unmerged:"):
                    _count = switch_block_reason.split(":", 1)[1]
                    print(
                        f"  ✓ Checkout was parked on '{current_branch}' — "
                        f"switched back to {branch}; {_count} unmerged "
                        f"commit(s) kept on '{current_branch}'."
                    )
                else:
                    print(
                        f"  ✓ Checkout was parked on '{current_branch}' (fully "
                        f"merged) — switched back to {branch}."
                    )
            elif current_branch not in {branch, "HEAD"}:
                subprocess.run(
                    git_cmd + ["checkout", current_branch],
                    cwd=_m().PROJECT_ROOT,
                    capture_output=True,
                    text=True, encoding="utf-8", errors="replace",
                    check=False,
                )

            # "No new commits" does not mean the managed interpreter is safe.
            # uv can retain the same CPython patch while python-build-standalone
            # refreshes the embedded SQLite underneath it. Keep the existing
            # update-boundary hook active on this retry path too.
            from hermes_cli.managed_uv import ensure_uv, update_managed_uv

            runtime_repairs = []
            update_managed_uv(repair_observer=runtime_repairs.append)
            ensure_uv(repair_observer=runtime_repairs.append)
            runtime_repaired = next(
                (result for result in runtime_repairs if result.repaired),
                None,
            )

            # A current checkout does NOT imply a healthy install: a previous
            # dependency sync may have failed partway (classic on Windows,
            # where a running gateway/desktop backend keeps .pyd files locked
            # and uv/pip dies with access-denied, stranding the venv between
            # versions). Probe the venv's core imports and repair if broken —
            # otherwise "Already up to date!" gaslights the user while their
            # install stays bricked.
            healthy, detail = _venv_core_imports_healthy()
            # The Windows shim hand-off spawns this child precisely to run a
            # sync its parent could not. The parent already pulled, so the
            # checkout is current BY DESIGN and venv health is not the
            # question — the pending sync is. Without this the child prints
            # "Already up to date!" and exits without doing the one job it
            # was spawned for.
            handed_off_sync = os.environ.get(_m()._UPDATE_REEXEC_ENV) == "1"
            if handed_off_sync:
                print("→ Finishing the dependency install handed off by hermes.exe...")
            elif not healthy:
                print("⚠ Checkout is current, but the venv is unhealthy:")
                print(f"  {detail}")
                print("→ Repairing Python dependencies...")
            if handed_off_sync or not healthy:
                # Self-lock deferral (#86735): the repair rewrites the venv
                # too — same mapped-extension hazard as the update sync.
                _m()._abort_dependency_sync_if_self_locked(_windows_gateway_resume)
                _write_update_incomplete_marker()
                from hermes_cli.managed_uv import ensure_uv

                repair_uv = ensure_uv()
                # A managed install whose venv is gone entirely (interrupted
                # repair after the old venv was moved aside) needs the venv
                # recreated before dependencies can be installed into it.
                venv_python_missing = not (
                    venv_python_path(
                        _m().PROJECT_ROOT / "venv", windows=_m()._is_windows()
                    )
                ).exists()
                if venv_python_missing and repair_uv:
                    print("→ Recreating virtual environment...")
                    subprocess.run(
                        [repair_uv, "venv", "venv"],
                        cwd=_m().PROJECT_ROOT,
                        check=False,
                    )
                if repair_uv:
                    # Isolated from third-party UV env vars (#83914), same as
                    # the main-path and git-path dependency syncs.
                    from hermes_cli.managed_uv import managed_python_env

                    repair_env = managed_python_env()
                    repair_env["VIRTUAL_ENV"] = str(_m().PROJECT_ROOT / "venv")
                    _m()._install_python_dependencies_with_optional_fallback(
                        [repair_uv, "pip"], env=repair_env, group="all"
                    )
                    _m()._refresh_active_lazy_features(
                        [repair_uv, "pip"],
                        env=repair_env,
                        features=active_lazy_features,
                    )
                    _m()._restore_active_tool_dependencies(
                        active_tool_dependencies,
                        [repair_uv, "pip"],
                        env=repair_env,
                    )
                else:
                    _m()._install_python_dependencies_with_optional_fallback(
                        [sys.executable, "-m", "pip"], group="all"
                    )
                    _m()._refresh_active_lazy_features(
                        [sys.executable, "-m", "pip"],
                        features=active_lazy_features,
                    )
                    _m()._restore_active_tool_dependencies(
                        active_tool_dependencies,
                        [sys.executable, "-m", "pip"],
                    )
                _m()._clear_update_incomplete_marker()
                healthy_after, detail_after = _venv_core_imports_healthy()
                if healthy_after:
                    print("✓ Dependencies repaired!")
                    _print_update_completion("✓ Update complete!")
                else:
                    print(f"⚠ Venv still unhealthy after repair: {detail_after}")
                    print("  Close all Hermes windows/gateways and re-run: hermes update")
            else:
                _repair_node_deps_on_current_checkout(_print_update_completion)
            if runtime_repaired is not None and not _m()._is_windows():
                print()
                print(
                    "⚠ Restart required to finish the managed Python runtime repair."
                )
                print(
                    "  Any running Hermes gateways, Desktop backends, or other "
                    "long-lived processes still use the previous runtime."
                )
                print("  Restart each of them to pick up the repaired runtime.")
            _m()._resume_windows_gateways_after_update(_windows_gateway_resume)
            return

        if commit_count > 0:
            print(f"→ Found {commit_count} new commit(s)")
        else:
            # Shallow checkout, exact count unrecoverable (offline/rate-limited
            # compare API) — the tips differ, so there IS an update.
            print("→ Updates available (commit count unknown on this shallow checkout)")

        print("→ Pulling updates...")
        update_succeeded = False
        # Capture the pre-pull SHA so we can auto-roll-back if the new code
        # has a syntax error in a critical-path file (PR #28452 incident:
        # orphan merge-conflict markers in hermes_cli/config.py bricked
        # every user who ran ``hermes update`` for the 7 minutes between
        # the bad commit and the fix landing).
        pre_pull_sha = _capture_head_sha(git_cmd, _m().PROJECT_ROOT)
        try:
            # Merge the ref we already fetched above (→ Fetching updates...)
            # instead of `git pull`, which performs a SECOND network fetch of
            # the same branch (~0.5-1.5 s of redundant round-trip per update).
            # `merge --ff-only origin/<branch>` is byte-identical in effect to
            # `pull --ff-only origin <branch>` given the fresh tracking ref;
            # the divergence fallback below is unchanged.
            pull_result = subprocess.run(
                git_cmd + ["merge", "--ff-only", f"origin/{branch}"],
                cwd=_m().PROJECT_ROOT,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            )
            if pull_result.returncode != 0:
                # ff-only failed — local and remote have diverged. Before
                # assuming an upstream force-push, check WHY: a checkout on a
                # custom branch (local commits on top of origin/<branch>) also
                # cannot fast-forward, and `reset --hard` here would silently
                # discard that work. Merge instead and stop cleanly on
                # conflict — an update must never destroy local commits.
                _cur_branch = (
                    subprocess.run(
                        git_cmd + ["branch", "--show-current"],
                        cwd=_m().PROJECT_ROOT,
                        capture_output=True,
                        text=True, encoding="utf-8", errors="replace",
                    ).stdout
                    or ""
                ).strip()
                if _cur_branch and _cur_branch != branch:
                    print(
                        f"  ⚠ Checkout is on custom branch '{_cur_branch}' — "
                        f"merging origin/{branch} instead of resetting so local commits survive..."
                    )
                    # Best-effort safety tag; recovery anchor if anything goes wrong.
                    subprocess.run(
                        git_cmd
                        + ["tag", f"pre-update-{_time.strftime('%Y%m%d-%H%M%S')}"],
                        cwd=_m().PROJECT_ROOT,
                        capture_output=True,
                        check=False,
                    )
                    merge_result = subprocess.run(
                        git_cmd + ["merge", "--no-edit", f"origin/{branch}"],
                        cwd=_m().PROJECT_ROOT,
                        capture_output=True,
                        text=True, encoding="utf-8", errors="replace",
                    )
                    if merge_result.returncode != 0:
                        subprocess.run(
                            git_cmd + ["merge", "--abort"],
                            cwd=_m().PROJECT_ROOT,
                            capture_output=True,
                            check=False,
                        )
                        print(
                            "✗ Merge conflict between local commits and upstream — "
                            "update stopped, nothing was changed."
                        )
                        print(
                            f"  Resolve manually: cd {_m().PROJECT_ROOT} && "
                            f"git merge origin/{branch}"
                        )
                        print(
                            "  Then re-run the update. Local work is untouched."
                        )
                        sys.exit(1)
                else:
                    # Same branch as the update target — a true upstream
                    # force-push/rebase. Local changes are already stashed;
                    # reset to match the remote exactly (original behaviour).
                    print(
                        "  ⚠ Fast-forward not possible (history diverged), resetting to match remote..."
                    )
                    reset_result = subprocess.run(
                        git_cmd + ["reset", "--hard", f"origin/{branch}"],
                        cwd=_m().PROJECT_ROOT,
                        capture_output=True,
                        text=True, encoding="utf-8", errors="replace",
                    )
                    if reset_result.returncode != 0:
                        print(f"✗ Failed to reset to origin/{branch}.")
                        if reset_result.stderr.strip():
                            print(f"  {reset_result.stderr.strip()}")
                        print(
                            f"  Try manually: git fetch origin && git reset --hard origin/{branch}"
                        )
                        sys.exit(1)

            # Post-pull syntax guard: validate critical-path files actually
            # parse before declaring the update successful. If a bad commit
            # made it through CI (e.g. admin-merge bypass of a failing
            # ruff check), this catches it on the user side and rolls back
            # so the CLI stays bootable. The user can then retry ``hermes
            # update`` later once a fix lands upstream.
            syntax_ok, failing_path, syntax_error = _validate_critical_files_syntax(
                _m().PROJECT_ROOT
            )
            if not syntax_ok:
                print()
                print("✗ Pulled code has a syntax error in a critical file:")
                print(f"  {failing_path}")
                if syntax_error:
                    # py_compile errors can be multi-line; show the first
                    # ~6 lines so the user sees the actual SyntaxError text.
                    for line in str(syntax_error).splitlines()[:6]:
                        print(f"    {line}")
                if pre_pull_sha:
                    print()
                    print(f"→ Rolling back to {pre_pull_sha[:10]}...")
                    rollback_result = subprocess.run(
                        git_cmd + ["reset", "--hard", pre_pull_sha],
                        cwd=_m().PROJECT_ROOT,
                        capture_output=True,
                        text=True, encoding="utf-8", errors="replace",
                    )
                    if rollback_result.returncode == 0:
                        print("  ✓ Rollback complete — your install is unchanged.")
                        print("  Try ``hermes update`` again later once a fix lands.")
                    else:
                        print("  ✗ Rollback failed. Recover manually with:")
                        print(f"    cd {_m().PROJECT_ROOT} && git reset --hard {pre_pull_sha}")
                        if rollback_result.stderr.strip():
                            print(f"    ({rollback_result.stderr.strip().splitlines()[0]})")
                else:
                    print()
                    print("  Could not capture pre-pull SHA — recover manually with:")
                    print(f"    cd {_m().PROJECT_ROOT} && git reflog && git reset --hard <prev-sha>")
                sys.exit(1)

            update_succeeded = True
        finally:
            if auto_stash_ref is not None:
                # Don't attempt stash restore if the code update itself failed —
                # working tree is in an unknown state.
                if not update_succeeded:
                    print(
                        f"  ℹ️  Local changes preserved in stash (ref: {auto_stash_ref})"
                    )
                    print("  Restore manually with: git stash apply")
                elif discard_local_changes:
                    # Non-interactive update + user opted into discarding local
                    # source edits (updates.non_interactive_local_changes:
                    # discard). Throw the stash away instead of re-applying it.
                    _m()._discard_stashed_changes(
                        git_cmd,
                        _m().PROJECT_ROOT,
                        auto_stash_ref,
                    )
                elif keep_stash:
                    # --keep-stash (desktop updater): the update landed; leave
                    # local edits parked in the stash instead of silently
                    # re-applying them onto the updated code.
                    _m()._park_stashed_changes(auto_stash_ref)
                else:
                    _m()._restore_stashed_changes(
                        git_cmd,
                        _m().PROJECT_ROOT,
                        auto_stash_ref,
                        prompt_user=prompt_for_restore,
                        input_fn=gw_input_fn,
                    )

        _invalidate_update_cache()

        # Verify HEAD actually moved (issue #79678). ``merge --ff-only``
        # succeeding only means the merge completed, not that the update
        # applied: a checkout that is pinned to a raw SHA (detached HEAD) can
        # report "N new commit(s)" against origin yet still sit on the old
        # commit afterward (the branch-switch step re-detaches to the SHA).
        # Before this guard, ``hermes update`` printed "✓ Code updated!" and
        # reinstalled deps + rebuilt the desktop app against the stale tree —
        # no error, no warning, ``hermes doctor`` healthy. Compare pre-pull
        # and post-pull HEAD; if they match, surface the no-op instead of
        # claiming success.
        post_pull_sha = _capture_head_sha(git_cmd, _m().PROJECT_ROOT)
        if pre_pull_sha and post_pull_sha == pre_pull_sha:
            print()
            print("✗ Code did not move — update was a no-op.")
            print(
                f"  HEAD is pinned to {pre_pull_sha[:10]} (detached checkout); "
                f"origin/{branch} advanced but the working tree stayed put."
            )
            print(
                "  Reattach to the branch and retry: "
                f"git -C {_m().PROJECT_ROOT} checkout {branch} && hermes update"
            )
            _m()._resume_windows_gateways_after_update(_windows_gateway_resume)
            sys.exit(1)

        # And verify HEAD actually sits on the target branch. The parked-
        # branch guard above should make this unreachable, but if any path
        # leaves the checkout attached elsewhere, "✓ Code updated!" would be
        # a lie — refuse to claim success (2026-08-17 incident class).
        #
        # An IN-PLACE branch update is the one legitimate way to end on a
        # non-target branch: origin/<target> was merged INTO the checked-out
        # branch, so the running code *is* up to date and HEAD staying put is
        # the whole point. Claiming failure there would make every update on a
        # real working branch exit 1 after doing exactly the right thing.
        post_pull_branch = subprocess.run(
            git_cmd + ["rev-parse", "--abbrev-ref", "HEAD"],
            cwd=_m().PROJECT_ROOT,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        ).stdout.strip()
        if (
            not in_place_update
            and post_pull_branch
            and post_pull_branch not in {branch, "HEAD"}
        ):
            print()
            print(
                f"✗ Update pulled origin/{branch}, but the checkout is on "
                f"'{post_pull_branch}' — not claiming success."
            )
            print(
                "  Switch to the target branch and retry: "
                f"git -C {_m().PROJECT_ROOT} checkout {branch} && hermes update"
            )
            _m()._resume_windows_gateways_after_update(_windows_gateway_resume)
            sys.exit(1)

        # Clear stale .pyc bytecode cache — prevents ImportError on gateway
        # restart when updated source references names that didn't exist in
        # the old bytecode (e.g. get_hermes_home added to hermes_constants).
        removed = _m()._clear_bytecode_cache(_m().PROJECT_ROOT)
        if removed:
            print(
                f"  ✓ Cleared {removed} stale __pycache__ director{'y' if removed == 1 else 'ies'}"
            )
        _m()._record_bytecode_fingerprint()
        _m()._refresh_bootstrap_cache_scripts(branch)

        # Fork upstream sync logic (only for main branch on forks)
        if is_fork and branch == "main":
            _m()._sync_with_upstream_if_needed(git_cmd, _m().PROJECT_ROOT)

        # Reinstall Python dependencies. Prefer .[all], but if one optional extra
        # breaks on this machine, keep base deps and reinstall the remaining extras
        # individually so update does not silently strip working capabilities.
        #
        # Self-lock deferral (relocated preflight — #86735): if THIS process
        # holds a native extension the sync must rewrite, defer NOW — after
        # the code swap, so only the dependency install is pending and the
        # next fresh launch completes it via the marker.
        _m()._abort_dependency_sync_if_self_locked(_windows_gateway_resume)
        #
        # Drop the core-install breadcrumb BEFORE touching the venv. If the
        # install is killed mid-flight (Ctrl-C, terminal close, WSL OOM), the
        # marker survives and the next ``hermes`` launch finishes the install
        # via ``_recover_from_interrupted_install``. Cleared after the core
        # ``.[all]`` install completes — lazy refresh uses a separate marker.
        _write_update_incomplete_marker()
        deps_current = _editable_install_is_current(
            git_cmd, _m().PROJECT_ROOT, pre_pull_sha
        )
        if deps_current:
            print("→ Python dependencies unchanged — skipping reinstall")
        else:
            print("→ Updating Python dependencies...")
        from hermes_cli.managed_uv import ensure_uv, update_managed_uv

        # Keep managed uv current — runs `uv self update` if we already have one.
        update_managed_uv()

        uv_bin = ensure_uv()

        pip_cmd = [sys.executable, "-m", "pip"]
        if not uv_bin:
            uv_bin = _ensure_uv_for_termux(pip_cmd)
        install_group = "all"

        if uv_bin:
            # Use official managed_python_env() isolation so third-party
            # UV_PYTHON_INSTALL_DIR (e.g. WorkBuddy) cannot hijack uv; then
            # point VIRTUAL_ENV at this install's venv.
            from hermes_cli.managed_uv import managed_python_env

            uv_env = managed_python_env()
            uv_env["VIRTUAL_ENV"] = str(_m().PROJECT_ROOT / "venv")
            if _m()._is_termux_env(uv_env):
                uv_env.pop("PYTHONPATH", None)
                uv_env.pop("PYTHONHOME", None)
                install_group = "termux-all"
                print("  → Termux detected: using uv + curated termux-all optional profile...")
            if not deps_current:
                if _m()._is_termux_env(uv_env) and _is_android_python():
                    print("  → Termux/Android detected: prebuilding psutil with Linux source path compatibility...")
                    _install_psutil_android_compat([uv_bin, "pip"], env=uv_env)
                _m()._install_python_dependencies_with_optional_fallback(
                    [uv_bin, "pip"], env=uv_env, group=install_group
                )
        else:
            # Use sys.executable to explicitly call the venv's pip module,
            # avoiding PEP 668 'externally-managed-environment' errors on Debian/Ubuntu.
            # Some environments lose pip inside the venv; bootstrap it back with
            # ensurepip before trying the editable install.
            pip_cmd = [sys.executable, "-m", "pip"]
            try:
                subprocess.run(
                    pip_cmd + ["--version"],
                    cwd=_m().PROJECT_ROOT,
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError:
                subprocess.run(
                    [sys.executable, "-m", "ensurepip", "--upgrade", "--default-pip"],
                    cwd=_m().PROJECT_ROOT,
                    check=True,
                )
            if _m()._is_termux_env():
                install_group = "termux-all"
                print("  → Termux detected: using curated termux-all optional profile...")
            if not deps_current:
                if _m()._is_termux_env() and _is_android_python():
                    print("  → Termux/Android detected: prebuilding psutil with Linux source path compatibility...")
                    _install_psutil_android_compat(pip_cmd)
                _m()._install_python_dependencies_with_optional_fallback(pip_cmd, group=install_group)

        install_prefix = [uv_bin, "pip"] if uv_bin else pip_cmd
        lazy_env = uv_env if uv_bin else None

        if deps_current:
            # The verification normally runs inside the install we just
            # skipped. Run it here so a wrong skip self-heals into a real
            # install (both verifiers reinstall what they find missing)
            # instead of leaving a venv nobody checked.
            _m()._verify_core_dependencies_installed(
                install_prefix, env=lazy_env, group=install_group
            )
            _m()._verify_console_scripts_installed(install_prefix, env=lazy_env)

        # Core ``.[all]`` install finished. Clear the generic core breadcrumb
        # before the lazy-refresh phase — that phase uses its own marker so a
        # later lazy failure cannot be "healed" by clearing the core marker
        # based on a narrow 7-package import probe (#58004 review).
        _m()._clear_update_incomplete_marker()

        # The update process is still the old Python interpreter process. Run
        # one final cache/module refresh immediately before lazy backend
        # refresh, which imports newly-pulled modules that may depend on fresh
        # symbols in hermes_constants or lazy_deps. The dependency install
        # above may also have regenerated bytecode from build-cache copies —
        # this second sweep catches those stragglers (#60242, #65240).
        removed = _m()._clear_bytecode_cache(_m().PROJECT_ROOT)
        if removed:
            print(
                f"  ✓ Cleared {removed} stale __pycache__ director{'y' if removed == 1 else 'ies'}"
            )
        _m()._record_bytecode_fingerprint()
        _m()._refresh_bootstrap_cache_scripts(branch)
        _m()._reload_updated_runtime_modules()

        # Upgrade pip before lazy refreshes — stale pip can fail source builds
        # and leave partially-written packages (#57828).
        _write_lazy_refresh_incomplete_marker()
        _m()._upgrade_pip_before_lazy_refresh(install_prefix, env=lazy_env)

        # Lazy refresh can corrupt the venv when a backend install fails.
        # Clear the lazy marker only when refresh/repair is confirmed healthy.
        lazy_ok = _m()._refresh_active_lazy_features(
            install_prefix,
            env=lazy_env,
            features=active_lazy_features,
        )
        if lazy_ok:
            _m()._clear_lazy_refresh_incomplete_marker()
        else:
            print(
                "  ⚠ Lazy-refresh recovery incomplete — run `hermes` again "
                "to finish import-based venv repair."
            )

        _m()._restore_active_tool_dependencies(
            active_tool_dependencies,
            install_prefix,
            env=lazy_env,
        )

        # Heal the active memory provider's bridge packages last — the core
        # reinstall + lazy refresh above may have stripped or downgraded
        # plugin.yaml-declared deps that aren't in extras (#53272, #70636).
        _m()._refresh_active_memory_provider_dependencies()

        # Everything that can legitimately produce a transient ImportError has
        # now run (bytecode sweep, dependency reinstall, lazy refresh), so a
        # module that still won't import is real breakage. Warn only — never
        # roll back here: `cannot import name X` is also the signature of the
        # stale-bytecode class (#6207, #60242), and the launch-time sweep in
        # _sweep_stale_bytecode_if_checkout_changed() self-heals that on the
        # next run. A destructive reset would undo a good update over a state
        # that fixes itself.
        import_ok, failing_module, import_error = _validate_critical_modules_import(
            _m().PROJECT_ROOT
        )
        if not import_ok:
            print()
            print(f"  ⚠ {failing_module} still fails to import after updating:")
            print(f"      {import_error}")
            print("    Run `hermes update` again — if it persists, reinstall:")
            print("    https://hermes-agent.nousresearch.com")

        node_failures = _update_node_dependencies()
        _m()._build_web_ui(_m().PROJECT_ROOT / "web")

        desktop_build_ok = _rebuild_desktop_after_update(
            desktop_dir,
            had_desktop_app_before_update=had_desktop_app_before_update,
        )

        print()
        print(f"✓ Code updated!{_branch_head_suffix(git_cmd, _m().PROJECT_ROOT)}")

        # ── macOS TCC stale-grant notice (#86385) ──────────────────────
        # Locally-built desktop bundles are re-signed on every update. With the
        # post-#73681 identifier-pinned DR, new grants survive rebuilds — but a
        # grant made to a pre-fix binary stays stale: the System Settings toggle
        # shows ON while macOS re-prompts on every capture, and the modern prompt
        # has no Allow button, so users loop. One line of guidance after update
        # tells affected users how to complete the one-time re-grant.
        if sys.platform == "darwin" and had_desktop_app_before_update:
            print()
            print(
                "  ℹ macOS: if Hermes re-prompts for permissions you already "
                "granted (toggle shows ON), the stored grant is stale — run "
                "`tccutil reset ScreenCapture com.nousresearch.hermes` (repeat "
                "per affected service), toggle it ON in System Settings, then "
                "fully quit & relaunch once."
            )

        # NOTE: the macOS TCC interpreter anchor that used to refresh here
        # (#95131/#95478) is REVERTED: the anchored real-file copy could not
        # load libpython (LC_RPATH resolved into venv/lib/), bricking every
        # hermes command on real Macs (#95425), and re-pointed aliases lost
        # the stdlib (#95541). `hermes doctor` now heals already-anchored
        # venvs back to symlinks. Re-land requires a dylib-complete design
        # verified on macOS hardware first.

        # ── Post-update state.db integrity guard (#68474) ─────────────────
        # Verify that state.db survived the update intact.  If the live file
        # is now corrupted (zeroed, missing header, integrity failure),
        # automatically restore from the pre-update snapshot rather than
        # letting the user discover silently that their sessions are gone.
        try:
            from hermes_cli.backup import _quick_snapshot_root, verify_sqlite_integrity

            _state_path = get_hermes_home() / "state.db"
            if _state_path.exists():
                _state_ok = verify_sqlite_integrity(
                    _state_path,
                    check_header=True,
                    run_pragma=True,
                )
                if _state_ok.get("valid"):
                    logger.debug(
                        "Post-update state.db integrity check: %s",
                        _state_ok.get("message"),
                    )
                else:
                    print()
                    print(
                        "⚠ state.db is corrupted after update: "
                        + _state_ok.get("message", "unknown error")
                    )
                    _pre_snap_id = pre_update_snapshot_id
                    if _pre_snap_id:
                        _snap_state = (
                            _quick_snapshot_root(get_hermes_home())
                            / _pre_snap_id
                            / "state.db"
                        )
                        if _snap_state.exists():
                            _snap_ok = verify_sqlite_integrity(
                                _snap_state, check_header=True, run_pragma=True
                            )
                            if _snap_ok.get("valid"):
                                try:
                                    if _restore_state_db_from_snapshot(
                                        _state_path, _snap_state
                                    ):
                                        print(
                                            "  ✓ Auto-restored from pre-update "
                                            f"snapshot ({_pre_snap_id})"
                                        )
                                    else:
                                        print(
                                            "  ✗ Auto-restore FAILED — restored "
                                            "copy also failed integrity"
                                        )
                                except OSError as _exc:
                                    print(
                                        f"  ✗ Auto-restore file copy failed: {_exc}"
                                    )
                            else:
                                print(
                                    "  ✗ Pre-update snapshot also failed integrity"
                                )
                        else:
                            print(
                                "  ⚠ Pre-update snapshot does not contain state.db"
                            )
                    else:
                        print("  ⚠ No pre-update snapshot was taken")
                    print()
        except Exception as exc:
            logger.debug("Post-update state.db integrity check failed: %s", exc)

        # Seed the model-catalog disk cache from the freshly-pulled checkout.
        # The repo ships the canonical catalog at
        # website/static/api/model-catalog.json, and `git pull` just made it
        # current — so copy it straight over ~/.hermes/cache/model_catalog.json
        # instead of waiting on a network fetch (which can be bot-gated or hit a
        # Portal hiccup). Keeps the model picker's curated/free lists in sync
        # with the version the user just installed. Non-fatal on failure: the
        # normal network refresh still applies on the next picker open.
        try:
            from hermes_cli.model_catalog import seed_cache_from_checkout

            if seed_cache_from_checkout(_m().PROJECT_ROOT):
                print("  ✓ Model catalog cache refreshed from checkout")
        except Exception as e:
            logger.debug("Model catalog seed during update failed: %s", e)

        # Sync bundled skills (copies new, updates changed, respects user deletions)
        try:
            from tools.skills_sync import sync_skills

            print()
            print("→ Syncing bundled skills...")
            result = sync_skills(quiet=True)
            if result["copied"]:
                print(f"  + {len(result['copied'])} new: {', '.join(result['copied'])}")
            if result.get("updated"):
                print(
                    f"  ↑ {len(result['updated'])} updated: {', '.join(result['updated'])}"
                )
            if result.get("user_modified"):
                print(f"  ~ {len(result['user_modified'])} user-modified (kept)")
                print(
                    "    → see them: hermes skills list-modified  "
                    "(diff/reset to resume updates)"
                )
            if result.get("cleaned"):
                print(f"  − {len(result['cleaned'])} removed from manifest")
            if result.get("relocated"):
                print(
                    f"  → {len(result['relocated'])} moved to new upstream paths: "
                    f"{', '.join(result['relocated'])}"
                )
            if not result["copied"] and not result.get("updated"):
                print("  ✓ Skills are up to date")
        except Exception as e:
            logger.debug("Skills sync during update failed: %s", e)

        # Sync bundled skills to all profiles (including the active one).
        # seed_profile_skills() uses subprocess with an explicit HERMES_HOME so
        # it is not affected by sync_skills()'s module-level HERMES_HOME cache,
        # which means the active profile is reliably synced regardless of whether
        # the caller's HERMES_HOME env var points at the default or a named profile.
        try:
            from hermes_cli.profiles import (
                list_profiles,
                seed_profile_skills,
            )

            all_profiles = list_profiles()
            if all_profiles:
                print()
                print("→ Syncing bundled skills to all profiles...")
                for p in all_profiles:
                    try:
                        r = seed_profile_skills(p.path, quiet=True)
                        if r and r.get("skipped_opt_out"):
                            status = "opted out (--no-skills)"
                        elif r:
                            copied = len(r.get("copied", []))
                            updated = len(r.get("updated", []))
                            modified = len(r.get("user_modified", []))
                            parts = []
                            if copied:
                                parts.append(f"+{copied} new")
                            if updated:
                                parts.append(f"↑{updated} updated")
                            if modified:
                                parts.append(f"~{modified} user-modified")
                            status = ", ".join(parts) if parts else "up to date"
                        else:
                            status = "sync failed"
                        print(f"  {p.name}: {status}")
                    except Exception as pe:
                        print(f"  {p.name}: error ({pe})")
        except Exception:
            pass  # profiles module not available or no profiles

        # Backfill per-profile .env files for profiles created before the
        # .env-seeding fix (#44792). Copies the default install's .env so
        # those profiles keep the credentials they were effectively using.
        try:
            from hermes_cli.profiles import backfill_profile_envs

            backfilled = backfill_profile_envs(quiet=True)
            if backfilled:
                print()
                print(
                    f"→ Seeded .env for {len(backfilled)} profile(s) "
                    f"(copied from default): {', '.join(backfilled)}"
                )
        except Exception:
            pass  # profiles module not available or no profiles

        # Sync Honcho host blocks to all profiles
        try:
            from plugins.memory.honcho.cli import sync_honcho_profiles_quiet

            synced = sync_honcho_profiles_quiet()
            if synced:
                print(f"\n-> Honcho: synced {synced} profile(s)")
        except Exception:
            pass  # honcho plugin not installed or not configured

        # Check for config migrations.
        #
        # CRITICAL: check_config_version and migrate_config must use
        # freshly-reloaded modules, not the sys.modules cache. The
        # ``hermes update`` process is the PRE-pull Python process — its
        # ``sys.modules`` cache holds the OLD ``hermes_cli.config`` and
        # ``hermes_cli.config_migrations`` from before ``git pull`` updated
        # the source files. A function-level ``from hermes_cli.config import
        # check_config_version`` returns the cached module, so
        # ``DEFAULT_CONFIG["_config_version"]`` is the OLD value and
        # ``check_config_version()`` reports ``(33, 33)`` — "up to date" —
        # even though the freshly-pulled code has v34 with a migration to
        # run. The personality reset migration (#81946) was silently skipped
        # this way, leaving ``display.personality: kawaii`` active after
        # updates that should have reset it.
        print()
        print("→ Checking configuration for new options...")

        # Reload config modules BEFORE any config reads so get_missing_*,
        # check_config_version, and migrate_config all use the updated code.
        _reload_config_modules()

        from hermes_cli.config import (
            get_missing_env_vars,
            get_missing_config_fields,
        )

        missing_env = get_missing_env_vars(required_only=True)
        missing_config = get_missing_config_fields()
        current_ver, latest_ver = _run_config_check_fresh()

        has_new_options = bool(missing_env or missing_config)
        version_bump_only = (
            not has_new_options and current_ver < latest_ver
        )
        needs_migration = has_new_options or current_ver < latest_ver

        if version_bump_only:
            # Nothing for the user to fill in — only the config format version
            # changed (new defaults already merge in transparently). Asking
            # "configure new options now?" here is misleading: saying yes just
            # bumps the version and looks like a no-op (issue: ScottFive /
            # Tt2021). Apply it silently and say what actually happened.
            print()
            print(
                f"  ℹ Updating config format (v{current_ver} → v{latest_ver})…"
            )
            try:
                _mig_results = _run_migrate_config_fresh(
                    interactive=False, quiet=True
                )
                print("  ✓ Config format updated (no new settings to configure)")
                # quiet=True also mutes migration steps that RESET or REMOVE an
                # existing setting (e.g. the v33→v34 personality reset from
                # #81946, which records its note only in the results dict).
                # Re-surface those notes so an unattended update never silently
                # changes user configuration (#86656). In this branch
                # missing_config is empty, so config_added can only contain
                # migration-step mutations, not missing-key listings.
                for _note in _mig_results.get("config_added") or []:
                    print(f"  ℹ {_note}")
                for _warn in _mig_results.get("warnings") or []:
                    print(f"  ⚠️  {_warn}")
            except Exception as _mig_err:
                print(f"  ⚠️  Config format update failed: {_mig_err}")
                print("     Run 'hermes config migrate' to retry.")
        elif needs_migration:
            print()
            # Show WHAT changed, not just a count, so the user can make an
            # informed yes/no decision (previously the prompt named nothing).
            def _print_items(items, label, key, fallback_key=None):
                if not items:
                    return
                print(f"  {label}:")
                shown = items[:8]
                for it in shown:
                    if isinstance(it, dict):
                        name = it.get(key) or (fallback_key and it.get(fallback_key)) or "?"
                        desc = (it.get("description") or "").strip()
                    else:
                        # Defensive: some callers/mocks pass bare name strings.
                        name = str(it)
                        desc = ""
                    if desc:
                        print(f"      • {name} — {desc}")
                    else:
                        print(f"      • {name}")
                extra = len(items) - len(shown)
                if extra > 0:
                    print(f"      … and {extra} more")

            if missing_env:
                print(
                    f"  ⚠️  {len(missing_env)} new required setting(s) need configuration"
                )
                _print_items(missing_env, "New settings", "name")
            if missing_config:
                print(f"  ℹ️  {len(missing_config)} new config option(s) available")
                _print_items(missing_config, "New options", "key")

            print()
            if assume_yes:
                print(
                    "  ℹ --yes: auto-applying config migration (skipping API-key prompts)."
                )
                response = "y"
            elif gateway_mode:
                response = (
                    _gateway_prompt(
                        "Would you like to configure new options now? [Y/n]", "n"
                    )
                    .strip()
                    .lower()
                )
            elif not (sys.stdin.isatty() and sys.stdout.isatty()):
                print("  ℹ Non-interactive session — applying safe config migrations.")
                response = "auto"
            else:
                try:
                    response = (
                        input("Would you like to configure them now? [Y/n]: ")
                        .strip()
                        .lower()
                    )
                except EOFError:
                    response = "n"
                except UnicodeDecodeError:
                    # input() can raise this when the terminal encoding can't
                    # decode the byte sequence (e.g. a non-UTF-8 locale, or an
                    # embedded terminal). Without this, the exception escapes
                    # here and crashes the update at this prompt.
                    print(
                        "  ⚠ Could not read input (encoding issue). Skipping. "
                        "Run 'hermes config migrate' manually to configure."
                    )
                    response = "n"

            if response in {"", "y", "yes", "auto"}:
                print()
                # Gateway mode, --yes, and non-interactive update contexts
                # (dashboard / web server actions) cannot prompt for API keys.
                # Still run the non-interactive migration pass before restarting
                # so new default config fields and version bumps are written
                # before the freshly updated gateway validates config at startup.
                interactive_migration = not (
                    gateway_mode or assume_yes or response == "auto"
                )
                results = _run_migrate_config_fresh(interactive=interactive_migration, quiet=False)

                if results["env_added"] or results["config_added"]:
                    print()
                    print("✓ Configuration updated!")
                if (gateway_mode or assume_yes or response == "auto") and missing_env:
                    print("  ℹ API keys require manual entry: hermes config migrate")
            else:
                print()
                print("Skipped. Run 'hermes config migrate' later to configure.")
        else:
            print("  ✓ Configuration is up to date")

        # Fleet-wide config migration (#91277 Phase 2; #20438 earliest report,
        # #54926, #79048): the shared checkout serves EVERY profile, but the
        # migration above only touched the active profile's config.yaml.
        # Sibling profiles kept their old _config_version and silently
        # drifted (field repro: sibling gateway restarted onto new code but
        # stayed at config v33 vs v37). Run the same NON-INTERACTIVE safe
        # migration for every sibling profile home, scoped via the
        # context-local HERMES_HOME override (never os.environ — other
        # threads must not see it).
        try:
            _migrated_siblings = _migrate_sibling_profile_configs()
            for _name, _from_ver, _to_ver in _migrated_siblings:
                print(
                    f"  ✓ Profile '{_name}': config format updated "
                    f"(v{_from_ver} → v{_to_ver})"
                )
        except Exception as exc:
            logger.debug("Sibling config migration failed: %s", exc)

        # Safety net: config-version migrations have been observed to leave
        # cron/jobs.json valid-but-empty, silently dropping every scheduled
        # job (issue #34600). The desktop scheduler can also overwrite with
        # its own small set, causing partial loss (issue #52144). If the
        # live file now has fewer jobs than the pre-update snapshot, restore
        # it and warn loudly.
        try:
            from hermes_cli.backup import restore_cron_jobs_if_emptied

            cron_restore = restore_cron_jobs_if_emptied(pre_update_snapshot_id)
            if cron_restore:
                print()
                print(
                    "  ⚠️  cron/jobs.json lost jobs during this update — "
                    f"restored {cron_restore['job_count']} job(s) from "
                    f"pre-update snapshot {cron_restore['snapshot_id']}."
                )
        except Exception as exc:
            # Never let the cron safety net break an otherwise-good update.
            logger.debug("Cron jobs auto-restore check failed: %s", exc)

        # #66140: run the same cron-jobs safety net for every sibling
        # profile against ITS OWN pre-update snapshot (same-generation by
        # construction — both taken by this run).
        try:
            from hermes_cli.backup import restore_cron_jobs_all_profiles

            for _restored in restore_cron_jobs_all_profiles(
                _LAST_SIBLING_SNAPSHOTS
            ):
                print()
                print(
                    f"  ⚠️  Profile '{_restored['profile']}': cron/jobs.json "
                    f"lost jobs during this update — restored "
                    f"{_restored['job_count']} job(s) from pre-update "
                    f"snapshot {_restored['snapshot_id']}."
                )
        except Exception as exc:
            logger.debug("Sibling cron auto-restore check failed: %s", exc)

        _print_update_summary(
            node_failures=node_failures,
            desktop_build_ok=desktop_build_ok,
            pre_update_version=pre_update_version,
        )

        # Search-index optimization notice (v23). Existing installs keep their
        # working search index untouched on update; the compact v23 layout —
        # which reclaims a large fraction of state.db on heavy users — is
        # opt-in. Surface it here (the moment the user is already thinking
        # about their install) with the exact command and the concrete size
        # win. Show-once-ish: only when a legacy index is actually present.
        try:
            _print_fts_optimize_available_notice()
        except Exception as e:
            logger.debug("FTS optimize notice failed: %s", e)

        # Curator first-run heads-up. Only prints when curator is enabled AND
        # has never run — i.e. the window where the ticker would otherwise
        # have fired against a fresh skill library. Kept silent on steady
        # state so we don't nag.
        try:
            _print_curator_first_run_notice()
        except Exception as e:
            logger.debug("Curator first-run notice failed: %s", e)

        # Most-recent curator run notice — show-once per run. Surfaces the
        # rename map (`old-name → umbrella`) on the high-attention update
        # surface so users learn about consolidations without having to
        # check `hermes curator status`. Self-stamps after printing so it
        # never repeats for the same run.
        try:
            _print_curator_recent_run_notice()
        except Exception as e:
            logger.debug("Curator recent-run notice failed: %s", e)

        # Repair RHEL-family root installs where /usr/local/bin isn't on PATH
        # for non-login interactive shells.  No-op on every other platform.
        try:
            _ensure_fhs_path_guard()
        except Exception as e:
            logger.debug("FHS PATH guard check failed: %s", e)

        # Self-heal the hermes-acp launcher for installs that predate it, so
        # ACP hosts (Zed, JetBrains, Buzz) can resolve Hermes on PATH without
        # a reinstall.  No-op on Windows (the launcher migration below owns
        # that) and when already present.
        try:
            _ensure_acp_launcher()
        except Exception as e:
            logger.debug("hermes-acp launcher self-heal failed: %s", e)

        # Migrate the Windows hermes launchers to the managed binary dir
        # (the default Hermes root's bin, next to the managed uv) and repair
        # them if they are missing. Earlier layouts put them inside the git
        # checkout (hermes-agent\bin) or put venv\Scripts itself on PATH; the
        # in-checkout copies were swept by this command's own pre-update
        # autostash (git stash push --include-untracked) and, with
        # --keep-stash (the desktop updater), never restored — `hermes`
        # stopped resolving in every new terminal. Updates never run
        # install.ps1, so this tail call is how existing installs reach the
        # new layout. No-op on POSIX and on source checkouts (root is not
        # the managed clone under the default Hermes root).
        try:
            from hermes_cli._install_repair import migrate_windows_bin_path

            migrate_windows_bin_path(_m().PROJECT_ROOT)
        except Exception as e:
            logger.debug("Windows bin launcher migration failed: %s", e)

        # Refresh the cua-driver binary used by the Computer Use toolset.
        # The upstream installer is gated on supported platforms and on the
        # binary already being on PATH, so this is a no-op for users who
        # don't have it. Tying the refresh to ``hermes update`` gives users a
        # predictable cadence (matches when they pull new agent code) without
        # adding startup latency or a per-launch GitHub API call.
        try:
            refresh_cua_driver = True
            try:
                from hermes_cli.config import load_config

                _update_cfg = (load_config() or {}).get("updates", {})
                if isinstance(_update_cfg, dict):
                    refresh_cua_driver = bool(
                        _update_cfg.get("refresh_cua_driver", True)
                    )
            except Exception as cfg_exc:
                logger.debug("Could not read updates.refresh_cua_driver: %s", cfg_exc)

            if (
                refresh_cua_driver
                and sys.platform in ("darwin", "win32", "linux")
                and shutil.which("cua-driver")
            ):
                from hermes_cli.tools_config import install_cua_driver

                print()
                print("→ Refreshing cua-driver (Computer Use)...")
                # require_confirmed_update: only run the (multi-minute,
                # silent) upstream installer when the driver's native
                # check-update verb positively reports a newer release.
                # An indeterminate check (offline, rate-limited, old
                # driver) keeps the installed version — `hermes update`
                # must stay fast; `hermes computer-use install --upgrade`
                # remains the force path. Windows also defers confirmed
                # updates and contract repairs to that explicit command
                # because the upstream installer may prompt for console/UAC
                # consent that this hidden updater cannot provide.
                install_cua_driver(
                    upgrade=True,
                    require_confirmed_update=True,
                    show_installer_progress=False,
                )
        except Exception as e:
            logger.debug("cua-driver refresh failed: %s", e)

        # Write exit code *before* the gateway restart attempt.
        # When running as ``hermes update --gateway`` (spawned by the gateway's
        # /update command), this process lives inside the gateway's systemd
        # cgroup.  A graceful SIGUSR1 restart keeps the drain loop alive long
        # enough for the exit-code marker to be written below, but the
        # fallback ``systemctl restart`` path (see below) kills everything in
        # the cgroup (KillMode=mixed → SIGKILL to remaining processes),
        # including us and the wrapping bash shell.  The shell never reaches
        # its ``printf $status > .update_exit_code`` epilogue, so the
        # exit-code marker file would never be created.  The new gateway's
        # update watcher would then poll for 30 minutes and send a spurious
        # timeout message.
        #
        # Writing the marker here — after git pull + pip install succeed but
        # before we attempt the restart — ensures the new gateway sees it
        # regardless of how we die. Gated on desktop_build_ok (#88251): a
        # Desktop rebuild failure must not be reported as "0" — the gateway's
        # /update watcher (gateway/run.py) polls this file.
        if gateway_mode:
            _write_gateway_update_exit_code(desktop_build_ok)

        gateway_fleet_restart_incomplete = False
        # Snapshot of gateways running before we touch anything. Stays empty
        # until we successfully import the probe and are about to stop/drain —
        # so an exception raised before we touch any gateway keeps this empty
        # (nothing to fail closed on), while a failure after we have stopped a
        # discovered gateway lets the handler fail closed on an empty survivor
        # probe rather than reporting a clean update (#78574).
        _pre_restart_gateway_pids: list | None = []
        # Declared outside the restart try/except below (and never reset
        # to None) so it's always safe to read afterwards even if that
        # block raises before reaching its own restart bookkeeping —
        # needed to forward already-restarted units to
        # ``_finish_dashboard_update_cleanup`` (review on #83595).
        restarted_services: list = []
        # Same outside-the-try treatment: the post-restart fleet version
        # check consults killed_pids to decide whether to wait for
        # freshly-restarted gateways to settle, and the phase's except
        # path forwards it to the update receipt.
        killed_pids: set = set()

        # Auto-restart ALL gateways after update.
        # The code update (git pull) is shared across all profiles, so every
        # running gateway needs restarting to pick up the new code.
        #
        # Purge stale cached Hermes modules FIRST: the import below pulls
        # freshly-updated gateway source into this pre-update interpreter,
        # and any already-cached sibling module (cli_output, status, ...)
        # that the new source expects a new symbol from would otherwise
        # ImportError and abort this whole phase (2026-08-20 field failure:
        # new gateway.py ← stale cli_output missing line_input).
        _m()._purge_stale_hermes_modules()
        try:
            from hermes_cli.gateway import (
                is_macos,
                supports_systemd_services,
                _ensure_user_systemd_env,
                find_gateway_pids,
                find_profile_gateway_processes,
                _prepare_profile_gateway_update_restart,
                _get_service_pids,
                _graceful_restart_via_sigusr1,
                _wait_for_gateway_exit,
            )
            import signal as _signal

            def _wait_for_service_active(
                scope_cmd_: list,
                svc_name_: str,
                timeout: float = 10.0,
            ) -> bool:
                """Poll ``systemctl is-active`` until the unit reports active.

                systemd's Stopped -> Started transition after a graceful exit
                (or a hard restart) is not instantaneous; a one-shot check
                races that window and falsely reports the unit as down.
                Poll every 0.5s up to ``timeout`` seconds before giving up.
                """
                deadline = _time.monotonic() + max(timeout, 0.5)
                while True:
                    try:
                        _verify = subprocess.run(
                            scope_cmd_ + ["is-active", svc_name_],
                            capture_output=True,
                            text=True, encoding="utf-8", errors="replace",
                            timeout=5,
                        )
                        if _verify.stdout.strip() == "active":
                            return True
                    except (FileNotFoundError, subprocess.TimeoutExpired):
                        pass
                    if _time.monotonic() >= deadline:
                        return False
                    _time.sleep(0.5)

            def _service_restart_sec(
                scope_cmd_: list,
                svc_name_: str,
                default: float = 0.0,
            ) -> float:
                """Read the unit's ``RestartUSec`` (RestartSec) in seconds.

                After a graceful exit-75, systemd waits ``RestartSec`` before
                respawning the unit.  Callers that poll for ``is-active``
                must use a timeout >= ``RestartSec`` + transition slack, or
                they'll give up *during* the cooldown window and wrongly
                conclude the unit didn't relaunch.
                """
                try:
                    _show = subprocess.run(
                        scope_cmd_
                        + [
                            "show",
                            svc_name_,
                            "--property=RestartUSec",
                            "--value",
                        ],
                        capture_output=True,
                        text=True, encoding="utf-8", errors="replace",
                        timeout=5,
                    )
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    return default
                raw = (_show.stdout or "").strip()
                # systemd emits values like "30s", "100ms", "1min 30s", or
                # "infinity".  Parse conservatively; on any miss return default.
                if not raw or raw == "infinity":
                    return default
                total = 0.0
                matched = False
                for part in raw.split():
                    for _suf, _mult in (
                        ("ms", 0.001),
                        ("us", 0.000001),
                        ("min", 60.0),
                        ("s", 1.0),
                    ):
                        if part.endswith(_suf):
                            try:
                                total += float(part[: -len(_suf)]) * _mult
                                matched = True
                            except ValueError:
                                pass
                            break
                return total if matched else default

            _manage_cmd_cache: dict = {}

            def _resolve_manage_cmd(scope_: str, scope_cmd_: list, svc_name_: str):
                """Resolve the command prefix for manage-units operations.

                Read-only systemctl calls (``is-active``, ``show``,
                ``list-units``) work unprivileged, but manage-units verbs
                (``reset-failed``, ``start``, ``restart``) on a *system*
                service trigger a polkit ``org.freedesktop.systemd1.manage-units``
                authentication prompt when run as a non-root user.  That
                interactive prompt runs inside our captured subprocess with a
                10-15s timeout — the user sees the prompt flash and "exit
                directly" before they can answer, and the resulting
                TimeoutExpired used to be swallowed silently.

                Strategy: if root, plain systemctl.  If not root, try
                non-interactive sudo (``sudo -n``) — first a blanket probe,
                then a targeted ``systemctl reset-failed`` probe so a
                least-privilege sudoers entry scoped to
                ``systemctl ... hermes-gateway*`` also qualifies
                (``reset-failed`` is an idempotent no-op we run before every
                privileged restart anyway).  If neither works, return None —
                the caller must SKIP the restart (without draining the
                gateway first!) and tell the user how to restart manually.
                ``--no-ask-password`` guarantees polkit can never hang a
                captured subprocess on this path.
                """
                if scope_ in _manage_cmd_cache:
                    return _manage_cmd_cache[scope_]
                cmd = scope_cmd_ + ["--no-ask-password"]
                if (
                    scope_ == "system"
                    and hasattr(os, "geteuid")
                    and os.geteuid() != 0  # windows-footgun: ok — systemd path, Linux-only
                ):
                    sudo_cmd = ["sudo", "-n"] + scope_cmd_ + ["--no-ask-password"]
                    sudo_ok = False
                    try:
                        _probe = subprocess.run(
                            ["sudo", "-n", "true"],
                            capture_output=True,
                            timeout=5,
                        )
                        sudo_ok = _probe.returncode == 0
                        if not sudo_ok:
                            # Blanket sudo refused — a targeted sudoers entry
                            # (NOPASSWD for systemctl ... hermes-gateway*)
                            # may still allow the exact commands we need.
                            _probe = subprocess.run(
                                sudo_cmd + ["reset-failed", svc_name_],
                                capture_output=True,
                                timeout=5,
                            )
                            sudo_ok = _probe.returncode == 0
                    except (FileNotFoundError, subprocess.TimeoutExpired):
                        sudo_ok = False
                    cmd = sudo_cmd if sudo_ok else None
                _manage_cmd_cache[scope_] = cmd
                return cmd

            # Wait budget for graceful SIGUSR1 restarts.  In-band restart
            # may defer stop() until active turns finish
            # (``restart_after_turn_timeout``, #77184) and then spend up to
            # ``restart_drain_timeout`` inside stop(). Cover both phases so
            # we don't fall back to a hard kill while the gateway is still
            # patiently waiting for the requesting turn. On older systemd
            # units without SIGUSR1 wiring this wait just times out and we
            # fall back to ``systemctl restart`` (the old behaviour).
            try:
                from hermes_cli.gateway import _get_restart_exit_wait_budget

                _drain_budget = max(float(_get_restart_exit_wait_budget()), 45.0)
            except Exception:
                _drain_budget = 45.0

            failed_or_stale_units = []
            killed_pids = set()
            relaunched_profiles = []
            externally_supervised_profiles = []

            # Record which gateways are running before any stop/drain, so a
            # later failure that leaves the survivor probe empty can still be
            # recognised as "a running gateway was stopped and did not come
            # back" rather than "nothing was running" (#78574). Best-effort:
            # if the probe itself raises, leave the snapshot as-is (the
            # survivor probe's own None result already fails closed).
            try:
                _pre_restart_gateway_pids = list(find_gateway_pids(all_profiles=True))
            except Exception:
                _pre_restart_gateway_pids = None

            # --- Systemd services (Linux) ---
            # Discover all hermes-gateway* units (default + profiles) plus
            # hermes-serve* units (the Desktop app's backend, #83438).
            if supports_systemd_services():
                try:
                    _ensure_user_systemd_env()
                except Exception:
                    pass

                for scope, scope_cmd in [
                    ("user", ["systemctl", "--user"]),
                    ("system", ["systemctl"]),
                ]:
                    try:
                        result = subprocess.run(
                            scope_cmd
                            + [
                                "list-units",
                                "hermes-gateway*",
                                "hermes-serve*",
                                "--plain",
                                "--no-legend",
                                "--no-pager",
                            ],
                            capture_output=True,
                            text=True, encoding="utf-8", errors="replace",
                            timeout=10,
                        )
                    except FileNotFoundError:
                        continue
                    except subprocess.TimeoutExpired as exc:
                        # Discovery timeout — skip this scope, keep the other.
                        print(
                            f"  ⚠ systemctl timed out listing {scope}-scope "
                            f"gateway units ({exc.cmd if exc.cmd else 'unknown command'}). "
                            f"Check the gateway with: hermes gateway status"
                        )
                        continue

                    def _restart_one_systemd_gateway_unit(svc_name: str) -> None:
                        # Check if active
                        check = subprocess.run(
                            scope_cmd + ["is-active", svc_name],
                            capture_output=True,
                            text=True, encoding="utf-8", errors="replace",
                            timeout=5,
                        )
                        if check.stdout.strip() != "active":
                            return

                        # Resolve how we may run manage-units verbs
                        # (reset-failed/start/restart) for this scope.
                        # None ⇒ no non-interactive privilege path; we
                        # must avoid those verbs entirely or polkit will
                        # throw an interactive auth prompt inside our
                        # captured 10-15s subprocess (the user sees it
                        # flash and "exit directly" — reported June 2026).
                        _manage_cmd = _resolve_manage_cmd(
                            scope, scope_cmd, svc_name
                        )

                        # Prefer a graceful SIGUSR1 restart so in-flight
                        # agent runs drain instead of being SIGKILLed.
                        # The gateway's SIGUSR1 handler calls
                        # request_restart(via_service=True) → drain →
                        # exit; systemd's Restart=always respawns the unit.
                        # hermes-serve has no such handler (it isn't
                        # gateway/run.py), so skip straight to the blunt
                        # restart below rather than sending it an unhandled
                        # signal and waiting out the drain budget for
                        # nothing.
                        _main_pid = 0
                        if _service_unit_supports_graceful_sigusr1_restart(svc_name):
                            try:
                                _show = subprocess.run(
                                    scope_cmd
                                    + [
                                        "show",
                                        svc_name,
                                        "--property=MainPID",
                                        "--value",
                                    ],
                                    capture_output=True,
                                    text=True, encoding="utf-8", errors="replace",
                                    timeout=5,
                                )
                                _main_pid = int((_show.stdout or "").strip() or 0)
                            except (
                                ValueError,
                                subprocess.TimeoutExpired,
                                FileNotFoundError,
                            ):
                                _main_pid = 0

                        _graceful_ok = False
                        if _main_pid > 0:
                            from hermes_cli.gateway import (
                                GATEWAY_LOOP_WEDGED,
                                _escalate_wedged_gateway,
                                probe_gateway_loop_liveness,
                            )

                            if (
                                probe_gateway_loop_liveness(_main_pid)
                                == GATEWAY_LOOP_WEDGED
                            ):
                                # Loop-liveness probe says the gateway's event
                                # loop is provably dead (#81642): SIGUSR1 can
                                # never drain it, so waiting the full budget
                                # (180s default) only wedges the update too.
                                # Bounded escalation (SIGTERM grace → SIGKILL,
                                # ~10s) then restart the unit. A busy gateway
                                # keeps a fresh heartbeat and never takes this
                                # path — its drain (incl. the #86684 cron
                                # floor) is untouched.
                                print(
                                    f"  ⚠ {svc_name}: gateway event loop is "
                                    "unresponsive — skipping drain, forcing "
                                    "a bounded stop..."
                                )
                                _escalate_wedged_gateway(_main_pid)
                                _graceful_ok = True
                            else:
                                print(
                                    f"  → {svc_name}: draining (up to {int(_drain_budget)}s)..."
                                )
                                _graceful_ok = _graceful_restart_via_sigusr1(
                                    _main_pid,
                                    drain_timeout=_drain_budget,
                                )

                        if _graceful_ok:
                            # Gateway exited after a planned restart.
                            # ``Restart=always`` means systemd WILL respawn
                            # the unit — but only after
                            # ``RestartSec`` (default 60s on our unit
                            # file). That 60s wait is a crash-loop guard,
                            # and is the right default when the gateway
                            # dies unexpectedly. For a voluntary restart
                            # on update, it's dead time the user watches.
                            #
                            # Shortcut it: ``reset-failed`` + ``start``
                            # skips RestartSec entirely (we're manually
                            # initiating the unit, not waiting for
                            # systemd's auto-restart logic). Takes about
                            # as long as the process takes to come up
                            # (~1-3s on a warm box).
                            #
                            # If the unit is already active because
                            # RestartSec elapsed while we were draining,
                            # ``start`` is a no-op and we fall through to
                            # the poll below. Either way we collapse the
                            # 60s+ delay to a ~5s one.
                            #
                            # The shortcut needs manage-units privileges.
                            # Without them (system service, non-root, no
                            # passwordless sudo) skip it — systemd's own
                            # auto-restart still relaunches the unit after
                            # RestartSec, no privileges required.
                            if _manage_cmd is not None:
                                subprocess.run(
                                    _manage_cmd + ["reset-failed", svc_name],
                                    capture_output=True,
                                    text=True, encoding="utf-8", errors="replace",
                                    timeout=10,
                                )
                                subprocess.run(
                                    _manage_cmd + ["start", svc_name],
                                    capture_output=True,
                                    text=True, encoding="utf-8", errors="replace",
                                    timeout=15,
                                )
                                # Short poll: the gateway should be up
                                # within a few seconds now that we
                                # bypassed RestartSec.
                                if _wait_for_service_active(
                                    scope_cmd,
                                    svc_name,
                                    timeout=10.0,
                                ):
                                    restarted_services.append(svc_name)
                                    return
                            # Passive poll: systemd's auto-restart fires
                            # after RestartSec regardless of privileges.
                            # This is the primary path when _manage_cmd is
                            # None, and the fallback when the explicit
                            # start didn't take.
                            _restart_sec = _service_restart_sec(
                                scope_cmd,
                                svc_name,
                                default=0.0,
                            )
                            _post_drain_timeout = max(
                                10.0,
                                _restart_sec + 10.0,
                            )
                            if _manage_cmd is None and _restart_sec > 5.0:
                                print(
                                    f"  → {svc_name}: waiting for systemd "
                                    f"auto-restart (~{int(_restart_sec)}s; "
                                    "no root for an immediate restart)..."
                                )
                            if _wait_for_service_active(
                                scope_cmd,
                                svc_name,
                                timeout=_post_drain_timeout,
                            ):
                                restarted_services.append(svc_name)
                                return
                            # Process exited but wasn't respawned (older
                            # unit without Restart=on-failure or
                            # RestartForceExitStatus=75).  Fall through
                            # to systemctl start/restart.
                            print(
                                f"  ⚠ {svc_name} drained but didn't relaunch — forcing restart"
                            )

                        # Forcing a restart requires manage-units
                        # privileges.  Without a non-interactive path,
                        # running systemctl here would spawn a polkit
                        # auth prompt inside a captured 10-15s subprocess
                        # — it flashes and dies before the user can
                        # answer.  Skip with clear instructions instead.
                        if _manage_cmd is None:
                            failed_or_stale_units.append(svc_name)
                            print(
                                f"  ⚠ {svc_name} is a system service and restarting it needs root.\n"
                                f"    Restart it manually to load the new version:\n"
                                f"      sudo systemctl restart {svc_name}\n"
                                f"    To let `hermes update` restart it automatically, allow\n"
                                f"    passwordless sudo for systemctl, or run updates with sudo."
                            )
                            return

                        # Fallback: blunt systemctl restart.  This is
                        # what the old code always did; we get here only
                        # when the graceful path failed (unit missing
                        # SIGUSR1 wiring, drain exceeded the budget,
                        # restart-policy mismatch).
                        #
                        # Always `reset-failed` first.  If systemd's own
                        # auto-restart attempts already parked the unit
                        # in a failed state (transient CHDIR / OOM /
                        # filesystem race after our drain + exit-75),
                        # a plain `systemctl restart` can wedge against
                        # the RestartSec backoff and leave the unit
                        # dead.  Clearing the failed state first makes
                        # the restart idempotent.  Mirrors the recovery
                        # path in `hermes gateway restart`
                        # (`systemd_restart()`) as of PR #20949.
                        subprocess.run(
                            _manage_cmd + ["reset-failed", svc_name],
                            capture_output=True,
                            text=True, encoding="utf-8", errors="replace",
                            timeout=10,
                        )
                        restart = subprocess.run(
                            _manage_cmd + ["restart", svc_name],
                            capture_output=True,
                            text=True, encoding="utf-8", errors="replace",
                            timeout=15,
                        )
                        if restart.returncode == 0:
                            # Verify the service actually survived the
                            # restart.  systemctl restart returns 0 even
                            # if the new process crashes immediately.
                            if _wait_for_service_active(
                                scope_cmd,
                                svc_name,
                                timeout=10.0,
                            ):
                                restarted_services.append(svc_name)
                            else:
                                # Retry once — transient startup failures
                                # (stale module cache, import race) often
                                # resolve on the second attempt.  Again
                                # clear any failed state first so the
                                # retry isn't blocked by the previous
                                # crash.
                                print(
                                    f"  ⚠ {svc_name} died after restart, retrying..."
                                )
                                subprocess.run(
                                    _manage_cmd + ["reset-failed", svc_name],
                                    capture_output=True,
                                    text=True, encoding="utf-8", errors="replace",
                                    timeout=10,
                                )
                                subprocess.run(
                                    _manage_cmd + ["restart", svc_name],
                                    capture_output=True,
                                    text=True, encoding="utf-8", errors="replace",
                                    timeout=15,
                                )
                                if _wait_for_service_active(
                                    scope_cmd,
                                    svc_name,
                                    timeout=10.0,
                                ):
                                    restarted_services.append(svc_name)
                                    print(f"  ✓ {svc_name} recovered on retry")
                                else:
                                    failed_or_stale_units.append(svc_name)
                                    _scope_flag = "--user " if scope == "user" else ""
                                    _sudo_hint = "sudo " if scope == "system" else ""
                                    print(
                                        f"  ✗ {svc_name} failed to stay running after restart.\n"
                                        f"    Check logs: {_sudo_hint}journalctl {_scope_flag}-u {svc_name} --since '2 min ago'\n"
                                        f"    Recover manually:\n"
                                        f"      {_sudo_hint}systemctl {_scope_flag}reset-failed {svc_name}\n"
                                        f"      {_sudo_hint}systemctl {_scope_flag}restart {svc_name}"
                                    )
                        else:
                            failed_or_stale_units.append(svc_name)
                            print(
                                f"  ⚠ Failed to restart {svc_name}: {restart.stderr.strip()}"
                            )

                    def _on_unit_timeout(svc_name: str, exc: subprocess.TimeoutExpired) -> None:
                        # Isolate the timeout to this unit and keep going
                        # (#68523). A scope-wide handler used to abort every
                        # later gateway and leave the fleet on mixed code.
                        failed_or_stale_units.append(svc_name)
                        print(
                            f"  ⚠ systemctl timed out restarting {svc_name} "
                            f"({exc.cmd if exc.cmd else 'unknown command'}); "
                            f"continuing with remaining gateways"
                        )

                    _for_each_systemd_gateway_unit(
                        result.stdout,
                        process_unit=_restart_one_systemd_gateway_unit,
                        on_unit_timeout=_on_unit_timeout,
                    )

            # --- Launchd services (macOS) ---
            # Restart EVERY ai.hermes.gateway* LaunchAgent, not only the
            # invoking profile's — parity with the systemd branch above
            # (#41403). Per-label TimeoutExpired isolation happens inside.
            if is_macos():
                try:
                    _restart_macos_launchd_gateways(
                        restarted_services,
                        failed_or_stale_units,
                        _drain_budget,
                    )
                except (FileNotFoundError, ImportError):
                    pass

            # --- Manual (non-service) gateways ---
            # Kill any remaining gateway processes not managed by a service.
            # Exclude PIDs that belong to just-restarted services so we don't
            # immediately kill the process that systemd/launchd just spawned.
            service_pids = _get_service_pids(all_profiles=True)
            manual_pids = find_gateway_pids(
                exclude_pids=service_pids, all_profiles=True
            )
            profile_processes = {
                proc.pid: proc
                for proc in find_profile_gateway_processes(exclude_pids=service_pids)
                if proc.pid in manual_pids
            }
            # Profile gateways we could not arm a relaunch for.  These must
            # NOT be left running: their modules are the pre-update ones and
            # every lazy import from here on mixes versions against the new
            # code on disk (#88654).  Handing them to the unmapped sweep
            # below stops them and surfaces them in the "Stopped N manual
            # gateway process(es) / Restart manually" summary, which is the
            # contract already used for gateways with no profile mapping.
            unrestartable_pids = set()
            for pid, proc in profile_processes.items():
                restart_mode = _prepare_profile_gateway_update_restart(
                    proc.profile, pid
                )
                if restart_mode is None:
                    # Previously a bare ``continue``: the gateway was neither
                    # relaunched nor stopped nor mentioned, so it kept serving
                    # from stale modules with no operator signal at all.
                    print(
                        f"  ⚠ {proc.profile}: could not arm an automatic "
                        f"gateway restart for PID {pid} — stopping it instead "
                        "so it cannot keep running pre-update code"
                    )
                    unrestartable_pids.add(pid)
                    continue
                # Prefer a graceful SIGUSR1 drain so in-flight agent runs
                # finish before the watcher respawns the gateway.  If the
                # gateway doesn't support SIGUSR1 or doesn't exit within
                # the drain budget, fall back to SIGTERM — the watcher
                # still sees the exit and relaunches either way.
                # Announce the drain first: this wait can hold for the full
                # budget per gateway with no other output, and on surfaces
                # that stream update progress (the desktop updater most of
                # all) the silence reads as a hung update (#44515).
                print(
                    f"  → {proc.profile}: draining gateway PID {pid} "
                    f"(up to {int(_drain_budget)}s)..."
                )
                from hermes_cli.gateway import (
                    GATEWAY_LOOP_WEDGED,
                    _escalate_wedged_gateway,
                    probe_gateway_loop_liveness,
                )

                if probe_gateway_loop_liveness(pid) == GATEWAY_LOOP_WEDGED:
                    # Loop-liveness probe: this gateway's event loop is
                    # provably dead (#81642) — SIGUSR1/SIGTERM shutdown can
                    # never run, so the drain wait would burn the full budget
                    # and stall the update. Bounded stop instead (SIGTERM
                    # grace → SIGKILL, ~10s). A busy-but-alive gateway keeps
                    # a fresh heartbeat and never takes this branch, so live
                    # drains (incl. the #86684 cron floor) are unaffected.
                    print(
                        f"  ⚠ {proc.profile}: gateway event loop is "
                        "unresponsive — skipping drain, forcing a bounded stop..."
                    )
                    _escalate_wedged_gateway(pid)
                    drained = True
                else:
                    drained = _graceful_restart_via_sigusr1(
                        pid,
                        drain_timeout=_drain_budget,
                    )
                if not drained:
                    try:
                        os.kill(pid, _signal.SIGTERM)
                    except (ProcessLookupError, PermissionError):
                        pass
                # Wait for the old process to fully exit before the watcher
                # spawns the new gateway.  Telegram holds the previous
                # getUpdates long-poll session open on its servers for up to
                # ~30s after the client disconnects.  If the new gateway
                # connects before that window expires it receives a 409
                # Conflict, which _handle_polling_conflict() recovers from
                # via back-off retries — but a brief wait here reduces the
                # chance of hitting that path at all, especially on fast
                # machines where the watcher loop restarts in < 1s.
                # We wait up to 5s for the process to exit (the OS-level
                # close, not the Telegram server-side expiry), then let the
                # watcher take over.  The Telegram adapter's retry logic
                # handles any remaining 409s if the server session is still
                # live when the new gateway polls.
                _wait_for_gateway_exit(timeout=5.0, force_after=None)
                killed_pids.add(pid)
                if restart_mode == "external-supervisor":
                    externally_supervised_profiles.append(proc.profile)
                else:
                    relaunched_profiles.append(proc.profile)

            for pid in manual_pids:
                if pid in profile_processes and pid not in unrestartable_pids:
                    continue
                try:
                    os.kill(pid, _signal.SIGTERM)
                    killed_pids.add(pid)
                except (ProcessLookupError, PermissionError):
                    pass

            if restarted_services or killed_pids:
                print()
                for svc in restarted_services:
                    print(f"  ✓ Restarted {svc}")
                if relaunched_profiles:
                    names = ", ".join(relaunched_profiles)
                    print(f"  ✓ Restarting manual gateway profile(s): {names}")
                if externally_supervised_profiles:
                    names = ", ".join(externally_supervised_profiles)
                    print(
                        "  ✓ Handed gateway profile(s) back to their external "
                        f"supervisor: {names}"
                    )
                unmapped_count = (
                    len(killed_pids)
                    - len(relaunched_profiles)
                    - len(externally_supervised_profiles)
                )
                if unmapped_count:
                    print(f"  → Stopped {unmapped_count} manual gateway process(es)")
                    print("    Restart manually: hermes gateway run")
                    if unmapped_count > 1:
                        print(
                            "    (or: hermes -p <profile> gateway run  for each profile)"
                        )

            if failed_or_stale_units:
                gateway_fleet_restart_incomplete = True
                if gateway_mode:
                    _exit_code_path = get_hermes_home() / ".update_exit_code"
                    try:
                        _exit_code_path.write_text("1", encoding="utf-8")
                    except OSError:
                        pass
            _warn_incomplete_gateway_fleet_restart(failed_or_stale_units)

            try:
                from hermes_cli.update_receipt import record_gateway_restart

                record_gateway_restart(
                    restarted_services=restarted_services,
                    relaunched_profiles=relaunched_profiles,
                    externally_supervised_profiles=externally_supervised_profiles,
                    killed_pids=sorted(killed_pids),
                    failed_units=failed_or_stale_units,
                    incomplete=bool(failed_or_stale_units),
                )
            except Exception:
                pass

            if not restarted_services and not killed_pids:
                # No gateways were running — nothing to do
                pass

            # --- Post-restart survivor sweep -----------------------------
            # Issue #17648: some gateways ignore SIGTERM (stuck drain,
            # blocked I/O, PID dead but zombie).  The detached profile
            # watchers wait 120s for the old PID to exit — if it never
            # does, no respawn happens and the user keeps hitting
            # ImportError against a stale sys.modules.  Give the
            # graceful paths a brief window to complete, then SIGKILL
            # any remaining pre-update PIDs so the watcher / service
            # manager can relaunch with fresh code.
            try:
                _time.sleep(3.0)
                _service_pids_after = _get_service_pids(all_profiles=True)
                _surviving = find_gateway_pids(
                    exclude_pids=_service_pids_after,
                    all_profiles=True,
                )
                # Scope to PIDs we already tried to kill during this
                # update (killed_pids).  Anything new is a gateway that
                # started AFTER our restart attempt — respecting user
                # intent, we don't kill those.
                _stuck = [pid for pid in _surviving if pid in killed_pids]
                if _stuck:
                    print()
                    print(
                        f"  ⚠ {len(_stuck)} gateway process(es) ignored SIGTERM — force-killing"
                    )
                    from gateway.status import terminate_pid as _terminate_pid
                    for pid in _stuck:
                        try:
                            # Routes through taskkill /T /F on Windows,
                            # SIGKILL on POSIX — _signal.SIGKILL doesn't
                            # exist on Windows so the old raw os.kill call
                            # used to crash the entire update path.
                            _terminate_pid(pid, force=True)
                        except (ProcessLookupError, PermissionError, OSError):
                            pass
                    # Give the OS a beat to reap the processes so the
                    # watchers see them exit and respawn.
                    _time.sleep(1.5)
            except Exception as _sweep_exc:
                logger.debug("Post-restart survivor sweep failed: %s", _sweep_exc)

        except Exception as e:
            logger.debug("Gateway restart during update failed: %s", e)
            # An exception escaping the whole phase means the drain/restart
            # output the user relies on never printed. Don't let that pass for
            # a clean update: surface it and treat the fleet as stale unless we
            # can positively prove no gateway is running (#78574).
            #
            # A positive-empty ``_surviving`` is only proof-of-safety when
            # nothing was running before we touched anything. If a gateway was
            # discovered pre-restart and none survive now, it was stopped and
            # its replacement was never verified — the same fail-open contract
            # this fix closes — so we must still fail closed on ``[]``.
            _surviving = _surviving_gateway_pids_after_failed_restart()
            if _restart_phase_failure_is_incomplete(
                _surviving, _pre_restart_gateway_pids
            ):
                gateway_fleet_restart_incomplete = True
                _warn_gateway_restart_phase_aborted(e, _surviving)
                if gateway_mode:
                    _exit_code_path = get_hermes_home() / ".update_exit_code"
                    try:
                        _exit_code_path.write_text("1", encoding="utf-8")
                    except OSError:
                        pass
            try:
                from hermes_cli.update_receipt import record_gateway_restart

                record_gateway_restart(
                    restarted_services=restarted_services,
                    incomplete=gateway_fleet_restart_incomplete,
                    phase_error=str(e),
                )
            except Exception:
                pass

        _m()._resume_windows_gateways_after_update(_windows_gateway_resume)

        # Warn if legacy Hermes gateway unit files are still installed.
        # When both hermes.service (from a pre-rename install) and the
        # current hermes-gateway.service are enabled, they SIGTERM-fight
        # for the same bot token (see PR #11909). Flagging here means
        # every `hermes update` surfaces the issue until the user migrates.
        try:
            from hermes_cli.gateway import (
                has_legacy_hermes_units,
                _find_legacy_hermes_units,
                supports_systemd_services,
            )

            if supports_systemd_services() and has_legacy_hermes_units():
                print()
                print("⚠ Legacy Hermes gateway unit(s) detected:")
                for name, path, is_sys in _find_legacy_hermes_units():
                    scope = "system" if is_sys else "user"
                    print(f"    {path}  ({scope} scope)")
                print()
                print("  These pre-rename units (hermes.service) fight the current")
                print("  hermes-gateway.service for the bot token and cause SIGTERM")
                print("  flap loops. Remove them with:")
                print()
                print("    hermes gateway migrate-legacy")
                print()
                print("  (add `sudo` if any are in system scope)")
        except Exception as e:
            logger.debug("Legacy unit check during update failed: %s", e)

        # Restart a managed dashboard through systemd, or stop stale manual
        # dashboard processes. Raw-killing a systemd-owned dashboard PID makes
        # systemd treat it as a clean stop, leaving the Cloudflare origin dead.
        # Preserve the safety rule above: a failed Node refresh leaves the
        # currently running dashboard untouched.
        #
        # Forward the systemd units restarted above (includes hermes-serve*,
        # #83438) so a Serve-only install's freshly restarted process isn't
        # found and restarted again below (review on #83595).
        _finish_dashboard_update_cleanup(
            node_failures, already_restarted_units=set(restarted_services)
        )

        print()
        print("Tip: You can now select a provider and model:")
        print("  hermes model              # Select provider and model")

        # Phase 1 (#91277): post-update fleet version verification. Compare
        # every live gateway's stamped code_sha against the freshly-updated
        # checkout and surface any gateway still serving pre-update code —
        # instead of assuming the restart phase worked (#88654, #69754).
        _fleet_snapshot: list = []
        try:
            from hermes_cli.update_receipt import (
                collect_fleet_versions,
                print_fleet_version_matrix,
            )

            # Cross-platform "we expected fleet rows" signal (#93406). The
            # old (restarted_services or killed_pids) condition never fires
            # on Windows: the pause/resume phase populates neither list, so
            # a healthy resumed gateway yielded zero rows and exit 0.
            _fleet_rows_expected = _m()._fleet_probe_expected_runtimes(
                _pre_update_plan,
                _pre_restart_gateway_pids,
                _windows_gateway_resume,
                restarted_services,
                killed_pids,
            )
            # A brief settle window: freshly restarted/resumed gateways need
            # a moment to rewrite gateway_state.json with their new identity.
            # Skipped when the restart phase touched nothing (no gateways
            # were running) — nothing to settle.
            if _fleet_rows_expected:
                _time.sleep(2.0)
            # Pass the pre-restart PID snapshot so a gateway the restart
            # phase stopped WITHOUT a verified replacement shows as a DOWN
            # row (exit 1) instead of silently producing no row at all.
            _fleet_snapshot = collect_fleet_versions(
                pre_restart_pids=_pre_restart_gateway_pids
            )
            if print_fleet_version_matrix(_fleet_snapshot):
                gateway_fleet_restart_incomplete = True
            elif not _fleet_snapshot and _fleet_rows_expected:
                # Fleet probe returned zero rows even though at least one
                # gateway runtime was (or may have been) live pre-update —
                # POSIX restart bookkeeping, the pre-restart PID snapshot,
                # the pre-update plan inventory, or the Windows pause/resume
                # token all count as that signal.  Every failure path inside
                # collect_fleet_versions() is swallowed via logger.debug(),
                # so an empty list is indistinguishable from a healthy fleet
                # in the current output.  Treat it as verification failure
                # so the receipt records "partial" and the exit code is 1
                # (#93406).
                print(
                    "\n⚠ Fleet version check returned no rows even though"
                    " gateway runtimes were expected — verification incomplete."
                )
                gateway_fleet_restart_incomplete = True
        except Exception as _fleet_exc:
            logger.debug("Fleet version verification failed: %s", _fleet_exc)

        # Plan-vs-execution reconciliation (#91277 Phase 2, restart via
        # declared mechanism): every runtime the PLAN saw must be accounted
        # for by the restart phase's bookkeeping. An unaccounted runtime is
        # the silent-miss class (a platform branch re-discovered its own
        # targets and skipped one the inventory knew about) — escalate it
        # exactly like a STALE/DOWN fleet row.
        _runtime_outcomes: list = []
        try:
            if _pre_update_plan is not None and _pre_update_plan.runtimes:
                from hermes_cli.update_inventory import (
                    match_runtime_outcomes,
                    report_unaccounted_runtimes,
                )

                _runtime_outcomes = match_runtime_outcomes(
                    _pre_update_plan,
                    restarted_services=restarted_services,
                    relaunched_profiles=relaunched_profiles,
                    externally_supervised_profiles=externally_supervised_profiles,
                    killed_pids=killed_pids,
                    failed_units=failed_or_stale_units,
                )
                if report_unaccounted_runtimes(_runtime_outcomes):
                    gateway_fleet_restart_incomplete = True
                try:
                    import hermes_cli.update_receipt as _ur

                    if _ur._current is not None:
                        _ur._current.data["runtime_outcomes"] = _runtime_outcomes
                except Exception:
                    pass
        except Exception as _outcome_exc:
            logger.debug("Runtime-outcome reconciliation failed: %s", _outcome_exc)

        try:
            from hermes_cli.update_receipt import finalize_update_receipt

            _receipt_path = finalize_update_receipt(
                "partial" if gateway_fleet_restart_incomplete else "success",
                fleet=_fleet_snapshot,
            )
            if _receipt_path is not None:
                logger.info("Update receipt written: %s", _receipt_path)
        except Exception as _receipt_exc:
            logger.debug("Update receipt finalize failed: %s", _receipt_exc)

        if gateway_fleet_restart_incomplete:
            # Code update itself succeeded, but at least one gateway still
            # runs pre-update modules — surface that as a failed update so
            # automation / operators do not treat the fleet as healthy.
            sys.exit(1)

    except _shim_quarantine_error_type() as e:
        # Fail-closed shim contention (#87331): strict quarantine refused
        # BEFORE any installer ran — defer via marker, exit 2, no ZIP.
        _refuse_update_for_contended_shims(e)
    except subprocess.CalledProcessError as e:
        stage = _format_update_failure_stage(e)
        if _should_zip_fallback_on_update_error(e):
            print(f"⚠ {stage}: {e}")
            print("→ Falling back to ZIP download...")
            print()
            desktop_build_ok = _update_via_zip(
                args,
                had_desktop_app_before_update=had_desktop_app_before_update,
            )
            if gateway_mode:
                _write_gateway_update_exit_code(desktop_build_ok)
        else:
            print(f"✗ {stage}: {e}")
            _print_called_process_error_tail(e)
            if _called_process_error_is_python_dep_install(e):
                print(
                    "  The git update already finished. Re-downloading the source "
                    "ZIP cannot fix a dependency install error and would overwrite "
                    "local files."
                )
                if _m()._is_windows():
                    print("  Retry through the venv interpreter:")
                    print(
                        '    venv\\Scripts\\python.exe -c '
                        '"from hermes_cli.main import main; main()" update --yes'
                    )
            try:
                from hermes_cli.update_receipt import finalize_update_receipt

                finalize_update_receipt("failed")
            except Exception:
                pass
            sys.exit(1)

# --- Hoisted from the body of _cmd_update_impl (self-contained, no closure state) ---

def _restart_phase_failure_is_incomplete(surviving, pre_restart_pids) -> bool:
    """Whether an escaped gateway-restart-phase exception must fail the update.

    Fail closed unless we can positively prove the fleet is safe:

    * ``surviving is None`` — the survivor probe could not determine state
      (typically the freshly-pulled ``hermes_cli.gateway`` no longer imports,
      one of the ways the phase aborts). Assume stale.
    * ``surviving`` non-empty — a gateway is still running pre-update code.
    * ``surviving == []`` — nothing is running now. That is proof-of-safety
      ONLY when nothing was running before we touched anything. If a gateway
      was discovered pre-restart (``pre_restart_pids`` non-empty, or ``None``
      meaning the pre-state could not be read), it was stopped without a
      verified replacement, so we still fail closed (#78574).
    """
    if surviving is None or surviving:
        return True
    # surviving == []: safe only if we know nothing was running beforehand.
    return pre_restart_pids is None or bool(pre_restart_pids)


def _fleet_probe_expected_runtimes(
    pre_update_plan,
    pre_restart_pids,
    windows_resume_token,
    restarted_services,
    killed_pids,
) -> bool:
    """Whether the post-update fleet probe should have produced rows.

    The zero-rows fail-open (#93406): ``collect_fleet_versions()`` swallows
    every probe failure via ``logger.debug()`` and ``print_fleet_version_matrix([])``
    early-returns ``False``, so an empty snapshot reads as \"healthy fleet\" and
    the update exits 0.  An empty snapshot is only proof-of-safety when NOTHING
    says a gateway existed before the update.  Any of these signals means at
    least one runtime was (or may have been) live pre-update, so zero rows is
    verification failure, not health:

    * ``restarted_services`` / ``killed_pids`` — the POSIX restart phase
      touched live gateways.
    * ``pre_restart_pids`` non-empty, or ``None`` (pre-state unreadable —
      cannot prove nothing was running; same contract as
      ``_restart_phase_failure_is_incomplete``, #78574).
    * the pre-update plan inventoried ≥1 runtime.
    * the Windows pause/resume token carries paused ``profiles`` or
      ``unmapped`` entries — ``_pause_windows_gateways_for_update`` /
      ``_resume_windows_gateways_after_update`` populate NEITHER
      ``restarted_services`` NOR ``killed_pids``, which is exactly why the
      original ``(restarted_services or killed_pids)`` guard never fires on
      Windows.

    The same condition gates the 2.0s settle sleep: a freshly resumed Windows
    gateway needs the settle window to rewrite ``gateway_state.json`` just
    like a systemd-restarted one.

    Note this keys ONLY on zero-rows-despite-expected-runtimes.  A non-empty
    snapshot — including rows in ``unknown`` state — is still judged solely by
    ``print_fleet_version_matrix``.
    """
    if restarted_services or killed_pids:
        return True
    if pre_restart_pids is None or pre_restart_pids:
        return True
    try:
        if pre_update_plan is not None and pre_update_plan.runtimes:
            return True
    except Exception:
        pass
    if isinstance(windows_resume_token, dict) and (
        windows_resume_token.get("profiles")
        or windows_resume_token.get("unmapped")
    ):
        return True
    return False


def _print_items(items, label, key, fallback_key=None):
    if not items:
        return
    print(f"  {label}:")
    shown = items[:8]
    for it in shown:
        if isinstance(it, dict):
            name = it.get(key) or (fallback_key and it.get(fallback_key)) or "?"
            desc = (it.get("description") or "").strip()
        else:
            # Defensive: some callers/mocks pass bare name strings.
            name = str(it)
            desc = ""
        if desc:
            print(f"      • {name} — {desc}")
        else:
            print(f"      • {name}")
    extra = len(items) - len(shown)
    if extra > 0:
        print(f"      … and {extra} more")

def _wait_for_service_active(
    scope_cmd_: list,
    svc_name_: str,
    timeout: float = 10.0,
) -> bool:
    """Poll ``systemctl is-active`` until the unit reports active.

    systemd's Stopped -> Started transition after a graceful exit
    (or a hard restart) is not instantaneous; a one-shot check
    races that window and falsely reports the unit as down.
    Poll every 0.5s up to ``timeout`` seconds before giving up.
    """
    deadline = _time.monotonic() + max(timeout, 0.5)
    while True:
        try:
            _verify = subprocess.run(
                scope_cmd_ + ["is-active", svc_name_],
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=5,
            )
            if _verify.stdout.strip() == "active":
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        if _time.monotonic() >= deadline:
            return False
        _time.sleep(0.5)

def _service_restart_sec(
    scope_cmd_: list,
    svc_name_: str,
    default: float = 0.0,
) -> float:
    """Read the unit's ``RestartUSec`` (RestartSec) in seconds.

    After a graceful exit-75, systemd waits ``RestartSec`` before
    respawning the unit.  Callers that poll for ``is-active``
    must use a timeout >= ``RestartSec`` + transition slack, or
    they'll give up *during* the cooldown window and wrongly
    conclude the unit didn't relaunch.
    """
    try:
        _show = subprocess.run(
            scope_cmd_
            + [
                "show",
                svc_name_,
                "--property=RestartUSec",
                "--value",
            ],
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return default
    raw = (_show.stdout or "").strip()
    # systemd emits values like "30s", "100ms", "1min 30s", or
    # "infinity".  Parse conservatively; on any miss return default.
    if not raw or raw == "infinity":
        return default
    total = 0.0
    matched = False
    for part in raw.split():
        for _suf, _mult in (
            ("ms", 0.001),
            ("us", 0.000001),
            ("min", 60.0),
            ("s", 1.0),
        ):
            if part.endswith(_suf):
                try:
                    total += float(part[: -len(_suf)]) * _mult
                    matched = True
                except ValueError:
                    pass
                break
    return total if matched else default
