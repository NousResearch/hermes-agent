"""Runtime reload, config migration, and post-swap validation.

Extracted mechanically from :mod:`hermes_cli.update_cmd`.  Runtime
references to the historical module surface resolve through the
compatibility facade so imports and monkeypatches remain effective.
"""

from hermes_cli._update_compat import facade as _u


_UPDATE_RUNTIME_RELOAD_MODULES = (
    "hermes_constants",
    "tools.environments.local",
    "tools.lazy_deps",
)


_STALE_PURGE_PREFIXES = (
    "hermes_cli",
    "gateway",
    "tools",
    "tui_gateway",
    "agent",
)


_STALE_PURGE_PROTECTED = frozenset(
    {
        "hermes_cli",
        "hermes_cli.main",
        "hermes_cli._update_compat",
        "hermes_cli.update_cmd",
        "hermes_cli.update_backup",
        "hermes_cli.update_dependencies",
        "hermes_cli.update_desktop",
        "hermes_cli.update_fleet_restart",
        "hermes_cli.update_gateway_posix",
        "hermes_cli.update_gateway_windows",
        "hermes_cli.update_notices",
        "hermes_cli.update_orchestrator",
        "hermes_cli.update_process_guard",
        "hermes_cli.update_reconciliation",
        "hermes_cli.update_runtime_refresh",
        "hermes_cli.update_source",
        "hermes_cli.update_zip",
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


def _check_and_apply_config_migration(
    *,
    assume_yes: bool = False,
    gateway_mode: bool = False,
    pre_update_snapshot_id: str | None = None,
) -> None:
    """Check and apply configuration migrations on an update completion path (#91360).

    CRITICAL: ``check_config_version`` and ``migrate_config`` must use
    freshly-reloaded modules, not the ``sys.modules`` cache (see
    ``_reload_config_modules``). This must run on EVERY update completion
    path — the normal post-pull path, the venv-repair retry and the
    Node-deps repair on the ``commit_count == 0`` "Already up to date"
    branch — so an interrupted update that previously pulled new code does
    not strand the user on an older config version.
    """
    print()
    print("→ Checking configuration for new options...")

    # Reload config modules BEFORE any config reads so get_missing_*,
    # check_config_version, and migrate_config all use the updated code.
    _reload_config_modules()

    from hermes_cli.config import (
        get_missing_env_vars,
        get_missing_config_fields,
    )

    # Defensive (#91360): this helper runs on repair/retry completion paths
    # too — a config-check failure must not break an otherwise-successful
    # update. Log, point at the manual command, and return.
    try:
        missing_env = get_missing_env_vars(required_only=True)
        missing_config = get_missing_config_fields()
        current_ver, latest_ver = _run_config_check_fresh()
    except Exception as exc:
        logger.debug("Config check during update failed: %s", exc)
        print("  ⚠️  Could not check config version.")
        print("     Run 'hermes config migrate' to check manually.")
        return

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

    # #64160: config.yaml model/provider + MoA safety net. Desktop
    # update/repair cycles have rewritten user-set model.provider /
    # model.default and dropped the moa: section — settings the gateway
    # and cron jobs also consume. Compare the live config against the
    # same pre-update snapshot and restore only the protected keys.
    try:
        from hermes_cli.backup import restore_config_model_settings_if_rewritten

        cfg_restore = restore_config_model_settings_if_rewritten(
            pre_update_snapshot_id
        )
        if cfg_restore:
            print()
            print(
                "  ⚠️  config.yaml user model settings were rewritten during "
                f"this update — restored {', '.join(cfg_restore['keys'])} "
                f"from pre-update snapshot {cfg_restore['snapshot_id']}."
            )
    except Exception as exc:
        # Never let the config safety net break an otherwise-good update.
        logger.debug("Config model-settings auto-restore check failed: %s", exc)

    # #66140: run the same cron-jobs safety net for every sibling
    # profile against ITS OWN pre-update snapshot (same-generation by
    # construction — both taken by this run).
    try:
        from hermes_cli.backup import restore_cron_jobs_all_profiles

        for _restored in restore_cron_jobs_all_profiles(
            _u()._LAST_SIBLING_SNAPSHOTS
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

    # #64160: same config model-settings safety net for sibling profiles.
    try:
        from hermes_cli.backup import restore_config_model_settings_all_profiles

        for _cfg_restored in restore_config_model_settings_all_profiles(
            _u()._LAST_SIBLING_SNAPSHOTS
        ):
            print()
            print(
                f"  ⚠️  Profile '{_cfg_restored['profile']}': config.yaml "
                f"user model settings were rewritten during this update — "
                f"restored {', '.join(_cfg_restored['keys'])} from "
                f"pre-update snapshot {_cfg_restored['snapshot_id']}."
            )
    except Exception as exc:
        logger.debug("Sibling config auto-restore check failed: %s", exc)


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
    return _validate_python_files_syntax(root, _UPDATE_CRITICAL_FILES)


_UPDATE_CRITICAL_MODULES = (
    "hermes_cli.main",
    "run_agent",
    "model_tools",
    "toolsets",
)


def _validate_critical_modules_import(
    root, *, report_runtime_errors: bool = False
) -> tuple[bool, str | None, str | None]:
    """Return the first critical-module import failure, if any."""
    failures = _critical_module_import_failures(
        root, report_runtime_errors=report_runtime_errors
    )
    if failures:
        module = next(iter(failures))
        return False, module, failures[module][1]
    return True, None, None


def _validate_python_files_syntax(
    root, relpaths
) -> tuple[bool, str | None, str | None]:
    """Compile *relpaths* under *root* without writing bytecode into the tree."""
    import py_compile
    import tempfile

    root = Path(root)
    with tempfile.TemporaryDirectory(prefix="hermes-syntax-check-") as tmpdir:
        for relpath in relpaths:
            path = root / relpath
            if not path.exists():
                continue
            cfile = Path(tmpdir) / (str(relpath).replace("/", "__") + "c")
            try:
                py_compile.compile(str(path), cfile=str(cfile), doraise=True)
            except py_compile.PyCompileError as exc:
                return False, str(path), str(exc)
            except OSError as exc:
                return False, str(path), f"could not read: {exc}"
    return True, None, None


def _critical_module_import_failures(
    root, *, report_runtime_errors: bool = False
) -> dict[str, tuple[str, str]]:
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

    Returns every failing module in probe order. Generic import-time exceptions
    remain tolerated by default because they can depend on local config or
    environment. ``report_runtime_errors=True`` exposes them so a caller can
    compare two states of the same checkout without an earlier failure masking
    a later one.
    """
    from hermes_constants import FIRST_PARTY_MODULE_ROOTS

    import secrets

    marker = f"__HERMES_IMPORT_HEALTH_{secrets.token_hex(16)}__"
    probe = (
        "import importlib, json, sys\n"
        "failures = []\n"
        "for name in %r:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except ModuleNotFoundError as exc:\n"
        # A missing *third-party* module means dependencies aren't installed
        # yet, not a skewed checkout. Only our own packages count as breakage.
        # The root set is injected from hermes_constants so this can't drift
        # from the hint the user is shown (they disagreed once already).
        "        missing = (getattr(exc, 'name', '') or '').split('.')[0]\n"
        "        if missing in %r or missing.startswith('hermes_') or %r:\n"
        "            failures.append((name, type(exc).__name__, str(exc)))\n"
        "    except ImportError as exc:\n"
        "        failures.append((name, type(exc).__name__, str(exc)))\n"
        "    except Exception as exc:\n"
        "        if %r:\n"
        "            failures.append((name, type(exc).__name__, str(exc)))\n"
        "    except BaseException as exc:\n"
        "        failures.append((name, type(exc).__name__, str(exc)))\n"
        "sys.stdout.write('\\n%s' + json.dumps(failures))\n"
        % (
            _UPDATE_CRITICAL_MODULES,
            tuple(sorted(FIRST_PARTY_MODULE_ROOTS)),
            report_runtime_errors,
            report_runtime_errors,
            marker,
        )
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
    except subprocess.TimeoutExpired:
        return {
            "critical-module probe": (
                "TimeoutExpired",
                "timed out before reporting import health",
            )
        }
    except (OSError, subprocess.SubprocessError):
        # Can't run the probe — don't block the update on our own tooling.
        return {}
    output = result.stdout or ""
    if marker not in output:
        return {
            "critical-module probe": (
                "ProbeTerminated",
                "terminated before reporting import health "
                f"(exit code {result.returncode})",
            )
        }
    try:
        import json

        failures = json.loads(output.rsplit(marker, 1)[1])
        if not isinstance(failures, list) or any(
            not isinstance(item, list)
            or len(item) != 3
            or not all(isinstance(value, str) for value in item)
            for item in failures
        ):
            raise ValueError("invalid import-health payload")
        return {
            str(module): (str(kind), str(detail))
            for module, kind, detail in failures
        }
    except (TypeError, ValueError):
        return {
            "critical-module probe": (
                "MalformedPayload",
                "reported malformed import health data",
            )
        }


def _post_update_sqlite_runtime_status():
    """Return whether the interpreter used after update has safe SQLite."""
    from hermes_constants import project_venv_dir
    from hermes_cli.sqlite_runtime import probe_sqlite_runtime

    venv_dir = project_venv_dir(_m().PROJECT_ROOT)
    python = (
        venv_python_path(venv_dir, windows=_m()._is_windows())
        if venv_dir is not None
        else Path(sys.executable)
    )
    info = probe_sqlite_runtime(python)
    return info is not None and not info.wal_reset_vulnerable, info


def _print_verified_update_completion(message: str) -> bool:
    """Print a success completion only after probing the next Hermes runtime."""
    if not message.startswith("✓"):
        _print_update_completion(message)
        return False
    sqlite_runtime_ok, sqlite_info = _post_update_sqlite_runtime_status()
    if sqlite_info is None:
        # Grace path: an unprobeable interpreter (no venv in a dev checkout,
        # probe subprocess unavailable) must not fail an otherwise-successful
        # update — only a POSITIVE vulnerable probe withholds success
        # (same contract as _venv_core_imports_healthy's unknown states).
        logger.debug("Post-update SQLite runtime probe unavailable; not blocking")
        _print_update_completion(message)
        return True
    if sqlite_runtime_ok:
        _print_update_completion(message)
        return True
    print()
    detail = (
        f"SQLite {sqlite_info.sqlite_version_string} still has the "
        "WAL-reset corruption bug"
    )
    print(f"⚠ Update partially complete — {detail}.")
    print(
        "  Rebuild the Hermes venv with a uv-managed Python, restart Hermes, "
        "then verify with `hermes doctor`."
    )
    return False
