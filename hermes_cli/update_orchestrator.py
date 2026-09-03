"""Top-level update phase orchestration.

Extracted mechanically from :mod:`hermes_cli.update_cmd`.  Runtime
references to the historical module surface resolve through the
compatibility facade so imports and monkeypatches remain effective.
"""


from hermes_cli.update_fleet_restart import _restart_gateway_fleet


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
                if _refuse_gateway_ancestor_tree_kill(
                    _gateway_holders, gateway_mode=gateway_mode
                ):
                    _m()._resume_windows_gateways_after_update(
                        _windows_gateway_resume
                    )
                    sys.exit(2)
                # Every remaining holder is a gateway the pause machinery
                # already owns — respawned by its supervisor inside the
                # pause→guard window, or up through a spawn path discovery
                # does not map. Stop them and re-check instead of
                # dead-ending; the post-update resume (and the supervisor
                # that respawned them) brings gateways back afterwards.
                from gateway.status import get_process_start_time, terminate_pid

                print(
                    f"  ⚠ {len(_gateway_holders)} gateway process(es) still "
                    "hold the venv after the pause; stopping them"
                )
                for _pid in _gateway_holders:
                    try:
                        pid_int = int(_pid)
                        terminate_pid(
                            pid_int,
                            force=True,
                            expected_start_time=get_process_start_time(pid_int),
                        )
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
            # Manual serve/dashboard rung (#63206): a network-bound
            # `hermes serve --host <ip>` powering a REMOTE Desktop holds the
            # venv and used to dead-end the update with exit 2 — the user's
            # only option was killing the backend by hand, and nothing ever
            # brought it back (the remote client's endpoint stayed dead).
            # Positive ledger identity only: self-registered serve/dashboard
            # whose recorded spawner is not alive (Desktop-owned backends
            # keep the refusal — the app respawns what we kill). Stop them,
            # and register an idempotent atexit relaunch built from the
            # ledger's structured host/port/profile so the endpoint comes
            # back on the SAME bind after the update — success or failure.
            _serve_entries = _m()._ledger_manual_serve_holders(_venv_holders)
            if _serve_entries:
                print(
                    f"  ⚠ {len(_serve_entries)} manual serve/dashboard "
                    "backend(s) hold the venv; stopping them for the update "
                    "(they will be relaunched on their recorded endpoints)"
                )
                _m()._stop_process_trees(
                    [int(e["pid"]) for e in _serve_entries]
                )
                _serve_resume_token = {
                    "pending": True,
                    "entries": _serve_entries,
                }
                try:
                    from hermes_cli.update_receipt import record_step

                    record_step(
                        "serve_pause",
                        True,
                        f"stopped={len(_serve_entries)}",
                    )
                except Exception:
                    pass
                import atexit as _serve_atexit

                _serve_atexit.register(
                    _m()._relaunch_stopped_serves, _serve_resume_token
                )
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
    # A broken Git-for-Windows trampoline refuses every git call with a
    # "BUG (fork bomb)" guard instead of running; swap in a real binary up
    # front so the normal git path survives instead of degrading to ZIP
    # (#87876).
    git_cmd = _ensure_non_trampoline_git(git_cmd)

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
        from hermes_cli.gitlock import clear_stale_git_locks, clear_stale_tmp_packs

        cleared = clear_stale_git_locks(_m().PROJECT_ROOT)
        if cleared:
            print("  (removed stale git lock(s): %s)" % ", ".join(cleared))
        swept = clear_stale_tmp_packs(_m().PROJECT_ROOT)
        if swept:
            print("  (removed %d aborted-fetch pack temp file(s))" % len(swept))

        # Surface autostash entries left behind by earlier updates (#63717
        # problem 6) — parked --keep-stash runs and failed restores preserve
        # the stash but nothing ever mentioned it again.
        _m()._warn_orphaned_update_autostashes(git_cmd, _m().PROJECT_ROOT)

        print("→ Fetching updates...")
        fetch_result = subprocess.run(
            git_cmd + ["fetch", "origin", branch],
            cwd=_m().PROJECT_ROOT,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            **_no_prompt_git_kwargs(),
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
        # Non-fork checkouts have no upstream question: origin IS the official
        # repo, so "Already up to date!" is fully verified there.
        upstream_checked = True
        if commit_count == 0 and is_fork and branch == "main":
            pre_sync_sha = _capture_head_sha(git_cmd, _m().PROJECT_ROOT)
            upstream_checked = _m()._sync_with_upstream_if_needed(
                git_cmd,
                _m().PROJECT_ROOT,
                assume_yes=assume_yes,
                input_fn=gw_input_fn,
            )
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
            current_checkout_complete = True
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
                    _check_and_apply_config_migration(
                        assume_yes=assume_yes,
                        gateway_mode=gateway_mode,
                        pre_update_snapshot_id=pre_update_snapshot_id,
                    )
                    # The Windows hand-off child lands here after doing the
                    # sync its parent could not, and the commits-pulled
                    # rebuild below is never reached — rebuild the Desktop
                    # app here or it silently stays on the old build
                    # (#97343).
                    if _rebuild_desktop_after_update(
                        desktop_dir,
                        had_desktop_app_before_update=had_desktop_app_before_update,
                    ):
                        current_checkout_complete = _print_verified_update_completion(
                            "✓ Update complete!"
                        )
                    else:
                        current_checkout_complete = False
                        _print_update_completion(
                            "⚠ Update partially complete — the desktop app was "
                            "not rebuilt and is still on the previous build."
                        )
                else:
                    current_checkout_complete = False
                    print(f"⚠ Venv still unhealthy after repair: {detail_after}")
                    print("  Close all Hermes windows/gateways and re-run: hermes update")
            else:
                current_checkout_complete = _repair_node_deps_on_current_checkout(
                    _print_verified_update_completion,
                    assume_yes=assume_yes,
                    gateway_mode=gateway_mode,
                    pre_update_snapshot_id=pre_update_snapshot_id,
                    completion_message=(
                        "✓ Already up to date!"
                        if upstream_checked
                        else "✓ Up to date with your fork (official repo not checked)."
                    ),
                    had_desktop_app_before_update=had_desktop_app_before_update,
                )
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
            # Git is current, but a prior pull may still owe the fleet a
            # restart (#95294). Catch up even on the "Already up to date"
            # path — that early return is what left the gateway on stale
            # code for two days. Runs BEFORE the runtime-verification exit
            # gate below: a vulnerable SQLite runtime demotes the outcome to
            # partial, but must not strand the fleet on stale code (#91277
            # fleet contract — the pending-restart check always executes).
            _apply_pending_fleet_restart_catchup()
            if not current_checkout_complete:
                if gateway_mode:
                    _write_gateway_update_exit_code(False)
                try:
                    from hermes_cli.update_receipt import finalize_update_receipt

                    finalize_update_receipt("partial")
                except Exception as _receipt_exc:
                    logger.debug(
                        "Update receipt finalize (current checkout) failed: %s",
                        _receipt_exc,
                    )
                sys.exit(1)
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
                    #
                    # Divergence here comes in two shapes: ordinary (a common
                    # ancestor still exists — the reset below is a normal,
                    # safe re-sync onto the remote) and orphan (no common
                    # ancestor at all, e.g. a corrupted local HEAD or a repo
                    # re-init — see #87694). In the orphan case `reset --hard`
                    # would silently discard the entire local commit graph
                    # with no way back, so park pre_pull_sha behind a rescue
                    # ref first.
                    merge_base_result = subprocess.run(
                        git_cmd + ["merge-base", "HEAD", f"origin/{branch}"],
                        cwd=_m().PROJECT_ROOT,
                        capture_output=True,
                        text=True, encoding="utf-8", errors="replace",
                    )
                    has_common_ancestor = bool(
                        merge_base_result.returncode == 0
                        and merge_base_result.stdout.strip()
                    )
                    if not has_common_ancestor and pre_pull_sha:
                        from datetime import datetime as _dt, timezone

                        # Suffix with the pre-pull SHA (not just a second-
                        # resolution timestamp) so two updates racing within
                        # the same second (e.g. a retried update) get distinct
                        # refs instead of the second `update-ref` silently
                        # overwriting the first backup.
                        rescue_ref = (
                            f"refs/hermes-update-backups/orphan-{branch}-"
                            f"{_dt.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
                            f"-{pre_pull_sha[:12]}"
                        )
                        update_ref_result = subprocess.run(
                            git_cmd + ["update-ref", rescue_ref, pre_pull_sha],
                            cwd=_m().PROJECT_ROOT,
                            capture_output=True,
                            text=True, encoding="utf-8", errors="replace",
                        )
                        if update_ref_result.returncode == 0:
                            print(
                                "  ⚠ Local history shares no common ancestor with "
                                f"origin/{branch} (orphan divergence) — backed up "
                                f"current HEAD to {rescue_ref} before resetting. "
                                f"This backup expires after "
                                f"{_ORPHAN_RESCUE_REF_MAX_AGE_DAYS} days."
                            )
                        else:
                            # update-ref's return code is intentionally not
                            # fatal (disk full, permissions) — but don't tell
                            # the user a backup exists when the write failed.
                            print(
                                "  ⚠ Local history shares no common ancestor with "
                                f"origin/{branch} (orphan divergence) — attempted "
                                f"to back up current HEAD to {rescue_ref} before "
                                "resetting, but the backup write failed "
                                f"(pre-reset SHA was {pre_pull_sha})."
                            )
                        _prune_orphan_rescue_refs(git_cmd, _m().PROJECT_ROOT, branch)
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

        # #95294: HEAD advanced; running gateways still serve pre-pull
        # modules until the restart phase below. Any interrupt between here
        # and a completed (or no-op) restart leaves this marker so the next
        # ``hermes update`` can catch up even when git is already up to date.
        # Distinct from ``.update-incomplete`` (venv/install repair).
        _write_fleet_restart_pending_marker(expected_sha=post_pull_sha or "")

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
            _m()._sync_with_upstream_if_needed(
                git_cmd,
                _m().PROJECT_ROOT,
                assume_yes=assume_yes,
                input_fn=gw_input_fn,
            )

        # Reinstall Python dependencies. Prefer .[all], but if one optional extra
        # breaks on this machine, keep base deps and reinstall the remaining extras
        # individually so update does not silently strip working capabilities.
        #
        # Ownership preflight (#83529): refuse before the first venv mutation
        # if the venv contains foreign-owned files (sudo-pip residue) — the
        # install below would die mid-mutation and brick the CLI.
        _refuse_update_if_venv_foreign_owned(_m().PROJECT_ROOT)
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

        # macOS TCC interpreter anchor (#95596): dylib-complete re-land.
        # Boot-gated — a failed probe leaves the venv untouched.
        try:
            from hermes_cli.macos_tcc_anchor import ensure_tcc_anchor

            ensure_tcc_anchor()
        except Exception:
            logger.debug("macOS TCC anchor refresh skipped", exc_info=True)

        # ── Post-update state.db integrity guard (#68474, #97994) ─────────
        # Verify that state.db survived the update intact in the root home
        # AND every sibling profile. If a live file is now corrupted (zeroed,
        # missing header, integrity failure), automatically restore from that
        # home's own pre-update snapshot rather than letting the user discover
        # silently that their sessions are gone.
        try:
            _verify_and_restore_state_dbs_post_update()
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

        # Check for config migrations (#91360).
        _check_and_apply_config_migration(
            assume_yes=assume_yes,
            gateway_mode=gateway_mode,
            pre_update_snapshot_id=pre_update_snapshot_id,
        )

        update_complete = _print_update_summary(
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

        (
            gateway_fleet_restart_incomplete,
            gateway_restart_phase_errors,
            _pre_restart_gateway_pids,
            restarted_services,
            failed_or_stale_units,
            relaunched_profiles,
            externally_supervised_profiles,
            killed_pids,
        ) = _restart_gateway_fleet(
            gateway_mode=gateway_mode,
            desktop_build_ok=desktop_build_ok,
            pre_update_plan=_pre_update_plan,
            windows_gateway_resume=_windows_gateway_resume,
        )

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

        # Check if any pre-update serve/dashboard runtimes survived on
        # pre-update code generations (#100479). This is the success-path
        # counterpart of abort recovery and participates in reconciliation.
        _stale_serve_rows: "list | None" = None
        try:
            _stale_serve_rows = _surviving_pre_update_serve_runtimes(_pre_update_plan)
            if _stale_serve_rows:
                _warn_stale_serve_runtimes(_stale_serve_rows)
        except Exception as _serve_warn_exc:
            logger.debug(
                "Failed to check for surviving serve runtimes: %s",
                _serve_warn_exc,
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
            #
            # On Windows the resume path relaunches the gateway DETACHED, and
            # that process must boot before it stamps gateway_state.json or
            # answers the control socket (a Telegram gateway reconnects its
            # polling loop — ~10s).  A single 2s sleep therefore races the
            # gateway's own startup and reports "no rows" (exit 1) for a
            # healthy resume, which then triggers a full retry that re-kills
            # the gateway the first attempt just started.  Poll a bounded
            # window for the resumed gateway to publish its identity instead.
            _fleet_snapshot = []
            if _fleet_rows_expected:
                _fleet_deadline = _time.monotonic() + 30.0
                while True:
                    _time.sleep(2.0)
                    # Pass the pre-restart PID snapshot so a gateway the
                    # restart phase stopped WITHOUT a verified replacement
                    # shows as a DOWN row (exit 1) instead of silently
                    # producing no row at all.
                    _fleet_snapshot = collect_fleet_versions(
                        pre_restart_pids=_pre_restart_gateway_pids
                    )
                    # A "down" row here is the stale pre-restart record of a
                    # gateway whose detached replacement is still booting —
                    # not a confirmed failure.  Keep polling until every
                    # resumed gateway has published (no "down" rows remain)
                    # or the deadline passes, so a slow second gateway can't
                    # be misread as down and re-trigger the retry loop.
                    if _fleet_snapshot and not any(
                        row.get("state") == "down" for row in _fleet_snapshot
                    ):
                        break
                    if _time.monotonic() >= _fleet_deadline:
                        break
            else:
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
                    stale_serve_pids=(
                        {row.get("pid") for row in _stale_serve_rows}
                        if _stale_serve_rows is not None
                        else None
                    ),
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
                (
                    "partial"
                    if gateway_fleet_restart_incomplete or not update_complete
                    else "success"
                ),
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
            # Leave ``fleet_restart_pending`` in place so the next
            # ``hermes update`` still runs the catch-up restart.
            sys.exit(1)
        _clear_fleet_restart_pending_marker()

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
