"""Frozen historical updater function, executed by test_old_updater_shims.

Verbatim extraction from hermes_cli/update_cmd.py at
096826bf7ded2170eafe6a0781af22c807fba6c2 (_sync_python_dependencies_after_pull
and _install_psutil_android_compat).
Only the module scaffolding is supplied by the test. Lazy imports stay intact:
they must resolve to the real NEW tree, not test doubles or the old module.
This fixture avoids depending on a deep Git history in CI/shallow checkouts.
"""

# ruff: noqa: F821 -- globals came from the old process; the test supplies them.
import sys
from pathlib import Path


def _install_psutil_android_compat(
    install_cmd_prefix: list[str],
    *,
    env: dict[str, str] | None = None,
) -> None:
    """Install psutil on Android by patching upstream platform detection.

    psutil's setup gates Linux sources behind ``sys.platform.startswith('linux')``;
    Termux reports ``'android'``, so setup aborts although the Linux source path
    compiles fine. Only the extracted build tree for this attempt is patched.

    Stopgap: remove (together with the standalone installer's use of the same
    helper) once https://github.com/giampaolo/psutil/pull/2762 ships.
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


def _sync_python_dependencies_after_pull(
    git_cmd,
    branch,
    pre_pull_sha,
    *,
    active_lazy_features,
    active_tool_dependencies,
    _windows_gateway_resume,
):
    """Reinstall Python dependencies for the freshly pulled checkout.

    Order matters: ownership preflight -> self-lock deferral -> core-install
    marker -> ``.[all]`` (uv or pip) -> bytecode sweep -> lazy-feature and
    tool-dependency refresh (own marker) -> memory-provider bridge deps ->
    critical-import probe (warn only; stale-bytecode self-heals next launch).
    """
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
        # sys.executable -m pip avoids PEP 668 'externally-managed-environment' errors.
        pip_cmd = [sys.executable, "-m", "pip"]
        _ensure_venv_pip(pip_cmd, sys.executable)
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
    _sweep_bytecode_after_update(branch)
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

    # All transient-ImportError sources have run, so a module that still
    # won't import is real breakage. Warn only — never roll back: `cannot
    # import name X` is also the stale-bytecode signature (#6207, #60242),
    # which _sweep_stale_bytecode_if_checkout_changed() self-heals next launch.
    import_ok, failing_module, import_error = _validate_critical_modules_import(
        _m().PROJECT_ROOT
    )
    if not import_ok:
        print()
        print(f"  ⚠ {failing_module} still fails to import after updating:")
        print(f"      {import_error}")
        print("    Run `hermes update` again — if it persists, reinstall:")
        print("    https://hermes-agent.nousresearch.com")
