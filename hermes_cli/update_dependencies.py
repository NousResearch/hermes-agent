"""Python, lazy, Node, Termux, FHS, and ACP dependency refresh.

Extracted mechanically from :mod:`hermes_cli.update_cmd`.  Runtime
references to the historical module surface resolve through the
compatibility facade so imports and monkeypatches remain effective.
"""

from pathlib import Path


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


def _repair_node_deps_on_current_checkout(
    print_completion,
    *,
    assume_yes: bool = False,
    gateway_mode: bool = False,
    pre_update_snapshot_id: str | None = None,
    completion_message: str = "✓ Already up to date!",
    had_desktop_app_before_update: bool = False,
) -> bool:
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
        return False
    # Pair the refresh with the web build like every other
    # _update_node_dependencies call site; it staleness-checks internally,
    # so this is a no-op when nothing changed.
    _m()._build_web_ui(_m().PROJECT_ROOT / "web")
    _check_and_apply_config_migration(
        assume_yes=assume_yes,
        gateway_mode=gateway_mode,
        pre_update_snapshot_id=pre_update_snapshot_id,
    )
    # A current checkout can still owe a Desktop rebuild (#97343): the
    # packaged app is built from source the pull already landed — or, on the
    # Windows hand-off, by a child that never reaches the commits-pulled
    # rebuild. Skipping it leaves a stale desktop app behind a
    # successful-looking update. Self-gates on the build stamp, so this is a
    # no-op when nothing changed.
    if not _rebuild_desktop_after_update(
        _m().PROJECT_ROOT / "apps" / "desktop",
        had_desktop_app_before_update=had_desktop_app_before_update,
    ):
        # _rebuild_desktop_after_update already printed the retry hint; withhold
        # success rather than claiming the update finished (#88251).
        print_completion(
            "⚠ Update partially complete — the desktop app was not rebuilt "
            "and is still on the previous build."
        )
        return False
    return bool(print_completion(completion_message))


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
    if "could not read Username" in stderr or "terminal prompts disabled" in stderr:
        # Anonymous fetch of a public repo got HTTP 401. GitHub does this
        # during outages (and for renamed/private repos) — it is not a
        # credentials problem on the user's side.
        return (
            "✗ GitHub rejected the anonymous fetch (asked for a login) — this"
            " usually means a GitHub outage; try again in a few minutes"
            " (https://www.githubstatus.com). If it persists, check"
            " `git remote -v` points at a public repo."
        )
    if "Authentication failed" in stderr:
        return "✗ Authentication failed — check your git credentials or SSH key."
    return "✗ Failed to fetch updates from origin."


def _print_fetch_failure(stderr: str) -> None:
    """Print the classified diagnosis plus the first raw stderr line."""
    stderr = (stderr or "").strip()
    print(_classify_fetch_failure(stderr))
    if stderr:
        print(f"  {stderr.splitlines()[0]}")


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
