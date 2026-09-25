"""hermes pm: lock / install / repair / env / doctor / gc / bundle."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from pm import termux_libs
from pm.install import _facts, _lockfile, _store, ensure, stage_only
from pm.operations import lock_project
from pm.package import InstallError
from pm.paths import repo_root
from pm.registry import get_package, source_install_packages, tool_roots
from pm.store import ALL_TARGETS, current_target, hash_url
from pm.update import Resolved, resolve_package, reuse_index_responses


def cmd_lock(args) -> int:
    """No arguments: relock uv.lock from pyproject.toml. --bump: pin a tool in pm/lock.json."""
    if args.name is None:
        return _relock_project(repo_root())
    return _pin_tool(args)


def _relock_project(root: Path) -> int:
    """The contributor's one step after editing pyproject.toml.

    The same PM operations as `python -m pm.build_env --source . --check-lock`
    and `--lock-only`, so the exclude-newer quarantine and resolver settings
    cannot drift between the two entry points. No environment changes here:
    activation owns syncing, and a relock that also installed would hide
    which of the two failed.
    """
    import pm

    print("→ Checking uv.lock…", flush=True)
    try:
        # Quiet: stale is the expected case here, so uv's failure tail would
        # read like an error. An unreachable index fails the relock below too,
        # and that error names the index knobs.
        pm.check_project_lock(root, explicit=True, quiet=True)
    except InstallError:
        print("uv.lock is out of date with pyproject.toml; relocking", flush=True)
    else:
        print("✓ uv.lock is already current with pyproject.toml; nothing written")
        return 0
    before = _locked_extras(root)
    try:
        pm.lock_project(root, explicit=True)
    except InstallError as exc:
        print(f"✗ uv.lock refresh failed: {exc}")
        return 1
    print("✓ uv.lock updated from pyproject.toml")
    windows = sys.platform == "win32"
    activate = r". .\activate.ps1" if windows else "source ./activate"
    print(f"Next: re-source activation to sync the environments: {activate}")
    opt_in = _new_opt_in_extras(root, before)
    if opt_in:
        # Activation syncs [all] plus extras already recorded, and --test-extras
        # REPLACES the default rather than adding to it, so [all] stays listed.
        names = ",".join(["all", *opt_in])
        flag = f"-TestExtras '{names}'" if windows else f"--test-extras {names}"
        print(f"New extras outside [all] ({', '.join(opt_in)}) need: {activate} {flag}")
    print("Then commit pyproject.toml and uv.lock together.")
    return 0


def _locked_extras(root: Path) -> set[str]:
    """Extras the current uv.lock already resolves for the root project."""
    import tomllib

    try:
        with (root / "uv.lock").open("rb") as f:
            lock = tomllib.load(f)
        with (root / "pyproject.toml").open("rb") as f:
            name = tomllib.load(f)["project"]["name"]
    except (OSError, ValueError, KeyError):
        return set()
    for package in lock.get("package", ()):
        if package.get("name") == name:
            return set(package.get("metadata", {}).get("provides-extras", ()))
    return set()


def _new_opt_in_extras(root: Path, before: set[str]) -> list[str]:
    """Extras this relock introduced that the default [all] closure does not reach."""
    import re
    import tomllib

    with (root / "pyproject.toml").open("rb") as f:
        project = tomllib.load(f)["project"]
    extras = project.get("optional-dependencies", {})
    self_ref = re.compile(rf"\s*{re.escape(project['name'])}\s*\[([^\]]+)\]")
    covered, pending = set(), ["all"]
    while pending:
        extra = pending.pop()
        if extra in covered:
            continue
        covered.add(extra)
        for requirement in extras.get(extra, ()):
            match = self_ref.match(requirement)
            if match:
                pending.extend(part.strip() for part in match.group(1).split(","))
    return sorted(set(extras) - before - covered)


@reuse_index_responses()
def _pin_tool(args) -> int:
    """--bump <name> <version>: resolve every target's archives, hash them,
    write. A target with one archive pins the object; several pin a list.
    Target-independent urls collapse to one "any" artifact."""
    lockfile = _lockfile()
    package = get_package(args.name)
    artifacts: dict[str, object] = {}

    def pin(url: str) -> dict:
        print(f"    {url}")
        digest = package.known_sha256(args.version, url) or hash_url(url)
        print(f"      sha256 {digest}")
        return {"url": url, "sha256": digest}

    urls = {
        target: package.fetch_urls(args.version, target)
        for target in ALL_TARGETS
        if package.missing_reason(target) is None
    }
    distinct = {tuple(u) for u in urls.values()}
    if len(distinct) == 1:
        print("  any:")
        pinned = [pin(url) for url in next(iter(urls.values()))]
        artifacts["any"] = pinned[0] if len(pinned) == 1 else pinned
    else:
        for target, target_urls in urls.items():
            print(f"  {target}:")
            pinned = [pin(url) for url in target_urls]
            artifacts[target] = pinned[0] if len(pinned) == 1 else pinned
    lockfile.set_pin(args.name, args.version, artifacts)
    lockfile.save()
    print(f"pinned {args.name} {args.version} ({len(artifacts)} targets)")
    return 0


def _fmt_bytes(n: int) -> str:
    return f"{n / (1024 * 1024):.1f} MiB"


def _enable_vt(stream) -> bool:
    """A Windows console prints ESC as a glyph (`←[2K`) until its handle opts
    in to VT processing, and PowerShell hands children a console with VT off.
    NUL also reports isatty() but is no console, so it fails here too."""
    if sys.platform != "win32":
        return True
    import ctypes
    import msvcrt
    from ctypes import wintypes

    try:
        handle = msvcrt.get_osfhandle(stream.fileno())
    except (AttributeError, OSError, ValueError):
        return False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    mode = wintypes.DWORD()
    if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        return False
    vt = 0x0004  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    return bool(mode.value & vt or kernel32.SetConsoleMode(handle, mode.value | vt))


def _progress_stream():
    """Stream for in-place progress, or None off a terminal. Prefer stdout;
    fall back to stderr because activate.ps1 pipes stdout through Out-Host
    while stderr stays on the console."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream.isatty() and _enable_vt(stream):
                return stream
        except (AttributeError, ValueError):
            continue
    return None


def _interactive() -> bool:
    return _progress_stream() is not None


def _live_progress(name: str):
    """Per-package progress for ensure(): download as % + MiB, unpack as a
    phase line. On a terminal the line redraws in place (~10 Hz); in a piped
    (CI) log each tick prints its own line, throttled to ~4 MiB steps so a
    slow line proves it's moving without flooding the log (a 1 MiB tick on a
    1.5 GiB model would be ~1,500 lines). ``finish()`` returns the cursor to
    a clean line so the caller's status glyph starts on its own row."""
    last = 0
    last_time = 0.0
    stream = _progress_stream()

    def write(line: str) -> None:
        if stream is not None:
            stream.write("\r\x1b[2K" + line)
            stream.flush()
        else:
            print(line, flush=True)

    def finish() -> None:
        if stream is not None:
            stream.write("\r\x1b[2K")
            stream.flush()

    def report(stage: str, done: int, total: int, label: str) -> None:
        nonlocal last, last_time
        if stage == "unpack":
            last = 0
            last_time = 0.0
            write(f"  {name}: unpacking{(' ' + label) if label else ''}")
            return
        if total <= 0:
            return
        if done < total:
            if stream is not None:
                now = time.monotonic()
                if now - last_time < 0.1:
                    return
                last_time = now
            elif done - last < 4 * 1024 * 1024:
                return
        last = done
        write(f"  {name}: {done / total * 100:5.1f}%  {_fmt_bytes(done)} / {_fmt_bytes(total)}")

    report.finish = finish  # type: ignore[attr-defined]
    return report


def _install_names(names: list[str], target: str | None = None, *, verify: bool = True) -> int:
    from pm.install import _install_operation

    failed = 0
    with _install_operation() as operation:
        for name in names:
            progress = _live_progress(name)
            try:
                if target is not None:
                    # Cross-target staging: publish the entry, touch no facts.
                    entry = stage_only(name, target)
                    print(f"✓ {name} (staged for {target}: {entry.name})")
                else:
                    ensure(name, explicit=True, verify=verify, progress=progress, _operation=operation)
                    if name == "python":
                        from hermes_cli.venv_sync import publish_launchers

                        publish_launchers(repo_root(), create=False)
                    progress.finish()
                    print(f"✓ {name}", flush=True)
            except InstallError as e:
                progress.finish()
                print(f"✗ {e}", flush=True)
                failed += 1
    return failed




def _install_flag_error(args, *, extras: list[str], cross_target, tools_only: bool,
                        trust_recorded: bool, test_environment) -> str | None:
    """The first incompatible flag combination, as its user-facing message."""
    if cross_target:
        if cross_target not in ALL_TARGETS:
            return f"unknown target {cross_target!r}; known: {', '.join(ALL_TARGETS)}"
        if not args.names:
            return "--target requires explicit package names"
    if test_environment is not None and (extras or cross_target or args.names or tools_only):
        return "--test-environment builds beside the default closure; it does not take names, --extra, --target, or --tools-only"
    if tools_only and (extras or cross_target or args.names):
        return "--tools-only installs the tool closure and then stops; it does not take names, --extra, or --target"
    if trust_recorded and (extras or cross_target or args.names):
        return "--trust-recorded installs the default closure and then stops; it does not take names, --extra, or --target"
    if extras and cross_target:
        return "--extra syncs this install's venv and cannot combine with --target"
    without = getattr(args, "without", None) or ()
    if without and (extras or cross_target or args.names):
        return "--without shapes the default closure; it does not take names, --extra, or --target"
    if without:
        from pm.defaults import default_package_names

        allowed = default_package_names()
        unknown = [name for name in without if name not in allowed]
        if unknown:
            return (f"--without accepts only optional default packages ({', '.join(allowed)}); "
                    f"not {', '.join(unknown)}")
    return None


def _install_defaults(names: list[str], *, verify: bool) -> None:
    """Install the optional defaults; a failure warns and never fails the install.

    They are conveniences (the browser tools), not what Hermes needs to run:
    a Chromium download that fails behind a proxy must not abort an install
    whose required closure and venv are fine.
    """
    for name in names:
        if _install_names([name], verify=verify):
            print(f"⚠ optional {name} was not installed; its tools stay unavailable until "
                  f"`hermes pm install {name}` succeeds", flush=True)


def _install_python_environments(extras: list[str], *, sync: bool, test_environment) -> int:
    """Sync the venv and build the side test environment; return the failure count."""
    failed = 0
    if sync:
        from pm.install import sync_venv

        try:
            # Default the venv to the [all] feature set — the same thing
            # `hermes update` force-syncs on every run (update_cmd.py) and
            # the installers' old `--extra all` did. sync_venv unions, so
            # any lazy extras already recorded survive this; it only makes
            # a fresh bootstrap match what the first update would do.
            sync_venv(extras or ["all"], explicit=True)
            print(f"✓ venv{' +' + ' +'.join(extras) if extras else ''}")
        except InstallError as e:
            print(f"✗ {e}")
            failed += 1
    if test_environment is not None and not failed:
        from pm import check_project_lock
        from pm.testenv import ensure_testenv, parse_extras

        # Before the input stamps: they then cover this environment too, so
        # the shebang/run_tests.sh staleness check rebuilds it when it drifts.
        try:
            check_project_lock(repo_root(), explicit=True)
            ensure_testenv(repo_root(), parse_extras(test_environment))
            print("✓ test environment")
        except InstallError as e:
            print(f"✗ {e}")
            failed += 1
    return failed


def cmd_install(args) -> int:
    cross_target = getattr(args, "target", None)
    # Source-install launchers require the store interpreter, even though
    # Python remains optional when provisioning individual tools.
    extras = list(dict.fromkeys(getattr(args, "extra", None) or ()))
    tools_only = bool(getattr(args, "tools_only", False))
    trust_recorded = bool(getattr(args, "trust_recorded", False))
    test_environment = getattr(args, "test_environment", None)
    error = _install_flag_error(args, extras=extras, cross_target=cross_target, tools_only=tools_only,
                                trust_recorded=trust_recorded, test_environment=test_environment)
    if error:
        print(f"✗ {error}")
        return 1
    names = args.names if args.names or extras else source_install_packages(_lockfile().names())
    without = list(dict.fromkeys(getattr(args, "without", None) or ()))
    if without:
        from pm.defaults import record_declined

        # Persisted before anything installs: later bare installs and
        # `hermes update` read the same record, so the opt-out sticks.
        record_declined(add=without)
    defaults: list[str] = []
    if not (args.names or extras):
        from pm.defaults import default_packages

        defaults = default_packages(_lockfile().names())
    # Only the whole default closure verifies everything an activated shell
    # composes from, so only it advances the prologue's input stamps.
    full_closure = not (args.names or tools_only or cross_target)
    from pm.environments import activation_input_mtimes

    input_mtimes = activation_input_mtimes(repo_root()) if full_closure else {}
    # Tools before the venv. A bare `pm install` used to install tools and
    # sync the venv in one breath, so a native build (Windows ARM64 source
    # wheels) resolved compilers and git from the host PATH. Publish every
    # tool first and put it on PATH; sync only after that.
    tool_names = names if args.names else tool_roots(names)
    if _install_names(tool_names, target=cross_target, verify=not trust_recorded):
        return 1
    if args.names and not cross_target:
        from pm.defaults import record_declined

        # Naming a declined default is the opt-back-in: updates carry it again.
        record_declined(remove=args.names)
    if not cross_target and (not args.names or tools_only):
        from pm.install import activate

        problems = activate(allow_incomplete=True)
        if problems:
            print(f"✗ tools not on PATH before venv sync: {'; '.join(problems)}", flush=True)
            return 1
    if tools_only:
        _install_defaults(defaults, verify=not trust_recorded)
        return 0
    failed = _install_python_environments(extras, sync=bool(extras or not args.names),
                                          test_environment=test_environment)
    # Defaults are optional and large (agent-browser + Chromium): fetch them
    # only once the venv, and on Windows ARM64 its build tools, succeeded.
    if not failed:
        _install_defaults(defaults, verify=not trust_recorded)
    if full_closure and not failed:
        from pm.environments import activation_inputs_dir, record_activation_inputs

        record_activation_inputs(activation_inputs_dir(repo_root()), input_mtimes, repo_root(),
                                 test_environment=test_environment is not None)
    return 1 if failed else 0


def cmd_env(args) -> int:
    from pm.install import env_for

    names = args.names or _lockfile().names()
    # Show only package exports, never the caller's environment (which can
    # contain credentials and is commonly pasted into support reports).
    print(json.dumps(env_for(*names, base_env={}), indent=2, sort_keys=True))
    return 0


def cmd_doctor(args) -> int:
    from pm.install import _identity, _installed_location
    from pm.store import tree_digest

    lockfile = _lockfile()
    facts = _facts()
    store = _store()
    target = current_target()
    bad = 0
    for name in lockfile.names():
        package = get_package(name)
        reason = package.missing_reason(target)
        if reason is not None:
            print(f"- {name}: n/a on {target} ({reason})")
            continue
        facts, store = _installed_location(package, lockfile, target) or (_facts(), _store())
        fact = facts.get(name)
        soft = package.optional or package.internal
        identity = _identity(lockfile, name, target)
        if (
            fact is not None
            and identity is not None
            and ("target" not in fact or "artifacts" not in fact)
        ):
            # Legacy fact: pre-dates digest-bound identity; installed()
            # treats it as not installed and forces one reinstall.
            print(f"{'?' if soft else '✗'} {name}: legacy fact: no recorded identity, run `hermes pm install`")
            bad += 0 if soft else 1
            continue
        if not facts.installed(name, lockfile.version(name), store.root, identity):
            state = "not installed" if fact is None else "outdated"
            print(f"{'?' if soft else '✗'} {name}: {state}")
            bad += 0 if soft else 1
            continue
        entry = store.entry(fact["entry"])
        reason = package.verify(entry, target)
        if reason:
            print(f"✗ {name}: installed but failed verification: {reason}")
            bad += 1
            continue
        recorded = fact.get("digest")
        if recorded is not None and tree_digest(entry) != recorded:
            # Doctor is the expensive-path tool: re-hash the realized
            # bytes. Boot checks stay O(1) json compares.
            print(f"✗ {name}: realized bytes do not match recorded digest")
            bad += 1
            continue
        print(f"✓ {name} {fact['version']}")
    return 1 if bad else 0


def _gc_store(store, facts) -> tuple[int, int]:
    """The sweep core shared by `pm gc` and `pm bundle`.

    Removes every store entry nothing references: fetch-<sha> download-cache
    dirs (the raw archives — needed only at install time, dead weight in a
    staged payload or a CI cache), orphaned package versions from an older
    lock, and expired partials. Keeps live package entries (recorded in
    facts) and partials an in-flight download still owns. Returns
    (removed, kept).
    """
    from pm.download_state import collect_partials
    from pm import paths

    partials_dir = paths.partials_root()
    if not store.root.is_dir() and not partials_dir.is_dir():
        return (0, 0)
    removed = 0
    with store.install_lock():
        facts.reload()
        keep = facts.entries_in_use()
        collect_partials(partials_dir)
        for item in sorted(store.root.iterdir()):
            if not item.is_dir():
                continue
            # Scratch dirs are created and removed under this same lock, so any
            # that remain belong to a killed installer. Other dot-dirs stay:
            # .previous-* is the restore point the next install of that entry
            # consumes, and it is only safe to drop after that verification.
            if item.name.startswith(".") and not item.name.startswith(".staging-"):
                continue
            if item.name in keep:
                continue
            print(f"removing {item.name}")
            shutil.rmtree(item, ignore_errors=True)
            removed += 1
    return (removed, len(keep))


def cmd_gc(args) -> int:
    from pm.paths import writable_store_root
    from pm.lock import Facts
    from pm.store import Store
    store = Store(writable_store_root())
    facts = _facts() if store.root == _store().root else Facts(store.root / "facts.json")
    removed, kept = _gc_store(store, facts)
    from hermes_cli.runtime_state import collect_generations
    from pm.environments import install_state_dir
    from pm.paths import repo_root
    from pm.runtime import collect_runtime_generations
    generations = collect_generations(repo_root())
    runtimes = collect_runtime_generations(install_state_dir(repo_root()) / "pm-runtime")
    print(f"gc: removed {removed}, kept {kept}; removed {len(generations)} dependency generations, "
          f"{len(runtimes)} PM runtime generations")
    return 0


def _run_live(cmd: list[str], *, cwd, env, timeout: int = 3600) -> tuple[int, str]:
    """Stream CI progress with the Python engine's bounded drain and diagnostic tail."""
    from pm.environment import _run_streaming

    try:
        result = _run_streaming(cmd, cwd=cwd, env=env, timeout=timeout, output=sys.stdout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{cmd[0]} timed out after {timeout}s") from exc
    return result.returncode, result.stderr


def _resolve_updates(names: list[str], lockfile, only_target: str | None) -> tuple[list, list[str]]:
    """Resolve each package's latest; an upstream outage records a failure instead of aborting."""
    resolved = []
    failures = []
    for name in names:
        package = get_package(name)
        targets = [t for t in ALL_TARGETS if package.missing_reason(t) is None]
        if only_target:  # cross-target check: only the requested target matters
            targets = [t for t in targets if t == only_target]
        if not targets:
            continue
        try:
            decision = resolve_package(package, targets, lockfile.version(name),
                                       artifacts=lockfile.pinned_artifacts(name))
        except Exception as e:  # an upstream index outage must not kill the whole check
            decision = Resolved(name, lockfile.version(name), package.version_style, reason=f"resolve failed: {e}")
            failures.append(name)
        resolved.append(decision)
    return resolved, failures


def _report_update(decision, width: int) -> None:
    d = decision
    if d.version is None:
        print(f"{d.name:<{width}}  {d.reason or 'up to date'}")
    elif not d.changed:
        print(f"{d.name:<{width}}  {d.locked} up to date")
    elif d.version == d.locked and d.artifact_updates:
        print(f"{d.name:<{width}}  {d.version}: newer artifacts for {', '.join(sorted(d.artifact_updates))}")
    else:
        per = ""
        if d.per_target and len(set(d.per_target.values())) > 1:
            per = " (" + ", ".join(f"{t}={v}" for t, v in sorted(d.per_target.items())) + ")"
        print(f"{d.name:<{width}}  {d.locked or '—'} → {d.version}{per}")


def _sync_venv_step() -> bool:
    from pm.install import sync_venv

    try:
        sync_venv(explicit=True)
    except InstallError as e:
        print(f"✗ {e}")
        return False
    print("✓ venv")
    return True


def _apply_pins(changed: list, lockfile) -> int:
    if not changed:
        print("pm update: nothing to update")
        return 0
    for d in changed:
        package = get_package(d.name)
        artifacts = _pin_artifacts(package, d, lockfile.pinned_artifacts(d.name))
        lockfile.set_pin(d.name, d.version, artifacts)
        print(f"✓ {d.name} pinned {d.locked or '—'} → {d.version}")
    lockfile.save()
    if _install_names([d.name for d in changed]):
        return 1
    return 0 if _sync_venv_step() else 1


def _refresh_uv_lock() -> int:
    try:
        lock_project(repo_root(), upgrade=True, explicit=True)
    except InstallError as exc:
        print(f"✗ Python lock refresh failed: {exc}")
        return 1
    print("✓ uv.lock refreshed")
    return 0 if _sync_venv_step() else 1


def _refresh_npm_lock() -> int:
    from pm.install import env_for, installed_package
    from pm.packages import npm_env
    from pm.paths import writable_store_root

    npm = installed_package("npm")
    node = installed_package("node")
    if npm is None or npm.binary is None or node is None or node.binary is None:
        print("✗ npm or Node: not installed; run `hermes pm install`")
        return 1
    env = npm_env(writable_store_root() / ".npm-cache", env_for("npm"))
    code, tail = _run_live([str(npm.binary), "update"], cwd=str(repo_root()), env=env)
    if code != 0:
        print(f"✗ npm update failed:\n{tail}")
        return 1
    print("✓ package-lock.json refreshed")
    return 0


def cmd_update(args) -> int:
    """`hermes pm update [names...] [--check] [--target T] [--uv] [--npm] [--termux]`.

    Resolve each package's latest via its own latest_versions() hook,
    intersect across targets, and (real mode) re-pin the lockfile + install
    the changed ones. --check is dry-run: hits upstream indexes, writes
    nothing. --uv / --npm also refresh uv.lock (+sync venv) / package-lock.
    --termux is its own repair pass: it repins only the pool archives the
    rolling termux-main pool has retired under our pins.
    """
    if args.termux:
        ignored = [name for name, given in (("names", args.names), ("--target", args.target),
                                            ("--uv", args.uv), ("--npm", args.npm)) if given]
        if ignored:
            print(f"::warning::--termux repairs the termux pool pins only; ignoring {', '.join(ignored)}")
        return _termux_pass(check=args.check)
    lockfile = _lockfile()
    names = args.names or [n for n in lockfile.names() if not get_package(n).internal or n == "uv"]
    if args.target and not args.check:
        print("::warning::--target is a CHECK-only cross-resolution flag; ignoring it for apply (the lockfile pins every target)")
        args.target = None

    resolved, failures = _resolve_updates(names, lockfile, args.target)
    changed = [d for d in resolved if d.changed]
    if not resolved:
        print("pm update: nothing to check (no resolvable packages)")
        return 0
    width = max(len(d.name) for d in resolved)
    for d in resolved:
        _report_update(d, width)
    if failures:
        print(f"pm update: resolution failed for {', '.join(failures)}; no changes applied")
        return 1
    if args.check:
        if args.uv:
            print("uv deps: would run `uv lock --upgrade` + venv sync")
        if args.npm:
            print("npm deps: would run `npm update`")
        return 1 if changed else 0

    if _apply_pins(changed, lockfile):
        return 1
    if args.uv and _refresh_uv_lock():
        return 1
    if args.npm:
        return _refresh_npm_lock()
    return 0


def _termux_pass(*, check: bool) -> int:
    """Repin the pool archives Termux has retired under our pins.

    The bionic lock rows and the runtime-lib table have no shared version axis
    to resolve (each pins what its own supplier ships), so this is a repair,
    not an update: only rows whose archive is gone are touched.
    """
    table = termux_libs.load_table()
    lockfile = _lockfile()
    total = len(termux_libs.pins(table, lockfile))
    stale = termux_libs.retired(table, lockfile, termux_libs.index())
    if not stale:
        print(f"termux pins: {total} rows still served by the pool")
        return 0
    unresolved = [entry for entry in stale if entry.replacement is None]
    width = max(len(entry.pin.name) for entry in stale)
    for entry in stale:
        if entry.replacement is None:
            print(f"{entry.pin.name:<{width}}  {entry.pin.version}: the pool no longer carries it")
        else:
            print(f"{entry.pin.name:<{width}}  {entry.pin.version} → {entry.replacement.version}")
    if check:
        return 1
    applied = termux_libs.repair(table, lockfile, stale)
    if applied:
        termux_libs.save_table(table)
        lockfile.save()
        print(f"✓ {applied} termux pins repinned; restage the bionic payload to pick them up")
    return 1 if unresolved else 0


@reuse_index_responses()
def _pin_artifacts(package, decision, current: dict) -> dict:
    """Retain unresolved targets and reuse hashes for unchanged artifact URLs."""
    per_target = decision.per_target or {t: decision.version for t in ALL_TARGETS}
    artifacts = dict(current)
    hashes: dict[str, str] = {}
    for target, version in per_target.items():
        if package.missing_reason(target) is not None:
            continue
        old = current.get(target, current.get("any", []))
        old = old if isinstance(old, list) else [old]
        known = {row["url"]: row["sha256"] for row in old}
        urls = decision.artifact_updates.get(target)
        if urls is None:
            urls = package.fetch_urls(version, target)
        if urls == [row["url"] for row in old]:
            continue
        pinned = []
        for url in urls:
            digest = known.get(url) or hashes.get(url)
            if not digest:
                digest = package.known_sha256(version, url) or hash_url(url)
                hashes[url] = digest
            pinned.append({"url": url, "sha256": digest})
        artifacts[target] = pinned[0] if len(pinned) == 1 else pinned
    return artifacts


def cmd_status(args) -> int:
    """Print the latest pm sync receipt — the reader surface for the
    CLI/TUI/desktop (same schema as update receipts; a failed venv
    rebuild or a plugin bisect is as reportable as a failed update)."""
    import json as _json

    from pm import receipt

    data = receipt.latest()
    if data is None:
        print("no pm sync receipt yet (no venv operation has run)")
        return 0
    print(_json.dumps(data, indent=2))
    return 0


def cmd_repair(args) -> int:
    from hermes_cli._early_recovery import recover_if_needed
    from pm.paths import repo_root

    if not recover_if_needed(repo_root(), explicit=True):
        return 1
    print("Restart Hermes to use the repaired dependency environment.")
    return 0


def cmd_bundle(args) -> int:
    from scripts.bundles.native import stage_native
    return stage_native(args)



def main(argv=None) -> int:
    # Windows consoles default to cp1252; pm prints ✓/✗. Never let the
    # status glyphs crash the command reporting them. line_buffering:
    # pm output must stream live in a piped (CI) log, not sit in a block
    # buffer and flush only at exit.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace", line_buffering=True)
        except (AttributeError, OSError):
            pass
    parser = argparse.ArgumentParser(prog="hermes pm")
    sub = parser.add_subparsers(dest="cmd", required=True)

    lock_parser = p = sub.add_parser(
        "lock", help="relock uv.lock from pyproject.toml (or --bump a tool pin in pm/lock.json)",
        description="With no arguments: re-resolve uv.lock from pyproject.toml; changes no "
                    "environment and writes nothing when the lock is current. With --bump NAME "
                    "VERSION: pin a pm tool's artifacts in pm/lock.json; uv.lock is untouched.")
    p.add_argument("--bump", dest="name", metavar="NAME",
                   help="pin tool NAME at VERSION in pm/lock.json instead of relocking uv.lock")
    p.add_argument("version", nargs="?", metavar="VERSION", help="the tool version (only with --bump)")
    p.set_defaults(func=cmd_lock)

    p = sub.add_parser("install", help="install packages (default: all required + optional defaults)")
    p.add_argument("names", nargs="*")
    p.add_argument("--extra", action="append", default=[], metavar="NAME",
                   help="enable a declared dependency extra in the venv (repeatable)")
    p.add_argument("--without", action="append", default=[], metavar="NAME",
                   help="leave an optional default package (agent-browser) out of this and every later "
                        "default install and update; `hermes pm install NAME` opts back in (repeatable)")
    p.add_argument("--tools-only", action="store_true",
                   help="install the tool closure, put it on PATH, and stop before the venv sync")
    p.add_argument("--trust-recorded", action="store_true",
                   help="trust the recorded tool digest instead of re-hashing every entry. "
                        "shell activation only. a deliberate install re-checks the bytes")
    p.add_argument("--test-environment", nargs="?", const="", default=None, metavar="EXTRAS",
                   help="also make this checkout's isolated test environment current (activation). "
                        "EXTRAS is comma-separated; omitted selects [all]")
    p.add_argument(
        "--target",
        help="stage for a cross target (e.g. linux-arm64-bionic on a glibc "
        "CI host); requires explicit package names",
    )
    p.set_defaults(func=cmd_install)

    p = sub.add_parser("env", help="print composed env of installed packages")
    p.add_argument("names", nargs="*")
    p.set_defaults(func=cmd_env)

    p = sub.add_parser("doctor", help="check installed state against the lockfile")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("repair", help="rebuild the recorded dependency environment without changing its graph")
    p.set_defaults(func=cmd_repair)

    p = sub.add_parser("gc", help="remove store entries nothing references")
    p.set_defaults(func=cmd_gc)

    p = sub.add_parser("bundle", help="stage a payload (repo+store+facts+relocatable venv) into --out")
    p.add_argument("--out", required=True)
    p.add_argument("--ref", help="git ref for the repo snapshot (default HEAD)")
    p.add_argument("--cache", type=Path, help="persistent build cache (default: UV_CACHE_DIR or PM's shared cache)")
    p.set_defaults(func=cmd_bundle)

    p = sub.add_parser("status", help="print the latest pm sync receipt (machine-readable)")
    p.set_defaults(func=cmd_status)
    p = sub.add_parser("update", help="resolve latest versions and re-pin the lockfile")
    p.add_argument("names", nargs="*", help="packages to check/update (default: all with a latest source)")
    p.add_argument("--check", action="store_true",
                   help="dry-run: print what would change, write nothing (exit 1 if updates exist)")
    p.add_argument("--target", help="resolve for a different target instead of this machine (e.g. win32-arm64)")
    p.add_argument("--uv", action="store_true", help="also refresh uv.lock + venv (uv update + sync)")
    p.add_argument("--npm", action="store_true", help="also refresh package-lock.json (npm update)")
    p.add_argument("--termux", action="store_true",
                   help="repin the termux pool archives the pool has retired (the runtime-lib "
                        "table + the bionic lock rows); --check reports without writing")
    p.set_defaults(func=cmd_update)

    args = parser.parse_args(argv)
    if args.cmd == "lock" and (args.name is None) != (args.version is None):
        lock_parser.error("--bump NAME and VERSION go together; run with neither to relock uv.lock")
    from pm.runtime import is_runtime, run_cli

    try:
        if not is_runtime():
            return run_cli(list(sys.argv[1:] if argv is None else argv))
        return args.func(args)
    except InstallError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
