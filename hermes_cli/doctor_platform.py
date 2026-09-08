"""Host-platform checks for hermes doctor: interpreter, SQLite, certificates, macOS TCC, gateway supervision, command install.
Split out of ``hermes_cli/doctor.py``, which re-exports every name so ``hermes_cli.doctor.<name>`` keeps resolving (and monkeypatching)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from hermes_cli.colors import Colors, color
from hermes_cli.config import is_nix_install_method, recommended_update_command_for_method
from hermes_cli.doctor_report import (
    Finding, _fail_and_issue, _section, check_bool, check_fail, check_info, check_ok, check_warn, doctor_check,
    warn_on_error,
)
from hermes_constants import is_termux as _is_termux


def _python_install_cmd() -> str:
    return "python -m pip install" if _is_termux() else "uv pip install"


def _system_package_install_cmd(pkg: str) -> str:
    return f"{'pkg' if _is_termux() else 'brew' if sys.platform == 'darwin' else 'sudo apt'} install {pkg}"


def _sqlite_upgrade_hint(install_method: str | None = None) -> str:
    """Return an actionable SQLite upgrade hint for this install layout."""
    from hermes_cli.doctor import PROJECT_ROOT
    from hermes_cli.config import detect_install_method
    method = install_method or detect_install_method(PROJECT_ROOT)
    cmd = recommended_update_command_for_method(method)
    action = cmd if is_nix_install_method(method) else {  # nix: prose guidance, not a shell command
        "docker": f"run `{cmd}`, then recreate all Hermes containers", "apt": f"run `{cmd}`"}.get(method, "run `hermes update`")
    return f"({action}; fixed versions: 3.51.3+ / 3.50.7 / 3.44.6 — see https://sqlite.org/wal.html#walresetbug)"


def _hermes_database_paths(hermes_home: Path) -> list[tuple[str, Path]]:
    """(display name, path) pairs for Hermes-managed SQLite databases: backup.py's per-profile store list + per-board kanban.db."""
    from hermes_cli.backup import _QUICK_STATE_FILES
    entries = [(name, hermes_home / name) for name in _QUICK_STATE_FILES if name.endswith(".db")]
    for board_db in sorted((hermes_home / "kanban" / "boards").glob("*/kanban.db")):
        entries.append((str(board_db.relative_to(hermes_home)), board_db))
    return entries


_SQLITE_HEADER_MAGIC = b"SQLite format 3\x00"


def _unreadable_reason(db_path: Path) -> str:
    """Explain why a database file could not be read, without opening it.

    ``read_header_bytes_preopen`` collapses every ``OSError`` into ``None``, but doctor must say *which*
    problem it hit. ``stat()``/``access()`` answer from directory metadata alone — no descriptor, no lock loss.
    """
    try:
        db_path.stat()
    except OSError as exc:
        return str(exc)
    return "file could not be read" if os.access(db_path, os.R_OK) else f"permission denied: {db_path}"


def _read_journal_mode(db_path: Path) -> tuple[str | None, str | None]:
    """Return (journal mode, error) from header byte 18 (2=WAL, 1=rollback) without opening the database.

    Opening through SQLite — even read-only — creates -wal/-shm sidecars, which a diagnostic must not do.
    ``read_header_bytes_preopen`` rather than a bare ``open()``: closing *any* descriptor cancels this
    process's POSIX advisory locks (see ``hermes_cli.sqlite_safe_read``), and the dashboard console runs
    ``run_doctor`` in-process with live ``SessionDB`` connections — the helper refuses then (unreadable).
    """
    from hermes_cli.sqlite_safe_read import has_live_connection, read_header_bytes_preopen
    header = read_header_bytes_preopen(db_path, length=20)
    if header is None:
        return None, "database is open in this process" if has_live_connection(db_path) else _unreadable_reason(db_path)
    if len(header) == 0:
        return None, "file is empty"
    if len(header) < 20 or not header.startswith(_SQLITE_HEADER_MAGIC):
        return None, "file is not a database"
    mode = {2: "wal", 1: "rollback"}.get(header[18])
    return (mode, None) if mode else (None, f"unrecognized file-format version {header[18]}")


def _format_db_size(db_path: Path) -> str:
    from hermes_cli.sizefmt import format_bytes as _format_size
    try:
        return _format_size(db_path.stat().st_size)
    except OSError:
        return "size unknown"


def _report_database_journal_modes(hermes_home: Path | None = None, version_info: tuple[int, ...] | None = None) -> None:
    """List each database's journal mode; warn on WAL under a vulnerable SQLite."""
    from hermes_cli.doctor import HERMES_HOME
    from hermes_state_wal import _wal_reset_repair_hint, is_sqlite_wal_reset_vulnerable
    vulnerable = is_sqlite_wal_reset_vulnerable(version_info)
    try:
        databases = _hermes_database_paths(hermes_home if hermes_home is not None else HERMES_HOME)
    except Exception as exc:
        check_warn(f"Could not list Hermes databases: {exc}")
        return
    exposed = []
    for name, path in databases:
        if not path.is_file():
            continue
        mode, error = _read_journal_mode(path)
        size = _format_db_size(path)
        if error is not None:
            if vulnerable:
                check_warn(f"{name}: journal mode could not be read", f"({error}; cannot rule out WAL exposure)")
            else:
                check_info(f"{name}: journal mode could not be read ({error})")
        elif mode == "wal" and vulnerable:
            exposed.append(name)
            check_warn(f"{name} is in WAL mode ({size})", "(exposed to the WAL-reset bug until SQLite is upgraded)")
        elif mode == "wal":
            check_info(f"{name}: WAL journal mode ({size})")
        else:
            check_info(f"{name}: rollback journal mode ({size}{', not exposed' if vulnerable else ''})")
    if exposed:
        check_info(f"To clear the exposure: {_wal_reset_repair_hint()}")


def _read_pyproject_version() -> str | None:
    """Read the ``[project]`` version from pyproject.toml; None for installed wheels (no pyproject) or unreadable files."""
    from hermes_cli.doctor import PROJECT_ROOT
    try:
        text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return None
    in_project = False
    for line in map(str.strip, text.splitlines()):
        if line.startswith("[") and line.endswith("]"):
            in_project = line == "[project]"
        elif in_project and line.startswith("version") and "=" in line:
            return line.split("=", 1)[1].split("#", 1)[0].strip().strip("\"'") or None
    return None


def _check_version_consistency(issues: list[str]) -> None:
    """Detect pyproject.toml vs hermes_cli.__version__ drift (a conflict resolution can revert one but not the
    other). Silent no-op for installed wheels (no pyproject)."""
    try:
        from hermes_cli import __version__ as init_version
    except Exception:
        return
    pyproject_version = _read_pyproject_version()
    if pyproject_version is None:
        return
    if pyproject_version == init_version:
        return check_ok("Version files consistent", f"({init_version})")
    _fail_and_issue("Version mismatch between source files", f"(pyproject.toml {pyproject_version} != hermes_cli/__init__.py {init_version})",
                    "Re-sync version files (e.g. run 'hermes update', or set hermes_cli/__init__.py __version__ to match pyproject.toml)", issues)


def _check_s6_supervision(issues: list[str]) -> None:
    """Under our s6 /init, report static services and per-profile gateway slots that are ``up``; no-op elsewhere.
    Counterpart to :func:`_check_gateway_service_linger` (systemd-on-host)."""
    try:
        from hermes_cli.service_manager import S6ServiceManager, detect_service_manager
    except Exception:
        return
    if detect_service_manager() != "s6":
        return
    _section("s6 Supervision")
    mgr = S6ServiceManager()
    for static in ("main-hermes", "dashboard"):  # s6-rc symlinks under /run/service/, same s6-svstat probe
        up = mgr.is_running(static)
        (check_ok if up else check_info)(f"{static}: up" if up else f"{static}: down (expected if not enabled via env)")
    profiles = mgr.list_profile_gateways()
    if not profiles:
        return check_info("No per-profile gateways registered yet — create one with `hermes profile create <name>`")
    up_count = sum(1 for p in profiles if mgr.is_running(f"gateway-{p}"))
    check_ok(f"Per-profile gateways: {up_count}/{len(profiles)} supervised up"
             + (f" ({', '.join(sorted(profiles))})" if len(profiles) <= 8 else ""))


def check_certificates(should_fix: bool = False, issues: "list | None" = None) -> None:
    """Verify the certifi CA bundle is loadable before the first HTTPS call tracebacks.

    ``--fix`` repairs a broken bundle (e.g. a brew Python upgrade rebuilt the venv) by force-reinstalling
    certifi into THIS interpreter's environment and re-verifying.
    """
    try:
        from agent.ssl_guard import verify_ca_bundle
        from agent.errors import SSLConfigurationError
    except Exception as e:
        return check_warn("SSL certificate check skipped", str(e))
    if issues is None:
        issues = []
    try:
        verify_ca_bundle()
        return check_ok("SSL CA certificate bundle is valid")
    except SSLConfigurationError as e:
        first_error = str(e)
    except Exception as e:
        return check_warn("SSL certificate check skipped", str(e))
    check_fail("SSL CA certificate bundle is broken", first_error)
    pip_cmd = f"{sys.executable} -m pip install --force-reinstall certifi"
    if not should_fix:
        issues.append(f"Repair the CA bundle: run `hermes doctor --fix`, or `{pip_cmd}`")
        return
    print("    → Repairing: force-reinstalling certifi...")
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "install", "--force-reinstall", "certifi"],
                                capture_output=True, text=True, timeout=300)
        failure = ("certifi reinstall failed", (result.stderr or result.stdout or "")[-500:]) if result.returncode != 0 else None
    except Exception as exc:
        failure = ("certifi repair could not run pip", str(exc))
    if failure:
        return _fail_and_issue(*failure, f"Reinstall certifi manually: {pip_cmd}", issues)
    # Drop cached certifi modules so where() resolves the fresh install without a restart.
    import importlib
    for mod_name in [m for m in sys.modules if m == "certifi" or m.startswith("certifi.")]:
        sys.modules.pop(mod_name, None)
    importlib.invalidate_caches()
    try:
        verify_ca_bundle()
        check_ok("SSL CA certificate bundle repaired (certifi reinstalled)")
    except SSLConfigurationError as e:
        _fail_and_issue("SSL CA certificate bundle still broken after reinstall", str(e),
                        "certifi reinstall did not restore the CA bundle — check for a custom CA env var "
                        "(SSL_CERT_FILE/REQUESTS_CA_BUNDLE) pointing at a missing file, or recreate the venv.", issues)


def _check_gateway_service_linger(issues: list[str]) -> None:
    """Warn when a systemd user gateway service will stop after logout (skipped under s6: no linger concept)."""
    try:
        from hermes_cli.gateway import get_systemd_linger_status, get_systemd_unit_path, is_linux
        from hermes_cli.service_manager import detect_service_manager
    except Exception as e:
        return check_warn("Gateway service linger", f"(could not import gateway helpers: {e})")
    if not is_linux() or detect_service_manager() == "s6" or not get_systemd_unit_path().exists():
        return
    _section("Gateway Service")
    linger_enabled, linger_detail = get_systemd_linger_status()
    if linger_enabled is None:
        return check_warn("Could not verify systemd linger", f"({linger_detail})")
    if not check_bool(linger_enabled, ("Systemd linger enabled", "(gateway service survives logout)"),
                      ("Systemd linger disabled", "(gateway may stop after logout)")):
        check_info("Run: sudo loginctl enable-linger $USER")
        issues.append("Enable linger for the gateway user service: sudo loginctl enable-linger $USER")


_TCC_CDHASH_DETAIL = (
    "the desktop bundle's designated requirement is cdhash-pinned (pre-#73681 build) — rebuilds invalidate "
    "all permission grants. Run `hermes update` to get the stable identifier-pinned signing identity, "
    "then re-grant permissions once.")
_TCC_STABLE_DETAIL = {
    True: "(certificate-anchored DR; grants survive rebuilds)",
    False: "(identifier-pinned DR; grants survive rebuilds — for the strongest anchor, see `hermes desktop --setup-tcc-identity`)",
}


def check_macos_tcc_grants() -> None:
    """Check macOS TCC grant persistence for a locally-built desktop bundle; silent on non-macOS / no bundle.

    TCC keys grants to the app's designated requirement (DR). A cdhash-pinned ad-hoc DR changes on every
    rebuild, so grants silently stop matching while the Settings toggle stays ON; identifier-pinned builds
    survive rebuilds, but grants made to older binaries stay stale until re-granted once. TCC.db needs Full
    Disk Access, so the DR string is the only readable signal (a proxy for the signing class, not DR wording).

    See #86385.
    """
    app = _desktop_app_bundle() if sys.platform == "darwin" else None
    if app is None:
        return
    dr = _macos_desktop_dr(app)
    if not dr:
        return check_warn("macOS TCC grant check", "(could not read code-signing requirement of the desktop bundle)")
    if "cdhash" in dr.lower():
        return check_warn("macOS TCC grants will reset after every update", _TCC_CDHASH_DETAIL)
    # --setup-tcc-identity or notarized build (certificate-anchored) is the strongest anchor.
    check_ok("macOS TCC signing identity is stable", _TCC_STABLE_DETAIL["certificate" in dr.lower()])
    check_info("If macOS still re-prompts for permissions (toggle shows ON): the stored grant is stale — run "
               "`tccutil reset ScreenCapture com.nousresearch.hermes` (repeat per affected service), toggle it ON in "
               "System Settings, then fully quit & relaunch Hermes once.")


def _desktop_app_bundle() -> Path | None:
    """Locate the locally-built desktop bundle (``apps/desktop/release/mac-<arch>/Hermes.app``), newest first.

    The only layout whose ad-hoc re-signed bundle can invalidate TCC grants. ``/Applications/Hermes.app`` is
    deliberately not probed: it is the separately-signed, certificate-anchored Hermes-Setup launcher.
    """
    release_dir = Path(__file__).resolve().parents[1] / "apps" / "desktop" / "release"
    candidates = [p for p in release_dir.glob("mac*/Hermes.app") if p.is_dir()]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def _macos_desktop_dr(app: Path) -> str | None:
    """Return the bundle's designated requirement string, or None on failure (a hanging codesign must never abort doctor)."""
    codesign = shutil.which("codesign")
    try:
        proc = subprocess.run([codesign, "-d", "--requirements", "-", str(app)], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15) if codesign else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return None if proc is None or proc.returncode != 0 else (proc.stdout or "") + (proc.stderr or "")


def check_macos_tcc_anchor(should_fix: bool = False) -> None:
    """Report (and with --fix install) the dylib-complete TCC anchor; silent on non-macOS / non-uv interpreters.
    Never raises. Install is gated by the module's pre-install boot probe, so ``--fix`` cannot brick the CLI.

    See #95596.
    """
    with warn_on_error("macOS TCC anchor check failed"):
        from hermes_cli import macos_tcc_anchor as tcc
        status, detail = tcc.tcc_anchor_state()
        if status == "skip":
            return
        if status == "active":
            return check_ok("macOS TCC anchor active", f"({detail})")
        anchored = tcc.ensure_tcc_anchor() if should_fix else None
        if anchored is not None:
            return check_ok("macOS TCC anchor installed", f"({anchored})")
        check_warn("macOS TCC anchor missing" if status == "missing" else "macOS TCC anchor stale", f"({detail})")


def check_macos_full_disk_access() -> None:
    """One-grant guidance: Full Disk Access silences every per-folder TCC prompt. Silent on non-macOS.

    Probe: listdir of ``~/Library/Application Support/com.apple.TCC`` — FDA-gated, and probing it does NOT
    trigger a prompt (the TCC dir just returns EPERM). A missing dir / other error is indeterminate: stay silent.

    macOS TCC prompts per-category (Desktop, then Downloads, then Documents, ...), so first-run agents
    drip-feed permission dialogs as they touch each folder. ONE Full Disk Access grant covers all of them,
    permanently — and with the stable signing identities now in place (#73681/#95091/#95131), it survives
    updates too. This check probes whether the terminal context already has FDA and, when it doesn't, prints
    the exact one-switch setup with the System Settings deep link.
    """
    if sys.platform != "darwin":
        return
    try:
        os.listdir(Path.home() / "Library" / "Application Support" / "com.apple.TCC")
    except PermissionError:
        check_info("One switch silences all macOS folder prompts: grant your terminal app Full Disk Access and Hermes "
                   "will never trip per-folder dialogs (Desktop/Downloads/Documents/...) again. Open: System Settings → "
                   "Privacy & Security → Full Disk Access — or run:\n"
                   "      open \"x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles\"\n"
                   "    then enable your terminal (and Hermes.app if you use Desktop), and restart them once. "
                   "With Hermes' stable signing identities the grant survives every update.")
    except OSError:
        pass  # missing dir / other error: indeterminate, stay silent
    else:
        check_ok("macOS Full Disk Access granted", "(no per-folder permission prompts will occur)")


@doctor_check("Security advisory check failed: {e}")
def _check_security_advisories(should_fix: bool, f: Finding) -> None:
    """Compromised-package advisories, funnelled into manual issues; a bug here must never block the rest of doctor."""
    from hermes_cli.security_advisories import detect_compromised, filter_unacked, full_remediation_text, get_acked_ids
    all_hits = detect_compromised()
    fresh_hits = filter_unacked(all_hits)
    if not fresh_hits:
        return check_ok("No active security advisories")
    for hit in fresh_hits:
        # Fail row + remediation text indented under it as one section; also into the summary action list.
        _fail_and_issue(f"{hit.advisory.title}", f"({hit.package}=={hit.installed_version})",
                        f"Resolve security advisory {hit.advisory.id}: uninstall {hit.package}=={hit.installed_version} "
                        f"and rotate credentials, then run `hermes doctor --ack {hit.advisory.id}`.", f.manual_issues)
        for line in full_remediation_text(hit):
            print(f"    {color(line, Colors.YELLOW)}" if line else "")
    acked_ids = get_acked_ids()  # acked-but-still-installed stays visible
    for h in all_hits:
        if h.advisory.id in acked_ids:
            check_warn(f"{h.package}=={h.installed_version} still installed (advisory {h.advisory.id} acknowledged)")


@doctor_check()
def _check_python_environment(should_fix: bool, f: Finding) -> None:
    """Interpreter, linked SQLite, venv, macOS TCC anchors/FDA/grants, version-file drift."""
    v, label = sys.version_info, f"Python {'.'.join(map(str, sys.version_info[:3]))}"
    if v < (3, 8):
        _fail_and_issue(label, "(3.10+ required)", "Upgrade Python to 3.10+", f.issues)
    elif check_bool(v >= (3, 10), label, (label, "(3.10+ recommended)")) and v < (3, 11):
        check_warn("Python 3.11+ recommended for RL Training tools (tinker requires >= 3.11)")
    # Linked SQLite: version + source id matter independently of the Python minor (uv's
    # python-build-standalone can keep a vulnerable SQLite across upgrades).
    with warn_on_error("SQLite version probe failed: {e}", ""):
        import sqlite3
        from hermes_state_wal import is_sqlite_wal_reset_vulnerable, sqlite_source_id
        src = sqlite_source_id()
        # Warn-only: Hermes already refuses WAL on fresh DBs and runtime repair is best-effort.
        check_bool(not is_sqlite_wal_reset_vulnerable(), f"SQLite {sqlite3.sqlite_version}",
                   (f"SQLite {sqlite3.sqlite_version} (WAL-reset bug)", _sqlite_upgrade_hint()))
        if src:
            check_info(f"SQLite source id: {(src[:48] + '…') if len(src) > 48 else src}")
        _report_database_journal_modes()
    check_bool(sys.prefix != sys.base_prefix, "Virtual environment active", ("Not in virtual environment", "(recommended)"))
    # macOS TCC interpreter anchor (#95596): dylib-complete re-land of the mechanism reverted in #95563.
    # Silent on non-macOS.
    check_macos_tcc_anchor(should_fix=should_fix)
    # macOS Full Disk Access (issue #52010 follow-up): one grant silences every per-folder prompt
    # permanently. Silent on non-macOS.
    check_macos_full_disk_access()
    _check_version_consistency(f.issues)
    # macOS TCC grant persistence (issue #86385): a locally-built desktop bundle whose DR is cdhash-pinned
    # loses every permission grant on each rebuild; a post-#73681 identifier-pinned DR survives, but grants
    # made to older binaries stay stale (toggle shows ON while macOS re-prompts).
    check_macos_tcc_grants()


@doctor_check()
def _check_certificates(should_fix: bool, f: Finding) -> None:
    check_certificates(should_fix=should_fix, issues=f.manual_issues)


# (import name, display name, optional)
_PACKAGES = (
    ("openai", "OpenAI SDK", False), ("rich", "Rich (terminal UI)", False), ("dotenv", "python-dotenv", False),
    ("yaml", "PyYAML", False), ("httpx", "HTTPX", False),
    ("croniter", "Croniter (cron expressions)", True), ("telegram", "python-telegram-bot", True), ("discord", "discord.py", True),
)


@doctor_check()
def _check_required_packages(should_fix: bool, f: Finding) -> None:
    for module, name, optional in _PACKAGES:
        try:
            __import__(module)
            check_ok(name, "(optional)" if optional else "")
        except ImportError:
            if optional:
                check_warn(name, "(optional, not installed)")
            else:
                _fail_and_issue(name, "(missing)", f"Install {name}: {_python_install_cmd()} {module}", f.issues)


@doctor_check()
def _check_gateway_supervision(should_fix: bool, f: Finding) -> None:
    _check_gateway_service_linger(f.issues)
    _check_s6_supervision(f.issues)


@doctor_check()
def _check_command_installation(should_fix: bool, f: Finding) -> None:
    """Venv entry point and the ~/.local/bin (or $PREFIX/bin) symlink; skipped on Windows."""
    from hermes_cli.doctor import PROJECT_ROOT
    if sys.platform == "win32":
        return
    _section("Command Installation")
    venv_bin = next((c for c in (PROJECT_ROOT / n / "bin" / "hermes" for n in ("venv", ".venv")) if c.exists()), None)
    if venv_bin is None:
        check_warn("Venv entry point not found", "(hermes not in venv/bin/ or .venv/bin/ — reinstall with pip install -e '.[all]')")
        return f.manual_issues.append(f"Reinstall entry point: cd {PROJECT_ROOT} && source venv/bin/activate && pip install -e '.[all]'")
    check_ok(f"Venv entry point exists ({venv_bin.relative_to(PROJECT_ROOT)})")
    # Expected command link directory (mirrors install.sh logic).
    prefix = os.environ.get("PREFIX", "")
    termux = prefix and (os.environ.get("TERMUX_VERSION") or "com.termux/files/usr" in prefix)
    link_dir, display = (Path(prefix) / "bin", "$PREFIX/bin") if termux else (Path.home() / ".local" / "bin", "~/.local/bin")
    link = link_dir / "hermes"
    if link.is_symlink():
        target, expected = link.resolve(), venv_bin.resolve()
        if target == expected:
            return check_ok(f"{display}/hermes → correct target")
        check_warn(f"{display}/hermes points to wrong target", f"(→ {target}, expected → {expected})")
        if not should_fix:
            return f.issues.append(f"Broken symlink at {display}/hermes — run 'hermes doctor --fix'")
        link.unlink()
        verb = "Fixed"
    elif link.exists():  # regular file (wrapper script), not a symlink
        return check_ok(f"{display}/hermes exists (non-symlink)")
    else:
        check_fail(f"{display}/hermes not found", "(hermes command may not work outside the venv)")
        if not should_fix:
            return f.issues.append(f"Missing {display}/hermes symlink — run 'hermes doctor --fix'")
        link_dir.mkdir(parents=True, exist_ok=True)
        verb = "Created"
    link.symlink_to(venv_bin)
    check_ok(f"{verb} symlink: {display}/hermes → {venv_bin}")
    f.fixed += 1
    if verb == "Created" and str(link_dir) not in os.environ.get("PATH", "").split(os.pathsep):
        check_warn(f"{display} is not on your PATH", "(add it to your shell config: export PATH=\"$HOME/.local/bin:$PATH\")")
        f.manual_issues.append(f"Add {display} to your PATH")
