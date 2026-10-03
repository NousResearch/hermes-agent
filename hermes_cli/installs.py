"""Find the other Hermes installs on this machine, and remove the ones that are safe to remove.

Several installs can share one ``HERMES_HOME``: a source checkout, a desktop
bundle, a packaged app. ``hermes uninstall`` acts on the install it runs from
and also on state that every install shares (User PATH, User environment
variables, the gateway, PortableGit and Node, ``<home>\\bin``). If you remove a
second install with it, the first install breaks.

This module removes only the tree of the install it is asked to remove, plus
the launchers and shortcuts that point into that tree. It never touches
``HERMES_HOME`` data, User PATH, User environment variables, or the gateway.

State lives in ``installs.json`` in the default Hermes root, so it does not
depend on the active profile. Each command records its own install there. That
is how an install learns about installs that discovery cannot find.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

logger = logging.getLogger(__name__)

REGISTRY_NAME = "installs.json"
_SCHEMA = 1
_TOUCH_INTERVAL_S = 3600
_POWERSHELL = ("powershell", "-NoProfile", "-NonInteractive", "-Command")
_POWERSHELL_TIMEOUT_S = 60
_APPX_REMOVE_TIMEOUT_S = 300
_GIT_STATUS_TIMEOUT_S = 30
_PACKAGE_FULL_NAME = re.compile(r"[A-Za-z0-9._~-]+")
# Anything these print is work that deleting the checkout would destroy. `HEAD --branches --not --remotes`
# is every local commit no remote branch has, including a detached HEAD.
_UNSAVED_WORK_PROBES = (
    (("status", "--porcelain"), "it has uncommitted changes. Commit or discard them first"),
    (("log", "-n1", "--format=%h", "HEAD", "--branches", "--not", "--remotes"),
     "it has commits that no remote has. Push or discard them first"),
    (("stash", "list"), "it has stashed changes. Apply or drop them first"),
)
_LAUNCHER_NAME = "hermes"
_LAUNCHER_TEXT_BYTES = 262144

_UTF8_OUTPUT = "[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false); "
_APPX_SCRIPT = _UTF8_OUTPUT + (
    "Get-AppxPackage -Name 'NousResearch*' | "
    "Select-Object PackageFullName,InstallLocation | ConvertTo-Json -Compress"
)
_SHORTCUTS_SCRIPT = _UTF8_OUTPUT + (
    "$shell = New-Object -ComObject WScript.Shell; "
    "@('Desktop','CommonDesktopDirectory','Programs','CommonPrograms') | "
    "ForEach-Object { [Environment]::GetFolderPath($_) } | "
    "Where-Object { $_ -and (Test-Path -LiteralPath $_) } | "
    "ForEach-Object { Get-ChildItem -LiteralPath $_ -Filter *.lnk -Recurse -ErrorAction SilentlyContinue } | "
    "ForEach-Object { [pscustomobject]@{ Path = $_.FullName; "
    "Target = $shell.CreateShortcut($_.FullName).TargetPath } } | ConvertTo-Json -Compress"
)

Runner = Callable[..., subprocess.CompletedProcess]


@dataclass(frozen=True)
class Install:
    """One Hermes install. ``steward`` uses the vocabulary of ``hermes_cli.steward``."""

    id: str
    root: Path
    steward: str
    sources: tuple[str, ...]
    current: bool = False
    package_full_name: str | None = None


@dataclass(frozen=True)
class Launcher:
    """A ``hermes`` launcher on PATH. ``owner`` is an install id, or None when no install claims it."""

    path: Path
    owner: str | None


@dataclass(frozen=True)
class RemovalPlan:
    """What removing one install does. ``action`` is ``tree``, ``appx``, or ``refuse``."""

    install: Install
    action: str
    steps: tuple[str, ...] = ()
    refusal: str | None = None
    tree: Path | None = None
    launchers: tuple[Path, ...] = ()
    shortcuts: tuple[Path, ...] = ()
    command: tuple[str, ...] | None = None


def _windows(flag: bool | None) -> bool:
    return sys.platform == "win32" if flag is None else flag


def current_root() -> Path:
    """The tree this process runs from."""
    from pm.paths import install_root

    return Path(install_root()).resolve()


def _default_root() -> Path:
    from hermes_constants import get_default_hermes_root

    return Path(get_default_hermes_root())


def install_id(root: Path | str) -> str:
    normal = os.path.normcase(str(Path(root).resolve()))
    return hashlib.sha256(normal.encode("utf-8")).hexdigest()[:12]


def fingerprint(others: Iterable[Install]) -> str:
    return hashlib.sha256("\n".join(sorted(i.id for i in others)).encode("utf-8")).hexdigest()[:12]


# --- registry ---------------------------------------------------------------


def registry_path() -> Path:
    return _default_root() / REGISTRY_NAME


def _empty_registry() -> dict:
    return {"schema": _SCHEMA, "seen": {}, "dismissed": ""}


def read_registry() -> dict:
    try:
        data = json.loads(registry_path().read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return _empty_registry()
    if not isinstance(data, dict) or data.get("schema") != _SCHEMA:
        return _empty_registry()
    seen = data.get("seen")
    dismissed = data.get("dismissed")
    return {
        "schema": _SCHEMA,
        "seen": seen if isinstance(seen, dict) else {},
        "dismissed": dismissed if isinstance(dismissed, str) else "",
    }


def write_registry(data: dict) -> None:
    target = registry_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    staging.write_text(json.dumps(data, indent=2), encoding="utf-8", newline="\n")
    os.replace(staging, target)


def record_launch(root: Path | None = None, *, now: float | None = None) -> None:
    """Record this install in the registry. Never raises: a launch must not fail on a notice file.

    Does nothing when the Hermes home does not exist yet, and for a linked git worktree: those are
    development trees, and recording them would list every worktree as a removable "other install".
    """
    import time

    try:
        if not registry_path().parent.is_dir():
            return
        resolved = Path(root if root is not None else current_root()).resolve()
        if (resolved / ".git").is_file():
            return
        moment = time.time() if now is None else now
        data = read_registry()
        key = install_id(resolved)
        entry = data["seen"].get(key)
        if (isinstance(entry, dict) and entry.get("root") == str(resolved)
                and moment - float(entry.get("last_launched", 0)) < _TOUCH_INTERVAL_S):
            return
        data["seen"][key] = {"root": str(resolved), "last_launched": int(moment)}
        write_registry(data)
    except Exception:
        logger.debug("could not record this install", exc_info=True)


# --- discovery --------------------------------------------------------------

Candidate = tuple[Path, str, "str | None"]


def _json_records(text: str) -> list[dict]:
    text = (text or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except ValueError:
        return []
    records = data if isinstance(data, list) else [data]
    return [r for r in records if isinstance(r, dict)]


def _home_checkout(default_root: Path) -> list[Candidate]:
    return [(default_root / "hermes-agent", "home", None)]


def _packaged_app_dirs() -> list[Path]:
    from hermes_cli.gui_uninstall import _env_dir, packaged_gui_app_paths

    dirs = list(packaged_gui_app_paths())
    if sys.platform == "win32":
        # Only discovery looks here. `packaged_gui_app_paths()` feeds `hermes uninstall --gui`,
        # which must not delete the folder of the other desktop product.
        local_base = _env_dir("LOCALAPPDATA", Path.home() / "AppData" / "Local")
        dirs.append(local_base / "Programs" / "HermesBundled")
    return [p for p in dirs if p.is_dir()]


def _registry_roots(registry: dict) -> list[Candidate]:
    roots = []
    for entry in registry["seen"].values():
        if isinstance(entry, dict) and isinstance(entry.get("root"), str):
            roots.append((Path(entry["root"]), "registry", None))
    return roots


def _bundle_repo_dirs(location: Path) -> list[Path]:
    """The agent tree inside an unpacked desktop package, read from the payload manifest."""
    from hermes_cli.bundled_app import PAYLOAD_DIR_NAME

    repos = []
    try:
        manifests = sorted(location.glob(f"*/resources/{PAYLOAD_DIR_NAME}/manifest.json"))
    except OSError:
        return repos
    for manifest in manifests:
        try:
            repo_dir = json.loads(manifest.read_text(encoding="utf-8-sig")).get("runtime", {}).get("repoDir")
        except (OSError, ValueError, AttributeError):
            continue
        if isinstance(repo_dir, str) and (manifest.parent / repo_dir).is_dir():
            repos.append(manifest.parent / repo_dir)
    return repos


def _appx_packages(run: Runner, windows: bool) -> list[Candidate]:
    if not windows:
        return []
    try:
        proc = run([*_POWERSHELL, _APPX_SCRIPT], capture_output=True, text=True,
                   encoding="utf-8", errors="replace",
                   stdin=subprocess.DEVNULL, timeout=_POWERSHELL_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("Get-AppxPackage failed: %s", exc)
        return []
    found: list[Candidate] = []
    for record in _json_records(proc.stdout):
        name, location = record.get("PackageFullName"), record.get("InstallLocation")
        if not isinstance(name, str) or not isinstance(location, str):
            continue
        repos = _bundle_repo_dirs(Path(location)) or [Path(location)]
        found.extend((repo, "appx", name) for repo in repos)
    return found


def _version_label(root: Path) -> str | None:
    """The version as the app shows it (``0.21.5+1913``). Only ``list`` pays for the git reads."""
    from hermes_cli.version_info import version_info_for

    info = version_info_for(root)
    if info.source == "commit-build" and info.commit:
        # A commit build carries pyproject's placeholder 0.0.0; the commit is its identity.
        return f"commit build {info.commit[:7]}"
    return None if info.source == "unknown" else info.display_version


def discover(
    *,
    current: Path | None = None,
    registry: dict | None = None,
    appx: bool = True,
    windows: bool | None = None,
    run: Runner = subprocess.run,
) -> list[Install]:
    """Every Hermes install this process can see, the running one first.

    ``appx=False`` skips the PowerShell query, which is too slow for launch.
    """
    running = Path(current if current is not None else current_root()).resolve()
    registry = read_registry() if registry is None else registry
    candidates: list[Candidate] = [(running, "running", None)]
    candidates += _home_checkout(_default_root())
    candidates += [(d, "packaged-app", None) for d in _packaged_app_dirs()]
    candidates += _registry_roots(registry)
    if appx:
        candidates += _appx_packages(run, _windows(windows))

    merged: dict[str, dict] = {}
    for root, source, package in candidates:
        try:
            root = root.resolve()
            if not root.is_dir():
                continue
        except OSError:
            continue
        slot = merged.setdefault(install_id(root), {"root": root, "sources": [], "package": None})
        if source not in slot["sources"]:
            slot["sources"].append(source)
        slot["package"] = slot["package"] or package

    from hermes_cli.steward import classify_install

    running_id = install_id(running)
    found = []
    for key, slot in merged.items():
        found.append(Install(
            id=key, root=slot["root"], steward=classify_install(slot["root"])[0],
            sources=tuple(slot["sources"]), current=key == running_id,
            package_full_name=slot["package"]))
    found.sort(key=lambda i: (not i.current, str(i.root).lower()))
    return found


def _others(current: Path | None) -> list[Install]:
    """The other installs the launch notice counts: the ones the user could actually remove.

    A sealed install (nix, a store package) is refused by ``remove``, so counting it would nag about
    something no command can fix.
    """
    from hermes_constants import get_hermes_home

    running = Path(current if current is not None else current_root()).resolve()
    found = discover(current=running, appx=False)
    home, default_root = Path(get_hermes_home()), _default_root()
    return [
        i for i in found
        if not i.current and classify_removal(
            i, current=running, hermes_home=home, default_root=default_root, found=found)[0] != "refuse"
    ]


# --- launchers on PATH ------------------------------------------------------


def _windows_path_entries() -> list[str]:
    import winreg

    entries: list[str] = []
    for hive, subkey in (
        (winreg.HKEY_CURRENT_USER, "Environment"),
        (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
    ):
        try:
            with winreg.OpenKey(hive, subkey) as key:
                value, _ = winreg.QueryValueEx(key, "Path")
        except OSError:
            continue
        entries.extend(os.path.expandvars(e) for e in str(value).split(";") if e)
    return entries


def _path_entries(windows: bool) -> list[str]:
    """PATH as a new terminal sees it: the registry on Windows, the environment elsewhere."""
    return _windows_path_entries() if windows else os.environ.get("PATH", "").split(os.pathsep)


def _launcher_owner(path: Path, found: Sequence[Install], windows: bool) -> str | None:
    resolved = path.resolve()
    for install in found:
        if resolved.is_relative_to(install.root):
            return install.id
    if windows and path.parent.name.lower() == "windowsapps" and path.parent.parent.name.lower() == "microsoft":
        packaged = [i for i in found if i.package_full_name]
        if len(packaged) == 1:
            return packaged[0].id
    try:
        text = path.read_bytes()[:_LAUNCHER_TEXT_BYTES].decode("latin-1")
    except OSError:
        return None
    text = text.lower() if windows else text
    for install in sorted(found, key=lambda i: len(str(i.root)), reverse=True):
        needle = str(install.root).lower() if windows else str(install.root)
        if needle in text:
            return install.id
    return None


def path_launchers(
    found: Sequence[Install],
    *,
    windows: bool | None = None,
    entries: Sequence[str] | None = None,
    pathext: Sequence[str] | None = None,
) -> list[Launcher]:
    """Every ``hermes`` launcher on PATH in the order a new terminal resolves them."""
    win = _windows(windows)
    if entries is None:
        entries = _path_entries(win)
    if pathext is None:
        pathext = [e.lower() for e in os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD").split(";") if e]
    launchers = []
    visited: set[str] = set()
    for entry in entries:
        directory = Path(entry)
        key = os.path.normcase(os.path.normpath(entry))
        if not entry or key in visited or not directory.is_dir():
            continue
        visited.add(key)
        names = [f"{_LAUNCHER_NAME}{ext}" for ext in pathext] if win else [_LAUNCHER_NAME]
        for name in names:
            candidate = directory / name
            if candidate.is_file() and (win or os.access(candidate, os.X_OK)):
                launchers.append(Launcher(candidate, _launcher_owner(candidate, found, win)))
                break
    return launchers


# --- removal ----------------------------------------------------------------


def _tree_guard(root: Path, current: Path, protected: Iterable[Path]) -> str | None:
    if not (root / "hermes_cli").is_dir():
        return "it is not a Hermes source tree"
    if current.resolve().is_relative_to(root):
        return "it contains the running install"
    for home in protected:
        if Path(home).resolve().is_relative_to(root):
            return "it contains HERMES_HOME"
    return None


def _work_at_risk(install: Install, found: Sequence[Install], run: Runner | None) -> str | None:
    """Why deleting this folder would destroy more than the install itself.

    A checkout can hold uncommitted work, commits no remote has, stashes, and other installs (git
    worktrees live inside it). A clean ``git status`` alone is not enough: ``.git`` goes with the
    folder, and so do local-only commits and stashes. The git probes run only when ``run`` is given:
    ``list`` and the launch notice skip them because they would pay three git calls per registered
    checkout. A probe that cannot run refuses, since this check guards an ``rmtree``.
    """
    for other in found:
        if other.id != install.id and other.root.is_relative_to(install.root):
            return f"it contains another install, {other.root}. Remove that one first"
    if run is None or not (install.root / ".git").exists():
        return None
    from hermes_cli.boot_bootstrap import _git_binary

    git = _git_binary()
    if git is None:
        return "git is not available to check it for unsaved work"
    for args, reason in _UNSAVED_WORK_PROBES:
        try:
            probe = run([git, "-C", str(install.root), *args], capture_output=True, text=True,
                        encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL, timeout=_GIT_STATUS_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("git %s failed for %s: %s", args[0], install.root, exc)
            return "it could not be checked for unsaved work"
        if probe.returncode != 0:
            return "it could not be checked for unsaved work"
        if probe.stdout.strip():
            return reason
    return None
def _is_elevated(windows: bool | None = None) -> bool:
    """True when this process runs as an administrator on Windows."""
    if not _windows(windows):
        return False
    import ctypes

    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except (AttributeError, OSError):
        return False


def classify_removal(
    install: Install, *, current: Path, hermes_home: Path, default_root: Path, windows: bool | None = None,
    all_users: bool = False, elevated: bool = False, found: Sequence[Install] = (), run: Runner | None = None,
) -> tuple[str, str | None]:
    """``(action, refusal)`` for removing ``install``. The action is ``tree``, ``appx``, or ``refuse``.

    ``found`` is every discovered install: a tree that holds another one is refused. ``run`` enables the
    uncommitted-changes probe (see ``_work_at_risk``).
    """
    from hermes_cli.steward import STEWARD_DESKTOP, steward_uninstall_message

    if install.id == install_id(current):
        return "refuse", "this is the running install"
    if install.steward == "git":
        reason = (_tree_guard(install.root, current, (hermes_home, default_root))
                  or _work_at_risk(install, found, run))
        return ("refuse", f"not removed: {reason}") if reason else ("tree", None)
    if install.steward == STEWARD_DESKTOP and install.package_full_name and _windows(windows):
        if not _PACKAGE_FULL_NAME.fullmatch(install.package_full_name):
            return "refuse", f"not removed: unsafe package name {install.package_full_name!r}"
        if all_users and not elevated:
            return "refuse", (
                "not removed: removing a package for all users needs an elevated terminal. "
                "Open PowerShell as Administrator and run this command again.")
        return "appx", None

    return "refuse", steward_uninstall_message(install.steward)


def _launcher_bin_dirs(default_root: Path) -> tuple[Path, ...]:
    """Where ``expose_cli`` publishes the ``hermes`` commands on Linux and macOS."""
    return (Path.home() / ".local" / "bin", default_root / "bin", Path("/usr/local/bin"))


def _owned_launchers(root: Path, default_root: Path, windows: bool) -> tuple[Path, ...]:
    """Launcher files that belong to the install at ``root``.

    On Windows, the managed clone owns the launchers in ``<default root>\\bin``. On Linux and macOS,
    a launcher belongs to ``root`` when ``_owns_launcher`` says so: a symlink into the tree, or a
    script that names the tree. A launcher that names another tree stays.
    """
    from hermes_cli._launchers import WINDOWS_BIN_LAUNCHERS, _owns_launcher

    if windows:
        if root.parent != default_root.resolve():
            return ()
        bin_dir = default_root / "bin"
        candidates = (bin_dir / f"{name}{ext}" for name in WINDOWS_BIN_LAUNCHERS for ext in (".exe", ".cmd"))
        return tuple(p for p in candidates if p.exists())
    names = (*WINDOWS_BIN_LAUNCHERS, "hermes-agent")
    candidates = (directory / name for directory in _launcher_bin_dirs(default_root) for name in names)
    owned = (p for p in candidates if (p.is_symlink() or p.exists()) and _owns_launcher(p, root))
    return tuple(dict.fromkeys(owned))


def _shortcuts_into(root: Path, run: Runner) -> tuple[Path, ...]:
    try:
        proc = run([*_POWERSHELL, _SHORTCUTS_SCRIPT], capture_output=True, text=True,
                   encoding="utf-8", errors="replace",
                   stdin=subprocess.DEVNULL, timeout=_POWERSHELL_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("shortcut scan failed: %s", exc)
        return ()
    shortcuts = []
    for record in _json_records(proc.stdout):
        target, path = record.get("Target"), record.get("Path")
        if isinstance(target, str) and target and isinstance(path, str) and Path(target).is_relative_to(root):
            shortcuts.append(Path(path))
    return tuple(shortcuts)


def plan_removal(
    install: Install,
    *,
    current: Path,
    hermes_home: Path,
    default_root: Path,
    windows: bool | None = None,
    run: Runner = subprocess.run,
    all_users: bool = False,
    elevated: bool | None = None,
    found: Sequence[Install] = (),
) -> RemovalPlan:
    win = _windows(windows)
    if elevated is None:
        elevated = all_users and _is_elevated(win)
    action, refusal = classify_removal(
        install, current=current, hermes_home=hermes_home, default_root=default_root, windows=win,
        all_users=all_users, elevated=elevated, found=found, run=run)
    if action == "refuse":
        return RemovalPlan(install, "refuse", refusal=refusal)
    if action == "appx":
        command = (*_POWERSHELL, f"Remove-AppxPackage -Package {install.package_full_name}"
                   + (" -AllUsers" if all_users else ""))
        scope = "all users" if all_users else "the current user"
        return RemovalPlan(
            install, "appx", command=command,
            steps=(f"Remove the package {install.package_full_name} for {scope}",
                   "Close the Hermes app first: removal stops it"))
    launchers = _owned_launchers(install.root, default_root, win)
    shortcuts = _shortcuts_into(install.root, run) if win else ()
    steps = [f"Remove the folder {install.root}"]
    steps += [f"Remove the launcher {p}" for p in launchers]
    steps += [f"Remove the shortcut {p}" for p in shortcuts]
    steps.append("Keep HERMES_HOME data, User PATH, User environment variables, the gateway, and other installs")
    return RemovalPlan(
        install, "tree", steps=tuple(steps), tree=install.root, launchers=launchers, shortcuts=shortcuts)


def execute_removal(plan: RemovalPlan, *, run: Runner = subprocess.run) -> int:
    """Run ``plan``. Returns 0 on success, 1 on any failure."""
    from hermes_cli.fs_utils import rmtree_force

    if plan.action == "tree":
        try:
            rmtree_force(plan.tree)
        except OSError as exc:
            print(f"Could not remove {plan.tree}: {exc}", file=sys.stderr)
            return 1
        failed = 0
        for leftover in (*plan.launchers, *plan.shortcuts):
            try:
                leftover.unlink(missing_ok=True)
            except OSError as exc:
                failed += 1
                print(f"Could not remove {leftover}: {exc}", file=sys.stderr)
        if failed:
            print(f"The folder is removed, but {failed} launcher or shortcut file(s) remain.", file=sys.stderr)
            return 1
        return 0
    if plan.action == "appx":
        try:
            proc = run([*plan.command[:-1], _UTF8_OUTPUT + plan.command[-1]], capture_output=True, text=True,
                       encoding="utf-8", errors="replace",
                       stdin=subprocess.DEVNULL, timeout=_APPX_REMOVE_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"Could not run PowerShell: {exc}", file=sys.stderr)
            return 1
        if proc.returncode != 0:
            print((proc.stderr or proc.stdout or "Remove-AppxPackage failed").strip(), file=sys.stderr)
            return 1
        return 0
    return 1


# --- launch notice ----------------------------------------------------------


def _notice_line(count: int, *, partial: bool) -> str:
    """The notice text. ``partial`` hedges the count: the launch path skips a query that can find more."""
    return (
        f"{'At least ' if partial else ''}{count} other Hermes install{'s' if count != 1 else ''} "
        "found on this machine. "
        f"Run `hermes installs` to see {'them' if count != 1 or partial else 'it'}, "
        "or `hermes installs dismiss` to hide this.")


def notice_state(*, current: Path | None = None) -> dict:
    """How many other installs the launch notice counts, and whether the user hid the notice.

    Uses only cheap sources, so it is safe at launch. ``list --json`` reports the same values.
    ``partial`` is True when the count can be low: ``hermes installs`` also lists Windows Store
    packages, which the launch path leaves out (``discover(appx=False)``).
    """
    others = _others(current)
    hidden = bool(others) and read_registry()["dismissed"] == fingerprint(others)
    return {"count": len(others), "dismissed": hidden, "partial": _windows(None)}


def launch_notice(*, current: Path | None = None) -> str | None:
    """One line about other installs, or None."""
    state = notice_state(current=current)
    count = state["count"]
    if not count or state["dismissed"]:
        return None
    return _notice_line(count, partial=state["partial"])


def dismiss(*, current: Path | None = None) -> int:
    """Hide the launch notice until the set of other installs changes. Returns how many others exist."""
    others = _others(current)
    data = read_registry()
    data["dismissed"] = fingerprint(others)
    write_registry(data)
    return len(others)


# --- command line -----------------------------------------------------------


def _install_json(
    install: Install, current: Path, hermes_home: Path, default_root: Path, found: Sequence[Install],
) -> dict:
    action, refusal = classify_removal(
        install, current=current, hermes_home=hermes_home, default_root=default_root, found=found)
    return {
        "id": install.id,
        "root": str(install.root),
        "steward": install.steward,
        "version": _version_label(install.root),
        "sources": list(install.sources),
        "current": install.current,
        "package_full_name": install.package_full_name,
        "removable": action != "refuse",
        "action": action,
        "refusal": refusal,
    }


def _print_list(entries: list[dict], launchers: list[Launcher]) -> None:
    print("Hermes installs on this machine")
    print()
    print(f"  {'':1} {'ID':12}  {'STEWARD':12}  {'VERSION':10}  {'REMOVABLE':9}  PATH")
    for e in entries:
        mark = "*" if e["current"] else " "
        print(f"  {mark} {e['id']:12}  {e['steward']:12}  {e['version'] or '-':10}  "
              f"{'yes' if e['removable'] else 'no':9}  {e['root']}")
    print()
    print("  * is the running install. Remove another one with `hermes installs remove <id>`.")
    print()
    if not launchers:
        print("No `hermes` launcher was found on PATH.")
        return
    print("`hermes` on PATH, in the order a new terminal finds them (the first one wins):")
    for position, launcher in enumerate(launchers, 1):
        print(f"  {position}. {launcher.path}  ->  {launcher.owner or 'no known install'}")


def _confirm() -> bool:
    if not sys.stdin.isatty():
        print("This needs a terminal to confirm. Pass --yes to skip the question.", file=sys.stderr)
        return False
    try:
        return input("Type 'yes' to continue: ").strip() == "yes"
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def _cmd_remove(args, found: list[Install], current: Path, default_root: Path, run: Runner) -> int:
    from hermes_constants import get_hermes_home

    wanted = args.install_id or ""
    matches = [i for i in found if wanted and i.id.startswith(wanted)]
    if len(matches) != 1:
        print(f"No single install matches {wanted!r}. Run `hermes installs` to see the ids.", file=sys.stderr)
        return 1
    plan = plan_removal(
        matches[0], current=current, hermes_home=Path(get_hermes_home()), default_root=default_root, run=run,
        all_users=bool(getattr(args, "all_users", False)), found=found)
    print(f"Install {plan.install.id}  {plan.install.root}")
    if plan.action == "refuse":
        print(plan.refusal, file=sys.stderr)
        return 1
    if getattr(args, "all_users", False) and plan.action != "appx":
        print("--all-users applies only to Windows package installs.", file=sys.stderr)
        return 1
    print("Will do:")
    for step in plan.steps:
        print(f"  - {step}")
    if args.dry_run:
        print("Dry run: nothing changed.")
        return 0
    if not args.yes and not _confirm():
        return 1
    if execute_removal(plan, run=run) != 0:
        return 1
    print("Removed.")
    remaining = path_launchers(discover(current=current, appx=False))
    if remaining:
        print(f"A new terminal now runs `hermes` from {remaining[0].path}.")
    return 0


def run_cli(args, *, run: Runner = subprocess.run) -> int:
    command = getattr(args, "installs_command", None) or "list"
    if command == "dismiss":
        count = dismiss()
        print(f"Hidden until the set of other installs changes ({count} other install{'s' if count != 1 else ''}).")
        return 0
    current = current_root()
    default_root = _default_root()
    found = discover(current=current, run=run)
    if command == "remove":
        return _cmd_remove(args, found, current, default_root, run)
    if command != "list":
        print(f"Unknown installs command: {command}", file=sys.stderr)
        return 2
    from hermes_constants import get_hermes_home

    home = Path(get_hermes_home())
    entries = [_install_json(i, current, home, default_root, found) for i in found]
    launchers = path_launchers(found)
    if getattr(args, "json", False):
        print(json.dumps({
            "current": install_id(current),
            "installs": entries,
            "launchers": [{"path": str(lnch.path), "owner": lnch.owner} for lnch in launchers],
            "notice": notice_state(current=current),
        }, indent=2))
    else:
        _print_list(entries, launchers)
    return 0
