"""Profiles whose backend a plugin runs somewhere other than this host.

A host-side plugin that runs a profile's whole backend elsewhere (for example inside a sandbox)
marks the profile for as long as it does. While a profile is marked, nothing on this host runs it,
whether or not the plugin's own process is running:

- ``AIAgent`` refuses to build an agent for it, whichever frontend asks;
- CLI commands that run a turn (chat, ``acp``, ``cron run``/``tick``, ...) exit 1 and
  ``hermes serve``/``dashboard`` exit 78 before they start, and the gateway process exits 78;
- ``profiles_to_serve()`` leaves it out, so the host gateway (platforms, cron, Kanban watchers,
  ``/p/<profile>/`` routes), the Desktop cron ticker and the Kanban dispatcher skip it;
- ``hermes gateway ensure`` refuses it (``profile_runs_elsewhere``), and the dashboard's cron,
  session, file and Webapp requests for it answer 409.

The marker is the file ``<hermes root>/profiles_run_elsewhere/<profile>.json``. It lives under the
Hermes root and never inside a profile home, so the profile's backend cannot remove it. The file's
presence is the marker; its contents (``{"plugin": ..., "since": ...}``) only name the owner in
messages, so a marker that cannot be read still marks the profile.

The plugin keeps this folder from the backend it runs elsewhere. A process that is denied the
folder (permission error) is therefore that backend, not the host, and refuses nothing: it keeps
running its own profile. Any other error reading the folder refuses every profile.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from hermes_constants import PROFILE_ID_RE

logger = logging.getLogger(__name__)

RUN_ELSEWHERE_DIRNAME = "profiles_run_elsewhere"
_SUFFIX = ".json"
_logged: set[str] = set()


@dataclass(frozen=True)
class RunElsewhere:
    """Why this host does not run ``profile``.

    ``plugin`` is None when the marker could not be read. ``folder_error`` is set when the marker
    folder itself could not be read, which refuses every profile.
    """

    profile: str
    path: Path
    plugin: str | None = None
    since: str | None = None
    folder_error: str | None = None

    def message(self) -> str:
        if self.folder_error is not None:
            return (f"Hermes cannot read {self.path.parent} ({self.folder_error}), so it cannot tell "
                    f"whether a plugin runs profile '{self.profile}' elsewhere. It does not run the "
                    "profile on this host until it can read that folder.")
        if self.plugin is None:
            return (f"Profile '{self.profile}' runs elsewhere: a plugin runs its backend (its marker "
                    f"{self.path} could not be read), so Hermes does not run it on this host. Use the "
                    "profile from the Hermes desktop app, or turn this off in the plugin (or delete the "
                    "marker) to run it here.")
        return (f"Profile '{self.profile}' runs elsewhere: the '{self.plugin}' plugin runs its backend, "
                "so Hermes does not run it on this host. Use the profile from the Hermes desktop app, "
                f"or turn this off in the '{self.plugin}' plugin to run it here.")


class ProfileRunsElsewhere(RuntimeError):
    """Raised when this host is asked to run a profile a plugin runs elsewhere."""

    def __init__(self, entry: RunElsewhere):
        super().__init__(entry.message())
        self.entry = entry


def run_elsewhere_dir(root: str | Path | None = None) -> Path:
    """The marker folder under *root* (default: this process's Hermes root)."""
    if root is None:
        from hermes_constants import get_default_hermes_root
        root = get_default_hermes_root()
    return Path(root) / RUN_ELSEWHERE_DIRNAME


def _is_profile_id(name: str) -> bool:
    return name == "default" or bool(PROFILE_ID_RE.match(name))


def _log_once(level: int, message: str, *args) -> None:
    key = message % args
    if key not in _logged:
        _logged.add(key)
        logger.log(level, message, *args)


def _marked_names(folder: Path) -> frozenset[str] | OSError:
    """Profile ids with a marker in *folder*, or the error that made the folder unreadable."""
    try:
        names = os.listdir(folder)
    except (FileNotFoundError, NotADirectoryError):
        return frozenset()
    except PermissionError:
        _log_once(logging.INFO, "Cannot list %s (permission denied): this process is a backend a "
                  "plugin runs elsewhere, so it refuses no profile", folder)
        return frozenset()
    except OSError as exc:
        _log_once(logging.ERROR, "Cannot read %s (%s); running no profile until it can be read", folder, exc)
        return exc
    # Writers rename a dot-prefixed temp file over <profile>.json; any other name is not a marker.
    return frozenset(name[:-len(_SUFFIX)] for name in names
                     if name.endswith(_SUFFIX) and _is_profile_id(name[:-len(_SUFFIX)]))


def _read_entry(folder: Path, profile: str) -> RunElsewhere:
    path = folder / f"{profile}{_SUFFIX}"
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return RunElsewhere(profile, path)
    if not isinstance(data, dict):
        return RunElsewhere(profile, path)
    plugin, since = data.get("plugin"), data.get("since")
    return RunElsewhere(profile, path, plugin if isinstance(plugin, str) and plugin.strip() else None,
                        since if isinstance(since, str) else None)


def run_elsewhere(profile: str, *, root: str | Path | None = None) -> RunElsewhere | None:
    """Why this host must not run *profile* (a canonical profile id), or None when it may."""
    if not isinstance(profile, str) or not _is_profile_id(profile):
        return None
    folder = run_elsewhere_dir(root)
    marked = _marked_names(folder)
    if isinstance(marked, OSError):
        return RunElsewhere(profile, folder / f"{profile}{_SUFFIX}", folder_error=str(marked))
    return _read_entry(folder, profile) if profile in marked else None


def run_elsewhere_for_home(home: str | Path) -> RunElsewhere | None:
    """:func:`run_elsewhere` for the profile whose home is *home*: ``<root>/profiles/<id>`` is that
    named profile of ``<root>``; any other home is the default profile of the root it is."""
    home = Path(home).expanduser()
    if home.parent.name == "profiles" and home.name != "default" and _is_profile_id(home.name):
        return run_elsewhere(home.name, root=home.parent.parent)
    return run_elsewhere("default", root=home)


def without_profiles_run_elsewhere(profiles: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    """*profiles* (``(name, home)`` pairs of this process's root) minus the marked ones. An
    unreadable marker folder leaves none: the host cannot tell which profiles it may run."""
    marked = _marked_names(run_elsewhere_dir())
    if isinstance(marked, OSError):
        return []
    return [(name, home) for name, home in profiles if name not in marked]


def require_home_runs_here(home: str | Path | None = None) -> None:
    """Raise :class:`ProfileRunsElsewhere` when a plugin runs the profile at *home* elsewhere.

    *home* defaults to this process's Hermes home, which a multi-profile host binds per agent
    build, so the host still serves the profiles that are not marked.
    """
    if home is None:
        from hermes_constants import get_hermes_home
        home = get_hermes_home()
    entry = run_elsewhere_for_home(home)
    if entry is not None:
        raise ProfileRunsElsewhere(entry)


def exit_if_profile_runs_elsewhere(command: str | None, *, exit_code: int = 1) -> None:
    """Print the refusal and exit when this process's profile runs elsewhere: for commands that
    run the profile, before any banner, TUI, server or gateway contact. ``gateway run`` has its own
    refusal (exit 78, ``gateway.run_bootstrap``)."""
    if command == "gateway":
        return
    try:
        require_home_runs_here()
    except ProfileRunsElsewhere as exc:
        import sys
        print(exc, file=sys.stderr)
        raise SystemExit(exit_code) from None


def profile_exists_here(name: str) -> bool:
    """Whether this host may start work for *name*: the profile exists and no plugin runs it
    elsewhere."""
    from hermes_cli.profiles import normalize_profile_name, profile_exists
    if not profile_exists(name):
        return False
    return run_elsewhere(normalize_profile_name(name)) is None


def _same_path(a: Path, b: Path) -> bool:
    try:
        return os.path.normcase(str(a.resolve(strict=False))) == os.path.normcase(str(b.resolve(strict=False)))
    except (OSError, RuntimeError):
        return False


def own_profile_dir_when_unlistable(profiles_root: Path, *, live_only: bool) -> list[Path]:
    """The named-profile dirs a process denied *profiles_root* can see: its own home, when that is
    a named profile directly under it (a backend a plugin runs elsewhere is denied the folder)."""
    from hermes_constants import get_hermes_home, named_profile_has_identity, named_profile_is_deleted
    own = Path(get_hermes_home())
    try:
        if (not _same_path(own.parent, profiles_root) or not PROFILE_ID_RE.match(own.name)
                or not named_profile_has_identity(own)):
            return []
    except OSError:
        return []
    try:
        deleted = live_only and named_profile_is_deleted(own)
    except PermissionError:
        deleted = False  # the tombstone folder is denied too; a running backend's own profile is live
    return [] if deleted else [own]


def _require_plugin(plugin: str) -> str:
    if not isinstance(plugin, str) or not plugin.strip():
        raise ValueError("plugin must be a non-empty plugin name")
    return plugin.strip()


def _require_profile(profile: str) -> str:
    from hermes_cli.profiles import normalize_profile_name
    name = normalize_profile_name(profile)
    if not _is_profile_id(name):
        raise ValueError(f"'{profile}' is not a valid profile id")
    return name


def mark_profile_run_elsewhere(profile: str, plugin: str, *, root: str | Path | None = None) -> RunElsewhere:
    """Record that *plugin* runs *profile*'s backend elsewhere, so this host stops running it.

    Keeps ``since`` when *plugin* already marked the profile; raises ``ValueError`` when another
    plugin did. Running host services drop the profile at their next rescan; a running host
    gateway drops a named profile at once through ``gateway.control_socket.request_unserve_profile``.
    The host gateway runs as the default profile, so stop it before marking ``default``.
    """
    from utils import atomic_json_write
    name, plugin = _require_profile(profile), _require_plugin(plugin)
    folder = run_elsewhere_dir(root)
    marked = _marked_names(folder)
    current = _read_entry(folder, name) if not isinstance(marked, OSError) and name in marked else None
    if current is not None and current.plugin not in (None, plugin):
        raise ValueError(f"profile '{name}' already runs elsewhere under the '{current.plugin}' plugin")
    since = (current.since if current is not None and current.plugin == plugin and current.since
             else datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))
    folder.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        # Host services that run as another local user must be able to list it: a permission
        # error reads as "this process is the backend running elsewhere" and refuses nothing.
        os.chmod(folder, 0o755)
    atomic_json_write(folder / f"{name}{_SUFFIX}", {"plugin": plugin, "since": since}, fsync_dir=True)
    return RunElsewhere(name, folder / f"{name}{_SUFFIX}", plugin, since)


def clear_profile_run_elsewhere(profile: str, plugin: str, *, root: str | Path | None = None) -> bool:
    """Remove *plugin*'s marker for *profile*; True when one was removed. Raises ``ValueError``
    when the marker names another plugin. Clear it only once the backend elsewhere has stopped."""
    name, plugin = _require_profile(profile), _require_plugin(plugin)
    folder = run_elsewhere_dir(root)
    current = _read_entry(folder, name)
    if current.plugin is not None and current.plugin != plugin:
        raise ValueError(f"profile '{name}' runs elsewhere under the '{current.plugin}' plugin, not '{plugin}'")
    try:
        (folder / f"{name}{_SUFFIX}").unlink()
    except FileNotFoundError:
        return False
    return True
