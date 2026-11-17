"""Profiles whose backend a plugin runs somewhere other than this host.

A host-side plugin that runs a profile's whole backend elsewhere (for example inside a sandbox)
marks the profile for as long as it does. While a profile is marked, nothing on this host runs it,
whether or not the plugin's own process is running:

- ``AIAgent`` refuses to build an agent for it, whichever frontend asks, and
  ``run_conversation()`` checks again before each turn, so an agent built before the profile was
  marked runs no further turn;
- ``cron.scheduler_tick.tick()`` returns before the tick lock or the due scan, and
  ``CronScheduler.claim_fire()`` claims nothing, so no in-process cron ticker and no provider's
  inbound fire runs its jobs;
- CLI commands that run a turn (chat, ``acp``, ``cron run``/``tick``, ...) exit 1 and
  ``hermes serve``/``dashboard`` exit 78 before they start, and the gateway process exits 78;
- ``profiles_to_serve()`` leaves it out, so the host gateway (platforms, cron, Kanban watchers,
  ``/p/<profile>/`` routes), the Desktop cron ticker and the Kanban dispatcher skip it;
- ``hermes gateway ensure`` refuses it (``profile_runs_elsewhere``), the dashboard's cron,
  session, file and Webapp requests for it answer 409, and a dashboard session's deferred agent
  build refuses before the profile's MCP servers are started.

While a profile is marked, its home is written by the backend the plugin runs, so host processes
load no env file and no secret source from it: ``load_hermes_dotenv`` (run at the import of every
``hermes`` start, before any command can refuse) reads no ``.env`` or ``.op.env`` there, takes no
config-backend selector from them and resolves none of its ``secrets:`` sources, and a host that
serves the profile resolves none of them either (``hydrate_profile_secret_sources``). The marking
plugin supplies what its own host process needs. Two readers still open the ``.env`` file: code
that asks for one named value (``hermes_cli.config.get_env_value``), and the secret scope a host
builds for a profile it serves.

The marker is the file ``<hermes root>/profiles_run_elsewhere/<profile>.json``. It lives under the
Hermes root and never inside a profile home, so the profile's backend cannot remove it. The file's
presence is the marker; its contents (``{"plugin": ..., "since": ...}``) only name the owner in
messages, so a marker that cannot be read still marks the profile.

The plugin keeps this folder from the backend it runs elsewhere. A process that is denied the
folder (permission error) is therefore that backend, not the host, and refuses nothing: it keeps
running its own profile and loads its own env files. Any other error reading the folder refuses
every profile, and no profile's env files or secret sources are loaded until it can be read.
"""

from __future__ import annotations

import contextlib
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


@dataclass(frozen=True)
class _MarkerFolder:
    """One listing of the marker folder: the profile ids it marks, and ``error`` when the folder
    could not be read (no profile may run here then)."""

    names: frozenset[str] = frozenset()
    error: OSError | None = None


def _read_folder(folder: Path) -> _MarkerFolder:
    try:
        names = os.listdir(folder)
    except FileNotFoundError:
        return _MarkerFolder()
    except PermissionError:
        _log_once(logging.INFO, "Cannot list %s (permission denied): this process is a backend a "
                  "plugin runs elsewhere, so it refuses no profile", folder)
        return _MarkerFolder()
    except OSError as exc:  # a file where the folder belongs, an I/O error, ...
        _log_once(logging.ERROR, "Cannot read %s (%s); running no profile and loading no profile's "
                  "env files until it can be read", folder, exc)
        return _MarkerFolder(error=exc)
    # Writers stage a marker under a dot-prefixed name; any name but <profile>.json is not a marker.
    return _MarkerFolder(frozenset(name[:-len(_SUFFIX)] for name in names
                                   if name.endswith(_SUFFIX) and _is_profile_id(name[:-len(_SUFFIX)])))


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
    listing = _read_folder(folder)
    if listing.error is not None:
        return RunElsewhere(profile, folder / f"{profile}{_SUFFIX}", folder_error=str(listing.error))
    return _read_entry(folder, profile) if profile in listing.names else None


def _home_readings(home: Path) -> list[tuple[str, Path]]:
    """``(profile, root)`` for *home* as written and as resolved, without repeats:
    ``<root>/profiles/<id>`` is that named profile of ``<root>``; any other home is the default
    profile of the root it is."""
    spellings = [home]
    try:
        spellings.append(home.resolve(strict=False))
    except (OSError, RuntimeError):  # a link loop: only the written path is left to read
        pass
    readings: list[tuple[str, Path]] = []
    for path in spellings:
        name = path.name.lower()  # ids are stored lowercase; a case-insensitive volume opens any case
        named = path.parent.name == "profiles" and name != "default" and _is_profile_id(name)
        reading = (name, path.parent.parent) if named else ("default", path)
        if reading not in readings:
            readings.append(reading)
    return readings


def run_elsewhere_for_home(home: str | Path) -> RunElsewhere | None:
    """:func:`run_elsewhere` for the profile whose home is *home*.

    *home* is read both as written and resolved, so another path to a profile home (a symlink or
    junction, a relative or ``..`` path, another letter case) reaches that profile's marker. A
    marker under either reading refuses, so neither reading can hide the other's.
    """
    for profile, root in _home_readings(Path(home).expanduser()):
        entry = run_elsewhere(profile, root=root)
        if entry is not None:
            return entry
    return None


def without_profiles_run_elsewhere(profiles: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    """*profiles* (``(name, home)`` pairs of this process's root) minus the marked ones. An
    unreadable marker folder leaves none: the host cannot tell which profiles it may run."""
    listing = _read_folder(run_elsewhere_dir())
    if listing.error is not None:
        return []
    return [(name, home) for name, home in profiles if name not in listing.names]


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


def own_profile_dir_when_unlistable(profiles_root: Path, denied: PermissionError, *, live_only: bool) -> list[Path]:
    """The named-profile dirs a process that is denied *profiles_root* can see.

    A process whose home is a named profile directly under it is a backend a plugin runs elsewhere
    (the plugin denies it the folder) and sees its own profile. Any other process gets *denied*
    raised, as before this fallback existed, so a host that fails closed on a profiles folder it
    cannot list (``activate_multi_profile_hosting_eagerly``) still does.
    """
    from hermes_constants import get_hermes_home, named_profile_has_identity, named_profile_is_deleted
    own = Path(get_hermes_home())
    if not _same_path(own.parent, profiles_root) or not PROFILE_ID_RE.match(own.name):
        raise denied
    try:
        if not named_profile_has_identity(own):
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


def _publish_new(path: Path, payload: dict) -> bool:
    """Write *payload* to *path* only where no file is yet; False when one is.

    The payload is written whole under a dot-prefixed name, which readers skip, and hard-linked
    into place: a link never replaces an existing file, where a POSIX rename would.
    """
    from utils import atomic_json_write, fsync_directory
    staged = path.with_name(f".{path.stem}.{os.getpid()}.{os.urandom(4).hex()}.new")
    try:
        atomic_json_write(staged, payload)
        try:
            os.link(staged, path)
        except FileExistsError:
            return False
        except OSError:
            # No hard links on this filesystem (FAT, Android app storage): rename into place, which
            # cannot tell two first writers apart.
            atomic_json_write(path, payload, fsync_dir=True)
            return True
        fsync_directory(path.parent)
        return True
    finally:
        with contextlib.suppress(OSError):
            staged.unlink()


def mark_profile_run_elsewhere(profile: str, plugin: str, *, root: str | Path | None = None) -> RunElsewhere:
    """Record that *plugin* runs *profile*'s backend elsewhere, so this host stops running it.

    Keeps ``since`` when *plugin* already marked the profile; raises ``ValueError`` when another
    plugin did, also when both found no marker and the other's landed first. Running host services
    drop the profile at their next rescan; a running host gateway drops a named profile at once
    through ``gateway.control_socket.request_unserve_profile``. The host gateway runs as the default
    profile, so stop it before marking ``default``.
    """
    from utils import atomic_json_write
    name, plugin = _require_profile(profile), _require_plugin(plugin)
    folder = run_elsewhere_dir(root)
    path = folder / f"{name}{_SUFFIX}"
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    folder.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        # Host services that run as another local user must be able to list it: a permission
        # error reads as "this process is the backend running elsewhere" and refuses nothing.
        os.chmod(folder, 0o755)
    current = _read_entry(folder, name) if name in _read_folder(folder).names else None
    if current is None:
        if _publish_new(path, {"plugin": plugin, "since": now}):
            return RunElsewhere(name, path, plugin, now)
        current = _read_entry(folder, name)  # another writer's marker landed first: it decides
    if current.plugin not in (None, plugin):
        raise ValueError(f"profile '{name}' already runs elsewhere under the '{current.plugin}' plugin")
    since = current.since if current.plugin == plugin and current.since else now
    atomic_json_write(path, {"plugin": plugin, "since": since}, fsync_dir=True)
    return RunElsewhere(name, path, plugin, since)


def clear_profile_run_elsewhere(profile: str, plugin: str, *, root: str | Path | None = None) -> bool:
    """Remove *plugin*'s marker for *profile*; True when one was removed. Raises ``ValueError``
    when the marker names another plugin. Clear it only once the backend elsewhere has stopped."""
    name, plugin = _require_profile(profile), _require_plugin(plugin)
    folder = run_elsewhere_dir(root)
    path = folder / f"{name}{_SUFFIX}"
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False  # nothing to remove; a marker published from here on is not this call's to delete
    current = _read_entry(folder, name)
    if current.plugin is not None and current.plugin != plugin:
        raise ValueError(f"profile '{name}' runs elsewhere under the '{current.plugin}' plugin, not '{plugin}'")
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True
