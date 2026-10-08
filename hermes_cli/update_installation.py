"""Installation-owned update configuration, shared by every profile and entrypoint.

Older releases wrote an install-keyed record into whichever profile was active.
Before a record is consolidated, only agreement between those explicit records
can select a channel. The installation-scoped record then becomes authoritative;
legacy files stay intact so migration never erases another profile's settings.
"""

from __future__ import annotations

from copy import deepcopy
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
import stat

from pm.environments import install_key
from hermes_cli.update_installation_owner import ensure_installation_home, installation_home


def installation_config_path(project_root: Path, *, home: Path | None = None) -> Path:
    return installation_home(project_root, home=home) / "config.yaml"


def profile_homes(home: Path) -> list[Path]:
    """Strict, read-only roster under an explicit installation data root."""
    from hermes_constants import named_profile_has_identity, named_profile_is_deleted
    from hermes_cli.profiles import _PROFILE_ID_RE

    profiles = home / "profiles"
    try:
        entries = sorted(profiles.iterdir())
    except FileNotFoundError:
        entries = []
    homes = [home]
    for entry in entries:
        if entry.name == "default" or not _PROFILE_ID_RE.fullmatch(entry.name):
            continue
        if not stat.S_ISDIR(entry.stat().st_mode):
            continue
        list(entry.iterdir())  # An unreadable candidate is unknown, not an absent profile.
        if named_profile_has_identity(entry) and not named_profile_is_deleted(entry):
            homes.append(entry)
    return homes


def _record(path: Path, root: Path) -> dict:
    from hermes_cli.config import require_readable_config_before_write

    config = require_readable_config_before_write(path)
    update = config.get("update")
    update = {} if update is None else update
    if not isinstance(update, dict):
        raise ValueError(f"config key 'update' is not a mapping: {path}")
    installs = update.get("installs")
    installs = {} if installs is None else installs
    if not isinstance(installs, dict):
        raise ValueError(f"config key 'update.installs' is not a mapping: {path}")
    record = installs.get(install_key(root))
    record = {} if record is None else record
    if not isinstance(record, dict):
        raise ValueError(f"installation update record is not a mapping: {path}")
    if record.get("scope") is not None and record.get("scope") != "installation":
        raise ValueError(f"Unrecognized installation update scope: {path}")
    return record


def read_install_channel_record(project_root: Path, *, home: Path | None = None) -> dict:
    from hermes_cli.release_channels import validate_name

    root = Path(project_root).resolve()
    owner = installation_home(root, home=home)
    canonical = _record(owner / "config.yaml", root)
    if canonical.get("scope") == "installation":
        validate_name(canonical.get("channel"))
        return canonical
    selections = []
    for candidate in profile_homes(owner):
        path = candidate / "config.yaml"
        record = canonical if candidate == owner else _record(path, root)
        if record.get("channel") is not None:
            selections.append((path, validate_name(record["channel"])))
    channels = {channel for _, channel in selections}
    if len(channels) > 1:
        choices = "; ".join(f"{path}: {channel}" for path, channel in selections)
        raise ValueError(f"Conflicting update channels for installation {root}: {choices}. "
                         "Choose the installation's channel with hermes update --set-channel CHANNEL.")
    if not channels:
        return canonical
    return {**canonical, "path": str(root), "channel": channels.pop()}


def resolve_install_channel(project_root: Path, *, home: Path | None = None) -> str:
    from hermes_cli.update_channel import _package_channel, _read_stamp, default_channel

    root = Path(project_root).resolve()
    if _package_channel(_read_stamp(root)):
        return default_channel(root)
    return read_install_channel_record(root, home=home).get("channel") or default_channel(root)


@dataclass
class ChannelMigration:
    path: Path
    root: Path
    before: dict
    after: dict
    changed: bool

    def rollback(self) -> dict:
        from hermes_cli.config import atomic_config_replace, require_readable_config_before_write
        from hermes_cli.update_channel import _channel_write_lock

        if not self.changed:
            return {"ok": True, "channelConfig": str(self.path)}
        try:
            with _channel_write_lock(self.path):
                current = require_readable_config_before_write(self.path)
                key = install_key(self.root)
                if _record(self.path, self.root) != self.after:
                    raise ValueError("channel configuration changed during migration; refusing to overwrite it")
                previous_update = self.before.get("update") or {}
                previous_installs = previous_update.get("installs") or {}
                if key not in previous_installs:
                    current["update"]["installs"].pop(key)
                    if not current["update"]["installs"] and not previous_update.get("installs"):
                        if "installs" in previous_update:
                            current["update"]["installs"] = previous_update["installs"]
                        else:
                            current["update"].pop("installs")
                    if not current["update"] and not self.before.get("update"):
                        if "update" in self.before:
                            current["update"] = self.before["update"]
                        else:
                            current.pop("update")
                else:
                    current["update"]["installs"][key] = previous_installs[key]
                atomic_config_replace(self.path, current)
                self.changed = False
            return {"ok": True, "channelConfig": str(self.path)}
        except (OSError, ValueError, RuntimeError) as exc:
            return {"ok": False, "channelConfig": str(self.path), "errors": [str(exc)]}


def migrate_install_channel(project_root: Path, *, home: Path | None = None) -> ChannelMigration:
    """Consolidate only an unambiguous legacy choice; explicit changes use --set-channel."""
    from hermes_cli.config import require_readable_config_before_write
    from hermes_cli.update_channel import _channel_write_lock, _write_channel_record_locked, default_channel

    root = Path(project_root).resolve()
    path = ensure_installation_home(root, home=home) / "config.yaml"
    with _channel_write_lock(path), ExitStack() as locks:
        before = deepcopy(require_readable_config_before_write(path))
        existing = _record(path, root)
        if existing.get("scope") == "installation":
            record = read_install_channel_record(root, home=home)
            return ChannelMigration(path, root, before, record, False)
        seen = {path.resolve()}
        for candidate in profile_homes(path.parent):
            profile_path = candidate / "config.yaml"
            if profile_path.resolve() not in seen:
                locks.enter_context(_channel_write_lock(profile_path))
                seen.add(profile_path.resolve())
        record = read_install_channel_record(root, home=home)
        channel = record.get("channel") or default_channel(root)
        _write_channel_record_locked(install_key(root), str(root), channel, None, path)
        after = _record(path, root)
        return ChannelMigration(path, root, before, after, existing != after)


def prepare_install_channel_record(project_root: Path, *, transient: str | None) -> dict:
    """The persistent subscription a source completion may adopt after retirement."""
    if transient is not None:
        return {}
    migrate_install_channel(project_root)
    return deepcopy(read_install_channel_record(project_root))
