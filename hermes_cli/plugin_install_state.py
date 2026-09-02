"""Profile-local plugin install metadata, locking, and crash recovery."""

from __future__ import annotations

import json
import os
import stat
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_home
from utils import (
    secure_atomic_write_text,
    secure_open_file,
    secure_parent_directory,
    secure_replace,
    secure_rmtree,
    secure_unlink,
)

_INSTALL_METADATA_FILE = ".install-metadata.json"
_INSTALL_TRANSACTION_FILE = ".install-transaction.json"
_PROCESS_INSTALL_METADATA_LOCK = threading.RLock()


class PluginOperationError(Exception):
    """Recoverable plugin install/update failure (CLI exits; HTTP maps to 4xx)."""


def _plugins_dir() -> Path:
    plugins = get_hermes_home() / "plugins"
    plugins.mkdir(parents=True, exist_ok=True)
    return plugins


def _lock_file(handle) -> None:
    """Acquire a blocking one-byte cross-process lock."""
    if os.name == "nt":
        import errno
        import msvcrt

        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        while True:
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise
                time.sleep(0.1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock_file(handle) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _open_lock_path(path: Path):
    """Open a regular lock file beneath a held no-follow parent."""
    if path.is_symlink():
        raise PluginOperationError(f"Lock path must not be a symlink: {path}")
    flags = os.O_RDWR | os.O_CREAT | os.O_APPEND | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = secure_open_file(path, get_hermes_home(), flags, create_parent=True)
    except OSError as exc:
        raise PluginOperationError(
            f"Could not safely open lock file {path}: {exc}"
        ) from exc
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise PluginOperationError(f"Lock path must be a regular file: {path}")
    return os.fdopen(fd, "a+b")


def _read_control_json(path: Path, label: str) -> object:
    """Read one regular control file beneath a held no-follow parent."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = secure_open_file(path, get_hermes_home(), flags)
    except OSError as exc:
        raise PluginOperationError(f"Could not read {label}: {exc}") from exc
    try:
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            fd = -1
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise PluginOperationError(f"Could not read {label}: {exc}") from exc
    finally:
        if fd >= 0:
            os.close(fd)


@contextmanager
def _install_metadata_lock():
    """Serialize profile-local install metadata and target commits."""
    get_hermes_home().mkdir(parents=True, exist_ok=True)
    path = _install_metadata_path().with_suffix(".lock")
    namespace = secure_parent_directory(path, get_hermes_home(), create=True)
    try:
        namespace.__enter__()
    except OSError as exc:
        raise PluginOperationError(
            f"Plugin install namespace is unsafe: {exc}"
        ) from exc
    try:
        with _PROCESS_INSTALL_METADATA_LOCK, _open_lock_path(path) as handle:
            _lock_file(handle)
            try:
                _recover_install_transaction()
                yield
            finally:
                _unlock_file(handle)
    finally:
        namespace.__exit__(None, None, None)


def _install_metadata_path() -> Path:
    return get_hermes_home() / "plugins" / _INSTALL_METADATA_FILE


def _install_transaction_path() -> Path:
    return get_hermes_home() / "plugins" / _INSTALL_TRANSACTION_FILE


def _read_install_metadata() -> dict[str, dict[str, object]]:
    """Read profile-local, non-secret plugin source metadata from disk."""
    path = _install_metadata_path()
    if path.is_symlink():
        raise PluginOperationError(
            "Plugin install metadata path must not be a symlink."
        )
    if not path.exists():
        return {}
    value = _read_control_json(path, "plugin install metadata")
    if not isinstance(value, dict):
        raise PluginOperationError("Plugin install metadata must be a JSON object.")
    for key, entry in value.items():
        if not isinstance(key, str) or not isinstance(entry, dict):
            raise PluginOperationError(
                "Plugin install metadata entries must map plugin names to objects."
            )
    return value


def _write_install_metadata(metadata: dict[str, dict[str, object]]) -> None:
    """Atomically replace the profile-local plugin install metadata sidecar."""
    secure_atomic_write_text(
        _install_metadata_path(),
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        get_hermes_home(),
    )


def _plugin_membership(name: str) -> tuple[bool, bool]:
    from hermes_cli.config import config_write_lock, load_config

    with config_write_lock():
        config = load_config()
        plugins = config.get("plugins")
        plugins = plugins if isinstance(plugins, dict) else {}
        enabled = plugins.get("enabled")
        disabled = plugins.get("disabled")
        return (
            isinstance(enabled, list) and name in enabled,
            isinstance(disabled, list) and name in disabled,
        )


def _restore_plugin_membership(
    name: str, was_enabled: bool, was_disabled: bool
) -> None:
    """Restore only *name* without overwriting newer state for other plugins."""
    from hermes_cli.config import config_write_lock, load_config, save_config

    with config_write_lock():
        config = load_config()
        plugins = config.setdefault("plugins", {})
        if not isinstance(plugins, dict):
            plugins = {}
            config["plugins"] = plugins
        raw_enabled = plugins.get("enabled")
        raw_disabled = plugins.get("disabled")
        enabled = set(raw_enabled) if isinstance(raw_enabled, list) else set()
        disabled = set(raw_disabled) if isinstance(raw_disabled, list) else set()
        (enabled.add if was_enabled else enabled.discard)(name)
        (disabled.add if was_disabled else disabled.discard)(name)
        plugins["enabled"] = sorted(enabled)
        plugins["disabled"] = sorted(disabled)
        save_config(
            config,
            preserve_plugin_state=False,
            preserve_platform_toolsets=False,
            merge_existing=True,
        )


def _write_install_transaction(value: dict[str, object]) -> None:
    secure_atomic_write_text(
        _install_transaction_path(),
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        get_hermes_home(),
    )


def _recover_install_transaction() -> None:
    """Roll back an interrupted install before any new plugin mutation."""
    journal = _install_transaction_path()
    if journal.is_symlink():
        raise PluginOperationError(
            "Plugin install transaction path must not be a symlink."
        )
    if not journal.exists():
        return
    value = _read_control_json(journal, "plugin install transaction")
    if not isinstance(value, dict) or value.get("version") != 1:
        raise PluginOperationError("Plugin install transaction is malformed.")

    name = value.get("plugin_name")
    transaction_dir = value.get("transaction_dir")
    old_metadata = value.get("old_metadata")
    if (
        not isinstance(name, str)
        or not isinstance(transaction_dir, str)
        or not transaction_dir.startswith(".install-")
        or "/" in transaction_dir
        or "\\" in transaction_dir
        or not isinstance(old_metadata, dict)
        or any(
            not isinstance(key, str) or not isinstance(entry, dict)
            for key, entry in old_metadata.items()
        )
        or not isinstance(value.get("was_enabled"), bool)
        or not isinstance(value.get("was_disabled"), bool)
    ):
        raise PluginOperationError("Plugin install transaction is malformed.")

    from hermes_cli.plugins_cmd import _sanitize_plugin_name

    plugins_dir = _plugins_dir()
    try:
        target = _sanitize_plugin_name(name, plugins_dir)
    except ValueError as exc:
        raise PluginOperationError(str(exc)) from exc
    transaction_root = plugins_dir / transaction_dir
    if (
        transaction_root.is_symlink()
        or not transaction_root.is_dir()
        or transaction_root.resolve().parent != plugins_dir.resolve()
    ):
        raise PluginOperationError("Plugin install transaction directory is unsafe.")
    backup = transaction_root / "previous-plugin"
    if backup.is_symlink() or (backup.exists() and not backup.is_dir()):
        raise PluginOperationError("Plugin install transaction backup is unsafe.")
    replaced_existing = value.get("replaced_existing") is True

    from hermes_cli.config import config_write_lock

    with config_write_lock():
        if replaced_existing and backup.exists():
            if target.exists():
                if target.is_dir():
                    secure_rmtree(target, plugins_dir)
                else:
                    secure_unlink(target, plugins_dir)
            secure_replace(backup, target, plugins_dir)
        elif replaced_existing and not target.exists():
            raise PluginOperationError(
                "Interrupted plugin install is missing both target and backup."
            )
        elif not replaced_existing and target.exists():
            if target.is_dir():
                secure_rmtree(target, plugins_dir)
            else:
                secure_unlink(target, plugins_dir)

        if old_metadata:
            _write_install_metadata(old_metadata)
        else:
            secure_unlink(_install_metadata_path(), get_hermes_home(), missing_ok=True)
        _restore_plugin_membership(
            name,
            value.get("was_enabled") is True,
            value.get("was_disabled") is True,
        )
        secure_unlink(journal, get_hermes_home(), missing_ok=True)

    if transaction_root.exists():
        try:
            secure_rmtree(transaction_root, plugins_dir)
        except OSError:
            pass


def _marketplace_metadata(entry) -> dict[str, object]:
    """Return private-marketplace provenance for atomic install metadata."""
    return {
        "marketplace_id": entry.source_id,
        "marketplace_name": entry.source_name,
        "marketplace_plugin_name": entry.name,
        "source": entry.repo,
        "subdir": entry.subdir,
        "installed_repo_sha": entry.sha,
        "installed_tree_sha": entry.tree_sha,
    }


def _marketplace_install(plugin_name: str) -> Optional[dict[str, object]]:
    value = _read_install_metadata().get(plugin_name)
    if not isinstance(value, dict) or not value.get("marketplace_id"):
        return None
    return value
