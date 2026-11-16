"""``hermes import --keep-host``: restore a backup over an agent whose host set part of it up.

A Hermes Cloud agent is provisioned by its platform: it is signed in to Nous Portal, its model and
tools route through Nous, its dashboard is gated by the portal's OAuth client, and its scheduler is
tuned for the container. Replacing that agent with someone's local backup must bring over
everything that makes the agent theirs (memory, persona, skills, conversations, jobs, settings, and
their other API keys and logins) while the host-owned pieces stay as the host set them:

* ``auth.json``: the destination's Nous Portal entries (``providers.nous``, the ``nous`` pool rows,
  ``active_provider`` when either side used ``nous``); every other provider comes from the archive.
  Merged under the auth-store lock against the LIVE store, so a Nous token refresh racing the
  restore is never rolled back, and the archive's own Nous refresh token never lands on disk.
* ``config.yaml``: the top-level sections in :data:`HOST_CONFIG_KEYS` keep the destination's value
  (or stay absent); every other key comes from the archive.
* ``.env``: keys :func:`host_env_key` names keep the destination's value; the rest come over.
* ``shared/nous_auth.json`` (the cross-profile Nous token store) is never taken from the archive.
* ``skills/`` and ``memories/`` become exactly the archive's: a destination file the archive does
  not carry is removed, so the result is a replacement, not a merge.

Messaging bot tokens come over like any other key. A bot token answers to one running Hermes at a
time, so the user stops the old one; the Desktop says so before the transfer.

Before anything is written a full backup of the destination lands in ``backups/pre-transfer-*.zip``:
``hermes import <it> --force --keep-host`` puts the agent back (its host keys are the archive's then).

In an s6 container (the image Hermes Cloud runs) the gateway owns the profile, and ``hermes import``
refuses ``state.db`` while it does. :func:`gateway_paused` asks the gateway to drain and exit through
its control socket (the ``pause-for-update`` verb ``hermes update`` uses) with its s6 slot marked
run-once so the supervisor does not bring it back mid-restore, then starts the slot again.
"""

from __future__ import annotations

import copy
import io
import json
import logging
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Top-level config.yaml sections the hosting platform provisions. Hermes Cloud seeds the model and
# provider (Nous), the Nous tool gateway routing (web/image_gen/video_gen/tts/stt), the dashboard's
# portal OAuth client (``dashboard.oauth``) and Web Access listener, and the container's scheduler and
# session caps. Keep this ONE list: the import, its docs and the Desktop confirmation name this set.
HOST_CONFIG_KEYS = (
    "model", "providers", "fallback_providers", "fallback", "auxiliary", "nous", "auth",
    "web", "image_gen", "video_gen", "tts", "stt", "dashboard", "webapp", "cron",
    "max_concurrent_sessions", "max_live_sessions",
)

# .env names the host provisions: the Nous sign-in and portal endpoints, the Tool Gateway, Hermes'
# own deployment switches, and the container's generated API server key (docker/stage2-hook.sh).
_HOST_ENV_PREFIXES = ("NOUS_", "HERMES_", "TOOL_GATEWAY_")
_HOST_ENV_NAMES = frozenset({"API_SERVER_KEY"})

_SHARED_NOUS_STORE = "shared/nous_auth.json"
_REPLACED_TREES = ("skills", "memories")
_UNDO_PREFIX = "pre-transfer-"
_UNDO_KEEP = 3
_NOUS = "nous"


def host_env_key(name: str) -> bool:
    return name in _HOST_ENV_NAMES or name.startswith(_HOST_ENV_PREFIXES)


def _env_name(line: str) -> str:
    body = line.strip()
    if not body or body.startswith("#") or "=" not in body:
        return ""
    return body.removeprefix("export ").split("=", 1)[0].strip()


def merge_env(archive_text: str, host_text: str) -> str:
    """The archive's .env with every host-owned name set to the destination's line (in place, so an
    undo reproduces the file byte for byte), dropped when the destination lacks it, and the
    destination's other host-owned lines appended."""
    host = {_env_name(line): line for line in host_text.splitlines() if host_env_key(_env_name(line))}
    out = []
    for line in archive_text.splitlines():
        name = _env_name(line)
        if host_env_key(name):
            if name in host:
                out.append(host.pop(name))
        else:
            out.append(line)
    return "\n".join(out + list(host.values())) + "\n"


def merge_config(archive_cfg: dict[str, Any], host_cfg: dict[str, Any]) -> dict[str, Any]:
    merged = dict(archive_cfg)
    for key in HOST_CONFIG_KEYS:
        if key in host_cfg:
            merged[key] = copy.deepcopy(host_cfg[key])
        else:
            merged.pop(key, None)
    return merged


def merge_auth(archive_store: dict[str, Any], host_store: dict[str, Any]) -> dict[str, Any]:
    """The archive's auth store with the destination's Nous Portal sign-in in place of its own."""
    merged = copy.deepcopy(archive_store)
    for section in ("providers", "credential_pool"):
        raw_theirs, raw_ours = merged.get(section), host_store.get(section)
        theirs: dict[str, Any] = raw_theirs if isinstance(raw_theirs, dict) else {}
        ours: dict[str, Any] = raw_ours if isinstance(raw_ours, dict) else {}
        theirs.pop(_NOUS, None)
        if _NOUS in ours:
            theirs[_NOUS] = copy.deepcopy(ours[_NOUS])
        merged[section] = theirs
    if _NOUS in (merged.get("active_provider"), host_store.get("active_provider")):
        merged["active_provider"] = host_store.get("active_provider")
    return merged


def _yaml(raw: Optional[bytes]) -> dict[str, Any]:
    import hermes_yaml as yaml
    data = yaml.safe_load(raw.decode("utf-8-sig")) if raw else None  # config-reader: ok — raw keep-host merge of one file
    return data if isinstance(data, dict) else {}


def _json(raw: Optional[bytes]) -> dict[str, Any]:
    try:
        data = json.loads(raw.decode("utf-8-sig")) if raw else None
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _live(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return b""


class _KeepHostZip:
    """The archive as the import reads it, with config.yaml and .env members merged against the
    destination's live file at the moment each is published. Everything else passes through."""

    def __init__(self, zf, prefix: str, root: Path):
        self._zf, self._prefix, self._root = zf, prefix, root

    def __getattr__(self, name):
        return getattr(self._zf, name)

    def open(self, member, *args, **kwargs):
        rel = member[len(self._prefix):] if self._prefix and member.startswith(self._prefix) else member
        name = Path(rel).name
        if name not in ("config.yaml", ".env"):
            return self._zf.open(member, *args, **kwargs)
        archive, live = self._zf.read(member), _live(self._root / rel)
        if name == ".env":
            data = merge_env(archive.decode("utf-8-sig", "replace"), live.decode("utf-8", "replace")).encode()
        else:
            import hermes_yaml as yaml
            data = yaml.safe_dump(merge_config(_yaml(archive), _yaml(live)), sort_keys=False).encode()
        return io.BytesIO(data)


def _rel(member: str, prefix: str) -> str:
    return member[len(prefix):] if prefix and member.startswith(prefix) else member


def _restore_auth_members(zf, members: list[str], prefix: str, root: Path) -> list[str]:
    """Merge each archived auth.json into the live one under its lock; return per-file errors."""
    from hermes_cli.auth import _auth_store_lock, _load_auth_store, _save_auth_store
    errors = []
    for member in members:
        target = root / _rel(member, prefix)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with _auth_store_lock(target_path=target):
                _save_auth_store(merge_auth(_json(zf.read(member)), _load_auth_store(target)), target_path=target)
        except Exception as exc:  # health: allow BLE001 -- reported like any other unrestored member
            errors.append(f"  {_rel(member, prefix)}: {exc}")
    return errors


def _prune_replaced_trees(root: Path, archive_rels: set[str]) -> int:
    """Remove destination files under ``skills/`` and ``memories/`` the archive did not carry."""
    from hermes_cli.backup import _should_exclude
    profiles = root / "profiles"
    homes = [root, *(p for p in profiles.iterdir() if p.is_dir())] if profiles.is_dir() else [root]
    removed = 0
    for home in homes:
        for base in (home / tree for tree in _REPLACED_TREES):
            for path in sorted(base.rglob("*"), reverse=True) if base.is_dir() else ():
                rel = path.relative_to(root)
                if path.is_dir() and not path.is_symlink():
                    if not any(path.iterdir()):
                        path.rmdir()
                elif rel.as_posix() not in archive_rels and not _should_exclude(rel.relative_to(home.relative_to(root))):
                    path.unlink()
                    removed += 1
    return removed


def kept_summary(root: Path) -> list[str]:
    """Human lines for what a keep-host restore leaves as the destination has it."""
    cfg, env = _yaml(_live(root / "config.yaml")), _live(root / ".env").decode("utf-8", "replace")
    names = sorted({n for n in map(_env_name, env.splitlines()) if n and host_env_key(n)})
    signed_in = _NOUS in (_json(_live(root / "auth.json")).get("providers") or {})
    return [f"  config.yaml: {', '.join(k for k in HOST_CONFIG_KEYS if k in cfg) or '(none set here)'}",
            f"  .env: {', '.join(names) or '(none set here)'}",
            f"  auth.json: {'its Nous Portal sign-in' if signed_in else '(not signed in to Nous Portal)'}"]


def create_undo_point(
    root: Path, out_path: Optional[Path] = None, external_members: tuple[str, ...] = (),
) -> Optional[Path]:
    """Back up the destination, including existing external files this import will overwrite.

    The ordinary automatic full backup only walks HERMES_HOME. Normal ``hermes backup``
    archives may also contain ``_external/`` memory-provider files under the user's home,
    which the import publishes outside HERMES_HOME. Without these files the Undo cannot
    restore their previous contents.
    """
    from hermes_cli.backup import _create_prefixed_full_backup, _write_full_zip_backup, _write_zip_file

    if out_path is None:
        undo = _create_prefixed_full_backup(root, _UNDO_PREFIX, _UNDO_KEEP, "pre-transfer", "pre-transfer backup")
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        undo = _write_full_zip_backup(out_path, root)
    if undo is None or not external_members:
        return undo

    home = Path.home().resolve()
    try:
        old_files: list[tuple[Path, str]] = []
        for member in external_members:
            rel = member.removeprefix("_external/")
            target = home / rel
            if not target.resolve().is_relative_to(home):
                continue  # The import rejects paths outside the home as well.
            if target.is_symlink():
                raise OSError(f"cannot snapshot linked external state: {member}")
            if target.is_file():
                old_files.append((target, member))
        if old_files:
            with zipfile.ZipFile(undo, "a", zipfile.ZIP_DEFLATED) as archive:
                for source, member in old_files:
                    _write_zip_file(archive, source, member)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        logger.warning("Undo point could not include external state: %s", exc)
        undo.unlink(missing_ok=True)
        return None
    return undo


def _s6_slot(home: Path) -> Optional[str]:
    """The s6 slot whose gateway serves ``home`` and is up, else None (not an s6 container)."""
    from hermes_cli.service_manager import S6_SERVICE_PREFIX, detect_service_manager, get_service_manager
    from hermes_constants import profile_name_for_home
    if detect_service_manager() != "s6":
        return None
    slot = f"{S6_SERVICE_PREFIX}{profile_name_for_home(home) or 'default'}"
    try:
        return slot if get_service_manager().is_running(slot) else None
    except Exception:  # health: allow BLE001 -- an unprobeable slot is not a running gateway
        return None


@contextmanager
def gateway_paused(home: Path):
    """Hold the profile's s6-supervised gateway down for the restore, then start it again.

    The gateway drains its in-flight turns and exits (control-socket ``pause-for-update``); its
    slot is marked run-once first (``s6-svc -O``) so s6 does not relaunch it into the restore.
    Outside an s6 container this does nothing: ``hermes import`` keeps refusing a live gateway.
    """
    from gateway.control_socket import pause_gateway_for_update
    from gateway.runtime_ownership import OwnershipConflict
    from hermes_cli.profiles_remove import wait_profile_unowned
    from hermes_cli.service_manager import S6ServiceManager

    slot = _s6_slot(home)
    if slot is None:
        yield
        return
    manager = S6ServiceManager()
    manager._run_svc("-O", "pause", slot)
    try:
        ack = pause_gateway_for_update(home, timeout=8)
        if not (ack and (ack.get("pausing") or ack.get("already_stopping"))):
            raise OwnershipConflict("the gateway did not accept the pause; nothing was restored")
        print(f"Paused the gateway ({slot}) for the restore ...")
        if not wait_profile_unowned(home, timeout=float(ack.get("drain_timeout") or 30) + 30):
            raise OwnershipConflict("the gateway is still finishing work; nothing was restored. Retry shortly")
        yield
    finally:
        manager.start(slot)
        print(f"Started the gateway ({slot}) again.")


@contextmanager
def import_source(args, zf, members: list[str], prefix: str, root: Path):
    """``(zip, members)`` ``run_import`` restores from, plus the keep-host before/after steps.

    Without ``--keep-host`` the archive is restored as it is. With it: print what is kept, write
    the undo point, pause the gateway, publish the merged host files, then prune the replaced trees.
    """
    if not getattr(args, "keep_host", False):
        yield zf, members, []
        return
    print("\nKeeping what this host set up (--keep-host):")
    print("\n".join(kept_summary(root)))
    undo = create_undo_point(
        root, Path(args.undo_point) if getattr(args, "undo_point", None) else None,
        tuple(m for m in members if m.startswith("_external/")),
    )
    if undo is None:
        raise OSError("could not back up this agent first (the undo point); nothing was restored")
    print(f"Undo point: {undo}\n  To put this agent back: hermes import {undo} --force --keep-host")
    auth = [m for m in members if Path(_rel(m, prefix)).name == "auth.json"]
    shared = {m for m in members if _rel(m, prefix).endswith(_SHARED_NOUS_STORE)}
    rest = [m for m in members if m not in shared and m not in auth]
    with gateway_paused(root):
        errors: list[str] = []
        yield _KeepHostZip(zf, prefix, root), rest, errors
        errors.extend(_restore_auth_members(zf, auth, prefix, root))
        removed = _prune_replaced_trees(root, {_rel(m, prefix) for m in members})
        if removed:
            print(f"  Removed {removed} skill/memory file(s) the backup does not have.")
