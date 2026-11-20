"""``hermes import --keep-host``: restore a backup over an agent whose host set part of it up.

A Hermes Cloud agent is provisioned by its platform: it is signed in to Nous Portal, its model and
tools route through Nous, its dashboard is gated by the portal's OAuth client, and its scheduler is
tuned for the container. Replacing that agent with someone's local backup must bring over
everything that makes the agent theirs (memory, persona, skills, conversations, jobs, settings, and
their other API keys and logins) while the host-owned pieces stay as the host set them.

This is not a second restore. :class:`KeepHostImport` is a per-member policy that
``hermes import`` applies inside its own checked restore (the traversal and resolved-root checks
included), under the same ``exclusive_maintenance`` reservation, so nothing here writes outside it.

* The root home's ``config.yaml``: the top-level sections in :data:`HOST_CONFIG_KEYS` keep the
  destination's value (or stay absent); every other key comes from the archive. Written with the
  comment-preserving config writer onto the archive's own document.
* The root home's ``.env``: names :func:`host_env_key` matches keep the destination's line.
* ``auth.json`` at the root and at each ``profiles/<name>/``: the destination's Nous Portal rows
  (``providers.nous``, the ``nous`` pool rows, ``active_provider`` when either side used ``nous``);
  every other provider comes from the archive. Merged under the auth-store lock against the live
  store, so a token refresh racing the restore is never rolled back.
* A named profile's ``config.yaml`` and ``.env`` arrive whole: only the root home is the host's.
* ``--messaging-off`` (the Desktop transfer passes it; an undo does not): every messaging bot the
  archive would run, in the root and in each profile, arrives with its ``platforms.<id>.enabled``
  switch off (the flag the Messaging page writes), its token and settings kept, so the source
  computer keeps answering until the user switches the bot over. Written into the same merged
  ``config.yaml`` the restore publishes.
* Never taken from the archive: ``shared/nous_auth.json``, ``webapp/`` (the Webapp's sign-in
  state), ``install_id`` and lock files.
* Replace, not merge: in the root home and in every profile the archive carries, a file the archive
  lacks is removed (except the root's ``workspace/``, ``home/`` and non-Hermes dot-entries: HOME's,
  on Hermes Cloud), and the root's ``config.yaml``/``.env``/``auth.json`` keep only their host-owned
  part. Kept: databases (restored page by page, never unlinked under a live reader), logs and the
  host's runtime records. A profile only the destination has is left exactly as it is, except on
  an undo: the undo point lists the profiles the agent had, and one the transfer brought is removed.

The undo point is a full ``hermes backup`` of the destination (memory-provider files outside the
home included), taken under the reservation after the gateway drained, in
``backups/pre-transfer-*.zip`` (the last three are kept). An import that wrote files and did not
finish leaves a marker naming its undo point, and the next keep-host import (a retry) reuses it
instead of backing up a half-restored agent. ``hermes import <undo point> --force --keep-host`` puts
the agent back.

In an s6 container (the image Hermes Cloud runs) the gateway owns the profile, and ``hermes import``
refuses ``state.db`` while it does. :func:`gateway_paused` asks the gateway to drain and exit through
its control socket (the ``pause-for-update`` verb ``hermes update`` uses) with its s6 slot marked
run-once so the supervisor does not bring it back mid-restore, then starts the slot again.
"""

from __future__ import annotations

import copy
import json
import logging
import secrets
import zipfile
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional

from hermes_cli.archive_safe import normalize_archive_parts
from hermes_cli.home_data_layout import HOME_ENV_FILES

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

_SHARED_NOUS_STORE = ("shared", "nous_auth.json")
# The machine's identity: a copy would make two installs report as one.
_HOST_ROOT_FILES = frozenset({"install_id"})
# The Webapp's sign-in state (its browser credential, native grants, the Portal callback): the
# host's Web Access, like the ``webapp`` config section.
_HOST_ROOT_DIRS = frozenset({"webapp"})
# Written by the host's running services, so a replace leaves them.
_RUNTIME_ENTRIES = frozenset({
    "logs", "state", "runtime", ".local", "spawn-ledger.json", "channel_directory.json",
    "gateway-starts.log", "plugin-update-checks",
})
_RUNTIME_FILES = frozenset({("cron", "ticker_last_success"), ("cron", "ticker_heartbeat")})
# A skill's or plugin's own *.lock is content; every other lock file is a kernel-lock inode that
# other processes hold, never restored or removed.
_CONTENT_TREES = frozenset({"skills", "plugins"})
_ROOT_HOST_FILES = frozenset({("config.yaml",), (".env",), ("auth.json",)})
# Hermes Cloud runs with HOME set to the Hermes home, so the agent's workspace and the container's
# dotfiles (.ssh, .gitconfig, .config) sit in the root home, and no backup from another machine
# carries them. A replace never removes these root entries; the archive's own copies still land.
# Hermes' own root dot-files (credentials, CLI history) are agent files and are replaced as usual.
_HOME_ENTRIES = frozenset({"workspace", "home"})
_AGENT_DOTFILES = frozenset({*HOME_ENV_FILES, ".anthropic_oauth.json", ".hermes_history"})
_DB_SUFFIXES = (".db", ".db-wal", ".db-shm", ".db-journal")

_UNDO_PREFIX = "pre-transfer-"
_UNDO_KEEP = 3
_UNFINISHED_MARKER = ".transfer-unfinished"
# In an undo point: the profiles the agent had, so the undo removes the ones the transfer brought.
_PROFILES_MANIFEST = ".keep-host-profiles.json"
# The line the dashboard's action status reads the undo point from.
UNDO_POINT_LINE = "Undo point: "
# The line it reads the messaging platforms ``--messaging-off`` switched off from (card names, ", ").
MESSAGING_OFF_LINE = "Messaging switched off: "
# The line it reads a gateway that did not start again after the restore from: the files arrived,
# but the agent's bots and scheduled jobs are not running until it does.
GATEWAY_DOWN_LINE = "Gateway did not start again: "
# Inbound HTTP endpoints, not bots: two agents can serve them at once, and the API server is the host's.
_NOT_BOTS = frozenset({"api_server", "webhook", "msgraph_webhook", "local"})
_EXTERNAL = "_external/"
_NOUS = "nous"


class KeepHostRefused(RuntimeError):
    """The keep-host import stopped before it wrote anything."""


def host_env_key(name: str) -> bool:
    return name in _HOST_ENV_NAMES or name.startswith(_HOST_ENV_PREFIXES)


def _env_name(line: str) -> str:
    body = line.strip()
    if not body or body.startswith("#") or "=" not in body:
        return ""
    return body.removeprefix("export ").split("=", 1)[0].strip()


def _host_lines(text: str) -> list[str]:
    """Every host-owned assignment, in order: .env loading takes the last, so all of them count."""
    return [line for line in text.splitlines() if host_env_key(_env_name(line))]


def merge_env(archive_text: str, host_text: str) -> str:
    """The archive's .env with every host-owned name set to the destination's line (in place),
    dropped when the destination lacks it, and the destination's other host-owned lines appended.
    The archive's text unchanged when both already agree on the host-owned lines (an undo)."""
    if _host_lines(archive_text) == _host_lines(host_text):
        return archive_text
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
        raw_ours = host_store.get(section)
        ours: dict[str, Any] = raw_ours if isinstance(raw_ours, dict) else {}
        theirs = merged.get(section)
        if not isinstance(theirs, dict):
            if _NOUS not in ours:
                continue
            theirs = merged[section] = {}
        theirs.pop(_NOUS, None)
        if _NOUS in ours:
            theirs[_NOUS] = copy.deepcopy(ours[_NOUS])
    if _NOUS in (merged.get("active_provider"), host_store.get("active_provider")):
        merged["active_provider"] = host_store.get("active_provider")
    return merged


def _bot_platforms() -> dict[str, tuple[str, tuple[str, ...]]]:
    """``{platform_id: (card name, env names that turn it on)}`` for every messaging bot."""
    from gateway.config import Platform
    from gateway.config_env import _enable_credential_names
    from hermes_cli.profile_channels import _left_core_platforms, _registry_entries, platform_ids
    entries = {entry.name: entry for entry in _registry_entries()}
    left = {feature.platform: feature for feature in _left_core_platforms()}
    bots = {}
    for pid in platform_ids():
        if pid in _NOT_BOTS:
            continue
        entry, names = entries.get(pid), set(getattr(left.get(pid), "enable_env", ()))
        names.update(getattr(entry, "required_env", None) or ())
        try:
            names.update(_enable_credential_names(Platform(pid)))
        except ValueError:
            pass  # a platform that left core and whose plugin is absent: its LEFT_CORE row names it
        label = (entry and (entry.dashboard_label or entry.label)) or getattr(left.get(pid), "label", "") or pid.title()
        bots[pid] = (label, tuple(sorted(names)))
    return bots


def _env_on(text: str) -> set[str]:
    """Names ``text`` sets to a value (an ``*_ENABLED`` switch only when it is truthy)."""
    from utils import is_truthy_value
    on = set()
    for line in text.splitlines():
        name = _env_name(line)
        value = line.split("=", 1)[1].strip().strip("'\"") if name else ""
        if value and (not name.endswith("_ENABLED") or is_truthy_value(value)):
            on.add(name)
    return on


def _platform_blocks(cfg: dict[str, Any], pid: str) -> list[dict[str, Any]]:
    """Every block the gateway reads ``enabled`` from for ``pid`` (gateway/config_loader.py)."""
    gateway = cfg.get("gateway") if isinstance(cfg.get("gateway"), dict) else {}
    holders = (cfg.get("platforms"), cfg, gateway, gateway.get("platforms"))
    return [h[pid] for h in holders if isinstance(h, dict) and isinstance(h.get(pid), dict)]


def messaging_off(cfg: dict[str, Any], env_text: str,
                  bots: Optional[dict[str, tuple[str, tuple[str, ...]]]] = None) -> tuple[dict[str, Any], list[str]]:
    """``cfg`` with every messaging bot it would run switched off, and the names switched off.

    A bot runs when a block sets ``enabled: true``, or when none sets ``enabled`` and ``env_text``
    holds its token (the gateway enables it from env). Off is ``platforms.<id>.enabled: false``, the
    switch the Messaging page writes, and no ``enabled`` in any other block: a later block wins in the
    gateway's merge, so a ``false`` left there would outvote the page's switch. Tokens and every other
    setting are left as they are."""
    from utils import is_truthy_value
    bots = _bot_platforms() if bots is None else bots
    out, env, off = copy.deepcopy(cfg), _env_on(env_text), []
    for pid, (label, names) in bots.items():
        blocks = _platform_blocks(out, pid)
        flags = [block["enabled"] for block in blocks if "enabled" in block]
        if not (any(is_truthy_value(flag) for flag in flags) or (not flags and env.intersection(names))):
            continue
        for block in blocks:
            block.pop("enabled", None)
        if not isinstance(out.get("platforms"), dict):
            out["platforms"] = {}
        if not isinstance(out["platforms"].get(pid), dict):
            out["platforms"][pid] = {}
        out["platforms"][pid]["enabled"] = False
        off.append(label)
    return out, off


def _yaml_map(raw: bytes, what: str) -> dict[str, Any]:
    import hermes_yaml as yaml
    text = raw.decode("utf-8-sig")
    data = yaml.safe_load(text) if text.strip() else None  # config-reader: ok — raw keep-host merge of one file
    if data is not None and not isinstance(data, dict):
        raise ValueError(f"{what} is not a mapping")
    return dict(data or {})


def _auth_store(raw: bytes) -> dict[str, Any]:
    data = json.loads(raw.decode("utf-8-sig") or "{}")
    if not isinstance(data, dict):
        raise ValueError("the backup's auth.json is not an auth store")
    return data


def _live(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return b""


def _home_parts(parts: tuple[str, ...]) -> tuple[str, ...]:
    """``parts`` relative to the profile home they sit in (the root or ``profiles/<name>``)."""
    return parts[2:] if len(parts) >= 3 and parts[0] == "profiles" else parts


def _is_auth_store(parts: tuple[str, ...]) -> bool:
    return parts == ("auth.json",) or (len(parts) == 3 and parts[0] == "profiles" and parts[2] == "auth.json")


def kept_summary(root: Path) -> list[str]:
    """Human lines for what a keep-host restore leaves as the destination has it."""
    try:
        cfg = _yaml_map(_live(root / "config.yaml"), "config.yaml")
        store = _auth_store(_live(root / "auth.json"))
    except (OSError, ValueError) as exc:
        return [f"  (this agent's settings could not be read: {exc})"]
    env = _live(root / ".env").decode("utf-8", "replace")
    names = sorted({n for n in map(_env_name, env.splitlines()) if n and host_env_key(n)})
    signed_in = _NOUS in (store.get("providers") or {})
    return [f"  config.yaml: {', '.join(k for k in HOST_CONFIG_KEYS if k in cfg) or '(none set here)'}",
            f"  .env: {', '.join(names) or '(none set here)'}",
            f"  auth.json: {'its Nous Portal sign-in' if signed_in else '(not signed in to Nous Portal)'}"]


def create_undo_point(root: Path, external_members: tuple[str, ...] = ()) -> Optional[Path]:
    """A full backup of ``root`` as ``hermes backup`` collects it (the memory provider's files outside
    the home included), plus the existing files the archive's ``_external/`` members will overwrite.
    None when it could not be written whole."""
    from hermes_cli.backup import _collect_external_entries, _write_full_zip_backup
    entries = {arcname: path for path, arcname in _collect_external_entries()[0]}
    home = Path.home().resolve()
    for member in external_members:
        target = home / member.removeprefix(_EXTERNAL)
        try:
            inside = target.resolve().is_relative_to(home)
        except (OSError, ValueError):
            continue  # the import refuses the member too
        if inside and target.is_symlink():
            logger.warning("Undo point cannot include linked external state: %s", member)
            return None
        if inside and target.is_file():
            entries.setdefault(member, target)
    from utils import atomic_write_text
    backups = root / "backups"
    backups.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest = backups / _PROFILES_MANIFEST
    atomic_write_text(manifest, json.dumps(sorted(_profile_names(root))), mode=0o600)
    entries[_PROFILES_MANIFEST] = manifest
    out = backups / f"{_UNDO_PREFIX}{datetime.now():%Y-%m-%d-%H%M%S}-{secrets.token_hex(3)}.zip"
    return _write_full_zip_backup(out, root, [(path, Path(arc)) for arc, path in entries.items()])


def _profile_names(root: Path) -> set[str]:
    from hermes_constants import PROFILE_ID_RE, named_profile_has_identity
    profiles = root / "profiles"
    return {p.name for p in profiles.iterdir() if PROFILE_ID_RE.match(p.name) and not p.is_symlink()
            and p.is_dir() and named_profile_has_identity(p)} if profiles.is_dir() else set()


def _merge_auth_store(target: Path, archive: dict[str, Any], publish_archive: Optional[Callable[[Path], None]]) -> None:
    """``merge_auth`` against the live store under its lock, written by the auth store's writer;
    ``publish_archive`` writes the archive's own bytes when the merge changes nothing (an undo)."""
    from hermes_cli.auth import _auth_store_lock, _load_auth_store, _save_auth_store
    from hermes_cli.backup_restore import _auth_restore_target
    store = _auth_restore_target(target)
    if store is None:
        raise OSError("auth.json is hard-linked or cannot be resolved; it was left as it is")
    with _auth_store_lock(target_path=store):
        merged = merge_auth(archive, _load_auth_store(store) if store.exists() else {})
        if publish_archive is not None and merged == archive:
            publish_archive(store)
        else:
            _save_auth_store(merged, target_path=store)


class KeepHostImport:
    """The keep-host policy ``_import_members_exclusive`` applies to each member it has checked,
    plus :meth:`begin` (before the first write) and :meth:`finish` (after the last), all under the
    import's reservation."""

    def __init__(self, root: Path, members: list[str], prefix: str, source: Path,
                 had_profiles: Optional[set[str]] = None, messaging_off: bool = False):
        self.root, self.source = root, source.resolve()
        # ``--messaging-off``: the bot names switched off, per home, in archive order.
        self.messaging_off = messaging_off
        self.switched_off: dict[str, list[str]] = {}
        self.rels: set[str] = set()
        self.profiles: set[str] = set()
        self.external = tuple(m for m in members if m.startswith(_EXTERNAL))
        self.undo: Optional[Path] = None
        # The restore's error list (shared with ``run_import``) and the profiles to remove after it.
        self.errors: list[str] = []
        self.tombstoned: list[str] = []
        for member in members:
            if member in self.external:
                continue
            try:
                parts = normalize_archive_parts(member[len(prefix):] if prefix and member.startswith(prefix) else member)
            except ValueError:
                continue  # reported by the import as a blocked traversal
            self.rels.add("/".join(parts))
            if len(parts) >= 3 and parts[0] == "profiles" and not self.skips(tuple(parts)):
                self.profiles.add(parts[1])
        # Restoring an undo point: a profile the agent did not have then came with the transfer.
        self.created = (_profile_names(root) - had_profiles - self.profiles) if had_profiles is not None else set()
        # Where HOME is the Hermes home (the Cloud image), external files land inside it.
        home, base = Path.home().resolve(), root.resolve()
        for member in self.external:
            target = home / member.removeprefix(_EXTERNAL)
            if target.is_relative_to(base):
                self.rels.add(target.relative_to(base).as_posix())

    def homes(self) -> list[Path]:
        """Every profile home the import writes into or removes, reserved for the whole import."""
        return [self.root / "profiles" / name for name in sorted(self.profiles | self.created)]

    def around(self, restore: Callable[..., tuple]) -> Callable[..., tuple]:
        """``restore`` with this policy: the undo point first, the replace after the last write."""
        def run(*args):
            self.begin()
            result = restore(*args, self)
            self.errors = result[2]
            self.errors.extend(self.finish(self.errors))
            return result
        return run

    def refuse_linked_folders(self) -> None:
        """Every folder the archive writes into must be a real folder of this home: through a link the
        restore would land elsewhere, and a backup never carries what is behind one."""
        from hermes_cli.backup import _is_link_path
        folders = {Path(rel).parent for rel in self.rels} | {Path("profiles", name, "x") for name in self.created}
        for folder in {Path(*f.parts[:n]) for f in folders for n in range(1, len(f.parts) + 1)}:
            if _is_link_path(self.root / folder):
                raise KeepHostRefused(f"{folder.as_posix()} is a linked folder; nothing was restored. "
                                      "Replace the link with a real folder, or import without --keep-host")

    def _marker(self) -> Path:
        return self.root / "backups" / _UNFINISHED_MARKER

    def _unfinished_undo(self) -> Optional[Path]:
        try:
            undo = Path(self._marker().read_text(encoding="utf-8-sig").strip())
        except (OSError, ValueError):
            return None
        # Undoing the unfinished import itself takes a fresh undo point of the half-restored agent.
        if not undo.name.startswith(_UNDO_PREFIX) or undo.resolve() == self.source or not zipfile.is_zipfile(undo):
            return None
        return undo

    def begin(self) -> None:
        """Take the undo point (or reuse an unfinished import's) and mark the import unfinished."""
        from utils import atomic_write_text
        reused = self._unfinished_undo()
        try:
            self.undo = reused or create_undo_point(self.root, self.external)
            if self.undo is not None:
                atomic_write_text(self._marker(), str(self.undo), mode=0o600)
        except OSError as exc:
            raise KeepHostRefused(f"could not back up this agent first (the undo point): {exc}; "
                                  "nothing was restored") from exc
        if self.undo is None:
            raise KeepHostRefused("could not back up this agent first (the undo point); nothing was restored")
        if reused:
            print("  An earlier transfer into this agent did not finish; its undo point still applies.")
        print(f"{UNDO_POINT_LINE}{self.undo}\n  To put this agent back: hermes import {self.undo} --force --keep-host")

    def skips(self, parts: tuple[str, ...]) -> bool:
        """Members the destination keeps its own copy of."""
        home = _home_parts(parts)
        return (parts[-2:] == _SHARED_NOUS_STORE or (len(parts) == 1 and parts[0] in _HOST_ROOT_FILES)
                or parts == (_PROFILES_MANIFEST,)
                or (len(parts) > 1 and parts[0] in _HOST_ROOT_DIRS)
                or (parts[-1].endswith(".lock") and home[0] not in _CONTENT_TREES))

    def publish(self, zf: zipfile.ZipFile, member: str, parts: tuple[str, ...], target: Path,
                mode: Optional[int]) -> bool:
        """Write a host-owned member merged with the destination's copy; False for any other member."""
        from hermes_yaml import YAMLError
        if parts == ("config.yaml",):
            handler = partial(_publish_config, policy=self)
        elif self.messaging_off and _home_parts(parts) == ("config.yaml",):
            handler = partial(_publish_profile_config, policy=self)
        elif parts == (".env",):
            handler = _publish_env
        elif _is_auth_store(parts):
            handler = _publish_auth
        else:
            return False
        try:
            handler(zf, member, target, mode)
        except (ValueError, RuntimeError, YAMLError) as exc:
            raise OSError(str(exc)) from exc
        return True

    def bots_off(self, home: str, cfg: dict[str, Any], env_text: str) -> dict[str, Any]:
        """``cfg`` with its bots switched off under ``--messaging-off`` (as it is without the flag)."""
        if not self.messaging_off:
            return cfg
        out, off = messaging_off(cfg, env_text)
        if off:
            self.switched_off[home] = off
        return out

    def report_messaging_off(self) -> None:
        names = list(dict.fromkeys(name for off in self.switched_off.values() for name in off))
        if not names:
            return
        print(f"{MESSAGING_OFF_LINE}{', '.join(names)}")
        for home, off in sorted(self.switched_off.items()):
            print(f"  {home}: {', '.join(off)} (token and settings kept; turn on on the Messaging page)")

    def finish(self, errors: list[str]) -> list[str]:
        """After a restore with no errors, clear what the archive lacks; return problems."""
        problems = [] if errors else self._clear_absent_host_files()
        if not errors and not problems:
            try:
                removed = self._prune()
            except OSError as exc:
                problems.append(f"  removing files the backup does not have: {exc}")
            else:
                if removed:
                    print(f"  Removed {removed} file(s) the backup does not have.")
        if not errors and not problems and self.messaging_off:
            problems = self._profiles_without_config_bots_off()
        self.report_messaging_off()
        if not errors and not problems and self.created:
            self._tombstone_created_profiles()
        left = sorted(p.name for p in (self.root / "profiles").glob("*")
                      if p.is_dir() and p.name not in self.profiles | self.created)
        if left:
            print(f"  Left as they are (not in this backup): profile(s) {', '.join(left)}")
        if errors or problems:
            print(f"\n  The restore did not finish. To put this agent back: hermes import {self.undo} --force --keep-host")
            return problems
        from hermes_cli.backup import _prune_prefixed_zips
        self._marker().unlink(missing_ok=True)
        _prune_prefixed_zips(self.root / "backups", _UNDO_PREFIX, _UNDO_KEEP, "pre-transfer backup")
        return []

    def _root_env(self) -> str:
        """The root .env as the restore leaves it: already published, or about to keep only the host's lines."""
        live = _live(self.root / ".env").decode("utf-8", "replace")
        return live if ".env" in self.rels else merge_env("", live)

    def _profiles_without_config_bots_off(self) -> list[str]:
        """A profile the archive brings without a config.yaml runs the bots its .env has tokens for."""
        from hermes_cli.config import atomic_config_replace
        problems = []
        for name in sorted(self.profiles):
            home = self.root / "profiles" / name
            if f"profiles/{name}/config.yaml" in self.rels:
                continue
            cfg = self.bots_off(f"profile {name}", {}, _live(home / ".env").decode("utf-8", "replace"))
            try:
                if cfg:
                    atomic_config_replace(home / "config.yaml", cfg)
            except OSError as exc:
                problems.append(f"  profiles/{name}/config.yaml: {exc}")
        return problems

    def _tombstone_created_profiles(self) -> None:
        """Mark the profiles the undone transfer brought deleted, under the reservation, so nothing
        starts in them; :meth:`remove_created_profiles` removes them once it is released."""
        from hermes_constants import mark_named_profile_deleted
        for name in sorted(self.created):
            mark_named_profile_deleted(self.root / "profiles" / name)
        self.tombstoned = sorted(self.created)

    def remove_created_profiles(self) -> None:
        """Remove the tombstoned profiles through ``hermes profile delete``'s own flow (services,
        multiplexer, identity, the owner wait), after the import released their ``gateway.lock``:
        removing a home whose lock this process holds fails on Windows and frees the lock on POSIX."""
        from hermes_cli.profiles import ProfileIdentitySettlementPending, delete_profile
        for name in self.tombstoned:
            try:
                delete_profile(name, yes=True)
                print(f"  Removed profile {name} (the transfer brought it).")
            except ProfileIdentitySettlementPending as exc:  # the profile is gone; only its identity is not
                print(f"  Removed profile {name} (the transfer brought it). To finish: {exc.retry_command}")
            except (OSError, RuntimeError, ValueError) as exc:
                self.errors.append(f"  profiles/{name}: {exc}")
        if self.tombstoned and self.errors:
            print(f"\n  The restore did not finish. To put this agent back: hermes import {self.undo} --force --keep-host")

    def _clear_absent_host_files(self) -> list[str]:
        """A host file the archive does not carry keeps only its host-owned part."""
        from hermes_cli.config import _env_write_lock, atomic_config_replace
        from hermes_yaml import YAMLError
        from utils import atomic_write_text
        config, env = self.root / "config.yaml", self.root / ".env"
        steps = [(f"profiles/{name}/auth.json", self.root / "profiles" / name / "auth.json")
                 for name in sorted(self.profiles)]
        problems = []
        for rel, target in [("auth.json", self.root / "auth.json"), *steps, ("config.yaml", config), (".env", env)]:
            if rel in self.rels or not _live(target):
                continue
            try:
                if target is config:
                    kept = merge_config({}, _yaml_map(_live(config), "config.yaml"))
                    atomic_config_replace(config, self.bots_off("root", kept, self._root_env()))
                elif target is env:
                    with _env_write_lock(env):
                        atomic_write_text(env, merge_env("", _live(env).decode("utf-8")), preserve_mode=True)
                else:
                    _merge_auth_store(target, {}, None)
            except (OSError, ValueError, RuntimeError, YAMLError) as exc:
                problems.append(f"  {rel}: {exc}")
        return problems

    def _replaceable(self, parts: tuple[str, ...]) -> bool:
        """Whether a destination file the archive lacks goes."""
        from hermes_cli.backup import _import_skipped
        if parts[0] == "profiles" and (len(parts) < 3 or parts[1] not in self.profiles):
            return False
        if parts[0] in _HOME_ENTRIES or (parts[0].startswith(".") and parts[0] not in _AGENT_DOTFILES):
            return False
        home = _home_parts(parts)
        return not (parts in _ROOT_HOST_FILES or _is_auth_store(parts) or self.skips(parts)
                    or home[0] in _RUNTIME_ENTRIES or home in _RUNTIME_FILES
                    or parts[-1].endswith(_DB_SUFFIXES) or _import_skipped("/".join(parts)))

    def _prune(self) -> int:
        """Remove what the archive does not carry. The walk is the backup's own: it never enters a
        linked folder and yields regular files only, so nothing behind a link is touched."""
        from hermes_cli.backup import _iter_backup_files
        homes = {self.root, self.root / "profiles", *(self.root / "profiles" / n for n in self.profiles)}
        stale = [path for path, rel in _iter_backup_files(self.root, self.root / "backups" / "none.zip")
                 if rel.as_posix() not in self.rels and self._replaceable(rel.parts)]
        for path in stale:
            path.unlink()
        for folder in sorted({p.parent for p in stale}, key=lambda p: len(p.parts), reverse=True):
            while folder not in homes and folder.is_dir() and not any(folder.iterdir()):
                folder.rmdir()
                folder = folder.parent
        return len(stale)


def _archive_env(zf: zipfile.ZipFile, config_member: str) -> str:
    """The .env beside ``config_member`` in the archive ("" when it carries none)."""
    try:
        return zf.read(config_member[: -len("config.yaml")] + ".env").decode("utf-8-sig", "replace")
    except KeyError:
        return ""


def _publish_profile_config(zf: zipfile.ZipFile, member: str, target: Path, mode: Optional[int], *,
                            policy: KeepHostImport) -> None:
    """A named profile's config.yaml, whole, with its bots switched off (``--messaging-off``)."""
    archive = _yaml_map(zf.read(member), "the backup's config.yaml")
    _write_config_member(zf, member, target, mode, archive,
                         policy.bots_off(f"profile {target.parent.name}", archive, _archive_env(zf, member)))


def _publish_config(zf: zipfile.ZipFile, member: str, target: Path, mode: Optional[int], *,
                    policy: KeepHostImport) -> None:
    archive = _yaml_map(zf.read(member), "the backup's config.yaml")
    merged = merge_config(archive, _yaml_map(_live(target), "this agent's config.yaml"))
    _write_config_member(zf, member, target, mode, archive, policy.bots_off("root", merged, _archive_env(zf, member)))


def _write_config_member(zf: zipfile.ZipFile, member: str, target: Path, mode: Optional[int],
                         archive: dict[str, Any], merged: dict[str, Any]) -> None:
    from hermes_cli.backup_restore import _extract_member_atomically
    from hermes_cli.config import atomic_config_replace
    from utils import atomic_replace
    if merged == archive:
        _extract_member_atomically(zf, member, target, mode)
        return
    # The archive's own document (and its comments) with the host's sections written into it.
    staged = target.with_name(".config.yaml.keep-host")
    try:
        _extract_member_atomically(zf, member, staged, mode)
        atomic_config_replace(staged, merged)
        atomic_replace(staged, target)
    finally:
        staged.unlink(missing_ok=True)


def _publish_env(zf: zipfile.ZipFile, member: str, target: Path, mode: Optional[int]) -> None:
    from hermes_cli.backup_restore import _extract_member_atomically
    from hermes_cli.config import _env_write_lock
    from utils import atomic_write_text
    archive = zf.read(member).decode("utf-8-sig")
    with _env_write_lock(target):
        merged = merge_env(archive, _live(target).decode("utf-8"))
        if merged == archive:
            _extract_member_atomically(zf, member, target, mode)
        else:
            atomic_write_text(target, merged, preserve_mode=True)


def _publish_auth(zf: zipfile.ZipFile, member: str, target: Path, mode: Optional[int]) -> None:
    from hermes_cli.backup_restore import _extract_member_atomically
    _merge_auth_store(target, _auth_store(zf.read(member)),
                      lambda store: _extract_member_atomically(zf, member, store, mode))


def _s6_slot(home: Path) -> Optional[str]:
    """The s6 slot whose gateway serves ``home`` and is up; None outside an s6 container."""
    from hermes_cli.service_manager import S6_SERVICE_PREFIX, detect_service_manager, get_service_manager
    from hermes_constants import profile_name_for_home
    if detect_service_manager() != "s6":
        return None
    slot = f"{S6_SERVICE_PREFIX}{profile_name_for_home(home) or 'default'}"
    try:
        return slot if get_service_manager().is_running(slot) else None
    except Exception as exc:  # health: allow BLE001 -- a failed probe refuses; it never means "not running"
        raise KeepHostRefused(f"could not tell whether the gateway ({slot}) is running: {exc}; "
                              "nothing was restored") from exc


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
    from hermes_cli.service_manager import S6Error, S6ServiceManager

    slot = _s6_slot(home)
    if slot is None:
        yield
        return
    manager = S6ServiceManager()
    try:
        manager._run_svc("-O", "pause", slot)
        ack = pause_gateway_for_update(home, timeout=8)
        if not (ack and (ack.get("pausing") or ack.get("already_stopping"))):
            raise OwnershipConflict("the gateway did not accept the pause; nothing was restored")
        print(f"Paused the gateway ({slot}) for the restore ...")
        if not wait_profile_unowned(home, timeout=float(ack.get("drain_timeout") or 30) + 30):
            raise OwnershipConflict("the gateway is still finishing work; nothing was restored. Retry shortly")
        yield
    finally:
        try:
            manager.start(slot)
            print(f"Started the gateway ({slot}) again.")
        except (S6Error, OSError) as exc:
            print(f"{GATEWAY_DOWN_LINE}{slot}: {exc}\n  Start it with: hermes gateway start")


def _had_profiles(zf: zipfile.ZipFile, prefix: str) -> Optional[set[str]]:
    try:
        names = json.loads(zf.read(f"{prefix}{_PROFILES_MANIFEST}").decode("utf-8-sig"))
    except KeyError:
        return None
    except ValueError as exc:
        raise KeepHostRefused(f"the undo point's profile list is unreadable ({exc}); nothing was restored") from exc
    return {str(n) for n in names} if isinstance(names, list) else None


@contextmanager
def keep_host_import(args, zf: zipfile.ZipFile, members: list[str], prefix: str, root: Path):
    """The :class:`KeepHostImport` ``run_import`` restores with (None without ``--keep-host``), the
    profile's gateway drained around it. :class:`KeepHostRefused` means nothing was written."""
    if not getattr(args, "keep_host", False):
        if getattr(args, "messaging_off", False):
            raise KeepHostRefused("--messaging-off needs --keep-host")
        yield None
        return
    policy = KeepHostImport(root, members, prefix, Path(args.zipfile).expanduser(), _had_profiles(zf, prefix),
                            messaging_off=getattr(args, "messaging_off", False))
    policy.refuse_linked_folders()
    print("\nKeeping what this host set up (--keep-host):")
    print("\n".join(kept_summary(root)))
    with gateway_paused(root):
        yield policy
        policy.remove_created_profiles()
