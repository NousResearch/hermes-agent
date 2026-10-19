"""config.d — user-owned drop-in config overrides, layered between config.yaml and managed scope.

DISTINCT from ``hermes_cli.managed_scope``: drop-ins are the USER's own files (editable, not
immutable), so they sit BELOW admin-pinned values — ``DEFAULT_CONFIG < config.yaml <
config.d/*.yaml (lexicographic) < managed scope``. Every ``*.yaml``/``*.yml`` under
``<config.yaml's parent>/config.d/`` deep-merges OVER ``config.yaml`` in filename order, so shared
config can be symlinked per profile with small per-profile overrides. The dir derives from the
CONFIG PATH, not ``get_hermes_home()``, so the home-parameterized callers resolve the probed
profile's drop-ins and profile switching needs no env var — profiles are just HERMES_HOMEs.

Write-back paths (``read_raw_config``, ``read_user_config_raw``, ``save_config``, ``config set``)
deliberately stay config.yaml-ONLY: a drop-in that leaked into a load -> mutate -> save cycle would
freeze its values INTO config.yaml, shadowed by the drop-in forever. Bulk saves that pass a
``load_config()`` result strip drop-in keys via ``config._strip_config_d_keys_for_save``.

Fail-open like the managed layer, and LOUD: a broken drop-in is skipped with a warning naming the
file — silence would hide a dropped override, but one malformed file must not brick startup.

``hermes_cli.config`` imports this module at module scope, so the ``hermes_cli.config`` helpers
used below are imported inside each function to break the cycle (``managed_scope`` does the same).
"""
from __future__ import annotations

import copy
import logging
import sys
import threading
from pathlib import Path
from typing import Optional

# Stale-module bridge: this module binds ``utils.file_signature`` at import time, so a fresh
# import in a post-pull updater process (pre-handoff purge keeps root modules cached) dies
# unless the stale ``utils`` is dropped first. See hermes_cli.stale_modules.
from hermes_cli.stale_modules import drop_stale_root_modules

drop_stale_root_modules()

from utils import fast_safe_load, file_signature

logger = logging.getLogger(__name__)

CONFIG_D_SUFFIXES = (".yaml", ".yml")

# Sentinel key in config_d_key_sources() naming drop-ins that failed to parse. It is deliberately
# not a real dotted config key so it can never collide with a setting, and it tells the save guard
# that config.d holds an override it cannot read.
UNPARSEABLE_DROP_IN_KEY = "\x00unparseable-config.d-drop-in"

_CACHE_LOCK = threading.Lock()
# dir path -> (drop-in signature, merged overlay); see config_d_signature for the key.
_CONFIG_D_CACHE: dict[str, tuple] = {}
# Dangling-symlink paths already warned about: a moved shared drop-in would otherwise log on
# every signature probe (the hot path stats the dir once per cached load).
_WARNED_DANGLING: set[str] = set()


def get_config_d_dir(config_path: Optional[Path] = None) -> Path:
    """``<config.yaml's parent>/config.d``."""
    from hermes_cli.config import get_config_path
    return (config_path or get_config_path()).parent / "config.d"


def config_d_files(d: Path) -> list[Path]:
    """Drop-in files in *d*, lexicographic by name; ``[]`` when the dir is absent.

    ``*.yaml``/``*.yml`` case-insensitively (Windows), never dotfiles (editor backups), regular
    files only — ``is_file()`` follows symlinks, the core use case (a shared drop-in symlinked
    from ``~/agents/``), and skips directories named ``*.yaml``. A DANGLING symlink is warned
    about once and skipped.
    """
    try:
        entries = sorted(d.iterdir(), key=lambda p: p.name)
    except OSError as exc:
        if not d.is_dir():
            return []  # absent dir: an empty overlay, the common case
        logger.error(
            "config.d: %s exists but could not be listed (%s) — IGNORING every drop-in in it, so "
            "NO config.d override is being applied. Fix the directory's permissions (or rename it "
            "aside) and restart.", d, exc, exc_info=True)
        return []
    files: list[Path] = []
    for p in entries:
        name = p.name
        if name.startswith(".") or not name.lower().endswith(CONFIG_D_SUFFIXES):
            continue
        try:
            if p.is_file():
                files.append(p)
            elif p.is_symlink() and not p.exists() and str(p) not in _WARNED_DANGLING:
                _WARNED_DANGLING.add(str(p))
                logger.warning(
                    "config.d: %s is a dangling symlink — IGNORING it because the shared config it "
                    "pointed at has moved. Re-point it at the shared file or remove it; the other "
                    "drop-ins still apply.", p)
                _warn_to_terminal(
                    f"hermes config.d: {p} is a dangling symlink — IGNORING it because the shared "
                    "config it pointed at has moved. Re-point it at the shared file or remove it.")
        except OSError as exc:
            # An ACL-denied or otherwise unstat-able entry must not crash the config hot path;
            # skipping it silently would hide a dropped override, so warn.
            logger.warning(
                "config.d: could not inspect %s (%s) — IGNORING it, so any override it sets is NOT "
                "applied. Fix its permissions or remove it; the other drop-ins still apply.",
                p, exc)
            _warn_to_terminal(
                f"hermes config.d: could not inspect {p} — IGNORING it, so any override it sets "
                "is NOT applied. Fix its permissions or remove it.")
    return files


def config_d_signature(d: Path) -> tuple:
    """Change key for the drop-in set: ``((name, *file_signature(stat)), ...)`` sorted by name.

    Catches every edit class the merged result depends on: in-place edit (file signature),
    add/remove (tuple length), rename with identical content (the name set). ``stat()`` follows
    symlinks, so editing the SHARED target invalidates every profile linking it. ``()`` when the
    dir is absent — one extra probe on the hot path, the managed layer's cost class.
    """
    sig: list[tuple] = []
    for p in config_d_files(d):
        try:
            sig.append((p.name, *file_signature(p.stat())))
        except OSError:
            # Raced with a delete mid-scan: a zeroed entry still differs from the file being
            # present, so the next load re-parses. Never crash the hot path over it.
            sig.append((p.name, 0, 0, 0, 0))
    return tuple(sig)


def _warn_to_terminal(message: str) -> None:
    """Echo a drop-in problem to stderr as well as the log.

    ``hermes_logging`` installs file handlers only, so a log-only warning is invisible to a user
    we are telling to fix a file and restart. Same contract as
    ``hermes_cli.config_read_errors._warn_config_parse_failure`` for a broken config.yaml.
    """
    try:
        sys.stderr.write(f"⚠️  {message}\n")
        sys.stderr.flush()
    except (OSError, ValueError):
        # A closed or redirected stderr must never take the config load down with it.
        pass


def _load_drop_in_file(path: Path) -> Optional[dict]:
    """One drop-in parsed RAW, or None to skip it (fail-open + loud).

    A parse failure or non-mapping top level must not brick startup, but silence would hide a
    dropped override, so the warning names the file, the cause and the fix. An empty file is a
    valid empty layer, not an error.
    """
    # Broad catch by design: a user-owned drop-in can fail in ways we cannot enumerate
    # (YAMLError, UnicodeDecodeError, OSError on an ACL-denied file, ...), and every one of
    # them means "skip this file", not "stop loading config".
    try:
        parsed = fast_safe_load(path.read_text(encoding="utf-8-sig"))
    except Exception:
        logger.exception(
            "config.d: could not read %s — IGNORING this drop-in, so its overrides are NOT "
            "applied. Fix or remove the file and restart; the other drop-ins still apply.", path)
        _warn_to_terminal(
            f"hermes config.d: could not read {path} — IGNORING it, so its overrides are NOT "
            "applied. Fix or remove the file and restart; the other drop-ins still apply.")
        return None
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        logger.warning(
            "config.d: %s has a top-level %s, not a mapping — IGNORING this drop-in, so its "
            "overrides are NOT applied. Wrap its settings under top-level keys and restart; "
            "the other drop-ins still apply.", path, type(parsed).__name__)
        return None
    return parsed


def load_config_d(config_path: Optional[Path] = None) -> dict:
    """Merged RAW drop-in overlay (unexpanded), or ``{}`` when no drop-ins exist.

    Each caller expands ``${VAR}`` refs at its own existing expansion point, so the overlay is
    cached unexpanded on ``(dir, signature)`` (mirroring ``managed_scope._cached_read``) and
    returned as a deepcopy. Files merge in lexicographic order, so a later filename wins.
    """
    d = get_config_d_dir(config_path)
    sig = config_d_signature(d)
    if not sig:
        return {}
    dir_key = str(d)
    with _CACHE_LOCK:
        hit = _CONFIG_D_CACHE.get(dir_key)
        if hit is not None and hit[0] == sig:
            return copy.deepcopy(hit[1])
    from hermes_cli.config import _deep_merge
    merged: dict = {}
    for path in config_d_files(d):
        parsed = _load_drop_in_file(path)
        if parsed:
            merged = _deep_merge(merged, parsed)
    with _CACHE_LOCK:
        _CONFIG_D_CACHE[dir_key] = (sig, copy.deepcopy(merged))
    return merged


def apply_config_d_overlay(config: dict, config_path: Optional[Path] = None) -> dict:
    """``_deep_merge(config, load_config_d(config_path))`` — the one call sites use.

    Drop-ins win at the leaf; sibling keys they don't mention stay. Returns a NEW dict, so
    callers must reassign. Refs stay UNEXPANDED: each caller expands at its own point.
    """
    from hermes_cli.config import _deep_merge
    drop_ins = load_config_d(config_path)
    return _deep_merge(config, drop_ins) if drop_ins else config


def _override_leaves(mapping: dict, prefix: str = "") -> list[str]:
    """Dotted leaves a drop-in REALLY sets, in file order; an empty dict claims nothing.

    ``_deep_merge`` merges ``{}`` as a no-op (the sibling keys in config.yaml survive), so
    attributing the section itself to the drop-in made the save guard strip a whole section the
    drop-in never overrode — load and save must agree on what a drop-in owns.
    """
    leaves: list[str] = []
    for key, value in mapping.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            if value:
                leaves.extend(_override_leaves(value, dotted))
        else:
            leaves.append(dotted)
    return leaves


def config_d_key_sources() -> dict[str, list[str]]:
    """``{dotted leaf key: [drop-in filenames setting it]}`` — who overrides what.

    Re-parses each drop-in because the merged overlay loses per-file attribution. Used by
    ``hermes config set`` to name the file overriding a just-written value, and by the bulk-save
    guard.

    Names the leaves through ``_override_leaves`` rather than ``managed_scope._flatten_keys``:
    that helper also collapses an empty mapping to the SECTION's own name, which made the save
    guard strip a section a ``display: {}`` drop-in never overrode.

    A drop-in that FAILS to parse is reported under its own sentinel key rather than skipped: the
    save guard must still know a file in config.d claims authority over this profile, or a bulk
    save would freeze that file's intended value into config.yaml — the exact failure
    strip_config_d_keys_for_save exists to prevent. The parse warning has already been logged by
    _load_drop_in_file, so this only records the attribution.
    """
    sources: dict[str, list[str]] = {}
    for path in config_d_files(get_config_d_dir()):
        parsed = _load_drop_in_file(path)
        if parsed is None:
            sources.setdefault(UNPARSEABLE_DROP_IN_KEY, []).append(path.name)
            continue
        for key in _override_leaves(parsed):
            sources.setdefault(key, []).append(path.name)
    return sources


def invalidate_config_d_cache() -> None:
    """Drop cached drop-in overlays. For tests and post-edit reloads."""
    with _CACHE_LOCK:
        _CONFIG_D_CACHE.clear()


def strip_config_d_keys_for_save(config: dict) -> dict:
    """Drop every leaf a drop-in sets from *config* and say what was not saved (bulk safety net).

    A bulk save that passed a ``load_config()`` result (dashboard ``web_server_config.py``, setup
    flows) would otherwise freeze drop-in values INTO config.yaml — where they would sit shadowed
    by the drop-in forever, breaking the shared-config DRY use case. Mirrors
    ``_strip_managed_keys_for_save``, but a drop-in is the user's own file, so the note explains
    the override instead of hard-refusing. Single-key ``config set`` reads the raw file and is
    already drop-in-free.

    Raises RuntimeError when a drop-in cannot be parsed: its keys are unknown, so the guard cannot
    tell which leaves it would shadow, and guessing would either freeze its values into
    config.yaml or silently drop the caller's edit. The caller surfaces this as a clean refusal.
    """
    key_sources = config_d_key_sources()
    unparseable = key_sources.pop(UNPARSEABLE_DROP_IN_KEY, None)
    if unparseable:
        raise RuntimeError(
            f"config.d drop-in(s) {', '.join(sorted(unparseable))} could not be read, so their "
            "settings are unknown and this save was refused rather than risk overwriting them. "
            "Fix or remove those files and retry; see the warning above for the parse error.")
    if not key_sources:
        return config
    from hermes_cli.config import _strip_dotted_keys
    config, stripped = _strip_dotted_keys(copy.deepcopy(config), set(key_sources))
    if stripped:
        print(
            f"Note: {len(stripped)} drop-in setting(s) were not saved to config.yaml "
            f"(a config.d drop-in overrides them): " + "; ".join(
                f"{key} (set by {', '.join(key_sources[key])})" for key in sorted(stripped))
            + ". Edit the named drop-in to change them.",
            file=sys.stderr)
    return config


def show_config_d_banner() -> None:
    """List the active drop-in files — where a user looks when "my config.yaml value isn't
    taking effect": a drop-in silently overrides it, and nothing else names it."""
    drop_in_dir = get_config_d_dir()
    files = config_d_files(drop_in_dir)
    if not files:
        return
    from hermes_cli.colors import Colors, color
    print()
    print(color(
        f"  ⧉ Drop-in overrides active from {drop_in_dir} "
        f"(applied over config.yaml, in filename order):", Colors.YELLOW, Colors.BOLD))
    for path in files:
        print(color(f"    {path.name}", Colors.YELLOW))


def print_config_d_override_notice(key: str) -> None:
    """Name the drop-in that still overrides a key just written to config.yaml.

    Non-blocking: unlike the managed layer (immutable, hard-rejected) a drop-in is the user's
    own file, so the write stands and we explain that the drop-in wins at load time.
    """
    drop_in_sources = config_d_key_sources()
    if key not in drop_in_sources:
        return
    from hermes_cli.colors import Colors, color
    print(color(
        f"  Note: {key} is currently overridden by config.d drop-in(s) "
        f"{', '.join(drop_in_sources[key])} — the drop-in wins at load time. Edit the "
        f"drop-in (or remove it) to change the effective value.", Colors.YELLOW))
