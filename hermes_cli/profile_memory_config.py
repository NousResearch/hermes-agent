"""Carry the ACTIVE memory provider's own config into a ``--clone`` (#120115).

``--clone`` copies ``config.yaml`` — and with it ``memory.provider: hindsight`` — but the
provider keeps its settings outside config.yaml, so the clone booted with the provider
selected and silently unavailable. Providers store per-home config by convention (the same
convention ``hermes_cli.web_routers.memory_providers`` reads): a ``<home>/<provider>/``
directory (hindsight) or a flat ``<home>/<provider>.json`` (mem0, honcho, supermemory). Copying
by convention keeps this free of plugin imports: the provider may live in the catalog, not in
tree, so a hook the plugin must implement could not fix the reported case.
"""

import contextlib
import os
import re
import shutil
from pathlib import Path
from typing import Optional

# A provider name is a bare directory/file stem; anything else (path separators, ``..``, spaces)
# would let a hand-edited config.yaml aim the copy outside the source profile.
_PROVIDER_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

# Providers whose ``<provider>/`` directory doubles as the live data store: these subpaths are
# the SOURCE profile's episodic history, not config, and must not travel with a clone — same
# class as the state.db/sessions exclusion in profiles._CLONE_ALL_HISTORY_EXCLUDE_ROOT (#133308).
# Layout per mnemosyne_hermes 0.5.0: ``mnemosyne/config.yaml`` + ``mnemosyne/data/mnemosyne.db``.
_CLONE_DATA_EXCLUDES: dict[str, frozenset[str]] = {
    "mnemosyne": frozenset({"data"}),
}

# Optional per-provider manifest naming further subpaths to exclude: one relative subpath per
# line under ``<provider>/``, ``#`` comments allowed. A file convention — not a Python attribute
# the provider must export — because the provider may live in the catalog, out of tree, and the
# copy must stay free of plugin imports (#120115). The manifest itself is copied, so clones of
# clones stay protected. Unlisted providers keep copying whole (hindsight, ``*.json``).
_CLONE_EXCLUDE_MANIFEST = "CLONE-EXCLUDE"


def active_memory_provider(config: Optional[dict]) -> Optional[str]:
    """The external ``memory.provider`` named in a parsed config.yaml, or None for the built-in
    store or an unsafe name."""
    from agent.memory_provider import is_core_memory_provider

    memory = (config or {}).get("memory")
    name = memory.get("provider") if isinstance(memory, dict) else None
    if not isinstance(name, str) or is_core_memory_provider(name):
        return None
    name = name.strip()
    if name in {".", ".."} or not _PROVIDER_NAME_RE.match(name):
        return None
    return name


def provider_live_data_excludes(provider: str, provider_dir: Path) -> set:
    """Subpaths of ``<provider>/`` that a clone must leave behind: the built-in table for
    providers whose directory is the live data store, plus the provider's own CLONE-EXCLUDE
    manifest when present. Manifest lines must be relative, in-bounds subpaths; anything else
    (absolute paths, ``..``, empty) is ignored so a malformed line can only keep data flowing,
    never aim the copy outside the provider directory."""
    excludes = set(_CLONE_DATA_EXCLUDES.get(provider, ()))
    try:
        lines = (
            (provider_dir / _CLONE_EXCLUDE_MANIFEST)
            .read_text(encoding="utf-8-sig")
            .splitlines()
        )
    except OSError:
        return excludes
    for line in lines:
        stripped = line.strip()
        parts = [p for p in stripped.split("/") if p not in ("", ".")]
        if stripped.startswith("#") or stripped.startswith(("/", "\\")) or not parts:
            continue
        if any(p == ".." or not _PROVIDER_NAME_RE.match(p) for p in parts):
            continue
        excludes.add("/".join(parts))
    return excludes


def clone_memory_provider_config(source_dir: Path, profile_dir: Path, provider: Optional[str]) -> bool:
    """Copy ``<provider>/`` and/or ``<provider>.json`` from *source_dir* into *profile_dir* when
    present. Files land owner-only like ``.env``: they can hold an API key. Live-data subpaths
    (:func:`provider_live_data_excludes`) stay behind as empty dirs — the provider's DB is the
    source profile's episodic history, not config (#133308). Returns True when anything was
    copied."""
    if not provider:
        return False
    copied = False
    src_dir = source_dir / provider
    if src_dir.is_dir():
        excludes = provider_live_data_excludes(provider, src_dir)
        src_resolved = src_dir.resolve()

        def _drop_live_data(directory: str, names: list) -> set:
            try:
                below = Path(directory).resolve().relative_to(src_resolved).parts
            except (OSError, ValueError):
                return set()
            drop = set()
            for subpath in excludes:
                segs = subpath.split("/")
                if (
                    len(segs) > len(below)
                    and segs[: len(below)] == list(below)
                    and segs[len(below)] in names
                ):
                    drop.add(segs[len(below)])
            return drop

        shutil.copytree(
            src_dir,
            profile_dir / provider,
            dirs_exist_ok=True,
            ignore=_drop_live_data if excludes else None,
        )
        for subpath in sorted(excludes):
            (profile_dir / provider / subpath).mkdir(parents=True, exist_ok=True)
        for root, _dirs, files in os.walk(profile_dir / provider):
            for filename in files:
                with contextlib.suppress(OSError):
                    os.chmod(os.path.join(root, filename), 0o600)
        copied = True
    src_file = source_dir / f"{provider}.json"
    if src_file.is_file():
        dst = profile_dir / f"{provider}.json"
        shutil.copy2(src_file, dst)
        with contextlib.suppress(OSError):
            os.chmod(str(dst), 0o600)
        copied = True
    return copied


def cloned_memory_provider(profile_dir: Path) -> Optional[str]:
    """Name of the external provider whose config *profile_dir* now carries, for the CLI notice."""
    from hermes_cli.profiles import _load_yaml_dict

    provider = active_memory_provider(_load_yaml_dict(profile_dir / "config.yaml"))
    if provider and ((profile_dir / provider).is_dir() or (profile_dir / f"{provider}.json").is_file()):
        return provider
    return None


def strip_cloned_provider_live_data(profile_dir: Path) -> list:
    """Post-copy sweep for ``--clone-all``'s whole-tree copy: remove the active provider's
    live-data subpaths (see :func:`provider_live_data_excludes`), leaving empty dirs. Returns
    the ``<provider>/<subpath>`` entries that were dropped."""
    from hermes_cli.profiles import _load_yaml_dict

    provider = active_memory_provider(_load_yaml_dict(profile_dir / "config.yaml"))
    if not provider or not (profile_dir / provider).is_dir():
        return []
    dropped = []
    for subpath in sorted(
        provider_live_data_excludes(provider, profile_dir / provider)
    ):
        target = profile_dir / provider / subpath
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
            target.mkdir(parents=True, exist_ok=True)
            dropped.append(f"{provider}/{subpath}")
    return dropped
