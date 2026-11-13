"""Carry the ACTIVE memory provider's own config into a ``--clone`` (#120115).

``--clone`` copies ``config.yaml`` — and with it ``memory.provider: hindsight`` — but the
provider keeps its settings outside config.yaml, so the clone booted with the provider
selected and silently unavailable. Providers store per-home config by convention (the same
convention ``hermes_agent.hermes_cli.web_routers.memory_providers`` reads): a ``<home>/<provider>/``
directory (hindsight) or a flat ``<home>/<provider>.json`` (mem0, honcho, supermemory). Copying
by convention keeps this free of plugin imports: the provider may live in the catalog, not in
tree, so a hook the plugin must implement could not fix the reported case.
"""

import contextlib
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Callable, Optional

from hermes_agent.agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

# A provider name is a bare directory/file stem; anything else (path separators, ``..``, spaces)
# would let a hand-edited config.yaml aim the copy outside the source profile.
_PROVIDER_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def active_memory_provider(config: Optional[dict]) -> Optional[str]:
    """The external ``memory.provider`` named in a parsed config.yaml, or None for the built-in
    store or an unsafe name."""
    from hermes_agent.agent.memory_provider import is_core_memory_provider

    memory = (config or {}).get("memory")
    name = memory.get("provider") if isinstance(memory, dict) else None
    if not isinstance(name, str) or is_core_memory_provider(name):
        return None
    name = name.strip()
    if name in {".", ".."} or not _PROVIDER_NAME_RE.match(name):
        return None
    return name


def clone_memory_provider_config(source_dir: Path, profile_dir: Path, provider: Optional[str]) -> bool:
    """Copy ``<provider>/`` and/or ``<provider>.json`` from *source_dir* into *profile_dir* when
    present. Files land owner-only like ``.env``: they can hold an API key. Returns True when
    anything was copied."""
    if not provider:
        return False
    copied = False
    src_dir = source_dir / provider
    if src_dir.is_dir():
        shutil.copytree(src_dir, profile_dir / provider, dirs_exist_ok=True)
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


def _load_config_dict(profile_dir: Path) -> Optional[dict]:
    """A profile's config.yaml mapping read through the config backend; None when absent or unreadable."""
    from hermes_agent.hermes_yaml import YAMLError
    from hermes_agent.hermes_cli.config_backend import read_config_doc
    try:
        data = read_config_doc(profile_dir / "config.yaml") or {}
    # FileNotFoundError included (no config). ValueError: the YAML constructors raise it bare for
    # values like ``2024-02-30`` / ``!!int abc``; this read is a best-effort notice and never fails a clone.
    except (YAMLError, OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _names_external_provider(profile_dir: Path) -> bool:
    """Cheap pre-check before binding a profile's secrets: does its config.yaml, or the managed
    overlay, name a non-core ``memory.provider`` (a ``${VAR}`` selector counts)?"""
    from hermes_agent.agent.memory_provider import is_core_memory_provider
    from hermes_agent.hermes_cli import managed_scope

    from hermes_agent.hermes_cli.config import cfg_get

    for config in (_load_config_dict(profile_dir), managed_scope.load_managed_config()):
        name = cfg_get(config, "memory", "provider")
        if isinstance(name, str) and name.strip() and not is_core_memory_provider(name):
            return True
    return False


def notify_memory_provider(profile_dir: Path, call: Callable[[MemoryProvider], None]) -> None:
    """Run ``call`` on the memory provider *profile_dir*'s config selects, with that profile's home AND
    its own secrets bound (a profile lifecycle hook): ``get_secret`` reads the target profile's
    ``.env`` and never falls back to the caller's environment for a key it lacks. No
    provider selected or installed: nothing runs. Best-effort: a failing
    provider, or an unreadable config, is logged and never fails the create/rename that called it
    (the directory is already published). A plugin host the hook started for that profile is
    stopped afterwards (``plugins.isolation: host``), so a long-lived dashboard keeps none behind."""
    from hermes_agent.hermes_constants import reset_hermes_home_override, set_hermes_home_override

    name = None
    home_token = set_hermes_home_override(str(profile_dir))
    secret_token = None
    try:
        if not _names_external_provider(profile_dir):
            return  # the common case: no provider, so no secret scope, effective config or load
        from hermes_agent.agent.secret_scope import build_profile_secret_scope, set_secret_scope
        from hermes_agent.hermes_cli.plugin_isolation import transient_plugin_host
        from hermes_agent.plugins.memory import load_memory_provider, provider_registration_scope
        # Secrets first: the effective config (managed scope, ``${VAR}`` refs) expands through them.
        # Its configured secret sources too (1Password etc.); a brand-new profile was never hydrated.
        from hermes_agent.hermes_cli.env_loader import hydrate_profile_secret_sources
        hydrate_profile_secret_sources(profile_dir)
        secret_token = set_secret_scope(build_profile_secret_scope(profile_dir), profile_home=str(profile_dir))
        from hermes_agent.hermes_cli.config import load_config_readonly
        name = active_memory_provider(load_config_readonly())
        if not name:
            return
        with provider_registration_scope(name) as release, transient_plugin_host(before_stop=release):
            provider = load_memory_provider(name, register_skills=False)
            if provider is not None:
                call(provider)
    except (Exception, SystemExit):  # ConfigBackendUnavailable is a SystemExit; a plugin may sys.exit
        logger.warning("memory provider %r: profile hook failed for %s", name, profile_dir, exc_info=True)
    finally:
        if secret_token is not None:
            from hermes_agent.agent.secret_scope import reset_secret_scope
            reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


def cloned_memory_provider(profile_dir: Path) -> Optional[str]:
    """Name of the external provider whose config *profile_dir* now carries, for the CLI notice."""
    provider = active_memory_provider(_load_config_dict(profile_dir))
    if provider and ((profile_dir / provider).is_dir() or (profile_dir / f"{provider}.json").is_file()):
        return provider
    return None
