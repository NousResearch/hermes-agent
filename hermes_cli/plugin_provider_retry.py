"""One forced plugin re-discovery per unregistered provider name, not one per call.

Every plugin-provider dispatcher (TTS, STT, image_gen, video_gen) shares one shape: idempotent
discovery, a registry lookup, and — on a miss — ``_ensure_plugins_discovered(force=True)``
followed by a second lookup. The forced pass starts with ``PluginManager.unload()``: every
plugin's teardown runs again (``ctx.spawn_task`` work is cancelled) and every ``register()``
re-runs, while the name stays unregistered because nothing about the plugin sources changed.

So a ``<category>.provider`` typo, or a plugin disabled in this profile, pays a full plugin
reload on every call — in the gateway that is once per inbound voice note (STT) and once per
spoken reply (TTS). Remember the miss together with a stamp of everything that CAN change the
answer (the plugin roots' entry lists, ``plugins.enabled`` / ``plugins.disabled``, and the
profile home) and skip the forced pass while the stamp is unchanged. A plugin installed,
removed, enabled, disabled or renamed changes the stamp; the install/update paths also reload
the plugins, after which the ordinary lookup finds the provider and this module is not reached.

Scope: the memo is in-process and per profile. A fresh process pays one forced pass per
unregistered name, which is the pre-existing behaviour and cheap (nothing is loaded yet).

Deliberately NOT covered: a provider whose plugin is present but fails to import. The stamp
only tracks which plugins exist and whether they are enabled, so such a name keeps being
reported as unregistered until the sources change or the process restarts — the same answer
the forced pass produces today.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from hermes_constants import get_hermes_home, hermes_home_key

logger = logging.getLogger("hermes_cli.plugins")

# (profile home, provider kind, provider name) -> the plugin-source stamp seen when the name
# missed even after a forced pass. A different stamp means something changed; retry again.
_misses: dict[tuple[str, str, str], str] = {}


def _plugin_roots() -> list[Path]:
    """The directories a discovery sweep reads manifests from (project plugins included only when the
    env opt-in is on, matching ``plugins_discovery.collect_directory_manifests``)."""
    from hermes_cli import plugins as _origin
    roots = [_origin.get_bundled_plugins_dir(), get_hermes_home() / "plugins"]
    if _origin._env_enabled("HERMES_ENABLE_PROJECT_PLUGINS"):
        roots.append(Path.cwd() / ".hermes" / "plugins")
    return roots


def plugin_source_stamp() -> str:
    """Digest of what decides which providers a discovery pass can register. Never raises: a root that
    cannot be listed is stamped as such (the caller then retries, which is the pre-existing behaviour)."""
    from hermes_cli import plugins_discovery

    parts = [hermes_home_key()]
    for root in _plugin_roots():
        try:
            names = sorted(child.name for child in root.iterdir() if not child.name.startswith("."))
        except OSError:
            names = ["<unreadable>"]
        parts.append(f"{root}:{'|'.join(names)}")
    enabled = plugins_discovery._get_enabled_plugins()
    disabled = plugins_discovery._get_disabled_plugins()
    # ``plugins.enabled`` missing/malformed reads as None ("nothing enabled yet"), a different
    # answer from an explicitly empty list.
    parts.append("enabled=" + ("<absent>" if enabled is None else "|".join(sorted(enabled))))
    parts.append("disabled=" + "|".join(sorted(disabled)))
    return "\n".join(parts)


def forced_refresh_needed(kind: str, name: str) -> bool:
    """True when a forced re-discovery may still register ``name``: no miss recorded for it under the
    current plugin-source stamp. Fail-open — a stamp that cannot be built retries."""
    try:
        stamp = plugin_source_stamp()
    except Exception:  # config/registry read failed: keep the old "always retry" behaviour
        logger.debug("Plugin-source stamp unavailable; retrying the forced discovery", exc_info=True)
        return True
    return _misses.get((hermes_home_key(), kind, name)) != stamp


def remember_miss(kind: str, name: str) -> None:
    """Record that ``name`` stayed unregistered after a forced pass over the current sources."""
    try:
        stamp = plugin_source_stamp()
    except Exception:
        # No stamp to compare against, so the next call retries (the pre-existing behaviour).
        logger.debug("Plugin-source stamp unavailable; not recording the %s miss for '%s'", kind, name,
                     exc_info=True)
        return
    _misses[(hermes_home_key(), kind, name)] = stamp


def lookup_with_forced_refresh(kind: str, name: str, lookup: Callable[[], Any]) -> Any:
    """``lookup()`` after at most ONE forced re-discovery per plugin-source stamp.

    ``lookup`` re-reads the registry for ``name`` and returns None when the forced pass did not
    register it — the miss is then remembered so the next call answers None without unloading the
    other plugins. Exceptions from discovery propagate: callers already decide whether a failed
    discovery is fatal, and an unrecorded miss simply retries next time.

    ``kind`` is the config section the name came from (``"tts"``, ``"stt"``, ``"image_gen"``,
    ``"video_gen"``): the same name can be a registered STT plugin and an unregistered TTS one.
    """
    if not forced_refresh_needed(kind, name):
        logger.debug(
            "Skipping the forced plugin re-discovery for %s provider '%s': already unregistered "
            "under the current plugin sources", kind, name)
        return None
    from hermes_cli.plugins import _ensure_plugins_discovered
    _ensure_plugins_discovered(force=True)
    found = lookup()
    if found is None:
        remember_miss(kind, name)
    return found


def forget_misses() -> None:
    """Drop every remembered miss (tests; a caller that knows the sources changed behind our back)."""
    _misses.clear()
