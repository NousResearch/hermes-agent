"""Bundled-resource env pointers baked into generated gateway service units.

The ``hermes`` wrapper on packaged installs (Homebrew, Nix) exports ``HERMES_BUNDLED_*`` so bundled
plugins/skills/locales/TUI assets outside site-packages (e.g. ``<prefix>/share/hermes-agent/plugins``)
are found. launchd and systemd start the venv python directly, bypassing the wrapper, so these must be
written into the unit; otherwise the supervised gateway discovers zero bundled platform manifests and
logs "No adapter available for <platform>" (#85357).
"""

from __future__ import annotations

import plistlib
from pathlib import Path
from xml.sax.saxutils import escape
import os

BUNDLED_RESOURCE_ENV_VARS = (
    "HERMES_BUNDLED_PLUGINS",
    "HERMES_BUNDLED_SKILLS",
    "HERMES_BUNDLED_LOCALES",
    "HERMES_OPTIONAL_SKILLS",
    "HERMES_TUI_DIR",
)


def bundled_resource_env_pairs(prior: dict[str, str] | None = None) -> list[tuple[str, str]]:
    """``(name, value)`` per pointer to bake: the live wrapper env wins, else the on-disk unit's value.

    ``generate_*`` runs on every ordinary start, not just install, so without the on-disk fallback a
    later start from a context without the wrapper env (venv python run directly, a GUI launch, a
    supervised restart) would rewrite the unit with the pointers stripped. Empty on a standard pip/uv
    install with no prior unit, so those units stay byte-identical."""
    prior = prior or {}
    pairs: list[tuple[str, str]] = []
    for name in BUNDLED_RESOURCE_ENV_VARS:
        value = os.environ.get(name, "").strip() or prior.get(name, "").strip()
        if value:
            pairs.append((name, value))
    return pairs


def systemd_env_line(name: str, value: str) -> str:
    """One ``Environment="NAME=value"`` line: ``\\`` and ``"`` escaped for systemd's quoting, ``%``
    doubled so specifier expansion leaves the value alone."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    return f'Environment="{name}={escaped}"\n'


def systemd_bundled_env_block(unit_path: Path) -> str:
    """Extra ``Environment=`` lines for the systemd unit at ``unit_path`` (``""`` when none apply)."""
    from hermes_cli.gateway import _unit_environment_value

    prior = {name: (_unit_environment_value(unit_path, name) or "") for name in BUNDLED_RESOURCE_ENV_VARS}
    return "".join(systemd_env_line(name, value) for name, value in bundled_resource_env_pairs(prior))


def _prior_launchd_env(plist_path: Path) -> dict[str, str]:
    try:
        data = plistlib.loads(plist_path.read_bytes())
    except (OSError, plistlib.InvalidFileException, ValueError):  # absent or hand-mangled plist
        return {}
    env = data.get("EnvironmentVariables") if isinstance(data, dict) else None
    if not isinstance(env, dict):
        return {}
    return {name: env[name] for name in BUNDLED_RESOURCE_ENV_VARS if isinstance(env.get(name), str)}


def launchd_bundled_env_xml(plist_path: Path) -> str:
    """Extra ``<key>/<string>`` pairs for the plist's ``EnvironmentVariables`` dict (``""`` when none)."""
    return "".join(
        f"\n        <key>{name}</key>\n        <string>{escape(value)}</string>"
        for name, value in bundled_resource_env_pairs(_prior_launchd_env(plist_path))
    )
