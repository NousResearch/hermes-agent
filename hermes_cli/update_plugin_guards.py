"""Required, profile-scoped plugin checks for a pulled source update.

An update runs these after the checkout move, while gateways are still paused and
before dependencies or services are changed. A failing check uses the updater's
existing verified tree rollback, then resumes the old gateway fleet. The opt-in
list lives in each profile's config, not in the checkout: losing a plugin cannot
silently remove its own guard.
"""

from __future__ import annotations

import os
import hashlib
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


_PLUGIN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_CHECK_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class UpdateGuard:
    profile: str
    plugin_id: str
    script: Path
    digest: str


@contextmanager
def _profile_home(home: Path):
    token = set_hermes_home_override(home)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def prepare_update_guards() -> tuple[UpdateGuard, ...]:
    """Resolve every installed profile's required checks without importing plugin code."""
    from hermes_cli.config import FailedConfigRead, load_config_readonly
    from hermes_cli.profiles import profiles_to_serve
    from pm.plugin_declarations import native_manifest_file, read_native_manifest

    checks: list[UpdateGuard] = []
    # Updating a shared checkout can restart standalone and parked profiles too.
    for profile, home in profiles_to_serve(True, include_standalone=True, include_parked=True):
        with _profile_home(home):
            config = load_config_readonly()
        if isinstance(config, FailedConfigRead):
            raise ValueError(f"profile {profile!r} has an unreadable config.yaml")
        plugins = config.get("plugins") or {}
        if not isinstance(plugins, dict):
            raise ValueError(f"profile {profile!r} has invalid plugins configuration")
        required = plugins.get("update_guards", [])
        enabled = plugins.get("enabled", [])
        if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
            raise ValueError(f"profile {profile!r} plugins.update_guards must be a list of plugin ids")
        if not isinstance(enabled, list) or not all(isinstance(item, str) for item in enabled):
            if required:
                raise ValueError(f"profile {profile!r} plugins.enabled must be a list of plugin ids")
            enabled = []
        # An enabled plugin's declaration is enough to run a check. The profile-owned list
        # additionally makes a missing/disabled plugin a hard refusal.
        for plugin_id in dict.fromkeys([*required, *enabled]):
            if plugin_id not in required and not _PLUGIN_ID.fullmatch(plugin_id):
                continue  # other plugin kinds use namespaced IDs outside native plugins/<id>
            if not _PLUGIN_ID.fullmatch(plugin_id):
                raise ValueError(f"profile {profile!r} has an invalid guarded plugin id")
            if plugin_id in required and plugin_id not in enabled:
                raise ValueError(f"profile {profile!r} requires plugin {plugin_id!r}, but it is not enabled")
            plugin_dir = home / "plugins" / plugin_id
            if not plugin_dir.is_dir():
                if plugin_id in required:
                    raise ValueError(f"profile {profile!r} requires missing plugin {plugin_id!r}")
                continue
            manifest_file = native_manifest_file(plugin_dir)
            if manifest_file is None:
                if plugin_id in required:
                    raise ValueError(f"profile {profile!r} plugin {plugin_id!r} has no native manifest")
                continue
            manifest = read_native_manifest(manifest_file)
            if manifest.get("name") != plugin_id:
                raise ValueError(f"profile {profile!r} plugin {plugin_id!r} has a mismatched manifest name")
            relative = manifest.get("post_update_check")
            if relative is None and plugin_id not in required:
                continue
            if not isinstance(relative, str) or not relative.endswith(".py"):
                raise ValueError(f"profile {profile!r} plugin {plugin_id!r} has no Python post_update_check")
            relpath = Path(relative)
            if relpath.is_absolute() or not relpath.parts or any(part in {".", ".."} for part in relpath.parts):
                raise ValueError(f"profile {profile!r} plugin {plugin_id!r} has an unsafe post_update_check path")
            script = plugin_dir / relpath
            if script.is_symlink() or not script.is_file() or plugin_dir.resolve() not in script.resolve().parents:
                raise ValueError(f"profile {profile!r} plugin {plugin_id!r} post_update_check is missing or unsafe")
            checks.append(UpdateGuard(profile, plugin_id, script, hashlib.sha256(script.read_bytes()).hexdigest()))
    return tuple(checks)


def check_pulled_plugin_compatibility(source: Path, checks: tuple[UpdateGuard, ...]) -> str | None:
    """Return a refusal for the first failed required check, else ``None``.

    Checks run under the selected Python with an empty Hermes home, no inherited
    credentials, no stdin and a deadline. They receive the pulled source path,
    never the real profile home. A check is plugin code already trusted by its
    operator; this isolation prevents accidental live-state writes, not hostile
    code execution.
    """
    for check in checks:
        profile, plugin_id, script = check.profile, check.plugin_id, check.script
        try:
            if hashlib.sha256(script.read_bytes()).hexdigest() != check.digest:
                return f"profile {profile!r} plugin {plugin_id!r} update check changed during update"
        except OSError:
            return f"profile {profile!r} plugin {plugin_id!r} update check disappeared during update"
        with tempfile.TemporaryDirectory(prefix="hermes-update-plugin-check-") as temp:
            env = {key: value for key, value in os.environ.items()
                   if key in {"SYSTEMROOT", "WINDIR", "TMP", "TEMP", "TMPDIR", "LANG", "LC_ALL"}}
            env["HERMES_HOME"] = temp
            try:
                result = subprocess.run(
                    [sys.executable, "-I", "-B", str(script), "--hermes-root", str(source.resolve()),
                     "--profile", profile],
                    cwd=temp, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                    timeout=_CHECK_TIMEOUT_SECONDS, check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                reason = "timed out" if isinstance(exc, subprocess.TimeoutExpired) else "could not run"
                return f"profile {profile!r} plugin {plugin_id!r} update check {reason}"
            if result.returncode:
                # The child has no credentials, but keep arbitrary plugin output out of receipts.
                return f"profile {profile!r} plugin {plugin_id!r} update check exited {result.returncode}"
    return None
