"""Required, profile-scoped plugin checks for a pulled source update.

An update runs these after the checkout move, while gateways are still paused and
before dependencies or services are changed. A failing check uses the updater's
existing verified tree rollback, then resumes the old gateway fleet. The opt-in
list lives in each profile's config, not in the checkout: losing a plugin cannot
silently remove its own guard.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


_PLUGIN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_CHECK_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class UpdateGuard:
    profile: str
    plugin_id: str
    script: Path
    digest: str
    required: bool = False


@contextmanager
def _profile_home(home: Path):
    token = set_hermes_home_override(home)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _profile_plugin_lists(profile: str, home: Path) -> tuple[list[str], list[str], list[str]]:
    """Read the profile-owned requirement and effective selection."""
    from hermes_cli.config import FailedConfigRead, load_config_readonly
    with _profile_home(home):
        config = load_config_readonly()
    if isinstance(config, FailedConfigRead):
        raise ValueError(f"profile {profile!r} has an unreadable config.yaml")
    plugins = config.get("plugins") or {}
    if not isinstance(plugins, dict):
        raise ValueError(f"profile {profile!r} has invalid plugins configuration")
    required, enabled, disabled = (plugins.get(key, []) for key in ("update_guards", "enabled", "disabled"))
    if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
        raise ValueError(f"profile {profile!r} plugins.update_guards must be a list of plugin ids")
    for key, value in (("enabled", enabled), ("disabled", disabled)):
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            if required:
                raise ValueError(f"profile {profile!r} plugins.{key} must be a list of plugin ids")
    valid_enabled = isinstance(enabled, list) and all(isinstance(item, str) for item in enabled)
    valid_disabled = isinstance(disabled, list) and all(isinstance(item, str) for item in disabled)
    return required, enabled if valid_enabled else [], disabled if valid_disabled else []


def _declared_guard(profile: str, home: Path, plugin_id: str, *, required: bool,
                    enabled: list[str], disabled: list[str]) -> UpdateGuard | None:
    from pm.plugin_declarations import native_manifest_file, read_native_manifest

    if not _PLUGIN_ID.fullmatch(plugin_id):
        if required:
            raise ValueError(f"profile {profile!r} has an invalid guarded plugin id")
        return None  # other plugin kinds use namespaced IDs outside native plugins/<id>
    if required and plugin_id not in enabled:
        raise ValueError(f"profile {profile!r} requires plugin {plugin_id!r}, but it is not enabled")
    if plugin_id in disabled:
        if required:
            raise ValueError(f"profile {profile!r} requires plugin {plugin_id!r}, but it is disabled")
        return None
    plugin_dir = home / "plugins" / plugin_id
    if not plugin_dir.is_dir():
        if required:
            raise ValueError(f"profile {profile!r} requires missing plugin {plugin_id!r}")
        return None
    manifest_file = native_manifest_file(plugin_dir)
    if manifest_file is None:
        if required:
            raise ValueError(f"profile {profile!r} plugin {plugin_id!r} has no native manifest")
        return None
    manifest = read_native_manifest(manifest_file)
    if manifest.get("name") != plugin_id:
        raise ValueError(f"profile {profile!r} plugin {plugin_id!r} has a mismatched manifest name")
    relative = manifest.get("post_update_check")
    if relative is None and not required:
        return None
    if not isinstance(relative, str) or not relative.endswith(".py"):
        raise ValueError(f"profile {profile!r} plugin {plugin_id!r} has no Python post_update_check")
    relpath = Path(relative)
    if relpath.is_absolute() or not relpath.parts or any(part in {".", ".."} for part in relpath.parts):
        raise ValueError(f"profile {profile!r} plugin {plugin_id!r} has an unsafe post_update_check path")
    script = plugin_dir / relpath
    if script.is_symlink() or not script.is_file() or plugin_dir.resolve() not in script.resolve().parents:
        raise ValueError(f"profile {profile!r} plugin {plugin_id!r} post_update_check is missing or unsafe")
    return UpdateGuard(profile, plugin_id, script, hashlib.sha256(script.read_bytes()).hexdigest(),
                       required)


def prepare_update_guards() -> tuple[UpdateGuard, ...]:
    """Resolve every installed profile's required checks without importing plugin code."""
    from hermes_cli.profiles import profiles_to_serve

    checks: list[UpdateGuard] = []
    # Updating a shared checkout can restart standalone and parked profiles too.
    for profile, home in profiles_to_serve(True, include_standalone=True, include_parked=True):
        required, enabled, disabled = _profile_plugin_lists(profile, home)
        # An enabled plugin's declaration is enough to run a check. The profile-owned list
        # additionally makes a missing/disabled plugin a hard refusal.
        for plugin_id in dict.fromkeys([*required, *enabled]):
            check = _declared_guard(profile, home, plugin_id, required=plugin_id in required,
                                    enabled=enabled, disabled=disabled)
            if check is not None:
                checks.append(check)
    return tuple(checks)


def guard_records(checks: tuple[UpdateGuard, ...]) -> list[dict[str, Any]]:
    """Pass the frozen pre-update declarations to the fresh completion interpreter."""
    return [dict(profile=check.profile, plugin_id=check.plugin_id,
                 script=str(check.script), digest=check.digest, required=check.required)
            for check in checks]


def guards_from_records(records: list[dict[str, Any]]) -> tuple[UpdateGuard, ...]:
    """Restore only the simple immutable fields that the old updater captured."""
    return tuple(UpdateGuard(row["profile"], row["plugin_id"], Path(row["script"]), row["digest"],
                             row.get("required", False))
                 for row in records)


def _pending_guard_record(source: Path) -> tuple[dict, tuple[UpdateGuard, ...]] | None:
    """Read the previous admission obligation without trusting it as executable code."""
    path = pending_path(source)
    try:
        record = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise ValueError("the earlier plugin update admission marker is unreadable") from exc
    if (not isinstance(record, dict) or record.get("source") != str(source.resolve())
            or "pre_sha" not in record or "runtime_facts_digest" not in record
            or (record["pre_sha"] is not None and not isinstance(record["pre_sha"], str))
            or (record["runtime_facts_digest"] is not None
                and not isinstance(record["runtime_facts_digest"], str))
            or type(record.get("expanded", False)) is not bool
            or not isinstance(record.get("guards"), list) or not record["guards"]):
        raise ValueError("the earlier plugin update admission marker is invalid")
    rows = record["guards"]
    if not all(isinstance(row, dict)
               and set(row) in ({"profile", "plugin_id", "script", "digest"},
                                {"profile", "plugin_id", "script", "digest", "required"})
               and all(isinstance(row[key], str)
                       for key in ("profile", "plugin_id", "script", "digest"))
               and type(row.get("required", False)) is bool for row in rows):
        raise ValueError("the earlier plugin update guard records are invalid")
    previous = guards_from_records(rows)
    if (len(set(previous)) != len(previous)
            or any(not check.script.is_absolute() or not _PLUGIN_ID.fullmatch(check.plugin_id)
                   or not re.fullmatch(r"[0-9a-f]{64}", check.digest) for check in previous)):
        raise ValueError("the earlier plugin update guard records are invalid")
    return record, previous


def require_prior_guards(source: Path, checks: tuple[UpdateGuard, ...]) -> None:
    """A retry may add guards, but cannot drop or replace an unadmitted guard.

    Only freshly discovered profile-owned declarations can authorize execution.
    A marker is never used to run an arbitrary saved script path.
    """
    pending = _pending_guard_record(source)
    if pending is not None and not set(pending[1]).issubset(checks):
        raise ValueError("an earlier plugin update guard is missing or changed")


def pending_path(source: Path) -> Path:
    """One host-owned stop marker per source checkout, including multiplex profiles."""
    from hermes_constants import get_default_hermes_root

    key = hashlib.sha256(os.path.normcase(str(source.resolve())).encode()).hexdigest()[:12]
    return get_default_hermes_root() / f".hermes-update-plugin-guard.{key}.json"


def _runtime_facts_digest(source: Path) -> str | None:
    """Detect a dependency selection changed after the original source baseline."""
    from pm.environments import runtime_facts_path

    try:
        return hashlib.sha256(runtime_facts_path(source).read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def arm_pending(source: Path, checks: tuple[UpdateGuard, ...], *, pre_sha: str | None = None) -> None:
    """Keep a paused fleet stopped until the selected runtime passes every check."""
    pending = _pending_guard_record(source)
    if pending is not None and not set(pending[1]).issubset(checks):
        raise ValueError("an earlier plugin update guard is missing or changed")
    if not checks:
        return
    from hermes_cli._early_recovery import write_durable_text

    path = pending_path(source)
    if pending is not None:
        record, previous = pending
        combined = tuple(dict.fromkeys([*previous, *checks]))
        if combined != previous:
            # Keep the original admitted source/dependency baseline. Every new
            # guard becomes owed before this retry may move the checkout. The
            # old baseline cannot discharge an obligation added afterward.
            write_durable_text(path, json.dumps({**record, "guards": guard_records(combined),
                                                 "expanded": True}))
        return
    write_durable_text(path, json.dumps({"source": str(source.resolve()),
                                         "guards": guard_records(checks), "pre_sha": pre_sha,
                                         "runtime_facts_digest": _runtime_facts_digest(source)}))


def clear_pending(source: Path) -> None:
    pending_path(source).unlink(missing_ok=True)


def clear_after_selected_runtime(source: Path, checks: tuple[UpdateGuard, ...]) -> None:
    """Discharge only the obligations this selected runtime actually checked."""
    pending = _pending_guard_record(source)
    if pending is None:
        raise ValueError("required plugin update admission marker disappeared")
    if not set(pending[1]).issubset(checks):
        raise ValueError("an earlier plugin update guard is missing or changed")
    clear_pending(source)


def clear_after_verified_rollback(source: Path, pre_sha: str | None) -> None:
    """Only a return to the marker's original admitted revision discharges it."""
    if not pre_sha:
        return
    try:
        record = json.loads(pending_path(source).read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, ValueError):
        return  # an unreadable or earlier marker keeps gateways stopped
    if (isinstance(record, dict) and record.get("pre_sha") == pre_sha
            and record.get("expanded") is not True
            and "runtime_facts_digest" in record
            and record["runtime_facts_digest"] == _runtime_facts_digest(source)):
        try:
            require_prior_guards(source, prepare_update_guards())
        except (OSError, ValueError):
            return  # profile requirement drift cannot be discharged by old source bytes
        clear_pending(source)


def pending_reason(source: Path) -> str | None:
    """A missing marker admits startup; an unreadable marker holds it closed."""
    try:
        pending_path(source).lstat()
    except FileNotFoundError:
        return None
    except OSError:
        return "required plugin update admission state cannot be read"
    return "a required plugin has not passed the selected runtime check"


def prepare_or_stop(windows_resume):
    """Freeze every profile's requirement before any checkout move."""
    from hermes_cli.update_cmd import _m
    from hermes_cli.update_cmd_common import _record_stop

    try:
        checks = prepare_update_guards()
        require_prior_guards(Path(_m().PROJECT_ROOT), checks)
        return checks
    except (OSError, ValueError) as exc:
        print(f"✗ Required plugin update check cannot be prepared: {exc}")
        print("  No update was applied; your install is unchanged.")
        _m()._resume_windows_gateways_after_update(windows_resume)
        _record_stop("plugin_update_guard_invalid")
        raise SystemExit(1) from exc


def arm_noop_or_stop(source: Path, checks: tuple[UpdateGuard, ...], windows_resume,
                     *, on_failure=None, pre_sha: str | None = None) -> None:
    """Refuse before a tree move or completion if the durable stop cannot be armed."""
    from hermes_cli.update_cmd import _m
    from hermes_cli.update_cmd_common import _record_stop

    try:
        arm_pending(source, checks, pre_sha=pre_sha)
    except (OSError, ValueError) as exc:
        print(f"✗ Could not record the required plugin runtime check: {exc}")
        if on_failure is not None:
            on_failure()
        _m()._resume_windows_gateways_after_update(windows_resume)
        _record_stop("plugin_update_guard_record_failed")
        raise SystemExit(1) from exc


def check_pulled_plugin_compatibility(
        source: Path, checks: tuple[UpdateGuard, ...], *, python_executable: Path | None = None,
        phase: str = "source") -> str | None:
    """Return a refusal for the first failed required check, else ``None``.

    Checks run under the pre-update or selected Python with an empty Hermes home, no inherited
    credentials, no stdin and a deadline. They receive the pulled source path,
    never the real profile home. A check is plugin code already trusted by its
    operator; this isolation prevents accidental live-state writes, not hostile
    code execution.
    """
    if phase not in {"source", "runtime"}:
        raise ValueError("invalid plugin update check phase")
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
                    [str(python_executable or sys.executable), "-I", "-B", str(script),
                     "--hermes-root", str(source.resolve()),
                     "--profile", profile, "--phase", phase],
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
