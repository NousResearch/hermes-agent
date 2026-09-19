"""Offline memory-provider preparation for an unpublished profile clone.

The host owns the source installation lock and staging rollback. Companions are
trusted plugin code: they read explicit source paths and write explicit staging
paths, never call runtime activation or perform remote provisioning.
"""
from __future__ import annotations

from contextlib import contextmanager
import inspect
import json
from pathlib import Path

CLONE_REPORT_FILE = ".clone-report.json"


class _FileOnlySecrets(dict):
    """Empty tombstones prevent get_secret's single-profile environment fallback.

    A clone has no process-injected credentials of its own. Missing source keys
    are explicitly empty even when the caller is not a multiplexed gateway.
    """
    def get(self, key, default=None):
        return super().get(key, "" if default is None else default)


@contextmanager
def clone_source_scope(source_home: Path):
    from agent.secret_scope import load_env_file, reset_secret_scope, set_secret_scope
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home_token = set_hermes_home_override(source_home)
    try:
        secret_token = set_secret_scope(_FileOnlySecrets(load_env_file(source_home / ".env")))
        try:
            yield
        finally:
            reset_secret_scope(secret_token)
    finally:
        reset_hermes_home_override(home_token)


def _provider_name(name) -> bool:
    from plugins.memory.surfaces import is_provider_name

    return is_provider_name(name)


def _invoke_prepare(callback, arguments):
    parameters = inspect.signature(callback).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return callback(**arguments)
    return callback(**{key: value for key, value in arguments.items() if key in parameters})


def prepare_memory_clone(*, source_home: Path, source_name: str, staging_home: Path,
                         destination_home: Path, destination_name: str, clone_all: bool) -> dict:
    """Prepare all installed native companions, including inactive providers.

    Legacy config/env-only providers retain historical behavior in both modes.
    Without a companion, recognizable native state is refused: the host cannot
    invent its ownership semantics. Detection is bounded to installed/configured
    names and their matching home paths (name or name.json); arbitrary retired
    or unconfigured native data is not discovered or sanitized by this check.
    """
    from hermes_cli.config import read_user_config_raw
    from hermes_cli.config_effective import load_user_config_effective
    from plugins import memory
    from plugins.memory.surfaces import load_provider_companion

    arguments = dict(source_home=source_home, source_name=source_name,
                     staging_home=staging_home, destination_home=destination_home,
                     destination_name=destination_name, clone_all=clone_all)
    report = {"needs_auth": []}
    with clone_source_scope(source_home):
        try:
            # Resolve provider eligibility before migration can discard malformed input.
            # Companions still read source files; the copied config is only the host's
            # publication candidate, not a recovered source backup.
            raw = read_user_config_raw(staging_home / "config.yaml")
            effective = load_user_config_effective(staging_home / "config.yaml", fail_closed=True)
            mem = effective.get("memory", {}) or {}
            if not isinstance(mem, dict):
                raise ValueError("Invalid memory configuration")
            active = mem.get("provider") or ""
            if active in {"built-in", "builtin", "none"}:
                active = ""
            if active and not _provider_name(active):
                raise ValueError("Invalid memory provider name")
            installed = set(memory.list_memory_provider_names())
            configured = {name for name, value in mem.items() if isinstance(value, dict) and value}
            # Native provider top-level config is also used by older installations.
            configured.update(name for name in installed if isinstance(raw.get(name), dict) and raw[name])
            names = installed | configured | ({active} if active else set())
        except Exception:
            raise ValueError("Cannot prepare memory for this clone. Check the source memory configuration and provider installation, then retry.") from None
        for name in sorted(names):
            if not _provider_name(name):
                raise ValueError("Cannot clone an invalid memory provider name. Repair the source configuration, then retry.")
            try:
                provider_dir = memory.find_provider_dir(name)
                if (name == active and provider_dir is None
                        and memory.find_provider_entry_point(name) is None):
                    raise ValueError("Selected provider package is missing")
                # The companion resolver validates entry-point origins without
                # activating them; module-only providers have no sibling surface.
                companion = load_provider_companion(name, "clone")
                callback = getattr(companion, "prepare_clone", None) if companion is not None else None
                native = any((source_home / path).exists() or (source_home / path).is_symlink()
                             for path in (name, f"{name}.json"))
                if callback is None:
                    if companion is not None or native:
                        raise ValueError("Provider cannot establish safe native-state clone preparation")
                    continue
                result = _invoke_prepare(callback, arguments)
                # Only a boolean contract is persisted. Arbitrary provider text/config
                # must never become a CLI/API warning containing tokens or URLs.
                if isinstance(result, dict) and result.get("needs_auth") is True:
                    report["needs_auth"].append(name)
            except Exception:
                raise ValueError(
                    f"Cannot safely clone memory provider '{name}'. Check its installation and clone support; "
                    "repair the source configuration or create a fresh profile and configure memory there, then retry."
                ) from None
    return report


def write_clone_report(staging_home: Path, report: dict) -> None:
    """Replace (rather than follow) any copied receipt symlink before publication."""
    path = staging_home / CLONE_REPORT_FILE
    path.unlink(missing_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream)
    path.chmod(0o600)


def clone_needs_auth(profile_home: Path) -> list[str]:
    """Safe, additive reporting without changing create_profile's Path return API."""
    try:
        report = json.loads((profile_home / CLONE_REPORT_FILE).read_text(encoding="utf-8"))
        names = report.get("needs_auth", [])
        return [name for name in names if _provider_name(name)] if isinstance(names, list) else []
    except (OSError, ValueError, AttributeError):
        return []
