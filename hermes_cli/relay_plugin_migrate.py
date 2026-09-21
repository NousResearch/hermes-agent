"""Migrate legacy ``HERMES_NEMO_RELAY_ATIF_*`` / ``ATOF_*`` exporter vars into a Relay ``plugins.toml``.

The Relay cutover (Aug 2026) stopped honouring the legacy exporter variables: a profile that still
carries them and no ``HERMES_NEMO_RELAY_PLUGINS_TOML`` logs one warning and initialises NO exporters,
so users who followed the earlier docs lost every trace silently. This module turns those variables
into ``<profile home>/relay-plugins.toml`` (built from the ``nemo_relay.observability`` dataclasses so
the file is exactly what Relay validates), points ``HERMES_NEMO_RELAY_PLUGINS_TOML`` at it, and
comments the legacy lines out. It runs from ``hermes update`` for every profile home and from
``hermes relay migrate`` for the active one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from hermes_cli.relay_plugin_cutover import (
    LEGACY_RELAY_EXPORT_ENV_VARS, RELAY_PLUGINS_CONFIG_ENV, configured_legacy_relay_env_vars)

logger = logging.getLogger(__name__)

RELAY_PLUGINS_TOML_NAME = "relay-plugins.toml"

_TRUE = {"1", "true", "yes", "on"}


@dataclass
class RelayMigrationResult:
    home: Path
    toml_path: Optional[Path] = None
    migrated_vars: tuple[str, ...] = ()
    skipped_reason: Optional[str] = None
    validation_error: Optional[str] = None
    diagnostics: list = field(default_factory=list)

    @property
    def migrated(self) -> bool:
        return self.toml_path is not None and self.skipped_reason is None


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in _TRUE


def _s(env: Mapping[str, Any], name: str, default: str = "") -> str:
    return str(env.get(name) or "").strip() or default


def relay_plugin_payload_from_legacy_env(env: Mapping[str, Any]) -> dict[str, Any]:
    """The plugins.toml document (as a dict) equivalent to the legacy exporter variables. Defaults
    mirror the removed ``plugins/observability/nemo_relay`` plugin so a migrated user keeps the
    same files in the same places. Built through ``nemo_relay.observability`` so the sink
    discriminator (``type = "file"``) and every field name are the ones Relay validates."""
    from nemo_relay import observability as obs

    atif = atof = None
    if _truthy(env.get("HERMES_NEMO_RELAY_ATIF_ENABLED")):
        kwargs: dict[str, Any] = {
            "enabled": True,
            "agent_name": _s(env, "HERMES_NEMO_RELAY_ATIF_AGENT_NAME", "Hermes Agent"),
            "model_name": _s(env, "HERMES_NEMO_RELAY_ATIF_MODEL_NAME", "unknown"),
            "filename_template": _s(env, "HERMES_NEMO_RELAY_ATIF_FILENAME_TEMPLATE", "hermes-atif-{session_id}.json"),
        }
        if _s(env, "HERMES_NEMO_RELAY_ATIF_OUTPUT_DIRECTORY"):
            kwargs["output_directory"] = _s(env, "HERMES_NEMO_RELAY_ATIF_OUTPUT_DIRECTORY")
        if _s(env, "HERMES_NEMO_RELAY_ATIF_AGENT_VERSION"):
            kwargs["agent_version"] = _s(env, "HERMES_NEMO_RELAY_ATIF_AGENT_VERSION")
        atif = obs.AtifConfig(**kwargs)
    if _truthy(env.get("HERMES_NEMO_RELAY_ATOF_ENABLED")):
        mode = _s(env, "HERMES_NEMO_RELAY_ATOF_MODE", "append").lower()
        sink = obs.AtofFileSinkConfig(
            output_directory=_s(env, "HERMES_NEMO_RELAY_ATOF_OUTPUT_DIRECTORY") or None,
            filename=_s(env, "HERMES_NEMO_RELAY_ATOF_FILENAME", "hermes-atof.jsonl"),
            mode="overwrite" if mode == "overwrite" else "append",
        )
        atof = obs.AtofConfig(enabled=True, sinks=[sink])
    spec = obs.ComponentSpec(config=obs.ObservabilityConfig(atif=atif, atof=atof))
    return {"version": 1, "components": [spec.to_dict()]}


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _emit_table(lines: list[str], path: str, table: Mapping[str, Any], *, array_item: bool = False) -> None:
    """Minimal TOML emitter for the nested-dict/list-of-dict shape ``to_dict()`` produces."""
    scalars = {k: v for k, v in table.items() if not isinstance(v, (dict, list))}
    nested = {k: v for k, v in table.items() if isinstance(v, (dict, list))}
    if path:
        lines.append(f"[[{path}]]" if array_item else f"[{path}]")
    for key, value in scalars.items():
        if value is not None:
            lines.append(f"{key} = {_toml_scalar(value)}")
    for key, value in nested.items():
        child = f"{path}.{key}" if path else key
        if isinstance(value, dict):
            lines.append("")
            _emit_table(lines, child, value)
        else:
            for item in value:
                if isinstance(item, dict):
                    lines.append("")
                    _emit_table(lines, child, item, array_item=True)
                else:
                    raise ValueError(f"unsupported TOML array item at {child}: {item!r}")


def dumps_toml(document: Mapping[str, Any]) -> str:
    """Serialize via ``tomli_w`` when installed, else the minimal emitter above (nested tables + arrays
    of tables only — exactly the ``ComponentSpec.to_dict()`` shape)."""
    try:
        import tomli_w  # type: ignore
        return tomli_w.dumps(dict(document))
    except ImportError:
        lines: list[str] = []
        _emit_table(lines, "", document)
        return "\n".join(lines).strip() + "\n"


def validate_relay_plugin_payload(payload: Mapping[str, Any]) -> list:
    """Activate the payload once through Relay's own validator and clear it; returns the diagnostics
    (empty = clean). Raises when Relay rejects the document outright."""
    import asyncio
    from nemo_relay import plugin

    async def _probe():
        try:
            report = await plugin.initialize(dict(payload))
        finally:
            await plugin.clear_async()
        return list((report or {}).get("diagnostics") or []) if isinstance(report, dict) else []

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_probe())
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(_probe())).result()


def _comment_out_legacy_lines(lines: list[str], names: set[str]) -> list[str]:
    from hermes_cli.config import _env_line_defines_key
    out = []
    for line in lines:
        if any(_env_line_defines_key(line, name) for name in names):
            out.append(f"# migrated to {RELAY_PLUGINS_TOML_NAME}: {line.rstrip()}\n")
        else:
            out.append(line)
    return out


def migrate_profile_relay_env(home: Path, *, validate: bool = True) -> RelayMigrationResult:
    """Migrate ONE profile home's ``.env``. Never raises for a no-op; a Relay import/validation failure
    leaves ``.env`` untouched and is reported in ``validation_error``."""
    from hermes_cli.config import (
        _env_line_defines_key, _quote_env_value, _read_env_lines, _write_env_lines,
        env_store_lock_for_path,
    )
    result = RelayMigrationResult(home=home)
    env_path = home / ".env"
    if not env_path.is_file():
        result.skipped_reason = "no .env"
        return result
    with env_store_lock_for_path(env_path):
        return _migrate_profile_relay_env_locked(
            home, env_path, result, validate=validate,
            _env_line_defines_key=_env_line_defines_key,
            _quote_env_value=_quote_env_value,
            _read_env_lines=_read_env_lines,
            _write_env_lines=_write_env_lines,
        )


def _migrate_profile_relay_env_locked(
        home, env_path, result, *, validate, _env_line_defines_key, _quote_env_value,
        _read_env_lines, _write_env_lines):
    lines = _read_env_lines(env_path)
    env: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.removeprefix("export ").partition("=")
        env[key.strip()] = value.strip().strip("'\"")
    legacy = configured_legacy_relay_env_vars(env)
    if not legacy:
        result.skipped_reason = "no legacy exporter variables"
        return result
    if env.get(RELAY_PLUGINS_CONFIG_ENV, "").strip():
        result.skipped_reason = f"{RELAY_PLUGINS_CONFIG_ENV} already set"
        return result
    if not (_truthy(env.get("HERMES_NEMO_RELAY_ATIF_ENABLED")) or _truthy(env.get("HERMES_NEMO_RELAY_ATOF_ENABLED"))):
        result.skipped_reason = "no exporter enabled by the legacy variables"
        return result
    try:
        payload = relay_plugin_payload_from_legacy_env(env)
        if validate:
            result.diagnostics = validate_relay_plugin_payload(payload)
    except Exception as exc:  # nemo_relay missing or rejecting the payload: leave .env alone
        result.validation_error = f"{type(exc).__name__}: {exc}"
        return result
    toml_path = home / RELAY_PLUGINS_TOML_NAME
    header = (
        "# NeMo Relay plugin configuration for Hermes (selected via "
        f"{RELAY_PLUGINS_CONFIG_ENV} in .env).\n"
        "# Generated by `hermes update` from the legacy HERMES_NEMO_RELAY_ATIF_*/ATOF_* variables,\n"
        "# which Relay no longer reads. Edit this file to change exporters.\n"
    )
    toml_path.write_text(header + dumps_toml(payload), encoding="utf-8")
    new_lines = _comment_out_legacy_lines(lines, set(LEGACY_RELAY_EXPORT_ENV_VARS))
    if not any(_env_line_defines_key(line, RELAY_PLUGINS_CONFIG_ENV) for line in new_lines):
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"
        new_lines.append(f"{RELAY_PLUGINS_CONFIG_ENV}={_quote_env_value(str(toml_path))}\n")
    _write_env_lines(env_path, new_lines, preserve_mode=True)
    result.toml_path = toml_path
    result.migrated_vars = legacy
    return result


def migrate_all_profile_relay_envs(*, validate: bool = True) -> list[RelayMigrationResult]:
    """Default home + every live named profile (multiplex: each profile keeps its own TOML)."""
    from hermes_cli.profiles import _get_default_hermes_home, _iter_named_profile_dirs
    homes = [_get_default_hermes_home(), *_iter_named_profile_dirs()]
    return [migrate_profile_relay_env(home, validate=validate) for home in homes]


def print_relay_migration_report(results: list[RelayMigrationResult]) -> None:
    """Loud, actionable notice for `hermes update` / `hermes relay migrate`."""
    migrated = [r for r in results if r.migrated]
    failed = [r for r in results if r.validation_error]
    if not migrated and not failed:
        return
    print()
    if migrated:
        print("\033[1;33m⚠  NeMo Relay exporter configuration migrated\033[0m")
        print("   The legacy HERMES_NEMO_RELAY_ATIF_*/ATOF_* variables stopped producing traces after the")
        print("   Relay cutover. Each profile below now has a generated relay-plugins.toml selected by")
        print(f"   {RELAY_PLUGINS_CONFIG_ENV} in its .env (legacy lines commented out, not deleted):")
        for r in migrated:
            extra = f" ({len(r.diagnostics)} Relay diagnostic(s))" if r.diagnostics else ""
            label = r.home.name if r.home.parent.name == "profiles" else "default"
            print(f"     • {label}: {r.toml_path}{extra}")
        print("   Restart the gateway to resume exports. Review the file and adjust paths if needed.")
    for r in failed:
        print(f"   ✗ {r.home}: could not migrate Relay exporter vars — {r.validation_error}")
        print(f"     Write {r.home / RELAY_PLUGINS_TOML_NAME} by hand and set {RELAY_PLUGINS_CONFIG_ENV}.")


def run_relay_migration_after_update() -> None:
    """`hermes update` hook: migrate every profile home, print the notice. Best-effort by contract."""
    print_relay_migration_report(migrate_all_profile_relay_envs())


RELAY_MIGRATE_COMMAND = "hermes migrate relay"


def cmd_migrate_relay(args) -> None:
    """``hermes migrate relay [--all-profiles] [--no-validate]``."""
    from hermes_constants import get_hermes_home
    validate = not getattr(args, "no_validate", False)
    if getattr(args, "all_profiles", False):
        results = migrate_all_profile_relay_envs(validate=validate)
    else:
        results = [migrate_profile_relay_env(get_hermes_home(), validate=validate)]
    print_relay_migration_report(results)
    for r in results:
        if not r.migrated and not r.validation_error:
            print(f"  {r.home}: nothing to migrate ({r.skipped_reason}).")
