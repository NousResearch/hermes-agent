"""The hosted boot profile: one config switch, and the checks a shared process must pass to serve.

A hosted deployment runs ONE process for many tenants' agents. Nothing a tenant controls may run
in that process: their code runs in their agent's remote sandbox, their plugins and MCP servers
are served only through that sandbox, and their secrets live behind the secret store, never in a
profile file or the process environment. :func:`assert_hosted_boot` enforces that before the
gateway serves anything; each refusal names the condition and the fix. :func:`finish_hosted_boot`
runs at the end of hosted boot: it materialises the registrations the process serves and freezes
the registries, so no registration made while one agent's turn runs can change what another agent
sees.

A local install is untouched: :func:`is_hosted` answers ``False``, neither function does anything,
and plugins, MCP servers and providers keep loading, reloading and unloading at runtime as they
always have.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn, Optional

from hermes_scope_cache import tenant_neutral


class HostedBootRefused(RuntimeError):
    """A hosted process may not serve while the condition named in the message holds. The message
    names the condition and the fix."""


# The boot verdict: read once (by the first reader), then frozen for the process. Which kind of
# process this is decides what the turn paths may do, so a config edit must not flip it mid-process.
# Tenant-neutral by construction: it is a property of the deployment, identical for every tenant.
_VERDICT: dict[str, Optional[bool]] = tenant_neutral(
    {"hosted": None}, reason="the hosted boot verdict is decided once per process from deployment config")


def _hosted_root(config: Mapping[str, Any] | Any) -> Any:
    return config.get("hosted") if isinstance(config, Mapping) else None


def _hosted_enabled(config: Mapping[str, Any] | Any) -> bool:
    """The ``hosted`` root: ``hosted: {enabled: true, ...}``, or the shorthand ``hosted: true``."""
    hosted = _hosted_root(config)
    if isinstance(hosted, bool):
        return hosted
    return bool(isinstance(hosted, Mapping) and hosted.get("enabled"))


def _hosted_setting(config: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    hosted = _hosted_root(config)
    if not isinstance(hosted, Mapping):
        return default
    return hosted.get(key, default)


def _effective_config() -> dict[str, Any]:
    from hermes_cli.config_effective import load_user_config_effective

    return load_user_config_effective()


def is_hosted() -> bool:
    """Whether this process runs the hosted boot profile (``hosted.enabled`` in config.yaml).

    Read from the config backend on the first call, then frozen for the rest of the process: local
    behaviour and hosted refusals both depend on which kind of process this is, so a later config
    change must not flip a running process between the two.
    """
    cached = _VERDICT.get("hosted")
    if cached is None:
        cached = _hosted_enabled(_effective_config())
        _VERDICT["hosted"] = bool(cached)
    return cached


# --------------------------------------------------------------------------------------------------
# The boot refusals. Each names one condition and its fix; the boot stops at the first one.


def _refuse(condition: str, fix: str) -> NoReturn:
    raise HostedBootRefused(f"{condition}. {fix}")


def _looks_like_credential_env(name: str) -> bool:
    """Whether *name* is credential-shaped, by the same heuristics the env loader and the native
    child spawn scrub use (a credential the loader would sanitize, or a marker the child-scrub
    strips): the loader's credential suffixes, else a marker anywhere in the name."""
    from hermes_cli.env_loader import _CREDENTIAL_SUFFIXES
    from hermes_cli.local_runtime.processes import _CREDENTIAL_ENV_MARKERS

    upper = name.upper()
    return any(upper.endswith(suffix) for suffix in _CREDENTIAL_SUFFIXES) or any(
        marker in upper for marker in _CREDENTIAL_ENV_MARKERS)


def _operator_env_allowlist() -> tuple[frozenset[str], tuple[str, ...]]:
    """The operator's own process settings: what the kanban dispatcher already treats as host
    plumbing (exact names plus the locale prefix), plus the config backend's deployment names
    (which backend, which plane, which instance). Everything else in the environment is, at best,
    the operator's residue — and if it looks like a credential it refuses."""
    from hermes_cli.config_backend import deployment_env_names
    from hermes_cli.kanban_worker_env import _HOST_NAMES, _HOST_PREFIXES

    return frozenset(_HOST_NAMES | {name.upper() for name in deployment_env_names()}), _HOST_PREFIXES


def _refuse_credential_environment(env: Mapping[str, str] | None = None) -> None:
    """Refuse a credential-shaped environment variable outside the operator's allowlist."""
    names, prefixes = _operator_env_allowlist()
    for name in sorted(env if env is not None else os.environ):
        upper = name.upper()
        if upper in names or upper.startswith(prefixes):
            continue
        if _looks_like_credential_env(name):
            _refuse(
                f"a credential is set in the process environment: {name!r}",
                "A hosted process serves many tenants and nothing in its environment is a tenant's; "
                "delete the variable (tenant secrets live behind the secret store) or, if it is the "
                "operator's own deployment setting, give it a non-credential name")


def _refuse_profile_secret_files(profile_homes: tuple[Path, ...] = ()) -> None:
    """Refuse a profile ``.env`` or ``auth.json`` on disk: loading it would put one profile's
    secrets into the process every other tenant's agent shares."""
    from hermes_constants import get_hermes_home

    homes = [Path(get_hermes_home()), *(Path(home) for home in profile_homes)]
    seen: set[Path] = set()
    for home in homes:
        if home in seen:
            continue
        seen.add(home)
        for name in (".env", "auth.json"):
            if (home / name).is_file():
                _refuse(
                    f"a profile credential file exists on disk: {(home / name)}",
                    "Hosted secrets live behind the secret store, never in a profile's files; "
                    "move the credential into the deployment's secret store and delete the file")


def _refuse_non_sandbox_terminal(config: Mapping[str, Any]) -> None:
    """Refuse a terminal backend that is not a registered sandbox plugin serving hosted scopes:
    every tool call of a hosted agent must run in its own remote sandbox, never on this host."""
    terminal = config.get("terminal") if isinstance(config, Mapping) else None
    backend = str((terminal or {}).get("backend") or os.environ.get("TERMINAL_ENV") or "local").strip()
    lowered = backend.lower()
    from agent.terminal_env_registry import BUILTIN_BACKEND_NAMES, plugin_backend_names, provider_flag

    if lowered in ("", "local"):
        _refuse(
            f"terminal.backend is {backend!r}: a hosted agent's commands would run on this host",
            "Set terminal.backend to a sandbox backend plugin that declares serves_hosted_scopes "
            "(a remote sandbox per agent; hosted tools refuse every other backend)")
    if lowered in BUILTIN_BACKEND_NAMES:
        _refuse(
            f"terminal.backend {backend!r} is a built-in backend, not a sandbox plugin",
            "Hosted tools reach only a registered sandbox plugin backend that declares "
            "serves_hosted_scopes; point terminal.backend at the deployment's sandbox plugin")
    if lowered not in plugin_backend_names():
        _refuse(
            f"terminal.backend {backend!r} names no registered sandbox plugin",
            "Install the deployment's sandbox backend plugin (it registers a terminal environment "
            "provider that declares serves_hosted_scopes) and point terminal.backend at it")
    if provider_flag(lowered, "serves_hosted_scopes") is not True:
        _refuse(
            f"terminal.backend {backend!r} does not declare serves_hosted_scopes",
            "A hosted agent's tools run only in a backend that declares serves_hosted_scopes; "
            "use the deployment's sandbox plugin, or extend that plugin to serve hosted scopes")


def _refuse_direct_platform_adapters(gateway_config: Any) -> None:
    """Refuse any configured platform adapter other than the relay: hosted traffic arrives only
    through the relay, and the connector owns the bots (their tokens are never this process's)."""
    connected = gateway_config.get_connected_platforms() if gateway_config is not None else []
    direct = sorted(str(platform.value) for platform in connected if str(platform.value) != "relay")
    if direct:
        _refuse(
            f"direct platform adapter(s) configured: {', '.join(direct)}",
            "A hosted process accepts only relay traffic (the connector owns the platform bots); "
            "remove the platform credentials from the deployment's config")


_HOSTED_INPROCESS_PLUGIN_RULE = (
    "in-process plugins of a hosted process are accepted only from plugins bundled with Hermes "
    "or installed from the Hermes plugin catalog at tier verified or official")


def _plugins_that_would_load_inprocess() -> list[Any]:
    """The winning plugin manifests whose module this process would import: the full discovery
    order and gating, read from manifests only (nothing is imported here — an uncurated plugin
    must be refused before its code can load)."""
    from hermes_cli.plugins_discovery import (
        collect_directory_manifests, discover_entrypoint_manifests, resolve_manifest_winners,
    )
    from hermes_cli.plugins_manifest import manifest_key

    manifests = collect_directory_manifests()
    directory_keys = {manifest_key(manifest) for manifest in manifests}
    manifests.extend(m for m in discover_entrypoint_manifests() if manifest_key(m) not in directory_keys)
    return list(resolve_manifest_winners(manifests).values())


def _refuse_uncurated_inprocess_code(config: Mapping[str, Any]) -> None:
    """Refuse a plugin or MCP server that is neither bundled nor catalog-curated yet would load
    in-process. User-added plugins and MCP servers run in the agent's sandbox, never in the shared
    process; the operator's curated set is the bundled plugins plus catalog installs at tier
    verified or official."""
    servers = config.get("mcp_servers") if isinstance(config, Mapping) else None
    if isinstance(servers, Mapping) and servers:
        _refuse(
            f"MCP server(s) configured in this process's config: {', '.join(sorted(map(str, servers))[:5])}",
            "A hosted process runs no MCP server a user added (they are served through the agent's "
            "sandbox); remove the mcp_servers entries, and install operator MCP servers as "
            "catalog plugins")
    from hermes_cli.plugin_isolation import ISOLATION_HOST, isolation_mode
    from hermes_cli.plugins_cmd_catalog import login_backend_provenance_error
    from hermes_cli.plugins_discovery import IN_BOTH_LISTS, _get_disabled_plugins, _get_enabled_plugins, gate_manifest

    out_of_process = isolation_mode() == ISOLATION_HOST
    disabled, enabled = _get_disabled_plugins(), _get_enabled_plugins()
    for manifest in _plugins_that_would_load_inprocess():
        if manifest.source == "bundled" or out_of_process:
            # Bundled plugins ship with this process; under plugins.isolation: host every
            # non-bundled plugin runs on the profile's host instead of in this process.
            continue
        gate = gate_manifest(manifest, disabled, enabled)
        if gate.action == "placeholder" and gate.error is not None and (
                "disabled via config" in gate.error or "not enabled in config" in gate.error
                or gate.error == IN_BOTH_LISTS):
            continue  # config never runs it
        provenance = login_backend_provenance_error(manifest, rule=_HOSTED_INPROCESS_PLUGIN_RULE)
        if provenance:
            _refuse(
                f"plugin {manifest.name!r} would load in this process: {provenance}",
                "User-added plugins run in the agent's sandbox, never in the shared process; remove "
                "it, or install it from the Hermes plugin catalog at tier verified or official")


def _require_hosted_backends(config: Mapping[str, Any]) -> None:
    """Refuse a missing required backend: the identity verifier that admits an inbound frame, and
    the session storage provider that isolates tenants in storage (including its own isolation
    assertion)."""
    from hermes_state_port import resolve_session_storage

    verifier = str(_hosted_setting(config, "identity_verifier") or "").strip()
    if not verifier:
        _refuse(
            "no identity verifier is configured (hosted.identity_verifier)",
            "Every inbound frame must be admitted by a verifier that checks the connector's signed "
            "claim; install the deployment's verifier plugin and name it there")
    storage = str(_hosted_setting(config, "session_storage") or "").strip()
    if not storage:
        _refuse(
            "no session storage provider is configured (hosted.session_storage)",
            "Hosted sessions live behind a storage provider that isolates tenants; install the "
            "deployment's provider plugin and name it there")
    try:
        provider = resolve_session_storage(storage)
    except LookupError as exc:
        _refuse(
            f"the configured session storage provider is missing: {exc}",
            "Install the deployment's session storage provider plugin so the name in "
            "hosted.session_storage resolves; storage never falls back to another store")
    try:
        provider.assert_isolating()
    except Exception as exc:  # health: allow BLE001 -- a backend's isolation assertion names its own
        # failure mode (a role's grants, a server error); the boot reports it verbatim instead of
        # enumerating backends' exceptions here
        _refuse(
            f"the session storage provider {storage!r} does not isolate tenants in storage: {exc}",
            "The storage role must not be able to read or write another tenant's rows (a role that "
            "bypasses row security fails this); use the provider's isolating role")


def assert_hosted_boot(gateway_config: Any = None, *, profile_homes: tuple[Path, ...] = (),
                       config: Mapping[str, Any] | None = None,
                       env: Mapping[str, str] | None = None) -> None:
    """Every condition a hosted process must not serve with (see the module docstring). Each
    refusal raises :class:`HostedBootRefused` naming the condition and the fix. No-op when this
    process is not hosted.

    *gateway_config* is the gateway's own resolved config (its connected platforms); when it is
    ``None`` the platform check is skipped (a caller that serves nothing has no direct adapter).
    *profile_homes* are the profile directories the caller would serve besides the launch home.
    """
    if not is_hosted():
        return
    if config is None:
        config = _effective_config()
    _refuse_credential_environment(env)
    _refuse_profile_secret_files(profile_homes)
    _refuse_direct_platform_adapters(gateway_config)
    _refuse_uncurated_inprocess_code(config)
    # Discovery loads what the process serves, so the sandbox backend can be checked as
    # registered. Anything non-curated that would load was already refused above, before import.
    from hermes_cli.plugins import discover_plugins

    discover_plugins()
    _refuse_non_sandbox_terminal(config)
    _require_hosted_backends(config)


# --------------------------------------------------------------------------------------------------
# The end of hosted boot: materialise what the process serves, then freeze the registries.


def finish_hosted_boot() -> None:
    """Materialise the registrations this process serves, then freeze the registries
    (:func:`registry_freeze.freeze_registries`): from here on every registration entry point
    raises, and each per-agent toggle is a filter read from the bound scope. No-op when this
    process is not hosted.

    Called once the boot has loaded everything it serves (plugins, providers, MCP servers,
    adapters). Two registrations are still pending at that point and must be settled here, or the
    freeze would refuse and the boot would die: provider discovery (materialised by listing the
    providers once) and the deferred platform loaders (the bundled platforms this process does
    not serve are dropped — after the freeze their first lookup could not register them)."""
    if not is_hosted():
        return
    import providers
    from gateway.platform_registry import platform_registry
    from registry_freeze import freeze_registries

    providers.list_providers()  # provider discovery is one of the freeze's completeness checks
    dropped = platform_registry.drop_pending_deferred()
    if dropped:
        import logging

        logging.getLogger(__name__).info(
            "Hosted boot dropped platform loaders it does not serve (the registries freeze next): %s",
            ", ".join(dropped))
    freeze_registries()
