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
# ``config`` is the effective config the boot recorded when it resolved the deployment's own
# settings; while it is set, every hosted-mode read consults it and re-reads nothing.
# ``plugins_permitted`` is the discovery gate: a hosted process imports plugin code only after the
# import-free preflight (:func:`hosted_boot_preflight`) refused everything uncurated.
_VERDICT: dict[str, Any] = tenant_neutral(
    {"hosted": None, "config": None, "plugins_permitted": False},
    reason="the hosted boot verdict is decided once per process from deployment config")


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


def _boot_or_effective_config() -> dict[str, Any]:
    """The config the boot recorded when it resolved the deployment's settings, else one effective
    read. Recording exists so the boot's first read is the ONLY one: every later hosted-mode
    consultation (the boot checks, the discovery gate, the freeze) reads the recorded mapping
    with no further file I/O."""
    recorded = _VERDICT.get("config")
    if isinstance(recorded, Mapping):
        return dict(recorded)
    return _effective_config()


def record_boot_config(config: Mapping[str, Any]) -> None:
    """Record the effective config the boot itself resolved, so every later hosted-mode read
    (:func:`is_hosted`, the boot checks) consults it instead of re-reading the profile's files.
    A config that CONTRADICTS the process's frozen verdict refuses instead of flipping it:
    the kind of process this is was decided once, and a second, disagreeing record is a
    mismatch the boot must stop on — an armed code fence cannot be disarmed by a verdict
    change, so letting the two diverge would serve a boot whose checks answered a different
    config than its enforcement did."""
    hosted = _hosted_enabled(config)
    frozen = _VERDICT.get("hosted")
    if isinstance(frozen, bool) and hosted != frozen:
        _refuse(
            f"the config the boot would record disagrees with the verdict frozen for this "
            f"process (this process is {'hosted' if frozen else 'local'}, the config is not)",
            "The kind of process this is was decided once from the deployment's config; "
            "restart with one consistent config instead of recording a second one that "
            "contradicts it")
    _VERDICT["config"] = config
    _VERDICT["hosted"] = hosted


def is_hosted() -> bool:
    """Whether this process runs the hosted boot profile (``hosted.enabled`` in config.yaml).

    Read from the config backend on the first call, then frozen for the rest of the process: local
    behaviour and hosted refusals both depend on which kind of process this is, so a later config
    change must not flip a running process between the two.
    """
    cached = _VERDICT.get("hosted")
    if cached is None:
        cached = _hosted_enabled(_boot_or_effective_config())
        _VERDICT["hosted"] = bool(cached)
    return cached


def hosted_verdict_frozen() -> Optional[bool]:
    """The frozen boot verdict — ``True``/``False`` once this process's kind is decided,
    ``None`` before the first :func:`is_hosted` call resolves it. Readers that must never
    trigger (or re-enter) a config read — the bundled-plugins root among them — consult this
    instead of :func:`is_hosted`: until the boot's first read decides the process they keep
    the local behaviour, and a hosted verdict freezes before the preflight lets any plugin
    code load."""
    verdict = _VERDICT.get("hosted")
    return verdict if isinstance(verdict, bool) else None


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
    "in-process code of a hosted process is only what its deployment image ships: the plugins "
    "bundled with Hermes, or a package installed into the image's environment")


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
    """Refuse plugin or MCP-server code that is not what the deployment image ships yet would
    load in this process. The one shared predicate decides (:func:`hermes_cli.hosted_code_fence.
    hosted_code_refusal`): bundled inside the installed Hermes package, or an entry point of
    a package installed into the image's environment. Everything under the deployment home is
    refused, whatever its install record says — user-added plugins and MCP servers run in the
    agent's sandbox, and an operator's curated set is baked into the image, not installed into a
    tenant-writable home.

    This is the discovery-layer pre-check that gives the boot its clean, early refusal message;
    enforcement does not depend on it; the installed code fence
    (:func:`hermes_cli.hosted_code_fence.install_hosted_code_fence`) refuses any load the
    enumeration missed. In particular a dotted entry point's parent package is refused by the
    fence before its module-level code can run."""
    servers = config.get("mcp_servers") if isinstance(config, Mapping) else None
    if isinstance(servers, Mapping) and servers:
        _refuse(
            f"MCP server(s) configured in this process's config: {', '.join(sorted(map(str, servers))[:5])}",
            "A hosted process runs no MCP server a user added (they are served through the agent's "
            "sandbox); remove the mcp_servers entries, and install operator MCP servers as "
            "catalog plugins")
    from hermes_cli.hosted_code_fence import hosted_code_refusal
    from hermes_cli.plugin_isolation import ISOLATION_HOST, isolation_mode
    from hermes_cli.plugins_discovery import IN_BOTH_LISTS, _get_disabled_plugins, _get_enabled_plugins, gate_manifest

    out_of_process = isolation_mode() == ISOLATION_HOST
    disabled, enabled = _get_disabled_plugins(), _get_enabled_plugins()
    refusals: list[str] = []
    for manifest in _plugins_that_would_load_inprocess():
        if out_of_process:
            # Under plugins.isolation: host every non-bundled plugin runs on the profile's host
            # instead of in this process.
            continue
        gate = gate_manifest(manifest, disabled, enabled)
        if gate.action == "placeholder" and gate.error is not None and (
                "disabled via config" in gate.error or "not enabled in config" in gate.error
                or gate.error == IN_BOTH_LISTS):
            continue  # config never runs it
        # The one rule answers entry points and directories alike: an entry point's module is
        # judged by where its file resolves to — never by the distribution's metadata alone.
        if manifest.source == "entrypoint":
            refusal = hosted_code_refusal(
                None, module=str(manifest.path or "").partition(":")[0].strip())
        else:
            refusal = hosted_code_refusal(manifest.path)
        if refusal:
            refusals.append(f"plugin {manifest.name!r} would load in this process: {refusal}")
    if refusals:
        _refuse(
            "; ".join(refusals),
            "User-added code runs in the agent's sandbox, never in the shared process; a hosted "
            f"deployment's own plugins are baked into its image. {_HOSTED_INPROCESS_PLUGIN_RULE}")


def _require_hosted_backends(config: Mapping[str, Any]) -> None:
    """Refuse a missing required backend: the identity verifier that admits an inbound frame, and
    the session storage provider that isolates tenants in storage (including its own isolation
    assertion)."""
    from hermes_identity_verifier import resolve_identity_verifier
    from hermes_state_port import resolve_session_storage

    verifier = str(_hosted_setting(config, "identity_verifier") or "").strip()
    if not verifier:
        _refuse(
            "no identity verifier is configured (hosted.identity_verifier)",
            "Every inbound frame must be admitted by a verifier that checks the connector's signed "
            "claim; install the deployment's verifier plugin and name it there")
    try:
        identity = resolve_identity_verifier(verifier)
    except LookupError as exc:
        _refuse(
            f"the configured identity verifier is missing: {exc}",
            "Install the deployment's identity verifier plugin so the name in "
            "hosted.identity_verifier resolves; a hosted frame is admitted by the named verifier "
            "or by nobody, never by a fallback")
    if not identity.is_available():
        _refuse(
            f"the configured identity verifier {verifier!r} is not available here",
            "The verifier must be able to check every frame this process serves (the reference "
            "turn-claim verifier needs the deployment's relay id); fix its configuration")
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


def _prune_sys_path_outside_image() -> None:
    """Drop every ``sys.path`` entry that resolves outside the deployment image (a ``.pth``-added
    home directory, an inherited ``PYTHONPATH`` residue) and record what was pruned in the boot
    log: nothing outside the image stays searchable once the boot serves."""
    import logging

    from hermes_cli.hosted_code_fence import prune_sys_path_outside_image

    pruned = prune_sys_path_outside_image()
    if pruned:
        logging.getLogger(__name__).info(
            "Hosted boot removed sys.path entries outside the deployment image: %s",
            ", ".join(pruned))


def hosted_boot_preflight(*, profile_homes: tuple[Path, ...] = (),
                          config: Mapping[str, Any] | None = None,
                          env: Mapping[str, str] | None = None) -> None:
    """The import-free hosted checks, run before anything resolves config through the plugin
    loader: a config read discovers and imports plugins, and a plugin this process must refuse
    must be refused before its module-level code can run. No-op when this process is not hosted.

    The start-flag confirmation runs first, in both directions: the hosted start command's
    ``--hosted`` flag armed the process-wide code fence at process entry, before any
    application import (:func:`hermes_cli.hosted_code_fence.arm_from_start_argv`), and this
    preflight is where the flag is confirmed against the deployment's config. A hosted boot
    whose start command did not arm the fence (so its application imported unfenced) refuses,
    and so does a flagged start on a deployment whose config does not enable hosted — the
    verdict the flag decided and the verdict the config records must agree. Passing clears the
    discovery gate (:func:`plugin_loading_permitted`); a refusal raises before any plugin code
    loaded. Checks manifests, config and provenance records only.
    """
    from hermes_cli.hosted_code_fence import armed_by_start_flag

    if not is_hosted():
        if armed_by_start_flag():
            _refuse(
                "the start command passed --hosted but this deployment's config does not "
                "enable hosted",
                "The --hosted start flag exists to arm the process-wide code fence before any "
                "application import; a deployment that is not hosted must not pass it — "
                "remove the flag from the start command, or enable hosted in the "
                "deployment's config")
        return
    if not armed_by_start_flag():
        _refuse(
            "a hosted boot must start with the --hosted start flag",
            "The flag is what arms the process-wide code fence before any application "
            "import — a boot that arms it only here would import the application first; "
            "add --hosted to the hosted gateway's start command")
    from hermes_cli.hosted_code_fence import startup_isolation_refusal

    isolation_refusal = startup_isolation_refusal()
    if isolation_refusal:
        _refuse(
            f"a hosted boot requires an interpreter that started isolated from user and "
            f"data site paths, and this one did not: {isolation_refusal}",
            "Start the image's python with -I or -s plus PYTHONNOUSERSITE, from site "
            "directories the image owns — the fence can verify isolation, never retrofit "
            "it onto a start whose site initialization already ran")
    if config is None:
        config = _boot_or_effective_config()
    record_boot_config(config)  # also the config-versus-frozen-verdict confirmation
    from hermes_cli.hosted_code_fence import install_hosted_code_fence

    install_hosted_code_fence()  # idempotent: the start flag already armed the fence at entry
    _refuse_credential_environment(env)
    _refuse_profile_secret_files(profile_homes)
    _refuse_uncurated_inprocess_code(config)
    _prune_sys_path_outside_image()
    _VERDICT["plugins_permitted"] = True


def plugin_loading_permitted() -> bool:
    """Whether plugin code may load in this process yet. Never gated on a local install; on a
    hosted one, not until the import-free preflight (:func:`hosted_boot_preflight`) passed —
    before that, every discovery path is a no-op, so no plugin's module-level code runs ahead of
    the boot's refusal."""
    return not is_hosted() or _VERDICT.get("plugins_permitted") is True


def assert_hosted_boot(gateway_config: Any = None, *, profile_homes: tuple[Path, ...] = (),
                       config: Mapping[str, Any] | None = None,
                       env: Mapping[str, str] | None = None) -> None:
    """Every condition a hosted process must not serve with (see the module docstring). Each
    refusal raises :class:`HostedBootRefused` naming the condition and the fix. No-op when this
    process is not hosted.

    *gateway_config* is the gateway's own resolved config (its connected platforms); when it is
    ``None`` the platform check is skipped (a caller that serves nothing has no direct adapter).
    *profile_homes* are the profile directories the caller would serve besides the launch home.

    The preflight runs first — it is also what confirms the start command's ``--hosted`` flag
    against config, in both directions — then the checks that need loaded things run. A local
    install stops at the preflight's no-op, exactly as before.
    """
    hosted_boot_preflight(profile_homes=profile_homes, config=config, env=env)
    if not is_hosted():
        return
    if config is None:
        config = _boot_or_effective_config()  # the preflight recorded it: no further file I/O
    _refuse_direct_platform_adapters(gateway_config)
    # Discovery loads what the process serves, so the sandbox backend can be checked as
    # registered. Anything non-curated that would load was already refused above, before import.
    from hermes_cli.plugins import discover_plugins

    discover_plugins()
    _refuse_non_sandbox_terminal(config)
    _require_hosted_backends(config)


# --------------------------------------------------------------------------------------------------
# The end of hosted boot: materialise what the process serves, then freeze the registries.


def _refuse_incomplete_builtin_toolset() -> None:
    """Builtin tool discovery is part of hosted boot, not of the first turn: every ``tools/*.py``
    module self-registers at import, and a registration after the freeze raises, so a tool whose
    module was never imported would be missing for every tenant until restart. The boot imports
    them synchronously and refuses unless the core set is registered."""
    import model_tools  # importing runs builtin discovery: each tools/*.py self-registers
    from toolsets import _HERMES_CORE_TOOLS
    from tools.registry import registry

    registered = set(registry.get_all_tool_names())
    missing = sorted({*_HERMES_CORE_TOOLS, "terminal", "read_file", "write_file"} - registered)
    if missing:
        _refuse(
            f"builtin tool discovery did not register: {', '.join(missing)}",
            "A hosted process freezes the registries once boot ends, so every builtin tool must be "
            "registered before that; check the tool-discovery log lines above for the module that "
            "failed to import")


def finish_hosted_boot() -> None:
    """Materialise the registrations this process serves, then freeze the registries
    (:func:`registry_freeze.freeze_registries`): from here on every registration entry point
    raises, and each per-agent toggle is a filter read from the bound scope. No-op when this
    process is not hosted.

    Called once the boot has loaded everything it serves (plugins, providers, MCP servers,
    adapters). Registrations still pending at that point are settled here, or the freeze would
    refuse and the boot would die: builtin tool discovery (mandatory and synchronous — the
    freeze would refuse each tool module's later import), provider discovery (materialised by
    listing the providers once) and the deferred platform loaders (the bundled platforms this
    process does not serve are dropped — after the freeze their first lookup could not register
    them)."""
    if not is_hosted():
        return
    import providers
    from gateway.platform_registry import platform_registry
    from registry_freeze import freeze_registries

    _refuse_incomplete_builtin_toolset()
    providers.list_providers()  # provider discovery is one of the freeze's completeness checks
    dropped = platform_registry.drop_pending_deferred()
    if dropped:
        import logging

        logging.getLogger(__name__).info(
            "Hosted boot dropped platform loaders it does not serve (the registries freeze next): %s",
            ", ".join(dropped))
    freeze_registries()
