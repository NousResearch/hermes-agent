"""Toolset / terminal-backend dashboard routes (extracted verbatim from
web_server.py).

Handler bodies are byte-identical.  The toolset/terminal catalogs
(``_MODEL_CATALOG_TOOLSETS``, ``_TERMINAL_BACKENDS``,
``_TERMINAL_BACKEND_NAMES`` - some defined *after* this router's mount point
in web_server's body) and all helpers stay in web_server - reached via the
late-binding seam in :mod:`hermes_cli.web_deps` (``late`` for callables,
``LateState`` for the catalog constants) so monkeypatching on web_server
stays authoritative.
"""

import asyncio
import logging
import sys  # noqa: F401 — used by handlers
from typing import Any, Dict, List, Optional  # noqa: F401

from fastapi import APIRouter, HTTPException  # noqa: F401

from hermes_cli.web_deps import late, LateState
from hermes_cli.web_models import (
    TerminalBackendSelect,
    ToolsetEnvUpdate,
    ToolsetModelSelect,
    ToolsetPostSetup,
    ToolsetProviderSelect,
    ToolsetToggle,
)

# Same logger the handlers used before extraction (identical logger object).
_log = logging.getLogger("hermes_cli.web_server")

router = APIRouter()

# Late-bound web_server helpers (resolved at call time; cycle-safe,
# monkeypatch-transparent).
_find_toolset_provider_row = late("_find_toolset_provider_row")
_probe_terminal_backend = late("_probe_terminal_backend")
_profile_cli_args = late("_profile_cli_args")
_profile_scope = late("_profile_scope")
_resolve_toolset_model_plugin = late("_resolve_toolset_model_plugin")
_spawn_hermes_action = late("_spawn_hermes_action")
_toolset_model_catalog = late("_toolset_model_catalog")
load_config = late("load_config")
save_config = late("save_config")
run_in_threadpool = late("run_in_threadpool")

# Live proxies for web_server-owned module state (mutations/monkeypatches
# on web_server remain authoritative; resolved at operation time).
_MODEL_CATALOG_TOOLSETS = LateState("_MODEL_CATALOG_TOOLSETS")
_TERMINAL_BACKENDS = LateState("_TERMINAL_BACKENDS")
_TERMINAL_BACKEND_NAMES = LateState("_TERMINAL_BACKEND_NAMES")
# Dynamic variants: built-ins + plugin-registered backends, computed per
# request so a plugin installed after server start still shows up.
_terminal_backend_rows = late("_terminal_backend_rows")
_terminal_backend_names = late("_terminal_backend_names")
# Config read-modify-write serialization for off-loop handlers (defined in
# web_server.py; LateState supports ``with``-blocks, so this is the live lock).
_CONFIG_MUTATION_LOCK = LateState("_CONFIG_MUTATION_LOCK")


@router.get("/api/tools/toolsets")
async def get_toolsets(profile: Optional[str] = None):
    from hermes_cli.tools_config import (
        _CONFIG_ONLY_TOOLSETS,
        _get_effective_configurable_toolsets,
        _get_platform_tools,
        _toolset_configuration_platform,
        _toolset_has_keys,
        get_nous_subscription_features,
        gui_toolset_label,
    )
    from hermes_cli.platforms import platform_label
    from toolsets import resolve_toolset

    def _read():
        with _profile_scope(profile):
            config = load_config()
            toolset_rows = _get_effective_configurable_toolsets()
            target_platforms = {
                _toolset_configuration_platform(name) for name, _, _ in toolset_rows
            }
            enabled_by_platform = {
                platform: _get_platform_tools(
                    config,
                    platform,
                    include_default_mcp_servers=False,
                )
                for platform in target_platforms
            }
            features = get_nous_subscription_features(config)
        return config, toolset_rows, enabled_by_platform, features

    config, toolset_rows, enabled_by_platform, features = await run_in_threadpool(_read)
    result = []
    for name, label, desc in toolset_rows:
        try:
            tools = sorted(set(resolve_toolset(name)))
        except Exception:
            tools = []
        target_platform = _toolset_configuration_platform(name)
        if name in _CONFIG_ONLY_TOOLSETS:
            # Config-only capabilities (stt) have no per-platform toolset —
            # their switch is their own config section (e.g. stt.enabled).
            from utils import is_truthy_value

            section = config.get(name)
            section = section if isinstance(section, dict) else {}
            is_enabled = is_truthy_value(section.get("enabled", True), default=True)
        else:
            is_enabled = name in enabled_by_platform[target_platform]
        result.append({
            "name": name,
            "label": gui_toolset_label(label),
            "description": desc,
            "platform": target_platform,
            "platform_label": gui_toolset_label(
                platform_label(target_platform, target_platform)
            ),
            "enabled": is_enabled,
            "available": is_enabled,
            "configured": _toolset_has_keys(name, config, features=features),
            "tools": tools,
        })
    return result


@router.put("/api/tools/toolsets/{name}")
async def toggle_toolset(name: str, body: ToolsetToggle, profile: Optional[str] = None):
    """Enable/disable a configurable toolset for its configuration platform.

    Most toolsets persist to ``platform_toolsets.cli``. Platform-restricted
    toolsets instead target their supported platform (for example, Discord's
    native toolsets persist to ``platform_toolsets.discord``). The shared
    ``_save_platform_tools`` helper keeps the GUI and CLI in lockstep. Scoped
    to ``body.profile`` when provided. Returns 400 for unknown toolset keys.
    """
    from hermes_cli.tools_config import (
        _CONFIG_ONLY_TOOLSETS,
        _get_effective_configurable_toolsets,
        _get_platform_tools,
        _save_platform_tools,
        _toolset_configuration_platform,
    )

    valid = {ts_key for ts_key, _, _ in _get_effective_configurable_toolsets()}
    if name not in valid:
        raise HTTPException(status_code=400, detail=f"Unknown toolset: {name}")

    target_platform = _toolset_configuration_platform(name)

    def _run():
        if name in _CONFIG_ONLY_TOOLSETS:
            # Config-only capabilities (stt) toggle their own config section's
            # ``enabled`` flag — there is no platform_toolsets entry to write.
            with _profile_scope(body.profile or profile):
                with _CONFIG_MUTATION_LOCK:
                    config = load_config()
                    section = config.setdefault(name, {})
                    if not isinstance(section, dict):
                        section = {}
                        config[name] = section
                    section["enabled"] = bool(body.enabled)
                    save_config(config)
            return
        with _profile_scope(body.profile or profile):
            with _CONFIG_MUTATION_LOCK:
                config = load_config()
                enabled = set(
                    _get_platform_tools(
                        config,
                        target_platform,
                        include_default_mcp_servers=False,
                    )
                )
                if body.enabled:
                    enabled.add(name)
                else:
                    enabled.discard(name)
                _save_platform_tools(config, target_platform, enabled)

    await asyncio.to_thread(_run)

    # Install-on-enable: when the newly enabled toolset's provider carries a
    # post_setup hook with a registered, UNSATISFIED install-state predicate
    # (cua-driver binary missing, etc. — see _POST_SETUP_INSTALLED), spawn the
    # same background install `hermes tools` runs interactively. Without this,
    # a dashboard/desktop toggle "saves" but the tool silently never appears
    # in the schema because its check_fn can't find the binary — the exact
    # dead-end that forced users to discover `hermes computer-use install`
    # by hand. Best-effort: a spawn failure never fails the toggle.
    post_setup_started: Optional[str] = None
    if body.enabled and name not in _CONFIG_ONLY_TOOLSETS:
        def _pending_install_key() -> Optional[str]:
            from hermes_cli.tools_config import (
                TOOL_CATEGORIES,
                _post_setup_already_installed,
                _visible_providers,
            )

            cat = TOOL_CATEGORIES.get(name)
            if not cat:
                return None
            with _profile_scope(body.profile or profile):
                config = load_config()
                for prov in _visible_providers(cat, config):
                    key = prov.get("post_setup")
                    if key and not _post_setup_already_installed(key):
                        return key
            return None

        try:
            pending_key = await asyncio.to_thread(_pending_install_key)
            if pending_key:
                _spawn_hermes_action(
                    _profile_cli_args(body.profile or profile)
                    + ["tools", "post-setup", pending_key],
                    "tools-post-setup",
                )
                post_setup_started = pending_key
        except Exception:
            _log.exception("install-on-enable post-setup spawn failed for %s", name)

    return {
        "ok": True,
        "name": name,
        "platform": target_platform,
        "enabled": body.enabled,
        "post_setup_started": post_setup_started,
    }


@router.get("/api/tools/toolsets/{name}/config")
async def get_toolset_config(name: str, profile: Optional[str] = None):
    """Return the provider matrix + key status for a toolset's config panel.

    Surfaces the same provider rows the CLI ``hermes tools`` picker shows
    (via ``_visible_providers``), each with its ``env_vars`` annotated with
    current ``is_set`` state so the GUI can render provider selection + key
    entry. Toolsets without a ``TOOL_CATEGORIES`` entry return an empty
    provider list and ``has_category: false``. Returns 400 for unknown keys.
    """
    from hermes_cli.tools_config import (
        TOOL_CATEGORIES,
        _get_effective_configurable_toolsets,
        _is_provider_active,
        _visible_providers,
        provider_readiness_status,
        web_provider_capabilities,
    )
    from hermes_cli.config import get_env_value
    from hermes_cli.nous_subscription import get_nous_subscription_features

    valid = {ts_key for ts_key, _, _ in _get_effective_configurable_toolsets()}
    if name not in valid:
        raise HTTPException(status_code=400, detail=f"Unknown toolset: {name}")

    def _read():
        with _profile_scope(profile):
            config = load_config()
            cat = TOOL_CATEGORIES.get(name)
            providers = []
            active_provider = None
            active_search_backend = None
            active_extract_backend = None
            if cat:
                # Fetch portal/entitlement state once for the whole matrix — the
                # per-provider readiness computation below reuses it instead of
                # re-probing per row.
                features = get_nous_subscription_features(config, force_fresh=True)
                for prov in _visible_providers(cat, config, force_fresh=True):
                    env_vars = [
                        {
                            "key": e["key"],
                            "prompt": e.get("prompt", e["key"]),
                            "url": e.get("url"),
                            "default": e.get("default"),
                            "is_set": bool(get_env_value(e["key"])),
                        }
                        for e in prov.get("env_vars", [])
                    ]
                    # Surface the same active-provider determination the CLI picker
                    # uses (``_is_provider_active``) so the GUI highlights the provider
                    # actually written to config (e.g. web.backend), not just the first
                    # keyless one in the list.
                    is_active = _is_provider_active(prov, config, force_fresh=True)
                    if is_active and active_provider is None:
                        active_provider = prov["name"]
                    row = {
                        "name": prov["name"],
                        "badge": prov.get("badge", ""),
                        "tag": prov.get("tag", ""),
                        "env_vars": env_vars,
                        "post_setup": prov.get("post_setup"),
                        "requires_nous_auth": bool(prov.get("requires_nous_auth")),
                        "is_active": is_active,
                        # Honest server-side readiness. The GUI's old client-side
                        # heuristic showed "Ready" for every zero-env-var row —
                        # including logged-out Nous Subscription rows and never-run
                        # post_setup installs (see provider_readiness_status).
                        "status": provider_readiness_status(
                            prov, config, features=features, is_active=is_active
                        ),
                    }
                    if name == "web" and prov.get("web_backend"):
                        # The runtime split web into two capabilities long ago
                        # (web.search_backend / web.extract_backend); surface each
                        # row's backend key and which capabilities it can serve so
                        # the GUI can offer per-capability selection.
                        row["web_backend"] = prov["web_backend"]
                        row["capabilities"] = web_provider_capabilities(prov["web_backend"])
                    if name == "tts" and prov.get("tts_provider"):
                        # The provider key written to tts.provider on selection.
                        # Doubles as the config section holding the provider's
                        # voice/model settings (tts.<key>.*) so the GUI can render
                        # those fields inline in the Capabilities panel.
                        row["tts_provider"] = prov["tts_provider"]
                    providers.append(row)
            if name == "web":
                # Resolve the per-capability active backends exactly the way the
                # web_search / web_extract dispatchers do (per-capability key →
                # shared web.backend → credential auto-detect), so the GUI badges
                # reflect what a tool call would actually hit right now.
                try:
                    from tools.web_tools import _get_extract_backend, _get_search_backend

                    active_search_backend = _get_search_backend()
                    active_extract_backend = _get_extract_backend()
                except Exception:
                    active_search_backend = None
                    active_extract_backend = None
        return cat, providers, active_provider, active_search_backend, active_extract_backend

    cat, providers, active_provider, active_search_backend, active_extract_backend = await asyncio.to_thread(_read)

    payload = {
        "name": name,
        "has_category": cat is not None,
        "providers": providers,
        "active_provider": active_provider,
    }
    if name == "web":
        payload["active_search_backend"] = active_search_backend
        payload["active_extract_backend"] = active_extract_backend
    return payload


@router.get("/api/tools/toolsets/{name}/models")
async def get_toolset_models(
    name: str, provider: Optional[str] = None, profile: Optional[str] = None
):
    """Return the model catalog for a toolset backend (image/video gen).

    The GUI counterpart of the model picker `hermes tools` runs after a
    backend is selected — e.g. FAL's multi-model catalog (speed / strengths /
    price per model). ``provider`` names a picker row; omitted, the currently
    active provider is used. Toolsets without model catalogs return
    ``has_models: false``.
    """
    section = _MODEL_CATALOG_TOOLSETS.get(name)
    if section is None:
        return {"name": name, "has_models": False, "models": [], "current": None, "default": None}

    def _read():
        with _profile_scope(profile):
            config = load_config()
            row = _find_toolset_provider_row(name, config, provider)
            plugin = _resolve_toolset_model_plugin(name, row) if row else None
            if not plugin:
                return None

            catalog, default_model = _toolset_model_catalog(name, plugin)
            section_cfg = config.get(section)
            current = None
            if isinstance(section_cfg, dict):
                raw = section_cfg.get("model")
                if isinstance(raw, str) and raw.strip():
                    current = raw.strip()
            if current not in catalog:
                current = default_model if default_model in catalog else None
        return row, plugin, catalog, default_model, current

    resolved = await asyncio.to_thread(_read)
    if resolved is None:
        return {
            "name": name,
            "has_models": False,
            "models": [],
            "current": None,
            "default": None,
        }
    row, plugin, catalog, default_model, current = resolved

    models = [
        {
            "id": model_id,
            "display": meta.get("display", model_id),
            "speed": meta.get("speed", ""),
            "strengths": meta.get("strengths", ""),
            "price": meta.get("price", ""),
        }
        for model_id, meta in catalog.items()
    ]
    return {
        "name": name,
        "has_models": bool(models),
        "provider": row.get("name") if row else None,
        "plugin": plugin,
        "models": models,
        "current": current,
        "default": default_model,
    }


@router.put("/api/tools/toolsets/{name}/model")
async def select_toolset_model(
    name: str, body: ToolsetModelSelect, profile: Optional[str] = None
):
    """Persist a backend model selection (``image_gen.model`` / ``video_gen.model``).

    Validates the model against the resolved backend's catalog — the same
    write the CLI's post-selection model picker performs. Returns 400 for
    toolsets without model catalogs or unknown model ids.
    """
    section = _MODEL_CATALOG_TOOLSETS.get(name)
    if section is None:
        raise HTTPException(
            status_code=400, detail=f"Toolset has no model catalog: {name}"
        )

    model_id = (body.model or "").strip()
    if not model_id:
        raise HTTPException(status_code=400, detail="model is required")

    def _run():
        with _profile_scope(body.profile or profile):
            with _CONFIG_MUTATION_LOCK:
                config = load_config()
                row = _find_toolset_provider_row(name, config, body.provider)
                plugin = _resolve_toolset_model_plugin(name, row) if row else None
                if not plugin:
                    raise HTTPException(
                        status_code=400,
                        detail=f"No model-capable backend is active for {name}",
                    )

                catalog, _default = _toolset_model_catalog(name, plugin)
                if model_id not in catalog:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unknown model {model_id!r} for backend {plugin!r}",
                    )

                section_cfg = config.setdefault(section, {})
                if not isinstance(section_cfg, dict):
                    section_cfg = {}
                    config[section] = section_cfg
                section_cfg["model"] = model_id
                save_config(config)
        return plugin

    plugin = await asyncio.to_thread(_run)
    return {"ok": True, "name": name, "model": model_id, "plugin": plugin}


@router.put("/api/tools/toolsets/{name}/provider")
async def select_toolset_provider(
    name: str, body: ToolsetProviderSelect, profile: Optional[str] = None
):
    """Persist a provider selection for a toolset (no key prompting).

    Delegates to ``apply_provider_selection`` — the shared, non-interactive
    core extracted from the CLI configurator — so the GUI and ``hermes tools``
    write identical config keys (``web.backend``, ``tts.provider``, etc.).
    API keys and post-setup flows are handled by separate endpoints. Returns
    400 for unknown toolset or provider names.

    For the ``web`` toolset only, an optional ``capability`` ('search' |
    'extract') scopes the selection to ``web.search_backend`` /
    ``web.extract_backend`` — the same per-capability overrides the runtime
    dispatchers (``tools.web_tools._get_search_backend`` /
    ``_get_extract_backend``) resolve first. The provider must actually
    support the requested capability (a search-only backend can't be the
    extract backend). Omitting ``capability`` keeps the legacy whole-provider
    behavior (writes ``web.backend``).

    Managed Nous rows (``managed_nous_feature``) additionally report the
    Portal entitlement state: the CLI flow gates these selections on
    ``ensure_nous_portal_access`` (inline login), but the GUI has no inline
    prompt, so selecting one while logged out / unentitled used to write the
    config keys and then never activate (``_is_provider_active`` requires
    ``managed_by_nous``). The response now carries an additive
    ``needs_nous_auth: true`` + ``feature`` so the client can drive the
    existing Nous Portal OAuth flow (``POST /api/providers/oauth/nous/start``)
    and refetch.
    """
    from hermes_cli.tools_config import (
        TOOL_CATEGORIES,
        apply_provider_selection,
        web_provider_capabilities,
        _get_effective_configurable_toolsets,
        _visible_providers,
    )
    from hermes_cli.nous_subscription import (
        MANAGED_FEATURE_COVERAGE_CATEGORY,
        get_nous_subscription_features,
    )

    valid = {ts_key for ts_key, _, _ in _get_effective_configurable_toolsets()}
    if name not in valid:
        raise HTTPException(status_code=400, detail=f"Unknown toolset: {name}")

    if body.capability is not None:
        if name != "web":
            raise HTTPException(
                status_code=400,
                detail="capability selection is only supported for the web toolset",
            )
        if body.capability not in ("search", "extract"):
            raise HTTPException(
                status_code=400,
                detail=f"Unknown capability: {body.capability!r} (expected 'search' or 'extract')",
            )

    def _run():
        with _profile_scope(body.profile or profile):
            with _CONFIG_MUTATION_LOCK:
                config = load_config()
                if body.capability is not None:
                    # Per-capability path: resolve the picker row to its backend key
                    # and write web.<capability>_backend. Does NOT touch web.backend,
                    # so the other capability keeps resolving through the shared
                    # fallback chain.
                    cat = TOOL_CATEGORIES.get(name)
                    providers = _visible_providers(cat, config, force_fresh=True) if cat else []
                    prov = next((p for p in providers if p.get("name") == body.provider), None)
                    if prov is None:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Unknown provider {body.provider!r} for toolset {name!r}",
                        )
                    backend = prov.get("web_backend")
                    if not backend:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Provider {body.provider!r} has no web backend key",
                        )
                    if body.capability not in web_provider_capabilities(backend):
                        raise HTTPException(
                            status_code=400,
                            detail=f"{body.provider} does not support {body.capability}",
                        )
                    web_cfg = config.setdefault("web", {})
                    if not isinstance(web_cfg, dict):
                        web_cfg = {}
                        config["web"] = web_cfg
                    web_cfg[f"{body.capability}_backend"] = backend
                else:
                    try:
                        apply_provider_selection(name, body.provider, config)
                    except KeyError as exc:
                        raise HTTPException(status_code=400, detail=str(exc).strip('"'))
                save_config(config)
                response: Dict[str, Any] = {"ok": True, "name": name, "provider": body.provider}
                if body.capability is not None:
                    response["capability"] = body.capability

            # Entitlement check for managed Nous rows — mirrors the gate the CLI
            # applies via ensure_nous_portal_access at selection time. This hits
            # the network (Portal), so it runs AFTER releasing the mutation lock:
            # holding a process-wide config-write lock across a network fetch
            # would stall every other config writer behind a slow Portal call.
            # Still inside the worker thread + profile scope.
            cat = TOOL_CATEGORIES.get(name)
            row = None
            if cat:
                row = next(
                    (
                        p
                        for p in _visible_providers(cat, config, force_fresh=True)
                        if p.get("name") == body.provider
                    ),
                    None,
                )
            managed_feature = (row or {}).get("managed_nous_feature")
            if managed_feature:
                features = get_nous_subscription_features(config, force_fresh=True)
                acct = features.account_info
                category = MANAGED_FEATURE_COVERAGE_CATEGORY.get(managed_feature)
                entitled = bool(
                    acct
                    and acct.logged_in
                    and (
                        acct.tool_gateway_entitled_for(category)
                        if category
                        else acct.tool_gateway_entitled
                    )
                )
                if not entitled:
                    response["needs_nous_auth"] = True
                    response["feature"] = managed_feature
        return response

    response = await asyncio.to_thread(_run)
    return response


@router.put("/api/tools/toolsets/{name}/env")
async def save_toolset_env(name: str, body: ToolsetEnvUpdate, profile: Optional[str] = None):
    """Persist API keys for a toolset's provider env vars.

    Writes each ``key: value`` to ``~/.hermes/.env`` via ``save_env_value`` —
    the same store ``hermes tools`` writes when it prompts for keys. Keys are
    validated against the env-var allowlist for the toolset's category (the
    union of every visible provider's ``env_vars``), so the GUI can't write an
    arbitrary env var through this endpoint. A blank value is treated as
    "leave unchanged" and skipped. Returns the saved/skipped key lists and the
    refreshed ``is_set`` status. Returns 400 for unknown toolset or env keys.
    """
    from hermes_cli.tools_config import (
        TOOL_CATEGORIES,
        _get_effective_configurable_toolsets,
        _visible_providers,
    )
    from hermes_cli.config import get_env_value, save_env_value

    valid_ts = {ts_key for ts_key, _, _ in _get_effective_configurable_toolsets()}
    if name not in valid_ts:
        raise HTTPException(status_code=400, detail=f"Unknown toolset: {name}")

    def _run():
        with _profile_scope(body.profile or profile):
            config = load_config()
            cat = TOOL_CATEGORIES.get(name)
            allowed: set[str] = set()
            if cat:
                for prov in _visible_providers(cat, config, force_fresh=True):
                    for e in prov.get("env_vars", []):
                        allowed.add(e["key"])

            unknown = [k for k in body.env if k not in allowed]
            if unknown:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown env var(s) for toolset {name}: {', '.join(sorted(unknown))}",
                )

            saved: List[str] = []
            skipped: List[str] = []
            for key, value in body.env.items():
                if value and value.strip():
                    try:
                        save_env_value(key, value.strip())
                    except ValueError as exc:
                        raise HTTPException(status_code=400, detail=str(exc))
                    saved.append(key)
                else:
                    skipped.append(key)

            status = {k: bool(get_env_value(k)) for k in allowed}
        return saved, skipped, status

    saved, skipped, status = await asyncio.to_thread(_run)
    return {"ok": True, "name": name, "saved": saved, "skipped": skipped, "is_set": status}


@router.post("/api/tools/toolsets/{name}/post-setup")
async def run_toolset_post_setup(
    name: str, body: ToolsetPostSetup, profile: Optional[str] = None
):
    """Spawn a provider's post-setup install hook as a background action.

    Post-setup hooks (npm install for browser/Camofox, pip install for
    KittenTTS/Piper/ddgs, cua-driver fetch, etc.) are long-running and
    text-output, so this follows the spawn-action pattern: it launches
    ``hermes tools post-setup <key>`` and the frontend tails the log via
    ``GET /api/actions/tools-post-setup/status``. The ``key`` is validated
    against the declared post-setup allowlist before spawning. Returns 400
    for unknown toolset or post-setup key.

    ``profile`` spawns the hook as ``hermes -p <profile> tools post-setup``.
    Most hooks install machine-level artifacts (repo node_modules, shared
    pip packages) where the scope is inert, but hooks that read config or
    write per-profile state must see the same HERMES_HOME the rest of the
    drawer's writes targeted — so the scope is threaded for consistency.
    """
    from hermes_cli.tools_config import (
        _get_effective_configurable_toolsets,
        valid_post_setup_keys,
    )

    valid_ts = {ts_key for ts_key, _, _ in _get_effective_configurable_toolsets()}
    if name not in valid_ts:
        raise HTTPException(status_code=400, detail=f"Unknown toolset: {name}")

    if body.key not in valid_post_setup_keys():
        raise HTTPException(
            status_code=400, detail=f"Unknown post-setup key: {body.key}"
        )

    try:
        proc = _spawn_hermes_action(
            _profile_cli_args(body.profile or profile)
            + ["tools", "post-setup", body.key],
            "tools-post-setup",
        )
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("Failed to spawn tools post-setup")
        raise HTTPException(
            status_code=500, detail=f"Failed to run post-setup: {exc}"
        )
    return {"ok": True, "pid": proc.pid, "name": "tools-post-setup", "key": body.key}


@router.get("/api/tools/terminal/backends")
async def get_terminal_backends(profile: Optional[str] = None):
    """Terminal execution backend rows with health probes for the picker panel.

    Returns ``{active, backends: [{name, label, description, active, status,
    detail}]}`` where ``status`` is ``ready`` / ``needs_setup`` /
    ``unavailable`` and ``detail`` carries setup guidance for non-ready rows.
    Probes are fast (<~2s each) and defensive — a probe failure surfaces as a
    status, never an error response.
    """
    def _read():
        with _profile_scope(profile):
            config = load_config()
            terminal_cfg = config.get("terminal")
            if not isinstance(terminal_cfg, dict):
                terminal_cfg = {}
            rows = _terminal_backend_rows()
            active = str(terminal_cfg.get("backend") or "local").strip().lower()
            if active not in {row["name"] for row in rows}:
                active = "local"

            backends = []
            for row in rows:
                status, detail = _probe_terminal_backend(row["name"], terminal_cfg)
                backends.append({
                    "name": row["name"],
                    "label": row["label"],
                    "description": row["description"],
                    "active": row["name"] == active,
                    "status": status,
                    "detail": detail,
                })
        return {"active": active, "backends": backends}

    return await asyncio.to_thread(_read)


@router.put("/api/tools/terminal/backend")
async def select_terminal_backend(
    body: TerminalBackendSelect, profile: Optional[str] = None
):
    """Persist ``terminal.backend`` in config.yaml.

    Validates against the known backend set (the same enum the raw-config
    settings row exposes). Selecting a backend that still needs setup is
    allowed — the picker shows guidance instead of blocking, matching the CLI.
    """
    backend = (body.backend or "").strip().lower()
    valid_names = _terminal_backend_names()
    if backend not in valid_names:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown terminal backend: {body.backend!r}. "
            f"Use one of: {', '.join(sorted(valid_names))}",
        )

    def _run():
        with _profile_scope(body.profile or profile):
            with _CONFIG_MUTATION_LOCK:
                config = load_config()
                terminal_cfg = config.setdefault("terminal", {})
                if not isinstance(terminal_cfg, dict):
                    terminal_cfg = {}
                    config["terminal"] = terminal_cfg
                if backend == "apple_container":
                    status, detail = _probe_terminal_backend(backend, terminal_cfg)
                    if status == "unavailable":
                        raise HTTPException(
                            status_code=400,
                            detail=detail or "Apple Container is unavailable on this host.",
                        )
                terminal_cfg["backend"] = backend
                save_config(config)

    await asyncio.to_thread(_run)
    return {"ok": True, "backend": backend}


@router.get("/api/tools/computer-use/status")
async def get_computer_use_status(profile: Optional[str] = None):
    """Cross-platform Computer Use readiness for the desktop card.

    See ``tools.computer_use.permissions.computer_use_status`` for the payload
    shape. Read-only and fast (shells ``cua-driver doctor`` + macOS
    ``permissions status``).
    """
    from tools.computer_use.permissions import computer_use_status

    def _read():
        with _profile_scope(profile):
            return computer_use_status()

    return await asyncio.to_thread(_read)


@router.post("/api/tools/computer-use/permissions/grant")
async def grant_computer_use_permissions(profile: Optional[str] = None):
    """Spawn ``hermes computer-use permissions grant`` as a background action.

    macOS-only: ``cua-driver permissions grant`` launches CuaDriver via
    LaunchServices so the TCC dialog is attributed to com.trycua.driver, then
    waits for approval. The frontend polls ``GET /api/actions/computer-use-
    grant/status`` and re-reads ``/status`` once it exits. Windows/Linux have
    no TCC toggles to grant, so this returns 400 there.
    """
    if sys.platform != "darwin":
        raise HTTPException(
            status_code=400,
            detail="Computer Use permission grants are a macOS concept.",
        )
    try:
        proc = _spawn_hermes_action(
            _profile_cli_args(profile)
            + ["computer-use", "permissions", "grant"],
            "computer-use-grant",
        )
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("Failed to spawn computer-use permissions grant")
        raise HTTPException(
            status_code=500, detail=f"Failed to request permissions: {exc}"
        )
    return {"ok": True, "pid": proc.pid, "name": "computer-use-grant"}
