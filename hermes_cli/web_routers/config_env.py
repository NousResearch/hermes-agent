"""Config, env var and provider custom-endpoint dashboard routes.

Extracted from ``hermes_cli.web_server``; helpers/state that tests monkeypatch on
``web_server`` stay there and are late-bound (cycle-safe).
"""

import contextlib
import logging
import re
import asyncio
import time
import urllib.parse
from fastapi import APIRouter
from hermes_cli.web_routers._common import http_failure, scoped_to_thread
from hermes_cli.web_deps import LateState, late
from hermes_cli.web_server_config import (
    _apply_main_model_assignment, _denormalize_config_from_web, _normalize_config_for_web, _schema_with_dynamic_provider_options,
)
from hermes_cli.web_server_profiles import (
    _approval_mode_of, _broadcast_gateway_session_info, _is_other_profile, _parse_model_ids,
)
from fastapi import HTTPException, Request
from hermes_cli.config import DEFAULT_CONFIG, OPTIONAL_ENV_VARS, read_raw_config, custom_endpoint_key_env, coerce_provider_id, find_provider_entry, redact_key, _deep_merge
from hermes_cli.web_models import ConfigUpdate, EnvVarUpdate, EnvVarDelete, EnvVarReveal, CustomEndpointUpdate
from typing import Any, Dict, List, Optional, Tuple

_log = logging.getLogger("hermes_cli.web_server")
config_router = APIRouter()
router = APIRouter()

# Late-bound so a test's monkeypatch on the owning module wins at call time.
_channel_managed_env_keys = late("_channel_managed_env_keys", "hermes_cli.web_server_messaging")
_config_profile_scope = late("_config_profile_scope", "hermes_cli.web_server_profiles")
_profile_scope = late("_profile_scope", "hermes_cli.web_server_profiles")
_require_token = late("_require_token")
load_config = late("load_config", "hermes_cli.config")
load_env = late("load_env", "hermes_cli.config")
remove_env_value = late("remove_env_value", "hermes_cli.config")
save_config = late("save_config", "hermes_cli.config")
save_env_value = late("save_env_value", "hermes_cli.config")
_CONFIG_MUTATION_LOCK = LateState("_CONFIG_MUTATION_LOCK")

# Simple rate limiter for the reveal endpoint
_reveal_timestamps: List[float] = []
_REVEAL_MAX_PER_WINDOW = 5
_REVEAL_WINDOW_SECONDS = 30

# Display order for tabs — unlisted categories sort alphabetically after these.
_CATEGORY_ORDER = [
    "general", "agent", "terminal", "display", "delegation",
    "memory", "compression", "security", "browser", "voice",
    "tts", "stt", "logging", "discord", "auxiliary",
]


@contextlib.contextmanager
def _env_write_errors(log_msg: str, *, http_passthrough: bool):
    """``ValueError`` -> 400 with its message (save/remove_env_value reject
    invalid names and denylisted keys — LD_PRELOAD, PATH, PYTHONPATH, …, and
    the SPA needs the reason, not an opaque 500); anything else is logged and
    becomes 500 "Internal server error"."""
    try:
        yield
    except HTTPException:
        if http_passthrough:
            raise
        _log.exception(log_msg)
        raise HTTPException(status_code=500, detail="Internal server error")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        _log.exception(log_msg)
        raise HTTPException(status_code=500, detail="Internal server error")


@config_router.get("/api/config")
async def get_config(profile: Optional[str] = None):
    # _profile_scope blocks on the process-wide _SKILLS_PROFILE_LOCK and
    # load_config() reads from disk; a slow lock-holder on the event loop froze
    # the whole gateway for >1s. asyncio.to_thread copies the contextvar
    # context, so the profile override stays scoped to the worker thread.
    config = await scoped_to_thread(profile, lambda: _normalize_config_for_web(load_config()))
    # Strip internal keys that the frontend shouldn't see or send back
    return {k: v for k, v in config.items() if not k.startswith("_")}


@config_router.get("/api/config/defaults")
async def get_defaults():
    return DEFAULT_CONFIG


@config_router.get("/api/config/schema")
async def get_schema(profile: Optional[str] = None):
    # Discovery-driven provider options (voice command providers + memory
    # provider plugins) are merged per-request so providers added after server
    # start still show up, scoped to the requested profile's config.
    with _config_profile_scope(profile):
        fields = _schema_with_dynamic_provider_options()
    return {"fields": fields, "category_order": _CATEGORY_ORDER}


@config_router.get("/api/egress/status")
async def get_egress_status():
    """Dashboard/Desktop-readable egress proxy status and remediation text."""
    from hermes_cli.proxy_cli import format_status_text
    return {"text": format_status_text()}


@router.put("/api/config")
async def update_config(body: ConfigUpdate, profile: Optional[str] = None):
    def _run():
        approvals_mode_changed = False
        with _profile_scope(body.profile or profile):
            # The dashboard form is schema-driven; root keys absent from the
            # schema (``custom_providers``, ``agent.personalities``, ...) are not
            # in the PUT body, so deep-merge incoming over disk rather than
            # full-replace — the frontend can only overwrite what it sends.
            with _CONFIG_MUTATION_LOCK:
                existing = read_raw_config()
                incoming = _denormalize_config_from_web(body.config)
                merged = _deep_merge(existing, incoming)
                # Compare normalized approvals.mode across the in-memory
                # documents, not config blocks and not cache re-reads: the page
                # PUTs the defaulted GET record while disk holds sparse YAML (a
                # block compare is always-unequal), and a post-save reload can
                # serve the pre-save cache on an (mtime_ns, size) collision.
                # Only approvals.mode feeds session.info, so it is the trigger.
                approvals_mode_changed = _approval_mode_of(merged) != _approval_mode_of(existing)
                save_config(merged)
        # REST saves bypass the config.set RPC (which re-emits itself), so
        # refresh live sessions' cached approval/YOLO indicators after a mode
        # change. Own-profile saves only: a profile-scoped save targets a
        # different HERMES_HOME than this process's gateway sessions.
        if approvals_mode_changed and not _is_other_profile(body.profile or profile):
            _broadcast_gateway_session_info()
        return {"ok": True}

    with http_failure("PUT /api/config failed", 500, detail="Internal server error"):
        return await asyncio.to_thread(_run)


def _provider_card(d, description: str, url, *, is_password: bool, advanced: bool) -> dict:
    return {
        "provider": d.slug, "provider_label": d.label, "description": description, "url": url,
        "is_password": is_password, "advanced": advanced, "category": "provider",
    }


_AUTH_TYPE_ENV_VARS = {
    "aws_sdk": (
        ("AWS_REGION", lambda d, var: f"{d.label} ({var})"),
        ("AWS_PROFILE", lambda d, var: f"{d.label} ({var})"),
    ),
    "vertex": (("VERTEX_CREDENTIALS_PATH", lambda d, var: f"{d.label} — service account JSON path (or use ADC)"),),
}


def _catalog_provider_env_metadata() -> dict:
    """Map provider env vars -> desktop card metadata, derived from the catalog.

    Returns ``{env_var: {provider, provider_label, description, url, is_password,
    advanced}}`` for every API-key provider in the unified ``provider_catalog()``
    (the ``hermes model`` universe), so the desktop Keys tab renders a card even
    for providers never hand-added to ``OPTIONAL_ENV_VARS``. Hand
    ``OPTIONAL_ENV_VARS`` prose is layered on top in the endpoint; this only
    supplies membership + grouping + fallbacks.
    """
    try:
        from hermes_cli.provider_catalog import provider_catalog
    except Exception:
        return {}

    # Env vars declared with a NON-provider category (e.g. the shared
    # GITHUB_TOKEN, a Skills-Hub "tool" credential) must not be promoted into a
    # provider card even when a provider (Copilot) lists them as auth aliases.
    _non_provider_keys = {
        k for k, v in OPTIONAL_ENV_VARS.items()
        if (v or {}).get("category") and (v or {}).get("category") != "provider"
    }

    meta: dict = {}
    for d in provider_catalog():
        if d.tab != "keys":
            continue
        # API-key vars: the first is the primary (password) field; aliases are
        # kept as additional password fields so users can clear them too.
        for env_var in d.api_key_env_vars:
            if env_var in _non_provider_keys:
                continue  # don't hijack a shared tool/messaging credential
            meta.setdefault(
                env_var,
                _provider_card(d, d.description, d.signup_url or None, is_password=True, advanced=False),
            )
        # Base-URL override is an advanced, non-secret field for the same card.
        if d.base_url_env_var:
            meta.setdefault(
                d.base_url_env_var,
                _provider_card(d, f"{d.label} base URL override", None, is_password=False, advanced=True),
            )

        # Providers without api_key_env_vars would otherwise be invisible on
        # the Keys tab: AWS-SDK providers (Bedrock) authenticate via the AWS
        # credential chain, Vertex via OAuth2 (service-account JSON path or
        # ADC — a path, not a secret). Tag their env vars to the card.
        for var, describe in _AUTH_TYPE_ENV_VARS.get(d.auth_type, ()):
            existing = meta.get(var, {})
            meta[var] = _provider_card(
                d, existing.get("description") or describe(d, var), existing.get("url"),
                is_password=False, advanced=existing.get("advanced", True),
            )
    return meta


@router.get("/api/env")
async def get_env_vars(profile: Optional[str] = None):
    # _profile_scope takes _SKILLS_PROFILE_LOCK and load_env()/catalog
    # discovery read from disk — keep the whole build off the event loop.
    return await asyncio.to_thread(_get_env_vars_sync, profile)


def _get_env_vars_sync(profile: Optional[str] = None):
    with _profile_scope(profile):
        env_on_disk = load_env()
    channel_keys = _channel_managed_env_keys()
    catalog_meta = _catalog_provider_env_metadata()

    def _row(var_name: str, info: dict, *, custom: bool = False) -> dict:
        value = env_on_disk.get(var_name)
        cat_meta = catalog_meta.get(var_name) or {}
        # Hand OPTIONAL_ENV_VARS prose wins where present; the catalog fills any
        # gaps (description/url) and always supplies provider grouping hints.
        return {
            "is_set": bool(value),
            "redacted_value": redact_key(value) if value else None,
            "description": info.get("description") or cat_meta.get("description", ""),
            "url": info.get("url") if info.get("url") is not None else cat_meta.get("url"),
            "category": info.get("category") or cat_meta.get("category", ""),
            "is_password": info.get("password", cat_meta.get("is_password", False)),
            "tools": info.get("tools", []),
            "advanced": info.get("advanced", cat_meta.get("advanced", False)),
            # Messaging-platform credential owned by a Channels page card; the
            # Keys/Env page hides it rather than duplicate the richer UI.
            "channel_managed": var_name in channel_keys,
            # Provider grouping from the unified catalog, so the desktop groups
            # by the SAME provider identity the CLI `hermes model` picker uses.
            "provider": cat_meta.get("provider", ""),
            "provider_label": cat_meta.get("provider_label", ""),
            # True for a .env key in no catalog at all — an arbitrary/custom var
            # the user added directly, listed so the Keys page can manage it.
            "custom": custom,
        }

    result = {}
    for var_name, info in OPTIONAL_ENV_VARS.items():
        result[var_name] = _row(var_name, info)
    # Catalog provider env vars with no hand entry in OPTIONAL_ENV_VARS.
    for var_name in catalog_meta:
        if var_name not in result:
            result[var_name] = _row(var_name, {})
    # Custom keys from .env: always "set" (on disk), treated as secrets by
    # default (is_password=True -> redacted, reveal-gated) since an
    # unrecognised key could hold anything. Channel-managed credentials belong
    # to the Channels page. A key added via "add a custom key" round-trips here.
    for var_name in env_on_disk:
        if var_name in result or var_name in channel_keys:
            continue
        row = _row(var_name, {}, custom=True)
        row["category"] = "custom"
        row["is_password"] = True
        result[var_name] = row
    return result


@router.put("/api/env")
async def set_env_var(body: EnvVarUpdate, profile: Optional[str] = None):
    # Unified credential lifecycle: writes .env AND reconciles any config.yaml
    # mirror still holding the previous value of this var (model.api_key /
    # auxiliary.*.api_key / custom_providers[*]), so a rotation can't leave a
    # stale higher-precedence copy that keeps authenticating with the old key.
    with _env_write_errors("PUT /api/env failed", http_passthrough=False):
        from hermes_cli.credential_lifecycle import save_provider_env_credential

        return await scoped_to_thread(
            body.profile or profile, lambda: save_provider_env_credential(body.key, body.value)
        )


# Live credential probes keyed by env var: (url, auth) where auth is "bearer"
# (Authorization header) or "query" (?key=). A cheap read-only call that 401s
# on a bad token — enough to catch a mistyped key before it's persisted.
# Providers absent here (or local endpoints) are not network-validated; the
# client treats those as "unknown".
_CREDENTIAL_PROBES: dict[str, tuple[str, str]] = {
    "OPENROUTER_API_KEY": ("https://openrouter.ai/api/v1/key", "bearer"),
    "OPENAI_API_KEY": ("https://api.openai.com/v1/models", "bearer"),
    "XAI_API_KEY": ("https://api.x.ai/v1/models", "bearer"),
    "GEMINI_API_KEY": ("https://generativelanguage.googleapis.com/v1beta/models", "query"),
}


def _custom_endpoint_id(raw: str, fallback: str = "custom") -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", coerce_provider_id(raw)).strip("-_").lower()
    return slug or fallback


def _models_from_custom_endpoint_entry(entry: Dict[str, Any]) -> List[str]:
    models: List[str] = []
    raw_models = entry.get("models")
    if isinstance(raw_models, (dict, list)):
        models.extend(str(model).strip() for model in raw_models)

    default_model = str(entry.get("model") or entry.get("default_model") or "").strip()
    if default_model:
        models.insert(0, default_model)

    seen: set[str] = set()
    return [model for model in models if model and not (model in seen or seen.add(model))]


def _api_key_display(entry: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Return ``(has_api_key, preview)`` for a provider or model config block.

    Keys live in ``.env`` behind ``key_env``; only older entries still carry a
    plaintext ``api_key``. Checking both keeps the panel honest either way.

    See #69449.
    """
    plaintext = str(entry.get("api_key") or "").strip()
    if plaintext:
        return True, redact_key(plaintext)
    key_env = str(entry.get("key_env") or "").strip()
    if key_env:
        return True, f"${{{key_env}}}"
    return False, None


def _raw_provider_api_key(endpoint_id: str) -> Any:
    """The on-disk (un-expanded) ``api_key`` of a providers entry, or ``None``."""
    _stored, entry = find_provider_entry(read_raw_config().get("providers"), endpoint_id)
    return entry.get("api_key") if isinstance(entry, dict) else None


def _config_api_key_is_env_ref(endpoint_id: str) -> bool:
    """True when this endpoint's on-disk ``api_key`` is a ``${VAR}`` template.

    ``load_config()`` expands env refs, so a hand-written ``api_key: ${MY_KEY}``
    is indistinguishable from a literal secret by the time it reaches us. Such
    an entry already keeps its secret out of config.yaml, so migrating it would
    only copy that secret into a second env var the user didn't ask for.
    """
    raw_key = _raw_provider_api_key(endpoint_id)
    return bool(isinstance(raw_key, str) and re.search(r"\$\{[^}]+\}", raw_key))


def _endpoint_row(
    endpoint_id: str, name: str, base_url: str, model: str, models: List[str], context_length,
    discover_models: bool, key_entry: Dict[str, Any], is_current: bool, source: str,
) -> Dict[str, Any]:
    has_api_key, api_key_preview = _api_key_display(key_entry)
    return {
        "id": endpoint_id, "name": name, "base_url": base_url, "model": model, "models": models,
        "context_length": context_length, "discover_models": discover_models,
        "has_api_key": has_api_key, "api_key_preview": api_key_preview,
        "is_current": is_current, "source": source,
    }


def _custom_endpoint_response(cfg: Dict[str, Any]) -> Dict[str, Any]:
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    current_provider = str(model_cfg.get("provider", "") or "")
    current_model = str(model_cfg.get("default", model_cfg.get("name", "")) or "")
    current_base_url = str(model_cfg.get("base_url", "") or "")

    endpoints: List[Dict[str, Any]] = []
    providers = cfg.get("providers")
    if isinstance(providers, dict):
        for provider_id, raw_entry in providers.items():
            if not isinstance(raw_entry, dict):
                continue
            base_url = str(raw_entry.get("base_url") or raw_entry.get("url") or raw_entry.get("api") or "").strip()
            if not base_url:
                continue
            endpoint_id = str(provider_id)
            models = _models_from_custom_endpoint_entry(raw_entry)
            endpoints.append(_endpoint_row(
                endpoint_id, str(raw_entry.get("name") or endpoint_id), base_url,
                str(raw_entry.get("model") or raw_entry.get("default_model") or (models[0] if models else "")),
                models, raw_entry.get("context_length"), bool(raw_entry.get("discover_models", True)),
                raw_entry, endpoint_id == current_provider, "providers",
            ))

    if current_provider.lower() == "custom" and current_base_url and not any(e["id"] == "custom" for e in endpoints):
        endpoints.insert(0, _endpoint_row(
            "custom", "Custom", current_base_url, current_model, [current_model] if current_model else [],
            model_cfg.get("context_length"), True, model_cfg, True, "direct-config",
        ))

    return {
        "endpoints": endpoints,
        "current": {
            "provider": current_provider, "model": current_model, "base_url": current_base_url,
        },
    }


def _detach_main_model_from_provider(cfg: Dict[str, Any], provider_key: str) -> None:
    """Drop the main-slot mirror of a provider that no longer exists.

    ``activate_custom_endpoint`` copies the endpoint's ``base_url`` and
    ``api_key`` onto ``model``; that mirror outranks the environment at client
    construction, so deleting the endpoint without clearing it leaves the agent
    authenticating to the deleted host with the deleted key (and the key in
    config.yaml). Only touches ``model`` when it names the deleted provider.

    See #62269.
    """
    model_cfg = cfg.get("model")
    if not isinstance(model_cfg, dict):
        return
    if str(model_cfg.get("provider") or "").strip().lower() != provider_key:
        return
    for field in ("provider", "base_url", "api_key", "key_env"):
        model_cfg.pop(field, None)
    cfg["model"] = model_cfg


def _write_custom_endpoint(cfg: Dict[str, Any], body: CustomEndpointUpdate) -> Tuple[str, Dict[str, Any]]:
    endpoint_id = _custom_endpoint_id(body.id or body.name)
    name = (body.name or "").strip()
    base_url = (body.base_url or "").strip().rstrip("/")
    model = (body.model or "").strip()

    if not name:
        raise HTTPException(status_code=400, detail="name required")
    if not base_url:
        raise HTTPException(status_code=400, detail="base_url required")
    parsed = urllib.parse.urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise HTTPException(status_code=400, detail="base_url must include scheme and host")
    if not model:
        raise HTTPException(status_code=400, detail="model required")

    # Deliver the bearer token through a named provider entry. A bare ``provider: custom`` cannot carry a
    # credential for this host: OPENAI_API_KEY is deliberately gated to openai.com (#28660), so the token
    # was dropped and requests went out as "no-key-required".
    providers = cfg.get("providers")
    if not isinstance(providers, dict):
        providers = {}
    stored_key, existing = find_provider_entry(providers, endpoint_id)
    if existing is None:
        existing = {}

    # Merge onto the existing entry rather than replacing it: a providers.<name>
    # block can carry hand-written keys the dashboard has no field for
    # (``api_mode``, ``key_env``/``api_key_env``, ``extra_headers`` — possibly
    # with credentials — ``request_overrides``); rebuilding from scratch
    # silently dropped them on an unrelated edit.
    entry: Dict[str, Any] = dict(existing)
    entry.update({
        "name": name, "base_url": base_url, "model": model,
        "discover_models": bool(body.discover_models),
    })
    # Same for the model map, so existing models keep their context lengths.
    # ``body.models`` is the catalogue the panel's Test button discovered;
    # without it only the hand-typed model survived Save. A payload with no
    # ``models`` (older UI) still ensures the named default is present.
    # See #69988.
    existing_models = entry.get("models")
    models_map: Dict[str, Any] = dict(existing_models) if isinstance(existing_models, dict) else {}
    for candidate in (*(body.models or ()), model):
        model_id = str(candidate).strip()
        if not model_id:
            continue
        current = models_map.get(model_id)
        models_map[model_id] = dict(current) if isinstance(current, dict) else {}
    entry["models"] = models_map
    if body.context_length and body.context_length > 0:
        entry["context_length"] = int(body.context_length)
        entry["models"][model]["context_length"] = int(body.context_length)

    # API keys never belong in config.yaml: write to .env and reference it via
    # ``key_env`` — the indirection built-in providers use and that
    # runtime_provider.py resolves at load time.
    # See #69449.
    env_var = custom_endpoint_key_env(endpoint_id)
    submitted_key = body.api_key.strip() if body.api_key is not None else None
    if submitted_key:
        save_env_value(env_var, submitted_key)
        entry["key_env"] = env_var
        entry.pop("api_key", None)
    elif submitted_key is not None:
        # Blank field means "clear the key", not "leave it alone".
        remove_env_value(env_var)
        entry.pop("key_env", None)
        entry.pop("api_key", None)
    elif str(entry.get("api_key") or "").strip() and not _config_api_key_is_env_ref(endpoint_id):
        # Migrate a plaintext key an earlier release wrote, on the next save,
        # without the user having to re-enter it.
        save_env_value(env_var, entry["api_key"].strip())
        entry["key_env"] = env_var
        entry.pop("api_key", None)

    if stored_key is not None and stored_key != endpoint_id:
        providers.pop(stored_key, None)
    providers[endpoint_id] = entry
    cfg["providers"] = providers

    if body.make_default:
        cfg["model"] = _apply_main_model_assignment(
            cfg.get("model", {}), endpoint_id, model, base_url
        )
        if entry.get("key_env") and isinstance(cfg["model"], dict):
            cfg["model"]["key_env"] = entry["key_env"]
            cfg["model"].pop("api_key", None)

    return endpoint_id, entry


@router.get("/api/providers/custom-endpoints")
def list_custom_endpoints(profile: Optional[str] = None):
    """Return configured OpenAI-compatible custom endpoints for Desktop.

    Scoped to the requested profile's config.yaml: the desktop settings UI
    targets the active profile, so read/write must resolve that profile's home
    rather than the process-level HERMES_HOME (mirrors ``/api/config``).
    """
    with http_failure("GET /api/providers/custom-endpoints failed", 500, detail="Failed to list custom endpoints"):
        with _config_profile_scope(profile):
            return _custom_endpoint_response(load_config())


@router.post("/api/providers/custom-endpoints")
def upsert_custom_endpoint(body: CustomEndpointUpdate, profile: Optional[str] = None):
    """Create or update a v12+ ``providers`` custom endpoint entry."""
    with http_failure("POST /api/providers/custom-endpoints failed", 500, detail="Failed to save custom endpoint"):
        with _config_profile_scope(profile):
            cfg = load_config()
            endpoint_id, _entry = _write_custom_endpoint(cfg, body)
            save_config(cfg)
            response = _custom_endpoint_response(cfg)
        response["ok"] = True
        response["id"] = endpoint_id
        return response


@router.post("/api/providers/custom-endpoints/{endpoint_id}/activate")
def activate_custom_endpoint(endpoint_id: str, profile: Optional[str] = None):
    """Set a configured custom endpoint as the default model provider."""
    with http_failure(
        f"POST /api/providers/custom-endpoints/{endpoint_id}/activate failed", 500,
        detail="Failed to activate custom endpoint",
    ):
        with _config_profile_scope(profile):
            cfg = load_config()
            provider_key = _custom_endpoint_id(endpoint_id)
            _stored, entry = find_provider_entry(cfg.get("providers"), provider_key)
            if entry is None:
                raise HTTPException(status_code=404, detail="custom endpoint not found")

            models = _models_from_custom_endpoint_entry(entry)
            model = str(entry.get("model") or (models[0] if models else "")).strip()
            base_url = str(entry.get("base_url") or "").strip()
            if not model or not base_url:
                raise HTTPException(status_code=400, detail="custom endpoint is incomplete")

            model_cfg = _apply_main_model_assignment(cfg.get("model", {}), provider_key, model, base_url)
            if entry.get("key_env"):
                model_cfg["key_env"] = entry["key_env"]
                model_cfg.pop("api_key", None)
            elif entry.get("api_key"):
                # `cfg` is env-expanded, so a raw `${VAR}` api_key would land as
                # plaintext; copy the raw template when that's what's on disk.
                try:
                    _raw_key = str(_raw_provider_api_key(provider_key) or "").strip()
                except Exception:
                    _raw_key = ""
                if _raw_key.startswith("${") and _raw_key.endswith("}"):
                    model_cfg["api_key"] = _raw_key
                else:
                    model_cfg["api_key"] = entry["api_key"]
            cfg["model"] = model_cfg
            save_config(cfg)
        return {"ok": True, "provider": provider_key, "model": model}


@router.delete("/api/providers/custom-endpoints/{endpoint_id}")
def delete_custom_endpoint(endpoint_id: str, profile: Optional[str] = None):
    """Remove a configured custom endpoint from ``providers``."""
    with http_failure(
        f"DELETE /api/providers/custom-endpoints/{endpoint_id} failed", 500,
        detail="Failed to delete custom endpoint",
    ):
        with _config_profile_scope(profile):
            cfg = load_config()
            provider_key = _custom_endpoint_id(endpoint_id)
            providers = cfg.get("providers")
            stored_key, entry = find_provider_entry(providers, provider_key)
            if entry is None or not isinstance(providers, dict):
                raise HTTPException(status_code=404, detail="custom endpoint not found")
            providers.pop(stored_key, None)
            cfg["providers"] = providers
            _detach_main_model_from_provider(cfg, provider_key)
            remove_env_value(custom_endpoint_key_env(provider_key))
            save_config(cfg)
            response = _custom_endpoint_response(cfg)
        response["ok"] = True
        return response


@router.post("/api/providers/custom-endpoints/validate")
async def validate_custom_endpoint(body: CustomEndpointUpdate):
    """Probe a custom endpoint by calling its OpenAI-compatible /models URL."""
    import httpx

    base_url = (body.base_url or "").strip().rstrip("/")
    if not base_url:
        return {"ok": False, "reachable": True, "message": "Enter an endpoint URL first.", "models": []}

    url = base_url + "/models"
    headers = {"Accept": "application/json"}
    if body.api_key and body.api_key.strip():
        headers["Authorization"] = f"Bearer {body.api_key.strip()}"

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(8.0)) as client:
            resp = await client.get(url, headers=headers)
    except Exception:
        return {"ok": False, "reachable": False, "message": f"Could not reach {url}.", "models": []}

    if resp.status_code in (401, 403):
        return {"ok": False, "reachable": True, "message": "The endpoint rejected the API key.", "models": []}
    if not resp.is_success:
        return {"ok": False, "reachable": True, "message": f"Endpoint returned HTTP {resp.status_code}.", "models": []}

    return {"ok": True, "reachable": True, "message": "", "models": _parse_model_ids(resp)}


@router.post("/api/providers/validate")
async def validate_provider_credential(body: EnvVarUpdate, request: Request):
    """Live-probe a provider credential before it's saved.

    Returns {ok, reachable, message}. ok=True means the provider accepted the
    key; ok=False + reachable=True means the key is bad (caller should block);
    reachable=False means the network probe couldn't run (caller may save with
    a warning rather than hard-blocking offline users).
    """
    _require_token(request)
    import httpx

    key = (body.key or "").strip()
    value = (body.value or "").strip()
    if not value:
        return {"ok": False, "reachable": True, "message": "Enter a value first."}

    # Local / custom endpoint: validate connectivity, not auth — any HTTP
    # response (even 401) proves the endpoint is up. Also surface the model ids
    # it advertises (OpenAI ``/v1/models`` shape) so the GUI can auto-pick a
    # default. The optional API key is sent so servers that require auth on
    # ``/v1/models`` still enumerate instead of returning an empty list.
    if key == "OPENAI_BASE_URL":
        url = value.rstrip("/") + "/models"
        api_key = (body.api_key or "").strip()
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(8.0)) as client:
                resp = await client.get(url, headers=headers)
            return {"ok": True, "reachable": True, "message": "", "models": _parse_model_ids(resp)}
        except Exception:
            return {"ok": False, "reachable": False, "message": f"Could not reach {url}."}

    probe = _CREDENTIAL_PROBES.get(key)
    if not probe:
        # No probe for this provider — can't validate, don't block.
        return {"ok": True, "reachable": False, "message": ""}

    url, auth = probe
    headers = {"Accept": "application/json"}
    params = {}
    if auth == "bearer":
        headers["Authorization"] = f"Bearer {value}"
    else:
        params["key"] = value

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            resp = await client.get(url, headers=headers, params=params)
    except Exception:
        return {"ok": False, "reachable": False, "message": "Could not reach the provider to verify the key."}

    if resp.status_code in (401, 403):
        return {"ok": False, "reachable": True, "message": "That API key was rejected. Double-check it and try again."}
    if resp.status_code == 429 or resp.is_success:
        # 429 = key is valid but rate-limited; success = valid.
        return {"ok": True, "reachable": True, "message": ""}
    return {"ok": False, "reachable": True, "message": f"Provider returned HTTP {resp.status_code} for this key."}


@router.delete("/api/env")
async def remove_env_var(body: EnvVarDelete, profile: Optional[str] = None):
    # Unified credential lifecycle: clears the .env entry AND every mirror of
    # the credential — env-seeded credential_pool entries in auth.json (stale
    # ones kept providers alive in the model picker), the affected providers'
    # model-cache rows, and value-matched config.yaml api_key mirrors.
    # OAuth/device-code/manual pool entries for the same provider are preserved.
    with _env_write_errors("DELETE /api/env failed", http_passthrough=True):
        from hermes_cli.credential_lifecycle import remove_provider_env_credential

        result = await scoped_to_thread(
            body.profile or profile, lambda: remove_provider_env_credential(body.key)
        )
        if not result.get("found"):
            raise HTTPException(status_code=404, detail=f"{body.key} not found in .env")
        return result


@router.post("/api/env/reveal")
async def reveal_env_var(
    body: EnvVarReveal, request: Request, profile: Optional[str] = None
):
    """Return the real (unredacted) value of a single env var.

    Protected by the ephemeral session token (per server start, injected into
    the SPA), rate limiting (max 5 reveals per 30s window) and audit logging.
    """
    _require_token(request)

    now = time.time()
    cutoff = now - _REVEAL_WINDOW_SECONDS
    _reveal_timestamps[:] = [t for t in _reveal_timestamps if t > cutoff]
    if len(_reveal_timestamps) >= _REVEAL_MAX_PER_WINDOW:
        raise HTTPException(status_code=429, detail="Too many reveal requests. Try again shortly.")
    _reveal_timestamps.append(now)

    env_on_disk = await scoped_to_thread(body.profile or profile, load_env)
    value = env_on_disk.get(body.key)
    if value is None:
        raise HTTPException(status_code=404, detail=f"{body.key} not found in .env")

    _log.info("env/reveal: %s", body.key)
    return {"key": body.key, "value": value}
