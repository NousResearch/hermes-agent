"""Shared model-switching logic for the CLI and gateway /model commands.

Pipeline: parse flags -> alias resolution -> provider resolution -> credential resolution ->
normalize model name -> metadata lookup -> build result. Provider switching uses ``--provider``
exclusively; colons are reserved for OpenRouter variant suffixes (``:free``, ``:extended``)."""

from __future__ import annotations

import http.client
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Optional

from hermes_cli.providers import (
    ProviderDef, custom_provider_aliases, determine_api_mode, get_label, host_mandated_api_mode,
    is_aggregator, resolve_provider_full)
from hermes_cli.model_normalize import normalize_model_for_provider
from agent.models_dev import (
    ModelCapabilities, ModelInfo, get_model_capabilities, get_model_info, list_provider_models)
from utils import base_url_host_matches, base_url_hostname, base_url_origin, file_signature
# Re-exported: callers/tests patch hermes_cli.model_switch.<name>.
from hermes_cli.model_switch_providers import list_authenticated_providers


logger = logging.getLogger(__name__)


def _declared_model_ids(value: Any) -> list[str]:
    """Configured model IDs from ``{"id": {...}}``, ``["a", "b"]``, ``[{"id"|"name": ...}]`` or ``"a"``."""
    if isinstance(value, str):
        candidates: Any = [value]
    elif isinstance(value, dict):
        # Pre-fix Hermes wrote sentinel keys inside the user-facing ``models`` mapping.
        candidates = (k for k in value if k not in ("__explicit_model_allowlist__", "__discovered_model_catalog__"))
    elif isinstance(value, (list, tuple)):
        candidates = (_declared_item_id(item) if isinstance(item, dict) else item for item in value)
    else:
        return []
    ids: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue  # non-str items are dropped
        model_id = candidate.strip()
        if not model_id:
            return
        lowered = model_id.lower()
        if lowered in seen:
            return
        seen.add(lowered)
        ids.append(model_id)

    if isinstance(value, str):
        _add(value)
        return ids

    if isinstance(value, dict):
        for model_id in value:
            # Backward compat: pre-fix Hermes wrote sentinel keys inside the
            # user-facing ``models`` mapping. Never list them as model IDs.
            if model_id in {
                "__explicit_model_allowlist__",
                "__discovered_model_catalog__",
            }:
                continue
            _add(model_id)
        return ids

    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str):
                _add(item)
                continue
            if isinstance(item, dict):
                model_id = item.get("id")
                if not isinstance(model_id, str) or not model_id.strip():
                    model_id = item.get("name")
                _add(model_id)
        return ids

    return ids


def _entry_models_discovered(entry: Any) -> bool:
    """True when the entry's ``models`` mapping was auto-discovered by Hermes.

    The current shape is an entry-level ``models_discovered: true`` sibling of
    ``models``. Older Hermes versions wrote an in-mapping
    ``__discovered_model_catalog__: true`` sentinel instead — accept that on
    read for backward compatibility (the next discovery save migrates the
    entry to the clean shape).
    """
    if not isinstance(entry, dict):
        return False
    if entry.get("models_discovered") is True:
        return True
    models = entry.get("models")
    return (
        isinstance(models, dict)
        and models.get("__discovered_model_catalog__") is True
    )


def _models_config_is_allowlist(value: Any, discovered: bool = False) -> bool:
    """Return True when ``models:`` is an intentional ID allowlist.


    List/string shapes remain allowlists for no-key endpoints. To pin a
    dict-shaped catalog, set ``discover_models: false``.

    ``discovered`` is the entry-level ``models_discovered`` flag (see
    ``_entry_models_discovered``): a catalog Hermes itself persisted after a
    successful probe is never a user pin, whatever its shape.
    """
    if discovered:
        return False
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        return bool(_declared_model_ids(value))
    return False


def _save_discovered_models_to_config(
    api_url: str,
    model_ids: list[str],
    *,
    api_mode: Optional[str] = None,
    headers: Optional[dict[str, str]] = None,
) -> None:
    """Persist discovered models into ``custom_providers`` in config.yaml.

    Called after a successful ``/v1/models`` probe so that the next read
    with ``discover_models: false`` uses the cached list instead of a stale
    or minimal manually-configured subset.

    Matches entries by ``base_url`` (trailing-slash-normalised).  A failed
    config write is swallowed — the picker still shows the live models for
    this session.
    """
    if not api_url or not model_ids:
        return
    try:
        from hermes_cli.config import load_config, save_config

        cfg = load_config()
        providers = cfg.get("custom_providers") or []
        if not isinstance(providers, list):
            return

        norm_url = api_url.strip().rstrip("/").lower()
        changed = False
        for entry in providers:
            if not isinstance(entry, dict):
                continue
            entry_url = (entry.get("base_url", "") or entry.get("url", "")).strip()
            if entry_url.rstrip("/").lower() != norm_url:
                continue
            entry_mode = str(
                entry.get("api_mode") or entry.get("transport") or ""
            ).strip().lower() or None
            if entry_mode != api_mode:
                continue
            if headers is not None:
                entry_headers = _extra_headers_from_config(entry)
                if entry_headers != headers:
                    continue
            existing = entry.get("models")
            legacy_discovered = (
                isinstance(existing, dict)
                and existing.get("__discovered_model_catalog__") is True
            )
            entry_discovered = (
                entry.get("models_discovered") is True or legacy_discovered
            )
            # Preserve per-model metadata: when ``models`` is a mapping
            # (e.g. ``{"model-a": {"context_length": 8192}}``) or a list of
            # dicts (e.g. ``[{"id": "model-a", "context_length": 8192}]``),
            # the user has curated metadata per model — do not replace it.
            # A mapping Hermes itself discovered (``models_discovered: true``
            # or the legacy in-mapping sentinel) is ours to refresh.
            if isinstance(existing, dict) and not entry_discovered:
                continue
            if isinstance(existing, list) and any(
                isinstance(m, dict) for m in existing
            ):
                continue
            # Only update when models are stale — avoids unnecessary
            # config writes on every picker open.  A legacy-shape entry
            # (sentinel inside ``models``) is always rewritten so the next
            # save migrates it to the clean entry-level flag.
            if isinstance(existing, list) and existing == model_ids:
                continue
            if (
                isinstance(existing, dict)
                and entry_discovered
                and not legacy_discovered
                and list(existing) == model_ids
            ):
                continue
            entry["models"] = {model_id: {} for model_id in model_ids}
            entry["models_discovered"] = True
            changed = True

        if changed:
            cfg["custom_providers"] = providers
            save_config(cfg)
    except Exception:
        pass


def _bare_custom_provider_def(current_base_url: str) -> Optional[ProviderDef]:
    """ProviderDef for a direct ``model.provider: custom`` endpoint."""
    base_url = _clean(current_base_url)
    if not base_url:
        return None
    return ProviderDef(
        id="custom", name="Custom endpoint", transport="openai_chat", api_key_env_vars=(),
        base_url=base_url, is_aggregator=False, auth_type="api_key", source="model-config")


_MODEL_DISCOVERY_ERRORS = (
    ImportError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    http.client.HTTPException,
)


class _NativePickerModelList(list[str]):
    """A successful native catalog, including an authoritative empty one."""


def _fetch_picker_live_models(
    api_key: str,
    api_url: str,
    native_catalog_provider: str,
    preserve_native_models: bool,
    headers: dict[str, str] | None = None,
    timeout: float = 5.0,
    api_mode: str | None = None,
) -> list[str] | None:
    """Fetch picker models with native Ollama and cached generic discovery."""
    from hermes_cli.models import (
        _get_ollama_native_headers,
        _normalize_openai_base_url,
        cached_fetch_api_models,
        fetch_ollama_local_models,
        should_use_ollama_native_catalog,
    )

    candidate_headers = _get_ollama_native_headers(api_url, api_key=api_key)
    caller_has_authorization = any(
        key.lower() == "authorization" for key in (headers or {})
    )
    if caller_has_authorization:
        for key in tuple(candidate_headers):
            if key.lower() == "authorization":
                del candidate_headers[key]
    if headers:
        for key in tuple(candidate_headers):
            if any(key.lower() == existing.lower() for existing in headers):
                del candidate_headers[key]
        candidate_headers.update(headers)
    if api_key and not caller_has_authorization:
        for key in tuple(candidate_headers):
            if key.lower() == "authorization":
                del candidate_headers[key]
        candidate_headers["Authorization"] = f"Bearer {api_key}"
    use_native = should_use_ollama_native_catalog(
        native_catalog_provider, api_url, headers=candidate_headers or None
    )
    resolved_headers = candidate_headers or None if use_native else headers

    if use_native:
        if preserve_native_models:
            return None
        native_models = fetch_ollama_local_models(
            api_url, timeout=timeout, headers=resolved_headers
        )
        if native_models is not None:
            return _NativePickerModelList(native_models)
        # A failed native probe is not authoritative: retry the cached generic
        # OpenAI-compatible catalog before reporting no models.
        return cached_fetch_api_models(
            api_key,
            _normalize_openai_base_url(api_url),
            timeout=timeout,
            headers=resolved_headers,
            api_mode=api_mode,
        )
    generic_models = cached_fetch_api_models(
        api_key,
        api_url,
        timeout=timeout,
        headers=resolved_headers,
        api_mode=api_mode,
    )
    return generic_models if generic_models else None


# ---------------------------------------------------------------------------
# Non-agentic model warning
# ---------------------------------------------------------------------------

_HERMES_MODEL_WARNING = (
    "Nous Research Hermes 3 & 4 models are NOT agentic and are not designed "
    "for use with Hermes Agent. They lack the tool-calling capabilities "
    "required for agent workflows. Consider using an agentic model instead "
    "(Claude, GPT, Gemini, DeepSeek, etc.).")

# Match only the real Nous Research Hermes 3 / 4 chat families; a bare substring check
# false-positived on tool-capable local Modelfiles like ``hermes-brain:qwen3-14b-ctx16k``.
#   match:    NousResearch/Hermes-3-Llama-3.1-70B, hermes-4-405b, openrouter/hermes3:70b
#   no match: hermes-brain:qwen3-14b-ctx16k, qwen3:14b, claude-opus-4-6
_NOUS_HERMES_NON_AGENTIC_RE = re.compile(r"(?:^|[/:])hermes[-_ ]?[34](?:[-_.:]|$)", re.IGNORECASE)


# Opaque proxy model IDs (Palantir Foundry: ``ri.language-model-service..language-model.<slug>``)
# are noise in status output; the provider_label already carries the routing context. Stripped
# for DISPLAY ONLY — never for wire-side comparison, persistence, config writes or alias lookup.
_OPAQUE_MODEL_PREFIXES: tuple[str, ...] = ("ri.language-model-service..language-model.",)


def format_model_for_display(model_name: str) -> str:
    """Human-friendly form of *model_name* for CLI status output (display only, never wire-side)."""
    for prefix in _OPAQUE_MODEL_PREFIXES:
        if model_name and model_name.startswith(prefix):
            return model_name[len(prefix):] or model_name
    return model_name


def is_nous_hermes_non_agentic(model_name: str) -> bool:
    """True if *model_name* is a real Nous Hermes 3/4 chat model (single owner; cli.py uses it too)."""
    return bool(model_name and _NOUS_HERMES_NON_AGENTIC_RE.search(model_name))


def _check_hermes_model_warning(model_name: str) -> str:
    """Warning string if *model_name* is a Nous Hermes 3/4 chat model, else ""."""
    return _HERMES_MODEL_WARNING if is_nous_hermes_non_agentic(model_name) else ""


# --- Model aliases -- short names -> (vendor, family) with NO version numbers,
# resolved dynamically against the live models.dev catalog.

class ModelIdentity(NamedTuple):
    """Vendor slug and family prefix used for catalog resolution."""
    vendor: str
    family: str


MODEL_ALIASES: dict[str, ModelIdentity] = {
    "sonnet":    ModelIdentity("anthropic", "claude-sonnet"),
    "opus":      ModelIdentity("anthropic", "claude-opus"),
    "haiku":     ModelIdentity("anthropic", "claude-haiku"),
    "claude":    ModelIdentity("anthropic", "claude"),
    "gpt5":      ModelIdentity("openai", "gpt-5"),
    "gpt":       ModelIdentity("openai", "gpt"),
    "codex":     ModelIdentity("openai", "codex"),
    "o3":        ModelIdentity("openai", "o3"),
    "o4":        ModelIdentity("openai", "o4"),
    "gemini":    ModelIdentity("google", "gemini"),
    "deepseek":  ModelIdentity("deepseek", "deepseek-chat"),
    "grok":      ModelIdentity("x-ai", "grok"),
    "llama":     ModelIdentity("meta-llama", "llama"),
    "qwen":      ModelIdentity("qwen", "qwen"),
    "minimax":   ModelIdentity("minimax", "minimax"),
    "nemotron":  ModelIdentity("nvidia", "nemotron"),
    "kimi":      ModelIdentity("moonshotai", "kimi"),
    "glm":       ModelIdentity("z-ai", "glm"),
    "step":      ModelIdentity("stepfun", "step"),
    "mimo":      ModelIdentity("xiaomi", "mimo"),
    "trinity":   ModelIdentity("arcee-ai", "trinity")}


# --- Direct aliases — exact model+provider+base_url for endpoints outside the
# models.dev catalog (Ollama Cloud, local servers). Checked BEFORE catalog
# resolution; loaded from config.yaml ``model_aliases:`` / ``model.aliases``.

class DirectAlias(NamedTuple):
    """Exact model mapping that bypasses catalog resolution.

    ``api_key`` / ``key_env`` carry the alias endpoint's OWN credential. Without them the switch
    would keep the *default* provider's key, which 401s against the alias host and sends that
    provider's secret to an unrelated third party. Both default so positional
    ``DirectAlias(model, provider, base_url)`` keeps working.

    See #83612.
    """
    model: str
    provider: str
    base_url: str
    api_key: str = ""
    key_env: str = ""


# Built-in direct aliases (extended via config.yaml model_aliases:)
_BUILTIN_DIRECT_ALIASES: dict[str, DirectAlias] = {}


def _clean(value: Any) -> str:
    """``str(value or "").strip()`` — the config-field normaliser used throughout this module."""
    return str(value or "").strip()

# Merged dict (builtins + user config); populated by _load_direct_aliases()
DIRECT_ALIASES: dict[str, DirectAlias] = {}


def _load_direct_aliases() -> dict[str, DirectAlias]:
    """Load direct aliases from config.yaml.

    ``model_aliases:`` entries are dicts (``model``, ``provider``, ``base_url``, optional
    ``api_key`` — literal or ``"${VAR}"`` — / ``key_env``); with neither credential field the key
    is resolved from the alias HOST, never from the previously active provider. ``model.aliases``
    never overrides ``model_aliases``; its string entries (``ds-flash: deepseek/deepseek-v4-flash``)
    take the provider from the ``provider/`` prefix, else the current provider.

    See #83612.
    """
    merged = dict(_BUILTIN_DIRECT_ALIASES)
    try:
        from hermes_cli.config import load_config
        cfg = load_config()

        user_aliases = cfg.get("model_aliases")
        if isinstance(user_aliases, dict):
            for name, entry in user_aliases.items():
                if isinstance(entry, dict) and entry.get("model", ""):
                    merged[name.strip().lower()] = DirectAlias(
                        model=entry.get("model", ""), provider=entry.get("provider", "custom"),
                        base_url=entry.get("base_url", ""), api_key=_clean(entry.get("api_key", "")),
                        key_env=_clean(entry.get("key_env", "")))

        model_section = cfg.get("model", {})
        simple_aliases = model_section.get("aliases") if isinstance(model_section, dict) else None
        if isinstance(simple_aliases, dict):
            current_provider = model_section.get("provider", "")
            for name, value in simple_aliases.items():
                key = name.strip().lower()
                if not key or key in merged:
                    continue
                if isinstance(value, dict):
                    model = _clean(value.get("model"))
                    if model:
                        merged[key] = DirectAlias(
                            model=model, provider=_clean(value.get("provider")) or current_provider or "custom",
                            base_url=_clean(value.get("base_url")))
                elif isinstance(value, str) and value.strip():
                    val = value.strip()
                    provider, model = val.split("/", 1) if "/" in val else (current_provider, val)
                    merged[key] = DirectAlias(
                        model=model.strip(), provider=provider.strip() or current_provider, base_url="")
    except Exception:
        pass
    return merged


# Identity of the config the cached aliases were built from. The cache is process-global but its
# source is profile-local: unkeyed, the first profile to resolve an alias would pin its definitions
# — and, since entries carry `api_key`, its credentials — for every later profile. Same shape
# `load_config()` keys on, so a profile switch (path) and a key rotation (mtime/size) both invalidate.
_DIRECT_ALIAS_IDENTITY: Optional[tuple] = None
# Copy of what the loader last produced. Callers and tests seed DIRECT_ALIASES both by rebinding
# and by editing in place, so only comparing against what we wrote tells our stale cache from
# someone else's contents.
_DIRECT_ALIAS_LOADED: Optional[dict] = None


def _direct_alias_source_identity() -> Optional[tuple]:
    """Identity of the active profile's alias source; None means "do not reuse the cache"."""
    try:
        from hermes_constants import get_config_path
        path = get_config_path()
    except Exception:
        return None
    try:
        stat = path.stat()
    except OSError:
        # A missing config is still a definite identity for this profile.
        return (str(path), None)
    except Exception:
        return None
    return (str(path), file_signature(stat))


def _ensure_direct_aliases() -> None:
    """Load direct aliases for the ACTIVE profile, caching per config identity.

    Mutates DIRECT_ALIASES in place (never rebinds) so ``from ... import DIRECT_ALIASES``
    references in callers stay valid."""
    global _DIRECT_ALIAS_IDENTITY, _DIRECT_ALIAS_LOADED
    identity = _direct_alias_source_identity()
    if DIRECT_ALIASES and (
        # Contents are not what we loaded — seeded or edited by a caller. Not ours to discard.
        DIRECT_ALIASES != _DIRECT_ALIAS_LOADED
        # Ours, and still the same config file at the same signature.
        or (identity is not None and identity == _DIRECT_ALIAS_IDENTITY)):
        return
    loaded = _load_direct_aliases()
    DIRECT_ALIASES.clear()
    DIRECT_ALIASES.update(loaded)
    _DIRECT_ALIAS_IDENTITY = identity
    _DIRECT_ALIAS_LOADED = dict(loaded)


def direct_alias_api_key(alias: DirectAlias) -> str:
    """Resolve a direct alias's own credential, or "" when it has none.

    Precedence: ``api_key: "${VAR}"`` (env indirection) > literal ``api_key`` > ``key_env``.
    Env reads go through the per-profile secret scope: a raw ``os.environ`` read hands this
    profile whatever key the process env holds — another profile's, under the multiplexed gateway."""
    raw = (alias.api_key or "").strip()
    if raw.startswith("${") and raw.endswith("}"):
        return _scoped_key_env(raw[2:-1].strip())
    if raw:
        return raw
    return _scoped_key_env((alias.key_env or "").strip())


def direct_alias_runtime_request(alias: DirectAlias) -> tuple[str, Optional[str]]:
    """``(requested_provider, explicit_api_key)`` for resolving *alias*.

    Single owner of the invariant that a URL-bearing direct alias resolves its credential for
    the alias HOST, never for its provider label: a label like ``anthropic`` on an unrelated URL
    would otherwise reach that provider's explicit-runtime branch and put the live vendor token
    on the foreign wire. Bare ``custom`` is host-gated, so an authoritative URL still resolves
    its vendor key and a foreign one resolves none. An alias with no base_url keeps its label —
    there is no foreign host, and the label is the only routing information.

    See #28660.
    """
    return ("custom" if alias.base_url else (alias.provider or "custom")), direct_alias_api_key(alias) or None


# Hosts where plaintext HTTP is not a downgrade — no network hop to intercept.
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})


def _may_reuse_session_credential(session_base_url: str, alias_base_url: str) -> bool:
    """Whether the session's key may follow a switch to *alias_base_url*.

    Same hostname is NOT sufficient: ``http://h`` and ``https://h:8443`` are different trust
    boundaries, and an alias that drops the scheme would put a live bearer secret on the wire in
    the clear. Require an identical (scheme, host, port) and refuse plaintext outside loopback."""
    session = base_url_origin(session_base_url)
    alias = base_url_origin(alias_base_url)
    if not session[1] or session != alias:
        return False
    scheme, hostname, _ = alias
    return scheme == "https" or hostname in _LOOPBACK_HOSTS


class StartupModelRoute(NamedTuple):
    """Model/provider pair resolved before an agent is constructed."""
    model: str
    provider: str = ""
    base_url: str = ""
    api_key: str = ""


def resolve_startup_model_route(
    raw_model: str, *, explicit_provider: str = "", current_provider: str = "",
    user_providers: Optional[dict] = None,
    custom_providers: Optional[list] = None) -> Optional[StartupModelRoute]:
    """Resolve aliases and configured ``provider/model`` input at startup.

    ``HermesCLI`` is constructed before the interactive ``/model`` pipeline runs; resolving here
    keeps startup from attaching the configured default provider to an explicitly requested
    model. ``provider/model`` strings are consumed only for providers present in user config. When
    ``current_provider`` is a routing aggregator and the raw string is an aggregator-native slug
    (``anthropic/claude-opus-4.6`` on OpenRouter) the input stays on the aggregator — a
    ``providers:`` block for the same vendor must not steal the route."""
    raw = _clean(raw_model)
    if not raw:
        return None

    _ensure_direct_aliases()
    direct = DIRECT_ALIASES.get(raw.lower())
    if direct is not None:
        if explicit_provider:
            # An explicit --provider wins over the alias's own label; the alias contributes
            # model/base_url only.
            return StartupModelRoute(model=direct.model, provider=explicit_provider, base_url=direct.base_url)
        # Same owner as the interactive /model and oneshot paths: credential for the alias HOST.
        # Resolve through the SAME owner the interactive /model and oneshot paths use: a URL-bearing alias
        # must resolve its credential for the alias HOST, never for its provider label — a label like
        # ``anthropic`` on a foreign URL would otherwise reach that provider's explicit-runtime branch and
        # put the live vendor token on the foreign wire (#28660).
        alias_provider, alias_key = direct_alias_runtime_request(direct)
        return StartupModelRoute(
            model=direct.model, provider=alias_provider, base_url=direct.base_url, api_key=alias_key or "")

    if explicit_provider or "/" not in raw:
        return None
    prefix, model = (part.strip() for part in raw.split("/", 1))
    if not prefix or not model:
        return None

    if current_provider:
        try:
            from hermes_cli.providers import is_routing_aggregator, normalize_provider as _norm_prov
            if is_routing_aggregator(_norm_prov(current_provider)):
                from hermes_cli.models import _find_openrouter_slug
                if _find_openrouter_slug(raw):
                    return None
        except Exception:
            pass

    configured = {str(name).strip().lower() for name in (user_providers or {}) if str(name).strip()}
    configured.update(
        f"custom:{entry.get('name', '').strip().lower()}"
        for entry in (custom_providers or [])
        if isinstance(entry, dict) and _clean(entry.get("name")))
    try:
        from hermes_cli.models import normalize_provider
        canonical = normalize_provider(prefix)
    except Exception:
        canonical = prefix.lower()

    if prefix.lower() in configured:
        provider = prefix
    elif canonical.lower() in configured:
        provider = canonical
    else:
        return None
    return None if is_aggregator(canonical) else StartupModelRoute(model=model, provider=provider)


# --- Result dataclasses

@dataclass
class ModelSwitchResult:
    """Result of a model switch attempt."""
    success: bool
    new_model: str = ""
    target_provider: str = ""
    provider_changed: bool = False
    api_key: str = ""
    base_url: str = ""
    api_mode: str = ""
    request_overrides: Optional[dict] = None
    error_message: str = ""
    warning_message: str = ""
    provider_label: str = ""
    resolved_via_alias: str = ""
    capabilities: Optional[ModelCapabilities] = None
    runtime_capabilities: Optional[dict[str, bool]] = None
    model_info: Optional[ModelInfo] = None
    is_global: bool = False


@dataclass(frozen=True)
class ModelFlagParseResult:
    """Parsed flags for a /model command."""
    model_input: str
    explicit_provider: str = ""
    reasoning_effort: str = ""
    is_global: bool = False
    force_refresh: bool = False
    is_session: bool = False
    is_once: bool = False


# --- Flag parsing

_BOOL_FLAGS = {"--global": "is_global", "--session": "is_session", "--refresh": "force_refresh", "--once": "is_once"}
_VALUE_FLAGS = {"--provider": "explicit_provider", "--reasoning": "reasoning_effort"}


def parse_model_flags_detailed(raw_args: str) -> ModelFlagParseResult:
    """Parse /model flags: ``--provider X``, ``--reasoning <level>``, ``--global``, ``--session``,
    ``--refresh``, ``--once``.

    ``--once`` is parsed here but interpreted by each caller (each frontend has its own
    live-session restore hook). ``is_global`` / ``is_session`` are raw flag presences; the
    effective persistence decision belongs to :func:`resolve_persist_behavior`. ``reasoning_effort``
    is the raw level word (validated by :func:`hermes_constants.parse_reasoning_effort` at apply
    time) so a model pick and its effort travel as ONE request on every surface."""
    # Telegram/iOS auto-convert ``--`` to an em/en dash: normalize a single Unicode dash before
    # a flag keyword.
    raw_args = re.sub(r'[\u2012\u2013\u2014\u2015](provider|reasoning|global|session|refresh|once)', r'--\1', raw_args)

    # Hand-rolled: model IDs may contain colons/slashes and the historical parser did not
    # require shell quoting.
    flags: dict[str, bool] = dict.fromkeys(_BOOL_FLAGS.values(), False)
    values: dict[str, str] = dict.fromkeys(_VALUE_FLAGS.values(), "")
    filtered: list[str] = []
    tokens = iter(raw_args.split())
    for tok in tokens:
        if tok in _BOOL_FLAGS:
            flags[_BOOL_FLAGS[tok]] = True
        elif tok in _VALUE_FLAGS and (value := next(tokens, None)) is not None:
            values[_VALUE_FLAGS[tok]] = value
        else:
            filtered.append(tok)  # a trailing bare ``--provider`` stays part of the model text
    return ModelFlagParseResult(model_input=" ".join(filtered).strip(), **values, **flags)


def parse_model_flags(raw_args: str) -> tuple[str, str, bool, bool, bool]:
    """Legacy 5-tuple ``(model_input, explicit_provider, is_global, force_refresh, is_session)``."""
    p = parse_model_flags_detailed(raw_args)
    return (p.model_input, p.explicit_provider, p.is_global, p.force_refresh, p.is_session)


def resolve_persist_behavior(
    is_global: bool, is_session: bool, is_once: bool = False, explicit_provider: str = "") -> bool:
    """Decide whether a ``/model`` switch should persist to ``config.yaml``.

    Order: ``--once`` / ``--session`` -> False; ``--global`` -> True; no default configured yet
    (neither ``model.default`` nor ``model.provider`` — a fresh install's first pick) -> True, so
    the pick does not evaporate into whatever ``*_API_KEY`` is lying around on the next launch;
    ``--provider`` without a persist flag -> False (exploratory); else
    ``model.persist_switch_by_default`` (default False). A flat-string ``model`` IS a configured
    default; an unreadable config -> False.

    1. ``--once`` explicitly opts out → ``False`` (next turn only). 2. ``--session`` explicitly opts out →
    ``False`` (this session only). 3. 4. Applies to every surface (CLI, gateway, Desktop picker) so no
    client has to hardcode ``--global``. 5. Provider switches are typically exploratory — the user is trying
    a different backend for this conversation, not reconfiguring the default. 6. Otherwise defer to
    ``model.persist_switch_by_default`` in ``config.yaml`` (defaults to ``False``: a plain ``/model <name>``
    affects only the current session). Users who want the old persist-by-default behavior can set the key to
    ``true``; a one-off ``--global`` always persists. See #86414.
    """
    if is_once or is_session:
        return False
    if is_global:
        return True
    try:
        from hermes_cli.config import load_config
        model_cfg = load_config().get("model")
    except Exception:
        return False
    if isinstance(model_cfg, dict):
        if not (model_cfg.get("default") or model_cfg.get("provider")):
            return True
        if explicit_provider:
            return False
        return bool(model_cfg.get("persist_switch_by_default", False))
    return not model_cfg


# --- Single-owner /model request parsing + effective-model resolution. Surfaces
# (cli.py, gateway/slash_commands.py, tui_gateway/server.py, api_server.py)
# map error codes to their own copy but never re-derive the semantics.

# Error codes emitted by parse_model_switch_args().
MODEL_SWITCH_ERR_ONCE_WITH_GLOBAL = "once_with_global"
MODEL_SWITCH_ERR_ONCE_REQUIRES_TARGET = "once_requires_target"
MODEL_SWITCH_ERR_BAD_REASONING = "bad_reasoning"

# Canonical (surface-neutral) error copy. Surfaces prepend their own decoration ("  ✗ " in the
# CLI, "❌ " in the gateway) but MUST NOT change the core sentence — it is shared user-visible copy.
MODEL_SWITCH_ERROR_TEXT = {
    MODEL_SWITCH_ERR_ONCE_WITH_GLOBAL: "/model --once cannot be combined with --global",
    MODEL_SWITCH_ERR_ONCE_REQUIRES_TARGET: "/model --once requires a model or provider.",
    MODEL_SWITCH_ERR_BAD_REASONING: "/model --reasoning takes none, minimal, low, medium, high, xhigh, max or ultra."}


@dataclass(frozen=True)
class ModelSwitchRequest:
    """A fully parsed /model command request.

    ``scope`` is the *requested* persistence scope from the flags alone: ``"once"`` |
    ``"session"`` | ``"global"`` | ``"default"`` (the effective decision then belongs to
    :func:`resolve_persist_behavior`). ``errors`` carries ``MODEL_SWITCH_ERR_*`` codes rendered
    via :data:`MODEL_SWITCH_ERROR_TEXT`. ``model_input`` keeps it a drop-in for
    :class:`ModelFlagParseResult` consumers."""
    raw: str
    target: str
    explicit_provider: str = ""
    reasoning_effort: str = ""
    is_global: bool = False
    is_session: bool = False
    is_once: bool = False
    force_refresh: bool = False
    scope: str = "default"
    errors: tuple = ()

    @property
    def model_input(self) -> str:
        return self.target

    def error_messages(self) -> list:
        """Canonical (undercorated) error strings for this request."""
        return [MODEL_SWITCH_ERROR_TEXT[code] for code in self.errors]


def parse_model_switch_args(raw: str) -> ModelSwitchRequest:
    """The ONE parser for every /model surface: tokenization plus flag-conflict validation.

    ``--once`` + ``--global`` -> ``MODEL_SWITCH_ERR_ONCE_WITH_GLOBAL``; ``--once`` with neither
    a model nor ``--provider`` -> ``MODEL_SWITCH_ERR_ONCE_REQUIRES_TARGET``; an unknown
    ``--reasoning`` level -> ``MODEL_SWITCH_ERR_BAD_REASONING``. Targets pass through
    untouched (bare names, ``vendor/model``, ``vendor:model``) for :func:`switch_model`."""
    raw = str(raw or "")
    parsed = parse_model_flags_detailed(raw)

    errors: list = []
    if parsed.is_once and parsed.is_global:
        errors.append(MODEL_SWITCH_ERR_ONCE_WITH_GLOBAL)
    if parsed.is_once and not parsed.model_input and not parsed.explicit_provider:
        errors.append(MODEL_SWITCH_ERR_ONCE_REQUIRES_TARGET)
    if parsed.reasoning_effort:
        from hermes_constants import parse_reasoning_effort
        if parse_reasoning_effort(parsed.reasoning_effort) is None:
            errors.append(MODEL_SWITCH_ERR_BAD_REASONING)
    # First matching flag wins: once > session > global > default.
    scope = next((name for name, on in (("once", parsed.is_once), ("session", parsed.is_session),
                                        ("global", parsed.is_global)) if on), "default")
    return ModelSwitchRequest(
        raw=raw, target=parsed.model_input, scope=scope, errors=tuple(errors),
        **{f: getattr(parsed, f)
           for f in ("explicit_provider", "reasoning_effort", "is_global", "is_session", "is_once", "force_refresh")})


def _effective_model_candidate(value: Any) -> str:
    """Extract a model-name candidate from a str / dict / attr-object."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return _clean(value.get("model"))
    model_attr = getattr(value, "model", None)
    return _clean(model_attr) if model_attr is not None else ""


def resolve_effective_model(
    session_overrides: Any = None, channel_config: Any = None, global_config: Any = "") -> str:
    """Resolve the effective model: session override > channel > global.

    Single owner of the precedence rule gateway/run.py and api_server.py each used to encode.
    Each argument may be a model string, a dict with a ``"model"`` key, or an object with a
    ``.model`` attribute; empty entries fall through to the next tier."""
    for tier in (session_overrides, channel_config, global_config):
        candidate = _effective_model_candidate(tier)
        if candidate:
            return candidate
    return ""


# --- Alias resolution

def _model_sort_key(model_id: str, prefix: str) -> tuple:
    """Sort key preferring higher versions after the family prefix, then ranked suffix tokens.

    With prefix ``"mimo"``: ``mimo-v2.5-pro`` -> (-2.5, 0, 'pro'), ``mimo-v2.5`` -> (-2.5, 1, ''),
    ``mimo-v2-omni`` -> (-2.0, 1, 'omni')."""
    # Strip the prefix (and optional "/" separator for aggregator slugs)
    rest = model_id[len(prefix):].removeprefix("/").lstrip("-").strip()
    nums, suffix_buf = _split_version_suffix(rest)
    suffix = suffix_buf.lower().strip("-_.").strip()

    # YYYYMMDD date stamps (claude-opus-4-20250514) are snapshot markers, not version components,
    # and would dwarf real point versions; keep them as a trailing tiebreaker so bare IDs sort
    # before their dated snapshots and newer snapshots before older. The 19_000_101 threshold
    # reclassifies only 8-digit stamps (mistral-large-2411, gpt-4-0613 keep sorting as versions).
    version_key = tuple(-n for n in nums if n < 19_000_101)  # negate: higher sorts first
    date_stamp = max((n for n in nums if n >= 19_000_101), default=0.0)
    date_key = (0.0, 0.0) if date_stamp == 0.0 else (1.0, -date_stamp)

    # Suffix quality: pro/max/plus/turbo (0) > no suffix / omni / flash / mini (1). "sol" is the
    # flagship tier of the GPT-5.6 series (sol > terra > luna); without it `/model gpt` would
    # tiebreak alphabetically onto luna, the cheapest. Revisit if a vendor ships a non-flagship "-sol".
    suffix_rank = 0 if suffix in ("pro", "max", "plus", "turbo", "sol") else 1
    return version_key + (suffix_rank, suffix) + date_key


def _split_version_suffix(rest: str) -> tuple[list[float], str]:
    """``"v2.5-pro"`` -> ``([2.5], "pro")``; ``"-omni"`` -> ``([], "omni")``.

    Version tokens are ``v``-optional digit/dot runs separated by ``-``/``_``; a second dot inside
    a run starts a new component; the first character that is neither starts the suffix."""
    nums: list[float] = []
    run, pos = "", 0

    def _flush() -> None:
        nonlocal run
        try:
            nums.append(float(run.rstrip(".")))
        except ValueError:
            pass
        run = ""

    while pos < len(rest):
        ch = rest[pos]
        if ch in "-_.":
            pos += 1
            continue
        if not (ch in "vV" or ch.isdigit()):
            break
        if ch in "vV":
            pos += 1
        while pos < len(rest) and (rest[pos].isdigit() or rest[pos] == "."):
            if rest[pos] == "." and "." in run:
                _flush()
            else:
                run += rest[pos]
            pos += 1
        _flush()
        if pos < len(rest) and rest[pos] not in "-_":
            break
    return nums, rest[pos:]


class AmbiguousAliasError(Exception):
    """Alias family-matches multiple catalog models; caller must disambiguate.

    Raised by :func:`resolve_alias` instead of silently picking one via version-sort heuristics.
    ``candidates`` is sorted best-guess-first (see :func:`_model_sort_key`) for display only."""
    def __init__(self, alias: str, provider: str, candidates: list[str]):
        self.alias = alias
        self.provider = provider
        self.candidates = candidates
        super().__init__(f"alias {alias!r} matches {len(candidates)} models on {provider}")


def _ambiguous_alias_message(err: "AmbiguousAliasError") -> str:
    """User-facing disambiguation list for an ambiguous alias."""
    shown = err.candidates[:10]
    lines = "\n".join(f"  {i}. {m}" for i, m in enumerate(shown, 1))
    hidden = len(err.candidates) - len(shown)
    more = f"\n  … and {hidden} more" if hidden > 0 else ""
    return (
        f"'{err.alias}' matches {len(err.candidates)} models on "
        f"{err.provider} — not switching automatically:\n{lines}{more}\n"
        f"Pick one with /model <exact-model-name>.")


def resolve_alias(raw_input: str, current_provider: str) -> Optional[tuple[str, str, str]]:
    """Resolve a short alias against the current provider's catalog.

    Direct aliases (and reverse lookup by exact model id) win; then :data:`MODEL_ALIASES` is
    matched against the provider's models.dev catalog by ``vendor/family`` prefix (``family``
    for non-aggregators). Returns ``(provider, resolved_model_id, alias_name)`` or None; raises
    :class:`AmbiguousAliasError` when several catalog models match."""
    key = raw_input.strip().lower()

    _ensure_direct_aliases()
    direct = DIRECT_ALIASES.get(key)
    if direct is not None:
        return (direct.provider, direct.model, key)

    # Reverse lookup so full names ("kimi-k2.5") route through direct aliases instead of
    # falling through to the catalog/OpenRouter.
    for alias_name, da in DIRECT_ALIASES.items():
        if da.model.lower() == key:
            return (da.provider, da.model, alias_name)

    identity = MODEL_ALIASES.get(key)
    if identity is None:
        return None

    vendor, family = identity

    # models.dev catalog merged with static _PROVIDER_MODELS entries it may be missing.
    catalog = list_provider_models(current_provider)
    try:
        from hermes_cli.models import _PROVIDER_MODELS
        seen = {m.lower() for m in catalog}
        catalog.extend(m for m in _PROVIDER_MODELS.get(current_provider, []) if m.lower() not in seen)
    except Exception:
        pass

    prefix = f"{vendor}/{family}" if is_aggregator(current_provider) else family
    matches = [mid for mid in catalog if mid.lower().startswith(prefix.lower())]
    if not matches:
        return None

    # Version-sort for display, but NEVER silently pick among multiple candidates: the
    # heuristics have repeatedly guessed wrong (dated snapshots outranking point releases,
    # suffix tiebreaks landing on the cheapest tier).
    matches.sort(key=lambda m: _model_sort_key(m, prefix))
    if len(matches) > 1:
        raise AmbiguousAliasError(key, current_provider, matches)
    return (current_provider, matches[0], key)


def get_authenticated_provider_slugs(
    current_provider: str = "", user_providers: dict = None, custom_providers: list | None = None
) -> list[str]:
    """Slugs of providers that have credentials (models.dev in-memory cache; no extra network cost)."""
    try:
        return [p["slug"] for p in list_authenticated_providers(
            current_provider=current_provider, user_providers=user_providers,
            custom_providers=custom_providers, max_models=0)]
    except Exception:
        return []


def _resolve_alias_fallback(
    raw_input: str, authenticated_providers: list[str] = ()) -> Optional[tuple[str, str, str]]:
    """Resolve an alias on the user's authenticated providers (``("openrouter", "nous")`` when none given).

    AmbiguousAliasError propagates: the alias exists on this provider, the user just has to
    choose — trying the next provider would silently switch them somewhere they didn't ask for."""
    results = (resolve_alias(raw_input, p) for p in authenticated_providers or ("openrouter", "nous"))
    return next((r for r in results if r is not None), None)


def resolve_display_context_length(
    model: str, provider: str, base_url: str = "", api_key: str = "",
    model_info: Optional[ModelInfo] = None, custom_providers: list | None = None,
    config_context_length: int | None = None, configured_model: str | None = None,
    configured_provider: str | None = None,
    configured_base_url: str | None = None) -> Optional[int]:
    """Context length to show in /model output.

    models.dev reports per-vendor context but provider-enforced limits can be lower (Codex OAuth
    caps gpt-5.5 at 272k), so ``agent.model_metadata.get_model_context_length`` is authoritative
    (it also honors ``custom_providers[].models.<id>.context_length``); ``model_info.context_window``
    is the fallback. A ``config_context_length`` pin is dropped when the route changed.

    When ``custom_providers`` is provided, per-model ``context_length`` overrides from
    ``custom_providers[].models.<id>.context_length`` are honored — this closes #15779 where ``/model``
    switch ignored user-set overrides.
    """
    if config_context_length is not None and (configured_model or configured_provider or configured_base_url):
        try:
            from hermes_cli.route_identity import should_clear_context_pin
            if should_clear_context_pin(
                    configured_model, model, configured_base_url, base_url, configured_provider, provider):
                config_context_length = None
        except Exception:
            config_context_length = None

    try:
        from agent.model_metadata import get_model_context_length
        ctx = get_model_context_length(
            model, base_url=base_url or "", api_key=api_key or "", provider=provider or None,
            custom_providers=custom_providers, config_context_length=config_context_length)
        if ctx:
            return int(ctx)
    except Exception:
        pass
    if model_info is not None and model_info.context_window:
        return int(model_info.context_window)
    return None


async def resolve_display_context_length_async(model: str, provider: str, **kwargs) -> Optional[int]:
    """Thread-offloaded :func:`resolve_display_context_length` (same keyword arguments) — the sync
    version runs blocking provider probes that async gateway handlers must not run on the loop."""
    import asyncio
    return await asyncio.to_thread(resolve_display_context_length, model, provider, **kwargs)


# --- Configured-provider detection for typed model names

def _configured_provider_matches(
    model_name: str, user_providers: Optional[dict], custom_providers: Optional[list]
) -> dict[str, str]:
    """``{provider_slug: canonical_model_id}`` for every configured provider whose declared models
    (``models``, ``model``, ``default_model`` — exact, case-insensitive, never fuzzy) contain
    ``model_name``, so a typed name routes to the provider that declares it instead of being
    soft-accepted by the current provider (openai-codex) as an unknown hidden model.

    Used by :func:`switch_model` to route a *typed* model name to the provider that actually declares it in
    user/custom provider config, instead of leaving it on the current provider. See #45006.
    """
    if not model_name or not model_name.strip():
        return {}
    target = model_name.strip().lower()

    candidates: list[tuple[str, dict]] = []
    if isinstance(user_providers, dict):
        candidates += [(slug, cfg) for slug, cfg in user_providers.items()
                       if isinstance(slug, str) and isinstance(cfg, dict)]
    candidates += [(f"custom:{e['name']}", e) for e in _custom_entries(custom_providers)
                   if isinstance(e.get("name"), str) and e["name"].strip()]

    matches: dict[str, str] = {}
    for slug, cfg in candidates:
        hit = next((mid for key in ("models", "model", "default_model")
                    for mid in _declared_model_ids(cfg.get(key)) if mid.lower() == target), None)
        if hit:
            matches.setdefault(slug, hit)  # first declaration wins
    return matches


def _resolve_named_custom_model_id(model_name: str, target_provider: str, custom_providers: Optional[list]) -> str:
    """Map a picker-prefixed custom model selection (``prefix/model``) to its configured ID."""
    provider = _clean(target_provider).lower()
    if not provider.startswith("custom:") or "/" not in model_name:
        return model_name

    prefix, candidate = (part.strip() for part in model_name.split("/", 1))
    if not prefix or not candidate:
        return model_name
    for entry in _custom_entries(custom_providers):
        entry_slugs = _entry_aliases(entry)
        if provider in entry_slugs and f"custom:{prefix.lower()}" in entry_slugs:
            for model_id in _declared_model_ids(entry.get("models")):
                if model_id.lower() == candidate.lower():
                    return model_id
    return model_name


# ---------------------------------------------------------------------------
# Core model-switching pipeline
# ---------------------------------------------------------------------------

def switch_model(
    raw_input: str,
    current_provider: str,
    current_model: str,
    current_base_url: str = "",
    current_api_key: str = "",
    is_global: bool = False,
    explicit_provider: str = "",
    user_providers: dict = None,
    custom_providers: list | None = None,
) -> ModelSwitchResult:
    """Core model-switching pipeline shared between CLI and gateway.

    Resolution chain:

      If --provider given:
        a. Resolve provider via resolve_provider_full()
        b. Resolve credentials
        c. If model given, resolve alias on target provider or use as-is
        d. If no model, auto-detect from endpoint

      If no --provider:
        a. Try alias resolution on current provider
        b. If alias exists but not on current provider -> fallback
        c. On aggregator, try vendor/model slug conversion
        d. Aggregator catalog search
        e. detect_provider_for_model() as last resort
        f. Resolve credentials
        g. Normalize model name for target provider

      Finally:
        h. Get full model metadata from models.dev
        i. Build result

    Args:
        raw_input: The model name (after flag parsing).
        current_provider: The currently active provider.
        current_model: The currently active model name.
        current_base_url: The currently active base URL.
        current_api_key: The currently active API key.
        is_global: Whether to persist the switch.
        explicit_provider: From --provider flag (empty = no explicit provider).
        user_providers: The ``providers:`` dict from config.yaml (for user endpoints).
        custom_providers: The ``custom_providers:`` list from config.yaml.

    Returns:
        ModelSwitchResult with all information the caller needs.
    """
    from hermes_cli.models import (
        copilot_model_api_mode,
        detect_provider_for_model,
        validate_requested_model,
        opencode_model_api_mode,
        _get_ollama_request_headers,
        _get_provider_config_dict,
        _same_ollama_native_root,
    )
    from hermes_cli.runtime_provider import resolve_runtime_provider

    resolved_alias = ""
    request_overrides: dict = {}
    new_model = raw_input.strip()
    target_provider = current_provider
    resolved_moa_preset = False

    # =================================================================
    # PATH A: Explicit --provider given
    # =================================================================
    if explicit_provider:
        # Resolve the provider
        pdef = resolve_provider_full(
            explicit_provider,
            user_providers,
            custom_providers,
        )
        if pdef is None and explicit_provider.strip().lower() == "custom":
            pdef = _bare_custom_provider_def(current_base_url)
        if pdef is None:
            _switch_err = (
                f"Unknown provider '{explicit_provider}'. "
                f"Check 'hermes model' for available providers, or define it "
                f"in config.yaml under 'providers:'."
            )
            # Check for common config issues that cause provider resolution failures
            try:
                from hermes_cli.config import validate_config_structure
                _cfg_issues = validate_config_structure()
                if _cfg_issues:
                    _switch_err += "\n\nRun 'hermes doctor' — config issues detected:"
                    for _ci in _cfg_issues[:3]:
                        _switch_err += f"\n  • {_ci.message}"
            except Exception:
                pass
            return ModelSwitchResult(
                success=False,
                is_global=is_global,
                error_message=_switch_err,
            )

        target_provider = pdef.id
        if target_provider == "moa" and not new_model:
            try:
                from hermes_cli.config import load_config
                from hermes_cli.moa_config import normalize_moa_config

                new_model = normalize_moa_config(load_config().get("moa") or {})["default_preset"]
            except Exception:
                new_model = "default"

        # Guard against silent aggregator hops. A vendor name like bare
        # "openai" is an alias that resolves to an aggregator ("openrouter").
        # If the user explicitly asked for that vendor but the aggregator it
        # routes to has no credentials, do NOT silently switch them onto an
        # unauthed endpoint (the classic HTTP 401 "Missing Authentication
        # header"). Point them at the real direct provider instead.
        from hermes_cli.models import _AGGREGATOR_PROVIDERS as _AGG_PROVIDERS
        from hermes_cli.providers import ALIASES as _PROVIDER_ALIAS_TABLE
        _explicit_norm = explicit_provider.strip().lower()
        _alias_target = _PROVIDER_ALIAS_TABLE.get(_explicit_norm)
        if (
            _alias_target
            and _alias_target == target_provider
            and target_provider != _explicit_norm
            and target_provider in _AGG_PROVIDERS
        ):
            _authed = get_authenticated_provider_slugs(
                current_provider=current_provider,
                user_providers=user_providers,
                custom_providers=custom_providers,
            )
            if target_provider not in _authed:
                _suggestions = [
                    s for s in _authed
                    if s.startswith(_explicit_norm) and s != _explicit_norm
                ]
                _hint = (
                    f" Did you mean: {', '.join(_suggestions)}?"
                    if _suggestions else ""
                )
                return ModelSwitchResult(
                    success=False,
                    target_provider=target_provider,
                    provider_label=pdef.name,
                    is_global=is_global,
                    error_message=(
                        f"Provider '{_explicit_norm}' is an alias that routes "
                        f"through {get_label(target_provider)}, which "
                        f"has no credentials configured.{_hint}"
                    ),
                )

        # If no model specified, try auto-detect from endpoint
        if not new_model:
            if pdef.base_url:
                from hermes_cli.runtime_provider import _auto_detect_local_model
                detected = _auto_detect_local_model(pdef.base_url)
                if detected:
                    new_model = detected
                else:
                    return ModelSwitchResult(
                        success=False,
                        target_provider=target_provider,
                        provider_label=pdef.name,
                        is_global=is_global,
                        error_message=(
                            f"No model detected on {pdef.name} ({pdef.base_url}). "
                            f"Specify the model explicitly: /model <model-name> --provider {explicit_provider}"
                        ),
                    )
            else:
                return ModelSwitchResult(
                    success=False,
                    target_provider=target_provider,
                    provider_label=pdef.name,
                    is_global=is_global,
                    error_message=(
                        f"Provider '{pdef.name}' has no base URL configured. "
                        f"Specify a model: /model <model-name> --provider {explicit_provider}"
                    ),
                )

        # Resolve alias on the TARGET provider
        try:
            alias_result = resolve_alias(new_model, target_provider)
        except AmbiguousAliasError as err:
            return ModelSwitchResult(
                success=False,
                target_provider=target_provider,
                is_global=is_global,
                error_message=_ambiguous_alias_message(err),
            )
        if alias_result is not None:
            _, new_model, resolved_alias = alias_result

    # =================================================================
    # PATH B: No explicit provider — resolve from model input
    # =================================================================
    else:
        try:
            from hermes_cli.config import load_config
            from hermes_cli.moa_config import exact_moa_preset_name, normalize_moa_config

            _moa_cfg = normalize_moa_config(load_config().get("moa") or {})
            _moa_match = exact_moa_preset_name(_moa_cfg, raw_input)
            if _moa_match:
                target_provider = "moa"
                new_model = _moa_match
                resolved_alias = ""
                resolved_moa_preset = True
                alias_result = None
            else:
                alias_result = resolve_alias(raw_input, current_provider)
        except AmbiguousAliasError as err:
            return ModelSwitchResult(
                success=False,
                is_global=is_global,
                error_message=_ambiguous_alias_message(err),
            )
        except Exception:
            try:
                alias_result = resolve_alias(raw_input, current_provider)
            except AmbiguousAliasError as err:
                return ModelSwitchResult(
                    success=False,
                    is_global=is_global,
                    error_message=_ambiguous_alias_message(err),
                )

        # --- Step a: Try alias resolution on current provider ---

        if resolved_moa_preset:
            pass
        elif alias_result is not None:
            target_provider, new_model, resolved_alias = alias_result
            logger.debug(
                "Alias '%s' resolved to %s on %s",
                resolved_alias, new_model, target_provider,
            )
        else:
            # --- Step b: Alias exists but not on current provider -> fallback ---
            key = raw_input.strip().lower()
            if key in MODEL_ALIASES:
                authed = get_authenticated_provider_slugs(
                    current_provider=current_provider,
                    user_providers=user_providers,
                    custom_providers=custom_providers,
                )
                try:
                    fallback_result = _resolve_alias_fallback(raw_input, authed)
                except AmbiguousAliasError as err:
                    return ModelSwitchResult(
                        success=False,
                        is_global=is_global,
                        error_message=_ambiguous_alias_message(err),
                    )
                if fallback_result is not None:
                    target_provider, new_model, resolved_alias = fallback_result
                    logger.debug(
                        "Alias '%s' resolved via fallback to %s on %s",
                        resolved_alias, new_model, target_provider,
                    )
                else:
                    identity = MODEL_ALIASES[key]
                    return ModelSwitchResult(
                        success=False,
                        is_global=is_global,
                        error_message=(
                            f"Alias '{key}' maps to {identity.vendor}/{identity.family} "
                            f"but no matching model was found in any provider catalog. "
                            f"Try specifying the full model name."
                        ),
                    )
            elif not resolved_moa_preset:
                # --- Step c: On aggregator, convert vendor:model to vendor/model ---
                # Only convert when there's no slash — a slash means the name
                # is already in vendor/model format and the colon is a variant
                # tag (:free, :extended, :fast) that must be preserved.
                colon_pos = raw_input.find(":")
                if (
                    colon_pos > 0
                    and "/" not in raw_input
                    and is_aggregator(current_provider)
                    and not str(current_provider).strip().lower().startswith("custom")
                    and str(current_provider).strip().lower() != "ollama"
                ):
                    left = raw_input[:colon_pos].strip().lower()
                    right = raw_input[colon_pos + 1:].strip()
                    if left and right:
                        # Colons become slashes for aggregator slugs
                        new_model = f"{left}/{right}"
                        logger.debug(
                            "Converted vendor:model '%s' to aggregator slug '%s'",
                            raw_input, new_model,
                        )

        # --- Step d: Aggregator catalog search ---
        # Track whether the live catalog of the CURRENT provider resolved the
        # model — if so, step e must not second-guess and switch providers.
        # Critical for flat-namespace resellers like opencode-go / opencode-zen
        # whose live /v1/models returns bare IDs (e.g. "deepseek-v4-flash") that
        # coincidentally match entries in native providers' static catalogs.
        resolved_in_current_catalog = False
        if is_aggregator(target_provider) and not resolved_alias:
            catalog = list_provider_models(target_provider)
            if catalog:
                new_model_lower = new_model.lower()
                for mid in catalog:
                    if mid.lower() == new_model_lower:
                        new_model = mid
                        resolved_in_current_catalog = True
                        break
                else:
                    for mid in catalog:
                        if "/" in mid:
                            _, bare = mid.split("/", 1)
                            if bare.lower() == new_model_lower:
                                new_model = mid
                                resolved_in_current_catalog = True
                                break

        # --- Step d.5: configured-provider exact-match detection (#45006) ---
        # If the typed model is declared in user/custom provider config, route
        # to that provider BEFORE detect_provider_for_model() guesses from
        # static catalogs and BEFORE the common-path validation can let a
        # soft-accepting current provider (e.g. openai-codex) swallow the name
        # as an unknown hidden model.  Configured matches beat static-catalog
        # detection.  Unlike step e this is deliberately NOT gated on
        # ``not is_custom`` — switching from a local/custom provider A to a
        # configured provider B that declares the typed model is the point.
        config_routed = False
        if (
            not resolved_alias
            and not resolved_in_current_catalog
            and target_provider == current_provider
        ):
            cfg_matches = _configured_provider_matches(
                new_model, user_providers, custom_providers
            )
            if cfg_matches:
                if current_provider in cfg_matches:
                    # The current provider itself declares it — keep current.
                    new_model = cfg_matches[current_provider]
                    config_routed = True
                else:
                    match_slugs = sorted(cfg_matches)
                    if len(match_slugs) > 1:
                        return ModelSwitchResult(
                            success=False,
                            is_global=is_global,
                            error_message=(
                                f"'{new_model}' is declared by multiple configured "
                                f"providers ({', '.join(match_slugs)}). Re-run with "
                                f"--provider <slug> to choose which one to use."
                            ),
                        )
                    target_provider = match_slugs[0]
                    new_model = cfg_matches[target_provider]
                    config_routed = True
                    logger.debug(
                        "Configured-provider detection routed '%s' to %s",
                        new_model, target_provider,
                    )
                    # User-config providers (providers.<slug>) are resolved in
                    # the credential block via resolve_user_provider(), which is
                    # gated on explicit_provider.  Mirror the picker so the
                    # rerouted user provider's base_url/key load from the passed
                    # config rather than a from-scratch runtime re-resolve that
                    # doesn't know user-config slugs.  custom:* slugs resolve via
                    # resolve_runtime_provider() directly and need no hint.
                    if isinstance(user_providers, dict) and target_provider in user_providers:
                        explicit_provider = target_provider

        # --- Step e: detect_provider_for_model() as last resort ---
        _base = current_base_url or ""
        is_custom = (
            current_provider in {"custom", "local"}
            or current_provider.startswith("custom:")
            or base_url_hostname(_base) in ("localhost", "127.0.0.1")
        )

        if (
            target_provider == current_provider
            and not is_custom
            and not resolved_alias
            and not resolved_in_current_catalog
            and not config_routed
        ):
            detected = detect_provider_for_model(new_model, current_provider)
            if detected:
                target_provider, new_model = detected

    # =================================================================
    # COMMON PATH: Resolve credentials, normalize, get metadata
    # =================================================================

    provider_changed = target_provider != current_provider
    provider_label = get_label(target_provider)
    if target_provider == "custom" and current_base_url:
        provider_label = "Custom endpoint"
    if target_provider.startswith("custom:"):
        custom_pdef = resolve_provider_full(
            target_provider,
            user_providers,
            custom_providers,
        )
        if custom_pdef is not None:
            provider_label = custom_pdef.name

    # --- Resolve credentials ---
    api_key = current_api_key
    base_url = current_base_url
    api_mode = ""
    runtime_capabilities: dict[str, bool] = {}
    ollama_headers: dict[str, str] = {}
    validation_headers: dict[str, str] = {}
    suppress_ollama_headers = False

    if provider_changed or explicit_provider:
        # User-config providers (providers.<name> in config.yaml) carry their
        # own base_url + transport + key reference. resolve_runtime_provider()
        # resolves by provider NAME and doesn't know user-config slugs (e.g. a
        # block named "openai"), so it would re-resolve from scratch and fail
        # or hop to an aggregator. Use the pdef's endpoint directly instead.
        _user_pdef = None
        if explicit_provider and user_providers:
            from hermes_cli.providers import resolve_user_provider as _ruser
            _user_pdef = _ruser(explicit_provider.strip().lower(), user_providers)
            if _user_pdef is None:
                _user_pdef = _ruser(target_provider, user_providers)
        if _user_pdef is not None and _user_pdef.base_url:
            _ucfg = (user_providers or {}).get(explicit_provider.strip().lower()) \
                or (user_providers or {}).get(target_provider) or {}
            _ukey = str(_ucfg.get("api_key", "") or "").strip()
            if _ukey.startswith("${") and _ukey.endswith("}"):
                # Same class as the picker reads below: a raw os.environ read
                # here hands this profile whatever key the process env holds —
                # another profile's, under the multiplexed gateway. Route
                # through the per-profile secret scope (identical to
                # os.getenv when multiplexing is off, fail-closed otherwise).
                _ukey = _scoped_key_env(_ukey[2:-1])
            if not _ukey:
                _kenv = str(
                    _ucfg.get("key_env") or _ucfg.get("api_key_env") or ""
                ).strip()
                if _kenv:
                    _ukey = _scoped_key_env(_kenv)
            validation_headers = _extra_headers_from_config(_ucfg)
            try:
                runtime = resolve_runtime_provider(
                    requested=target_provider,
                    explicit_api_key=_ukey or None,
                    explicit_base_url=_user_pdef.base_url,
                    target_model=new_model,
                )
                api_key = runtime.get("api_key", "") or _ukey
                base_url = runtime.get("base_url", "") or _user_pdef.base_url
                api_mode = runtime.get("api_mode", "")
                runtime_capabilities = runtime.get("capabilities") or {}
                validation_headers = runtime.get("extra_headers") or validation_headers
            except Exception:
                api_key = _ukey
                base_url = _user_pdef.base_url
                api_mode = ""
        elif target_provider == "custom" and current_base_url:
            api_key = current_api_key
            base_url = current_base_url
            api_mode = determine_api_mode(target_provider, base_url)
        else:
            try:
                runtime = resolve_runtime_provider(
                    requested=target_provider,
                    target_model=new_model,
                )
                api_key = runtime.get("api_key", "")
                base_url = runtime.get("base_url", "")
                api_mode = runtime.get("api_mode", "")
                runtime_capabilities = runtime.get("capabilities") or {}
                validation_headers = runtime.get("extra_headers") or validation_headers
            except Exception as e:
                return ModelSwitchResult(
                    success=False,
                    target_provider=target_provider,
                    provider_label=provider_label,
                    is_global=is_global,
                    error_message=(
                        f"Could not resolve credentials for provider "
                        f"'{provider_label}': {e}"
                    ),
                )
    else:
        keep_current_ollama_endpoint = False
        if current_provider == "custom" and current_base_url:
            try:
                from hermes_cli.models import should_use_ollama_native_catalog
                ollama_headers = _get_ollama_request_headers()
                ollama_config = _get_provider_config_dict("ollama")
                configured_ollama_base = str(
                    ollama_config.get("base_url")
                    or ollama_config.get("api")
                    or ollama_config.get("url")
                    or ""
                ).strip()
                if configured_ollama_base and not _same_ollama_native_root(
                    current_base_url, configured_ollama_base
                ):
                    ollama_headers = {}
                    suppress_ollama_headers = True
                elif not configured_ollama_base:
                    # Without an explicit configured root there is no safe
                    # origin to associate provider-level Ollama headers with.
                    ollama_headers = {}
                    suppress_ollama_headers = True
                keep_current_ollama_endpoint = should_use_ollama_native_catalog(
                    current_provider,
                    current_base_url,
                    headers=ollama_headers,
                )
            except (ImportError, OSError, RuntimeError, TypeError, ValueError):
                keep_current_ollama_endpoint = False
        if keep_current_ollama_endpoint:
            # Mid-session `/model <name>` on a local Ollama-compatible endpoint
            # must keep the endpoint the session is already using. Re-resolving
            # bare `custom` from config can fall through to an unrelated default
            # provider, causing validation to probe the wrong model-list URL.
            api_key = current_api_key or "no-key-required"
            base_url = current_base_url
            api_mode = determine_api_mode(current_provider, base_url)
            validation_headers = ollama_headers
        else:
            try:
                runtime = resolve_runtime_provider(
                    requested=current_provider,
                    target_model=new_model,
                )
                api_key = runtime.get("api_key", "")
                base_url = runtime.get("base_url", "")
                api_mode = runtime.get("api_mode", "")
                runtime_capabilities = runtime.get("capabilities") or {}
                validation_headers = runtime.get("extra_headers") or validation_headers
            except Exception:
                pass

    # --- Direct alias override: use exact base_url from the alias if set ---
    if resolved_alias:
        _ensure_direct_aliases()
        _da = DIRECT_ALIASES.get(resolved_alias)
        if _da is not None and _da.base_url:
            base_url = _da.base_url
            api_mode = ""  # clear so determine_api_mode re-detects from URL
            if target_provider.strip().lower() == "ollama":
                _ollama_cfg = _get_provider_config_dict("ollama")
                _ollama_cfg_base = str(
                    _ollama_cfg.get("base_url")
                    or _ollama_cfg.get("api")
                    or _ollama_cfg.get("url")
                    or ""
                ).strip()
                if _ollama_cfg_base and _same_ollama_native_root(
                    base_url, _ollama_cfg_base
                ):
                    configured_key = str(_ollama_cfg.get("api_key") or "").strip()
                    if configured_key.startswith("${") and configured_key.endswith("}"):
                        configured_key = os.environ.get(configured_key[2:-1], "").strip()
                    if not configured_key:
                        key_env = str(
                            _ollama_cfg.get("key_env")
                            or _ollama_cfg.get("api_key_env")
                            or ""
                        ).strip()
                        if key_env:
                            configured_key = os.environ.get(key_env, "").strip()
                    if configured_key:
                        api_key = configured_key
                if _ollama_cfg_base and not _same_ollama_native_root(
                    base_url, _ollama_cfg_base
                ):
                    # Do not carry providers.ollama credentials to an alias
                    # endpoint with a different origin.
                    validation_headers = {}
                    suppress_ollama_headers = True
                    api_key = "no-key-required"
                elif not _ollama_cfg_base:
                    # Without an explicit configured root there is no safe
                    # origin to associate the provider-level headers with.
                    validation_headers = {}
                    suppress_ollama_headers = True
                    api_key = "no-key-required"
            if not api_key:
                api_key = "no-key-required"

    # --- Resolve api_mode from the final (provider, base_url) before validation ---
    # Two cases this closes, both surfaced when the switched model's reasoning
    # is actually applied (post the reasoning-unification refactor):
    #   1. api_mode empty (e.g. alias cleared it above) → fill from the endpoint.
    #   2. api_mode carried a STALE value from the previous session state
    #      (e.g. a same-provider /model switch to gpt-5.x on api.openai.com that
    #      kept the prior openrouter/chat_completions mode). A host that mandates
    #      one wire protocol must override the stale value — otherwise the request
    #      goes out on chat_completions and OpenAI 400s on tools+reasoning_effort.
    _mandated_mode = host_mandated_api_mode(base_url)
    if _mandated_mode is not None:
        api_mode = _mandated_mode
    elif not api_mode:
        api_mode = determine_api_mode(target_provider, base_url)

    # --- Normalize model name for target provider ---
    new_model = _resolve_named_custom_model_id(
        new_model, target_provider, custom_providers
    )
    new_model = normalize_model_for_provider(new_model, target_provider)

    # --- Validate ---
    try:
        validation = validate_requested_model(
            new_model,
            target_provider,
            api_key=api_key,
            base_url=base_url,
            api_mode=api_mode or None,
            headers=(
                (
                    {}
                    if suppress_ollama_headers
                    else (validation_headers or _get_ollama_request_headers())
                )
                if target_provider.strip().lower() == "ollama"
                else (
                    validation_headers
                    or (
                        _extra_headers_from_config(user_providers.get(target_provider))
                        if user_providers and target_provider in user_providers
                        else None
                    )
                )
            ),
        )
    except Exception as e:
        validation = {
            "accepted": False,
            "persist": False,
            "recognized": False,
            "message": f"Could not validate `{new_model}`: {e}",
        }

    # Override rejection if model is in the user's saved provider config.
    # API /v1/models may not list cloud/aliased models even though the server supports them.
    if not validation.get("accepted"):
        override = False
        if user_providers:
            from hermes_cli.config import is_provider_enabled
            # user_providers is a dict: {provider_slug: config_dict}
            for slug, cfg in user_providers.items():
                if not is_provider_enabled(cfg):
                    continue
                if slug == target_provider:
                    if new_model in _declared_model_ids(cfg.get("models", {})):
                        override = True
                        break
        # Also check custom_providers list — models declared there should be accepted
        # even if the remote /v1/models endpoint doesn't list them.
        if not override and custom_providers and isinstance(custom_providers, list):
            for entry in custom_providers:
                if not isinstance(entry, dict):
                    continue
                # Match by provider slug (custom:<name>) or by base_url
                entry_name = entry.get("name", "")
                entry_aliases = custom_provider_aliases(
                    str(entry_name or ""),
                    str(entry.get("provider_key") or ""),
                )
                entry_url = entry.get("base_url", "")
                if target_provider.lower() in entry_aliases or entry_url == base_url:
                    # Check if the requested model matches the entry's model
                    entry_model = entry.get("model", "")
                    entry_models = entry.get("models", {})
                    if new_model == entry_model:
                        override = True
                        break
                    if new_model in _declared_model_ids(entry_models):
                        override = True
                        break
        if override:
            validation = {"accepted": True, "persist": True, "recognized": False, "message": validation.get("message", "")}
        else:
            msg = validation.get("message", "Invalid model")
            return ModelSwitchResult(
                success=False,
                new_model=new_model,
                target_provider=target_provider,
                provider_label=provider_label,
                is_global=is_global,
                error_message=msg,
            )

    # Apply auto-correction if validation found a closer match
    if validation.get("corrected_model"):
        new_model = validation["corrected_model"]

    # --- Copilot api_mode override ---
    if target_provider in {"copilot", "github-copilot"}:
        api_mode = copilot_model_api_mode(new_model, api_key=api_key)

    # --- OpenCode api_mode override ---
    if target_provider in {"opencode-zen", "opencode-go", "opencode"}:
        api_mode = opencode_model_api_mode(target_provider, new_model)

    # --- Nous Portal dual-wire override ---
    # Portal serves anthropic/* on /v1/messages and everything else on
    # /chat/completions. resolve_runtime_provider already sets this when it
    # succeeds; always re-derive from the *final* (post-normalize) model so
    # alias clears / empty fallbacks cannot leave Claude on the OpenAI wire.
    if target_provider in {"nous", "nous-portal", "nousresearch"}:
        from hermes_cli.providers import nous_api_mode

        api_mode = nous_api_mode(new_model)

    # --- Determine api_mode if not already set ---
    if not api_mode:
        api_mode = determine_api_mode(
            target_provider, base_url, model=new_model
        )

    # OpenCode base URLs end with /v1 for OpenAI-compatible models, but the
    # Anthropic SDK prepends its own /v1/messages to the base_url.  Normalize
    # symmetrically (strip /v1 for anthropic_messages, re-append it for
    # chat_completions / codex_responses).  Mirrors the same logic in
    # hermes_cli.runtime_provider.resolve_runtime_provider; without the strip,
    # /model switches into an anthropic_messages-routed OpenCode model
    # (e.g. `/model minimax-m2.7` on opencode-go, `/model claude-sonnet-4-6`
    # on opencode-zen) hit a double /v1 and returned OpenCode's website 404
    # page — and without the re-append, a stripped URL persisted to
    # model.base_url broke every later chat_completions model (glm, deepseek,
    # kimi) the same way.
    from hermes_cli.models import opencode_provider_family as _oc_family_fn
    if _oc_family_fn(target_provider) is not None and isinstance(base_url, str):
        from hermes_cli.models import normalize_opencode_base_url
        base_url = normalize_opencode_base_url(target_provider, api_mode, base_url)

    # --- Get capabilities (legacy) ---
    capabilities = get_model_capabilities(target_provider, new_model, allow_network=True)
    from agent.native_compaction import resolve_native_compaction_capabilities
    runtime_capabilities = resolve_native_compaction_capabilities(
        model=new_model,
        base_url=base_url,
        provider=target_provider,
        is_codex_backend=target_provider.strip().lower() == "openai-codex",
    )

    # --- Get full model info from models.dev ---
    model_info = get_model_info(target_provider, new_model, allow_network=True)

    # --- Collect warnings ---
    warnings: list[str] = []
    if validation.get("message"):
        warnings.append(validation["message"])
    hermes_warn = _check_hermes_model_warning(new_model)
    if hermes_warn:
        warnings.append(hermes_warn)

    # Carry the switched provider's request_overrides (e.g. a custom_providers
    # ``extra_body`` such as chat_template_kwargs) so a ``/model`` switch to a
    # custom provider applies it on the gateway, matching the default-provider
    # path. resolve_runtime_provider surfaces these for named custom providers.
    request_overrides = None
    try:
        from hermes_cli.runtime_provider import (
            _get_named_custom_provider,
            _custom_provider_request_overrides,
        )
        _cp_for_ro = _get_named_custom_provider(target_provider)
        if _cp_for_ro:
            request_overrides = _custom_provider_request_overrides(_cp_for_ro) or None
    except Exception:
        request_overrides = None

    # --- Build result ---
    return ModelSwitchResult(
        success=True,
        new_model=new_model,
        target_provider=target_provider,
        provider_changed=provider_changed,
        api_key=api_key,
        base_url=base_url,
        api_mode=api_mode,
        request_overrides=dict(request_overrides or {}),
        warning_message=" | ".join(warnings) if warnings else "",
        provider_label=provider_label,
        resolved_via_alias=resolved_alias,
        capabilities=capabilities,
        runtime_capabilities={
            key: value
            for key, value in runtime_capabilities.items()
            if isinstance(key, str) and isinstance(value, bool)
        },
        model_info=model_info,
        is_global=is_global,
    )


def _entry_aliases(entry: dict) -> frozenset[str]:
    return custom_provider_aliases(str(entry.get("name") or ""), str(entry.get("provider_key") or ""))


# --- Core model-switching pipeline

def _entry_configured_key(cfg: dict, read_env) -> str:
    """Inline ``api_key`` (a ``${VAR}`` template resolves via *read_env*), else
    ``key_env``/``api_key_env`` via *read_env*."""
    key = _clean(cfg.get("api_key", ""))
    if key.startswith("${") and key.endswith("}"):
        key = read_env(key[2:-1])
    if not key:
        key_env = _clean(cfg.get("key_env") or cfg.get("api_key_env"))
        key = read_env(key_env) if key_env else ""
    return key


def _ollama_configured_base() -> tuple[dict, str]:
    from hermes_cli.models import _get_provider_config_dict
    cfg = _get_provider_config_dict("ollama")
    return cfg, _clean(cfg.get("base_url") or cfg.get("api") or cfg.get("url"))


def _unknown_provider_message(explicit_provider: str) -> str:
    msg = (
        f"Unknown provider '{explicit_provider}'. Check 'hermes model' for available "
        f"providers, or define it in config.yaml under 'providers:'.")
    try:  # Surface common config issues that cause provider resolution failures
        from hermes_cli.config import validate_config_structure
        issues = validate_config_structure()
        if issues:
            msg += "\n\nRun 'hermes doctor' — config issues detected:" + "".join(f"\n  • {ci.message}" for ci in issues[:3])
    except Exception:
        pass
    return msg


def _aggregator_alias_error(
    explicit_provider: str, target_provider: str, current_provider: str, user_providers, custom_providers,
) -> str:
    """Guard against silent aggregator hops: a vendor alias like bare "openai" resolves to an
    aggregator ("openrouter"); if that aggregator has no credentials, refuse instead of switching
    the user onto an unauthed endpoint (HTTP 401) and point at the real direct provider."""
    from hermes_cli.models import _AGGREGATOR_PROVIDERS
    from hermes_cli.providers import ALIASES
    explicit_norm = explicit_provider.strip().lower()
    alias_target = ALIASES.get(explicit_norm)
    if not (
        alias_target and alias_target == target_provider and target_provider != explicit_norm
        and target_provider in _AGGREGATOR_PROVIDERS):
        return ""
    authed = get_authenticated_provider_slugs(
        current_provider=current_provider, user_providers=user_providers, custom_providers=custom_providers)
    if target_provider in authed:
        return ""
    suggestions = [s for s in authed if s.startswith(explicit_norm) and s != explicit_norm]
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    return (
        f"Provider '{explicit_norm}' is an alias that routes "
        f"through {get_label(target_provider)}, which "
        f"has no credentials configured.{hint}")


def _aggregator_catalog_match(new_model: str, catalog: list) -> str | None:
    """Exact (case-insensitive) match on full id, then on the bare part after ``vendor/``."""
    wanted = new_model.lower()
    return next((mid for mid in catalog if mid.lower() == wanted), None) or next(
        (mid for mid in catalog if "/" in mid and mid.split("/", 1)[1].lower() == wanted), None)


def _config_declares_model(
    new_model: str, target_provider: str, base_url: str, user_providers, custom_providers) -> bool:
    """A model declared in the user's ``providers:``/``custom_providers:`` config is accepted even
    when the remote /v1/models does not list it (cloud/aliased models). Custom entries match by
    slug alias or by base_url."""
    if user_providers:
        from hermes_cli.config import is_provider_enabled
        cfg = user_providers.get(target_provider)
        if cfg is not None and is_provider_enabled(cfg) and new_model in _declared_model_ids(cfg.get("models", {})):
            return True
    for entry in _custom_entries(custom_providers):
        if (target_provider.lower() in _entry_aliases(entry) or entry.get("base_url", "") == base_url) and (
            new_model == entry.get("model", "") or new_model in _declared_model_ids(entry.get("models", {}))
        ):
            return True
    return False


def _apply_direct_alias_endpoint(st: "_Switch", da: DirectAlias) -> None:
    """Route a direct alias to its own base_url and decide its credential (mutates ``st``).

    Credentials were resolved against the DEFAULT provider; carrying that key onto the alias
    endpoint both 401s and ships the default provider's secret to an unrelated host. The alias's
    own endpoint decides: its declared key; else the session key only for the SAME ORIGIN; else a
    fresh resolution against the alias base_url (env-key fallbacks are host-gated: OLLAMA_API_KEY
    resolves for ollama.com, OPENROUTER_API_KEY never reaches an unrelated host)."""
    from hermes_cli.models_local import _same_ollama_native_root
    from hermes_cli.runtime_provider import resolve_runtime_provider
    alias_key = direct_alias_api_key(da)
    same_host = _may_reuse_session_credential(st.base_url, da.base_url)
    if alias_key:
        st.base_url, st.api_key = da.base_url, alias_key
    elif st.api_key and st.api_key != "no-key-required" and same_host:
        # Same origin: the key is host-appropriate and re-resolving would only repeat the work.
        st.base_url = da.base_url
    else:
        try:
            req, explicit = direct_alias_runtime_request(da)
            alias_runtime = resolve_runtime_provider(
                requested=req, explicit_api_key=explicit, explicit_base_url=da.base_url, target_model=st.new_model)
        except Exception:
            alias_runtime = {}
        st.base_url = alias_runtime.get("base_url", "") or da.base_url
        # The resolver reports "no key found" as the `no-key-required` placeholder; normalise so
        # a same-host credential still outranks it.
        resolved_key = alias_runtime.get("api_key", "")
        if resolved_key == "no-key-required":
            resolved_key = ""
        st.api_key = resolved_key or (st.api_key if same_host else "") or "no-key-required"

    # providers.ollama refinement: pick up the configured key only for the configured native
    # root; drop key and provider-level headers for any other origin. Skipped when the alias
    # declared its own credential (explicit api_key/key_env outranks a provider-level config key).
    if not alias_key and st.target_provider.strip().lower() == "ollama":
        ollama_cfg, ollama_cfg_base = _ollama_configured_base()
        if ollama_cfg_base and _same_ollama_native_root(st.base_url, ollama_cfg_base):
            configured_key = _entry_configured_key(ollama_cfg, lambda n: os.environ.get(n, "").strip())
            if configured_key:
                st.api_key = configured_key
        else:
            # Different origin, or no configured root to safely associate the headers with.
            st.validation_headers, st.suppress_ollama_headers, st.api_key = {}, True, "no-key-required"
    st.api_key = st.api_key or "no-key-required"
    st.api_mode = ""  # clear so determine_api_mode re-detects from URL


def _moa_default_preset() -> str:
    try:
        from hermes_cli.config import load_config
        from hermes_cli.moa_config import normalize_moa_config
        return normalize_moa_config(load_config().get("moa") or {})["default_preset"]
    except Exception:
        return "default"


@dataclass
class _Switch:
    """Mutable state threaded through the ``switch_model`` steps.

    The routing steps settle ``target_provider`` / ``new_model`` / ``resolved_alias`` (and may
    promote a config-routed ``providers.<slug>`` to ``explicit_provider`` so the credential step
    resolves its block); the credential step fills ``api_key`` / ``base_url`` / ``api_mode`` /
    ``validation_headers``."""
    raw_input: str
    current_provider: str
    current_model: str
    current_base_url: str
    current_api_key: str
    is_global: bool
    explicit_provider: str
    user_providers: Optional[dict]
    custom_providers: Optional[list]
    new_model: str = ""
    target_provider: str = ""
    resolved_alias: str = ""
    provider_label: str = ""
    api_key: str = ""
    base_url: str = ""
    api_mode: str = ""
    validation_headers: dict = field(default_factory=dict)
    suppress_ollama_headers: bool = False
    validation: dict = field(default_factory=dict)

    def fail(self, message: str, **fields) -> ModelSwitchResult:
        return ModelSwitchResult(success=False, is_global=self.is_global, error_message=message, **fields)

    def fail_on_target(self, message: str) -> ModelSwitchResult:
        """Failure carrying the already-settled ``target_provider`` / ``provider_label``."""
        return self.fail(message, target_provider=self.target_provider, provider_label=self.provider_label)

    @property
    def provider_changed(self) -> bool:
        return self.target_provider != self.current_provider

    def resolve_runtime(self, **kwargs) -> None:
        """Fill api_key / base_url / api_mode / validation_headers from ``resolve_runtime_provider``
        for ``new_model``; headers keep their current value when the resolver returns none."""
        from hermes_cli.runtime_provider import resolve_runtime_provider
        rt = resolve_runtime_provider(target_model=self.new_model, **kwargs)
        self.api_key, self.base_url = rt.get("api_key", ""), rt.get("base_url", "")
        self.api_mode = rt.get("api_mode", "")
        self.validation_headers = rt.get("extra_headers") or self.validation_headers


def _route_explicit_provider(st: _Switch) -> Optional[ModelSwitchResult]:
    """PATH A (``--provider`` given): resolve the provider, auto-detect a model from a local
    endpoint when none was typed, then resolve the alias on the TARGET provider."""
    pdef = resolve_provider_full(st.explicit_provider, st.user_providers, st.custom_providers)
    if pdef is None and st.explicit_provider.strip().lower() == "custom":
        pdef = _bare_custom_provider_def(st.current_base_url)
    if pdef is None:
        return st.fail(_unknown_provider_message(st.explicit_provider))

    st.target_provider, st.provider_label = pdef.id, pdef.name  # label is re-derived in the credential step
    if st.target_provider == "moa" and not st.new_model:
        st.new_model = _moa_default_preset()

    agg_err = _aggregator_alias_error(
        st.explicit_provider, st.target_provider, st.current_provider, st.user_providers, st.custom_providers)
    if agg_err:
        return st.fail_on_target(agg_err)

    if not st.new_model:
        if not pdef.base_url:
            return st.fail_on_target(
                f"Provider '{pdef.name}' has no base URL configured. "
                f"Specify a model: /model <model-name> --provider {st.explicit_provider}")
        from hermes_cli.runtime_provider import _auto_detect_local_model
        st.new_model = _auto_detect_local_model(pdef.base_url)
        if not st.new_model:
            return st.fail_on_target(
                f"No model detected on {pdef.name} ({pdef.base_url}). "
                f"Specify the model explicitly: /model <model-name> --provider {st.explicit_provider}")

    try:
        alias_result = resolve_alias(st.new_model, st.target_provider)
    except AmbiguousAliasError as err:
        return st.fail(_ambiguous_alias_message(err), target_provider=st.target_provider)
    if alias_result is not None:
        _, st.new_model, st.resolved_alias = alias_result
    return None


def _route_alias_fallback(st: _Switch, key: str) -> Optional[ModelSwitchResult]:
    """Step b: the alias exists but not on the current provider -> try the user's authenticated providers."""
    authed = get_authenticated_provider_slugs(
        current_provider=st.current_provider, user_providers=st.user_providers, custom_providers=st.custom_providers,
    )
    try:
        fallback_result = _resolve_alias_fallback(st.raw_input, authed)
    except AmbiguousAliasError as err:
        return st.fail(_ambiguous_alias_message(err))
    if fallback_result is None:
        identity = MODEL_ALIASES[key]
        return st.fail(
            f"Alias '{key}' maps to {identity.vendor}/{identity.family} "
            f"but no matching model was found in any provider catalog. "
            f"Try specifying the full model name.")
    st.target_provider, st.new_model, st.resolved_alias = fallback_result
    logger.debug(
        "Alias '%s' resolved via fallback to %s on %s", st.resolved_alias, st.new_model, st.target_provider)
    return None


def _convert_vendor_colon_slug(st: _Switch) -> None:
    """Step c: ``vendor:model`` -> ``vendor/model``. Only without a slash: with one, the colon is
    a variant tag (:free, :extended, :fast) that must be preserved.

    On an aggregator every ``left:right`` is a slug. Elsewhere the colon is converted only when
    ``left`` names a provider Hermes knows, so ``/model alibaba:qwen3.6-plus`` routes like
    ``alibaba/qwen3.6-plus`` (#9748) while Ollama-style tags (``qwen3.5:4b``) stay intact."""
    raw_input = st.raw_input
    colon_pos = raw_input.find(":")
    cur_norm = str(st.current_provider).strip().lower()
    if colon_pos <= 0 or "/" in raw_input or cur_norm.startswith("custom") or cur_norm == "ollama":
        return
    left = raw_input[:colon_pos].strip().lower()
    right = raw_input[colon_pos + 1:].strip()
    if not left or not right:
        return
    if not is_aggregator(st.current_provider) and not _names_known_provider(left, st):
        return
    st.new_model = f"{left}/{right}"
    logger.debug("Converted vendor:model '%s' to slug '%s'", raw_input, st.new_model)


def _names_known_provider(name: str, st: _Switch) -> bool:
    """Whether ``name`` is a built-in provider id/alias or a provider the user configured."""
    from hermes_cli.providers import get_provider
    if resolve_provider_full(name, st.user_providers, st.custom_providers) is not None:
        return True
    try:
        return get_provider(name, allow_network=False) is not None
    except Exception:
        return False


def _route_configured_provider(st: _Switch) -> Optional[ModelSwitchResult] | bool:
    """Step d.5: a model declared in user/custom provider config routes there BEFORE
    detect_provider_for_model() guesses from static catalogs and before a soft-accepting current
    provider (openai-codex) can swallow it as an unknown hidden model. Returns a failure result,
    ``True`` when routed, else ``False``."""
    cfg_matches = _configured_provider_matches(st.new_model, st.user_providers, st.custom_providers)
    if not cfg_matches:
        return False
    if st.current_provider in cfg_matches:
        st.new_model = cfg_matches[st.current_provider]
        return True
    match_slugs = sorted(cfg_matches)
    if len(match_slugs) > 1:
        return st.fail(
            f"'{st.new_model}' is declared by multiple configured "
            f"providers ({', '.join(match_slugs)}). Re-run with "
            f"--provider <slug> to choose which one to use.")
    st.target_provider = match_slugs[0]
    st.new_model = cfg_matches[st.target_provider]
    logger.debug("Configured-provider detection routed '%s' to %s", st.new_model, st.target_provider)
    # providers.<slug> endpoints resolve in the credential block via resolve_user_provider(),
    # which is gated on explicit_provider; custom:* slugs resolve at runtime directly.
    if isinstance(st.user_providers, dict) and st.target_provider in st.user_providers:
        st.explicit_provider = st.target_provider
    return True


def _route_from_model_input(st: _Switch) -> Optional[ModelSwitchResult]:
    """PATH B (no ``--provider``): MoA preset / alias on the current provider (a) -> alias
    fallback (b) or ``vendor:model`` conversion (c) -> aggregator catalog search (d) ->
    configured-provider match (d.5) -> detect_provider_for_model() as last resort (e)."""
    from hermes_cli.models import detect_provider_for_model
    raw_input, current_provider = st.raw_input, st.current_provider
    try:
        from hermes_cli.config import load_config
        from hermes_cli.moa_config import exact_moa_preset_name, normalize_moa_config
        moa_match = exact_moa_preset_name(normalize_moa_config(load_config().get("moa") or {}), raw_input)
    except Exception:
        moa_match = None  # MoA config unreadable: fall through to plain alias resolution
    if moa_match:
        st.target_provider, st.new_model, st.resolved_alias = "moa", moa_match, ""
    else:
        try:
            alias_result = resolve_alias(raw_input, current_provider)
        except AmbiguousAliasError as err:
            return st.fail(_ambiguous_alias_message(err))
        if alias_result is not None:
            st.target_provider, st.new_model, st.resolved_alias = alias_result
            logger.debug("Alias '%s' resolved to %s on %s", st.resolved_alias, st.new_model, st.target_provider)
        elif raw_input.strip().lower() in MODEL_ALIASES:
            fail = _route_alias_fallback(st, raw_input.strip().lower())
            if fail is not None:
                return fail
        else:
            _convert_vendor_colon_slug(st)

    # Step d: if the CURRENT provider's live catalog resolved the model, step e must not
    # second-guess and switch providers — flat-namespace resellers (opencode-go/zen) return bare
    # ids that coincidentally match native providers' static catalogs.
    resolved_in_current_catalog = False
    if is_aggregator(st.target_provider) and not st.resolved_alias:
        catalog = list_provider_models(st.target_provider)
        if catalog:
            matched = _aggregator_catalog_match(st.new_model, catalog)
            if matched is not None:
                st.new_model, resolved_in_current_catalog = matched, True

    # Steps d.5 / e only apply while the request is still unrouted on the current provider.
    if st.resolved_alias or resolved_in_current_catalog or st.target_provider != current_provider:
        return None
    if current_provider == "nous":
        # The welcome host serves nous/welcome only; a model outside it needs an account or a key.
        # Never hop to another provider on the user's behalf here (there is no key to hop to).
        from hermes_cli.anon_auth import GUEST_MODEL, route_is_welcome_host
        if route_is_welcome_host(st.current_base_url) and st.new_model != GUEST_MODEL:
            return st.fail(
                f"{st.new_model} needs a Nous account or an API key. "
                "Use /login to sign in, or /model to pick another provider.")
    config_routed = _route_configured_provider(st)  # d.5 — deliberately NOT gated on ``not is_custom``
    if isinstance(config_routed, ModelSwitchResult):
        return config_routed
    is_custom = (
        current_provider in {"custom", "local"} or current_provider.startswith("custom:")
        or base_url_hostname(st.current_base_url or "") in ("localhost", "127.0.0.1"))
    if not config_routed and not is_custom:  # e
        detected = detect_provider_for_model(st.new_model, current_provider)
        if detected:
            st.target_provider, st.new_model = detected
    return None


def _switch_provider_label(st: _Switch) -> str:
    label = get_label(st.target_provider)
    if st.target_provider == "custom" and st.current_base_url:
        label = "Custom endpoint"
    if st.target_provider.startswith("custom:"):
        custom_pdef = resolve_provider_full(st.target_provider, st.user_providers, st.custom_providers)
        if custom_pdef is not None:
            label = custom_pdef.name
    return label


def _creds_for_switched_provider(st: _Switch) -> Optional[ModelSwitchResult]:
    """Credentials when the provider changed or ``--provider`` was given.

    ``providers.<name>`` blocks carry their own base_url + transport + key reference;
    resolve_runtime_provider() resolves by provider NAME and would re-resolve a block named
    "openai" from scratch (or hop to an aggregator), so use the pdef's endpoint directly."""
    user_pdef = None
    explicit_norm = st.explicit_provider.strip().lower()
    if st.explicit_provider and st.user_providers:
        from hermes_cli.providers import resolve_user_provider
        user_pdef = (resolve_user_provider(explicit_norm, st.user_providers)
                     or resolve_user_provider(st.target_provider, st.user_providers))
    if user_pdef is not None and user_pdef.base_url:
        ucfg = st.user_providers.get(explicit_norm) or st.user_providers.get(st.target_provider) or {}
        # Key reads go through the per-profile secret scope (multiplexed gateway).
        ukey = _entry_configured_key(ucfg, _scoped_key_env)
        st.validation_headers = _extra_headers_from_config(ucfg)
        try:
            st.resolve_runtime(
                requested=st.target_provider, explicit_api_key=ukey or None, explicit_base_url=user_pdef.base_url)
            st.api_key, st.base_url = st.api_key or ukey, st.base_url or user_pdef.base_url
        except Exception:
            st.api_key, st.base_url, st.api_mode = ukey, user_pdef.base_url, ""
    elif st.target_provider == "custom" and st.current_base_url:
        st.api_key, st.base_url = st.current_api_key, st.current_base_url
        st.api_mode = determine_api_mode(st.target_provider, st.base_url)
    else:
        try:
            st.resolve_runtime(requested=st.target_provider)
        except Exception as e:
            return st.fail_on_target(
                f"{st.provider_label} is not connected: no API key or login was found for it. Add one with "
                f"`hermes auth add {st.target_provider}`, or pick a connected provider in /model.\n"
                f"  Details: {e}")
    return None


def _creds_for_current_provider(st: _Switch) -> None:
    """Credentials when staying on the current provider. Mid-session ``/model <name>`` on a local
    Ollama-compatible endpoint keeps the endpoint in use; re-resolving bare ``custom`` from config
    can fall through to an unrelated default provider."""
    from hermes_cli.models_local import _get_ollama_request_headers, _same_ollama_native_root
    keep_current_ollama_endpoint = False
    ollama_headers: dict[str, str] = {}
    if st.current_provider == "custom" and st.current_base_url:
        try:
            from hermes_cli.models_local import should_use_ollama_native_catalog
            ollama_headers = _get_ollama_request_headers()
            _, configured_ollama_base = _ollama_configured_base()
            # Provider-level Ollama headers only belong to the configured native root; without
            # one there is no safe origin for them.
            if not configured_ollama_base or not _same_ollama_native_root(st.current_base_url, configured_ollama_base):
                ollama_headers = {}
                st.suppress_ollama_headers = True
            keep_current_ollama_endpoint = should_use_ollama_native_catalog(
                st.current_provider, st.current_base_url, headers=ollama_headers)
        except (ImportError, OSError, RuntimeError, TypeError, ValueError):
            keep_current_ollama_endpoint = False
    if keep_current_ollama_endpoint:
        st.api_key = st.current_api_key or "no-key-required"
        st.base_url = st.current_base_url
        st.api_mode = determine_api_mode(st.current_provider, st.base_url)
        st.validation_headers = ollama_headers
    else:
        try:
            st.resolve_runtime(requested=st.current_provider)
        except Exception:
            pass
        # Bare ``custom``/``local`` sessions whose base_url is session-only (not a trusted config
        # ``model.base_url``) re-resolve to the OpenRouter DEFAULT — a host the user never picked
        # (#74143). Keep the session endpoint + key then; a config-backed custom URL still wins so
        # key/endpoint rotation is not pinned to a stale session.
        if (
            st.current_provider in {"custom", "local"} and st.current_base_url
            and (not st.base_url or base_url_host_matches(st.base_url, "openrouter.ai"))
            and not base_url_host_matches(st.current_base_url, "openrouter.ai")
        ):
            st.base_url, st.api_key = st.current_base_url, st.current_api_key
            st.api_mode = determine_api_mode(st.current_provider, st.base_url)


def _resolve_switch_credentials(st: _Switch) -> Optional[ModelSwitchResult]:
    """COMMON PATH part 1: credentials, direct-alias endpoint override, and the api_mode for the
    final (provider, base_url) before validation."""
    st.provider_label = _switch_provider_label(st)
    st.api_key, st.base_url = st.current_api_key, st.current_base_url
    if st.provider_changed or st.explicit_provider:
        fail = _creds_for_switched_provider(st)
        if fail is not None:
            return fail
    else:
        _creds_for_current_provider(st)

    # Direct alias override: use the alias's exact base_url if set.
    if st.resolved_alias:
        _ensure_direct_aliases()
        da = DIRECT_ALIASES.get(st.resolved_alias)
        if da is not None and da.base_url:
            _apply_direct_alias_endpoint(st, da)

    # Fills an empty mode (alias cleared it) and overrides a STALE mode carried from previous
    # session state when the host mandates one wire protocol (e.g. gpt-5.x on api.openai.com
    # would otherwise 400 on tools+reasoning).
    from hermes_cli.providers import is_actual_route
    mandated_mode = "chat_completions" if is_actual_route(st.target_provider, st.base_url) else host_mandated_api_mode(st.base_url)
    if mandated_mode is not None:
        st.api_mode = mandated_mode
    st.api_mode = st.api_mode or determine_api_mode(st.target_provider, st.base_url)
    return None


def _validate_switch(st: _Switch) -> Optional[ModelSwitchResult]:
    """COMMON PATH part 2: normalize the model name for the target provider, validate it, and
    accept config-declared models the remote catalog lacks."""
    from hermes_cli.models_local import _get_ollama_request_headers
    from hermes_cli.models_validate import validate_requested_model
    st.new_model = _resolve_named_custom_model_id(st.new_model, st.target_provider, st.custom_providers)
    st.new_model = normalize_model_for_provider(st.new_model, st.target_provider)

    if st.target_provider.strip().lower() == "ollama":
        headers = {} if st.suppress_ollama_headers else (st.validation_headers or _get_ollama_request_headers())
    else:
        headers = st.validation_headers or (
            _extra_headers_from_config(st.user_providers.get(st.target_provider))
            if st.user_providers and st.target_provider in st.user_providers else None)
    # A ``providers.<key>`` endpoint is the user's own: validate it as a custom endpoint (an id its
    # listing lacks is soft-accepted) whether the slug arrived as ``custom:<key>`` or the bare key
    # the picker rows carry — otherwise the bare spelling fell into the built-in live-listing
    # branch and hard-rejected the very model the user selected.
    validate_as = st.target_provider
    if not validate_as.lower().startswith("custom"):
        pdef = resolve_provider_full(validate_as, st.user_providers, st.custom_providers)
        if pdef is not None and pdef.source == "user-config":
            validate_as = f"custom:{validate_as}"
    try:
        validation = validate_requested_model(
            st.new_model, validate_as, api_key=st.api_key, base_url=st.base_url,
            api_mode=st.api_mode or None, headers=headers)
    except Exception as e:
        validation = {"accepted": False, "persist": False, "recognized": False,
                      "message": f"Could not validate `{st.new_model}`: {e}"}

    if not validation.get("accepted"):
        if not _config_declares_model(
                st.new_model, st.target_provider, st.base_url, st.user_providers, st.custom_providers):
            return st.fail(
                validation.get("message", "Invalid model"),
                new_model=st.new_model, target_provider=st.target_provider, provider_label=st.provider_label)
        validation = {"accepted": True, "persist": True, "recognized": False, "message": validation.get("message", "")}
    st.validation = validation
    return None


def _copilot_api_mode(provider: str, model: str, api_key: str) -> str:
    from hermes_cli.models import copilot_model_api_mode
    return copilot_model_api_mode(model, api_key=api_key)


def _opencode_api_mode(provider: str, model: str, api_key: str) -> str:
    # Re-derive api_mode from the effective model rather than the persisted api_mode: the opencode providers
    # serve both anthropic_messages and chat_completions models, so the previous session's mode must not
    # leak across /model switches. Refs #16878.
    # opencode-zen/go must always re-derive api_mode from the target model (not the stale persisted
    # api_mode), because the same provider serves both anthropic_messages (e.g. minimax-m2.7) and
    # chat_completions (e.g. deepseek-v4-flash) and switching models via /model would otherwise carry the
    # previous mode forward, stripping /v1 from base_url for chat_completions models and 404'ing. Refs
    # #16878.
    from hermes_cli.models import opencode_model_api_mode
    return opencode_model_api_mode(provider, model)


def _nous_api_mode(provider: str, model: str, api_key: str) -> str:
    # Portal serves anthropic/* on /v1/messages and everything else on /chat/completions;
    # re-derive from the FINAL model so alias clears / empty fallbacks cannot leave Claude on the
    # OpenAI wire.
    from hermes_cli.providers import nous_api_mode
    return nous_api_mode(model)


# Per-provider api_mode overrides applied after validation, keyed on the final target provider
# (the key sets are disjoint, so exactly one — or none — fires).
_PROVIDER_API_MODE_OVERRIDES: dict[str, Any] = {
    **dict.fromkeys(("copilot", "github-copilot"), _copilot_api_mode),
    **dict.fromkeys(("opencode-zen", "opencode-go", "opencode"), _opencode_api_mode),
    **dict.fromkeys(("nous", "nous-portal", "nousresearch"), _nous_api_mode)}


def _build_switch_result(st: _Switch) -> ModelSwitchResult:
    """COMMON PATH part 3: final api_mode / base_url shaping, metadata, warnings."""
    override = _PROVIDER_API_MODE_OVERRIDES.get(st.target_provider)
    if override is not None:
        st.api_mode = override(st.target_provider, st.new_model, st.api_key)
    if not st.api_mode:
        st.api_mode = determine_api_mode(st.target_provider, st.base_url, model=st.new_model)

    # OpenCode base URLs end with /v1 for OpenAI-compatible models but the Anthropic SDK prepends
    # its own /v1/messages: strip for anthropic_messages, re-append for
    # chat_completions/codex_responses (mirrors resolve_runtime_provider).
    from hermes_cli.models import normalize_opencode_base_url, opencode_provider_family
    if opencode_provider_family(st.target_provider) is not None and isinstance(st.base_url, str):
        st.base_url = normalize_opencode_base_url(st.target_provider, st.api_mode, st.base_url)

    capabilities = get_model_capabilities(st.target_provider, st.new_model, allow_network=True)
    from agent.native_compaction import resolve_native_compaction_capabilities
    runtime_capabilities = resolve_native_compaction_capabilities(
        model=st.new_model, base_url=st.base_url, provider=st.target_provider,
        is_codex_backend=st.target_provider.strip().lower() == "openai-codex")
    model_info = get_model_info(st.target_provider, st.new_model, allow_network=True)

    warnings = [w for w in (st.validation.get("message"), _check_hermes_model_warning(st.new_model)) if w]

    # Carry the switched provider's request_overrides (custom_providers ``extra_body`` such as
    # chat_template_kwargs) so the gateway applies them like the default-provider path does.
    request_overrides = None
    try:
        from hermes_cli.runtime_provider import _get_named_custom_provider, _custom_provider_request_overrides
        cp_for_ro = _get_named_custom_provider(st.target_provider)
        request_overrides = _custom_provider_request_overrides(cp_for_ro) or None if cp_for_ro else None
    except Exception:
        request_overrides = None
    return ModelSwitchResult(
        success=True, new_model=st.new_model, target_provider=st.target_provider,
        provider_changed=st.provider_changed, api_key=st.api_key, base_url=st.base_url, api_mode=st.api_mode,
        request_overrides=dict(request_overrides or {}), warning_message=" | ".join(warnings) if warnings else "",
        provider_label=st.provider_label, resolved_via_alias=st.resolved_alias, capabilities=capabilities,
        runtime_capabilities={
            k: v for k, v in runtime_capabilities.items() if isinstance(k, str) and isinstance(v, bool)},
        model_info=model_info, is_global=st.is_global)


def switch_model(
    raw_input: str, current_provider: str, current_model: str, current_base_url: str = "",
    current_api_key: str = "", is_global: bool = False, explicit_provider: str = "",
    user_providers: dict = None, custom_providers: list | None = None) -> ModelSwitchResult:
    """Core model-switching pipeline shared between CLI and gateway.

    Route (PATH A with ``--provider``, else PATH B) -> credentials -> validation -> result; each
    step returns a failure :class:`ModelSwitchResult` to stop the chain, or ``None`` to continue.
    ``user_providers`` / ``custom_providers`` are the config.yaml ``providers:`` dict and
    ``custom_providers:`` list."""
    st = _Switch(
        raw_input=raw_input, current_provider=current_provider, current_model=current_model,
        current_base_url=current_base_url, current_api_key=current_api_key, is_global=is_global,
        explicit_provider=explicit_provider, user_providers=user_providers, custom_providers=custom_providers,
        new_model=raw_input.strip(), target_provider=current_provider)
    route = _route_explicit_provider if explicit_provider else _route_from_model_input
    for step in (route, _resolve_switch_credentials, _validate_switch):
        fail = step(st)
        if fail is not None:
            return fail
    return _build_switch_result(st)


def model_selection_config_updates(result: ModelSwitchResult, current_model_cfg: Any) -> dict[str, Any]:
    """The ONE config.yaml shape a persisted model selection produces, as ``model.<key>`` -> value
    (``None`` = clear). ``current_model_cfg`` is the on-disk ``model:`` block (raw).

    base_url/api_mode are freshly resolved for the target route, so they are always synced —
    ``None`` when the target has none — otherwise the OLD provider's endpoint/wire-protocol lingers
    (#25106). A context pin is dropped only when its route identity changed (fail-closed).
    Non-custom targets resolve credentials from env/auth.json/the pool, so an inline
    ``model.api_key`` is a leftover that would contaminate later custom resolution. For custom
    targets the inline key belongs to ONE endpoint: it survives only a same-route re-pick (same
    provider and base_url) — ``custom:a`` -> ``custom:b`` must not hand endpoint A's secret to B.
    The ``key_env`` / ``api_key_env`` credential POINTER (written by custom-endpoint activation
    and, for REGISTRY providers too, by the Desktop settings UI (#106336); resolved by
    runtime_provider / auxiliary_client / ``auth._model_level_key_env``) clears only when the
    route changed: left behind it routes the NEW provider's requests to the OLD endpoint's env
    var, but a same-provider same-base_url model re-pick keeps it whatever the provider is. The
    dashboard re-adds an explicitly submitted key / the target provider's own pointer after this
    (``_apply_main_model_assignment`` / ``_resolve_assignment_credentials``)."""
    model_cfg = current_model_cfg if isinstance(current_model_cfg, dict) else {}
    updates: dict[str, Any] = {
        "default": result.new_model, "provider": result.target_provider,
        "base_url": result.base_url or None, "api_mode": result.api_mode or None,
    }
    if "context_length" in model_cfg:
        from hermes_cli.route_identity import should_clear_context_pin
        if should_clear_context_pin(
                model_cfg.get("default") or model_cfg.get("model"), result.new_model,
                model_cfg.get("base_url"), result.base_url, model_cfg.get("provider"), result.target_provider):
            updates["context_length"] = None
    target = str(result.target_provider or "").strip().lower()
    route_changed = _route_changed(model_cfg, result)
    stale = ["api_key", "api"] if (not target.startswith("custom") or route_changed) else []
    if route_changed:
        stale += ["key_env", "api_key_env"]
    for key in stale:
        if key in model_cfg:
            updates[key] = None
    return updates


def _route_changed(model_cfg: dict, result: ModelSwitchResult) -> bool:
    """Provider or endpoint differs between the on-disk ``model:`` block and the switch target."""
    from hermes_cli.route_identity import normalize_route_base_url
    if str(model_cfg.get("provider") or "").strip().lower() != str(result.target_provider or "").strip().lower():
        return True
    return normalize_route_base_url(model_cfg.get("base_url")) != normalize_route_base_url(result.base_url)


def apply_model_selection(model_cfg: Any, result: ModelSwitchResult) -> dict:
    """Apply the canonical shape to an in-memory ``model:`` dict (``None`` = key removed) for
    callers that save a whole config document they are already mutating."""
    model_cfg = dict(model_cfg) if isinstance(model_cfg, dict) else {}
    for key, value in model_selection_config_updates(result, model_cfg).items():
        if value is None:
            model_cfg.pop(key, None)
        else:
            model_cfg[key] = value
    return model_cfg


def persist_model_selection(result: ModelSwitchResult, config_path: Any = None) -> None:
    """Write a successful :func:`switch_model` result to ``config_path`` (default:
    ``HERMES_HOME/config.yaml`` — the context override or ``HERMES_HOME`` at call time).

    Targeted key writes, not a whole-``model:`` rewrite: a block rewrite destroys sibling keys the
    user set there (``model_slots``, ``model_fallback``, ...). ``should_clear_context_pin`` can do
    cold-start disk I/O — async callers run this on a worker thread."""
    from pathlib import Path
    from hermes_cli.config import get_config_path, read_user_config_raw, warn_unpinned_cron_jobs_after_model_config_change
    from utils import atomic_roundtrip_yaml_update
    path = Path(config_path) if config_path else get_config_path()
    for key, value in model_selection_config_updates(result, read_user_config_raw(path).get("model")).items():
        atomic_roundtrip_yaml_update(path, f"model.{key}", value)
        # Same unpinned-cron notice as `hermes config set` for every model switch.
        warn_unpinned_cron_jobs_after_model_config_change(f"model.{key}", value)
    try:  # owner-only: config files contain API keys
        os.chmod(path, 0o600)
    except (OSError, NotImplementedError):
        pass


def _extra_headers_from_config(entry: Any) -> dict[str, str]:
    if not isinstance(entry, dict):
        return {}
    from hermes_cli.config import normalize_extra_headers
    return normalize_extra_headers(entry.get("extra_headers"))


def _scoped_key_env(name: str) -> str:
    """Read a provider key env var the way the chat path does, honouring the per-profile scope.

    With a secret scope installed (multiplexed gateway turn, dashboard/kanban workers) the scope's
    verdict is authoritative: a hit is this profile's key, a miss must not borrow another profile's
    value from the process env or the default ``.env``. Multiplexing on with no scope fails closed
    (``UnscopedSecretError`` -> ""). Otherwise resolve through ``get_env_prefer_dotenv`` — the
    chain ``client_lifecycle`` uses for the actual request — so a ``key_env`` that lives only in
    ``$HERMES_HOME/.env`` authenticates the ``/model`` verification probe (#109315) and a rotated
    ``.env`` beats a stale value inherited from the parent shell."""
    if not name:
        return ""
    try:
        from agent.secret_scope import current_secret_scope, get_secret, is_multiplex_active
        if current_secret_scope() is not None or is_multiplex_active():
            return (get_secret(name, "") or "").strip()
        from agent.credential_pool import get_env_prefer_dotenv
        return (get_env_prefer_dotenv(name) or "").strip()
    except Exception:
        return ""


# --- Parallel prefetch for provider model catalogs -----------------------
#
# When the 1h disk cache lapses (or on first cold open), list_authenticated_providers()
# calls cached_provider_model_ids() serially for each authed provider.  Each call
# that misses the cache blocks on a live /v1/models HTTP round-trip (1-8s per
# provider depending on endpoint latency).  With 10+ authed providers the
# cumulative serial blocking time is 15-30+ seconds.
#
# This prefetch function runs those same cached_provider_model_ids() calls in
# parallel via ThreadPoolExecutor before the main picker build loop starts.
# The main loop then hits warm cache entries instead of blocking on live
# fetches.  Providers whose cache was already fresh (SWR or within TTL) are
# skipped entirely — no wasted network calls.
#
# Net effect on a 13-provider setup with an expired cache:
#   Before: ~20s serial blocking (sum of all provider latencies)
#   After:  ~8s parallel (max single provider latency), rest served from cache

_PARALLEL_PREFETCH_WORKERS = 8


def _prefetch_provider_models_parallel(provider_slugs: list[str]) -> None:
    """Fetch model catalogs for multiple providers in parallel.

    Only providers whose cache entry is stale or missing are fetched; fresh
    entries are skipped to avoid unnecessary network calls.  Each worker uses
    :func:`update_provider_cache_entry` (thread-safe) to persist its result,
    so concurrent writes to ``provider_models_cache.json`` don't clobber each
    other.

    :param provider_slugs: Hermes provider IDs to prefetch (e.g. ``["openrouter",
        "anthropic", "deepseek"]``).  Unknown providers are silently skipped.
    """
    from hermes_cli.models import cached_provider_model_ids

    # Quick-stale-check: skip providers whose cache is already fresh so we
    # don't waste network calls on a warm cache.  We check staleness the same
    # way cached_provider_model_ids does internally: load the cache, compare
    # age to TTL.  This is a read-only check — if the cache file changes
    # between this check and the actual fetch, cached_provider_model_ids will
    # still do the right thing (it re-reads the cache internally).
    from hermes_cli.models import (
        _load_provider_models_cache,
        _credential_fingerprint,
        _PROVIDER_MODELS_CACHE_TTL,
        normalize_provider,
    )

    now = time.time()
    stale_slugs: list[str] = []
    cache = _load_provider_models_cache()
    for slug in provider_slugs:
        normalized = normalize_provider(slug) or (slug or "")
        if not normalized:
            continue
        entry = cache.get(normalized)
        fp = _credential_fingerprint(normalized)
        if (
            isinstance(entry, dict)
            and entry.get("fp") == fp
            and isinstance(entry.get("models"), list)
            and entry["models"]
        ):
            age = now - float(entry.get("at", 0))
            if age < _PROVIDER_MODELS_CACHE_TTL:
                continue  # fresh, skip
        stale_slugs.append(normalized)

    if not stale_slugs:
        return

    import concurrent.futures

    def _fetch_one(slug: str) -> None:
        try:
            models = cached_provider_model_ids(slug, force_refresh=True)
            # cached_provider_model_ids already persists the result, but in a
            # non-locked read-modify-write.  Re-persist via the thread-safe
            # path to guarantee no lost writes under concurrency.
            if models:
                from hermes_cli.models import update_provider_cache_entry
                update_provider_cache_entry(slug, models)
        except Exception:
            pass  # best-effort; picker falls back to curated list

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(_PARALLEL_PREFETCH_WORKERS, len(stale_slugs)),
        thread_name_prefix="model-cache-prefetch",
    ) as executor:
        list(executor.map(_fetch_one, stale_slugs))


def _collect_authed_provider_slugs(
    models_dev_data: dict,
    curated: dict[str, list[str]],
    excluded: list[str],
) -> list[str]:
    """Quick-scan which providers have credentials, without fetching model lists.

    Mirrors the credential-check logic from sections 1, 2, and 2b of
    :func:`list_authenticated_providers` but **only** collects the provider
    slugs — it never calls ``cached_provider_model_ids``.  The returned list
    is consumed by :func:`_prefetch_provider_models_parallel` to warm the disk
    cache in parallel before the serial picker build loop starts.

    :param models_dev_data: The models.dev registry dict (from ``fetch_models_dev()``).
    :param curated: The curated model-lists dict (``_PROVIDER_MODELS`` + extras).
    :param excluded: Provider slugs to exclude (from ``model_catalog.excluded_providers``).
    :returns: List of normalized provider slugs that have credentials.
    """
    import os
    from agent.models_dev import PROVIDER_TO_MODELS_DEV
    from hermes_cli.auth import PROVIDER_REGISTRY, _load_auth_store
    from hermes_cli.providers import HERMES_OVERLAYS, ALIASES as _PROVIDER_ALIAS_TABLE
    from hermes_cli.models import _AGGREGATOR_PROVIDERS as _AGG_PROVIDERS, CANONICAL_PROVIDERS

    _excluded_set = {str(p).strip().lower() for p in excluded if p}
    slugs: list[str] = []
    seen: set[str] = set()

    # --- Section 1: Hermes-mapped providers (PROVIDER_TO_MODELS_DEV) ---
    for hermes_id, mdev_id in PROVIDER_TO_MODELS_DEV.items():
        _alias_target = _PROVIDER_ALIAS_TABLE.get(hermes_id)
        if (
            _alias_target
            and _alias_target != hermes_id
            and _alias_target in _AGG_PROVIDERS
        ):
            continue
        _canonical = hermes_id
        try:
            from providers import get_provider_profile as _gpp
            _prof = _gpp(hermes_id)
            if _prof is not None:
                _canonical = _prof.name
        except Exception:
            pass
        if _canonical != hermes_id:
            continue
        if hermes_id.lower() in seen:
            continue
        if hermes_id.lower() in _excluded_set or mdev_id.lower() in _excluded_set:
            continue
        pdata = models_dev_data.get(mdev_id)
        if not isinstance(pdata, dict):
            continue
        pconfig = PROVIDER_REGISTRY.get(hermes_id)
        if pconfig and pconfig.auth_type != "api_key":
            continue
        from hermes_cli.auth import is_runtime_provider_routable
        if not is_runtime_provider_routable(hermes_id):
            continue
        if pconfig and pconfig.api_key_env_vars:
            env_vars = list(pconfig.api_key_env_vars)
        else:
            env_vars = pdata.get("env", [])
            if not isinstance(env_vars, list):
                continue
        has_creds = any(_scoped_key_env(ev) for ev in env_vars)
        if not has_creds:
            try:
                store = _load_auth_store()
                raw_pool_present = bool(
                    store and store.get("credential_pool", {}).get(hermes_id)
                )
                if raw_pool_present:
                    has_creds = _credential_pool_is_usable(
                        hermes_id, raw_pool_present=True
                    )
            except Exception:
                pass
        if has_creds:
            slugs.append(hermes_id)
            seen.add(hermes_id.lower())

    # --- Section 2: Hermes-only providers (HERMES_OVERLAYS) ---
    _mdev_to_hermes = {v: k for k, v in PROVIDER_TO_MODELS_DEV.items()}
    for pid, overlay in HERMES_OVERLAYS.items():
        if pid.lower() in seen:
            continue
        hermes_slug = _mdev_to_hermes.get(pid, pid)
        if hermes_slug.lower() in seen:
            continue
        if pid.lower() in _excluded_set or hermes_slug.lower() in _excluded_set:
            continue
        has_creds = False
        if overlay.auth_type == "aws_sdk":
            # Skip AWS SDK providers in prefetch — credential detection is heavier
            continue
        elif overlay.auth_type == "vertex":
            try:
                from agent.vertex_adapter import has_vertex_credentials
                has_creds = has_vertex_credentials()
            except Exception:
                pass
        elif overlay.extra_env_vars:
            has_creds = any(_scoped_key_env(ev) for ev in overlay.extra_env_vars)
        if not has_creds and overlay.auth_type == "api_key":
            for _key in (pid, hermes_slug):
                pcfg = PROVIDER_REGISTRY.get(_key)
                if pcfg and pcfg.api_key_env_vars:
                    if any(_scoped_key_env(ev) for ev in pcfg.api_key_env_vars):
                        has_creds = True
                        break
        if not has_creds:
            try:
                store = _load_auth_store()
                providers_store = store.get("providers", {}) if store else {}
                if pid in providers_store or hermes_slug in providers_store:
                    has_creds = True
            except Exception:
                pass
        if not has_creds:
            try:
                if _credential_pool_is_usable(hermes_slug):
                    has_creds = True
            except Exception:
                pass
        if has_creds:
            slugs.append(hermes_slug)
            seen.add(pid.lower())
            seen.add(hermes_slug.lower())

    # --- Section 2b: Canonical providers cross-check ---
    for _cp in CANONICAL_PROVIDERS:
        if _cp.slug.lower() in seen:
            continue
        if _cp.slug.lower() in _excluded_set:
            continue
        _cp_config = PROVIDER_REGISTRY.get(_cp.slug)
        _cp_has_creds = False
        if _cp_config and _cp_config.api_key_env_vars:
            _cp_has_creds = any(_scoped_key_env(ev) for ev in _cp_config.api_key_env_vars)
        if not _cp_has_creds:
            try:
                _cp_store = _load_auth_store()
                _cp_providers_store = _cp_store.get("providers", {}) if _cp_store else {}
                if _cp.slug in _cp_providers_store:
                    _cp_has_creds = True
            except Exception:
                pass
        if not _cp_has_creds:
            try:
                if _credential_pool_is_usable(_cp.slug):
                    _cp_has_creds = True
            except Exception:
                pass
        if not _cp_has_creds and _cp_config and getattr(_cp_config, "auth_type", "") == "aws_sdk":
            continue  # skip AWS SDK in prefetch
        if _cp_has_creds:
            slugs.append(_cp.slug)
            seen.add(_cp.slug.lower())

    return slugs


def list_authenticated_providers(
    current_provider: str = "",
    current_base_url: str = "",
    user_providers: dict = None,
    custom_providers: list | None = None,
    *,
    force_fresh_nous_tier: bool = False,
    max_models: int | None = None,
    current_model: str = "",
    refresh: bool = False,
    probe_custom_providers: bool = True,
    probe_current_custom_provider: bool = False,
    for_picker: bool = False,
    excluded_providers: list | None = None,
) -> List[dict]:
    """Detect which providers have credentials and list their curated models.

    Uses the curated model lists from hermes_cli/models.py (OPENROUTER_MODELS,
    _PROVIDER_MODELS) — NOT the full models.dev catalog.  These are hand-picked
    agentic models that work well as agent backends.

    Returns a list of dicts, each with:
      - slug: str — the --provider value to use
      - name: str — display name
      - is_current: bool
      - is_user_defined: bool
      - models: list[str] — curated model IDs (up to max_models)
      - total_models: int — total curated count
      - source: str — "built-in", "models.dev", "user-config"

    Only includes providers that have API keys set or are user-defined endpoints.
    ``force_fresh_nous_tier`` bypasses the short Nous tier cache for explicit
    account-sensitive flows. UI picker opens should leave it false so they do
    not block on fresh Portal/account checks every time.

    ``refresh`` busts the per-provider model-id disk cache
    (``provider_models_cache.json``) up front so every row re-fetches its
    live catalog. Use for an explicit user-triggered "refresh models" action
    (e.g. the desktop picker's refresh control); leave false for normal picker
    opens so they stay snappy on the 1h cache.

    ``probe_custom_providers`` controls live ``/models`` discovery for saved
    custom OpenAI-compatible endpoints. Keep the default true for CLI parity;
    GUI picker opens can pass false to show configured models immediately
    without waiting on offline local endpoints.

    ``probe_current_custom_provider`` is the middle ground for GUI picker
    opens: probe only the currently-selected custom endpoint so its model list
    matches the active provider without blocking on every saved/offline custom
    endpoint.
    """
    import os
    from agent.models_dev import (
        PROVIDER_TO_MODELS_DEV,
        fetch_models_dev,
        get_provider_info as _mdev_pinfo,
    )
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.models import (
        OPENROUTER_MODELS, _PROVIDER_MODELS,
        _MODELS_DEV_PREFERRED, _merge_with_models_dev, cached_provider_model_ids,
        clear_provider_models_cache, get_curated_nous_model_ids,
    )

    # Explicit refresh: drop every provider's cached model-id list so the
    # cached_provider_model_ids() calls below all re-fetch live. Without this
    # a stale 1h cache can fall back to the curated static list when its live
    # fetch later fails, silently dropping live-only models (e.g. OpenCode
    # Zen's free tier) the user had seen before.
    if refresh:
        try:
            clear_provider_models_cache()
        except Exception:
            pass

    from hermes_cli.config import coerce_provider_id, stringify_provider_map

    results: List[dict] = []
    seen_slugs: set = set()  # lowercase-normalized to catch case variants (#9545)
    # PyYAML parses unquoted numeric names (`provider: 2070`) as int. Later
    # `.strip()` / `.lower()` on that raw value 500s GET /api/model/options.
    current_provider = coerce_provider_id(current_provider)
    current_base_url = str(current_base_url or "").strip()
    current_model = str(current_model or "").strip()
    _current_provider_norm = current_provider.lower()
    _current_base_url_norm = current_base_url.rstrip("/").lower()
    user_providers = stringify_provider_map(user_providers)

    def _can_probe_custom_provider(*, row_is_current: bool) -> bool:
        return bool(probe_custom_providers or (probe_current_custom_provider and row_is_current))

    # Normalize the excluded-providers list once for fast membership checks.
    # Compared against hermes_id / mdev_id (section 1), pid / hermes_slug
    # (section 2) and canonical slug (section 2b) so a single entry like
    # ``copilot`` hides the provider regardless of which key it surfaces under.
    _excluded: set = {str(p).strip().lower() for p in (excluded_providers or []) if p}
    # Effective base URLs of every built-in row we emit (normalized lower+rstrip).
    # Section 4 uses this to hide ``custom_providers`` entries that point at the
    # same endpoint as a built-in (e.g. a user-defined "my-dashscope" on
    # https://coding-intl.dashscope.aliyuncs.com/v1 collides with the built-in
    # alibaba-coding-plan row when DASHSCOPE_API_KEY is present). Fixes #16970.
    _builtin_endpoints: set = set()

    def _norm_url(url: str) -> str:
        return str(url or "").strip().rstrip("/").lower()

    def _record_builtin_endpoint(slug: str) -> None:
        """Record the effective base URL for a built-in provider row.

        Prefers the live env-override (e.g. DASHSCOPE_BASE_URL) over the
        static inference_base_url so the dedup matches what a user typing
        that URL into custom_providers would actually hit."""
        try:
            from hermes_cli.auth import PROVIDER_REGISTRY as _reg
        except Exception:
            return
        pcfg = _reg.get(slug)
        if not pcfg:
            return
        url = ""
        if getattr(pcfg, "base_url_env_var", ""):
            url = os.environ.get(pcfg.base_url_env_var, "") or ""
        if not url:
            url = getattr(pcfg, "inference_base_url", "") or ""
        normed = _norm_url(url)
        if normed:
            _builtin_endpoints.add(normed)

    def _has_fast_aws_sdk_signal() -> bool:
        """Return True when explicit AWS auth config is present.

        This intentionally avoids botocore's full credential chain. Provider
        picker/model-switch discovery can run for non-Bedrock providers, and
        botocore may otherwise probe EC2 IMDS (169.254.169.254) on local
        machines before returning no credentials.
        """
        if os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "").strip():
            return True
        if (
            os.environ.get("AWS_ACCESS_KEY_ID", "").strip()
            and os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip()
        ):
            return True
        return any(
            os.environ.get(name, "").strip()
            for name in (
                "AWS_PROFILE",
                "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
                "AWS_CONTAINER_CREDENTIALS_FULL_URI",
                "AWS_WEB_IDENTITY_TOKEN_FILE",
            )
        )

    def _has_aws_sdk_creds_for_listing(slug: str) -> bool:
        """Credential check for AWS SDK providers in non-runtime discovery."""
        slug_norm = str(slug or "").strip().lower()
        current_norm = str(current_provider or "").strip().lower()
        if _has_fast_aws_sdk_signal():
            return True
        if slug_norm != current_norm:
            return False
        try:
            from agent.bedrock_adapter import has_aws_credentials
            return bool(has_aws_credentials())
        except Exception:
            return False

    data = fetch_models_dev()

    # Build curated model lists keyed by hermes provider ID
    curated: dict[str, list[str]] = dict(_PROVIDER_MODELS)
    curated["openrouter"] = [mid for mid, _ in OPENROUTER_MODELS]
    # "nous" pulls from the remote model-catalog manifest published at
    # https://hermes-agent.nousresearch.com/docs/api/model-catalog.json so
    # newly added Portal models surface in the /model picker without
    # requiring a Hermes release. Falls back to the in-repo
    # _PROVIDER_MODELS["nous"] snapshot when the manifest is unreachable.
    curated["nous"] = get_curated_nous_model_ids()
    # Ollama Cloud uses dynamic discovery (no static curated list)
    if "ollama-cloud" not in curated:
        from hermes_cli.models import fetch_ollama_cloud_models
        curated["ollama-cloud"] = fetch_ollama_cloud_models()
    # LM Studio has no static catalog — probe its native /api/v1/models
    # endpoint live so the picker reflects whatever the user has loaded.
    # Base URL precedence: LM_BASE_URL env var > active config's base_url
    # (when current provider is lmstudio) > 127.0.0.1 default.
    # On auth rejection or unreachable server, fall back to the caller-supplied
    # current model so the picker still shows something when offline / mis-keyed.
    if "lmstudio" not in curated and (
        os.environ.get("LM_API_KEY") or os.environ.get("LM_BASE_URL") or current_provider.strip().lower() == "lmstudio"
    ):
        from hermes_cli.models import fetch_lmstudio_models
        from hermes_cli.auth import AuthError
        is_current_lmstudio = current_provider.strip().lower() == "lmstudio"
        lm_base = (
            os.environ.get("LM_BASE_URL")
            or (current_base_url if is_current_lmstudio and current_base_url else None)
            or "http://127.0.0.1:1234/v1"
        )
        try:
            live = fetch_lmstudio_models(
                api_key=os.environ.get("LM_API_KEY", ""),
                base_url=lm_base,
                timeout=1.5, # Smaller timeout for picker
            )
        except AuthError:
            live = []
        if not live and is_current_lmstudio and current_model:
            live = [current_model]
        curated["lmstudio"] = live

    # --- Parallel cache prefetch ---------------------------------------------
    # The serial loops below (sections 1, 2, 2b) each call
    # cached_provider_model_ids(slug) which blocks on a live /v1/models HTTP
    # round-trip when the disk cache is stale or missing.  With many authed
    # providers those serial round-trips stack to 15-30s on a cold/expired
    # cache.  Pre-scanning which providers have credentials (without fetching
    # their model lists) and warming their cache entries in parallel makes
    # the subsequent serial calls hit fresh cache entries instead.
    #
    # Skipped entirely when refresh=True (the serial path already force-refreshes)
    # and when there are 3 or fewer authed providers (serial is fast enough;
    # avoids thread-pool overhead for the common 1-2 provider case).
    _prefetch_slugs: list[str] = []
    if not refresh:
        _prefetch_slugs = _collect_authed_provider_slugs(
            data, curated, excluded_providers or []
        )
    if len(_prefetch_slugs) > 3:
        try:
            _prefetch_provider_models_parallel(_prefetch_slugs)
        except Exception:
            pass  # best-effort; serial path still works as fallback

    # --- 1. Check Hermes-mapped providers ---
    from hermes_cli.models import _AGGREGATOR_PROVIDERS as _AGG_PROVIDERS
    from hermes_cli.providers import ALIASES as _PROVIDER_ALIAS_TABLE
    for hermes_id, mdev_id in PROVIDER_TO_MODELS_DEV.items():
        # Skip vendor names that are merely aliases routing through an
        # aggregator (e.g. bare "openai" → "openrouter"). These are NOT
        # directly-routable providers: emitting them as their own picker
        # row produces a phantom entry that, when selected, resolves via
        # resolve_provider_full() to the aggregator (OpenRouter) — silently
        # switching a user off their real provider onto an endpoint they
        # may have no key for (HTTP 401). The user's real provider (e.g.
        # openai-api, or a providers.openai config row) covers this vendor.
        _alias_target = _PROVIDER_ALIAS_TABLE.get(hermes_id)
        if (
            _alias_target
            and _alias_target != hermes_id
            and _alias_target in _AGG_PROVIDERS
        ):
            continue
        # Resolve the canonical provider profile name.  Skip hermes_ids
        # that are mere aliases resolving to a different canonical profile
        # (e.g. "kimi" and "moonshot" both → "kimi-coding").  Only process
        # entries whose hermes_id matches the canonical profile name so
        # distinct profiles (e.g. kimi-coding, kimi-coding-cn) each get
        # their own picker row.
        _canonical = hermes_id
        try:
            from providers import get_provider_profile as _gpp
            _prof = _gpp(hermes_id)
            if _prof is not None:
                _canonical = _prof.name
        except Exception:
            pass
        if _canonical != hermes_id:
            continue

        # Skip duplicates: another entry with the same slug was already
        # emitted (e.g. two PROVIDER_TO_MODELS_DEV entries routing to the
        # same hermes_id).  Distinct canonical profiles that share a
        # models.dev ID (e.g. kimi-coding and kimi-coding-cn → kimi-for-coding)
        # are both allowed through since they have different slugs.
        slug = hermes_id
        if slug.lower() in seen_slugs:
            continue
        if hermes_id.lower() in _excluded or mdev_id.lower() in _excluded:
            continue
        pdata = data.get(mdev_id)
        if not isinstance(pdata, dict):
            continue

        # Prefer auth.py PROVIDER_REGISTRY for env var names — it's our
        # source of truth.  models.dev can have wrong mappings (e.g.
        # minimax-cn → MINIMAX_API_KEY instead of MINIMAX_CN_API_KEY).
        pconfig = PROVIDER_REGISTRY.get(hermes_id)
        # Skip non-API-key auth providers here — they are handled in
        # section 2 (HERMES_OVERLAYS) with proper auth store checking.
        if pconfig and pconfig.auth_type != "api_key":
            continue
        # models.dev catalogs include providers Hermes may not route yet.
        # Gate on runtime capability rather than registry membership: special
        # providers and plugin aliases can be routable without a registry row.
        from hermes_cli.auth import is_runtime_provider_routable
        if not is_runtime_provider_routable(hermes_id):
            continue
        if pconfig and pconfig.api_key_env_vars:
            env_vars = list(pconfig.api_key_env_vars)
        else:
            env_vars = pdata.get("env", [])
            if not isinstance(env_vars, list):
                continue

        # Check if any env var is set
        has_creds = any(os.environ.get(ev) for ev in env_vars)
        if not has_creds:
            try:
                from hermes_cli.auth import _load_auth_store
                store = _load_auth_store()
                raw_pool_present = bool(
                    store and store.get("credential_pool", {}).get(hermes_id)
                )
                if raw_pool_present:
                    has_creds = _credential_pool_is_usable(
                        hermes_id, raw_pool_present=True
                    )
            except Exception:
                pass
        if not has_creds:
            continue

        # Unified pathway: route through cached_provider_model_ids() so the
        # /model picker sees the SAME list `hermes model` would build, with
        # disk caching to keep the picker open snappy. Falls back to the
        # curated static list when the live fetcher returns nothing.
        model_ids = cached_provider_model_ids(hermes_id)
        if not model_ids:
            model_ids = curated.get(hermes_id, [])
            if hermes_id in _MODELS_DEV_PREFERRED:
                model_ids = _merge_with_models_dev(hermes_id, model_ids)
        # A providers.<built-in>.models block extends the provider's discovered
        # catalog. Section 3 cannot emit it later because this built-in row owns
        # the slug, so merge declarations here before applying max_models.
        configured_models: list[str] = []
        if isinstance(user_providers, dict):
            configured = user_providers.get(hermes_id)
            if isinstance(configured, dict):
                configured_models = _declared_model_ids(configured.get("models"))
        model_ids = list(dict.fromkeys([*configured_models, *model_ids]))
        total = len(model_ids)
        if hermes_id in _UNCAPPED_PICKER_PROVIDERS:
            top = model_ids  # Aggregator: show full catalog regardless of max_models
        else:
            top = model_ids[:max_models] if max_models is not None else model_ids

        pinfo = _mdev_pinfo(mdev_id)
        display_name = pconfig.name if pconfig and pconfig.name else (pinfo.name if pinfo else mdev_id)

        results.append({
            "slug": slug,
            "name": display_name,
            "is_current": (
                slug == current_provider
                or hermes_id == current_provider
                or mdev_id == current_provider
            ),
            "is_user_defined": False,
            "models": top,
            "total_models": total,
            "source": "built-in",
        })
        seen_slugs.add(slug.lower())
        _record_builtin_endpoint(slug)

    # --- 2. Check Hermes-only providers (nous, openai-codex, copilot, opencode-go) ---
    from hermes_cli.providers import HERMES_OVERLAYS
    from hermes_cli.auth import PROVIDER_REGISTRY as _auth_registry

    # Build reverse mapping: models.dev ID → Hermes provider ID.
    # HERMES_OVERLAYS keys may be models.dev IDs (e.g. "github-copilot")
    # while _PROVIDER_MODELS and config.yaml use Hermes IDs ("copilot").
    _mdev_to_hermes = {v: k for k, v in PROVIDER_TO_MODELS_DEV.items()}

    for pid, overlay in HERMES_OVERLAYS.items():
        if pid.lower() in seen_slugs:
            continue

        # Resolve Hermes slug — e.g. "github-copilot" → "copilot"
        hermes_slug = _mdev_to_hermes.get(pid, pid)
        if hermes_slug.lower() in seen_slugs:
            continue
        if pid.lower() in _excluded or hermes_slug.lower() in _excluded:
            continue

        # Check if credentials exist
        has_creds = False
        if getattr(overlay, "keyless", False):
            # Keyless providers (opencode-free) are served anonymously —
            # there is no credential to check, so everyone is authenticated.
            has_creds = True
        elif overlay.auth_type == "aws_sdk":
            has_creds = _has_aws_sdk_creds_for_listing(hermes_slug)
        elif overlay.auth_type == "vertex":
            # Vertex authenticates via OAuth2 (service-account JSON / ADC),
            # not an API key — mirror the aws_sdk gate above, otherwise the
            # provider is silently hidden from the /model picker even when
            # fully configured.
            try:
                from agent.vertex_adapter import has_vertex_credentials
                has_creds = has_vertex_credentials()
            except Exception as exc:
                logger.debug("Vertex credential check failed: %s", exc)
        elif overlay.extra_env_vars:
            has_creds = any(os.environ.get(ev) for ev in overlay.extra_env_vars)
        # Also check api_key_env_vars from PROVIDER_REGISTRY for api_key auth_type
        if not has_creds and overlay.auth_type == "api_key":
            for _key in (pid, hermes_slug):
                pcfg = _auth_registry.get(_key)
                if pcfg and pcfg.api_key_env_vars:
                    if any(os.environ.get(ev) for ev in pcfg.api_key_env_vars):
                        has_creds = True
                        break
        # Check auth store and credential pool for non-env-var credentials.
        # This applies to OAuth providers AND api_key providers that also
        # support OAuth (e.g. anthropic supports both API key and Claude Code
        # OAuth via external credential files).
        if not has_creds:
            try:
                from hermes_cli.auth import _load_auth_store
                store = _load_auth_store()
                providers_store = store.get("providers", {})
                if store and (pid in providers_store or hermes_slug in providers_store):
                    has_creds = True
            except Exception as exc:
                logger.debug("Auth store check failed for %s: %s", pid, exc)
        # Fallback: check the credential pool with full auto-seeding.
        # This catches credentials that exist in external stores (e.g.
        # Codex CLI ~/.codex/auth.json) which _seed_from_singletons()
        # imports on demand but aren't in the raw auth.json yet.
        if not has_creds:
            try:
                if _credential_pool_is_usable(hermes_slug):
                    has_creds = True
                elif for_picker:
                    # For the interactive /model picker, also show providers
                    # whose credential pool has entries but all are temporarily
                    # rate-limited.  Rate limits are per-model for many
                    # providers (e.g. Google Gemini) — switching to a different
                    # model under the same provider may work even when all keys
                    # are in cooldown.
                    try:
                        from agent.credential_pool import load_pool
                        _pool = load_pool(hermes_slug)
                        if _pool.has_credentials():
                            has_creds = True
                    except Exception:
                        pass
            except Exception as exc:
                logger.debug("Credential pool check failed for %s: %s", hermes_slug, exc)
        # Fallback: check external credential files directly.
        # The credential pool gates anthropic behind
        # is_provider_explicitly_configured() to prevent auxiliary tasks
        # from silently consuming Claude Code tokens (PR #4210).
        # But the /model picker is discovery-oriented — we WANT to show
        # providers the user can switch to, even if they aren't currently
        # configured.
        if not has_creds and hermes_slug == "anthropic":
            try:
                from agent.anthropic_adapter import (
                    read_claude_code_credentials,
                    read_hermes_oauth_credentials,
                )
                hermes_creds = read_hermes_oauth_credentials()
                cc_creds = read_claude_code_credentials()
                if (hermes_creds and hermes_creds.get("accessToken")) or \
                   (cc_creds and cc_creds.get("accessToken")):
                    has_creds = True
            except Exception as exc:
                logger.debug("Anthropic external creds check failed: %s", exc)
        if not has_creds:
            continue

        if hermes_slug in {"openai-codex", "copilot", "copilot-acp"}:
            # Use live OAuth-backed discovery so the gateway /model picker
            # matches what the user's authenticated Codex/Copilot backend
            # actually serves — including ChatGPT-Pro-only Codex slugs
            # (e.g. gpt-5.3-codex-spark) that aren't in the static curated
            # catalog. ``cached_provider_model_ids()`` falls back to the
            # curated list when the live endpoint is unreachable, so this
            # is safe for unauthenticated and offline cases too.
            model_ids = cached_provider_model_ids(hermes_slug)
        # For aws_sdk providers (bedrock), use live discovery so the list
        # reflects the active region (eu.*, ap.*) not the static us.* list.
        elif overlay.auth_type == "aws_sdk":
            try:
                _ids = cached_provider_model_ids(hermes_slug)
                model_ids = _ids if _ids else (curated.get(hermes_slug, []) or curated.get(pid, []))
            except Exception:
                model_ids = curated.get(hermes_slug, []) or curated.get(pid, [])
        elif hermes_slug == "nous":
            # Nous serves a large live /v1/models catalog (vendor-prefixed
            # models from many providers, returned alphabetically). The
            # `hermes model` picker deliberately shows ONLY the curated agentic
            # list — augmented with the Portal's free/paid recommendations so
            # newly-launched models surface without a CLI release — in curated
            # order. Mirror that exactly (see _model_flow_nous in main.py) so
            # the GUI picker matches the CLI. Was: falling through to
            # cached_provider_model_ids, which dumped the full alphabetical
            # catalog; then: curated-only, which dropped the 4 Portal
            # recommendations (e.g. stepfun/step-3.7-flash:free).
            model_ids = curated.get("nous", [])
            try:
                from hermes_cli.models import (
                    get_pricing_for_provider as _nous_pricing,
                    check_nous_free_tier as _nous_free,
                    union_with_portal_free_recommendations as _union_free,
                    union_with_portal_paid_recommendations as _union_paid,
                )
                from hermes_cli.auth import get_provider_auth_state as _nous_state

                _pricing = _nous_pricing("nous") or {}
                _portal = ""
                try:
                    _st = _nous_state("nous") or {}
                    _portal = _st.get("portal_base_url", "") or ""
                except Exception:
                    _portal = ""
                if _nous_free(force_fresh=force_fresh_nous_tier):
                    model_ids, _ = _union_free(model_ids, _pricing, _portal)
                else:
                    model_ids, _ = _union_paid(model_ids, _pricing, _portal)
            except Exception:
                # Portal recommendation fetch failed — fall back to the
                # curated list alone (still correct, just may lag newly
                # launched models, exactly like an offline CLI run).
                pass
        else:
            # Unified pathway — see Section 1 rationale. Fall back to the
            # curated dict (with models.dev merge for preferred providers)
            # when the live fetcher comes up empty.
            model_ids = cached_provider_model_ids(hermes_slug)
            if not model_ids:
                model_ids = curated.get(hermes_slug, []) or curated.get(pid, [])
                if hermes_slug in _MODELS_DEV_PREFERRED:
                    model_ids = _merge_with_models_dev(hermes_slug, model_ids)
        total = len(model_ids)
        if hermes_slug in _UNCAPPED_PICKER_PROVIDERS:
            top = model_ids  # Aggregator: show full catalog regardless of max_models
        else:
            top = model_ids[:max_models] if max_models is not None else model_ids

        results.append({
            "slug": hermes_slug,
            "name": get_label(hermes_slug),
            "is_current": hermes_slug == current_provider or pid == current_provider,
            "is_user_defined": False,
            "models": top,
            "total_models": total,
            "source": "hermes",
        })
        seen_slugs.add(pid.lower())
        seen_slugs.add(hermes_slug.lower())
        _record_builtin_endpoint(hermes_slug)

    # --- 2b. Cross-check canonical provider list ---
    # Catches providers that are in CANONICAL_PROVIDERS but weren't found
    # in PROVIDER_TO_MODELS_DEV or HERMES_OVERLAYS (keeps /model in sync
    # with `hermes model`).
    try:
        from hermes_cli.models import CANONICAL_PROVIDERS as _canon_provs
    except ImportError:
        _canon_provs = []

    for _cp in _canon_provs:
        if _cp.slug.lower() in seen_slugs:
            continue
        if _cp.slug.lower() in _excluded:
            continue

        # Check credentials via PROVIDER_REGISTRY (auth.py)
        _cp_config = _auth_registry.get(_cp.slug)
        _cp_has_creds = False
        if _cp_config and _cp_config.api_key_env_vars:
            _cp_has_creds = any(os.environ.get(ev) for ev in _cp_config.api_key_env_vars)
        # Also check auth store and credential pool
        if not _cp_has_creds:
            try:
                from hermes_cli.auth import _load_auth_store
                _cp_store = _load_auth_store()
                _cp_providers_store = _cp_store.get("providers", {})
                if _cp_store and _cp.slug in _cp_providers_store:
                    _cp_has_creds = True
            except Exception:
                pass
        if not _cp_has_creds:
            try:
                if _credential_pool_is_usable(_cp.slug):
                    _cp_has_creds = True
            except Exception:
                pass

        # Special case: aws_sdk auth (bedrock) — no API key env vars,
        # credentials come from the boto3 credential chain (env vars,
        # ~/.aws/credentials, instance roles, etc.)
        if not _cp_has_creds and _cp_config and getattr(_cp_config, "auth_type", "") == "aws_sdk":
            _cp_has_creds = _has_aws_sdk_creds_for_listing(_cp.slug)

        if not _cp_has_creds:
            continue

        # For bedrock, use live discovery so the list reflects the active
        # region (eu.*, us.*, ap.*) instead of the hardcoded us.* static list.
        if _cp_config and getattr(_cp_config, "auth_type", "") == "aws_sdk":
            try:
                _ids = cached_provider_model_ids(_cp.slug)
                _cp_model_ids = _ids if _ids else curated.get(_cp.slug, [])
            except Exception:
                _cp_model_ids = curated.get(_cp.slug, [])
        else:
            # Unified pathway — same as sections 1 and 2.
            _cp_model_ids = cached_provider_model_ids(_cp.slug)
            if not _cp_model_ids:
                _cp_model_ids = curated.get(_cp.slug, [])
        _cp_total = len(_cp_model_ids)
        _cp_top = _cp_model_ids[:max_models] if max_models is not None else _cp_model_ids

        results.append({
            "slug": _cp.slug,
            "name": _cp.label,
            "is_current": _cp.slug == current_provider,
            "is_user_defined": False,
            "models": _cp_top,
            "total_models": _cp_total,
            "source": "canonical",
        })
        seen_slugs.add(_cp.slug.lower())
        _record_builtin_endpoint(_cp.slug)

    # --- 3. User-defined endpoints from config ---
    # Track (name, base_url) of what section 3 emits so section 4 can skip
    # any overlapping ``custom_providers:`` entries.  Callers typically pass
    # both (gateway/CLI invoke ``get_compatible_custom_providers()`` which
    # merges ``providers:`` into the list) — without this, the same endpoint
    # produces two picker rows: one bare-slug ("openrouter") from section 3
    # and one "custom:openrouter" from section 4, both labelled identically.
    _section3_emitted_pairs: set = set()
    if user_providers and isinstance(user_providers, dict):
        # Group ``providers:`` entries by (api_url, key_env, api_mode) so that
        # multiple keyed providers pointing at the same endpoint with the
        # same credential and wire-protocol collapse into one picker row.
        # Mirrors section-4's grouping for ``custom_providers:`` lists.
        # Concrete case: a Palantir Foundry Anthropic-proxy with two
        # configured models (claude-4.6 + claude-4.7) — both share the same
        # api/key_env/api_mode and used to produce two near-duplicate rows
        # labelled "Palantir Claude 4.6 Opus" and "Palantir Claude 4.7 Opus";
        # now they appear as a single "Palantir Claude" row with both models
        # in the dropdown. Same-host entries with different ``key_env`` or
        # ``api_mode`` (e.g. an OpenAI-compat gpt-5.4 alongside the Anthropic
        # claude-4.7 on the same Palantir host) keep distinct rows since
        # the wire protocol differs.
        from collections import OrderedDict as _OD3

        from hermes_cli.config import is_provider_enabled

        ep_groups: "_OD3[tuple, dict]" = _OD3()
        for ep_name, ep_cfg in user_providers.items():
            if not isinstance(ep_cfg, dict):
                continue
            # Honour explicit ``providers.<name>.enabled: false`` from
            # config — these are hidden from the picker.
            if not is_provider_enabled(ep_cfg):
                continue
            if ep_name.lower() in seen_slugs:
                continue
            display_name = coerce_provider_id(ep_cfg.get("name")) or ep_name
            api_url = (
                ep_cfg.get("base_url", "")
                or ep_cfg.get("api", "")
                or ep_cfg.get("url", "")
                or ""
            )
            key_env = str(
                ep_cfg.get("key_env") or ep_cfg.get("api_key_env") or ""
            ).strip()
            inline_api_key = str(ep_cfg.get("api_key", "") or "").strip()
            api_mode = str(
                ep_cfg.get("api_mode")
                or ep_cfg.get("transport")
                or ""
            ).strip().lower() or None
            credential_identity = (
                inline_api_key
                if inline_api_key
                else (f"env:{key_env}" if key_env else "")
            )
            api_url_norm = str(api_url).strip().rstrip("/").lower()
            # Per-provider extra_headers participate in the group identity
            # (same invariant as section 4): two entries sharing
            # (api_url, credential, api_mode) but declaring different headers
            # are distinct endpoints (e.g. different tenants behind one proxy
            # URL, routed by header) and must keep distinct picker rows.
            entry_extra_headers = _extra_headers_from_config(ep_cfg)
            headers_identity = tuple(sorted(entry_extra_headers.items()))
            group_key = (api_url_norm, credential_identity, api_mode, headers_identity)

            # ``default_model`` is the legacy key; ``model`` matches what
            # custom_providers entries use, so accept either.
            default_model = ep_cfg.get("default_model", "") or ep_cfg.get("model", "")
            # Build models list from both default_model and full models array.
            # Hermes writes ``models:`` as a dict keyed by model id, but older
            # or hand-edited configs may use strings or ``[{id: ...}]`` rows —
            # _declared_model_ids() owns that contract.
            entry_models: list = []
            if default_model:
                entry_models.append(default_model)
            entry_declared_models = _declared_model_ids(ep_cfg.get("models", []))
            for model_id in entry_declared_models:
                if model_id not in entry_models:
                    entry_models.append(model_id)

            if group_key not in ep_groups:
                # Strip per-model suffix so "Palantir Claude 4.7 Opus" becomes
                # "Palantir Claude". Em dash and " - " are the separators
                # Hermes's own writer uses (mirrors section-4 grouping).
                grp_display = display_name
                for sep in ("—", " - "):
                    if sep in grp_display:
                        grp_display = grp_display.split(sep)[0].strip()
                        break
                # Drop trailing numeric/version tokens that distinguish per-model
                # entries ("Palantir Claude 4.7 Opus" → "Palantir Claude").
                # Keeps the row label short; the model dropdown carries the
                # per-version detail. Heuristic: split at the first token whose
                # stripped form contains a digit; keep the prefix only if it
                # is at least 2 words (avoids over-trimming single-word names).
                _toks = grp_display.split()
                _cut_at = None
                for _i, _t in enumerate(_toks):
                    _tl = _t.strip(".,()")
                    if _tl and any(c.isdigit() for c in _tl):
                        _cut_at = _i
                        break
                if _cut_at is not None and _cut_at >= 2:
                    grp_display = " ".join(_toks[:_cut_at]).strip()
                grp_slug = ep_name  # primary slug is the first ep_name encountered
                ep_groups[group_key] = {
                    "slug": grp_slug,
                    "name": grp_display or display_name,
                    "api_url": api_url,
                    "models": [],
                    "has_explicit_models": False,
                    "ep_cfg": ep_cfg,  # used below for discover_models / api_key
                    # Part of group_key, so it is constant across the group.
                    # The render loop below needs it to key the model cache:
                    # api_mode changes the wire protocol (``x-api-key`` vs
                    # ``Authorization: Bearer``), so two rows that differ only
                    # by it must not share a cached catalog.
                    "api_mode": api_mode,
                    "raw_names": [],
                    "aliases": set(),
                }
            # Aggregate models across all members of the group (preserve order).
            for _m in entry_models:
                if _m and _m not in ep_groups[group_key]["models"]:
                    ep_groups[group_key]["models"].append(_m)
            # Track allowlist-shaped ``models:`` separately from the merged
            # list: a singular ``default_model``/``model`` is only the active
            # selection and must not suppress discovery (see #40542 / PR
            # #61928). Dict-shaped ``models:`` is context_length metadata from
            # ``hermes model``, not an allowlist — see
            # ``_models_config_is_allowlist``.
            if _models_config_is_allowlist(
                ep_cfg.get("models"), _entry_models_discovered(ep_cfg)
            ):
                ep_groups[group_key]["has_explicit_models"] = True
            ep_groups[group_key]["raw_names"].append(display_name)
            ep_groups[group_key]["aliases"].update(
                custom_provider_aliases(display_name, str(ep_name))
            )

        for grp in ep_groups.values():
            ep_cfg = grp["ep_cfg"]
            ep_name = grp["slug"]
            display_name = grp["name"]
            api_url = grp["api_url"]
            models_list = list(grp["models"])

            # Official OpenAI API rows in providers: often have base_url but no
            # explicit models: dict — avoid a misleading zero count in /model.
            if not models_list:
                url_lower = str(api_url).strip().lower()
                if base_url_host_matches(url_lower, "api.openai.com"):
                    fb = curated.get("openai") or []
                    if fb:
                        models_list = list(fb)

            # Prefer the endpoint's live /models list when discoverable,
            # unless the provider explicitly opts out via discover_models: false.
            # Policy mirrors Section 4's should_probe logic:
            # - With an api_key: always probe (user opted into the endpoint).
            # - Without an api_key but with an allowlist-shaped ``models:``
            #   (list/string): skip — the user narrowed a public endpoint.
            #   A singular ``default_model``/``model`` does NOT count as
            #   narrowing (mirrors section 4 / #40542).
            # - A dict-shaped ``models:`` is per-model metadata
            #   (context_length), not an allowlist — still probe so local
            #   Ollama/llama.cpp match ``hermes model``. Pin with
            #   ``discover_models: false`` instead.
            # - Without an api_key AND no allowlist: probe anyway so bare
            #   local endpoints still show their full model catalog.
            api_key = str(ep_cfg.get("api_key", "") or "").strip()
            if not api_key:
                key_env = str(
                    ep_cfg.get("key_env") or ep_cfg.get("api_key_env") or ""
                ).strip()
                api_key = _scoped_key_env(key_env) if key_env else ""
            discover = ep_cfg.get("discover_models", True)
            if isinstance(discover, str):
                discover = discover.lower() not in {"false", "no", "0"}
            has_explicit_models = bool(grp.get("has_explicit_models"))
            _ep_url_norm = str(api_url).strip().rstrip("/").lower()
            _ep_slug_norm = str(ep_name).strip().lower()
            _ep_aliases = {
                str(alias).lower() for alias in grp.get("aliases", set())
            }
            _ep_is_current = (
                _ep_slug_norm == _current_provider_norm
                or _current_provider_norm in _ep_aliases
                or (
                    _current_provider_norm == "custom"
                    and bool(_current_base_url_norm)
                    and _ep_url_norm == _current_base_url_norm
                )
            )
            # See section 4: when live probing is suppressed for latency, a
            # warm same-fingerprint cache entry still serves the full catalog
            # with no network round-trip.
            #
            # ``has_explicit_models`` gates the *probe*, not the cache read:
            # it exists so a keyless endpoint with a declared catalog is not
            # hammered over the network (5f00f36ba, 1039e90b5). Reading a
            # catalog an earlier probe already paid for costs nothing, and
            # applying the probe gate to it re-pins the endpoint — see
            # ``_discovery_allowed`` in section 4 for the full rationale.
            _discovery_allowed = bool(api_url) and discover
            _probe_live = (
                _discovery_allowed
                and (bool(api_key) or not has_explicit_models)
                and _can_probe_custom_provider(row_is_current=_ep_is_current)
            )
            native_catalog_empty = False
            if _probe_live:
                try:
                    native_catalog_provider = (
                        ep_name
                        if str(ep_name).strip().lower()
                        in {"ollama", "custom:ollama"}
                        else "custom"
                    )
                    live_models = _fetch_picker_live_models(
                        api_key,
                        api_url,
                        native_catalog_provider,
                        has_explicit_models,
                        headers=_extra_headers_from_config(ep_cfg) or None,
                        timeout=(1.5 if for_picker else 5.0),
                        api_mode=ep_cfg.get("api_mode"),
                    )
                    if isinstance(live_models, _NativePickerModelList):
                        native_catalog_empty = not live_models
                    if live_models is not None and (
                        live_models
                        or not has_explicit_models
                        or isinstance(live_models, _NativePickerModelList)
                    ):
                        models_list = live_models
                except Exception:
                    pass
            elif _discovery_allowed:
                try:
                    from hermes_cli.models import cached_fetch_api_models

                    cached_models = cached_fetch_api_models(
                        api_key,
                        api_url,
                        cache_only=True,
                        timeout=(1.5 if for_picker else 5.0),
                        headers=_extra_headers_from_config(ep_cfg) or None,
                        api_mode=ep_cfg.get("api_mode"),
                    )
                    if cached_models:
                        models_list = cached_models
                except _MODEL_DISCOVERY_ERRORS:
                    pass

            results.append({
                "slug": ep_name,
                "name": display_name,
                "is_current": _ep_is_current,
                "is_user_defined": True,
                "models": models_list,
                "total_models": len(models_list) if models_list else 0,
                "source": "user-config",
                "api_url": api_url,
                "native_catalog_empty": native_catalog_empty,
            })
            seen_slugs.add(ep_name.lower())
            seen_slugs.update(_ep_aliases)
            # Record (display_name, api_url) for each raw entry that joined
            # this group so section-4's _section3_emitted_pairs dedup can
            # match per-model custom_providers rows ("Palantir Claude 4.7 Opus")
            # even though we collapsed the group label to "Palantir Claude".
            _url_norm_for_pair = str(api_url).strip().rstrip("/").lower()
            for _raw_name in grp.get("raw_names") or [display_name]:
                _pair = (
                    str(_raw_name).strip().lower(),
                    _url_norm_for_pair,
                )
                if _pair[0] and _pair[1]:
                    _section3_emitted_pairs.add(_pair)
                    seen_slugs.add(custom_provider_slug(_raw_name).lower())
            _pair = (
                str(display_name).strip().lower(),
                _url_norm_for_pair,
            )
            if _pair[0] and _pair[1]:
                _section3_emitted_pairs.add(_pair)

    # --- 3b. Active bare custom endpoint from model config ---
    # A config can still use the direct one-off form:
    #   model.provider: custom
    #   model.base_url: https://some-openai-compatible/v1
    # In that shape there is no named providers:/custom_providers row for the
    # picker to render, but the gateway only passes this current model slice to
    # list_authenticated_providers(). Surface the active endpoint explicitly so
    # /model does not look like it ignored config.yaml.
    if (
        _current_provider_norm == "custom"
        and current_base_url
        and "custom" not in seen_slugs
        and not any(
            isinstance(_cp, dict)
            and str(
                _cp.get("base_url", "")
                or _cp.get("url", "")
                or _cp.get("api", "")
            ).strip().rstrip("/").lower()
            == str(current_base_url).strip().rstrip("/").lower()
            for _cp in (custom_providers or [])
        )
    ):
        _models = [current_model] if current_model else []
        # With live probing suppressed, use the shared stale/cache path;
        # otherwise probe through the native-aware picker helper.
        native_catalog_empty = False
        _probe_live = bool(refresh or probe_current_custom_provider)
        try:
            if _probe_live:
                _live_models = _fetch_picker_live_models(
                    "",
                    str(current_base_url).strip().rstrip("/"),
                    "custom",
                    False,
                    timeout=(1.5 if for_picker else 5.0),
                )
            else:
                from hermes_cli.models import cached_fetch_api_models

                _live_models = cached_fetch_api_models(
                    "",
                    str(current_base_url).strip().rstrip("/"),
                    cache_only=True,
                    timeout=(1.5 if for_picker else 5.0),
                )
            if _live_models is not None:
                native_catalog_empty = isinstance(
                    _live_models, _NativePickerModelList
                ) and not _live_models
                _models = _live_models
        except Exception:
            pass
        results.append({
            "slug": "custom",
            "name": "Custom endpoint",
            "is_current": True,
            "is_user_defined": True,
            "models": _models[:max_models] if max_models is not None else _models,
            "total_models": len(_models),
            "source": "model-config",
            "api_url": str(current_base_url).strip().rstrip("/"),
            "native_catalog_empty": native_catalog_empty,
        })
        seen_slugs.add("custom")

    # --- 4. Saved custom providers from config ---
    # Each ``custom_providers`` entry represents one model under a named
    # provider. Entries sharing the same endpoint, credential identity, and
    # wire protocol are grouped into a single picker row, so e.g. four Ollama
    # entries pointing at ``http://localhost:11434/v1`` with per-model display
    # names ("Ollama — GLM 5.1", "Ollama — Qwen3-coder", ...) appear as one
    # "Ollama" row with four models inside instead of four near-duplicates
    # that differ only by suffix. Same-host entries with different ``key_env``
    # or ``api_mode`` remain distinct providers.
    if custom_providers and isinstance(custom_providers, list):
        from collections import OrderedDict

        # Key by endpoint + credential identity + wire protocol + display
        # prefix instead of slug: names frequently differ per model
        # ("Ollama — X") while the endpoint stays the same.  Keep same-host
        # providers with distinct env-backed credentials or API protocols
        # separate so picker selection cannot route through the wrong
        # credential/mode pair. The display prefix (text before " — " /
        # " - ") is included so intentionally distinct providers sharing an
        # endpoint (e.g. a proxy fronting cerebras, groq and perplexity at
        # a single base_url) each get their own picker row instead of
        # collapsing into one. Per-model suffix entries that share the same
        # prefix ("Ollama — A", "Ollama — B") still group together.
        groups: "OrderedDict[tuple, dict]" = OrderedDict()
        for entry in custom_providers:
            if not isinstance(entry, dict):
                continue

            raw_name = coerce_provider_id(entry.get("name"))
            api_url = str(
                entry.get("base_url", "")
                or entry.get("url", "")
                or entry.get("api", "")
                or ""
            ).strip().rstrip("/")
            if not raw_name or not api_url:
                continue
            inline_api_key = str(entry.get("api_key") or "").strip()
            key_env = str(entry.get("key_env") or "").strip()
            api_key = inline_api_key or _scoped_key_env(key_env)
            api_mode = str(
                entry.get("api_mode")
                or entry.get("transport")
                or ""
            ).strip().lower() or None
            credential_identity = (
                inline_api_key
                if inline_api_key
                else (f"env:{key_env}" if key_env else "")
            )

            # Read discover_models from the entry (same semantics as
            # section 3: true by default, set false to keep the explicit
            # ``models:`` list instead of replacing it with live /models).
            discover = entry.get("discover_models", True)
            if isinstance(discover, str):
                discover = discover.lower() not in {"false", "no", "0"}

            # Per-provider extra_headers participate in the group identity:
            # two entries sharing (api_url, credential, api_mode) but declaring
            # different headers are distinct endpoints (e.g. different tenants
            # behind one proxy URL, routed by header) and must probe /models
            # with their own headers rather than collapsing into one row and
            # silently adopting whichever header set was seen first.
            entry_extra_headers = _extra_headers_from_config(entry)
            headers_identity = tuple(sorted(entry_extra_headers.items()))

            # Display-name prefix (text before " — " / " - "), used both
            # as a grouping dimension and to derive the row's display name.
            _display_prefix = raw_name
            for sep in ("—", " - "):
                if sep in _display_prefix:
                    _display_prefix = _display_prefix.split(sep)[0].strip()
                    break

            group_key = (api_url, credential_identity, api_mode, headers_identity, _display_prefix.lower())
            if group_key not in groups:
                # Reuse the prefix computed above as the row display name;
                # fall back to the raw name if stripping left it empty.
                display_name = _display_prefix or raw_name
                provider_key = str(entry.get("provider_key") or "").strip()
                slug = custom_provider_slug(display_name, provider_key)
                groups[group_key] = {
                    "slug": slug,
                    "name": display_name,
                    "api_url": api_url,
                    "api_key": api_key,
                    "models": [],
                    "has_explicit_models": False,
                    "discover_models": discover,
                    "api_mode": api_mode,
                    "extra_headers": entry_extra_headers,
                    # Part of group_key, so constant across the group. Needed
                    # in the render loop to key the model cache — api_mode
                    # selects the wire protocol, so rows differing only by it
                    # must not share a cached catalog.
                    "api_mode": api_mode,
                    "aliases": set(),
                }
            else:
                if api_key and not groups[group_key].get("api_key"):
                    groups[group_key]["api_key"] = api_key
                # extra_headers is part of group_key, so every entry in this
                # group already carries identical headers — nothing to merge.
                # If any entry in this group opts out of discovery,
                # honour that for the whole grouped row.
                if not discover:
                    groups[group_key]["discover_models"] = False
            groups[group_key]["aliases"].update(
                custom_provider_aliases(
                    raw_name,
                    str(entry.get("provider_key") or ""),
                )
            )

            # The singular ``model:`` field only holds the currently
            # active model. Hermes's own writer (main.py::_save_custom_provider)
            # stores every configured model as a dict under ``models:``;
            # downstream readers (agent/models_dev.py, gateway/run.py,
            # run_agent.py, hermes_cli/config.py) already consume that dict.
            default_model = (entry.get("model") or "").strip()
            if default_model and default_model not in groups[group_key]["models"]:
                groups[group_key]["models"].append(default_model)

            models_field = entry.get("models", {})
            declared_models = _declared_model_ids(models_field)
            # Dict-shaped models: is context_length metadata from
            # ``_save_custom_provider``, not an allowlist — see
            # ``_models_config_is_allowlist``.
            if _models_config_is_allowlist(
                models_field, _entry_models_discovered(entry)
            ):
                groups[group_key]["has_explicit_models"] = True
            for model_id in declared_models:
                if model_id not in groups[group_key]["models"]:
                    groups[group_key]["models"].append(model_id)

        _section4_emitted_slugs: set = set()
        _current_base_url_group_count = sum(
            1
            for _grp in groups.values()
            if _current_base_url_norm
            and str(_grp["api_url"]).strip().rstrip("/").lower() == _current_base_url_norm
        )
        for grp in groups.values():
            api_url = grp["api_url"]
            api_key = grp.get("api_key", "")
            slug = grp["slug"]
            # If the slug is already claimed by a built-in / overlay /
            # user-provider row (sections 1-3), skip this custom group
            # to avoid shadowing a real provider.
            if slug.lower() in seen_slugs and slug.lower() not in _section4_emitted_slugs:
                continue
            # If a prior section-4 group already used this slug (two custom
            # endpoints with the same cleaned name — e.g. two OpenAI-
            # compatible gateways named identically with different keys),
            # append a counter so both rows stay visible in the picker.
            if slug.lower() in _section4_emitted_slugs:
                base_slug = slug
                n = 2
                while f"{base_slug}-{n}".lower() in seen_slugs:
                    n += 1
                slug = f"{base_slug}-{n}"
                grp["slug"] = slug
            # Skip if section 3 already emitted this endpoint under its
            # ``providers:`` dict key — matches on (display_name, base_url).
            # Prevents two picker rows labelled identically when callers
            # pass both ``user_providers`` and a compatibility-merged
            # ``custom_providers`` list.
            _pair_key = (
                str(grp["name"]).strip().lower(),
                str(grp["api_url"]).strip().rstrip("/").lower(),
            )
            if _pair_key[0] and _pair_key[1] and _pair_key in _section3_emitted_pairs:
                continue
            # Skip if a built-in row (sections 1/2/2b) already represents this
            # endpoint. Fixes #16970: a user-defined "my-dashscope" pointing at
            # https://coding-intl.dashscope.aliyuncs.com/v1 duplicates the
            # built-in alibaba-coding-plan row whenever DASHSCOPE_API_KEY is
            # set. The built-in row carries the curated model list, correct
            # auth wiring, and canonical slug — keep it and hide the shadow.
            _grp_url_norm = _pair_key[1]
            if _grp_url_norm and _grp_url_norm in _builtin_endpoints:
                continue
            # Live model discovery from custom provider endpoints (matches
            # Section 3 behavior for user ``providers:`` entries).
            # Also probes when no api_key is set (e.g. local llama.cpp /
            # Ollama servers) — the /models endpoint often works without
            # auth.  The CLI's _model_flow_named_custom always probes, so
            # the Telegram/Discord picker should do the same for parity.
            # Live-discovery policy:
            # - With an api_key, the user has explicitly opted into the
            #   endpoint and live /models is the source of truth — replace
            #   the (possibly partial) ``models:`` subset with the full
            #   live catalog (Bifrost / aggregator-gateway case).
            # - Without an api_key but with an allowlist-shaped ``models:``
            #   (list/string), the user narrowed a public endpoint (e.g.
            #   ollama.com). Preserve that list and skip live discovery.
            # - A dict-shaped ``models:`` is per-model metadata written by
            #   ``_save_custom_provider`` for context_length — not an
            #   allowlist. Still probe so Desktop/Telegram match
            #   ``hermes model``. Pin a dict catalog with
            #   ``discover_models: false``.
            # - The singular ``model:`` field is only the current active
            #   selection and must not suppress discovery.
            # - When discover_models: false is set, skip live discovery and
            #   keep the configured ``models:`` list regardless of api_key.
            _grp_is_current = (
                slug.lower() == _current_provider_norm
                or _current_provider_norm in {
                    str(alias).lower()
                    for alias in grp.get("aliases", set())
                }
            ) or (
                _current_provider_norm == "custom"
                and bool(_current_base_url_norm)
                and _grp_url_norm == _current_base_url_norm
                and _current_base_url_group_count == 1
            )
            # Discovery is what the user's config asks for; probing is how we
            # get it. When the caller suppresses live probing for latency, the
            # already-discovered catalog on disk still answers the question
            # without a round-trip — skipping it too is what collapsed a
            # multi-model endpoint to its config-declared subset.
            #
            # ``has_explicit_models`` belongs on the probe side of that line.
            # It is a network-cost gate: don't hammer a keyless endpoint that
            # already declares its catalog (5f00f36ba, 1039e90b5). It is not a
            # user pin — ``discover_models: false`` is the documented way to
            # pin, and it is honored above.
            #
            # Keeping it on the discovery side re-pins the endpoint it was
            # meant to spare, because a successful probe calls
            # ``_save_discovered_models_to_config()``, which writes a plain
            # list — the exact shape ``_models_config_is_allowlist()`` reads
            # back as an explicit allowlist. A keyless local server therefore
            # self-pins on its first probe and can never widen again. f66319097
            # already carved the dict shape out of that trap for the same
            # reason; the list shape is the other door into it.
            _discovery_allowed = bool(api_url) and grp.get("discover_models", True)
            _probe_live = (
                _discovery_allowed
                and (bool(api_key) or not grp.get("has_explicit_models"))
                and _can_probe_custom_provider(row_is_current=_grp_is_current)
            )
            native_catalog_empty = False
            if _probe_live:
                try:
                    native_catalog_provider = (
                        "ollama"
                        if str(slug).strip().lower() == "ollama"
                        or str(grp.get("name") or "").strip().lower() == "ollama"
                        else "custom"
                    )
                    live_models = _fetch_picker_live_models(
                        api_key,
                        api_url,
                        native_catalog_provider,
                        bool(grp.get("has_explicit_models")),
                        headers=grp.get("extra_headers") or None,
                        timeout=(1.5 if for_picker else 5.0),
                        api_mode=grp.get("api_mode"),
                    )
                    if live_models is not None and (
                        live_models
                        or not bool(grp.get("has_explicit_models"))
                        or isinstance(live_models, _NativePickerModelList)
                    ):
                        if isinstance(live_models, _NativePickerModelList):
                            native_catalog_empty = not live_models
                        grp["models"] = live_models
                        grp["total_models"] = len(live_models)
                        _save_discovered_models_to_config(
                            api_url,
                            live_models,
                            api_mode=grp.get("api_mode"),
                            headers=grp.get("extra_headers") or None,
                        )
                except Exception:
                    pass
            elif _discovery_allowed:
                try:
                    from hermes_cli.models import cached_fetch_api_models

                    cached_models = cached_fetch_api_models(
                        api_key,
                        api_url,
                        cache_only=True,
                        timeout=(1.5 if for_picker else 5.0),
                        headers=grp.get("extra_headers") or None,
                        api_mode=grp.get("api_mode"),
                    )
                    if cached_models:
                        grp["models"] = cached_models
                        grp["total_models"] = len(cached_models)
                except _MODEL_DISCOVERY_ERRORS:
                    pass
            results.append({
                "slug": slug,
                "name": grp["name"],
                "is_current": _grp_is_current,
                "is_user_defined": True,
                "models": grp["models"],
                "total_models": len(grp["models"]),
                "source": "user-config",
                "api_url": grp["api_url"],
                "native_catalog_empty": native_catalog_empty,
            })
            seen_slugs.add(slug.lower())
            _section4_emitted_slugs.add(slug.lower())

    # Apply final ``providers.<name>.enabled: false`` post-filter — covers
    # built-in PROVIDER_REGISTRY rows (sections 1-2) which would otherwise
    # bypass the per-section gate. Indexed by lowercase slug AND by
    # ``provider_id`` so PROVIDER_REGISTRY entries that match user-config
    # blocks are filtered consistently.
    try:
        from hermes_cli.config import is_provider_enabled
        if isinstance(user_providers, dict):
            _disabled_slugs = {
                str(name).strip().lower()
                for name, cfg in user_providers.items()
                if isinstance(cfg, dict) and not is_provider_enabled(cfg)
            }
            if _disabled_slugs:
                results = [
                    r for r in results
                    if str(r.get("provider_id", "")).strip().lower() not in _disabled_slugs
                    and str(r.get("slug", "")).strip().lower() not in _disabled_slugs
                ]
    except Exception:
        pass

    # Surface a custom / uncurated model the user selected via the CLI.
    # Each row's model list is its curated/live catalog, so a model the user set
    # with `/model <provider>/<uncurated-name>` would otherwise be invisible in
    # every picker — the main model picker AND the MoA reference/aggregator slot
    # pickers, which read these same rows. Inject it at the front of the current
    # provider's row (matched by slug) so it is selectable and shown. Done as a
    # post-pass so it covers every provider section uniformly, regardless of
    # which branch emitted the row.
    if current_model:
        for _row in results:
            if not _row.get("is_current") or _row.get("native_catalog_empty"):
                continue
            _models = _row.get("models") or []
            if current_model not in _models:
                _row["models"] = [current_model, *_models]
                _row["total_models"] = _row.get("total_models", len(_models)) + 1
            break

    # Sort: current provider first, then by model count descending
    results.sort(key=lambda r: (not r["is_current"], -r["total_models"]))

    return results


def _prepend_moa_picker_provider(providers: List[dict], current_provider: str = "") -> List[dict]:
    """Add the virtual MoA provider row used by interactive model pickers.

    ``list_authenticated_providers()`` only returns real/auth-backed providers.
    The CLI model inventory adds MoA separately so named presets appear next to
    normal providers; gateway pickers call ``list_picker_providers()`` directly,
    so they need the same virtual row here. Reuse the inventory's single row
    builder so the row shape stays defined in one place.
    """
    try:
        from hermes_cli.inventory import _moa_provider_row

        moa_row = _moa_provider_row(current_provider)
        if moa_row is None:
            return providers
        return [moa_row] + [p for p in providers if str(p.get("slug", "")).lower() != "moa"]
    except Exception:
        return providers


def list_picker_providers(
    current_provider: str = "",
    current_base_url: str = "",
    user_providers: dict = None,
    custom_providers: list | None = None,
    max_models: int | None = None,
    current_model: str = "",
    include_moa: bool = False,
    excluded_providers: list | None = None,
) -> List[dict]:
    """Interactive-picker variant of :func:`list_authenticated_providers`.

    Post-processes the base list so the ``/model`` picker (Telegram/Discord
    inline keyboards) only surfaces models that are actually callable in the
    current install:

    - OpenRouter's model list is replaced with the output of
      :func:`hermes_cli.models.fetch_openrouter_models`, which filters the
      curated ``OPENROUTER_MODELS`` snapshot against the live OpenRouter
      catalog.  IDs the live catalog no longer carries drop out, so the
      picker never offers a model the user can't call.
    - Provider rows whose model list ends up empty are dropped, except
      custom endpoints (``is_user_defined=True`` with an ``api_url``) where
      the user may supply their own model set through config.

    All other providers and metadata fields are passed through unchanged.
    The typed ``/model <name>`` path is unaffected -- only the interactive
    picker payload is narrowed.
    """
    from hermes_cli.models import fetch_openrouter_models

    providers = list_authenticated_providers(
        current_provider=current_provider,
        current_base_url=current_base_url,
        user_providers=user_providers,
        custom_providers=custom_providers,
        max_models=max_models,
        current_model=current_model,
        for_picker=True,
        excluded_providers=excluded_providers,
    )
    if include_moa:
        providers = _prepend_moa_picker_provider(providers, current_provider=current_provider)

    filtered: List[dict] = []
    for p in providers:
        slug = str(p.get("slug", "")).lower()
        if slug == "openrouter":
            try:
                live = fetch_openrouter_models()
                live_ids = [mid for mid, _ in live]
            except Exception:
                live_ids = list(p.get("models", []))
            p = dict(p)
            p["models"] = live_ids[:max_models] if max_models is not None else live_ids
            p["total_models"] = len(live_ids)

        has_models = bool(p.get("models"))
        is_custom_endpoint = bool(p.get("is_user_defined")) and bool(p.get("api_url"))
        if not has_models and not is_custom_endpoint:
            continue
        filtered.append(p)

    return filtered
