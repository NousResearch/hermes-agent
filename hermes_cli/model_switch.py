"""Shared model-switching logic for the CLI and gateway /model commands.

Pipeline: parse flags -> canonical model selection -> credentials -> validation ->
metadata lookup -> build result. Known ``provider:model`` prefixes
are canonical model references; provider-native colons remain part of the model ID."""

from __future__ import annotations

import logging
import os
import re
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Optional

from providers import (
    ResolvedProvider, custom_provider_aliases, custom_provider_slug, get_provider_label,
    is_aggregator, list_providers, normalize_provider,
)
from models import (
    AmbiguousModelAliasError,
    MODEL_ALIASES,
    ModelRef,
    parse_configured_provider_ref,
    parse_model_ref,
)
from models.selection import (
    ExplicitAlias,
    ExplicitProviderFacts,
    ExplicitSelectionError,
    explicit_provider_hint,
    select_explicit_model,
)
from agent.model_warnings import nous_hermes_non_agentic_warning
from hermes_cli.providers import (
    LLAMACPP_ALIASES, resolve_provider_full,
)
from models.catalog_static import static_provider_model_ids
from agent.models_dev import (
    ModelMetadata, ModelInfo, query_model_metadata, get_model_info, list_provider_models)
from utils import base_url_host_matches, base_url_hostname, base_url_origin, file_signature
# Re-exported: callers/tests patch hermes_cli.model_switch.<name>.
from application_provider_discovery import list_authenticated_providers


logger = logging.getLogger(__name__)


from models.catalog_configured import (
    declared_model_ids as _declared_model_ids,
    entry_models_discovered as _entry_models_discovered,
    models_config_is_allowlist as _models_config_is_allowlist,
)


def _bare_custom_provider_def(current_base_url: str) -> Optional[ResolvedProvider]:
    """Resolved provider for a direct ``model.provider: custom`` endpoint."""
    base_url = _clean(current_base_url)
    if not base_url:
        return None
    return ResolvedProvider(
        id="custom", display_name="Custom endpoint", api_mode="chat_completions", env_vars=(),
        base_url=base_url, is_aggregator=False, auth_type="api_key", source="model-config")


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


# --- Direct aliases — exact model+provider+base_url for endpoints outside the
# models.dev catalog (Ollama Cloud, local servers). Checked BEFORE catalog
# resolution; loaded from config.yaml ``model_aliases:`` / ``model.aliases``.

@dataclass(frozen=True, slots=True)
class DirectAlias:
    """Exact model identity plus route/credential state for a configured alias."""

    ref: ModelRef
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

        user_providers = cfg.get("providers") if isinstance(cfg.get("providers"), dict) else {}
        custom_providers = cfg.get("custom_providers") if isinstance(cfg.get("custom_providers"), list) else []
        known_ids, named_custom_ids = _model_ref_context(user_providers, custom_providers)
        alias_provider_ids = known_ids | named_custom_ids

        configured_provider_ids: dict[str, str] = {}
        for provider_key, entry in user_providers.items():
            if not isinstance(entry, dict):
                continue
            display_name = str(entry.get("name") or provider_key)
            slug = custom_provider_slug(display_name, str(provider_key))
            for alias in custom_provider_aliases(display_name, str(provider_key)):
                configured_provider_ids[alias] = slug

        def alias_ref(provider: Any, model: Any) -> ModelRef:
            raw_provider = _clean(provider)
            provider_id = configured_provider_ids.get(raw_provider.lower(), raw_provider)
            return ModelRef(provider_id, model)

        user_aliases = cfg.get("model_aliases")
        if isinstance(user_aliases, dict):
            for name, entry in user_aliases.items():
                if isinstance(entry, dict) and entry.get("model", ""):
                    merged[name.strip().lower()] = DirectAlias(
                        ref=alias_ref(entry.get("provider", "custom"), entry.get("model", "")),
                        base_url=entry.get("base_url", ""),
                        api_key=_clean(entry.get("api_key", "")),
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
                            ref=alias_ref(
                                _clean(value.get("provider")) or current_provider or "custom",
                                model,
                            ),
                            base_url=_clean(value.get("base_url", "")),
                            api_key=_clean(value.get("api_key", "")),
                            key_env=_clean(value.get("key_env", "")))
                elif isinstance(value, str) and value.strip():
                    val = value.strip()
                    ref = parse_configured_provider_ref(val, alias_provider_ids)
                    if ref is not None:
                        raw_prefix = val.split("/", 1)[0].strip().lower()
                        configured_id = configured_provider_ids.get(raw_prefix)
                        if configured_id:
                            ref = ModelRef(configured_id, ref.model)
                    merged[key] = DirectAlias(
                        ref=ref or alias_ref(current_provider, val),
                        base_url="",
                    )
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
    return ("custom" if alias.base_url else (alias.ref.provider or "custom")), direct_alias_api_key(alias) or None


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


def _model_ref_context(
    user_providers: Optional[dict] = None,
    custom_providers: Optional[list] = None,
) -> tuple[set[str], set[str]]:
    """Known provider IDs plus configured named-custom IDs for model parsing."""

    known = {str(profile.name or "").strip().lower() for profile in list_providers()}
    known.discard("")
    named_custom: set[str] = set()
    for key, entry in (user_providers or {}).items():
        if not isinstance(entry, dict):
            continue
        raw_key = str(key or "").strip().lower()
        if raw_key:
            known.add(raw_key)
        named_custom.add(
            custom_provider_slug(str(entry.get("name") or key), str(key))
        )
    for entry in (custom_providers or []):
        if not isinstance(entry, dict):
            continue
        name = _clean(entry.get("name"))
        if name:
            named_custom.add(custom_provider_slug(name))
    return known, named_custom


def _configured_provider_ids(
    user_providers: Optional[dict] = None,
    custom_providers: Optional[list] = None,
) -> set[str]:
    configured = {
        str(name).strip().lower()
        for name in (user_providers or {})
        if str(name).strip()
    }
    configured.update(
        custom_provider_slug(_clean(entry.get("name")))
        for entry in (custom_providers or [])
        if isinstance(entry, dict) and _clean(entry.get("name"))
    )
    return configured


def resolve_startup_model_route(
    raw_model: str, *, explicit_provider: str = "", current_provider: str = "",
    user_providers: Optional[dict] = None,
    custom_providers: Optional[list] = None) -> Optional[StartupModelRoute]:
    """Resolve aliases, ``provider:model`` and configured ``provider/model`` input at startup.

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
            return StartupModelRoute(model=direct.ref.model, provider=explicit_provider, base_url=direct.base_url)
        # Same owner as the interactive /model and oneshot paths: credential for the alias HOST.
        # Resolve through the SAME owner the interactive /model and oneshot paths use: a URL-bearing alias
        # must resolve its credential for the alias HOST, never for its provider label — a label like
        # ``anthropic`` on a foreign URL would otherwise reach that provider's explicit-runtime branch and
        # put the live vendor token on the foreign wire (#28660).
        alias_provider, alias_key = direct_alias_runtime_request(direct)
        return StartupModelRoute(
            model=direct.ref.model, provider=alias_provider, base_url=direct.base_url, api_key=alias_key or "")

    if explicit_provider:
        return None
    known_ids, named_custom_ids = _model_ref_context(user_providers, custom_providers)
    qualified = parse_model_ref(
        raw,
        "",
        known_provider_ids=known_ids,
        named_custom_provider_ids=named_custom_ids,
    )
    if qualified.provider:
        return StartupModelRoute(model=qualified.model, provider=qualified.provider)

    if current_provider:
        try:
            from providers import is_routing_aggregator

            if is_routing_aggregator(current_provider):
                from hermes_cli.models import _find_openrouter_slug

                if _find_openrouter_slug(raw):
                    return None
        except Exception:
            pass

    configured = _configured_provider_ids(user_providers, custom_providers)
    configured_ref = parse_configured_provider_ref(raw, configured)
    if configured_ref is None:
        return None

    # Parsing belongs to the model domain; route-key precedence remains here.
    prefix = raw.split("/", 1)[0].strip()
    canonical = normalize_provider(prefix)
    if prefix.lower() in configured:
        provider = prefix
    elif canonical in configured:
        provider = canonical
    else:
        return None
    return None if is_aggregator(canonical) else StartupModelRoute(
        model=configured_ref.model,
        provider=provider,
    )


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
    runtime_kind: str = ""
    request_overrides: Optional[dict] = None
    error_message: str = ""
    warning_message: str = ""
    provider_label: str = ""
    resolved_via_alias: str = ""
    capabilities: Optional[ModelMetadata] = None
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

def _ambiguous_alias_message(err: AmbiguousModelAliasError) -> str:
    """User-facing disambiguation list for an ambiguous alias."""
    shown = err.candidates[:10]
    lines = "\n".join(f"  {i}. {m}" for i, m in enumerate(shown, 1))
    hidden = len(err.candidates) - len(shown)
    more = f"\n  … and {hidden} more" if hidden > 0 else ""
    return (
        f"'{err.alias}' matches {len(err.candidates)} models on "
        f"{err.provider} — not switching automatically:\n{lines}{more}\n"
        f"Pick one with /model <exact-model-name>.")


def _provider_identity(name: str, user_providers: Optional[dict] = None,
                       custom_providers: Optional[list] = None) -> str:
    """Id a provider name routes to, e.g. ``custom:<name>`` for a legacy ``custom_providers``
    entry, so an alias naming it by its bare name compares equal to the resolved provider."""
    pdef = resolve_provider_full(name, user_providers, custom_providers) if name else None
    return pdef.id if pdef is not None else normalize_provider(name or "")


def get_authenticated_provider_slugs(
    current_provider: str = "", user_providers: dict = None, custom_providers: list | None = None
) -> list[str]:
    """Slugs of providers that have credentials (models.dev in-memory cache + disk catalog cache;
    stale catalogs warm in the background, never in this call)."""
    try:
        return [p["slug"] for p in list_authenticated_providers(
            current_provider=current_provider, user_providers=user_providers,
            custom_providers=custom_providers, max_models=0, non_blocking_catalogs=True)]
    except Exception:
        return []


def resolve_display_context_length(
    model: str, provider: str, base_url: str = "", api_key: str = "",
    model_info: Optional[ModelInfo] = None, custom_providers: list | None = None,
    config_context_length: int | None = None, configured_model: str | None = None,
    configured_provider: str | None = None,
    configured_base_url: str | None = None) -> Optional[int]:
    """Context length to show in /model output.

    models.dev reports per-vendor context but provider-enforced limits can be lower (Codex OAuth
    caps gpt-5.5 at 272k), so ``models.metadata.context.get_model_context_length`` is authoritative
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
        from models.metadata.context import get_model_context_length
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
    # Callers (gateway, TUI, CLI) pass both ``providers:`` and the compat ``custom_providers`` view,
    # which re-lists every ``providers.<slug>`` row as ``custom:<name>``; a hand-migrated config may
    # also keep the same endpoint in both sections. Either is one provider, not two (#112788) — but
    # the duplicate is folded at the MATCH level, not dropped as a candidate: a model only the legacy
    # row declares still routes to the shared endpoint instead of falling through to the current
    # provider.
    rows = {slug: _configured_provider_identity(slug, cfg) for slug, cfg in candidates}
    entries = [(f"custom:{e['name']}", e) for e in _custom_entries(custom_providers)
               if isinstance(e.get("name"), str) and e["name"].strip()]

    matches: dict[str, str] = {}
    for slug, cfg in candidates + entries:
        hit = next((mid for key in ("models", "model", "default_model")
                    for mid in _declared_model_ids(cfg.get(key)) if mid.lower() == target), None)
        if hit:
            owner = _duplicates_configured_row(slug, cfg, rows) if slug not in rows else None
            matches.setdefault(owner or slug, hit)  # first declaration wins
    return matches


def _configured_provider_identity(slug: str, cfg: dict) -> tuple[str, str, str, str]:
    """``(name, endpoint, credential, api_mode)`` of one configured provider, read through the same
    normalizer that builds the compat view, so a ``providers.<slug>`` row, its ``custom:<name>``
    projection and a legacy duplicate of the same endpoint reduce to one tuple. Any difference in
    endpoint, credential identity or wire protocol keeps two rows distinct."""
    from hermes_cli.config_providers import _normalize_custom_provider_entry
    from providers.routing import canonicalize_api_mode
    # ``provider_key`` is the compat view's stamp, not a config key: drop it so the normalizer does
    # not warn about it as unknown.
    entry = _normalize_custom_provider_entry({k: v for k, v in cfg.items() if k != "provider_key"},
                                             provider_key="" if slug.startswith("custom:") else slug) or cfg
    name = (_clean(entry.get("name")).lower() or slug.removeprefix("custom:").lower()).replace(" ", "-")
    base_url = _clean(entry.get("base_url") or entry.get("url") or entry.get("api")).rstrip("/").lower()
    api_key, key_env = _clean(entry.get("api_key")), _clean(entry.get("key_env") or entry.get("api_key_env"))
    if api_key.startswith("${") and api_key.endswith("}"):
        api_key, key_env = "", api_key[2:-1].strip()
    credential = (f"key:{api_key}" if api_key else f"env:{key_env}" if key_env
                  else f"cmd:{_clean(entry.get('key_cmd'))}" if _clean(entry.get("key_cmd")) else "")
    api_mode = _clean(entry.get("api_mode") or entry.get("transport"))
    return name, base_url, credential, canonicalize_api_mode(api_mode).lower() if api_mode else ""


def _duplicates_configured_row(
        slug: str, entry: dict, rows: dict[str, tuple[str, str, str, str]]) -> Optional[str]:
    """Slug of the ``providers.<slug>`` row a ``custom_providers`` entry is a second view of, else
    None: the row's compat projection (``provider_key`` names the slug AND it points at the same
    endpoint with the same credential — on the raw-list fallback a hand-written provider_key aimed
    at another endpoint or another key stays a candidate) or a legacy duplicate with the same
    provider identity."""
    identity = _configured_provider_identity(slug, entry)
    provider_key = _clean(entry.get("provider_key")).lower()
    return next((row_slug for row_slug, row in rows.items()
                 if identity == row or (provider_key == row_slug.lower() and identity[1:3] == row[1:3])), None)


def _custom_entries(custom_providers: Any) -> list[dict]:
    """The dict-shaped entries of a ``custom_providers:`` list (anything else is ignored)."""
    return [e for e in custom_providers if isinstance(e, dict)] if isinstance(custom_providers, list) else []


def _entry_aliases(entry: dict) -> frozenset[str]:
    return custom_provider_aliases(str(entry.get("name") or ""), str(entry.get("provider_key") or ""))


def _selection_provider_facts(
    provider: str,
    user_providers: Optional[dict],
    custom_providers: Optional[list],
    *,
    include_models: bool = True,
) -> ExplicitProviderFacts:
    """Materialize selection facts without moving discovery/config ownership."""

    canonical = _provider_identity(provider, user_providers, custom_providers)
    from providers import get_provider_profile

    profile = get_provider_profile(canonical)
    identity_aliases: set[str] = {canonical, str(provider or "").strip().lower()}
    model_aliases: dict[str, str] = dict(getattr(profile, "model_aliases", {}) or {})

    providers_cfg = user_providers if isinstance(user_providers, dict) else {}
    for slug, cfg in providers_cfg.items():
        if not isinstance(cfg, dict):
            continue
        aliases = custom_provider_aliases(str(cfg.get("name") or slug), str(slug))
        if canonical in aliases or str(provider or "").strip().lower() in aliases:
            identity_aliases.update(aliases)

    for entry in _custom_entries(custom_providers):
        aliases = _entry_aliases(entry)
        if canonical not in aliases and str(provider or "").strip().lower() not in aliases:
            continue
        identity_aliases.update(aliases)
        models = _declared_model_ids(entry.get("models"))
        for alias in aliases:
            prefix = alias.removeprefix("custom:")
            if not prefix:
                continue
            for model_id in models:
                model_aliases.setdefault(f"{prefix}/{model_id}".lower(), model_id)

    catalog_models = tuple(list_provider_models(canonical)) if include_models else ()
    static_models = tuple(static_provider_model_ids(canonical))
    fallback_models = tuple(getattr(profile, "fallback_models", ()) or ())
    alias_models = tuple(dict.fromkeys((*catalog_models, *static_models, *fallback_models)))

    return ExplicitProviderFacts(
        provider=canonical,
        alias_models=alias_models,
        catalog_models=catalog_models,
        model_aliases=tuple(
            (str(key).strip().lower(), str(value).strip())
            for key, value in model_aliases.items()
            if str(key).strip() and str(value).strip()
        ),
        identity_aliases=tuple(sorted(value for value in identity_aliases if value)),
        aggregator=is_aggregator(canonical),
        normalization_ids=static_models,
    )


def _selection_direct_aliases(
    user_providers: Optional[dict], custom_providers: Optional[list]
) -> tuple[ExplicitAlias, ...]:
    _ensure_direct_aliases()
    return tuple(
        ExplicitAlias(
            name=name,
            ref=ModelRef(
                _provider_identity(alias.ref.provider, user_providers, custom_providers),
                alias.ref.model,
            ),
        )
        for name, alias in DIRECT_ALIASES.items()
    )


def _selection_provider_fact_set(
    providers: list[tuple[str, bool]],
    user_providers: Optional[dict],
    custom_providers: Optional[list],
) -> tuple[ExplicitProviderFacts, ...]:
    """Build one merged fact row per canonical provider."""

    merged: dict[str, ExplicitProviderFacts] = {}
    for provider, include_models in providers:
        if not provider:
            continue
        fact = _selection_provider_facts(
            provider,
            user_providers,
            custom_providers,
            include_models=include_models,
        )
        prior = merged.get(fact.provider)
        if prior is None:
            merged[fact.provider] = fact
            continue
        merged[fact.provider] = ExplicitProviderFacts(
            provider=fact.provider,
            alias_models=tuple(dict.fromkeys((*prior.alias_models, *fact.alias_models))),
            catalog_models=tuple(dict.fromkeys((*prior.catalog_models, *fact.catalog_models))),
            model_aliases=tuple(dict((*prior.model_aliases, *fact.model_aliases)).items()),
            identity_aliases=tuple(
                dict.fromkeys((*prior.identity_aliases, *fact.identity_aliases))
            ),
            aggregator=prior.aggregator or fact.aggregator,
            normalization_ids=tuple(
                dict.fromkeys((*prior.normalization_ids, *fact.normalization_ids))
            ),
        )
    return tuple(merged.values())


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
    explicit_norm = explicit_provider.strip().lower()
    alias_target = normalize_provider(explicit_norm)
    if not (
        alias_target and alias_target == target_provider and target_provider != explicit_norm
        and is_aggregator(target_provider)):
        return ""
    authed = get_authenticated_provider_slugs(
        current_provider=current_provider, user_providers=user_providers, custom_providers=custom_providers)
    if target_provider in authed:
        return ""
    suggestions = [s for s in authed if s.startswith(explicit_norm) and s != explicit_norm]
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    return (
        f"Provider '{explicit_norm}' is an alias that routes "
        f"through {get_provider_label(target_provider)}, which "
        f"has no credentials configured.{hint}")


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
    st.api_mode = ""  # canonical route resolver finalizes the wire after credentials settle


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
    runtime_kind: str = ""
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
        self.runtime_kind = rt.get("runtime_kind", "")
        self.validation_headers = rt.get("extra_headers") or self.validation_headers


def _select_switch_target(st: _Switch) -> Optional[ModelSwitchResult]:
    """Resolve explicit /model intent through the canonical selection domain."""

    raw = st.raw_input.strip()
    explicit_target = ""
    if st.explicit_provider:
        pdef = resolve_provider_full(
            st.explicit_provider, st.user_providers, st.custom_providers
        )
        if pdef is None and st.explicit_provider.strip().lower() == "custom":
            pdef = _bare_custom_provider_def(st.current_base_url)
        if pdef is None:
            return st.fail(_unknown_provider_message(st.explicit_provider))
        explicit_target, st.provider_label = pdef.id, pdef.display_name
        if explicit_target == "moa" and not raw:
            raw = _moa_default_preset()
        agg_err = _aggregator_alias_error(
            st.explicit_provider,
            explicit_target,
            st.current_provider,
            st.user_providers,
            st.custom_providers,
        )
        if agg_err:
            st.target_provider = explicit_target
            return st.fail_on_target(agg_err)
        if not raw:
            if not pdef.base_url:
                st.target_provider = explicit_target
                return st.fail_on_target(
                    f"Provider '{pdef.display_name}' has no base URL configured. "
                    f"Specify a model: /model <model-name> --provider {st.explicit_provider}"
                )
            from models.catalog_probe import detect_single_openai_model

            raw = detect_single_openai_model(pdef.base_url)
            if not raw:
                st.target_provider = explicit_target
                return st.fail_on_target(
                    f"No model detected on {pdef.display_name} ({pdef.base_url}). "
                    f"Specify the model explicitly: /model <model-name> "
                    f"--provider {st.explicit_provider}"
                )

    moa_ref = None
    if not explicit_target:
        try:
            from hermes_cli.config import load_config
            from hermes_cli.moa_config import (
                exact_moa_preset_name,
                normalize_moa_config,
            )

            moa_match = exact_moa_preset_name(
                normalize_moa_config(load_config().get("moa") or {}), raw
            )
            if moa_match:
                moa_ref = ModelRef("moa", moa_match)
        except Exception:
            pass

    known_ids, named_custom_ids = _model_ref_context(
        st.user_providers, st.custom_providers
    )
    qualified_provider = (
        explicit_provider_hint(
            raw,
            known_provider_ids=tuple(known_ids),
            named_custom_provider_ids=tuple(named_custom_ids),
        )
        if not explicit_target
        else ""
    )

    configured = _configured_provider_matches(
        raw, st.user_providers, st.custom_providers
    )
    configured_refs = tuple(
        ModelRef(provider, model) for provider, model in configured.items()
    )
    configured_credential_providers = tuple(
        str(provider).strip().lower()
        for provider in (st.user_providers or {})
        if str(provider).strip()
    )

    guest_block = False
    if not explicit_target and st.current_provider == "nous":
        try:
            from hermes_cli.anon_auth import GUEST_MODEL, route_is_welcome_host

            guest_block = (
                route_is_welcome_host(st.current_base_url)
                and raw != GUEST_MODEL
            )
        except Exception:
            guest_block = False

    hold_current = (
        st.current_provider in {"custom", "local"}
        or st.current_provider.startswith("custom:")
        or base_url_hostname(st.current_base_url or "") in ("localhost", "127.0.0.1")
    )

    detection = None
    if (
        not explicit_target
        and not guest_block
        and not hold_current
        and not configured_refs
        and moa_ref is None
    ):
        from hermes_cli.model_selection_facts import build_explicit_detection_facts

        detection = build_explicit_detection_facts(raw, st.current_provider)

    authenticated: list[str] = []
    if not explicit_target and raw.lower() in MODEL_ALIASES:
        authenticated = get_authenticated_provider_slugs(
            current_provider=st.current_provider,
            user_providers=st.user_providers,
            custom_providers=st.custom_providers,
        )

    semantic_alias = raw.lower() in MODEL_ALIASES
    requested_facts: list[tuple[str, bool]] = [
        (
            st.current_provider,
            semantic_alias or is_aggregator(st.current_provider),
        )
    ]
    for provider in (explicit_target, qualified_provider):
        if provider:
            requested_facts.append((provider, semantic_alias))
    detection_providers = detection.candidate_providers() if detection is not None else ()
    for provider in (
        *(ref.provider for ref in configured_refs),
        *detection_providers,
    ):
        if provider:
            requested_facts.append((provider, False))
    requested_facts.extend((provider, True) for provider in authenticated)

    facts = _selection_provider_fact_set(
        requested_facts, st.user_providers, st.custom_providers
    )
    try:
        selection = select_explicit_model(
            raw,
            st.current_provider,
            explicit_provider=explicit_target,
            provider_facts=facts,
            direct_aliases=_selection_direct_aliases(
                st.user_providers, st.custom_providers
            ),
            configured_matches=configured_refs,
            fallback_providers=tuple(authenticated),
            known_provider_ids=tuple(known_ids),
            named_custom_provider_ids=tuple(named_custom_ids),
            detection=detection,
            block_provider_fallback=guest_block,
            hold_current_provider=hold_current,
            moa_ref=moa_ref,
        )
    except AmbiguousModelAliasError as err:
        return st.fail(_ambiguous_alias_message(err))
    except ExplicitSelectionError as err:
        if err.code == "ambiguous_configured":
            return st.fail(
                f"'{raw}' is declared by multiple configured providers "
                f"({', '.join(err.providers)}). Re-run with --provider <slug> "
                "to choose which one to use."
            )
        if err.code == "alias_unavailable":
            identity = MODEL_ALIASES[raw.lower()]
            return st.fail(
                f"Alias '{raw.lower()}' maps to {identity.vendor}/{identity.family} "
                "but no matching model was found in any provider catalog. "
                "Try specifying the full model name."
            )
        return st.fail(f"Could not resolve model '{raw}'.")

    if selection.selected is None:
        return st.fail(f"Could not resolve model '{raw}'.")
    selected = selection.selected
    route_provider = selected.ref.provider
    if selection.matched_alias:
        # ModelRef carries canonical provider identity. A direct alias can additionally name a
        # concrete configured route key (for example custom:ollama -> providers.ollama); restore
        # that application-owned key only while applying the selection so credentials/endpoints
        # still resolve against the route the alias actually named.
        direct_alias = DIRECT_ALIASES.get(selection.matched_alias.lower())
        if direct_alias is not None:
            route_provider = _provider_identity(
                direct_alias.ref.provider, st.user_providers, st.custom_providers
            )
    st.target_provider = route_provider
    st.new_model = selected.ref.model
    st.resolved_alias = selection.matched_alias
    target_identity = _provider_identity(
        st.target_provider, st.user_providers, st.custom_providers
    )
    current_identity = _provider_identity(
        st.current_provider, st.user_providers, st.custom_providers
    )
    if (
        not st.explicit_provider
        and st.target_provider in configured_credential_providers
        and (
            selected.source == "explicit"
            or target_identity != current_identity
        )
    ):
        st.explicit_provider = st.target_provider

    if guest_block and selected.source == "current":
        from hermes_cli.anon_auth import GUEST_MODEL

        if st.new_model != GUEST_MODEL:
            return st.fail(
                f"{st.new_model} needs a Nous account or an API key. "
                "Use /login to sign in, or /model to pick another provider."
            )

    logger.debug(
        "Selection resolved %r to %s on %s via %s",
        st.raw_input,
        st.new_model,
        st.target_provider,
        selected.source,
    )
    return None


def _switch_provider_label(st: _Switch) -> str:
    label = get_provider_label(st.target_provider)
    if st.target_provider == "custom" and st.current_base_url:
        label = "Custom endpoint"
    if st.target_provider.startswith("custom:"):
        custom_pdef = resolve_provider_full(st.target_provider, st.user_providers, st.custom_providers)
        if custom_pdef is not None:
            label = custom_pdef.display_name
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
        # A bare-custom session switching models stays on its endpoint (#45597). Arriving from
        # ANOTHER provider (the per-turn config sync adopting ``provider: custom``) the configured
        # endpoint wins, or the new model is paired with the old provider's host and key (#73680).
        # With nothing configured the resolver either raises (st.* keep the session values) or
        # lands on OpenRouter's default or the ``OPENROUTER_BASE_URL`` mirror (#74143, #10622) —
        # the session endpoint is kept in all three cases.
        key, url = st.current_api_key, st.current_base_url
        if st.current_provider != "custom":
            with suppress(Exception):
                st.resolve_runtime(requested="custom")
            if st.base_url and not _fell_back_to_openrouter_default(st):
                key, url = st.api_key, st.base_url
        st.api_key, st.base_url = key, url
        st.api_mode = ""  # canonical route resolver finalizes this in _build_switch_result
    else:
        # A URL-bearing LOCAL direct alias (ollama, vllm — labels that resolve to `custom`)
        # supplies its endpoint HERE as well as in _apply_direct_alias_endpoint: the resolver
        # refuses such an alias with no endpoint configured anywhere, and this alias does have
        # one. A built-in label (anthropic, openai, …) must NOT get the alias URL: its resolver
        # would pair the vendor key with the foreign host, and _apply_direct_alias_endpoint then
        # sees a same-origin credential and keeps it (#28660).
        from providers import resolves_to_custom_provider
        da = DIRECT_ALIASES.get(st.resolved_alias) if st.resolved_alias else None
        alias_url = da.base_url if da is not None and resolves_to_custom_provider(st.target_provider) else None
        try:
            st.resolve_runtime(requested=st.target_provider, explicit_base_url=alias_url or None)
        except Exception as e:
            if st.target_provider.strip().lower() in LLAMACPP_ALIASES:
                # A local-runtime alias has no credential to add: the seam's own message ("server
                # isn't running" / "turned off") is the actionable one, the auth hint below is noise.
                return st.fail_on_target(str(e))
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
        st.validation_headers = ollama_headers
    else:
        try:
            st.resolve_runtime(requested=st.current_provider)
        except Exception:
            pass
        # Bare ``custom``/``local`` sessions whose base_url is session-only (not a trusted config
        # ``model.base_url``) re-resolve to the OpenRouter DEFAULT — a host the user never picked
        # (#74143). Keep the session endpoint + key then (also when the resolver came back empty,
        # whatever host the session is on); a config-backed custom URL still wins so key/endpoint
        # rotation is not pinned to a stale session.
        if (
            st.current_provider in {"custom", "local"} and st.current_base_url
            and (not st.base_url or _fell_back_to_openrouter_default(st))
        ):
            st.base_url, st.api_key = st.current_base_url, st.current_api_key


def _fell_back_to_openrouter_default(st: _Switch) -> bool:
    """The bare-``custom`` resolver ended on an OpenRouter endpoint that is not a custom endpoint
    the user configured: the built-in default host, or the ``OPENROUTER_BASE_URL`` mirror — the
    credential ladder's last rung (#10622), which ``provider: custom`` reaches only when no
    ``CUSTOM_BASE_URL`` / trusted ``model.base_url`` exists."""
    mirror = _openrouter_mirror_base_url()
    if mirror and st.base_url.rstrip("/") == mirror and not _custom_endpoint_source():
        return True
    return (base_url_host_matches(st.base_url, "openrouter.ai")
            and not base_url_host_matches(st.current_base_url, "openrouter.ai"))


def _custom_endpoint_source() -> str:
    """The endpoint the credential ladder prefers over its OpenRouter rung for bare ``custom``:
    ``CUSTOM_BASE_URL``, else the config's ``model.base_url`` when that config backs bare custom.
    Non-empty means a resolved URL matching the mirror came from a configured custom endpoint (two
    env vars pointed at one proxy), so the mirror guard must not call it a fallback."""
    from agent.secret_scope import get_secret_str
    try:
        env_url = (get_secret_str("CUSTOM_BASE_URL", "") or "").strip()
        if env_url:
            return env_url
        from hermes_cli.runtime_provider import (
            _config_base_url_trustworthy_for_bare_custom, _get_model_config)
        model_cfg = _get_model_config() or {}
        base = model_cfg.get("base_url") if isinstance(model_cfg.get("base_url"), str) else ""
        provider = model_cfg.get("provider") if isinstance(model_cfg.get("provider"), str) else ""
        base = (base or "").strip()
        return base if base and _config_base_url_trustworthy_for_bare_custom(base, provider) else ""
    except Exception:
        return ""


def _openrouter_mirror_base_url() -> str:
    """``OPENROUTER_BASE_URL``, read the way the resolver reads it (env, or the profile's secret
    scope). A guard read, not a credential fetch: a read that fails — unscoped under multiplexing —
    must leave the mirror undetected so its caller keeps the session endpoint, rather than raising
    out of ``switch_model`` where the resolver's own read of the same name is suppressed."""
    from agent.secret_scope import get_secret_str
    try:
        return (get_secret_str("OPENROUTER_BASE_URL", "") or "").strip().rstrip("/")
    except Exception:
        return ""


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

    return None


def _validate_switch(st: _Switch) -> Optional[ModelSwitchResult]:
    """COMMON PATH part 2: validate the already-selected canonical model."""
    from hermes_cli.models_local import _get_ollama_request_headers
    from hermes_cli.models_validate import validate_requested_model
    from models.catalog_chat import is_known_non_chat_model
    if is_known_non_chat_model(st.new_model):
        return st.fail(
            f"`{st.new_model}` is a generation model and cannot be used for chat. "
            "Pick a chat model, or use image generation for image models.",
            new_model=st.new_model, target_provider=st.target_provider,
            provider_label=st.provider_label,
        )

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


def _build_switch_result(st: _Switch) -> ModelSwitchResult:
    """COMMON PATH part 3: final api_mode / base_url shaping, metadata, warnings."""
    from providers.routing import InvocationRequest, resolve_invocation_route
    route = resolve_invocation_route(InvocationRequest(
        provider=st.target_provider,
        model=st.new_model,
        base_url=st.base_url,
        explicit_api_mode=None,
        configured_api_mode=st.api_mode or None,
        configured_provider=st.target_provider or None,
        openai_runtime="codex_app_server" if st.runtime_kind == "app_server" else None,
        requested_provider=st.explicit_provider or st.target_provider,
    ))
    st.target_provider, st.new_model = route.provider, route.model
    st.base_url, st.api_mode, st.runtime_kind = route.base_url, route.api_mode, route.runtime_kind

    # OpenCode base URLs end with /v1 for OpenAI-compatible models but the Anthropic SDK prepends
    # its own /v1/messages: strip for anthropic_messages, re-append for
    # chat_completions/codex_responses (mirrors resolve_runtime_provider).
    from providers import normalize_opencode_base_url, opencode_provider_family
    if opencode_provider_family(st.target_provider) is not None and isinstance(st.base_url, str):
        st.base_url = normalize_opencode_base_url(st.target_provider, st.api_mode, st.base_url)

    capabilities = query_model_metadata(st.target_provider, st.new_model, allow_network=True)
    from agent.native_compaction import resolve_native_compaction_capabilities
    runtime_capabilities = resolve_native_compaction_capabilities(
        model=st.new_model, base_url=st.base_url, provider=st.target_provider,
        is_codex_backend=st.target_provider.strip().lower() == "openai-codex")
    model_info = get_model_info(st.target_provider, st.new_model, allow_network=True)

    warnings = [w for w in (st.validation.get("message"), nous_hermes_non_agentic_warning(st.new_model)) if w]

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
        provider_changed=st.provider_changed, api_key=st.api_key, base_url=st.base_url,
        api_mode=st.api_mode, runtime_kind=st.runtime_kind,
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
    for step in (_select_switch_target, _resolve_switch_credentials, _validate_switch):
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
    from providers import normalize_route_base_url
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
    from hermes_cli.config import get_config_path, read_user_config_raw
    from utils import atomic_roundtrip_yaml_update
    path = Path(config_path) if config_path else get_config_path()
    for key, value in model_selection_config_updates(result, read_user_config_raw(path).get("model")).items():
        atomic_roundtrip_yaml_update(path, f"model.{key}", value)
    try:  # owner-only: config files contain API keys
        os.chmod(path, 0o600)
    except (OSError, NotImplementedError):
        pass


from application_provider_secret_inputs import (
    extra_headers_from_config as _extra_headers_from_config,
    scoped_key_env as _scoped_key_env,
)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import List  # noqa: F401,E402
import http.client  # noqa: F401,E402
import time  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'base_url_host_matches': ('utils', 'base_url_host_matches'),
    'list_picker_providers': ('application_provider_discovery', 'list_picker_providers'),
    'prewarm_picker_cache_async': ('application_picker_prewarm', 'prewarm_picker_cache_async'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
