"""Shared model-switching logic for the CLI and gateway /model commands.

Pipeline: parse flags -> alias resolution -> provider resolution -> credential resolution ->
normalize model name -> metadata lookup -> build result. Provider switching uses ``--provider``
exclusively; colons are reserved for OpenRouter variant suffixes (``:free``, ``:extended``)."""

from __future__ import annotations

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
from utils import base_url_hostname, base_url_origin
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
        if model_id and model_id.lower() not in seen:
            seen.add(model_id.lower())
            ids.append(model_id)
    return ids


def _declared_item_id(item: dict) -> Any:
    """``id`` of a ``[{"id": ...}]`` entry, falling back to ``name`` when blank/missing."""
    model_id = item.get("id")
    return model_id if isinstance(model_id, str) and model_id.strip() else item.get("name")


def _entry_models_discovered(entry: Any) -> bool:
    """True when the entry's ``models`` mapping was auto-discovered by Hermes.

    Current shape: entry-level ``models_discovered: true``. Older versions wrote an in-mapping
    ``__discovered_model_catalog__: true`` sentinel — accepted on read (the next save migrates it)."""
    if not isinstance(entry, dict):
        return False
    models = entry.get("models")
    return entry.get("models_discovered") is True or (
        isinstance(models, dict) and models.get("__discovered_model_catalog__") is True)


def _models_config_is_allowlist(value: Any, discovered: bool = False) -> bool:
    """True when ``models:`` is an intentional ID allowlist.

    A mapping like ``{model_id: {context_length: N}}`` is per-model *metadata* written by
    ``_save_custom_provider`` / the wizard, not a catalog narrow (treating it as one made GUI
    pickers show only the saved default for keyless Ollama while the CLI live-probed). List and
    string shapes remain allowlists for no-key endpoints; pin a dict catalog with
    ``discover_models: false``. A catalog Hermes itself persisted (``discovered``) is never a pin."""
    if discovered:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        return bool(_declared_model_ids(value))
    return False  # None, dict (per-model metadata), or anything else


def _bare_custom_provider_def(current_base_url: str) -> Optional[ProviderDef]:
    """ProviderDef for a direct ``model.provider: custom`` endpoint."""
    base_url = _clean(current_base_url)
    if not base_url:
        return None
    return ProviderDef(
        id="custom", name="Custom endpoint", transport="openai_chat", api_key_env_vars=(),
        base_url=base_url, is_aggregator=False, auth_type="api_key", source="model-config")


# --- Non-agentic model warning

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
        return (str(path), None, None)
    except Exception:
        return None
    return (str(path), stat.st_mtime_ns, stat.st_size)


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
    is_global: bool = False
    force_refresh: bool = False
    is_session: bool = False
    is_once: bool = False


# --- Flag parsing

_BOOL_FLAGS = {"--global": "is_global", "--session": "is_session", "--refresh": "force_refresh", "--once": "is_once"}


def parse_model_flags_detailed(raw_args: str) -> ModelFlagParseResult:
    """Parse /model flags: ``--provider X``, ``--global``, ``--session``, ``--refresh``, ``--once``.

    ``--once`` is parsed here but interpreted by each caller (each frontend has its own
    live-session restore hook). ``is_global`` / ``is_session`` are raw flag presences; the
    effective persistence decision belongs to :func:`resolve_persist_behavior`."""
    # Telegram/iOS auto-convert ``--`` to an em/en dash: normalize a single Unicode dash before
    # a flag keyword.
    raw_args = re.sub(r'[\u2012\u2013\u2014\u2015](provider|global|session|refresh|once)', r'--\1', raw_args)

    # Hand-rolled: model IDs may contain colons/slashes and the historical parser did not
    # require shell quoting.
    flags = dict.fromkeys(_BOOL_FLAGS.values(), False)
    explicit_provider = ""
    filtered: list[str] = []
    tokens = iter(raw_args.split())
    for tok in tokens:
        if tok in _BOOL_FLAGS:
            flags[_BOOL_FLAGS[tok]] = True
        elif tok == "--provider" and (value := next(tokens, None)) is not None:
            explicit_provider = value
        else:
            filtered.append(tok)  # a trailing bare ``--provider`` stays part of the model text
    return ModelFlagParseResult(model_input=" ".join(filtered).strip(), explicit_provider=explicit_provider, **flags)


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

# Canonical (surface-neutral) error copy. Surfaces prepend their own decoration ("  ✗ " in the
# CLI, "❌ " in the gateway) but MUST NOT change the core sentence — it is shared user-visible copy.
MODEL_SWITCH_ERROR_TEXT = {
    MODEL_SWITCH_ERR_ONCE_WITH_GLOBAL: "/model --once cannot be combined with --global",
    MODEL_SWITCH_ERR_ONCE_REQUIRES_TARGET: "/model --once requires a model or provider."}


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
    a model nor ``--provider`` -> ``MODEL_SWITCH_ERR_ONCE_REQUIRES_TARGET``. Targets pass through
    untouched (bare names, ``vendor/model``, ``vendor:model``) for :func:`switch_model`."""
    raw = str(raw or "")
    parsed = parse_model_flags_detailed(raw)

    errors: list = []
    if parsed.is_once and parsed.is_global:
        errors.append(MODEL_SWITCH_ERR_ONCE_WITH_GLOBAL)
    if parsed.is_once and not parsed.model_input and not parsed.explicit_provider:
        errors.append(MODEL_SWITCH_ERR_ONCE_REQUIRES_TARGET)
    # First matching flag wins: once > session > global > default.
    scope = next((name for name, on in (("once", parsed.is_once), ("session", parsed.is_session),
                                        ("global", parsed.is_global)) if on), "default")
    return ModelSwitchRequest(
        raw=raw, target=parsed.model_input, scope=scope, errors=tuple(errors),
        **{f: getattr(parsed, f)
           for f in ("explicit_provider", "is_global", "is_session", "is_once", "force_refresh")})


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
    user_slugs_lower = set()
    if isinstance(user_providers, dict):
        candidates += [(slug, cfg) for slug, cfg in user_providers.items()
                       if isinstance(slug, str) and isinstance(cfg, dict)]
        user_slugs_lower = {slug.lower() for slug, _ in candidates}
    # A new-style providers.<slug> entry mirrored into custom_providers (via
    # get_compatible_custom_providers) carries provider_key == slug; skip the
    # mirror so one model isn't reported as declared twice (custom:gpt + gpt).
    # Compare against a normalized (lowercased) set to catch mixed-case slugs.
    candidates += [(f"custom:{e['name']}", e) for e in _custom_entries(custom_providers)
                   if isinstance(e.get("name"), str) and e["name"].strip()
                   and str(e.get("provider_key") or "").strip().lower() not in user_slugs_lower]

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


def _custom_entries(custom_providers: Any) -> list[dict]:
    """The dict-shaped entries of a ``custom_providers:`` list (anything else is ignored)."""
    return [e for e in custom_providers if isinstance(e, dict)] if isinstance(custom_providers, list) else []


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
    """Step c: on an aggregator, ``vendor:model`` -> ``vendor/model``. Only without a slash: with
    one, the colon is a variant tag (:free, :extended, :fast) that must be preserved."""
    raw_input = st.raw_input
    colon_pos = raw_input.find(":")
    cur_norm = str(st.current_provider).strip().lower()
    if (
        colon_pos > 0 and "/" not in raw_input and is_aggregator(st.current_provider)
        and not cur_norm.startswith("custom") and cur_norm != "ollama"):
        left = raw_input[:colon_pos].strip().lower()
        right = raw_input[colon_pos + 1:].strip()
        if left and right:
            st.new_model = f"{left}/{right}"
            logger.debug("Converted vendor:model '%s' to aggregator slug '%s'", raw_input, st.new_model)


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
            return st.fail_on_target(f"Could not resolve credentials for provider '{st.provider_label}': {e}")
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
    mandated_mode = host_mandated_api_mode(st.base_url)
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
    try:
        validation = validate_requested_model(
            st.new_model, st.target_provider, api_key=st.api_key, base_url=st.base_url,
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
    st.new_model = validation.get("corrected_model") or st.new_model
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


def _extra_headers_from_config(entry: Any) -> dict[str, str]:
    if not isinstance(entry, dict):
        return {}
    from hermes_cli.config import normalize_extra_headers
    return normalize_extra_headers(entry.get("extra_headers"))


def _scoped_key_env(name: str) -> str:
    """Read a provider key env var through the per-profile secret scope.

    The multiplexed gateway installs a secret scope per turn; a raw ``os.environ`` read hands the
    current profile whatever key happens to be in the process environment — another profile's.
    Identical to ``os.getenv`` when multiplexing is off. A fail-closed ``UnscopedSecretError``
    (multiplexing on, no scope installed) means "no credential visible for this profile here",
    which is exactly how the picker already treats a missing key."""
    try:
        from agent.secret_scope import get_secret
        return (get_secret(name, "") or "").strip() if name else ""
    except Exception:
        return ""


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import List  # noqa: F401,E402
import http.client  # noqa: F401,E402
import time  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'base_url_host_matches': ('utils', 'base_url_host_matches'),
    'custom_provider_slug': ('hermes_cli.providers', 'custom_provider_slug'),
    'list_picker_providers': ('hermes_cli.model_switch_providers', 'list_picker_providers'),
    'prewarm_picker_cache_async': ('hermes_cli.model_switch_providers', 'prewarm_picker_cache_async'),
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
