"""Lightweight internationalization (i18n) for Hermes static user-facing messages.

Scope (thin slice, by design): only the highest-impact static strings shown
to the user by Hermes itself -- approval prompts, a handful of gateway slash
command replies, restart-drain notices.  Agent-generated output, log lines,
error tracebacks, tool outputs, and slash-command descriptions all stay in
English.

Catalog files live under ``locales/<lang>.yaml`` at the repo root.  Each
catalog is a flat dict keyed by dotted paths (e.g. ``approval.choose`` or
``gateway.approval_expired``).  Missing keys fall back to English; if English
is missing too, the key path itself is returned so a broken catalog never
crashes the agent.

Usage::

    from agent.i18n import t
    print(t("approval.choose_long"))                       # current lang
    print(t("gateway.draining", count=3))                  # {count} formatted
    print(t("approval.choose_long", lang="zh"))            # explicit override

Language resolution order:
    1. Explicit ``lang=`` argument passed to :func:`t`
    2. ``HERMES_LANGUAGE`` environment variable (for tests / quick override)
    3. ``display.language`` from config.yaml
    4. ``"en"`` (baseline)

Supported languages: en, zh, ja, de, es, fr, tr, uk.  Unknown values fall back to en.
"""

from __future__ import annotations

import logging
import os
import string
import sysconfig
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES: tuple[str, ...] = (
    "en", "zh", "zh-hant", "ja", "de", "es", "fr", "tr", "uk",
    "af", "ko", "it", "ga", "pt", "ru", "hu",
)
DEFAULT_LANGUAGE = "en"

# Accept a few natural aliases so users who type "chinese" / "zh-CN" / "jp"
# get the right catalog instead of silently falling back to English.
_LANGUAGE_ALIASES: dict[str, str] = {
    "english": "en", "en-us": "en", "en-gb": "en",
    # Simplified Chinese — explicit codes route here; bare "chinese" / "mandarin"
    # also default to Simplified since that's the larger user base.
    "chinese": "zh", "mandarin": "zh", "zh-cn": "zh", "zh-hans": "zh", "zh-sg": "zh",
    # Traditional Chinese — distinct catalog.  Cover Taiwan / Hong Kong / Macau
    # locale tags plus the common "traditional" alias.
    "traditional-chinese": "zh-hant", "traditional_chinese": "zh-hant",
    "zh-tw": "zh-hant", "zh-hk": "zh-hant", "zh-mo": "zh-hant",
    "japanese": "ja", "jp": "ja", "ja-jp": "ja",
    "german": "de", "deutsch": "de", "de-de": "de", "de-at": "de", "de-ch": "de",
    "spanish": "es", "español": "es", "espanol": "es", "es-es": "es", "es-mx": "es", "es-ar": "es",
    "french": "fr", "français": "fr", "france": "fr", "fr-fr": "fr", "fr-be": "fr", "fr-ca": "fr", "fr-ch": "fr",
    "ukrainian": "uk", "ukrainisch": "uk", "українська": "uk", "uk-ua": "uk", "ua": "uk",
    "turkish": "tr", "türkçe": "tr", "tr-tr": "tr",
    # Afrikaans — South African Dutch-derived language; "af-ZA" is the common BCP-47 tag.
    "afrikaans": "af", "af-za": "af",
    # Korean
    "korean": "ko", "한국어": "ko", "ko-kr": "ko",
    # Italian
    "italian": "it", "italiano": "it", "it-it": "it", "it-ch": "it",
    # Irish (Gaeilge) — ga is the BCP-47 code
    "irish": "ga", "gaeilge": "ga", "ga-ie": "ga",
    # Portuguese — bare "portuguese" routes to European Portuguese; pt-br
    # is in the same family but rendered identically here (no separate br catalog).
    "portuguese": "pt", "português": "pt", "portugues": "pt",
    "pt-pt": "pt", "pt-br": "pt", "brazilian": "pt", "brasileiro": "pt",
    # Russian
    "russian": "ru", "русский": "ru", "ru-ru": "ru",
    # Hungarian
    "hungarian": "hu", "magyar": "hu", "hu-hu": "hu",
}

_MUTABLE_CATEGORIES: frozenset[str] = frozenset({"progress", "lifecycle", "info"})

# PROVISIONAL (upstream migration): keys below are re-derived in Phase 3 against
# the gateway strings actually wrapped on current upstream/main. A key absent from
# the catalog never matches, so suppression is a harmless no-op for it until then.
# Full catalog key -> suppression category. ONLY notification-class keys whose
# emit path drops empty content (direct adapter.send guarded by
# _send_unless_empty, or status_callback via _prepare_gateway_status_message).
# Return-value reply messages are intentionally excluded (see plan §Global
# Constraints). errors/approval/commands keys are absent here => never suppressed.
GATEWAY_MESSAGE_CATEGORIES: dict[str, str] = {
    # progress — ephemeral work status
    "gateway.long_running": "progress",
    "gateway.no_activity_warning": "progress",
    "gateway.subagent_working": "progress",
    "gateway.queued_next_turn": "progress",
    "gateway.interrupting_task": "progress",
    "gateway.steered_into_run": "progress",
    # lifecycle — gateway daemon lifecycle notifications
    "gateway.restart_success": "lifecycle",
    "gateway.gateway_online": "lifecycle",
    "gateway.shutdown_restarting": "lifecycle",
    "gateway.shutdown_shutting_down": "lifecycle",
    # info — one-shot notices
    "gateway.codex_gpt55_autoraise_notice": "info",
    "gateway.kanban_done": "info",
    "gateway.kanban_blocked": "info",
    "gateway.kanban_crashed": "info",
    "gateway.kanban_gave_up": "info",
    "gateway.kanban_timed_out": "info",
    "gateway.compression_aux_unavailable": "info",
    "gateway.compression_no_provider": "info",
    "gateway.compress_aux_model_failed": "info",
    "gateway.preflight_compression": "info",
    "gateway.stale_connections_cleaned": "info",
    "gateway.iteration_budget_exhausted": "info",
    "gateway.thinking_prefill_retry": "info",
}

_catalog_cache: dict[str, dict[str, str]] = {}
_catalog_lock = threading.Lock()

_overrides_cache: dict[str, str] | None = None
_overrides_lock = threading.Lock()

_suppress_cache: frozenset[str] | None = None
_suppress_lock = threading.Lock()


def _locales_dir() -> Path:
    """Return the directory containing locale YAML files.

    Resolution order, first existing wins:

    1. ``HERMES_BUNDLED_LOCALES`` env var -- set by the Nix wrapper (or any
       sealed-packaging system) to point at the installed catalog directory.
    2. ``<repo-root>/locales`` -- source checkouts and ``pip install -e .``,
       where the working tree sits next to ``agent/``.
    3. ``<sysconfig data|purelib|platlib>/locales`` -- pip wheel installs.
       setuptools ``data-files`` extracts ``locales/*.yaml`` under the
       interpreter's ``data`` scheme; the other schemes are checked as a
       safety net for nonstandard layouts.

    Falling through to the source-style path (even when missing) keeps
    ``_load_catalog`` error messages informative -- it logs the path it
    looked at -- rather than raising.
    """
    override = os.getenv("HERMES_BUNDLED_LOCALES", "").strip()
    if override:
        candidate = Path(override)
        if candidate.is_dir():
            return candidate
        logger.warning(
            "HERMES_BUNDLED_LOCALES points to a non-directory path (%s); "
            "falling back to bundled/source locale resolution",
            override,
        )

    # agent/i18n.py -> agent/ -> repo root (source checkout, editable install)
    source_dir = Path(__file__).resolve().parent.parent / "locales"
    if source_dir.is_dir():
        return source_dir

    # pip wheel install: data-files lands under the interpreter data scheme.
    # ``data`` (== sys.prefix in a venv) is where setuptools data-files extract
    # and is checked first. ``purelib``/``platlib`` (site-packages) are a safety
    # net for nonstandard layouts. NOTE: this does NOT cover ``pip install
    # --user`` (user scheme, ~/.local/locales) or ``pip install --target`` --
    # both are out of scope; see the plan header.
    for scheme in ("data", "purelib", "platlib"):
        raw = sysconfig.get_path(scheme)
        if not raw:
            continue
        candidate = Path(raw) / "locales"
        if candidate.is_dir():
            return candidate

    # Last resort: return the source-style path so _load_catalog's catalog-missing
    # log (logger.debug "i18n catalog missing for %s at %s") stays informative.
    return source_dir


def _normalize_lang(value: Any) -> str:
    """Normalize a user-supplied language value to a supported code.

    Accepts supported codes directly, common aliases (``chinese`` -> ``zh``),
    and case-insensitive regional tags (``zh-CN`` -> ``zh``).  Returns the
    default language for unknown values.
    """
    if not isinstance(value, str):
        return DEFAULT_LANGUAGE
    key = value.strip().lower()
    if not key:
        return DEFAULT_LANGUAGE
    if key in SUPPORTED_LANGUAGES:
        return key
    if key in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[key]
    # Try stripping a region suffix (e.g. "pt-br" -> "pt" won't be supported,
    # but "zh-CN" -> "zh" will).
    base = key.split("-", 1)[0]
    if base in SUPPORTED_LANGUAGES:
        return base
    return DEFAULT_LANGUAGE


def _load_catalog(lang: str) -> dict[str, str]:
    """Load and flatten one locale YAML file into a dotted-key dict.

    YAML files can be nested for human readability; this produces the flat
    key space :func:`t` expects.  Cached per-language for the process.
    """
    with _catalog_lock:
        cached = _catalog_cache.get(lang)
        if cached is not None:
            return cached

    path = _locales_dir() / f"{lang}.yaml"
    if not path.is_file():
        logger.debug("i18n catalog missing for %s at %s", lang, path)
        with _catalog_lock:
            _catalog_cache[lang] = {}
        return {}

    try:
        import yaml  # PyYAML is already a hermes dependency
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning("Failed to load i18n catalog %s: %s", path, exc)
        with _catalog_lock:
            _catalog_cache[lang] = {}
        return {}

    flat: dict[str, str] = {}
    _flatten_into(raw, "", flat)
    with _catalog_lock:
        _catalog_cache[lang] = flat
    return flat


def _flatten_into(node: Any, prefix: str, out: dict[str, str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            child_key = f"{prefix}.{key}" if prefix else str(key)
            _flatten_into(value, child_key, out)
    elif isinstance(node, str):
        out[prefix] = node
    # Non-string, non-dict leaves are ignored -- catalogs are text-only.


def _load_config_dict() -> dict:
    """Read config.yaml as a plain dict, or ``{}`` on any failure.

    Single seam for every config-derived i18n value (language, agent name,
    overrides) so tests can patch one function and callers stay crash-proof.
    """
    try:
        from hermes_cli.config import load_config
        return load_config() or {}
    except Exception as exc:  # config missing / malformed / import error
        logger.debug("i18n could not read config.yaml: %s", exc)
        return {}


@lru_cache(maxsize=1)
def _config_language_cached() -> str | None:
    """Read ``display.language`` from config.yaml once per process.

    Cached because ``t()`` is called in hot paths (every approval prompt,
    every gateway reply) and re-reading YAML each call would be wasteful.
    ``reset_language_cache()`` clears this when config changes at runtime
    (e.g. after the setup wizard).
    """
    cfg = _load_config_dict()
    lang = (cfg.get("display") or {}).get("language")
    if lang:
        return _normalize_lang(lang)
    return None


@lru_cache(maxsize=1)
def agent_display_name() -> str:
    """Configured agent name for ``{name}`` substitution; ``"Hermes"`` fallback.

    Sourced from the skin selected by ``display.skin``, matching the CLI/TUI
    branding path. Loading is cached with the other config-derived i18n state.
    """
    config = _load_config_dict()
    display = config.get("display") or {}
    if not isinstance(display, dict):
        display = {}
    skin_name = display.get("skin", "default")
    if not isinstance(skin_name, str) or not skin_name.strip():
        skin_name = "default"

    try:
        from hermes_cli.skin_engine import load_skin

        name = load_skin(skin_name.strip()).get_branding("agent_name", "Hermes")
        if isinstance(name, str) and name.strip():
            return name.strip()
    except Exception as exc:
        logger.debug("i18n could not resolve display skin branding: %s", exc)
    return "Hermes"


def reset_language_cache() -> None:
    """Invalidate cached language resolution and catalogs.

    Call after :func:`hermes_cli.config.save_config` if a running process
    needs to pick up a changed ``display.language`` without restart.
    """
    _config_language_cached.cache_clear()
    agent_display_name.cache_clear()
    with _catalog_lock:
        _catalog_cache.clear()
    global _overrides_cache
    with _overrides_lock:
        _overrides_cache = None
    global _suppress_cache
    with _suppress_lock:
        _suppress_cache = None


def get_language() -> str:
    """Resolve the active language using env > config > default order."""
    env_lang = os.environ.get("HERMES_LANGUAGE")
    if env_lang:
        return _normalize_lang(env_lang)
    cfg_lang = _config_language_cached()
    if cfg_lang:
        return cfg_lang
    return DEFAULT_LANGUAGE


class _MissingField(str):
    """Sentinel returned by ``_SafeFormatter.get_value`` for an unknown key.

    Subclasses ``str`` so the value already IS the ``"{key}"`` token and
    normal string operations still work.  The distinct type lets
    ``format_field`` unambiguously distinguish a re-emitted placeholder from
    a real kwarg value that happens to look like ``"{...}"``.
    """

    __slots__ = ()


class _SafeFormatter(string.Formatter):
    """``str.format`` that leaves unknown placeholders intact instead of raising.

    A user-supplied override template (``gateway.system_messages.*``) may contain
    a placeholder the call site never provides -- a typo, or ``{minutes}`` on a
    key that isn't the heartbeat.  Stock ``str.format`` raises on the first such
    token and the whole string is lost.  This formatter substitutes the kwargs it
    knows and re-emits the rest as the literal ``{token}`` so a broken template
    degrades gracefully instead of crashing the gateway.
    """

    def get_value(self, key: Any, args: Any, kwargs: Any) -> Any:
        if isinstance(key, str):
            return kwargs[key] if key in kwargs else _MissingField("{" + key + "}")
        try:
            return args[key]
        except (IndexError, KeyError):
            return _MissingField("{" + str(key) + "}")

    def get_field(self, field_name: str, args: Any, kwargs: Any) -> tuple[Any, Any]:
        """Preserve an unresolved compound field as its complete raw token.

        ``string.Formatter.get_field`` performs ``.attr`` / ``[item]``
        traversal after :meth:`get_value`. A missing root is therefore not
        enough on its own: traversing the ``_MissingField`` sentinel would
        otherwise raise ``AttributeError`` or ``TypeError``. Known roots whose
        requested attribute/item is absent degrade the same way.
        """
        root = field_name.split(".", 1)[0].split("[", 1)[0]
        lookup_key: Any = int(root) if root.isdecimal() else root
        root_value = self.get_value(lookup_key, args, kwargs)
        if isinstance(root_value, _MissingField):
            return _MissingField("{" + field_name + "}"), lookup_key
        try:
            return super().get_field(field_name, args, kwargs)
        except (AttributeError, KeyError, IndexError, TypeError):
            return _MissingField("{" + field_name + "}"), lookup_key

    def format_field(self, value: Any, format_spec: str) -> str:
        # If get_value returned a sentinel for a missing placeholder, reconstruct
        # the raw token -- with or without a format spec.
        if isinstance(value, _MissingField):
            if format_spec:
                # Reconstruct e.g. "{n:02d}" from token "{n}" + spec "02d".
                return value[:-1] + ":" + format_spec + "}"
            return str(value)
        try:
            return super().format_field(value, format_spec)
        except (ValueError, TypeError):
            # A real value with an unsatisfiable format spec degrades without
            # crashing (e.g. a non-numeric value passed where ":02d" expected).
            return str(value)


_SAFE_FORMATTER = _SafeFormatter()


def _safe_format(template: str, **kwargs: Any) -> str:
    """Format ``template`` with ``kwargs``; unknown ``{tokens}`` stay literal."""
    try:
        return _SAFE_FORMATTER.vformat(template, (), kwargs)
    except (AttributeError, KeyError, IndexError, TypeError, ValueError) as exc:
        logger.warning("i18n safe-format failed for template %r: %s", template, exc)
        return template


def _gateway_overrides() -> dict[str, str]:
    """Load ``gateway.system_messages`` overrides as a ``{full_key: template}`` map.

    ``gateway.system_messages.restart_success`` -> ``gateway.restart_success``.
    Cached for the process; ``reset_language_cache()`` clears it on config reload.
    """
    global _overrides_cache
    with _overrides_lock:
        if _overrides_cache is not None:
            return _overrides_cache
    result: dict[str, str] = {}
    raw = (_load_config_dict().get("gateway") or {}).get("system_messages") or {}
    if isinstance(raw, dict):
        for short_key, template in raw.items():
            if isinstance(template, str):
                result[f"gateway.{short_key}"] = template
    with _overrides_lock:
        if _overrides_cache is None:
            _overrides_cache = result
        return _overrides_cache


def _suppressed_categories() -> frozenset[str]:
    """Categories the user has muted via ``gateway.system_messages.suppress``.

    Accepts a list of category names or the string ``"all"`` (== all mutable
    categories). Non-mutable / unknown names are ignored with a warning. Cached
    for the process; ``reset_language_cache()`` clears it.
    """
    global _suppress_cache
    with _suppress_lock:
        if _suppress_cache is not None:
            return _suppress_cache
    raw = (_load_config_dict().get("gateway") or {}).get("system_messages") or {}
    spec = raw.get("suppress") if isinstance(raw, dict) else None
    result: set[str] = set()
    items = [spec] if isinstance(spec, str) else (spec if isinstance(spec, list) else [])
    for item in items:
        if item == "all":
            result |= set(_MUTABLE_CATEGORIES)
        elif item in _MUTABLE_CATEGORIES:
            result.add(item)
        elif item is not None:
            logger.warning("Ignoring non-suppressible/unknown system-message category %r", item)
    frozen = frozenset(result)
    with _suppress_lock:
        if _suppress_cache is None:
            _suppress_cache = frozen
        return _suppress_cache


def t(key: str, lang: str | None = None, **format_kwargs: Any) -> str:
    """Translate a dotted key to the active language.

    Parameters
    ----------
    key
        Dotted path into the catalog, e.g. ``"approval.choose_long"``.
    lang
        Explicit language override.  Takes precedence over env + config.
    **format_kwargs
        ``str.format`` substitution arguments (``t("gateway.drain", count=3)``
        expects a catalog entry with a ``{count}`` placeholder).

    Returns
    -------
    The translated string, or the English fallback if the key is missing in
    the target language, or the bare key if English is also missing.

    Notes
    -----
    ``gateway.*`` keys are checked against :func:`_gateway_overrides` first;
    a matching override short-circuits both the language catalog and the
    English fallback.
    """
    category = GATEWAY_MESSAGE_CATEGORIES.get(key)
    if category is not None and category in _suppressed_categories():
        return ""
    target = _normalize_lang(lang) if lang else get_language()
    # Gateway overrides win over the catalog (populated from gateway.system_messages).
    value = _gateway_overrides().get(key)
    if value is None:
        catalog = _load_catalog(target)
        value = catalog.get(key)

    if value is None and target != DEFAULT_LANGUAGE:
        # Fall through to English rather than showing a key path to the user.
        value = _load_catalog(DEFAULT_LANGUAGE).get(key)

    if value is None:
        # Last-ditch: return the key itself.  A broken catalog should not
        # crash anything; it just looks ugly until someone fixes it.
        logger.debug("i18n miss: key=%r lang=%r", key, target)
        value = key

    if "{name}" in value and "name" not in format_kwargs:
        format_kwargs = {**format_kwargs, "name": agent_display_name()}

    if format_kwargs:
        return _safe_format(value, **format_kwargs)
    return value


__all__ = [
    "SUPPORTED_LANGUAGES",
    "DEFAULT_LANGUAGE",
    "GATEWAY_MESSAGE_CATEGORIES",
    "t",
    "get_language",
    "reset_language_cache",
    "agent_display_name",
]
