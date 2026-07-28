"""Lightweight i18n for Hermes' static user-facing strings (approval prompts, a few gateway replies).

Catalogs are ``locales/<lang>.yaml`` flattened to dotted keys. Missing keys
fall back to English, then to the key itself, so a broken catalog never crashes.
Language resolution: explicit ``lang=`` > ``HERMES_LANGUAGE`` > ``display.language`` > ``en``.
"""

from __future__ import annotations

import logging
import os
import re
import string
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES: tuple[str, ...] = (
    "en", "zh", "zh-hant", "ja", "de", "es", "fr", "tr", "uk",
    "af", "ko", "it", "ga", "pt", "ru", "hu", "ar",
)
DEFAULT_LANGUAGE = "en"

# Natural aliases so "chinese" / "zh-CN" / "jp" hit the right catalog instead of
# silently falling back to English. Bare "chinese" defaults to Simplified;
# Taiwan/HK/Macau tags route to the distinct Traditional catalog. pt-br shares
# the pt catalog (no separate br one).
_LANGUAGE_ALIASES: dict[str, str] = {
    "english": "en", "en-us": "en", "en-gb": "en",
    "chinese": "zh", "mandarin": "zh", "zh-cn": "zh", "zh-hans": "zh", "zh-sg": "zh",
    "traditional-chinese": "zh-hant", "traditional_chinese": "zh-hant",
    "zh-tw": "zh-hant", "zh-hk": "zh-hant", "zh-mo": "zh-hant",
    "japanese": "ja", "jp": "ja", "ja-jp": "ja",
    "german": "de", "deutsch": "de", "de-de": "de", "de-at": "de", "de-ch": "de",
    "spanish": "es", "español": "es", "espanol": "es", "es-es": "es", "es-mx": "es", "es-ar": "es",
    "french": "fr", "français": "fr", "france": "fr", "fr-fr": "fr", "fr-be": "fr", "fr-ca": "fr", "fr-ch": "fr",
    "ukrainian": "uk", "ukrainisch": "uk", "українська": "uk", "uk-ua": "uk", "ua": "uk",
    "turkish": "tr", "türkçe": "tr", "tr-tr": "tr",
    "afrikaans": "af", "af-za": "af",
    "korean": "ko", "한국어": "ko", "ko-kr": "ko",
    "italian": "it", "italiano": "it", "it-it": "it", "it-ch": "it",
    "irish": "ga", "gaeilge": "ga", "ga-ie": "ga",
    "portuguese": "pt", "português": "pt", "portugues": "pt",
    "pt-pt": "pt", "pt-br": "pt", "brazilian": "pt", "brasileiro": "pt",
    "russian": "ru", "русский": "ru", "ru-ru": "ru",
    "hungarian": "hu", "magyar": "hu", "hu-hu": "hu",
    "arabic": "ar", "العربية": "ar",
    "ar-sa": "ar", "ar-eg": "ar", "ar-ae": "ar", "ar-ma": "ar", "ar-dz": "ar",
}

_MUTABLE_CATEGORIES = frozenset({"progress", "lifecycle", "info"})
# Only notification messages whose callers treat an empty string as "do not
# send" belong here. Errors, approvals and command replies remain visible.
GATEWAY_MESSAGE_CATEGORIES: dict[str, str] = {
    "gateway.long_running": "progress",
    "gateway.no_activity_warning": "progress",
    "gateway.subagent_working": "progress",
    "gateway.queued_next_turn": "progress",
    "gateway.interrupting_task": "progress",
    "gateway.compaction_done": "progress",
    "gateway.steered_into_run": "progress",
<<<<<<< HEAD
=======
    # The compaction START status is already swallowed by the gateway noise
    # regex; this completion edge is deliberately delivered to chat instead
    # (test_compaction_completion_notice_reaches_chat), so muting it needs
    # this category entry — the noise filter will never do it.
    "gateway.compaction_done": "progress",
    # lifecycle — gateway daemon lifecycle notifications
>>>>>>> 92b3bde16c0f (feat(gateway): make the compaction-completion notice suppressible)
    "gateway.restart_success": "lifecycle",
    "gateway.gateway_online": "lifecycle",
    "gateway.shutdown_restarting": "lifecycle",
    "gateway.shutdown_shutting_down": "lifecycle",
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
_overrides_cache: dict[str, dict[str, str]] = {}
_overrides_lock = threading.Lock()
_suppress_cache: dict[str, frozenset[str]] = {}
_suppress_lock = threading.Lock()


def _locales_dir() -> Path:
    """Locale dir: ``HERMES_BUNDLED_LOCALES`` (sealed packaging, e.g. Nix) if it exists, else ``<repo-root>/locales``.

    The source path is returned even when missing so ``_load_catalog`` can log
    the path it looked at rather than raise.
    """
    override = os.getenv("HERMES_BUNDLED_LOCALES", "").strip()
    if override and Path(override).is_dir():
        return Path(override)
    if override:
        logger.warning(
            "HERMES_BUNDLED_LOCALES points to a non-directory path (%s); "
            "falling back to bundled/source locale resolution", override,
        )
    return Path(__file__).resolve().parent.parent / "locales"


def _normalize_lang(value: Any) -> str:
    """Map a user-supplied value (code, alias, or regional tag like ``zh-CN``) to a supported code, else default."""
    key = value.strip().lower() if isinstance(value, str) else ""
    if key in SUPPORTED_LANGUAGES:
        return key
    if key in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[key]
    base = key.split("-", 1)[0]  # strip region suffix
    return base if base in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE


def _cache_catalog(lang: str, flat: dict[str, str]) -> dict[str, str]:
    with _catalog_lock:
        _catalog_cache[lang] = flat
    return flat


def _load_catalog(lang: str) -> dict[str, str]:
    """Load one locale YAML flattened to dotted keys; cached per language (empty dict on any failure)."""
    with _catalog_lock:
        cached = _catalog_cache.get(lang)
        if cached is not None:
            return cached

    path = _locales_dir() / f"{lang}.yaml"
    flat: dict[str, str] = {}
    if not path.is_file():
        logger.debug("i18n catalog missing for %s at %s", lang, path)
        return _cache_catalog(lang, flat)
    try:
        import yaml
        with path.open("r", encoding="utf-8") as f:
            _flatten_into(yaml.safe_load(f) or {}, "", flat)
    except Exception as exc:
        logger.warning("Failed to load i18n catalog %s: %s", path, exc)
        flat = {}
    return _cache_catalog(lang, flat)


def _flatten_into(node: Any, prefix: str, out: dict[str, str]) -> None:
    # Non-string, non-dict leaves are ignored -- catalogs are text-only.
    if isinstance(node, dict):
        for key, value in node.items():
            _flatten_into(value, f"{prefix}.{key}" if prefix else str(key), out)
    elif isinstance(node, str):
        out[prefix] = node


@lru_cache(maxsize=8)
def _config_language_cached(hermes_home: str) -> str | None:
    """``display.language`` from config.yaml, read once per profile home (``t()`` is a hot path).
    Keyed by home so a multiplexed gateway serving several profiles doesn't freeze the first
    profile's language for every other profile."""
    try:
        from hermes_cli.config import load_config_readonly
        lang = (load_config_readonly().get("display") or {}).get("language")
        return _normalize_lang(lang) if lang else None
    except Exception as exc:
        logger.debug("Could not read display.language from config: %s", exc)
        return None


def _config_language() -> str | None:
    from hermes_constants import get_hermes_home
    return _config_language_cached(str(get_hermes_home()))


def _config_dict() -> dict[str, Any]:
    """Read the active profile's configuration without making translation fatal."""
    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
        return config if isinstance(config, dict) else {}
    except Exception as exc:
        logger.debug("Could not read config for i18n: %s", exc)
        return {}


def _profile_cache_key() -> str:
    from hermes_constants import get_hermes_home
    return str(get_hermes_home())


@lru_cache(maxsize=8)
def agent_display_name(hermes_home: str | None = None) -> str:
    """Configured branding name for ``{name}`` substitutions."""
    config = _config_dict()
    display = config.get("display") or {}
    skin_name = display.get("skin", "default") if isinstance(display, dict) else "default"
    try:
        from hermes_cli.skin_engine import load_skin
        name = load_skin(str(skin_name or "default")).get_branding("agent_name", "Hermes")
        return name.strip() if isinstance(name, str) and name.strip() else "Hermes"
    except Exception as exc:
        logger.debug("Could not resolve i18n branding: %s", exc)
        return "Hermes"


def _gateway_overrides() -> dict[str, str]:
    """Return the active profile's ``gateway.system_messages`` overrides.

    Config keys use short names (``restart_success``); catalog keys retain the
    ``gateway.`` prefix.  The cache is profile-scoped so a multiplexed gateway
    cannot leak one profile's custom wording into another profile's session.
    """
    profile_key = _profile_cache_key()
    with _overrides_lock:
        cached = _overrides_cache.get(profile_key)
        if cached is not None:
            return cached
    raw = (_config_dict().get("gateway") or {}).get("system_messages") or {}
    overrides = {
        f"gateway.{name}": template
        for name, template in raw.items()
        if isinstance(name, str) and isinstance(template, str)
    } if isinstance(raw, dict) else {}
    with _overrides_lock:
        return _overrides_cache.setdefault(profile_key, overrides)


def _suppressed_categories() -> frozenset[str]:
    """Return active-profile notification categories muted by configuration."""
    profile_key = _profile_cache_key()
    with _suppress_lock:
        cached = _suppress_cache.get(profile_key)
        if cached is not None:
            return cached
    raw = (_config_dict().get("gateway") or {}).get("system_messages") or {}
    spec = raw.get("suppress") if isinstance(raw, dict) else None
    values = [spec] if isinstance(spec, str) else spec if isinstance(spec, list) else []
    result: set[str] = set()
    for value in values:
        if value == "all":
            result.update(_MUTABLE_CATEGORIES)
        elif value in _MUTABLE_CATEGORIES:
            result.add(value)
        elif value is not None:
            logger.warning("Ignoring non-suppressible system-message category %r", value)
    frozen = frozenset(result)
    with _suppress_lock:
        return _suppress_cache.setdefault(profile_key, frozen)


class _MissingField(str):
    """A format placeholder that remains visible when an override omits data."""


class _SafeFormatter(string.Formatter):
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
        if isinstance(value, _MissingField):
            return value[:-1] + (":" + format_spec if format_spec else "") + "}"
        try:
            return super().format_field(value, format_spec)
        except (ValueError, TypeError):
            return str(value)


_SAFE_FORMATTER = _SafeFormatter()


@lru_cache(maxsize=8)
def _gateway_message_matchers(lang: str) -> tuple[dict[str, str], tuple[tuple[re.Pattern[str], str], ...]]:
    """Index English gateway templates for delivery-time compatibility translation.

    Hermes historically emitted many static gateway messages as raw strings.
    The refactored gateway now spreads those sites over dedicated modules; this
    bridge keeps existing catalog entries effective while those paths migrate
    to direct ``t()`` calls.  It only matches a complete known English system
    template, so model-generated replies and ordinary user text pass through.
    """
    exact: dict[str, str] = {}
    patterns: list[tuple[re.Pattern[str], str]] = []
    for key, template in _load_catalog(DEFAULT_LANGUAGE).items():
        if not key.startswith("gateway."):
            continue
        parsed = list(string.Formatter().parse(template))
        if not any(field is not None for _, field, _, _ in parsed):
            exact.setdefault(template, key)
            continue
        parts: list[str] = ["^"]
        fields: set[str] = set()
        usable = True
        for literal, field, _spec, _conversion in parsed:
            parts.append(re.escape(literal))
            if field is not None:
                if not field.isidentifier():
                    usable = False
                    break
                if field in fields:
                    parts.append(f"(?P={field})")
                else:
                    parts.append(f"(?P<{field}>.+?)")
                    fields.add(field)
        if usable:
            parts.append("$")
            patterns.append((re.compile("".join(parts), re.DOTALL), key))
    return exact, tuple(patterns)


def localize_gateway_message(message: str, lang: str | None = None) -> str:
    """Translate one complete known gateway system message at delivery time.

    New call sites should prefer :func:`t`; this is deliberately a narrow
    compatibility bridge for static strings that upstream moved into modules
    during the i18n branch's rebase.
    """
    if not isinstance(message, str) or not message:
        return message
    target = _normalize_lang(lang) if lang else get_language()
    if target == DEFAULT_LANGUAGE:
        return message
    exact, patterns = _gateway_message_matchers(target)
    key = exact.get(message)
    if key:
        return t(key, lang=target)
    for pattern, candidate in patterns:
        match = pattern.fullmatch(message)
        if match:
            return t(candidate, lang=target, **match.groupdict())
    return message


def _safe_format(template: str, **kwargs: Any) -> str:
    try:
        return _SAFE_FORMATTER.vformat(template, (), kwargs)
    except (AttributeError, KeyError, IndexError, TypeError, ValueError) as exc:
        logger.warning("i18n safe-format failed for template %r: %s", template, exc)
        return template


def reset_language_cache() -> None:
    """Invalidate cached language resolution and catalogs (call after ``save_config`` changes ``display.language``)."""
    _config_language_cached.cache_clear()
    agent_display_name.cache_clear()
    _gateway_message_matchers.cache_clear()
    with _catalog_lock:
        _catalog_cache.clear()
    with _overrides_lock:
        _overrides_cache.clear()
    with _suppress_lock:
        _suppress_cache.clear()


def get_language() -> str:
    """Resolve the active language using env > config > default order. ``HERMES_LANGUAGE`` is a
    per-profile ``.env`` value, so it is read through the secret scope: under multiplexing a raw
    environ read would impose the default profile's language on every other profile."""
    from agent.secret_scope import UnscopedSecretError, get_secret
    try:
        env_lang = get_secret("HERMES_LANGUAGE")
    except UnscopedSecretError:
        env_lang = os.environ.get("HERMES_LANGUAGE")  # unscoped default-profile path: environ IS its own value
    return _normalize_lang(env_lang) if env_lang else _config_language() or DEFAULT_LANGUAGE


def t(key: str, lang: str | None = None, **format_kwargs: Any) -> str:
    """Translate a dotted catalog key to the active (or explicit ``lang``) language.

    ``format_kwargs`` are applied with ``str.format``. Falls back to English,
    then to the bare key; a format failure returns the unformatted string.
    """
    category = GATEWAY_MESSAGE_CATEGORIES.get(key)
    if category in _suppressed_categories():
        return ""
    target = _normalize_lang(lang) if lang else get_language()
    value = _gateway_overrides().get(key) if key.startswith("gateway.") else None
    if value is None:
        value = _load_catalog(target).get(key)
    if value is None and target != DEFAULT_LANGUAGE:
        value = _load_catalog(DEFAULT_LANGUAGE).get(key)
    if value is None:
        logger.debug("i18n miss: key=%r lang=%r", key, target)
        value = key
    if "{name}" in value and "name" not in format_kwargs:
        format_kwargs = {**format_kwargs, "name": agent_display_name(_profile_cache_key())}
    return _safe_format(value, **format_kwargs) if format_kwargs else value


__all__ = [
    "SUPPORTED_LANGUAGES", "DEFAULT_LANGUAGE", "GATEWAY_MESSAGE_CATEGORIES", "t", "get_language",
    "localize_gateway_message", "reset_language_cache", "agent_display_name",
]
