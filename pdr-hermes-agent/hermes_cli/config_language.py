"""``display.language`` validation for ``hermes config set`` — pluggable languages.

The accepted set is ``agent.i18n.supported_languages()``: bundled catalogs ∪ the profile's overlay dir
∪ every loaded plugin language pack. Packs only exist after discovery, and ``hermes config`` never
imports ``model_tools`` (the usual discovery trigger), so this runs ``discover_plugins()`` first —
otherwise ``hermes config set display.language pl`` would refuse the very pack the user just installed.
"""

from __future__ import annotations

from typing import Optional

DISPLAY_LANGUAGE_KEY = "display.language"


def resolve_display_language(value: str) -> Optional[str]:
    """Canonical id for *value* (alias/region-tolerant) or ``None`` when no layer supplies it."""
    from agent.i18n import reset_language_cache, resolve_language_id
    try:
        from hermes_cli.plugins import discover_plugins
        discover_plugins()
    except Exception:
        pass  # a broken plugin tree must not block setting a bundled language
    reset_language_cache()  # packs registered during discovery must be visible to this check
    return resolve_language_id(value)


def display_language_error(value: str) -> Optional[str]:
    """User-facing refusal for an unsupported ``display.language`` value, ``None`` when acceptable.
    Auto-detect (empty) is always accepted."""
    if not str(value).strip():
        return None
    if resolve_display_language(value) is not None:
        return None
    from agent.i18n import supported_languages
    return (f"✗ Unknown language {value!r} for {DISPLAY_LANGUAGE_KEY}. Available: "
            f"{', '.join(supported_languages())}.\n  Install a language pack plugin "
            "(hermes plugins install <pack>) or drop <HERMES_HOME>/locales/<id>.yaml to add one.")


__all__ = ["DISPLAY_LANGUAGE_KEY", "display_language_error", "resolve_display_language"]
