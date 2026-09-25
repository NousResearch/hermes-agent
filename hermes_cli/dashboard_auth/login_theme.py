"""Active dashboard theme -> CSS overrides for the server-rendered sign-in pages.

The ``/login`` page is pre-auth and has no SPA bundle, so ``ThemeProvider`` never runs there.
This module reads the same active *user* theme the index.html critical-CSS shim uses
(:func:`hermes_cli.web_server_dashboard.resolve_active_user_theme`) and turns its palette and
type into one ``:root`` override block appended to the page's own ``<style>``.

Pre-auth rules:

* Only CSS values leave this module — never the theme name, label, file path or profile.
* Every value is allow-listed by shape (hex colour, font-family list, CSS length). Theme YAML is
  trusted like ``config.yaml`` in the SPA (``customCSS`` passes through there), but a page
  served to unauthenticated visitors gets strict grammars: anything else is dropped, so a
  value cannot close the declaration, the rule or the ``<style>`` element.
* ``fontUrl``, ``assets`` and ``customCSS`` are ignored: no new requests, no free-form CSS.
* A missing or broken theme logs a warning and renders the built-in look, byte-identical.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from hermes_cli.web_server_dashboard import (
    _THEME_DEFAULT_TYPOGRAPHY, ActiveThemeUnavailable, resolve_active_user_theme)

_log = logging.getLogger(__name__)

_HEX_COLOR = re.compile(r"#(?:[0-9a-fA-F]{3,4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})\Z")
# One family: a quoted name (no quotes, backslashes or markup inside) or bare identifiers.
_FONT_FAMILY = re.compile(r"""\s*(?:"[\w .\-]+"|'[\w .\-]+'|[A-Za-z_\-][\w\-]*(?:\s+[\w\-]+)*)\s*\Z""")
_LENGTH = re.compile(r"(?:\d{1,3}(?:\.\d{1,3})?)(?:px|rem|em|%)\Z")
_FONT_LIST_MAX = 512


def _hex(palette: Dict[str, Any], key: str) -> Optional[str]:
    layer = palette.get(key)
    value = layer.get("hex") if isinstance(layer, dict) else None
    value = value.strip() if isinstance(value, str) else ""
    return value if _HEX_COLOR.match(value) else None


def _font_list(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.strip() or len(value) > _FONT_LIST_MAX:
        return None
    families = value.split(",")
    if not all(_FONT_FAMILY.match(f) for f in families):
        return None
    return ", ".join(f.strip() for f in families)


def _declarations(theme: Dict[str, Any]) -> List[str]:
    """Validated ``:root`` declarations plus element rules, in emit order; [] when unusable."""
    palette = theme.get("palette") if isinstance(theme.get("palette"), dict) else {}
    typo = theme.get("typography") if isinstance(theme.get("typography"), dict) else {}
    rules: List[str] = []

    background, midground = _hex(palette, "background"), _hex(palette, "midground")
    if background and midground:
        # Same roles the SPA gives them: page ground, and ink/accent (the shim paints
        # ``color: var(--midground-base)``; the foreground layer is an overlay, not text).
        rules.append(
            ":root{"
            f"--background-base:{background};--background:{background};"
            f"--midground:{midground};--foreground:{midground};"
            f"--hairline:color-mix(in srgb, {midground} 18%, transparent);"
            f"--hairline-strong:color-mix(in srgb, {midground} 35%, transparent);"
            "}"
        )
    elif palette:
        _log.warning("dashboard theme palette is not two valid hex colours; login page keeps its default palette")

    # The normaliser back-fills typography defaults; only what the theme itself sets replaces the
    # page's own type, so a palette-only theme keeps the built-in faces and 16px scale.
    own = {k: v for k, v in typo.items()
           if isinstance(v, str) and v.strip() and v != _THEME_DEFAULT_TYPOGRAPHY.get(k)}
    font_sans, font_display = _font_list(own.get("fontSans")), _font_list(own.get("fontDisplay"))
    base_size = own.get("baseSize", "").strip()
    base_size = base_size if _LENGTH.match(base_size) else None
    if (own.get("fontSans") and not font_sans) or (own.get("fontDisplay") and not font_display) \
            or (own.get("baseSize") and not base_size):
        _log.warning("dashboard theme typography value rejected for the login page (not a plain family list / length)")
    font_display = font_display or font_sans
    if font_sans:
        rules.append(f"html,body,.provider-btn,.field-input{{font-family:{font_sans};}}")
    if font_display:
        rules.append(f".brand,h1,.form-title{{font-family:{font_display};}}")
    if base_size:
        rules.append(f"html,body{{font-size:{base_size};}}")
    return rules


def render_login_theme_css() -> str:
    """CSS text to append inside the sign-in page's ``<style>``; "" for the built-in look."""
    try:
        theme = resolve_active_user_theme()
    except ActiveThemeUnavailable:
        _log.warning("active dashboard theme could not be loaded; login page uses its default styling")
        return ""
    except Exception:
        _log.warning("dashboard theme lookup failed; login page uses its default styling", exc_info=True)
        return ""
    if not theme:
        return ""
    rules = _declarations(theme)
    if not rules:
        return ""
    return "  /* Active dashboard theme (validated values only). */\n  " + "\n  ".join(rules) + "\n"
