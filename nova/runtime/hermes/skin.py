"""Project an IdentitySpec into a Hermes skin document.

The runtime loads user skins from its skins directory ahead of its own built-ins, so
writing a skin file rebrands the terminal surface with **no runtime code change at all**.
That is the whole reason branding is compiled rather than looked up: the projector
produces a file the runtime already wanted to read.

This lives in the adapter package because the skin schema belongs to the runtime, not to
NOVA. A second runtime brands itself differently and brings its own projector; what stays
shared is :class:`~nova.spec.identity.IdentitySpec`, which is expressed in NOVA's terms.
"""

from __future__ import annotations

from typing import Any

from nova.spec import IdentitySpec

#: Fallback glyph for the response label. Neutral on purpose — an unbranded deployment
#: should not display a symbol that belongs to a specific product.
DEFAULT_SYMBOL = "*"


def skin_filename(tenant_id: str) -> str:
    """Skin file name for a tenant. One skin per tenant, named for it."""
    return f"{tenant_id}.yaml"


def build_skin(identity: IdentitySpec, *, tenant_id: str) -> dict[str, Any]:
    """The skin document for this tenant's identity.

    Only fields the tenant actually set are emitted; the runtime's own defaults fill the
    rest, so a partial identity produces a partially rebranded surface rather than a
    surface with blanks in it.
    """
    product = identity.product_name
    symbol = DEFAULT_SYMBOL

    branding: dict[str, Any] = {
        "agent_name": product,
        "response_label": f" {symbol} {product} ",
        "help_header": f"({symbol}) Available Commands",
    }
    branding["welcome"] = identity.welcome or (
        f"Welcome to {product}. Type your message or /help for commands."
    )
    if identity.goodbye:
        branding["goodbye"] = identity.goodbye

    skin: dict[str, Any] = {
        "name": tenant_id,
        "description": _description(identity),
        "branding": branding,
    }

    colors = _colors(identity)
    if colors:
        skin["colors"] = colors
    return skin


def _description(identity: IdentitySpec) -> str:
    if identity.company_name:
        return f"{identity.product_name} — {identity.company_name}"
    return identity.product_name


def _colors(identity: IdentitySpec) -> dict[str, str]:
    """Map NOVA's three theme tokens onto the display roles a skin colours.

    A deliberately small mapping: NOVA declares intent (accent, surface, on-accent) and
    the projector decides which display roles those drive. Exposing the runtime's full
    colour vocabulary in the tenant bundle would tie customer configuration to one
    runtime's internals.
    """
    theme = identity.theme
    colors: dict[str, str] = {}
    if theme.accent:
        colors["ui_accent"] = theme.accent
        colors["banner_title"] = theme.accent
        colors["response_border"] = theme.accent
    if theme.on_accent:
        colors["banner_text"] = theme.on_accent
    if theme.surface:
        colors["status_bar_bg"] = theme.surface
    return colors
