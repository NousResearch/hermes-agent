"""IdentitySpec — the white-label surface, declared once per tenant.

Branding is *compiled* into the surfaces a runtime already reads, never looked up by
runtime code at execution time. This module holds only the declaration; A runtime adapter's projector turns it
into that runtime's own artefacts.

The default product name lives here as a fallback, not as an identifier: nothing in
NOVA's code, paths, environment variables or wire formats carries the product name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from nova._fields import Doc

#: Used when a tenant does not name its own product. A display string only.
DEFAULT_PRODUCT_NAME = "NOVA AI Workforce"


@dataclass(frozen=True)
class ThemeSpec:
    """Colour tokens handed to display surfaces.

    Kept as opaque strings: NOVA does not validate colour formats, because the set of
    surfaces that consume them (terminal, web, email) accept different notations and a
    premature check here would reject a legitimate one.
    """

    accent: str = ""
    surface: str = ""
    on_accent: str = ""

    @classmethod
    def parse(cls, doc: Optional[Doc]) -> "ThemeSpec":
        if doc is None:
            return cls()
        spec = cls(
            accent=doc.str_("accent"),
            surface=doc.str_("surface"),
            on_accent=doc.str_("on_accent"),
        )
        doc.reject_unknown()
        return spec

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in (
                ("accent", self.accent),
                ("surface", self.surface),
                ("on_accent", self.on_accent),
            )
            if value
        }


@dataclass(frozen=True)
class SupportSpec:
    """Where a user of the branded product goes when they need a human."""

    email: str = ""
    url: str = ""

    @classmethod
    def parse(cls, doc: Optional[Doc]) -> "SupportSpec":
        if doc is None:
            return cls()
        spec = cls(email=doc.str_("email"), url=doc.str_("url"))
        doc.reject_unknown()
        return spec

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in (("email", self.email), ("url", self.url)) if value}


@dataclass(frozen=True)
class IdentitySpec:
    """One tenant's customer-facing identity."""

    product_name: str = DEFAULT_PRODUCT_NAME
    company_name: str = ""
    logo: str = ""
    favicon: str = ""
    theme: ThemeSpec = field(default_factory=ThemeSpec)
    support: SupportSpec = field(default_factory=SupportSpec)
    welcome: str = ""
    goodbye: str = ""
    #: agent id -> display name. Lets an operator see "Acme Support Assistant" while the
    #: identifier stays ``customer-support`` everywhere it is addressed.
    agent_display_names: Mapping[str, str] = field(default_factory=dict)
    source: Optional[Path] = None

    @classmethod
    def parse(
        cls,
        data: Mapping[str, Any],
        *,
        source: Optional[Path] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> "IdentitySpec":
        doc = Doc(data, source=source, env=env)
        agents_doc = doc.child("agents")
        display: dict[str, str] = {}
        if agents_doc is not None:
            for agent_id in agents_doc.keys():
                entry = agents_doc.child(agent_id)
                if entry is not None:
                    name = entry.str_("display_name")
                    entry.reject_unknown()
                    if name:
                        display[agent_id] = name
            agents_doc.reject_unknown()

        messages = doc.child("messages")
        welcome = goodbye = ""
        if messages is not None:
            welcome = messages.str_("welcome")
            goodbye = messages.str_("goodbye")
            messages.reject_unknown()

        spec = cls(
            product_name=doc.str_("product_name", default=DEFAULT_PRODUCT_NAME),
            company_name=doc.str_("company_name"),
            logo=doc.str_("logo"),
            favicon=doc.str_("favicon"),
            theme=ThemeSpec.parse(doc.child("theme")),
            support=SupportSpec.parse(doc.child("support")),
            welcome=welcome,
            goodbye=goodbye,
            agent_display_names=display,
            source=source,
        )
        doc.reject_unknown()
        return spec

    def display_name_for(self, agent_id: str, fallback: str) -> str:
        """The operator-facing name for an agent, or ``fallback`` when unbranded."""
        return self.agent_display_names.get(agent_id) or fallback

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"product_name": self.product_name}
        for key, value in (
            ("company_name", self.company_name),
            ("logo", self.logo),
            ("favicon", self.favicon),
            ("welcome", self.welcome),
            ("goodbye", self.goodbye),
        ):
            if value:
                out[key] = value
        for key, section in (("theme", self.theme), ("support", self.support)):
            rendered = section.to_dict()
            if rendered:
                out[key] = rendered
        if self.agent_display_names:
            out["agent_display_names"] = dict(sorted(self.agent_display_names.items()))
        return out
