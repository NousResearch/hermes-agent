"""Render opt-out / legal request text from templates/ with safe substitution.

Templates use {field} placeholders. Missing fields are left literal (never crash,
never inject blanks that look like real data). Field values come from the
least-disclosure selection in dossier.select_disclosure.
"""
from __future__ import annotations

from pathlib import Path

import paths


class _SafeDict(dict):
    def __missing__(self, key):  # leave unknown placeholders untouched
        return "{" + key + "}"


def template_path(name: str) -> Path:
    return paths.templates_dir() / name


def render(template_name: str, fields: dict) -> str:
    text = template_path(template_name).read_text(encoding="utf-8")
    return text.format_map(_SafeDict(fields))


def _join_listings(value) -> str:
    if isinstance(value, (list, tuple)):
        return "\n".join(str(v) for v in value)
    return str(value or "")


def _join_identifiers(value) -> str:
    """Render the subject's OWN identifiers as a bullet list for an indirect-exposure request."""
    if isinstance(value, (list, tuple)):
        return "\n".join(f"  - {v}" for v in value if v)
    return f"  - {value}" if value else ""


def render_optout_email(broker: dict, fields: dict) -> str:
    ctx = dict(fields)
    ctx.setdefault("broker_name", broker.get("name", "the data broker"))
    ctx["listing_urls"] = _join_listings(fields.get("listing_urls"))
    ctx.setdefault("full_name", fields.get("full_name", "[your name]"))
    ctx.setdefault("contact_email", fields.get("contact_email", "[your email]"))
    return render("emails/generic-optout.txt", ctx)


def render_request(kind: str, broker: dict, fields: dict) -> str:
    """kind: generic | ccpa | ccpa_agent | ccpa_indirect | gdpr"""
    template = {
        "generic": "emails/generic-optout.txt",
        "ccpa": "emails/ccpa-deletion.txt",
        "ccpa_agent": "emails/ccpa-authorized-agent.txt",
        "ccpa_indirect": "emails/ccpa-indirect-deletion.txt",
        "gdpr": "emails/gdpr-erasure.txt",
    }.get(kind, "emails/generic-optout.txt")
    ctx = dict(fields)
    ctx.setdefault("broker_name", broker.get("name", "the data broker"))
    ctx["listing_urls"] = _join_listings(fields.get("listing_urls"))
    ctx["my_identifiers"] = _join_identifiers(fields.get("my_identifiers"))
    return render(template, ctx)
