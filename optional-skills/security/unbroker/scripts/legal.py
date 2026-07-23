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
    """kind: generic | ccpa | ccpa_agent | ccpa_indirect | gdpr | gdpr_art21_only | gdpr_indirect

    Jurisdictional templates (gdpr_*) cite the relevant GDPR articles and the EU Charter
    Article 8 right to data protection. gdpr_art21_only targets brokers that process data
    under a legitimate-interest claim; gdpr_indirect mirrors ccpa_indirect for cases where
    the subject appears on someone else's record.
    """
    template = {
        "generic": "emails/generic-optout.txt",
        "ccpa": "emails/ccpa-deletion.txt",
        "ccpa_agent": "emails/ccpa-authorized-agent.txt",
        "ccpa_indirect": "emails/ccpa-indirect-deletion.txt",
        "gdpr": "emails/gdpr-erasure.txt",
        "gdpr_art21_only": "emails/gdpr-art21-only.txt",
        "gdpr_indirect": "emails/gdpr-indirect-deletion.txt",
    }.get(kind, "emails/generic-optout.txt")
    ctx = dict(fields)
    ctx.setdefault("broker_name", broker.get("name", "the data broker"))
    ctx["listing_urls"] = _join_listings(fields.get("listing_urls"))
    ctx["my_identifiers"] = _join_identifiers(fields.get("my_identifiers"))
    return render(template, ctx)


def render_dpa_complaint(dpa_id: str, fields: dict) -> str:
    """Render an Art. 77 supervisory-authority complaint for the named DPA.

    dpa_id: the DPA adapter id (e.g. 'garante', 'cnil', 'ico', 'bfdi', or 'generic' as
    a fallback when the subject's national DPA has no custom template). The template
    resolution prefers references/dpa/<dpa_id>.json's 'complaint_template' field, then
    falls back to templates/dpa-complaints/<dpa_id>.txt.

    Guaranteed never to crash: if neither the requested template nor the generic fallback
    exist (e.g. on a fresh install before phase-2 DPA templates land), we emit a minimal
    hand-editable complaint skeleton the subject can complete and send manually.
    """
    template_name = f"dpa-complaints/{dpa_id}.txt"
    try:
        import dpa as dpa_mod  # local import avoids coupling render_request() to the DPA registry
        adapter = dpa_mod.get(dpa_id)
        if adapter and adapter.get("complaint_template"):
            template_name = adapter["complaint_template"]
    except Exception:  # noqa: BLE001 - template rendering must never fail because metadata is absent
        pass
    path = template_path(template_name)
    if not path.exists():
        path = template_path("dpa-complaints/generic.txt")
    if not path.exists():
        # Last-resort skeleton — keeps the function's no-crash contract. The subject
        # gets a minimum-viable complaint they can paste into their DPA's web form.
        return (
            f"Subject: GDPR Article 77 complaint — {fields.get('broker_name', '[broker]')}\n\n"
            f"To Whom It May Concern,\n\n"
            f"I am {fields.get('full_name', '[your name]')} ({fields.get('contact_email', '[your email]')}). "
            f"I am filing this complaint under Article 77 of the General Data Protection Regulation "
            f"(EU) 2016/679 regarding {fields.get('broker_name', '[broker]')}, which has failed to "
            f"respond satisfactorily to my Article 17 erasure request of "
            f"{fields.get('request_date', '[date]')}.\n\n"
            f"[Please describe the facts, attach evidence of your prior request, and state the remedy you seek.]\n\n"
            f"Sincerely,\n{fields.get('full_name', '[your name]')}\n"
        )
    # `render()` expects the full template path including .txt suffix (it joins onto
    # templates_dir() and reads directly), unlike render_request which dispatches by name.
    return render(f"{path.parent.name}/{path.name}", fields)
