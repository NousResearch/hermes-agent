"""The tenant's own identity: who they are, and what their workforce looks like.

Two files in the bundle carry it. ``organization.yaml`` is who the deployment serves —
legal name, region, contact — and its ``tenant_id`` is stamped on every audit event.
``identity.yaml`` is the customer-facing surface: product name, logo, colours, support
links, and the display name each agent introduces itself by.

Both were editable only by editing files on the server. This module is the seam that lets
the Control Centre change them, and like every other write it goes through
:func:`nova.spec.writer.edit` — so a change is validated against the whole bundle before
anything lands, and a malformed colour or a missing required field fails the same way it
would have if somebody had typed it into the YAML by hand.

**The logo is a file, not a URL.** The Control Centre serves under a strict
``img-src 'self' data:`` policy, so an external image would be blocked by the browser and
the screen would show a broken logo with no explanation. Uploading writes the bytes into
the bundle and the control plane serves them from its own origin, which keeps the policy
untouched.
"""

from __future__ import annotations

import base64
import binascii
from pathlib import Path
from typing import Any, Mapping, Optional

from nova.errors import SpecError
from nova.spec.writer import BundleEdit, edit, safe_relative

ORGANIZATION_FILE = "organization.yaml"
IDENTITY_FILE = "identity.yaml"

#: Where an uploaded logo lands. Fixed rather than caller-supplied: the path ends up in a
#: file the control plane serves, and a caller-chosen one is a traversal question nobody
#: needs to have.
BRANDING_DIR = "branding"

#: Image types a browser renders and that carry no script. SVG is deliberately absent: it
#: can contain script, and an SVG logo served from the control plane's own origin would run
#: with the control plane's privileges. A customer with an SVG logo converts it to PNG.
LOGO_TYPES: Mapping[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}

#: A logo is a wordmark, not a photograph. Large enough for a 2x retina asset, small enough
#: that it cannot be used to park a payload in the bundle.
MAX_LOGO_BYTES = 1_500_000

#: Fields the Control Centre may set, per file. Allowlisted for the same reason the agent
#: fields are: a body that could set any key could set one the loader ignores today and
#: honours tomorrow.
ORGANIZATION_FIELDS = ("legal_name", "region", "timezone", "contact_email")
IDENTITY_FIELDS = ("product_name", "company_name", "theme", "support", "messages")

#: ``tenant_id`` is not settable. It is stamped on every audit event already written, names
#: the runtime home, and is what isolation is keyed on — renaming it through a form would
#: orphan the history and silently move the tenant.
IMMUTABLE = ("tenant_id",)


def _check(fields: Mapping[str, Any], allowed: tuple[str, ...], what: str) -> None:
    unknown = sorted(set(fields) - set(allowed))
    if unknown:
        immutable = [name for name in unknown if name in IMMUTABLE]
        if immutable:
            raise SpecError(
                f"{', '.join(immutable)} cannot be changed. It is stamped on every audit "
                "event already recorded and names this tenant's runtime home; changing it "
                "would orphan the history rather than rename the tenant"
            )
        raise SpecError(
            f"cannot set {', '.join(unknown)} on {what}. Settable: {', '.join(allowed)}"
        )


def update_organization(root: Path, fields: Mapping[str, Any]):
    """Change who this deployment serves."""
    _check(fields, ORGANIZATION_FIELDS, "the organization")

    def mutate(e: BundleEdit) -> None:
        document = e.read_yaml(ORGANIZATION_FILE)
        if not document:
            raise SpecError(f"{ORGANIZATION_FILE} is missing from this bundle")
        for key, value in fields.items():
            if value in (None, ""):
                document.pop(key, None)
            else:
                document[key] = value
        e.write_yaml(ORGANIZATION_FILE, document)

    return edit(root, mutate)


def update_identity(root: Path, fields: Mapping[str, Any]):
    """Change the customer-facing surface: name, colours, support links, messages."""
    _check(fields, IDENTITY_FIELDS, "the identity")

    def mutate(e: BundleEdit) -> None:
        document = e.read_yaml(IDENTITY_FILE)
        for key, value in fields.items():
            if value in (None, "", {}):
                document.pop(key, None)
            elif isinstance(value, Mapping):
                # Merged rather than replaced: the theme and support blocks carry keys this
                # form may not offer, and a replace would silently drop them.
                merged = dict(document.get(key) or {})
                for inner, inner_value in value.items():
                    if inner_value in (None, ""):
                        merged.pop(inner, None)
                    else:
                        merged[inner] = inner_value
                if merged:
                    document[key] = merged
                else:
                    document.pop(key, None)
            else:
                document[key] = value
        e.write_yaml(IDENTITY_FILE, document)

    return edit(root, mutate)


def set_agent_display_name(root: Path, agent_id: str, display_name: str):
    """What one agent introduces itself as, without touching its id."""
    from nova.spec.writer import validate_id

    agent_id = validate_id(agent_id, what="agent id")

    def mutate(e: BundleEdit) -> None:
        document = e.read_yaml(IDENTITY_FILE)
        agents = dict(document.get("agents") or {})
        if display_name.strip():
            agents[agent_id] = {**dict(agents.get(agent_id) or {}),
                                "display_name": display_name.strip()}
        else:
            agents.pop(agent_id, None)
        if agents:
            document["agents"] = agents
        else:
            document.pop("agents", None)
        e.write_yaml(IDENTITY_FILE, document)

    return edit(root, mutate)


def _decode_logo(data: str, content_type: str) -> tuple[bytes, str]:
    """The image bytes and the extension to store it under, or a refusal.

    The declared content type is checked against the bytes themselves rather than trusted:
    a caller claiming ``image/png`` while sending HTML would otherwise get a file the
    control plane serves from its own origin.
    """
    extension = LOGO_TYPES.get(content_type.strip().lower())
    if extension is None:
        raise SpecError(
            f"{content_type!r} is not an image type this control plane stores. Use one of: "
            f"{', '.join(sorted(LOGO_TYPES))}. SVG is excluded on purpose — it can carry "
            "script, and a logo is served from the control plane's own origin"
        )
    # Tolerate a data: URI, which is what a browser's FileReader produces.
    if data.startswith("data:"):
        _, _, tail = data.partition(",")
        data = tail
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise SpecError("the logo must be base64-encoded image data") from None
    if not raw:
        raise SpecError("the logo is empty")
    if len(raw) > MAX_LOGO_BYTES:
        raise SpecError(
            f"a logo may not exceed {MAX_LOGO_BYTES // 1000} kB; this one is "
            f"{len(raw) // 1000} kB. It is a wordmark, not a photograph"
        )
    if not _looks_like(raw, content_type):
        raise SpecError(
            f"these bytes are not {content_type}. The declared type and the content must "
            "agree — this file is served from the control plane's own origin"
        )
    return raw, extension


#: Leading bytes each format must start with. Cheap, and enough to catch a file that is
#: something else entirely — which is the case that matters, not a subtly malformed PNG.
_MAGIC: Mapping[str, tuple[bytes, ...]] = {
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/jpeg": (b"\xff\xd8\xff",),
    "image/gif": (b"GIF87a", b"GIF89a"),
    "image/webp": (b"RIFF",),
}


def _looks_like(raw: bytes, content_type: str) -> bool:
    prefixes = _MAGIC.get(content_type.strip().lower(), ())
    if not prefixes:
        return False
    if content_type.strip().lower() == "image/webp":
        return raw.startswith(b"RIFF") and raw[8:12] == b"WEBP"
    return any(raw.startswith(prefix) for prefix in prefixes)


def set_logo(root: Path, *, data: str, content_type: str, kind: str = "logo"):
    """Store a logo or favicon in the bundle and point identity.yaml at it."""
    if kind not in ("logo", "favicon"):
        raise SpecError("kind must be 'logo' or 'favicon'")
    raw, extension = _decode_logo(data, content_type)
    relative = f"{BRANDING_DIR}/{kind}{extension}"

    def mutate(e: BundleEdit) -> None:
        # Replacing a logo with a different format must not leave the old file behind, or
        # the bundle accumulates one image per format anyone ever uploaded.
        for old in LOGO_TYPES.values():
            e.remove(f"{BRANDING_DIR}/{kind}{old}")
        e.write_bytes(relative, raw)
        document = e.read_yaml(IDENTITY_FILE)
        document[kind] = relative
        e.write_yaml(IDENTITY_FILE, document)

    return edit(root, mutate)


def clear_logo(root: Path, kind: str = "logo"):
    """Remove a logo and the reference to it."""
    if kind not in ("logo", "favicon"):
        raise SpecError("kind must be 'logo' or 'favicon'")

    def mutate(e: BundleEdit) -> None:
        for old in LOGO_TYPES.values():
            e.remove(f"{BRANDING_DIR}/{kind}{old}")
        document = e.read_yaml(IDENTITY_FILE)
        document.pop(kind, None)
        e.write_yaml(IDENTITY_FILE, document)

    return edit(root, mutate)


def logo_bytes(bundle, kind: str = "logo") -> Optional[tuple[bytes, str]]:
    """The stored image and its content type, or None when there is none.

    Resolved against the bundle root and refused if it escapes — the path comes from a YAML
    file an operator edits, which is not the same as a path NOVA wrote.
    """
    declared = str(getattr(bundle.identity, kind, "") or "").strip()
    if not declared:
        return None
    try:
        path = safe_relative(bundle.root, declared)
    except SpecError:
        return None
    if not path.is_file():
        return None
    suffix = path.suffix.lower()
    for content_type, extension in LOGO_TYPES.items():
        if extension == suffix:
            try:
                return path.read_bytes(), content_type
            except OSError:
                return None
    return None
