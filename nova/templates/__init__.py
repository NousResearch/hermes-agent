"""Ready-made business templates: a working tenant bundle for a kind of business.

A template is a complete bundle (agents, prompts, policy, knowledge, channels, a recurring
job, an objective, budgets) with a handful of placeholders — the company's name, tenant id,
support email, region, time zone, model and monthly budget. ``new_bundle`` fills them in
and then **loads the result as a real bundle**; a template that does not load is refused
and nothing is left behind, so a new client never starts from something half-working.

Placeholder values are checked before they are used. The company name in particular is
limited to characters that cannot change the meaning of the YAML it is written into, so a
name like ``Acme: "Ltd"`` is refused with a sentence rather than producing a bundle that
parses into something else.

Templates never carry secrets. What a client must still provide — tokens, a service
account, channel credentials — is listed in each template's ``template.yaml`` under
``needs`` and printed by ``nova template show``.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from nova._contact import check_email, check_timezone
from nova.errors import NovaError, SpecError

#: Where the templates live, one directory each.
LIBRARY = Path(__file__).parent / "library"

#: The metadata file inside each template; everything else is the bundle.
META_FILE = "template.yaml"

#: Files whose text is rendered. Anything else (none today) is copied as-is.
_RENDERED = {".yaml", ".yml", ".md", ".txt"}

_PLACEHOLDER = re.compile(r"\{\{\s*([a-z_]+)\s*\}\}")
_TENANT = re.compile(r"^[a-z0-9][a-z0-9-]{0,38}[a-z0-9]$")
#: Letters and digits of any script, spaces and a few punctuation marks — nothing that
#: means something to YAML inside a double-quoted string or at the start of a scalar.
_COMPANY = re.compile(r"^[\w][\w .,&'()\-]{0,79}$")
_REGION = re.compile(r"^[a-z]{2}(-[a-z]+)+-\d$")
_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")

DEFAULTS = {
    "region": "eu-west-2",
    "timezone": "Europe/London",
    "model": "eu.anthropic.claude-sonnet-4-6",
    "monthly_budget": "150",
}


@dataclass(frozen=True)
class Template:
    """One template's description, from its ``template.yaml``."""

    id: str
    title: str
    summary: str
    for_whom: str
    agents: tuple[dict, ...] = ()
    needs: tuple[dict, ...] = ()
    path: Path = field(default=LIBRARY, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "summary": self.summary,
                "for_whom": self.for_whom, "agents": list(self.agents), "needs": list(self.needs)}


def catalogue() -> list[Template]:
    """Every template in the library, by id."""
    out = []
    for directory in sorted(LIBRARY.iterdir()) if LIBRARY.is_dir() else ():
        meta_path = directory / META_FILE
        if not meta_path.is_file():
            continue
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
        out.append(Template(
            id=directory.name,
            title=str(meta.get("title", directory.name)),
            summary=str(meta.get("summary", "")),
            for_whom=str(meta.get("for_whom", "")),
            agents=tuple(meta.get("agents") or ()),
            needs=tuple(meta.get("needs") or ()),
            path=directory,
        ))
    return out


def get(template_id: str) -> Template:
    for template in catalogue():
        if template.id == template_id:
            return template
    known = ", ".join(t.id for t in catalogue()) or "(none)"
    raise NovaError(f"no template {template_id!r}; available: {known}")


def check_values(values: Mapping[str, Any]) -> dict[str, str]:
    """The placeholder values, checked and with defaults filled. Raises ``SpecError``."""
    merged = {**DEFAULTS, **{k: str(v).strip() for k, v in values.items() if v not in (None, "")}}
    for required in ("tenant_id", "company", "support_email"):
        if not merged.get(required):
            raise SpecError(f"{required} is required", field=required)
    if not _TENANT.match(merged["tenant_id"]):
        raise SpecError("tenant_id must be 2-40 lowercase letters, digits or hyphens, starting "
                        "and ending with a letter or digit", field="tenant_id")
    if not _COMPANY.match(merged["company"]):
        raise SpecError("company may use letters, digits, spaces and . , & ' ( ) - only "
                        "(up to 80 characters)", field="company")
    check_email(merged["support_email"], field="support_email", source=None)
    if not _REGION.match(merged["region"]):
        raise SpecError(f"{merged['region']!r} is not an AWS region such as eu-west-2", field="region")
    check_timezone(merged["timezone"], field="timezone", source=None)
    if not _MODEL.match(merged["model"]):
        raise SpecError(f"{merged['model']!r} is not a model id", field="model")
    try:
        budget = float(merged["monthly_budget"])
    except ValueError:
        raise SpecError("monthly_budget must be a number of US dollars", field="monthly_budget") from None
    if budget <= 0:
        raise SpecError("monthly_budget must be more than zero", field="monthly_budget")
    merged["monthly_budget"] = f"{budget:g}"
    return merged


def _render(text: str, values: Mapping[str, str], where: Path) -> str:
    def replace(match: "re.Match[str]") -> str:
        key = match.group(1)
        if key not in values:
            raise NovaError(f"{where}: placeholder {{{{{key}}}}} has no value")
        return values[key]

    return _PLACEHOLDER.sub(replace, text)


def new_bundle(template_id: str, destination: Path, values: Mapping[str, Any]):
    """Write a filled-in bundle to ``destination`` and return it loaded.

    Refuses a destination that exists and is not empty. On any failure — a bad value, a
    template that does not load — the destination is removed, so a failed attempt leaves
    nothing a later ``nova apply`` could pick up by mistake.
    """
    from nova.spec import load_bundle

    template = get(template_id)
    checked = check_values(values)
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise NovaError(f"{destination} is not empty; choose a new directory for the bundle")
    created = not destination.exists()
    try:
        for source in sorted(template.path.rglob("*")):
            relative = source.relative_to(template.path)
            if relative.name == META_FILE and relative.parent == Path("."):
                continue
            target = destination / relative
            if source.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix in _RENDERED:
                target.write_text(_render(source.read_text(encoding="utf-8"), checked, relative),
                                  encoding="utf-8")
            else:
                shutil.copy2(source, target)
        return load_bundle(destination)
    except Exception:
        if created:
            shutil.rmtree(destination, ignore_errors=True)
        else:
            for child in destination.iterdir():
                shutil.rmtree(child) if child.is_dir() else child.unlink()
        raise
