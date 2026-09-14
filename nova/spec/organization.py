"""OrganizationSpec — who this deployment belongs to.

The tenant id is the one value that must be stamped on every task and every structured
log line. NOVA is deliberately single-tenant per deployment today; carrying the id
anyway costs nothing and is what keeps multi-tenancy from being foreclosed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from nova._fields import Doc


@dataclass(frozen=True)
class OrganizationSpec:
    """The customer this deployment serves."""

    tenant_id: str
    legal_name: str = ""
    region: str = ""
    timezone: str = ""
    contact_email: str = ""
    source: Optional[Path] = None

    @classmethod
    def parse(
        cls,
        data: Mapping[str, Any],
        *,
        source: Optional[Path] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> "OrganizationSpec":
        doc = Doc(data, source=source, env=env)
        spec = cls(
            tenant_id=doc.identifier("tenant_id"),
            legal_name=doc.str_("legal_name"),
            region=doc.str_("region"),
            timezone=doc.str_("timezone"),
            contact_email=doc.str_("contact_email"),
            source=source,
        )
        doc.reject_unknown()
        return spec

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"tenant_id": self.tenant_id}
        for key, value in (
            ("legal_name", self.legal_name),
            ("region", self.region),
            ("timezone", self.timezone),
            ("contact_email", self.contact_email),
        ):
            if value:
                out[key] = value
        return out
