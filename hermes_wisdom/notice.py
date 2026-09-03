"""Shared presentation copy for profile-local qualification notices."""

from __future__ import annotations

from typing import Any


def qualification_notice(event: dict[str, Any]) -> str:
    if event.get("notice_variant") == "first":
        organization_name = event.get("organization_name")
        organization = (
            f"Your organisation ({str(organization_name).strip()})"
            if isinstance(organization_name, str) and organization_name.strip()
            else "Your organisation"
        )
        return (
            f"{organization} has enabled Collective Wisdom, a feature designed to "
            "automatically detect useful skills across all team members.\n\n"
            "Congratulations! Hermes detected a skill that could be useful to your team."
        )
    return "Hermes detected another skill that could be useful to your team."
