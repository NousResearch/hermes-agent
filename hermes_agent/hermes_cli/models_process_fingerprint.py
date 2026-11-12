"""Catalog-cache identity of an external-process provider (``hermes_agent.hermes_cli.models._credential_fingerprint``)."""

from __future__ import annotations

import os
from typing import Any


def process_catalog_identity(provider: str) -> tuple[Any | None, list[str]]:
    """``(profile, parts)`` for an external-process provider, else ``(None, [])``.

    Such a provider discovers models through the launched program, so the command / argv env
    overrides and the resolved binary identify the catalog the way an API key identifies an HTTP one.
    """
    from hermes_agent.providers import get_provider_profile

    profile = get_provider_profile(provider)
    if profile is None or profile.auth_type != "external_process":
        return None, []
    parts = [f"{ev}={os.environ.get(ev, '')}"
             for ev in (*profile.process_command_env_vars, profile.process_args_env_var) if ev]
    from hermes_agent.hermes_cli.auth_external_process import external_process_command, resolve_external_process_command
    binary = resolve_external_process_command(external_process_command(profile))
    parts.append(f"binary={os.path.realpath(binary) if binary else ''}")
    return profile, parts
