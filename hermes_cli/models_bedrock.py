"""AWS Bedrock catalog seam for ``hermes_cli.models``.

Split out of ``hermes_cli.models`` along the ``<stem>_<topic>`` decomposition that produced
``models_catalog_static``, ``models_local`` and ``models_validate``. ``hermes_cli.models``
re-imports the names, so ``hermes_cli.models.<name>`` stays the stable import/monkeypatch surface.

- ``_bedrock_catalog`` — live discovery via ``agent.bedrock_adapter``; ``None`` when there is no
  live catalog, so ``provider_model_ids`` falls through to the curated static list.
"""

from __future__ import annotations

from typing import Optional

__all__ = ["_bedrock_catalog"]


def _bedrock_catalog(normalized: str, force_refresh: bool) -> Optional[list[str]]:
    # Live discovery keyed by the resolved AWS region so EU/AP users see eu.*/ap.* ids.
    try:
        from agent.bedrock_adapter import bedrock_model_ids_or_none

        return bedrock_model_ids_or_none()
    except Exception:
        return None
