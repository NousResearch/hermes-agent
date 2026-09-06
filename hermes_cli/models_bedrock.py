"""AWS Bedrock catalog seam for ``hermes_cli.models``: live discovery, the curated fallback with
its cache provenance, and the ``bedrock.discovery.model_allowlist`` policy identity.

Split out of ``hermes_cli.models`` along the ``<stem>_<topic>`` decomposition that produced
``models_catalog_static``, ``models_local`` and ``models_validate``. ``hermes_cli.models``
re-imports the names, so ``hermes_cli.models.<name>`` stays the stable import/monkeypatch surface.

- ``_bedrock_catalog`` — live discovery via ``agent.bedrock_adapter``. Under an allowlist the
  curated static table (plus the Mantle ids the control plane never lists) is the fallback,
  projected through the same allowlist: the table names the very ids an allowlist is usually
  written to hide, so an unfiltered fallback would silently re-admit them.
- ``_StaticFallbackModelIds`` — provenance of that fallback. The disk-cache layer serves it for
  the current open only: persisted with live authority, the stub would cap the picker at the
  offline list for the full TTL after credentials recover (#74151; #74207 applies the same rule
  to the unfiltered ``None`` fall-through in ``provider_model_ids``).
- ``_bedrock_policy_fingerprint_part`` — the normalized allowlist, serialized for
  ``models._credential_fingerprint`` so a policy change invalidates the cache row instead of
  serving the old projection until TTL expiry.
"""

from __future__ import annotations

from typing import Optional

from hermes_cli.models_catalog_static import _PROVIDER_MODELS

__all__ = [
    "_StaticFallbackModelIds",
    "_bedrock_catalog",
    "_bedrock_policy_fingerprint_part",
]


class _StaticFallbackModelIds(list):
    """Model ids taken from the curated static table because live discovery returned nothing.

    A plain ``list`` to every consumer; the type is the provenance ``cached_provider_model_ids``,
    ``update_provider_cache_entry`` and the SWR refresh check before writing a row to
    ``provider_models_cache.json``. It rides on the returned value so the picker prefetch, which
    re-persists what ``cached_provider_model_ids`` returned, can refuse it as well.
    """


def _bedrock_catalog(normalized: str, force_refresh: bool) -> Optional[list[str]]:
    # Live discovery keyed by the resolved AWS region so EU/AP users see eu.*/ap.* ids.
    try:
        from agent.bedrock_adapter import (bedrock_model_ids_or_none, configured_bedrock_model_allowlist,
                                           filter_bedrock_model_ids, merge_bedrock_openai_model_ids)

        live = bedrock_model_ids_or_none()
        if live is not None:
            return live
        allowlist = configured_bedrock_model_allowlist()
        if not allowlist:
            return None
        # No live catalog under an allowlist (no credentials, denied ListFoundationModels, or nothing
        # matched). The curated fallback must obey the allowlist too: _PROVIDER_MODELS["bedrock"]
        # lists the very ids an allowlist is usually written to hide (openai.gpt-5.6-*, deepseek.v3.2,
        # us.meta.llama4-*). The Mantle ids are merged into the pool first, so an allowlist naming
        # only a Mantle id does not depend on the curated table happening to carry it. An empty
        # projection is returned as-is (no Bedrock models); the tag keeps the stub out of the disk
        # cache either way.
        curated = merge_bedrock_openai_model_ids(list(_PROVIDER_MODELS.get("bedrock", [])))
        return _StaticFallbackModelIds(filter_bedrock_model_ids(curated, allowlist))
    except Exception:
        return None


def _bedrock_policy_fingerprint_part() -> str:
    """Serialized ``bedrock.discovery.model_allowlist`` for the cache credential fingerprint.

    The allowlist decides which ids discovery may return, so a disk row written under one policy
    must not be served under another: folding the normalized policy into the fingerprint makes a
    widening ``[A] -> [A, B]`` (or any change) invalidate the row, and the next picker open
    re-discovers instead of serving the old projection until TTL expiry.
    """
    try:
        from agent.bedrock_adapter import configured_bedrock_model_allowlist
        allowlist = configured_bedrock_model_allowlist()
        return "bedrock.discovery.model_allowlist=" + "|".join(sorted(allowlist))
    except Exception:
        return "bedrock.discovery.model_allowlist=unreadable"
