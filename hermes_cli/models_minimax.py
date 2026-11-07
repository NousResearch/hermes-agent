"""MiniMax OAuth model catalog discovery.

The ``minimax-oauth`` picker row in ``hermes_cli.models_catalog_static`` is a hand-written
snapshot, so a model MiniMax ships (e.g. MiniMax-M3.1-Flash) stays missing from every picker
surface until Hermes ships a release. MiniMax exposes the account's real catalog on the
Anthropic-compatible route at ``{inference_base_url}/v1/models``; this module probes it.

Split out of ``hermes_cli.models`` (a facade, already at its line cap): the provider profile
in ``plugins/model-providers/minimax`` wires this in through its ``picker_catalog`` hook, and
the helpers used here are looked up on ``hermes_cli.models`` at call time so
``patch("hermes_cli.models.<name>")`` mocks keep intercepting.
"""

from __future__ import annotations

import logging
from typing import Optional

# Log-record parity with the origin module.
logger = logging.getLogger("hermes_cli.models")


def minimax_oauth_catalog(force_refresh: bool = False) -> Optional[list[str]]:
    """Live MiniMax OAuth catalog from the Anthropic-compatible ``/v1/models`` endpoint.

    Live ids lead (the account's actual list), then curated-only extras so a curated model
    the endpoint omits (stale rollout, region gating) still appears.

    Read-only like the Codex picker (#68004): a picker never imports, refreshes or persists a
    credential. The stored access token is read as-is; when it has expired the probe 401s and
    the curated row is served until the runtime lease refreshes it.
    ``resolve_minimax_oauth_runtime_credentials`` is deliberately NOT used here — it refreshes
    and rewrites auth.json.
    """
    from hermes_cli.models import _get_json, _merge_unique, _PROVIDER_MODELS

    try:
        from hermes_cli.auth import get_provider_auth_state

        state = get_provider_auth_state("minimax-oauth") or {}
        token = str(state.get("access_token") or "").strip()
        base_url = str(state.get("inference_base_url") or "").strip().rstrip("/")
        if not token or not base_url:
            return None
        headers = {
            "Authorization": f"Bearer {token}",
            "x-api-key": token,
            "anthropic-version": "2023-06-01",
        }
        data = _get_json(f"{base_url}/v1/models", timeout=5.0, headers=headers)
        live = [str(m["id"]) for m in (data.get("data") or []) if isinstance(m, dict) and m.get("id")]
        if not live:
            return None
        curated = list(_PROVIDER_MODELS.get("minimax-oauth", []))
        return _merge_unique(live, curated)
    except Exception as e:
        logger.debug("Failed to fetch MiniMax OAuth models: %s", e, exc_info=True)
        return None
