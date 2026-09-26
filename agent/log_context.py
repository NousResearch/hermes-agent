"""Canonical ``provider=/base_url=/model=`` log context formatting.

Several conversation-loop log lines report which provider/base URL/model an
API call went to. Historically each call site formatted these fields by hand,
which produced inconsistent key orders ``(model=X, provider=Y)``,
``(provider=X, model=Y)``, missing fields entirely (no ``base_url``), and
inconsistent fallbacks when a field was empty. With custom providers the
bare ``provider=custom`` value is ambiguous -- only ``base_url`` identifies
the actual gateway -- so omitting it makes failure triage guesswork.

This module is the single formatter every such line should use. The
``key=value value ...`` shape matches the existing convention in
``run_agent.AIAgent._client_log_context()`` and ``agent/stream_diag.py``.
"""
from __future__ import annotations

from typing import Any


def model_provider_fields(agent: Any) -> str:
    """Return ``provider=<p> base_url=<u> model=<m>`` for *agent*.

    Missing values render as ``unknown`` so the keys stay greppable even on
    partially-initialized agents.
    """
    provider = str(getattr(agent, "provider", "") or "unknown").strip()
    base_url = str(getattr(agent, "base_url", "") or "unknown").strip()
    model = str(getattr(agent, "model", "") or "unknown").strip()
    return f"provider={provider} base_url={base_url} model={model}"
