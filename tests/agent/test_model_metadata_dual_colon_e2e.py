"""Cross-validation: ::-style id resolves a real context window end-to-end (#148755).

Mirrors the E2E leg of our forge/148755 commit's tests: with the ::-guard in place,
the full id reaches the metadata catalog and returns its real window; without it
(base), the mangled id misses and the resolver warn-logs the 256K fallback — the
production symptom verbatim.
"""

from unittest.mock import MagicMock, patch

import providers
from providers.base import ProviderProfile


def test_dual_colon_id_resolves_real_window_not_256k_fallback(monkeypatch):
    import logging

    from agent.model_metadata import (
        _FALLBACK_WARNED,
        _strip_provider_prefix,
        get_model_context_length,
    )

    monkeypatch.setattr(providers, "_REGISTRY", {})
    monkeypatch.setattr(providers, "_ALIASES", {})
    monkeypatch.setattr(providers, "_PROVIDER_LIST_CACHE", None)
    monkeypatch.setattr(providers, "_discovered", True)
    providers.register_provider(ProviderProfile(name="tensorzero"))
    _FALLBACK_WARNED.clear()

    model = "tensorzero::function_name::mechanical"
    assert _strip_provider_prefix(model) == model

    mock_fetch = MagicMock()
    mock_fetch.return_value = {
        model: {"context_length": 131072},
    }
    records = []
    handler = logging.Handler()
    handler.emit = lambda record: records.append(record.getMessage())
    lg = logging.getLogger("agent.model_metadata")
    old_level = lg.level
    lg.setLevel(logging.WARNING)
    lg.addHandler(handler)
    try:
        with patch("agent.model_metadata.fetch_model_metadata", mock_fetch):
            got = get_model_context_length(model)
    finally:
        lg.removeHandler(handler)
        lg.setLevel(old_level)

    assert got == 131072, f"expected real window 131072, got {got}"
    warns = [m for m in records if "falling back to" in m]
    assert not warns, f"unexpected fallback warning: {warns}"
