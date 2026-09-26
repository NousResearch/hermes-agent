"""The picker's per-model ``fast`` capability must respect the route.

The request builders only send fast-mode params to the first-party endpoint
that bills for them (``resolve_fast_mode_overrides``), and ``session.info``
reports Fast through the same gate (fd602278c7). The picker payload is the
third consumer: a ``fast: true`` entry on a route that can never receive the
param makes the desktop/TUI composer offer a Fast toggle that silently does
nothing — the exact inconsistency fd602278c7 fixed for session.info.
"""

import hermes_cli.inventory as inv


def _caps(row: dict) -> dict:
    rows = [row]
    inv._apply_capabilities(rows)
    slug_models = rows[0]["capabilities"]
    return {model: caps["fast"] for model, caps in slug_models.items()}


def test_aggregator_row_never_claims_fast():
    """A first-party fast model served through OpenRouter cannot receive the param."""
    caps = _caps({"slug": "openrouter", "models": ["anthropic/claude-opus-5", "gpt-5.4"]})
    assert caps["anthropic/claude-opus-5"] is False
    assert caps["gpt-5.4"] is False


def test_first_party_row_keeps_fast():
    caps = _caps({"slug": "anthropic", "models": ["claude-opus-5"]})
    assert caps["claude-opus-5"] is True
    caps = _caps({"slug": "openai", "models": ["gpt-5.4"]})
    assert caps["gpt-5.4"] is True


def test_custom_proxy_row_never_claims_fast():
    """A user-defined endpoint proxying gpt-5.4 is not the billing endpoint."""
    caps = _caps({
        "slug": "custom:my-proxy", "api_url": "https://proxy.example.com/v1",
        "models": ["gpt-5.4"],
    })
    assert caps["gpt-5.4"] is False


def test_local_row_never_claims_fast():
    caps = _caps({"slug": "llamacpp", "api_url": "http://127.0.0.1:18434/v1", "models": ["gpt-5.4"]})
    assert caps["gpt-5.4"] is False
