"""grok-4.20 has a 1M context window, not 2M.

docs.x.ai/docs/models/grok-4.20 lists "Context window 1,000,000", and the models index carries
grok-4.20-0309-reasoning, grok-4.20-0309-non-reasoning and grok-4.20-multi-agent-0309 at 1M. The
2M entry over-advertised the window, so a session only started compressing after the endpoint had
already begun rejecting the request."""

from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS, get_model_context_length

# Dated/variant ids resolve to the "grok-4.20" catalog key by substring match.
_GROK_420_IDS = (
    "grok-4.20",
    "grok-4.20-0309-reasoning",
    "grok-4.20-0309-non-reasoning",
    "grok-4.20-multi-agent-0309",
)


def test_grok_420_and_its_variants_resolve_to_1m():
    for model_id in _GROK_420_IDS:
        assert get_model_context_length(model_id, provider="xai") == 1_000_000, model_id


def test_grok_420_catalog_entry_is_1m():
    assert DEFAULT_CONTEXT_LENGTHS["grok-4.20"] == 1_000_000


def test_grok_4_fast_keeps_its_own_2m_window():
    """Only the 4.20 row moved; grok-4-fast is documented at 2M and shares the stale-key comment."""
    assert DEFAULT_CONTEXT_LENGTHS["grok-4-fast"] == 2_000_000
