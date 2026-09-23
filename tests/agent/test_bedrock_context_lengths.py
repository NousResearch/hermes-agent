"""The Bedrock Mantle OpenAI models carry a 1M context window, not 272K.

Every AWS model card for these ids states "Context window: 1M tokens"; 272K appears only as the
pricing-tier boundary ("short context (272K input tokens or fewer)" vs "long context (more than
272K input tokens)"), and GPT-5.5's card spells it out: "Long-context rates apply to all input and
output tokens, not just tokens above 272K." Pinning the table to the tier boundary compressed these
sessions at 27% of the real window."""

from agent.bedrock_adapter import (
    BEDROCK_CONTEXT_LENGTHS,
    BEDROCK_OPENAI_RESPONSES_MODEL_IDS,
    get_bedrock_context_length,
)


def test_bedrock_openai_responses_models_are_1m_in_the_static_table():
    assert BEDROCK_OPENAI_RESPONSES_MODEL_IDS, "allowlist must not be empty"
    for model_id in BEDROCK_OPENAI_RESPONSES_MODEL_IDS:
        assert BEDROCK_CONTEXT_LENGTHS[model_id] == 1_000_000, model_id


def test_resolver_returns_1m_without_a_live_probe():
    """The substring resolver must not fall through to a shorter key or the 128K default."""
    for model_id in BEDROCK_OPENAI_RESPONSES_MODEL_IDS:
        assert get_bedrock_context_length(model_id, probe=False) == 1_000_000, model_id
