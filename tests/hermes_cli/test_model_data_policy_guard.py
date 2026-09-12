"""Tests for the data-training-tier selection guard."""

from hermes_cli.model_data_policy_guard import (
    DataTrainingWarning,
    data_training_warning,
)


def test_fires_on_meta_contributor():
    w = data_training_warning("muse-spark-1.2-contributor", provider="meta-ai")
    assert isinstance(w, DataTrainingWarning)
    assert w.model == "muse-spark-1.2-contributor"
    assert "train" in w.message.lower()
    assert "-contributor" in w.message.lower() or "contributor" in w.message.lower()  # mentions the tier
    # Points to Meta's live pricing page instead of hardcoding prices.
    assert "pricing and rate limits" in w.message.lower()
    assert "$0.10" not in w.message and "$0.20" not in w.message and "$0.002" not in w.message
    assert "prompts and completions" in w.message.lower()
    assert "dev.meta.ai/docs/pricing-rate-limits" in w.message


def test_silent_on_non_contributor_muse():
    assert data_training_warning("muse-spark-1.2", provider="meta-ai") is None
    assert data_training_warning("muse-spark-1.1", provider="meta-ai") is None


def test_silent_on_unrelated_models():
    for m in ("anthropic/claude-opus-4.8", "gpt-5.6-sol", "deepseek-v4-pro", ""):
        assert data_training_warning(m, provider="anthropic") is None


def test_fires_regardless_of_provider_string():
    # id-keyed: must fire even if selected via custom/gateway (no meta-ai provider)
    assert data_training_warning("muse-spark-1.2-contributor", provider="custom") is not None
    assert data_training_warning("muse-spark-1.2-contributor") is not None


def test_case_insensitive():
    assert data_training_warning("MUSE-SPARK-1.2-CONTRIBUTOR", provider="meta-ai") is not None
