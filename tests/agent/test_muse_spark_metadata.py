"""Contract tests: Meta Muse Spark family metadata in Hermes.

Grounded in Meta's own docs (Muse Spark = 1M context window; 1.3 spec =
1M context / 131,072 max output). No live metadata source carries muse
models (Zen/Go /v1/models cards are bare; models.dev has no muse entries),
so Hermes must resolve the family from its own static defaults — otherwise
every muse-spark* slug silently probing-down to the 256K fallback.
"""

from agent.model_metadata import get_model_context_length

MUSE_1M = 1_048_576


def test_muse_spark_family_context_length_is_1m():
    """Every muse-spark* slug resolves to the family 1M window, not 256K."""
    for model in (
        "muse-spark-1.2",
        "muse-spark-1.2-contributor",
        "muse-spark-1.2-contributor-free",
        "muse-spark-1.3-contributor-free",
        # future checkpoint ids keep matching via longest-first substring
        "muse-spark-1.4-contributor-free",
    ):
        ctx = get_model_context_length(
            model, base_url="http://127.0.0.1:1/v1"  # unreachable endpoint
        )
        assert ctx == MUSE_1M, f"{model} -> {ctx}, expected {MUSE_1M}"


def test_opencode_floors_carry_muse_13_free():
    """Picker floors list the live free 1.3 checkpoint (both opencode families)."""
    from hermes_cli.models import _PROVIDER_MODELS

    for family in ("opencode-zen", "opencode-free"):
        floors = _PROVIDER_MODELS[family]
        assert "muse-spark-1.3-contributor-free" in floors, family
        # the 1.2 free checkpoint stays as offline fallback
        assert "muse-spark-1.2-contributor-free" in floors, family