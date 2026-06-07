"""Contract test: hindsight-embed must be pinned below 0.7.

Version 0.7.x removed the ``HindsightEmbedded`` class from the ``hindsight``
package namespace, breaking the ``local_embedded`` mode import at runtime.
This test ensures the pin is not accidentally removed.

See: https://github.com/NousResearch/hermes-agent/issues/41126
"""

from pathlib import Path

import pytest


def test_hindsight_embed_pinned_below_07():
    """hindsight-embed must be pinned <0.7 to prevent ImportError."""
    pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
    text = pyproject.read_text()

    for line in text.splitlines():
        if line.strip().startswith("hindsight ="):
            assert "hindsight-embed" in line, (
                "hindsight-embed must be listed in the hindsight extras to pin <0.7"
            )
            assert "<0.7" in line or "<=0.6" in line, (
                "hindsight-embed must be capped below 0.7 — "
                "0.7.x removed HindsightEmbedded class"
            )
            return

    pytest.fail("hindsight extras not found in pyproject.toml")
