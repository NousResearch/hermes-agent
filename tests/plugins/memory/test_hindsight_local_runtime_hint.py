"""NousResearch/hermes-agent#7718 — actionable message when local_embedded
runtime (`hindsight-all`) is missing.

`local_embedded` imports `from hindsight import HindsightEmbedded`, provided
only by `hindsight-all`. When it's absent the provider disables itself; the
disable warning should point the user at the fix rather than just echoing
`No module named 'hindsight'`.
"""

import sys

from plugins.memory.hindsight import _local_runtime_hint


def test_hint_for_missing_hindsight_all():
    hint = _local_runtime_hint("No module named 'hindsight'")
    assert "hindsight-all" in hint
    assert "hermes memory setup" in hint
    assert sys.executable in hint


def test_hint_for_missing_hindsight_embed():
    hint = _local_runtime_hint("No module named 'hindsight_embed.daemon_embed_manager'")
    assert "hindsight-all" in hint


def test_no_hint_for_unrelated_runtime_error():
    # e.g. the NumPy-on-old-CPU failure _check_local_runtime also guards against
    assert _local_runtime_hint("Illegal instruction (NumPy SIMD)") == ""
    assert _local_runtime_hint(None) == ""
