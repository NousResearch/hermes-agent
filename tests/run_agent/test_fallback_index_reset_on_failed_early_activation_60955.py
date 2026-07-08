"""Regression test for #60955: fallback chain index consumed by failed early
rate-limit activation prevents terminal retry-exhaustion path from retrying.

Uses direct file-inspection instead of module import (avoids the
conversation_loop.py import cascade that can hang on Windows).

Bug: when the rate-limit handler in conversation_loop.py calls
_try_activate_fallback() and it fails (returns False), the internal recursive
call already incremented _fallback_index past len(_fallback_chain).  The
terminal "max retries — trying fallback" path checks _has_pending_fallback()
which sees _fallback_index >= len(chain) and returns False — fallback is never
retried after the retry budget burns out.

Fix: reset _fallback_index = 0 after the early fallback activation fails,
so the terminal path can retry instead of finding the chain exhausted.
"""
from __future__ import annotations

import re
from pathlib import Path

# Resolve the source file relative to this test file
_HERE = Path(__file__).resolve().parent
_SRC = (_HERE / "../../agent/conversation_loop.py").resolve()


def _get_source() -> str:
    return _SRC.read_text(encoding="utf-8")


class TestFallbackIndexResetOnFailedEarlyActivation:
    """The rate-limit early-fallback block must reset ``_fallback_index = 0``
    when ``_try_activate_fallback`` returns False, so the retry-exhaustion
    terminal path can attempt fallback after the retry budget is consumed."""

    def test_fallback_index_reset_present(self):
        src = _get_source()
        # The fix block: just after `continue` at end of the
        # `if _try_activate_fallback:` block in the rate-limit handler,
        # there must be a reset guarded by a "Fallback activation failed"
        # comment referencing #60955.
        match = re.search(
            r"(?s)"  # DOTALL
            r"# Fallback activation failed.*?"
            r"agent\._fallback_index\s*=\s*0",
            src,
        )
        assert match is not None, (
            "Failed early-activation fallback index reset NOT found in "
            "conversation_loop.py — the rate-limit fallback block must reset "
            "_fallback_index = 0 after _try_activate_fallback returns False. "
            "See #60955."
        )

    def test_exactly_one_fallback_reset(self):
        """Ensure exactly one reset guarded by the 'failed activation' comment
        (distinct from the primary-recovery reset and other resets)."""
        src = _get_source()
        matches = re.findall(
            r"(?s)# Fallback activation failed.*?_fallback_index\s*=\s*0",
            src,
        )
        assert len(matches) == 1, (
            f"Expected exactly 1 fallback-activation-failure reset, "
            f"found {len(matches)}"
        )

    def test_both_resets_exist(self):
        """The primary-recovery reset (in the `_try_recover_primary_transport`
        block) and our new rate-limit reset must both exist independently."""
        src = _get_source()
        all_resets = re.findall(r"_fallback_index\s*=\s*0", src)
        assert len(all_resets) >= 2, (
            f"Expected at least 2 _fallback_index = 0 assignments "
            f"(primary-recovery + rate-limit-fallback), found {len(all_resets)}"
        )
