"""The post-compression "still large" warning must be reachable.

Pre-agent session hygiene compresses when the transcript passes its fixed
0.85 safety threshold, then warns if the result is *still* large. That warning
was pinned at 0.95 of the window, so it could not fire at the point that
triggered hygiene.

On a 500K-window model, hygiene fires at ~425K and the warning needs ~475K,
so it never fires - the operator gets no signal when
compression achieves nothing and the session immediately re-compresses.
"""

import pytest

from gateway.run import hygiene_warn_token_threshold


class TestHygieneWarnTokenThreshold:
    def test_never_above_the_compression_trigger(self):
        """Compression left the session at or above the point that triggered
        it: it will re-fire next turn. That is the treadmill, and it is the
        signal worth logging."""
        window = 500_000
        compress_at = int(window * 0.85)

        warn_at = hygiene_warn_token_threshold(window, compress_at)

        assert warn_at <= compress_at

    @pytest.mark.parametrize(
        "window",
        [500_000, 1_000_000, 272_000, 128_000],
    )
    def test_reachable_at_the_fixed_hygiene_trigger(self, window):
        """A no-op compression at the real 0.85 hygiene trigger must warn."""
        compress_at = int(window * 0.85)

        warn_at = hygiene_warn_token_threshold(window, compress_at)

        # A compression that reclaimed nothing lands right back at the trigger.
        assert compress_at >= warn_at, (
            f"window={window} threshold=0.85: a no-op compression "
            f"({compress_at:,} tokens) would not warn (needs {warn_at:,})"
        )

    def test_effective_compression_does_not_warn(self):
        """The common case stays quiet: compression did its job."""
        warn_at = hygiene_warn_token_threshold(500_000, 425_000)

        assert 300_000 < warn_at, "a healthy post-compression size must not warn"

    def test_degenerate_inputs_do_not_crash(self):
        for window, compress_at in ((0, 0), (-1, 10), (100, 0)):
            assert hygiene_warn_token_threshold(window, compress_at) >= 0
