"""A ``type: exec`` quick command's raw bytes must decode lossily.

``_hm_run_exec_quick_command`` drains the subprocess with ``asyncio.subprocess.PIPE``
(bytes) and called bare ``.decode()`` — strict UTF-8. A quick command printing one
locale-encoded byte (``ls`` of a Latin-1 filename, ``git`` output, ``uptime`` on a
non-UTF-8 host) turned the captured output into a ``Quick command error`` instead
of preserving the usable parts of the reply. Same class as #105582 / #124990:
capture pipes decode with ``errors="replace"``.
"""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_exec_quick_command_decodes_non_utf8_output_lossily():
    """One bad byte in the command's stdout must not raise out of the handler."""
    from gateway.run_inbound import GatewayInboundMixin

    class Runner(GatewayInboundMixin):
        pass

    result = await Runner()._hm_run_exec_quick_command(
        "limits", "printf 'caf\\xe9 noir'")
    # The reply survives, with the stray byte replaced instead of raising.
    assert "caf" in result and "noir" in result
    assert "\ufffd" in result
