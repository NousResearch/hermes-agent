import asyncio
from types import SimpleNamespace

import pytest

from gateway.run import GatewayRunner


@pytest.mark.asyncio
async def test_auto_vision_enrichment_times_out_and_degrades_gracefully(monkeypatch):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = SimpleNamespace()

    async def _slow_vision_analyze_tool(*, image_url, user_prompt, model=None):
        await asyncio.sleep(0.05)
        return (
            '{"success": true, "analysis": "should not arrive before timeout"}'
        )

    monkeypatch.setattr(
        "tools.vision_tools.vision_analyze_tool",
        _slow_vision_analyze_tool,
    )
    monkeypatch.setattr(
        "gateway.run._AUTO_VISION_ANALYSIS_TIMEOUT_SECONDS",
        0.01,
        raising=False,
    )

    result = await asyncio.wait_for(
        runner._enrich_message_with_vision(
            "看看这个图",
            ["/tmp/demo-image.jpg"],
        ),
        timeout=0.03,
    )

    assert "couldn't quite see it this time" in result
    assert "vision_analyze using image_url: /tmp/demo-image.jpg" in result
    assert result.endswith("看看这个图")
