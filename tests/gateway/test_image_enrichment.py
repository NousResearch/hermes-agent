from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_image_enrichment_uses_athabasca_upload_guidance_without_stale_r2_warning():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)

    with patch(
        "tools.vision_tools.vision_analyze_tool",
        return_value='{"success": true, "analysis": "A painted serpent warrior."}',
    ):
        enriched = await runner._enrich_message_with_vision(
            "caption",
            ["/tmp/test.jpg"],
        )

    assert "R2 not configured" not in enriched
    assert "Gateway media URL available for reference" not in enriched
    assert "POST /api/uploads" in enriched
    assert "Do not store the local cache path" in enriched
    assert "caption" in enriched
