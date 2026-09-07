"""Unknown destinations must not acknowledge a successful local delivery."""

from pathlib import Path

import pytest

from gateway.config import GatewayConfig
from gateway.delivery import DeliveryRouter, DeliveryTarget


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ["MisspelledPlatform:ChatID:ThreadID", "", "   "])
async def test_unknown_destination_fails_without_writing_and_local_still_works(tmp_path, monkeypatch, raw):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    router = DeliveryRouter(GatewayConfig())
    target = DeliveryTarget.parse(raw)
    raw = raw.strip()
    result = await router.deliver("must not save", [target], job_id="bad")
    assert not any(receipt["success"] for receipt in result.values()), result
    assert raw in result
    assert result[raw]["error"] == f"unknown_platform: {raw}"
    assert not list(tmp_path.rglob("*.txt"))
    assert not list(tmp_path.rglob("*.md"))
    result = await router.deliver("explicit local marker", [DeliveryTarget.parse("local")], job_id="good")
    assert result["local"]["success"]
    assert "explicit local marker" in Path(result["local"]["result"]["path"]).read_text(encoding="utf-8")
