"""Text-batch debounce defaults must not exceed Telegram's cadence (#44883, #25056).

WhatsApp (5s/10s) and Weixin (3s/5s) used to hold every reply for seconds
before dispatching; Telegram waits 0.3s (1.0s near a split chunk).
"""

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.weixin import WeixinAdapter
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter




@pytest.mark.parametrize(
    "adapter_cls", [WhatsAppAdapter, WeixinAdapter], ids=["whatsapp", "weixin"],
)
def test_text_batch_delays_clamped_to_shared_ceilings(adapter_cls, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = adapter_cls(PlatformConfig(
        enabled=True, extra={"text_batch_delay_seconds": "2.5", "text_batch_split_delay_seconds": 7}))

    assert adapter._text_batch_delay_seconds == adapter._TEXT_BATCH_MAX_DELAY_S
    assert adapter._text_batch_split_delay_seconds == adapter._TEXT_BATCH_MAX_SPLIT_DELAY_S
