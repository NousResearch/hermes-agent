"""Basic tests for the MQTT platform adapter plugin."""

from unittest import mock
import pytest

# Import after ensuring path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

try:
    from plugins.platforms.mqtt.adapter import MQTTAdapter, register
except Exception as e:
    MQTTAdapter = None
    register = None


class TestMQTTAdapterBasics:
    def test_register_does_not_crash(self):
        if register is None:
            pytest.skip("adapter not importable")
        class FakeCtx:
            def register_platform(self, **kwargs):
                assert kwargs.get("name") == "mqtt"
        ctx = FakeCtx()
        register(ctx)

    def test_adapter_inits_with_defaults(self):
        if MQTTAdapter is None:
            pytest.skip("adapter not importable")
        class FakeConfig:
            extra = {}
            token = ""
        adapter = MQTTAdapter(FakeConfig())
        assert adapter._client_id == "hermes-mqtt"
        assert adapter._log_retained is False
        assert adapter._observational is True

    def test_retained_suppressed_by_default(self):
        if MQTTAdapter is None:
            pytest.skip("adapter not importable")
        class FakeConfig:
            extra = {}
            token = ""
        adapter = MQTTAdapter(FakeConfig())
        # Simulate retained message
        class FakeMsg:
            topic = "test/topic"
            payload = b"data"
            retain = True
        with mock.patch.object(adapter, "_log_event") as mock_log:
            # Call private for test; in real it is triggered by paho
            adapter._on_message(None, None, FakeMsg())
            mock_log.assert_not_called()  # suppressed


if __name__ == "__main__":
    pytest.main([__file__])
