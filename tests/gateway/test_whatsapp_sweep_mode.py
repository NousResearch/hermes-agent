"""Regression contracts for optional WhatsApp bridge sweep mode."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BRIDGE = REPO / "scripts/whatsapp-bridge/bridge.js"
ADAPTER = REPO / "plugins/platforms/whatsapp/adapter.py"


def test_sweep_mode_is_disabled_by_default_and_configurable():
    source = BRIDGE.read_text()
    assert "WHATSAPP_SWEEP_INTERVAL_MS || '0'" in source
    assert "WHATSAPP_SWEEP_WINDOW_MS || '5000'" in source
    assert "const SWEEP_INTERVAL_MS" in source
    assert "const SWEEP_WINDOW_MS" in source


def test_only_intentional_sweep_close_waits_for_sweep_interval():
    source = BRIDGE.read_text()
    assert "let intentionalSweepDisconnect = false" in source
    assert "const wasIntentionalSweepDisconnect = intentionalSweepDisconnect" in source
    assert "} else if (wasIntentionalSweepDisconnect)" in source
    assert "setTimeout(startSocket, SWEEP_INTERVAL_MS)" in source
    # A remote/network 428 is indistinguishable by status code alone and must
    # keep the established rapid-reconnect behavior.
    assert "setTimeout(startSocket, reason === 515 ? 1000 : 3000)" in source
    assert source.count("intentionalSweepDisconnect = true") == 2


def test_sweep_extends_the_drain_window_for_inbound_batches():
    source = BRIDGE.read_text()
    assert "messages.some(message => message.message && !message.key.fromMe)" in source
    assert "sweepMessagesSeen = true" in source
    assert "clearSweepTimer()" in source


def test_adapter_propagates_optional_sweep_configuration():
    source = ADAPTER.read_text()
    assert 'self.config.extra.get("sweep_interval_ms")' in source
    assert 'bridge_env["WHATSAPP_SWEEP_INTERVAL_MS"]' in source
    assert 'self.config.extra.get("sweep_window_ms")' in source
    assert 'bridge_env["WHATSAPP_SWEEP_WINDOW_MS"]' in source
