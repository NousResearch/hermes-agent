"""RetainDB BASE_URL always-blocked floor (salvage of incomplete #4984)."""

from unittest.mock import MagicMock, patch

import plugins.memory.retaindb as mod


def test_retaindb_initialize_resets_metadata_base_url(monkeypatch, tmp_path):
    monkeypatch.setenv("RETAINDB_API_KEY", "test-key")
    monkeypatch.setenv("RETAINDB_BASE_URL", "http://169.254.169.254/latest/")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    provider = mod.RetainDBMemoryProvider()
    fake_client = MagicMock()
    with patch.object(mod, "_Client", return_value=fake_client) as client_cls, patch.object(
        mod, "_WriteQueue", return_value=MagicMock()
    ):
        provider.initialize(session_id="s1", hermes_home=str(tmp_path))

    assert client_cls.call_args.args[1] == mod._DEFAULT_BASE_URL


def test_retaindb_allows_private_self_host_url():
    """Full is_safe_url would wrongly reject LAN; always-blocked must not."""
    from tools.url_safety import is_always_blocked_url

    assert not is_always_blocked_url("http://192.168.1.50:8080")
    assert not is_always_blocked_url("http://127.0.0.1:8080")
