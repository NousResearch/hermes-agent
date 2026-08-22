"""RetainDB BASE_URL always-blocked floor (salvage of incomplete #4984)."""

from unittest.mock import MagicMock, patch

import pytest

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


def test_retaindb_file_read_rejects_public_to_metadata_redirect(monkeypatch):
    response = MagicMock(
        is_redirect=True,
        headers={"location": "http://169.254.169.254/latest/meta-data/"},
        url="https://api.retaindb.com/v1/files/file-1/content",
    )

    def redirecting_get(*args, **kwargs):
        for hook in kwargs["hooks"]["response"]:
            hook(response)
        return response

    monkeypatch.setattr("requests.get", redirecting_get)
    client = mod._Client("test-key", "https://api.retaindb.com", "default")

    with pytest.raises(RuntimeError, match="redirect target is always blocked"):
        client.read_file_content("file-1")
