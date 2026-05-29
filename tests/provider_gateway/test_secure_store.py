import os
import tempfile
import builtins
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from provider_gateway.secure_store import DynamicCredentialStore


@pytest.fixture(autouse=True)
def disable_native_keyring(monkeypatch):
    """Isolate tests from native OS keyring and WSL to prevent pollution and test interference."""
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "keyring":
            raise ImportError("Mocked native keyring unavailable")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        with patch("provider_gateway.secure_store.DynamicCredentialStore.is_wsl", return_value=False):
            yield


@pytest.fixture
def temp_secrets_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_secure_store_local_aes_encryption(temp_secrets_dir) -> None:
    store = DynamicCredentialStore(secrets_dir=temp_secrets_dir)

    # Initially empty
    assert store.get_credential("openrouter") is None

    # Store key
    api_key = "sk-or-v1-unique-key-12345"
    assert store.store_credential("openrouter", api_key) is True

    # Retrieve key
    retrieved = store.get_credential("openrouter")
    assert retrieved == api_key


def test_secure_store_delete_credential(temp_secrets_dir) -> None:
    store = DynamicCredentialStore(secrets_dir=temp_secrets_dir)

    # Store and then delete
    api_key = "key-to-delete"
    assert store.store_credential("anthropic", api_key) is True
    assert store.get_credential("anthropic") == api_key

    # Delete
    assert store.delete_credential("anthropic") is True
    assert store.get_credential("anthropic") is None


def test_secure_store_machine_binding_protection(temp_secrets_dir) -> None:
    """Verify that credentials cannot be decrypted if the hardware fingerprint changes (machine-bound)."""
    store = DynamicCredentialStore(secrets_dir=temp_secrets_dir)
    api_key = "extremely-sensitive-token"

    # Store with current fingerprint
    assert store.store_credential("cohere", api_key) is True

    # Mock the fingerprint method to simulate another machine trying to decrypt the file
    with patch.object(store, "_get_machine_fingerprint", return_value=b"totally_different_machine_fingerprint_123"):
        # Decryption should fail and return None
        assert store.get_credential("cohere") is None


@patch("subprocess.run")
def test_secure_store_wsl_powershell_mock(mock_run, temp_secrets_dir) -> None:
    """Test WSL powershell interop routing flow."""
    store = DynamicCredentialStore(secrets_dir=temp_secrets_dir)

    # Force is_wsl and has_powershell_interop to be True
    with patch.object(store, "is_wsl", return_value=True), \
         patch.object(store, "has_powershell_interop", return_value=True):
        
        # Mock successful subprocess execution for storing
        mock_run.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")
        
        # Store (will fallback to WSL Powershell since keyring import fails)
        res = store.store_credential("mock_wsl_prov", "wsl-secret-key")
        assert res is True
        assert mock_run.call_count >= 1


class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
