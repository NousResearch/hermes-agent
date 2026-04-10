"""Tests for the write-only secrets tool (tools/secrets_tool.py)."""

from __future__ import annotations

import pytest

from proxy.store import CredentialStore
from tools.secrets_tool import handle_secrets, set_store


@pytest.fixture(autouse=True)
def _setup_store():
    """Provide a fresh store for each test."""
    store = CredentialStore()
    set_store(store)
    yield store
    set_store(None)


@pytest.mark.asyncio
class TestSecretsTool:
    """Secrets tool must be write-only with no read/get action."""

    async def test_store(self, _setup_store):
        result = await handle_secrets({"action": "store", "name": "tok", "value": "secret"})
        assert result == {"stored": True}
        assert _setup_store._resolve("tok") == "secret"

    async def test_store_missing_name(self):
        result = await handle_secrets({"action": "store", "value": "v"})
        assert "error" in result

    async def test_store_missing_value(self):
        result = await handle_secrets({"action": "store", "name": "n"})
        assert "error" in result

    async def test_rotate(self, _setup_store):
        _setup_store.store("key", "old")
        result = await handle_secrets({"action": "rotate", "name": "key", "value": "new"})
        assert result == {"rotated": True}
        assert _setup_store._resolve("key") == "new"

    async def test_delete(self, _setup_store):
        _setup_store.store("key", "val")
        result = await handle_secrets({"action": "delete", "name": "key"})
        assert result == {"deleted": True}
        assert _setup_store._resolve("key") is None

    async def test_delete_nonexistent(self):
        result = await handle_secrets({"action": "delete", "name": "nope"})
        assert result == {"deleted": False}

    async def test_list(self, _setup_store):
        _setup_store.store("beta", "b")
        _setup_store.store("alpha", "a")
        result = await handle_secrets({"action": "list"})
        assert result == {"names": ["alpha", "beta"]}

    async def test_list_empty(self):
        result = await handle_secrets({"action": "list"})
        assert result == {"names": []}

    async def test_unknown_action(self):
        result = await handle_secrets({"action": "read"})
        assert "error" in result
        assert "read" in result["error"]

    async def test_no_read_action(self):
        """The security invariant: there is no way to read credential values."""
        result = await handle_secrets({"action": "get"})
        assert "error" in result

    async def test_no_store_returns_error(self):
        set_store(None)
        result = await handle_secrets({"action": "list"})
        assert "error" in result
        assert "not running" in result["error"]
