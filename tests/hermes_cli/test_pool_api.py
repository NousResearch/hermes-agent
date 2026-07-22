"""
Tests for the credential pool REST API endpoints.

These extend the existing /api/credentials/pool family (list/add/remove)
and the new strategy/reset/health endpoints.

API shape:
  GET    /api/credentials/pool                          → {providers: [{provider, entries: [...]}]}
  POST   /api/credentials/pool                          → {ok, provider, count}
  DELETE /api/credentials/pool/{provider}/{index}       → {ok, provider, count}
  PUT    /api/credentials/pool/{provider}/strategy      → {ok, strategy}
  POST   /api/credentials/pool/{provider}/{index}/reset → {ok, index, status}
  GET    /api/credentials/pool/{provider}/health        → {provider, total, available, strategy}
"""

import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def test_client(monkeypatch):
    """Create a test client with auth via session token header."""
    monkeypatch.setenv("HERMES_HOME", tempfile.mkdtemp())
    import hermes_cli.web_server as ws
    client = TestClient(ws.app, raise_server_exceptions=False)
    client.headers[ws._SESSION_HEADER_NAME] = ws._SESSION_TOKEN
    return client


@pytest.fixture
def fake_pool():
    """Mock credential pool with 2 entries."""
    pool = MagicMock()
    entry1 = MagicMock(id="abc123", label="key-1", auth_type="api_key", source="manual",
                       base_url="https://api.z.ai/api/coding/paas/v4",
                       last_status="ok", last_status_at=None,
                       last_error_code=None, last_error_reason=None,
                       last_error_reset_at=None, last_error_message=None,
                       request_count=42, priority=0,
                       access_token="sk-secret-123", refresh_token=None)
    entry2 = MagicMock(id="def456", label="key-2", auth_type="api_key", source="manual",
                       base_url="",
                       last_status="exhausted", last_status_at=1234567890,
                       last_error_code=1308, last_error_reason="Usage limit reached",
                       last_error_reset_at=9999999999, last_error_message="rate limited",
                       request_count=15, priority=0,
                       access_token="sk-secret-456", refresh_token=None)
    pool.entries.return_value = [entry1, entry2]
    pool.peek.return_value = entry1
    pool._available_entries.return_value = [entry1]
    pool.add_entry.return_value = True
    pool.remove_index.return_value = entry1
    pool._replace_entry.return_value = None
    pool._persist.return_value = None
    return pool


def _get_entries(resp, provider="zai"):
    """Extract entries for a provider from the GET response."""
    data = resp.json()
    providers = data.get("providers", [])
    p = next((x for x in providers if x["provider"] == provider), None)
    return p["entries"] if p else []


# ════════════════════════════════════════════════════════════════════════════
# GET /api/credentials/pool
# ════════════════════════════════════════════════════════════════════════════

class TestGetPool:
    def test_get_empty_pool(self, test_client):
        """GET returns empty list when no pool exists."""
        with patch("agent.credential_pool.load_pool") as mock_load, \
             patch("hermes_cli.auth.read_credential_pool", return_value={}):
            pool = MagicMock()
            pool.entries.return_value = []
            mock_load.return_value = pool
            resp = test_client.get("/api/credentials/pool")
        assert resp.status_code == 200
        assert resp.json()["providers"] == []

    def test_get_pool_after_add(self, test_client, fake_pool):
        """GET returns entries after they're added."""
        with patch("agent.credential_pool.load_pool", return_value=fake_pool), \
             patch("hermes_cli.auth.read_credential_pool", return_value={"zai": [{}]}):
            resp = test_client.get("/api/credentials/pool")
        assert resp.status_code == 200
        entries = _get_entries(resp, "zai")
        assert len(entries) == 2

    def test_get_pool_includes_base_url(self, test_client, fake_pool):
        """GET response includes base_url for each entry."""
        with patch("agent.credential_pool.load_pool", return_value=fake_pool), \
             patch("hermes_cli.auth.read_credential_pool", return_value={"zai": [{}]}):
            resp = test_client.get("/api/credentials/pool")
        entries = _get_entries(resp, "zai")
        assert entries[0]["base_url"] == "https://api.z.ai/api/coding/paas/v4"
        assert entries[1]["base_url"] == ""


# ════════════════════════════════════════════════════════════════════════════
# POST /api/credentials/pool
# ════════════════════════════════════════════════════════════════════════════

class TestAddPoolEntry:
    def test_add_entry_basic(self, test_client, fake_pool):
        """POST adds a new entry."""
        with patch("agent.credential_pool.load_pool", return_value=fake_pool):
            resp = test_client.post("/api/credentials/pool", json={
                "provider": "zai",
                "api_key": "sk-test-12345678",
            })
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        assert resp.json()["provider"] == "zai"

    def test_add_entry_with_label(self, test_client, fake_pool):
        """POST with a custom label."""
        with patch("agent.credential_pool.load_pool", return_value=fake_pool):
            resp = test_client.post("/api/credentials/pool", json={
                "provider": "zai",
                "api_key": "sk-test-12345678",
                "label": "My Custom Key",
            })
        assert resp.status_code == 200

    def test_add_entry_with_base_url(self, test_client, fake_pool):
        """POST with a custom base_url."""
        with patch("agent.credential_pool.load_pool", return_value=fake_pool):
            resp = test_client.post("/api/credentials/pool", json={
                "provider": "zai",
                "api_key": "sk-test-12345678",
                "base_url": "https://api.z.ai/api/coding/paas/v4",
            })
        assert resp.status_code == 200

    def test_add_multiple_entries(self, test_client, fake_pool):
        """POST multiple entries in sequence."""
        with patch("agent.credential_pool.load_pool", return_value=fake_pool):
            for i in range(3):
                resp = test_client.post("/api/credentials/pool", json={
                    "provider": "zai",
                    "api_key": f"sk-test-key-{i:08d}",
                })
                assert resp.status_code == 200


# ════════════════════════════════════════════════════════════════════════════
# DELETE /api/credentials/pool/{provider}/{index}
# ════════════════════════════════════════════════════════════════════════════

class TestRemovePoolEntry:
    def test_remove_entry(self, test_client, fake_pool):
        """DELETE removes entry by 1-based index."""
        with patch("agent.credential_pool.load_pool", return_value=fake_pool):
            resp = test_client.delete("/api/credentials/pool/zai/1")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_remove_nonexistent_entry(self, test_client, fake_pool):
        """DELETE with invalid index returns 404."""
        fake_pool.remove_index.return_value = None
        with patch("agent.credential_pool.load_pool", return_value=fake_pool):
            resp = test_client.delete("/api/credentials/pool/zai/999")
        assert resp.status_code in (400, 404)


# ════════════════════════════════════════════════════════════════════════════
# PUT /api/credentials/pool/{provider}/strategy
# ════════════════════════════════════════════════════════════════════════════

class TestSetPoolStrategy:
    def test_strategy_persists_after_set(self, test_client, fake_pool):
        """PUT strategy changes the rotation strategy."""
        with patch("agent.credential_pool.SUPPORTED_POOL_STRATEGIES",
                   {"fill_first", "round_robin", "least_used", "random"}), \
             patch("hermes_cli.config.set_config_value") as mock_set:
            resp = test_client.put("/api/credentials/pool/zai/strategy", json={
                "strategy": "round_robin",
            })
        assert resp.status_code == 200
        assert resp.json()["strategy"] == "round_robin"
        mock_set.assert_called_once_with("credential_pool_strategies.zai", "round_robin")


# ════════════════════════════════════════════════════════════════════════════
# POST /api/credentials/pool/{provider}/{index}/reset
# ════════════════════════════════════════════════════════════════════════════

class TestResetPoolEntry:
    def test_reset_entry(self, test_client, fake_pool):
        """POST reset clears the exhausted status."""
        # Use real PooledCredential for replace() to work
        from agent.credential_pool import PooledCredential, STATUS_EXHAUSTED
        real_entry = PooledCredential(
            provider="zai", id="abc123", label="key-1",
            auth_type="api_key", priority=0, source="manual",
            access_token="sk-secret",
            last_status=STATUS_EXHAUSTED,
            last_error_code=1308,
        )
        pool = MagicMock()
        pool.entries.return_value = [real_entry]
        pool._replace_entry.return_value = None
        pool._persist.return_value = None
        with patch("agent.credential_pool.load_pool", return_value=pool):
            resp = test_client.post("/api/credentials/pool/zai/1/reset")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True


# ════════════════════════════════════════════════════════════════════════════
# GET /api/credentials/pool/{provider}/health
# ════════════════════════════════════════════════════════════════════════════

class TestPoolHealth:
    def test_health_with_entries(self, test_client, fake_pool):
        """GET health returns summary."""
        with patch("agent.credential_pool.load_pool", return_value=fake_pool), \
             patch("agent.credential_pool.get_pool_strategy", return_value="fill_first"), \
             patch("agent.credential_pool.STATUS_OK", "ok"):
            resp = test_client.get("/api/credentials/pool/zai/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert data["available"] >= 0


# ════════════════════════════════════════════════════════════════════════════
# End-to-end
# ════════════════════════════════════════════════════════════════════════════

class TestEndToEndFlow:
    def test_full_lifecycle(self, test_client, fake_pool):
        """Add → list → strategy → reset → delete → verify."""
        # Add
        resp = test_client.post("/api/credentials/pool", json={
            "provider": "zai",
            "api_key": "sk-test-lifecycle",
        })
        assert resp.status_code == 200

        # List
        with patch("agent.credential_pool.load_pool", return_value=fake_pool), \
             patch("hermes_cli.auth.read_credential_pool", return_value={"zai": [{}]}):
            resp = test_client.get("/api/credentials/pool")
        assert resp.status_code == 200
        entries = _get_entries(resp, "zai")
        assert len(entries) >= 1

        # Strategy
        with patch("agent.credential_pool.SUPPORTED_POOL_STRATEGIES",
                   {"fill_first", "round_robin"}), \
             patch("hermes_cli.config.set_config_value"):
            resp = test_client.put("/api/credentials/pool/zai/strategy", json={
                "strategy": "round_robin",
            })
        assert resp.status_code == 200

        # Reset - use real PooledCredential
        from agent.credential_pool import PooledCredential
        real_entry = PooledCredential(
            provider="zai", id="abc123", label="key-1",
            auth_type="api_key", priority=0, source="manual",
            access_token="sk-secret",
        )
        reset_pool = MagicMock()
        reset_pool.entries.return_value = [real_entry]
        reset_pool._replace_entry.return_value = None
        reset_pool._persist.return_value = None
        with patch("agent.credential_pool.load_pool", return_value=reset_pool):
            resp = test_client.post("/api/credentials/pool/zai/1/reset")
        assert resp.status_code == 200

        # Delete
        with patch("agent.credential_pool.load_pool", return_value=fake_pool):
            resp = test_client.delete("/api/credentials/pool/zai/1")
        assert resp.status_code == 200



# ════════════════════════════════════════════════════════════════════════════
# POST /api/credentials/pool/oauth-login
# ════════════════════════════════════════════════════════════════════════════

class TestOAuthLogin:
    """OAuth device-flow login for OAuth-capable providers."""

    def test_oauth_login_anthropic_success(self, test_client, monkeypatch):
        """OAuth login for anthropic succeeds and returns 200."""
        monkeypatch.setattr(
            "hermes_cli.web_server._run_oauth_login_blocking",
            lambda provider, body: True,
        )
        monkeypatch.setattr(
            "agent.credential_pool.load_pool",
            lambda provider: MagicMock(entries=lambda: []),
        )
        resp = test_client.post(
            "/api/credentials/pool/oauth-login",
            json={"provider": "anthropic", "no_browser": True, "timeout": 30.0},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["provider"] == "anthropic"
        assert data["status"] == "logged_in"

    def test_oauth_login_non_oauth_provider_rejected(self, test_client):
        """Non-OAuth providers (e.g. zai) return 400."""
        resp = test_client.post(
            "/api/credentials/pool/oauth-login",
            json={"provider": "zai", "timeout": 30.0},
        )
        assert resp.status_code == 400
        assert "OAuth" in resp.json()["detail"]

    def test_oauth_login_invalid_provider_name(self, test_client):
        """Path-traversal in provider name is rejected by _validate_provider_name."""
        resp = test_client.post(
            "/api/credentials/pool/oauth-login",
            json={"provider": "../etc/passwd", "timeout": 30.0},
        )
        assert resp.status_code == 400

    def test_oauth_login_failure_returns_401(self, test_client, monkeypatch):
        """When the OAuth flow returns False, the endpoint returns 401."""
        monkeypatch.setattr(
            "hermes_cli.web_server._run_oauth_login_blocking",
            lambda provider, body: False,
        )
        resp = test_client.post(
            "/api/credentials/pool/oauth-login",
            json={"provider": "anthropic", "timeout": 30.0},
        )
        assert resp.status_code == 401


# ════════════════════════════════════════════════════════════════════════════
# GET /api/credentials/pool/{provider}/status
# ════════════════════════════════════════════════════════════════════════════

class TestPoolStatus:
    """GET provider auth status, mirroring `hermes auth status`."""

    def test_status_logged_in_returns_full_dict(self, test_client, monkeypatch):
        """Logged-in provider returns full status dict (without secrets)."""
        fake_status = {
            "logged_in": True,
            "auth_type": "oauth",
            "client_id": "client-abc",
            "scope": "read write",
            "expires_at": "2027-01-01T00:00:00Z",
        }
        monkeypatch.setattr(
            "hermes_cli.auth.get_auth_status",
            lambda provider: fake_status,
        )
        resp = test_client.get("/api/credentials/pool/anthropic/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["logged_in"] is True
        assert data["provider"] == "anthropic"
        assert "access_token" not in data

    def test_status_logged_out_returns_simple(self, test_client, monkeypatch):
        """Logged-out provider returns minimal status."""
        monkeypatch.setattr(
            "hermes_cli.auth.get_auth_status",
            lambda provider: {"logged_in": False},
        )
        resp = test_client.get("/api/credentials/pool/zai/status")
        assert resp.status_code == 200
        assert resp.json()["logged_in"] is False

    def test_status_redacts_secrets(self, test_client, monkeypatch):
        """access_token/refresh_token/api_key are redacted in response."""
        monkeypatch.setattr(
            "hermes_cli.auth.get_auth_status",
            lambda provider: {
                "logged_in": True,
                "access_token": "sk-secret-12345-abcde",
                "refresh_token": "rt-secret",
            },
        )
        resp = test_client.get("/api/credentials/pool/anthropic/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["access_token"] != "sk-secret-12345-abcde"
        assert "sk-" not in data["access_token"] or "..." in data["access_token"]

    def test_status_invalid_provider_name(self, test_client):
        """Provider with invalid chars is rejected by _validate_provider_name."""
        resp = test_client.get("/api/credentials/pool/..invalid../status")
        assert resp.status_code == 400


# ════════════════════════════════════════════════════════════════════════════
# POST /api/credentials/pool/{provider}/logout
# ════════════════════════════════════════════════════════════════════════════

class TestPoolLogout:
    """POST logout clears the provider's stored auth state."""

    def test_logout_success(self, test_client, monkeypatch):
        """Successful logout returns {ok: True, provider: ...}."""
        monkeypatch.setattr(
            "hermes_cli.auth.logout_command",
            lambda args: None,
        )
        resp = test_client.post("/api/credentials/pool/anthropic/logout")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "provider": "anthropic"}

    def test_logout_unknown_provider_404(self, test_client, monkeypatch):
        """SystemExit(1) from logout_command is converted to 404."""
        def fake_logout(args):
            raise SystemExit(1)
        monkeypatch.setattr("hermes_cli.auth.logout_command", fake_logout)
        resp = test_client.post("/api/credentials/pool/zai/logout")
        assert resp.status_code == 404

    def test_logout_invalid_provider_name(self, test_client):
        """Provider with invalid chars rejected by _validate_provider_name."""
        resp = test_client.post("/api/credentials/pool/..invalid../logout")
        assert resp.status_code == 400


# ════════════════════════════════════════════════════════════════════════════
# GET /api/credentials/pool/summary
# ════════════════════════════════════════════════════════════════════════════

class TestPoolSummary:
    """Dashboard top-level view: one summary per provider in the registry."""

    def test_summary_returns_all_providers(self, test_client, monkeypatch):
        """Summary includes all providers from PROVIDER_REGISTRY."""
        from hermes_cli.auth import PROVIDER_REGISTRY
        fake_registry = {
            "zai": MagicMock(name="Z.AI", auth_type="api_key"),
            "deepseek": MagicMock(name="DeepSeek", auth_type="api_key"),
        }
        monkeypatch.setattr(
            "hermes_cli.auth.PROVIDER_REGISTRY", fake_registry
        )
        monkeypatch.setattr(
            "agent.credential_pool.load_pool",
            lambda provider: MagicMock(entries=lambda: []),
        )
        monkeypatch.setattr(
            "agent.credential_pool.get_pool_strategy",
            lambda provider: "fill_first",
        )
        resp = test_client.get("/api/credentials/pool/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_providers"] == 2
        provider_ids = {p["id"] for p in data["providers"]}
        assert provider_ids == {"zai", "deepseek"}

    def test_summary_total_credentials(self, test_client, monkeypatch):
        """total_credentials equals sum of per-provider credentials_count."""
        from agent.credential_pool import STATUS_OK
        fake_registry = {"zai": MagicMock(name="Z.AI", auth_type="api_key")}

        def fake_load(provider):
            pool = MagicMock()
            entry = MagicMock(last_status=STATUS_OK, last_error_reason=None)
            pool.entries.return_value = [entry, entry, entry]
            return pool

        monkeypatch.setattr("hermes_cli.auth.PROVIDER_REGISTRY", fake_registry)
        monkeypatch.setattr("agent.credential_pool.load_pool", fake_load)
        monkeypatch.setattr(
            "agent.credential_pool.get_pool_strategy", lambda p: "fill_first"
        )
        resp = test_client.get("/api/credentials/pool/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_credentials"] == 3
        assert data["providers"][0]["credentials_count"] == 3

    def test_summary_handles_provider_failure(self, test_client, monkeypatch):
        """If one provider raises, summary still includes it with last_error."""
        from agent.credential_pool import STATUS_OK
        fake_registry = {
            "zai": MagicMock(name="Z.AI", auth_type="api_key"),
            "broken": MagicMock(name="Broken", auth_type="api_key"),
        }

        def fake_load(provider):
            if provider == "broken":
                raise IOError("disk full")
            pool = MagicMock()
            pool.entries.return_value = []
            return pool

        monkeypatch.setattr("hermes_cli.auth.PROVIDER_REGISTRY", fake_registry)
        monkeypatch.setattr("agent.credential_pool.load_pool", fake_load)
        monkeypatch.setattr(
            "agent.credential_pool.get_pool_strategy", lambda p: "fill_first"
        )
        resp = test_client.get("/api/credentials/pool/summary")
        assert resp.status_code == 200
        data = resp.json()
        broken = next(p for p in data["providers"] if p["id"] == "broken")
        assert "pool_load_failed" in (broken.get("last_error") or "")


# ════════════════════════════════════════════════════════════════════════════
# POST /api/credentials/pool/{provider}/probe
# ═ ════════════════════════════════════════════════════════════════════════════

class TestPoolProbe:
    """Live probe of pool entries — used by the dashboard 'Test all' button."""

    def test_probe_returns_results(self, test_client, monkeypatch):
        """Probe without filter probes all entries in the pool."""
        fake_pool = MagicMock()
        entry = MagicMock(
            id="e1",
            label="key-1",
            base_url="https://example.com",
        )
        fake_pool.entries.return_value = [entry]

        def fake_probe(entry, timeout=8.0):
            return {
                "label": entry.label,
                "base_url": entry.base_url,
                "http_code": 200,
                "latency_ms": 12.3,
                "ok": True,
            }

        monkeypatch.setattr("agent.credential_pool.load_pool", lambda p: fake_pool)
        monkeypatch.setattr(
            "hermes_cli.web_server._probe_one_entry", fake_probe
        )
        resp = test_client.post("/api/credentials/pool/zai/probe", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["http_code"] == 200

    def test_probe_specific_entry_by_id(self, test_client, monkeypatch):
        """With entry_id set, only the matching entry is probed."""
        fake_pool = MagicMock()
        e1 = MagicMock(id="e1", label="key-1", base_url="https://a.com")
        e2 = MagicMock(id="e2", label="key-2", base_url="https://b.com")
        fake_pool.entries.return_value = [e1, e2]

        probed_labels = []

        def fake_probe(entry, timeout=8.0):
            probed_labels.append(entry.label)
            return {
                "label": entry.label,
                "base_url": entry.base_url,
                "http_code": 200,
                "ok": True,
            }

        monkeypatch.setattr("agent.credential_pool.load_pool", lambda p: fake_pool)
        monkeypatch.setattr("hermes_cli.web_server._probe_one_entry", fake_probe)
        resp = test_client.post(
            "/api/credentials/pool/zai/probe", json={"entry_id": "e2"}
        )
        assert resp.status_code == 200
        assert probed_labels == ["key-2"]

    def test_probe_invalid_entry_id_returns_404(self, test_client, monkeypatch):
        """Unknown entry_id (with non-empty pool) returns 404."""
        fake_pool = MagicMock()
        # Pool has an entry, but we're asking for one that doesn't exist
        e1 = MagicMock(id="real-id", label="key-1", base_url="https://a.com")
        fake_pool.entries.return_value = [e1]
        monkeypatch.setattr("agent.credential_pool.load_pool", lambda p: fake_pool)
        monkeypatch.setattr(
            "hermes_cli.web_server._probe_one_entry", lambda e, t=8.0: {"ok": True}
        )
        resp = test_client.post(
            "/api/credentials/pool/zai/probe", json={"entry_id": "ghost"}
        )
        assert resp.status_code == 404

    def test_probe_empty_pool(self, test_client, monkeypatch):
        """Empty pool returns 200 with empty results."""
        fake_pool = MagicMock()
        fake_pool.entries.return_value = []
        monkeypatch.setattr("agent.credential_pool.load_pool", lambda p: fake_pool)
        monkeypatch.setattr(
            "hermes_cli.web_server._probe_one_entry", lambda e, t=8.0: {"ok": True}
        )
        resp = test_client.post("/api/credentials/pool/zai/probe", json={})
        assert resp.status_code == 200
        assert resp.json()["results"] == []
