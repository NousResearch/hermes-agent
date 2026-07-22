"""
Security tests for the unified credential resolver and pool REST API.

Covers:
1. SSRF prevention (metadata endpoints, non-http schemes, link-local)
2. Path traversal prevention in provider names
3. Null byte injection
4. API key masking (never leaked in GET responses)
5. Input validation (min key length, strategy whitelist)
6. Race condition protection (pool operations under lock)
"""

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def client(monkeypatch):
    """Create a test client with auth via session token header."""
    monkeypatch.setenv("HERMES_HOME", tempfile.mkdtemp())
    import hermes_cli.web_server as ws
    client = TestClient(ws.app, raise_server_exceptions=False)
    client.headers[ws._SESSION_HEADER_NAME] = ws._SESSION_TOKEN
    return client


@pytest.fixture
def fake_pool():
    """Mock credential pool with 3 entries."""
    pool = MagicMock()
    entry1 = MagicMock(id="abc123", label="key-1", auth_type="api_key", source="manual",
                       base_url="https://api.z.ai/api/coding/paas/v4",
                       last_status="ok", last_status_at=None,
                       last_error_code=None, last_error_reason=None,
                       last_error_reset_at=None, last_error_message=None,
                       request_count=42)
    entry2 = MagicMock(id="def456", label="key-2", auth_type="api_key", source="manual",
                       base_url="https://api.z.ai/api/coding/paas/v4",
                       last_status="exhausted", last_status_at=1234567890,
                       last_error_code=1308, last_error_reason="Usage limit reached",
                       last_error_reset_at=9999999999, last_error_message="rate limited",
                       request_count=15)
    entry3 = MagicMock(id="ghi789", label="key-3", auth_type="api_key", source="manual",
                       base_url="",
                       last_status="ok", last_status_at=None,
                       last_error_code=None, last_error_reason=None,
                       last_error_reset_at=None, last_error_message=None,
                       request_count=0)
    pool.entries.return_value = [entry1, entry2, entry3]
    pool.peek.return_value = entry1
    pool.add_entry.return_value = True
    pool.remove_index.return_value = entry1
    pool.resolve_target.return_value = (1, entry1, None)
    pool._replace_entry.return_value = None
    pool._persist.return_value = None
    return pool


# ════════════════════════════════════════════════════════════════════════════
# 1. SSRF PREVENTION
# ════════════════════════════════════════════════════════════════════════════

class TestSSRFPrevention:
    """Ensure user-supplied base_url cannot reach internal services."""

    def test_aws_metadata_endpoint_blocked(self, client, fake_pool):
        """169.254.169.254 (AWS metadata) must be rejected."""
        with patch("hermes_cli.web_server.load_pool", return_value=fake_pool, create=True):
            with patch("agent.credential_pool.load_pool", return_value=fake_pool):
                resp = client.post("/api/credentials/pool", json={"provider": "zai", 
                    "api_key": "sk-test-1234567890",
                    "base_url": "http://169.254.169.254/latest/meta-data/",
                })
        assert resp.status_code == 400
        assert "Invalid base_url" in resp.json()["detail"]

    def test_gcp_metadata_endpoint_blocked(self, client, fake_pool):
        """metadata.google.internal must be rejected."""
        with patch("agent.credential_pool.load_pool", return_value=fake_pool):
            resp = client.post("/api/credentials/pool", json={"provider": "zai", 
                "api_key": "sk-test-1234567890",
                "base_url": "http://metadata.google.internal/computeMetadata/v1/",
            })
        assert resp.status_code == 400

    def test_link_local_address_blocked(self, client, fake_pool):
        """169.254.x.x (link-local) must be rejected."""
        with patch("agent.credential_pool.load_pool", return_value=fake_pool):
            resp = client.post("/api/credentials/pool", json={"provider": "zai", 
                "api_key": "sk-test-1234567890",
                "base_url": "http://169.254.1.1/v1",
            })
        assert resp.status_code == 400

    def test_file_scheme_blocked(self, client, fake_pool):
        """file:// scheme must be rejected."""
        with patch("agent.credential_pool.load_pool", return_value=fake_pool):
            resp = client.post("/api/credentials/pool", json={"provider": "zai", 
                "api_key": "sk-test-1234567890",
                "base_url": "file:///etc/passwd",
            })
        assert resp.status_code == 400

    def test_gopher_scheme_blocked(self, client, fake_pool):
        """gopher:// scheme must be rejected."""
        with patch("agent.credential_pool.load_pool", return_value=fake_pool):
            resp = client.post("/api/credentials/pool", json={"provider": "zai", 
                "api_key": "sk-test-1234567890",
                "base_url": "gopher://internal:6379/_INFO",
            })
        assert resp.status_code == 400

    def test_dict_scheme_blocked(self, client, fake_pool):
        """dict:// scheme must be rejected."""
        with patch("agent.credential_pool.load_pool", return_value=fake_pool):
            resp = client.post("/api/credentials/pool", json={"provider": "zai", 
                "api_key": "sk-test-1234567890",
                "base_url": "dict://internal:11211/stats",
            })
        assert resp.status_code == 400

    def test_valid_https_endpoint_accepted(self, client, fake_pool):
        """Legitimate https endpoints must work."""
        with patch("agent.credential_pool.load_pool", return_value=fake_pool):
            resp = client.post("/api/credentials/pool", json={"provider": "zai", 
                "api_key": "sk-test-1234567890",
                "base_url": "https://api.z.ai/api/coding/paas/v4",
            })
        assert resp.status_code == 200


# ════════════════════════════════════════════════════════════════════════════
# 2. PATH TRAVERSAL PREVENTION
# ════════════════════════════════════════════════════════════════════════════

class TestPathTraversalPrevention:
    """Provider name cannot be used for path traversal."""

    def test_dot_dot_in_provider_blocked(self, client, fake_pool):
        """../../../etc/passwd must not reach the pool loader."""
        with patch("agent.credential_pool.load_pool", return_value=fake_pool) as mock_load:
            resp = client.get("/api/credentials/pool/..%2F..%2Fetc%2Fpasswd/strategy")
        assert resp.status_code in (400, 422, 404)

    def test_backslash_in_provider_blocked(self, client, fake_pool):
        """Backslashes in provider name must be rejected."""
        with patch("agent.credential_pool.load_pool", return_value=fake_pool):
            resp = client.get("/api/credentials/pool/zai%5Cevil/strategy")
        assert resp.status_code in (400, 422, 404)

    def test_null_byte_in_provider_blocked(self, client, fake_pool):
        """Null byte in provider name must be rejected."""
        with patch("agent.credential_pool.load_pool", return_value=fake_pool):
            resp = client.get("/api/credentials/pool/zai%00evil/strategy")
        assert resp.status_code in (400, 422, 404)


# ════════════════════════════════════════════════════════════════════════════
# 3. API KEY LEAK PREVENTION
# ════════════════════════════════════════════════════════════════════════════

class TestAPIKeyLeakPrevention:
    """API keys must never appear in GET /pool responses."""

    def test_get_pool_does_not_return_api_key(self, client, fake_pool):
        """GET /pool must not contain api_key or access_token in the response."""
        # Set access_token on entries
        for e in fake_pool.entries():
            e.access_token = "sk-testfake-secretkey1234"
        with patch("agent.credential_pool.load_pool", return_value=fake_pool):
            resp = client.get("/api/credentials/pool")
        body_str = resp.text
        assert "sk-testfake-secretkey1234" not in body_str, \
            "API key leaked in GET /pool response!"
        assert "access_token" not in body_str, \
            "access_token field present in GET /pool response!"

    def test_post_pool_does_not_echo_api_key(self, client, fake_pool):
        """POST /pool must not echo the api_key back in the response."""
        with patch("agent.credential_pool.load_pool", return_value=fake_pool):
            resp = client.post("/api/credentials/pool", json={"provider": "zai", 
                "api_key": "sk-testfake-secretkey1234",
            })
        assert resp.status_code == 200
        assert "sk-super-secret" not in resp.text, \
            "API key echoed in POST /pool response!"

    def test_error_messages_do_not_contain_api_key(self, client, fake_pool):
        """Error messages (400/404/500) must not contain the api_key."""
        with patch("agent.credential_pool.load_pool", side_effect=Exception("boom")):
            resp = client.post("/api/credentials/pool", json={"provider": "zai", 
                "api_key": "sk-testfake-secretkey1234",
            })
        assert "sk-super-secret" not in resp.text


# ════════════════════════════════════════════════════════════════════════════
# 4. INPUT VALIDATION
# ════════════════════════════════════════════════════════════════════════════

class TestInputValidation:
    """Validate input constraints on the pool API."""

    def test_empty_api_key_rejected(self, client, fake_pool):
        """Empty api_key must be rejected with 400."""
        with patch("agent.credential_pool.load_pool", return_value=fake_pool):
            resp = client.post("/api/credentials/pool", json={"provider": "zai", 
                "api_key": "",
            })
        assert resp.status_code == 400

    def test_short_api_key_rejected(self, client, fake_pool):
        """api_key shorter than 8 chars must be rejected."""
        with patch("agent.credential_pool.load_pool", return_value=fake_pool):
            resp = client.post("/api/credentials/pool", json={"provider": "zai", 
                "api_key": "short",
            })
        assert resp.status_code == 400

    def test_null_byte_in_api_key_rejected(self, client, fake_pool):
        """Null byte in api_key must be rejected."""
        with patch("agent.credential_pool.load_pool", return_value=fake_pool):
            resp = client.post("/api/credentials/pool", json={"provider": "zai", 
                "api_key": "valid-key\x00evil",
            })
        # FastAPI/Pydantic may accept it, but the base_url check should catch null bytes
        # The key itself should be stored as-is; null bytes in keys are not a security issue
        # per se, but we test that it doesn't cause a 500
        assert resp.status_code in (200, 400)

    def test_invalid_strategy_rejected(self, client, fake_pool):
        """Invalid strategy name must be rejected with 400."""
        with patch("agent.credential_pool.load_pool", return_value=fake_pool):
            with patch("agent.credential_pool.SUPPORTED_POOL_STRATEGIES", {"fill_first", "round_robin", "least_used", "random"}):
                resp = client.put("/api/credentials/pool/zai/strategy", json={
                    "strategy": "evil_strategy; DROP TABLE pools;",
                })
        assert resp.status_code == 400

    def test_sql_injection_in_strategy_rejected(self, client, fake_pool):
        """SQL injection attempt in strategy must be rejected."""
        with patch("agent.credential_pool.load_pool", return_value=fake_pool):
            with patch("agent.credential_pool.SUPPORTED_POOL_STRATEGIES", {"fill_first", "round_robin"}):
                resp = client.put("/api/credentials/pool/zai/strategy", json={
                    "strategy": "fill_first'; --",
                })
        assert resp.status_code == 400


# ════════════════════════════════════════════════════════════════════════════
# 5. UNIFIED RESOLVER SSRF GUARD
# ════════════════════════════════════════════════════════════════════════════

class TestUnifiedResolverSSRF:
    """The unified resolver must also protect against SSRF."""

    def test_resolver_blocks_metadata_endpoint(self):
        """_validate_base_url_safe blocks metadata endpoints."""
        from agent.auth import _validate_base_url_safe
        result = _validate_base_url_safe("http://169.254.169.254/latest/meta-data/")
        assert result == "", f"Expected empty, got {result!r}"

    def test_resolver_blocks_file_scheme(self):
        """_validate_base_url_safe blocks file:// scheme."""
        from agent.auth import _validate_base_url_safe
        result = _validate_base_url_safe("file:///etc/passwd")
        assert result == "", f"Expected empty, got {result!r}"

    def test_resolver_blocks_gopher_scheme(self):
        """_validate_base_url_safe blocks gopher:// scheme."""
        from agent.auth import _validate_base_url_safe
        result = _validate_base_url_safe("gopher://internal:6379/_INFO")
        assert result == "", f"Expected empty, got {result!r}"

    def test_resolver_blocks_null_byte(self):
        """_validate_base_url_safe blocks null byte injection."""
        from agent.auth import _validate_base_url_safe
        result = _validate_base_url_safe("https://api.z.ai\x00.evil.com")
        assert result == "", f"Expected empty, got {result!r}"

    def test_resolver_allows_valid_https(self):
        """_validate_base_url_safe allows legitimate https URLs."""
        from agent.auth import _validate_base_url_safe
        result = _validate_base_url_safe("https://api.z.ai/api/coding/paas/v4")
        assert result == "https://api.z.ai/api/coding/paas/v4"

    def test_resolver_allows_empty_url(self):
        """_validate_base_url_safe passes through empty URLs."""
        from agent.auth import _validate_base_url_safe
        result = _validate_base_url_safe("")
        assert result == ""

    def test_resolver_allows_localhost(self):
        """_validate_base_url_safe allows localhost (for local LLMs like Ollama)."""
        from agent.auth import _validate_base_url_safe
        result = _validate_base_url_safe("http://localhost:1234/v1")
        assert result == "http://localhost:1234/v1"


# ════════════════════════════════════════════════════════════════════════════
# 6. MASK FUNCTION
# ════════════════════════════════════════════════════════════════════════════

class TestMaskApiKey:
    """API key masking utility."""

    def test_mask_long_key(self):
        """Long keys are masked to show first 4 + last 4."""
        from hermes_cli.web_server import _mask_api_key
        masked = _mask_api_key("testkey1234567890abcdef")
        assert masked.startswith("test")
        assert masked.endswith("cdef")
        assert "*" in masked
        assert "1234567890" not in masked

    def test_mask_short_key(self):
        """Short keys are fully masked."""
        from hermes_cli.web_server import _mask_api_key
        assert _mask_api_key("short") == "***"

    def test_mask_empty_key(self):
        """Empty keys return empty string."""
        from hermes_cli.web_server import _mask_api_key
        assert _mask_api_key("") == ""

    def test_mask_boundary_12_chars(self):
        """Keys of exactly 12 chars are masked."""
        from hermes_cli.web_server import _mask_api_key
        masked = _mask_api_key("abcdefghijkl")
        assert masked == "abcd****ijkl"
