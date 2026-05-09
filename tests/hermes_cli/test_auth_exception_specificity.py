"""Regression tests: auth.py exception handlers narrowed from Exception to specific types.

Stash-verify anchors use exceptions outside the narrowed set (e.g. RuntimeError)
to confirm the handler no longer swallows arbitrary errors.
"""

import base64
import json
import pytest

from hermes_cli.auth import (
    _decode_jwt_claims,
    _coerce_ttl_seconds,
    _parse_iso_timestamp,
    _load_auth_store,
)


class TestDecodeJwtClaims:
    def test_returns_empty_on_malformed_base64(self):
        """binascii.Error (subclass of ValueError) from bad base64 returns {}."""
        bad_token = "header.!!!notbase64!!!.sig"
        result = _decode_jwt_claims(bad_token)
        assert result == {}

    def test_returns_empty_on_non_json_payload(self):
        """JSONDecodeError (subclass of ValueError) from non-JSON payload returns {}."""
        payload = base64.urlsafe_b64encode(b"not-json{").decode()
        bad_token = f"header.{payload}.sig"
        result = _decode_jwt_claims(bad_token)
        assert result == {}

    def test_returns_claims_on_valid_jwt(self):
        """Sanity: valid JWT payload returns parsed claims."""
        claims = {"sub": "user-123", "exp": 9999999999}
        payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
        token = f"header.{payload}.sig"
        result = _decode_jwt_claims(token)
        assert result["sub"] == "user-123"

    def test_propagates_runtime_error(self):
        """RuntimeError outside the narrowed set must propagate, not be swallowed.

        Stash-verify anchor: fails with old ``except Exception`` (swallows),
        passes after narrowing to ``except (ValueError, UnicodeDecodeError)``.
        """
        from unittest.mock import patch

        with patch("base64.urlsafe_b64decode", side_effect=RuntimeError("unexpected")):
            with pytest.raises(RuntimeError):
                _decode_jwt_claims("a.b.c")


class TestCoerceTtlSeconds:
    def test_returns_zero_on_none(self):
        """None raises TypeError — must be caught, return 0."""
        assert _coerce_ttl_seconds(None) == 0

    def test_returns_zero_on_non_numeric_string(self):
        """Non-numeric string raises ValueError — must be caught, return 0."""
        assert _coerce_ttl_seconds("not-a-number") == 0

    def test_returns_value_on_valid_int(self):
        assert _coerce_ttl_seconds(3600) == 3600

    def test_returns_value_on_numeric_string(self):
        assert _coerce_ttl_seconds("1800") == 1800

    def test_propagates_runtime_error(self):
        """RuntimeError from unexpected source must propagate, not be swallowed.

        Stash-verify anchor: fails under old ``except Exception``.
        """
        from unittest.mock import patch

        with patch("builtins.int", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                _coerce_ttl_seconds("3600")


class TestParseIsoTimestamp:
    def test_returns_none_on_invalid_format(self):
        """ValueError from fromisoformat must be caught, return None."""
        result = _parse_iso_timestamp("not-a-date")
        assert result is None

    def test_returns_float_on_valid_utc(self):
        result = _parse_iso_timestamp("2025-01-01T00:00:00Z")
        assert isinstance(result, float)
        assert result > 0

class TestLoadAuthStore:
    def test_returns_empty_store_on_malformed_json(self, tmp_path):
        """JSONDecodeError from corrupt auth.json returns empty store, not raise."""
        auth_file = tmp_path / "auth.json"
        auth_file.write_text("{ bad json [[[")

        result = _load_auth_store(auth_file)

        assert isinstance(result.get("providers"), dict)
        assert result["providers"] == {}

    def test_creates_corrupt_backup_on_malformed_json(self, tmp_path):
        """Corrupt auth.json is backed up to auth.json.corrupt before resetting."""
        auth_file = tmp_path / "auth.json"
        auth_file.write_text("{ bad json [[[")

        _load_auth_store(auth_file)

        corrupt = tmp_path / "auth.json.corrupt"
        assert corrupt.exists(), "backup of corrupt file should be created"
