"""Test coverage for agent/secret_sources/onepassword.py — 19 functions had LOW coverage.

Tests the pure helper functions: disk key formatting, reference validation,
auth fingerprinting, cache path, error classification, and scrubbing.
All network and subprocess calls are mocked — no real 1Password access.
"""

import pytest
from pathlib import Path

from agent.secret_sources.onepassword import (
    _auth_fingerprint,
    _classify_op_error,
    _disk_key_str,
    _refs_fingerprint,
    _scrub,
    _validate_references,
)


class TestDiskKeyStr:
    def test_formats_cache_key_with_all_fields(self):
        result = _disk_key_str(("provider", "field", "ref", "extra"))
        assert isinstance(result, str)
        assert len(result) > 0
        # Deterministic: same input -> same output
        assert result == _disk_key_str(("provider", "field", "ref", "extra"))

    def test_different_keys_different_results(self):
        a = _disk_key_str(("a", "b", "c", "d"))
        b = _disk_key_str(("x", "y", "z", "w"))
        assert a != b


class TestValidateReferences:
    def test_returns_tuple_of_dict_and_list(self):
        valid, warnings = _validate_references({})
        assert isinstance(valid, dict)
        assert isinstance(warnings, list)

    def test_valid_op_ref_is_kept(self):
        valid, warnings = _validate_references({"DB_PASS": "op://Vault/Item/Field"})
        assert valid == {"DB_PASS": "op://Vault/Item/Field"}
        assert warnings == []

    def test_non_op_ref_is_dropped_with_warning(self):
        valid, warnings = _validate_references({"DB_PASS": "plaintext-password"})
        assert "DB_PASS" not in valid
        assert len(warnings) == 1
        assert "DB_PASS" in warnings[0]

    def test_invalid_env_name_is_dropped(self):
        valid, warnings = _validate_references({"1invalid-name!": "op://V/I/F"})
        assert "1invalid-name!" not in valid
        assert len(warnings) == 1

    def test_non_string_ref_is_dropped(self):
        valid, warnings = _validate_references({"DB_PASS": 12345})
        assert "DB_PASS" not in valid
        assert len(warnings) == 1

    def test_empty_dict_returns_empty(self):
        valid, warnings = _validate_references({})
        assert valid == {}
        assert warnings == []

    def test_none_returns_empty(self):
        valid, warnings = _validate_references(None)
        assert valid == {}
        assert warnings == []

    def test_whitespace_in_ref_is_stripped(self):
        valid, _ = _validate_references({"K": "  op://V/I/F  "})
        assert valid["K"] == "op://V/I/F"


class TestAuthFingerprint:
    def test_returns_nonempty_string(self):
        fp = _auth_fingerprint("my-service-token")
        assert isinstance(fp, str)
        assert len(fp) > 0

    def test_deterministic_same_input(self):
        assert _auth_fingerprint("token-abc") == _auth_fingerprint("token-abc")


class TestRefsFingerprint:
    def test_deterministic(self):
        refs = {"KEY": "op://Vault/Item/Field"}
        assert _refs_fingerprint(refs) == _refs_fingerprint(refs)

    def test_different_refs_different_fps(self):
        assert _refs_fingerprint({"K": "op://a/b/c"}) != _refs_fingerprint({"K": "op://x/y/z"})

    def test_order_independent(self):
        a = _refs_fingerprint({"K1": "op://1", "K2": "op://2"})
        b = _refs_fingerprint({"K2": "op://2", "K1": "op://1"})
        assert a == b


class TestScrub:
    def test_strips_ansi_escape_sequences(self):
        result = _scrub("\x1b[31mError:\x1b[0m something failed")
        assert "\x1b[" not in result
        assert "Error:" in result
        assert "something failed" in result

    def test_plain_text_unchanged(self):
        assert _scrub("no special characters") == "no special characters"


class TestClassifyOpError:
    def test_auth_error_classified(self):
        kind = _classify_op_error("authentication failed: invalid session token")
        assert kind is not None

    def test_not_found_classified(self):
        kind = _classify_op_error("item not found in vault")
        assert kind is not None

    def test_generic_error_classified(self):
        kind = _classify_op_error("connection timed out unexpectedly")
        assert kind is not None
