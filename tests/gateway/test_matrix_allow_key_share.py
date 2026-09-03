"""Tests for the Matrix cross-user key-share opt-in.

Covers two surfaces:

- ``matrix.allow_key_share`` -> ``MATRIX_ALLOW_KEY_SHARE`` mapping (the
  ``_apply_yaml_config`` path, mirroring the other ``matrix:`` keys).
- ``_normalize_allow_key_share``, which folds the loose user-facing values
  (booleans, ``allowed_users``, ``all``) into the canonical ``false`` /
  ``allowed-users`` / ``all`` modes.

The knob opts the bot into honoring ``m.room_key_request`` from other users,
which makes a client's "Request Key" button a working recovery path. mautrix's
default key-share policy silently drops cross-user key requests, so the flag
must be opt-in (default ``false``).
"""
import os

import pytest

from plugins.platforms.matrix.adapter import (
    _apply_yaml_config,
    _normalize_allow_key_share,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("MATRIX_ALLOW_KEY_SHARE", raising=False)
    yield
    monkeypatch.delenv("MATRIX_ALLOW_KEY_SHARE", raising=False)


class TestAllowKeyShareMapping:
    def test_yaml_true_sets_env_true(self):
        _apply_yaml_config({}, {"allow_key_share": True})
        assert os.environ["MATRIX_ALLOW_KEY_SHARE"] == "true"

    def test_yaml_false_sets_env_false(self):
        _apply_yaml_config({}, {"allow_key_share": False})
        assert os.environ["MATRIX_ALLOW_KEY_SHARE"] == "false"

    def test_yaml_string_is_normalized(self):
        _apply_yaml_config({}, {"allow_key_share": "True"})
        assert os.environ["MATRIX_ALLOW_KEY_SHARE"] == "true"

    def test_env_precedence_over_yaml(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ALLOW_KEY_SHARE", "false")
        _apply_yaml_config({}, {"allow_key_share": True})
        assert os.environ["MATRIX_ALLOW_KEY_SHARE"] == "false"

    def test_absent_key_leaves_env_unset(self):
        _apply_yaml_config({}, {})
        assert "MATRIX_ALLOW_KEY_SHARE" not in os.environ


class TestNormalizeAllowKeyShare:
    def test_default_is_false(self):
        assert _normalize_allow_key_share("") == "false"
        assert _normalize_allow_key_share("false") == "false"
        assert _normalize_allow_key_share("off") == "false"
        assert _normalize_allow_key_share("garbage") == "false"

    def test_booleans_map_to_all(self):
        assert _normalize_allow_key_share("true") == "all"
        assert _normalize_allow_key_share("1") == "all"
        assert _normalize_allow_key_share("yes") == "all"

    def test_all_spellings(self):
        assert _normalize_allow_key_share("all") == "all"

    def test_allowed_users_spellings(self):
        assert _normalize_allow_key_share("allowed-users") == "allowed-users"
        assert _normalize_allow_key_share("allowed_users") == "allowed-users"
        assert _normalize_allow_key_share("allowed") == "allowed-users"

    def test_case_and_whitespace_insensitive(self):
        assert _normalize_allow_key_share("  ALL  ") == "all"
        assert _normalize_allow_key_share("Allowed-Users") == "allowed-users"
