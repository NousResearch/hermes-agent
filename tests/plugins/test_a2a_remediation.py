"""Focused regression tests for the two bounded remediations:

1. MEDIUM: _persist_context_peers must create the temp file with 0o600
   permissions before the atomic replace (matching session persistence).
2. LOW: _patience_for must cap peer-supplied sender.timeout at
   _ORPHAN_TIMEOUT - _PATIENCE_MARGIN (270s) and reject non-finite/
   negative/over-ceiling values consistently.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from plugins.platforms.a2a import adapter as adapter_mod
from plugins.platforms.a2a import protocol
from plugins.platforms.a2a import a2a_persistence


# ═════════════════════════════════════════════════════════════════════════════
# 1. Context-peer persistence mode — 0o600
# ═════════════════════════════════════════════════════════════════════════════


class TestContextPeerPersistenceMode:
    """Verify _persist_context_peers writes the temp file with 0o600."""

    def test_tmp_file_gets_0o600_before_replace(self, tmp_path, monkeypatch):
        """The final persisted file must end up with mode 0o600."""
        peers_file = tmp_path / "a2a_context_peers.json"
        monkeypatch.setattr(
            a2a_persistence,
            "_context_peers_path",
            lambda: peers_file,
        )
        peers = {"ctx-alpha": "peer-alpha", "ctx-beta": "peer-beta"}
        adapter_mod._persist_context_peers(peers)
        assert peers_file.exists(), "peer file should exist after persist"
        mode = os.stat(peers_file).st_mode & 0o777
        assert mode == 0o600, (
            f"Expected 0o600, got octal {oct(mode)} — "
            "context-peer file must be user-rw only"
        )

    def test_content_is_valid_json(self, tmp_path, monkeypatch):
        """The file must contain valid JSON after persist."""
        peers_file = tmp_path / "a2a_context_peers.json"
        monkeypatch.setattr(
            a2a_persistence,
            "_context_peers_path",
            lambda: peers_file,
        )
        adapter_mod._persist_context_peers({"ctx-x": "peer-x"})
        data = json.loads(peers_file.read_text())
        assert data == {"ctx-x": "peer-x"}

    def test_empty_dict_persists(self, tmp_path, monkeypatch):
        """Persisting an empty dict still produces a valid file with 0o600."""
        peers_file = tmp_path / "a2a_context_peers.json"
        monkeypatch.setattr(
            a2a_persistence,
            "_context_peers_path",
            lambda: peers_file,
        )
        adapter_mod._persist_context_peers({})
        assert peers_file.exists()
        mode = os.stat(peers_file).st_mode & 0o777
        assert mode == 0o600, f"Empty-dict persist: expected 0o600, got {oct(mode)}"
        assert json.loads(peers_file.read_text()) == {}


# ═════════════════════════════════════════════════════════════════════════════
# 2. Sender-timeout ceiling — _patience_for
# ═════════════════════════════════════════════════════════════════════════════

_TIMEOUT_CEILING = adapter_mod._ORPHAN_TIMEOUT - adapter_mod._PATIENCE_MARGIN  # 270s


def _bare_adapter():
    from gateway.config import PlatformConfig
    return adapter_mod.A2AAdapter(PlatformConfig(enabled=True))


def _patience_for(adapter, params, peer="test-peer"):
    """Call _patience_for on a bare adapter with the given params."""
    return adapter._patience_for(params, peer)


class TestPatienceForTimeoutCeiling:
    """Verify _patience_for applies the timeout ceiling consistently."""

    def test_absent_sender_returns_default(self):
        """No sender → default 120s."""
        adapter = _bare_adapter()
        result = _patience_for(adapter, {"message": {}})
        assert result == 120.0

    def test_absent_message_returns_default(self):
        """No message key → default 120s."""
        adapter = _bare_adapter()
        result = _patience_for(adapter, {})
        assert result == 120.0

    def test_normal_timeout_passes_through(self):
        """A peer-supplied timeout under the ceiling is returned as-is."""
        adapter = _bare_adapter()
        result = _patience_for(
            adapter,
            {"message": protocol.text_message(protocol.ROLE_USER, "test", sender={"timeout": 60.0})},
        )
        assert result == 60.0

    def test_timeout_at_ceiling(self):
        """A timeout exactly at the ceiling is returned as-is."""
        adapter = _bare_adapter()
        result = _patience_for(
            adapter,
            {"message": protocol.text_message(protocol.ROLE_USER, "test", sender={"timeout": float(_TIMEOUT_CEILING)})},
        )
        assert result == float(_TIMEOUT_CEILING)

    def test_timeout_over_ceiling_is_clamped(self):
        """A timeout above the ceiling is clamped to the ceiling."""
        adapter = _bare_adapter()
        result = _patience_for(
            adapter,
            {"message": protocol.text_message(protocol.ROLE_USER, "test", sender={"timeout": 500.0})},
        )
        assert result == float(_TIMEOUT_CEILING), (
            f"Over-ceiling 500.0 should clamp to {_TIMEOUT_CEILING}"
        )

    def test_timeout_very_large_is_clamped(self):
        """An absurdly large timeout is clamped to the ceiling."""
        adapter = _bare_adapter()
        result = _patience_for(
            adapter,
            {"message": protocol.text_message(protocol.ROLE_USER, "test", sender={"timeout": 999999.0})},
        )
        assert result == float(_TIMEOUT_CEILING)

    def test_zero_timeout_falls_through_to_default(self):
        """Zero sender.timeout is treated as absent → default 120s."""
        adapter = _bare_adapter()
        result = _patience_for(
            adapter,
            {"message": protocol.text_message(protocol.ROLE_USER, "test", sender={"timeout": 0})},
        )
        assert result == 120.0

    def test_negative_timeout_falls_through_to_default(self):
        """Negative sender.timeout is rejected → default 120s."""
        adapter = _bare_adapter()
        result = _patience_for(
            adapter,
            {"message": protocol.text_message(protocol.ROLE_USER, "test", sender={"timeout": -5.0})},
        )
        assert result == 120.0

    def test_none_timeout_falls_through_to_default(self):
        """sender.timeout=None is treated as absent → default 120s."""
        adapter = _bare_adapter()
        result = _patience_for(
            adapter,
            {"message": protocol.text_message(protocol.ROLE_USER, "test", sender={"timeout": None})},
        )
        assert result == 120.0

    def test_nan_timeout_falls_through_to_default(self):
        """NaN sender.timeout is non-finite → default 120s."""
        adapter = _bare_adapter()
        result = _patience_for(
            adapter,
            {"message": protocol.text_message(protocol.ROLE_USER, "test", sender={"timeout": float("nan")})},
        )
        assert result == 120.0

    def test_inf_timeout_falls_through_to_default(self):
        """Inf sender.timeout is non-finite → default 120s."""
        adapter = _bare_adapter()
        result = _patience_for(
            adapter,
            {"message": protocol.text_message(protocol.ROLE_USER, "test", sender={"timeout": float("inf")})},
        )
        assert result == 120.0

    def test_neg_inf_timeout_falls_through_to_default(self):
        """-Inf sender.timeout is non-finite → default 120s."""
        adapter = _bare_adapter()
        result = _patience_for(
            adapter,
            {"message": protocol.text_message(protocol.ROLE_USER, "test", sender={"timeout": float("-inf")})},
        )
        assert result == 120.0

    def test_string_timeout_falls_through_to_default(self):
        """Non-numeric sender.timeout → default 120s."""
        adapter = _bare_adapter()
        result = _patience_for(
            adapter,
            {"message": protocol.text_message(protocol.ROLE_USER, "test", sender={"timeout": "not-a-number"})},
        )
        assert result == 120.0

    def test_sender_not_dict_falls_through(self):
        """sender is a string, not a dict → default 120s."""
        adapter = _bare_adapter()
        result = _patience_for(
            adapter,
            {"message": {"sender": "just-a-string"}},
        )
        assert result == 120.0

    def test_ceiling_is_derived_from_constants(self):
        """The ceiling is _ORPHAN_TIMEOUT - _PATIENCE_MARGIN."""
        assert adapter_mod._ORPHAN_TIMEOUT == 300
        assert adapter_mod._PATIENCE_MARGIN == 30
        assert _TIMEOUT_CEILING == 270
