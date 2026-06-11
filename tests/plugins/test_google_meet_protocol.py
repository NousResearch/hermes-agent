"""Tests for the google_meet node RPC wire protocol.

Focus: the shared-token check in ``validate_request`` must be timing-safe
(CWE-208) so the bearer token cannot be recovered byte-by-byte via a
validation-timing side channel.
"""
import importlib.util
from pathlib import Path

import pytest

# The protocol module ships inside a plugin tree that is not an importable
# package, so load it directly from its file path.
_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[2]
    / "plugins"
    / "google_meet"
    / "node"
    / "protocol.py"
)


def _load_protocol():
    spec = importlib.util.spec_from_file_location(
        "google_meet_node_protocol", _PROTOCOL_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def protocol():
    return _load_protocol()


def _request(protocol, token):
    return protocol.make_request("ping", token, {})


def test_correct_token_accepted(protocol):
    ok, reason = protocol.validate_request(_request(protocol, "sekret"), "sekret")
    assert ok is True
    assert reason == ""


def test_wrong_token_rejected(protocol):
    ok, reason = protocol.validate_request(_request(protocol, "wrong"), "sekret")
    assert ok is False
    assert reason == "token mismatch"


def test_token_check_uses_constant_time_compare(protocol, monkeypatch):
    calls = []

    def spy(a, b):
        calls.append((a, b))
        return False

    monkeypatch.setattr(protocol.hmac, "compare_digest", spy)
    ok, reason = protocol.validate_request(_request(protocol, "sekret"), "expected")

    assert ok is False
    assert reason == "token mismatch"
    # Both operands must flow through the constant-time comparator; a plain
    # ``!=`` fast path would never touch hmac.compare_digest.
    assert calls, "token validation must use hmac.compare_digest"
    assert calls[0][0] == b"sekret"
    assert calls[0][1] == b"expected"
