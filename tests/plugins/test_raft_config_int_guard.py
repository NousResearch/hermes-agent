"""Raft adapter must tolerate malformed port / max_body_bytes in config.extra."""

from __future__ import annotations

from gateway.config import PlatformConfig
from plugins.platforms.raft.adapter import (
    DEFAULT_MAX_BODY_BYTES,
    DEFAULT_PORT,
    RaftAdapter,
)


def _make_config(**extra):
    data = {
        "bridge_token": "bridge-secret",
        "runtime_session": "default",
        "port": DEFAULT_PORT,
    }
    data.update(extra)
    return PlatformConfig(enabled=True, extra=data)


def test_malformed_port_falls_back_to_default():
    adapter = RaftAdapter(_make_config(port="not-a-port"))
    assert adapter._port == DEFAULT_PORT


def test_malformed_max_body_bytes_falls_back_to_default():
    adapter = RaftAdapter(_make_config(max_body_bytes="nope"))
    assert adapter._max_body_bytes == DEFAULT_MAX_BODY_BYTES


def test_valid_port_from_extra_is_honored():
    adapter = RaftAdapter(_make_config(port=8088))
    assert adapter._port == 8088
