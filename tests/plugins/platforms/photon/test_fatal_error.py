"""#68693: UPSTREAM_STREAM_DEGRADED must be non-retryable so the gateway
exits and the service manager (launchd/systemd) restarts it."""

from __future__ import annotations

import inspect

import pytest

from plugins.platforms.photon.adapter import PhotonAdapter


def test_upstream_stream_degraded_is_non_retryable():
    """The _set_fatal_error call for UPSTREAM_STREAM_DEGRADED must pass
    retryable=False so the gateway exits for service-manager restart."""
    src = inspect.getsource(PhotonAdapter)
    # Find the UPSTREAM_STREAM_DEGRADED block
    idx = src.find("UPSTREAM_STREAM_DEGRADED")
    assert idx >= 0, "UPSTREAM_STREAM_DEGRADED not found in PhotonAdapter"
    # Check the nearby code for retryable=False
    block = src[idx:idx + 400]
    assert "retryable=False" in block, (
        "UPSTREAM_STREAM_DEGRADED must set retryable=False so the gateway "
        "exits for launchd/systemd restart (#68693)"
    )


def test_sidecar_crashed_still_retryable():
    """SIDECAR_CRASHED should remain retryable — the sidecar can be
    re-started by the reconnect watcher without a full gateway restart."""
    src = inspect.getsource(PhotonAdapter)
    idx = src.find("SIDECAR_CRASHED")
    assert idx >= 0
    block = src[idx:idx + 400]
    assert "retryable=True" in block, (
        "SIDECAR_CRASHED should remain retryable for background reconnection"
    )