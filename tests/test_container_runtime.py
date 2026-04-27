"""Tests for container_runtime detection."""

from container_runtime import detect_container_runtime


def test_detect_container_runtime_returns_string() -> None:
    r = detect_container_runtime()
    assert r in ("docker", "podman", "none")
