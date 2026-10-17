"""Regression coverage for systemd unit detection from /proc/self/cgroup."""

from __future__ import annotations

import builtins
import io
from types import SimpleNamespace

import pytest

from gateway import shutdown_forensics as sf


def _patch_cgroup(monkeypatch: pytest.MonkeyPatch, content: str) -> None:
    original_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if path == "/proc/self/cgroup":
            return io.StringIO(content)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)


def test_transient_scope_does_not_resolve_parent_user_service(monkeypatch):
    monkeypatch.setenv("INVOCATION_ID", "test-invocation")
    _patch_cgroup(
        monkeypatch,
        "0::/user.slice/user-1000.slice/user@1000.service/app.slice/hermes-worker-abc.scope\n",
    )

    def unexpected_systemctl(*args, **kwargs):
        pytest.fail("systemctl should not be queried for a transient scope")

    monkeypatch.setattr(sf.subprocess, "run", unexpected_systemctl)

    assert sf.check_systemd_timing_alignment(180.0) is None


def test_direct_service_leaf_is_queried(monkeypatch):
    monkeypatch.setenv("INVOCATION_ID", "test-invocation")
    _patch_cgroup(
        monkeypatch,
        "0::/user.slice/user-1000.slice/user@1000.service/app.slice/hermes-gateway.service\n",
    )
    calls = []

    def fake_systemctl(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout="TimeoutStopUSec=240000000\n")

    monkeypatch.setattr(sf.subprocess, "run", fake_systemctl)

    result = sf.check_systemd_timing_alignment(180.0)

    assert result is not None
    assert result["unit"] == "hermes-gateway.service"
    assert result["mismatch"] is False
    assert calls == [
        [
            "systemctl",
            "--user",
            "show",
            "hermes-gateway.service",
            "--property=TimeoutStopUSec",
        ]
    ]


def test_delegated_subgroup_resolves_owning_service(monkeypatch):
    monkeypatch.setenv("INVOCATION_ID", "test-invocation")
    _patch_cgroup(
        monkeypatch,
        "0::/user.slice/user-1000.slice/user@1000.service/app.slice/hermes-gateway.service/worker\n",
    )
    calls = []

    def fake_systemctl(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout="TimeoutStopUSec=240000000\n")

    monkeypatch.setattr(sf.subprocess, "run", fake_systemctl)

    result = sf.check_systemd_timing_alignment(180.0)

    assert result is not None
    assert result["unit"] == "hermes-gateway.service"
    assert result["mismatch"] is False
    assert calls == [
        [
            "systemctl",
            "--user",
            "show",
            "hermes-gateway.service",
            "--property=TimeoutStopUSec",
        ]
    ]
