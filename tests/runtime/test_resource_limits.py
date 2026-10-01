"""Tests for runtime-owned configurable RLIMIT_NOFILE handling."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import textwrap
import pytest

from runtime import resource_limits


class _FakeResource:
    RLIMIT_NOFILE = 7
    RLIM_INFINITY = 2**63 - 1

    def __init__(self, soft: int, hard: int) -> None:
        self.limits = (soft, hard)
        self.set_calls: list[tuple[int, tuple[int, int]]] = []

    def getrlimit(self, resource: int) -> tuple[int, int]:
        assert resource == self.RLIMIT_NOFILE
        return self.limits

    def setrlimit(self, resource: int, limits: tuple[int, int]) -> None:
        assert resource == self.RLIMIT_NOFILE
        self.set_calls.append((resource, limits))
        self.limits = limits


def test_default_is_clamped_to_hard_limit(monkeypatch):
    fake_resource = _FakeResource(soft=256, hard=1024)
    monkeypatch.setattr(resource_limits, "_resource", fake_resource)

    assert resource_limits.apply_nofile_soft_limit({}) is True
    assert fake_resource.limits == (1024, 1024)


def test_finite_soft_limit_raises_when_hard_limit_is_infinite(monkeypatch):
    fake_resource = _FakeResource(soft=256, hard=_FakeResource.RLIM_INFINITY)
    monkeypatch.setattr(resource_limits, "_resource", fake_resource)

    assert resource_limits.apply_nofile_soft_limit({}) is True
    assert fake_resource.set_calls == [
        (
            fake_resource.RLIMIT_NOFILE,
            (4096, fake_resource.RLIM_INFINITY),
        ),
    ]


def test_never_lowers_an_already_higher_soft_limit(monkeypatch):
    fake_resource = _FakeResource(soft=8192, hard=16384)
    monkeypatch.setattr(resource_limits, "_resource", fake_resource)

    assert resource_limits.apply_nofile_soft_limit(
        {"runtime": {"nofile_soft_limit": 4096}}
    ) is False
    assert fake_resource.set_calls == []
    assert fake_resource.limits == (8192, 16384)


@pytest.mark.parametrize("disabled", [0, False, None])
def test_explicit_values_disable(monkeypatch, disabled):
    fake_resource = _FakeResource(soft=256, hard=4096)
    monkeypatch.setattr(resource_limits, "_resource", fake_resource)

    assert resource_limits.apply_nofile_soft_limit(
        {"runtime": {"nofile_soft_limit": disabled}}
    ) is False
    assert fake_resource.set_calls == []


def test_unsupported_platform_is_a_safe_noop(monkeypatch):
    monkeypatch.setattr(resource_limits, "_resource", None)

    assert resource_limits.apply_nofile_soft_limit({}) is False


def test_fresh_process_import_without_posix_resource_is_a_safe_noop():
    code = textwrap.dedent(
        """
        import importlib.util
        import pathlib
        import sys

        sys.modules["resource"] = None
        module_path = pathlib.Path(sys.argv[1])
        spec = importlib.util.spec_from_file_location(
            "runtime._resource_limits_without_posix_resource",
            module_path,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert module._resource is None
        assert module.apply_nofile_soft_limit({}) is False
        """
    )

    subprocess.run(
        [sys.executable, "-c", code, resource_limits.__file__],
        check=True,
        cwd=Path(resource_limits.__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("invalid", [True, -1, 4096.0, "4096", object()])
def test_invalid_values_are_safe_noops(monkeypatch, invalid):
    fake_resource = _FakeResource(soft=256, hard=4096)
    monkeypatch.setattr(resource_limits, "_resource", fake_resource)

    assert resource_limits.apply_nofile_soft_limit(
        {"runtime": {"nofile_soft_limit": invalid}}
    ) is False
    assert fake_resource.set_calls == []


def test_setrlimit_denial_is_a_safe_noop(monkeypatch):
    class _DeniedResource(_FakeResource):
        def setrlimit(self, resource: int, limits: tuple[int, int]) -> None:
            raise PermissionError("simulated EPERM")

    fake_resource = _DeniedResource(soft=256, hard=4096)
    monkeypatch.setattr(resource_limits, "_resource", fake_resource)

    assert resource_limits.apply_nofile_soft_limit({}) is False
    assert fake_resource.limits == (256, 4096)


def test_getrlimit_failure_is_a_safe_noop(monkeypatch):
    class _BrokenResource(_FakeResource):
        def getrlimit(self, resource: int) -> tuple[int, int]:
            raise OSError("simulated getrlimit failure")

    fake_resource = _BrokenResource(soft=256, hard=4096)
    monkeypatch.setattr(resource_limits, "_resource", fake_resource)

    assert resource_limits.apply_nofile_soft_limit({}) is False
    assert fake_resource.set_calls == []


def test_never_lowers_an_unlimited_soft_limit(monkeypatch):
    fake_resource = _FakeResource(soft=-1, hard=-1)
    fake_resource.RLIM_INFINITY = -1
    monkeypatch.setattr(resource_limits, "_resource", fake_resource)

    assert resource_limits.apply_nofile_soft_limit({}) is False
    assert fake_resource.set_calls == []
    assert fake_resource.limits == (-1, -1)
