"""HERMES_WRITE_SAFE_ROOT must honor the active profile secret scope (#136204).

A desktop/gateway-served profile's ``.env`` is installed as a secret scope, never
exported to ``os.environ``, so the shared process's plain env read ignored that
profile's write limit. These tests pin the resolution order:

1. a scoped value wins over the process env (the routed profile's own limit);
2. a scope miss falls back to the process env (a deployment-wide limit keeps
   constraining profiles that set none — the fallback is fail-closed);
3. an unscoped multiplexed reader (outside any turn scope) still sees the
   process env instead of refusing to classify writes;
4. single-profile CLI behavior (no scope, multiplex off) is unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent import file_safety as fs
from agent.secret_scope import (
    reset_multiplex_context,
    reset_secret_scope,
    set_multiplex_context,
    set_secret_scope,
)


@pytest.fixture
def multiplex():
    token = set_multiplex_context(True)
    try:
        yield
    finally:
        reset_multiplex_context(token)


@pytest.fixture
def launch_limit(monkeypatch, tmp_path):
    """A process-wide (launch profile / deployment) write limit."""
    launch_root = tmp_path / "launch-allowed"
    launch_root.mkdir()
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(launch_root))
    return launch_root


def _bind_demo_scope(tmp_path: Path, value: str | None) -> object:
    """Install the routed profile's secret scope; ``None`` means its ``.env`` has no key."""
    mapping = {} if value is None else {"HERMES_WRITE_SAFE_ROOT": value}
    return set_secret_scope(mapping, profile_home=str(tmp_path / "profiles" / "demo"))


class TestScopedSafeRootResolution:
    def test_scoped_profile_limit_wins_over_process_env(
        self, tmp_path, multiplex, launch_limit
    ):
        demo_root = tmp_path / "demo-allowed"
        demo_root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        token = _bind_demo_scope(tmp_path, str(demo_root))
        try:
            assert fs.is_write_denied(str(outside / "f.txt")) is True
            assert fs.is_write_denied(str(demo_root / "f.txt")) is False
            # The launch root is NOT part of the scoped limit anymore.
            assert fs.is_write_denied(str(launch_limit / "f.txt")) is True
        finally:
            reset_secret_scope(token)

    def test_scope_miss_keeps_process_env_limit(
        self, tmp_path, multiplex, launch_limit
    ):
        other = tmp_path / "other"
        other.mkdir()
        token = _bind_demo_scope(tmp_path, None)
        try:
            assert fs.get_safe_write_roots() == {os.path.realpath(str(launch_limit))}
            assert fs.is_write_denied(str(other / "f.txt")) is True
            assert fs.is_write_denied(str(launch_limit / "f.txt")) is False
        finally:
            reset_secret_scope(token)

    def test_unscoped_multiplex_reader_still_sees_process_env(
        self, tmp_path, multiplex, launch_limit
    ):
        other = tmp_path / "other"
        other.mkdir()
        # No secret scope installed: an out-of-turn reader must not crash or
        # silently lose the deployment-wide limit.
        assert fs.get_safe_write_roots() == {os.path.realpath(str(launch_limit))}
        assert fs.is_write_denied(str(other / "f.txt")) is True

    def test_denied_error_names_the_scoped_roots(
        self, tmp_path, multiplex, launch_limit
    ):
        demo_root = tmp_path / "demo-allowed"
        demo_root.mkdir()
        target = tmp_path / "outside" / "f.txt"
        token = _bind_demo_scope(tmp_path, str(demo_root))
        try:
            message = fs.get_write_denied_error(str(target))
        finally:
            reset_secret_scope(token)
        assert message is not None
        assert str(demo_root) in message
        assert str(launch_limit) not in message

    def test_single_profile_cli_reads_process_env(self, tmp_path, monkeypatch):
        allowed = tmp_path / "cli-allowed"
        allowed.mkdir()
        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(allowed))
        assert fs.get_safe_write_roots() == {os.path.realpath(str(allowed))}
        assert fs.is_write_denied(str(tmp_path / "elsewhere" / "f.txt")) is True
        assert fs.is_write_denied(str(allowed / "f.txt")) is False
