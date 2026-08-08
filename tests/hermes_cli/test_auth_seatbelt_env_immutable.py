"""Regression coverage for env-immutable auth-store test seat belts."""

from contextlib import contextmanager
import os
from pathlib import Path

import pytest

pwd = pytest.importorskip(
    "pwd",
    reason="dangerous-path derivation requires the POSIX passwd database",
)

from agent import credential_pool
from hermes_cli import auth


def _real_auth_path() -> Path:
    """Derive the dangerous path independently of the helpers under test."""
    home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    return (home / ".hermes" / "auth.json").resolve(strict=False)


def _configure_dangerous_profile(monkeypatch, tmp_path) -> Path:
    """Point a profile process at the independently derived protected store."""
    dangerous_auth = _real_auth_path()
    monkeypatch.setenv("HOME", str(tmp_path / "patched-home"))
    monkeypatch.setenv(
        "HERMES_HOME",
        str(dangerous_auth.parent / "profiles" / "seatbelt-test"),
    )
    global_path = auth._global_auth_file_path()
    assert global_path is not None
    assert global_path.resolve(strict=False) == dangerous_auth
    return dangerous_auth


def _forbid_auth_store_io(monkeypatch):
    """Instrument every explicit-target persistence boundary."""
    boundary_hits = []

    @contextmanager
    def fail_lock(*args, **kwargs):
        boundary_hits.append(("lock", args, kwargs))
        raise AssertionError("explicit-target persistence reached the auth-store lock")
        yield  # pragma: no cover

    def fail_load(*args, **kwargs):
        boundary_hits.append(("load", args, kwargs))
        raise AssertionError("explicit-target persistence reached auth-store load")

    def fail_save(*args, **kwargs):
        boundary_hits.append(("save", args, kwargs))
        raise AssertionError("explicit-target persistence reached auth-store save")

    monkeypatch.setattr(auth, "_auth_store_lock", fail_lock)
    monkeypatch.setattr(auth, "_load_auth_store", fail_load)
    monkeypatch.setattr(auth, "_save_auth_store", fail_save)
    return boundary_hits


def test_auth_file_path_refuses_real_store_after_home_is_patched(
    monkeypatch,
    tmp_path,
):
    dangerous_auth = _real_auth_path()
    monkeypatch.setenv("HOME", str(tmp_path / "patched-home"))
    monkeypatch.setenv("HERMES_HOME", str(dangerous_auth.parent))

    with pytest.raises(RuntimeError, match="HERMES_HOME"):
        auth._auth_file_path()


def test_auth_file_path_keeps_production_resolution_unchanged(
    monkeypatch,
    tmp_path,
):
    dangerous_auth = _real_auth_path()
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "patched-home"))
    monkeypatch.setenv("HERMES_HOME", str(dangerous_auth.parent))

    assert auth._auth_file_path().resolve(strict=False) == dangerous_auth


def test_global_loader_refuses_real_store_before_file_access(
    monkeypatch,
    tmp_path,
):
    dangerous_auth = _real_auth_path()
    monkeypatch.setenv("HOME", str(tmp_path / "patched-home"))
    monkeypatch.setenv(
        "HERMES_HOME",
        str(dangerous_auth.parent / "profiles" / "seatbelt-test"),
    )

    global_path = auth._global_auth_file_path()
    assert global_path is not None
    assert global_path.resolve(strict=False) == dangerous_auth

    original_exists = Path.exists

    def fail_dangerous_exists(path):
        if path.resolve(strict=False) == dangerous_auth:
            raise AssertionError("global loader reached exists() before its seat belt")
        return original_exists(path)

    def fail_load(path):
        raise AssertionError(f"global loader attempted auth-store access: {path}")

    monkeypatch.setattr(Path, "exists", fail_dangerous_exists)
    monkeypatch.setattr(auth, "_load_auth_store", fail_load)

    assert auth._load_global_auth_store() == {}


def test_explicit_target_writer_refuses_real_store_before_io(monkeypatch, tmp_path):
    dangerous_auth = _configure_dangerous_profile(monkeypatch, tmp_path)
    boundary_hits = _forbid_auth_store_io(monkeypatch)

    with pytest.raises(RuntimeError, match="Refusing.*auth store"):
        auth._persist_provider_state_to_store(
            "test-provider",
            {"access_token": "test-sentinel"},
            dangerous_auth,
            set_active=False,
        )

    assert boundary_hits == []


def test_explicit_target_writer_keeps_production_behavior(monkeypatch, tmp_path):
    dangerous_auth = _configure_dangerous_profile(monkeypatch, tmp_path)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    boundary_hits = []

    @contextmanager
    def tracking_lock(*, target_path):
        boundary_hits.append(("lock", target_path))
        yield

    def tracking_load(target_path):
        boundary_hits.append(("load", target_path))
        return {"version": 1, "providers": {}}

    def tracking_save(auth_store, *, target_path):
        boundary_hits.append(("save", target_path))
        return target_path

    monkeypatch.setattr(auth, "_auth_store_lock", tracking_lock)
    monkeypatch.setattr(auth, "_load_auth_store", tracking_load)
    monkeypatch.setattr(auth, "_save_auth_store", tracking_save)

    assert (
        auth._persist_provider_state_to_store(
            "test-provider",
            {"access_token": "test-sentinel"},
            dangerous_auth,
            set_active=False,
        )
        == dangerous_auth
    )
    assert [kind for kind, _path in boundary_hits] == ["lock", "load", "save"]


def test_direct_write_through_stops_before_real_store_io(monkeypatch, tmp_path):
    _configure_dangerous_profile(monkeypatch, tmp_path)
    boundary_hits = _forbid_auth_store_io(monkeypatch)

    auth._write_through_xai_oauth_to_global_root(
        {"tokens": {"access_token": "test-sentinel"}}
    )

    assert boundary_hits == []


def test_pool_write_through_stops_before_real_store_io(monkeypatch, tmp_path):
    _configure_dangerous_profile(monkeypatch, tmp_path)
    boundary_hits = _forbid_auth_store_io(monkeypatch)

    credential_pool._write_through_provider_state_to_global_root(
        "nous",
        {"access_token": "test-sentinel"},
    )

    assert boundary_hits == []
