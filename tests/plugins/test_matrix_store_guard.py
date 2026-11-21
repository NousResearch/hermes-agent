"""Retained fork store locking: real contention and credential refusal cleanup."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace


_spec = importlib.util.spec_from_file_location(
    "matrix_store_guard_under_test",
    Path(__file__).parents[2] / "plugins/platforms/matrix/store_guard.py",
)
assert _spec is not None and _spec.loader is not None
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)


def adapter(directory, *, credential_allowed=True):
    errors = []
    releases = []
    return SimpleNamespace(
        _resolve_store_dir=lambda: directory,
        _access_token="fixture-token",
        _homeserver="https://example.invalid",
        _user_id="@fixture:example.invalid",
        _set_fatal_error=lambda *args, **kwargs: errors.append((args, kwargs)),
        _acquire_platform_lock=lambda *args: credential_allowed,
        _release_platform_lock=lambda: releases.append(True),
        errors=errors,
        releases=releases,
    )


def test_store_contention_and_release(tmp_path):
    first, second = adapter(tmp_path), adapter(tmp_path)
    assert guard.claim_store(first)
    try:
        assert not guard.claim_store(second)
        assert second.errors[0][0][0] == "matrix_store_busy"
        assert second.errors[0][1]["retryable"] is True
    finally:
        guard.release_store(first)
    assert guard.claim_store(second)
    guard.release_store(second)
    guard.release_store(second)
    assert second.releases == [True]


def test_credential_refusal_releases_store(tmp_path):
    refused = adapter(tmp_path, credential_allowed=False)
    assert not guard.claim_store(refused)
    assert refused._matrix_store_lock is None
    successor = adapter(tmp_path)
    assert guard.claim_store(successor)
    guard.release_store(successor)
