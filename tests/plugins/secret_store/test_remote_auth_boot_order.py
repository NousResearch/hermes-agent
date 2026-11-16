"""Bounded real auth-lock/event inversion, no patched lock implementation."""
import json
import threading
import traceback
import sys

import pytest

pytest_plugins = ["tests.plugins.secret_store.test_remote_secret_store"]


def test_boot_cleanup_vs_auth_transaction(plane, monkeypatch):
    from hermes_secret_routing import routed_provider
    from plugins.secret_store.remote import importer
    from hermes_cli import auth

    provider = routed_provider()
    home = plane.home
    monkeypatch.setattr(provider, "_ensure_poller", lambda: None)
    (home / ".env").write_text("LOCAL_KEY=local-marker\n", encoding="utf-8")
    path = home / "auth.json"
    path.write_text(json.dumps({"providers": {"example": {"key": "auth-marker"}}}), encoding="utf-8")
    cleanup, release, auth_held, waiting = (threading.Event() for _ in range(4))
    original = importer.finish
    first = True
    errors, results = [], []

    def finish(*args):
        nonlocal first
        if first:
            first = False
            cleanup.set()
            assert release.wait(10)
        return original(*args)

    monkeypatch.setattr(importer, "finish", finish)

    def boot():
        try:
            provider.boot(home)
        except BaseException as exc:
            errors.append(exc)

    def transaction():
        try:
            from hermes_cli.auth_codex import _load_auth_store_maybe_locked
            results.append(_load_auth_store_maybe_locked(True))
        except BaseException as exc:
            errors.append(exc)

    kernel_lock = auth._kernel_lock
    captured = False

    def observed_kernel_lock(file, acquire):
        nonlocal captured
        try:
            result = kernel_lock(file, acquire)
            if acquire and threading.current_thread().name == "review-auth-reader":
                auth_held.set()
            return result
        except BlockingIOError:
            if not captured and threading.current_thread().name == "review-boot-owner":
                captured = True
                for name, thread in (("BOOT_OWNER", owner), ("AUTH_READER", other)):
                    print(name, "".join(traceback.format_stack(sys._current_frames()[thread.ident])))
            raise

    monkeypatch.setattr(auth, "_kernel_lock", observed_kernel_lock)

    owner = threading.Thread(target=boot, daemon=True, name="review-boot-owner")
    owner.start()
    assert cleanup.wait(10)
    event = provider._boot_events[provider._key(home)]
    wait = event.wait

    def observed_wait(*args, **kwargs):
        waiting.set()
        return wait(*args, **kwargs)

    monkeypatch.setattr(event, "wait", observed_wait)
    other = threading.Thread(target=transaction, daemon=True, name="review-auth-reader")
    other.start()
    try:
        assert waiting.wait(10)
        assert not auth_held.is_set()
    finally:
        release.set()
        owner.join(45)
        other.join(45)
    print("REAL_AUTH_EVENT_ERRORS", [(type(exc).__name__, str(exc)) for exc in errors])
    assert not owner.is_alive() and not other.is_alive()
    assert not errors, [(type(exc).__name__, str(exc)) for exc in errors]
    assert results


def test_auth_reader_after_boot_control(plane, monkeypatch):
    from hermes_secret_routing import routed_provider
    from hermes_cli.auth_codex import _load_auth_store_maybe_locked
    provider = routed_provider()
    monkeypatch.setattr(provider, "_ensure_poller", lambda: None)
    (plane.home / "auth.json").write_text(
        json.dumps({"providers": {"example": {"key": "auth-marker"}}}), encoding="utf-8")
    provider.boot(plane.home)
    assert _load_auth_store_maybe_locked(True)["providers"]["example"] == {"key": "auth-marker"}
