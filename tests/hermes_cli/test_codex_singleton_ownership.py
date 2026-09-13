"""Synthetic real-store regressions for #87503 (no live OAuth requests)."""
import base64
import json
import os
from pathlib import Path

import pytest

from hermes_cli import auth


def _jwt(exp):
    payload = base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode()).decode().rstrip("=")
    return f"synthetic.{payload}.signature"


def _store(path, refresh, access=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "version": 1, "active_provider": "other", "providers": {"openai-codex": {"tokens": {
            "access_token": access or _jwt(1), "refresh_token": refresh}}},
    }), encoding="utf-8")


@pytest.fixture
def homes(tmp_path, monkeypatch):
    import hermes_constants

    root = tmp_path / "root"
    profile = root / "profiles" / "borrower"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "no-cli"))
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: root)
    monkeypatch.setattr(auth, "_global_auth_store_cache", None)
    return root / "auth.json", profile / "auth.json"


@pytest.mark.parametrize("owner", ["root", "profile"])
def test_refresh_writes_only_the_owner(homes, monkeypatch, owner):
    root, profile = homes
    _store(root, "synthetic-root")
    if owner == "profile":
        _store(profile, "synthetic-local")
    else:
        profile.write_text('{"providers": {"other": {"sentinel": true}}}', encoding="utf-8")
    source, untouched = (root, profile) if owner == "root" else (profile, root)
    before = untouched.read_bytes()
    calls = []

    def refresh(access, token, **kwargs):
        calls.append(token)
        return {"access_token": _jwt(4102444800), "refresh_token": "synthetic-rotated"}

    monkeypatch.setattr(auth, "refresh_codex_oauth_pure", refresh)
    assert auth.resolve_codex_runtime_credentials()["api_key"] == _jwt(4102444800)
    assert calls == ["synthetic-root" if owner == "root" else "synthetic-local"]
    assert untouched.read_bytes() == before
    if owner == "root":
        assert json.loads(source.read_text(encoding="utf-8"))["active_provider"] == "other"
    assert json.loads(source.read_text(encoding="utf-8"))["providers"]["openai-codex"]["tokens"] == {
        "access_token": _jwt(4102444800), "refresh_token": "synthetic-rotated"}


def _wait_for(path):
    import time

    deadline = time.monotonic() + 15
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"Borrower rendezvous timed out: {path.name}")
        time.sleep(0.01)


def _borrower(root, name, mode):
    """Fresh interpreter, real OS locks; only the OAuth endpoint is synthetic."""
    from contextlib import contextmanager
    import hermes_constants

    root = Path(root)
    hermes_constants.get_default_hermes_root = lambda: root
    auth._global_auth_store_cache = None
    real_read, real_lock = auth._read_codex_tokens, auth._auth_store_lock
    other = "b" if name == "a" else "a"

    def read(**kwargs):
        result = real_read(**kwargs)
        if kwargs.get("_lock", True):
            (root / f"{name}.read").touch()
            _wait_for(root / f"{other}.read")
        return result

    @contextmanager
    def lock(*args, **kwargs):
        if kwargs.get("target_path") == root / "auth.json":
            (root / f"{name}.locking").touch()
        with real_lock(*args, **kwargs):
            yield

    def refresh(access, token, **kwargs):
        _wait_for(root / f"{other}.locking")
        # Exclusive creation models consumption of a single-use refresh grant.
        # Any second endpoint call fails, even if it submits the newly rotated token.
        with (root / "consumed").open("x", encoding="utf-8") as handle:
            handle.write("one endpoint call")
        assert token == "synthetic-root"
        return {"access_token": _jwt(4102444800), "refresh_token": "synthetic-rotated"}

    auth._read_codex_tokens = read
    auth._auth_store_lock = lock
    auth.refresh_codex_oauth_pure = refresh
    assert auth.fcntl is not None or auth.msvcrt is not None, "Real kernel lock required"
    if __import__("os").name == "nt":
        assert auth.msvcrt is not None and auth.fcntl is None
    assert auth.resolve_codex_runtime_credentials(force_refresh=mode != "expiry")["api_key"] == _jwt(4102444800)


@pytest.mark.parametrize("mode", ["expiry", "force", "refresh-only"])
def test_cross_process_single_refresh(homes, mode):
    import os
    import subprocess
    import sys

    root_auth, profile_auth = homes
    root = root_auth.parent
    _store(root_auth, "synthetic-root", _jwt(4102444800) if mode == "refresh-only" else _jwt(1))
    worker_code = """
import os, pathlib, runpy, sys
root = pathlib.Path(sys.argv[2]).resolve()
violations = []
def guard(event, args):
    denied = event in {'socket.connect', 'socket.getaddrinfo', 'socket.sendto'}
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = pathlib.Path(os.fsdecode(args[0])).resolve()
        denied |= path.name in {'auth.json', 'auth.lock'} and not path.is_relative_to(root)
    if denied:
        violations.append(event)
        raise RuntimeError('Worker isolation guard')
sys.addaudithook(guard)
module = runpy.run_path(sys.argv[1])
module['_borrower'](*sys.argv[2:])
assert not violations, violations
"""
    workers = []
    try:
        for name in ("a", "b"):
            home = root / "profiles" / name
            home.mkdir()
            env = dict(os.environ, HERMES_HOME=str(home), CODEX_HOME=str(root / "no-cli"),
                       PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=os.pathsep.join(sys.path))
            workers.append(subprocess.Popen(
                [sys.executable, "-B", "-c", worker_code, __file__, str(root), name, mode],
                env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                encoding="utf-8"))
        results = [worker.communicate(timeout=35) for worker in workers]
        for worker, (stdout, stderr) in zip(workers, results):
            assert worker.returncode == 0, stdout + stderr
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
            worker.communicate(timeout=5)
    assert (root / "consumed").read_text(encoding="utf-8") == "one endpoint call"
    assert not profile_auth.exists()
    assert all(not (root / "profiles" / name / "auth.json").exists() for name in ("a", "b"))
    assert json.loads(root_auth.read_text(encoding="utf-8"))["providers"]["openai-codex"]["tokens"] == {
        "access_token": _jwt(4102444800), "refresh_token": "synthetic-rotated"}


@pytest.mark.parametrize("failure", ["transient", "rejected-with-cli", "persistence"])
def test_failed_refresh_does_not_fork(homes, monkeypatch, failure):
    root, profile = homes
    _store(root, "synthetic-root")
    before = root.read_bytes()

    def refresh(*args, **kwargs):
        if failure == "persistence":
            return {"access_token": _jwt(4102444800), "refresh_token": "synthetic-rotated"}
        raise auth.AuthError("Synthetic refresh failure", provider="openai-codex",
                             code="invalid_grant" if failure == "rejected-with-cli" else "temporary",
                             relogin_required=failure == "rejected-with-cli")

    monkeypatch.setattr(auth, "refresh_codex_oauth_pure", refresh)
    monkeypatch.setattr(auth, "_import_codex_cli_tokens", lambda: {
        "access_token": _jwt(4102444800), "refresh_token": "synthetic-external"})
    if failure == "persistence":
        def save(*args, **kwargs):
            raise OSError("Synthetic persistence failure")
        monkeypatch.setattr(auth, "_save_auth_store", save)
    with pytest.raises((auth.AuthError, OSError)):
        auth.resolve_codex_runtime_credentials()
    assert not profile.exists()
    assert root.read_bytes() == before




@pytest.mark.windows_only
@pytest.mark.skipif(os.name != "nt", reason="Native Windows kernel locking")
@pytest.mark.parametrize("mode", ["expiry", "force", "refresh-only"])
def test_windows_cross_process_single_refresh(homes, mode):
    # The Windows CI lane selects both files and tests by this marker.
    test_cross_process_single_refresh(homes, mode)


def test_strict_refresh_respects_pytest_seat_belt(tmp_path, monkeypatch):
    # Entirely synthetic HOME: exercise the seat belt without touching user auth.
    monkeypatch.setenv("HOME", str(tmp_path))
    target = tmp_path / ".hermes" / "auth.json"
    monkeypatch.setattr(auth, "_global_auth_file_path", lambda: target)
    with pytest.raises(RuntimeError, match="Refusing Codex refresh"):
        auth._write_through_codex_tokens_to_global_root({}, {}, None, strict=True)
    assert not target.exists()
