"""HERMES_WRITE_SAFE_ROOT must resolve per-profile, never cross-profile - and
must keep the container-wide floor when a profile does not set its own.

Under gateway multiplexing one process serves many profiles off one shared
os.environ; the write-safety allowlist used to be read with a plain os.getenv,
so profile A's roots gated profile B's writes (proven live: felix saw
/home/jonas). It now resolves by layer: a profile's own scoped value wins; a
scope miss falls back to the container-wide floor (Dockerfile ENV /opt/data),
never to empty (which would fail OPEN) and never to another profile's value.
"""
from __future__ import annotations

import pytest

import agent.secret_scope as ss
import agent.file_safety as fs


@pytest.fixture(autouse=True)
def _reset_scope():
    ss.set_secret_scope(None)
    ss.set_multiplex_active(False)
    try:
        yield
    finally:
        ss.set_secret_scope(None)
        ss.set_multiplex_active(False)


def test_scope_hit_wins_over_os_environ(monkeypatch):
    # os.environ carries profile B's value (loaded last); scope binds profile A.
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", "/home/jonas")
    ss.set_multiplex_active(True)
    ss.set_secret_scope({"HERMES_WRITE_SAFE_ROOT": "/home/felix"})
    roots = fs.get_safe_write_roots()
    assert any(r.endswith("/home/felix") for r in roots), roots
    assert not any(r.endswith("/home/jonas") for r in roots), roots


def test_scope_miss_keeps_container_floor_not_allow_all(monkeypatch):
    # Profile bound a scope but did NOT set its own root: must keep the
    # container-wide floor (/opt/data), NOT fail open to an empty allowlist.
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", "/opt/data")
    ss.set_multiplex_active(True)
    ss.set_secret_scope({"OPENAI_API_KEY": "x"})  # no WRITE_SAFE_ROOT
    roots = fs.get_safe_write_roots()
    assert any(r.endswith("/opt/data") for r in roots), roots
    # And the floor actually confines: a path outside it is denied.
    assert fs.is_write_denied("/tmp/evil") is True
    assert fs.is_write_denied("/opt/data/profiles/x/notes.txt") is False


def test_explicit_empty_scoped_value_keeps_container_floor(monkeypatch):
    # BLOCKER regression: a multiplexed profile whose .env sets an EXPLICIT empty
    # HERMES_WRITE_SAFE_ROOT= is retained by load_env_file and reaches the scope.
    # It must NOT erase the container-wide floor (that would fail OPEN, allow-all);
    # the empty override falls through to the /opt/data floor, same as a miss.
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", "/opt/data")
    ss.set_multiplex_active(True)
    ss.set_secret_scope({"HERMES_WRITE_SAFE_ROOT": ""})  # explicit empty override
    roots = fs.get_safe_write_roots()
    assert any(r.endswith("/opt/data") for r in roots), roots
    # The floor still confines: a path outside it stays denied.
    assert fs.is_write_denied("/tmp/evil") is True
    assert fs.is_write_denied("/opt/data/profiles/x/notes.txt") is False


def test_unscoped_multiplex_keeps_container_floor(monkeypatch):
    # No scope + multiplex active: still honor the container floor, never crash,
    # never allow-all.
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", "/opt/data")
    ss.set_multiplex_active(True)
    ss.set_secret_scope(None)
    roots = fs.get_safe_write_roots()  # must not raise
    assert any(r.endswith("/opt/data") for r in roots), roots
    assert fs.is_write_denied("/tmp/evil") is True


def test_single_profile_unscoped_reads_os_environ(monkeypatch):
    # Multiplex OFF, no scope: behaves exactly as before (os.environ value).
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", "/srv/data")
    ss.set_multiplex_active(False)
    ss.set_secret_scope(None)
    roots = fs.get_safe_write_roots()
    assert any(r.endswith("/srv/data") for r in roots), roots


def test_truly_unset_is_permissive(monkeypatch):
    # No var anywhere = the historical unset baseline: no safe-root restriction.
    monkeypatch.delenv("HERMES_WRITE_SAFE_ROOT", raising=False)
    ss.set_multiplex_active(False)
    ss.set_secret_scope(None)
    assert fs.get_safe_write_roots() == set()
