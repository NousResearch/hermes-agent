"""Unreadable package bytes must not trigger a reinstall (#140575)."""

import errno
import json
import os

import pytest

from pm.install import _entry_verified, ensure
from pm.package import InstallError, Package
from pm.store import Store, current_target, tree_digest
from tests.pm.test_pm_core import pm_env as pm_env
from tests.pm._fixtures import served as served


def _deny_scan(monkeypatch, blocked):
    original = os.scandir

    def normalized(path):
        return os.path.normcase(os.path.abspath(path)).removeprefix("\\\\?\\")

    def scan(path):
        if normalized(path) == normalized(blocked):
            raise PermissionError(errno.EACCES, "permission denied", str(blocked))
        return original(path)

    monkeypatch.setattr(os, "scandir", scan)


def test_digest_refuses_partial_tree(tmp_path, monkeypatch):
    blocked = tmp_path / "unreadable"
    blocked.mkdir()
    (blocked / "payload").write_bytes(b"package bytes")
    _deny_scan(monkeypatch, blocked)
    with pytest.raises(PermissionError):
        tree_digest(tmp_path)


@pytest.mark.parametrize("blocked_name", ["unreadable", "bin"])
def test_explicit_install_keeps_unverifiable_entry(pm_env, monkeypatch, blocked_name):
    _, runtime, _, _ = pm_env
    ensure("faketool", base_env={}, explicit=True)
    entry = next(runtime.glob("faketool-*"))
    blocked = entry / blocked_name
    blocked.mkdir(exist_ok=True)
    (blocked / "payload").write_bytes(b"preserve me")
    facts_path = runtime / "facts.json"
    facts = json.loads(facts_path.read_text())
    facts["packages"]["faketool"]["digest"] = tree_digest(entry)
    facts_path.write_text(json.dumps(facts))
    facts_before = facts_path.read_bytes()
    _deny_scan(monkeypatch, blocked)

    with pytest.raises(InstallError, match="cannot verify.*permission"):
        ensure("faketool", base_env={}, explicit=True)

    assert (blocked / "payload").read_bytes() == b"preserve me"
    assert (runtime / "facts.json").read_bytes() == facts_before
    assert not list(runtime.glob(".previous-*"))


def test_digest_preserves_empty_and_cache_contract(tmp_path, monkeypatch):
    empty = tree_digest(tmp_path)
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "runtime.pyc").write_bytes(b"runtime only")
    _deny_scan(monkeypatch, cache)
    assert tree_digest(tmp_path) == empty
    (tmp_path / "payload").write_bytes(b"immutable")
    assert tree_digest(tmp_path) != empty


@pytest.mark.parametrize("failure", [None, FileNotFoundError, OSError, PermissionError])
def test_verification_preserves_nonpermission_outcomes(tmp_path, failure):
    class Probe(Package):
        name = "probe"

        def verify(self, entry, target):
            if failure is not None:
                raise failure(errno.EIO, "probe refused", str(entry))
            return ""

    entry = tmp_path / "entry"
    entry.mkdir()
    fact = {"entry": "entry", "digest": tree_digest(entry)}
    if failure is PermissionError:
        with pytest.raises(InstallError, match="cannot verify.*permission"):
            _entry_verified(Probe(), fact, Store(tmp_path), current_target())
    else:
        assert _entry_verified(Probe(), fact, Store(tmp_path), current_target()) is (failure is None)
