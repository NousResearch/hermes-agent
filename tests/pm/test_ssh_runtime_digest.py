"""SSH liveness markers are runtime state, not sealed Python bytes (#122827)."""

import os

import pytest

from pm.store import tree_digest


@pytest.mark.parametrize("purelib", ["Lib/site-packages", "lib/python3.14/site-packages"])
@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_runtime_marker_preserves_digest_but_not_neighbor_changes(tmp_path, monkeypatch, purelib, newline):
    import shutil
    from types import SimpleNamespace

    from hermes_cli import web_server
    from pm import paths, registry
    from pm.cli import cmd_doctor
    from pm.install import _entry_verified
    from pm.lock import Facts, Lockfile
    from pm.packages import BinaryPackage
    from pm.store import Store, current_target

    class FixturePython(BinaryPackage):
        name = "fixture-python"
        probe_version = False
        binary_rel = {"win32": "bin/python", "posix": "bin/python"}

    runtime = FixturePython()
    store = Store(tmp_path / "store")
    entry = store.entry("python")
    (entry / "bin").mkdir(parents=True)
    (entry / "bin/python").write_bytes(b"fixture interpreter, never executed")
    site = entry / purelib
    site.mkdir(parents=True)
    package = site / "installed.py"
    package.write_bytes(b"original package")
    sealed = tree_digest(entry)
    monkeypatch.setattr(paths, "store_root", lambda: store.root)
    monkeypatch.setattr(paths, "lockfile_path", lambda: tmp_path / "lock.json")
    monkeypatch.setattr(registry, "_packages", {runtime.name: runtime})
    lock = Lockfile(paths.lockfile_path())
    lock.set_pin(runtime.name, "1", {"any": {"url": "https://invalid.example/never-fetched", "sha256": "a" * 64}})
    lock.save()
    facts = Facts(paths.facts_path())
    facts.record(runtime.name, "1", entry.name, {}, store.root,
                 target=current_target(), artifacts=["a" * 64], digest=sealed)
    fact = facts.get(runtime.name)
    recorded_facts = facts.path.read_bytes()
    monkeypatch.setattr(web_server, "sysconfig", SimpleNamespace(get_paths=lambda: {"purelib": str(site)}))
    for name in ("_SSH_OWNER_NONCE", "_SSH_RUNTIME_PURELIB", "_SSH_RUNTIME_MARKER"):
        monkeypatch.setattr(web_server, name, None)
    web_server._apply_ssh_owner_nonce("0123456789abcdef")
    marker = site / ".hermes-ssh-runtime-0123456789abcdef"
    assert marker.read_text() == f"pid={os.getpid()}\n"
    marker.write_bytes(f"pid={os.getpid()}".encode() + newline)
    assert web_server._ssh_runtime_intact()
    assert tree_digest(entry) == sealed
    assert cmd_doctor(None) == 0
    assert _entry_verified(runtime, fact, store, current_target())
    package.write_bytes(b"changed package")
    assert tree_digest(entry) != sealed
    assert cmd_doctor(None) == 1
    assert not _entry_verified(runtime, fact, store, current_target())
    package.write_bytes(b"original package")
    assert tree_digest(entry) == sealed
    assert facts.path.read_bytes() == recorded_facts
    shutil.rmtree(site)
    site.mkdir()
    assert not web_server._ssh_runtime_intact()


@pytest.mark.parametrize("relative,payload", [
    ("Lib/site-packages/.hermes-ssh-runtime-invalid", b"pid=123\n"),
    ("Lib/site-packages/.hermes-ssh-runtime-0123456789abcdef.py", b"pid=123\n"),
    ("Lib/site-packages/.hermes-ssh-runtime-0123456789abcdef", b"arbitrary bytes\n"),
    ("Lib/site-packages/.hermes-ssh-runtime-0123456789abcdef", b"pid=123\nextra"),
    ("Lib/site-packages/.hermes-ssh-runtime-0123456789abcdef", b"pid=0\n"),
    ("Lib/site-packages/.hermes-ssh-runtime-0123456789abcdef", b"pid=" + b"1" * 100 + b"\n"),
    ("Lib/site-packages/pkg/.hermes-ssh-runtime-0123456789abcdef", b"pid=123\n"),
    ("other/site-packages/.hermes-ssh-runtime-0123456789abcdef", b"pid=123\n"),
    (".hermes-ssh-runtime-0123456789abcdef", b"pid=123\n"),
])
def test_marker_lookalikes_remain_integrity_bound(tmp_path, relative, payload):
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    sealed = tree_digest(tmp_path)
    path.write_bytes(payload)
    assert tree_digest(tmp_path) != sealed
