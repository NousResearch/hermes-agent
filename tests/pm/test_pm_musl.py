"""A native musl userland resolves PM to musl-native runtime artifacts (#123682)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pm.lock import Lockfile
from pm.registry import all_packages, source_install_packages, walk

pytestmark = pytest.mark.platforms("linux")

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_musl_userland_gets_a_satisfiable_native_musl_closure(monkeypatch, tmp_path):
    from pm import store

    # /bin/sh's ELF interpreter is musl; the bootstrap Python (this glibc test
    # host's) says otherwise and must not win.
    fake_sh = tmp_path / "sh"
    fake_sh.write_bytes(b"\x7fELF" + b"\0" * 60 + b"/lib/ld-musl-x86_64.so.1\0")
    monkeypatch.setattr(store, "_native_linux_uses_musl",
                        lambda: store._elf_loader_is_musl(fake_sh), raising=False)
    monkeypatch.setattr(store, "_native_machine", lambda: "x86_64")
    monkeypatch.setattr(store, "_is_bionic_libc", lambda: False)

    target = store.current_target()
    assert target == "linux-x64-musl"

    lock = Lockfile(REPO_ROOT / "pm" / "lock.json")
    from pm.defaults import default_packages
    roots = source_install_packages(all_packages())
    roots += [n for n in default_packages(lock.names(), target=target, declined_names=frozenset())
              if n not in roots]
    closure = walk(roots)
    assert {"python", "uv", "node"} <= {package.name for package in closure}
    for package in closure:
        assert package.missing_reason(target) is None, package.name
        if lock.version(package.name):
            assert lock.artifacts(package.name, target), f"{package.name} has no {target} artifact"
    for name in ("python", "uv", "node"):
        assert "musl" in lock.artifacts(name, target)[0]["url"], name
