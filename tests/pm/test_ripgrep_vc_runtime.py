"""ARM64 Windows ripgrep carries the VC++ runtime it links against.

The aarch64-pc-windows-msvc rg.exe imports VCRUNTIME140.dll (the x64 build
embeds the CRT). A fresh Windows has no copy, so staging failed with
``rg.exe --version exited 3221225781`` (0xC0000135, STATUS_DLL_NOT_FOUND).
"""

from __future__ import annotations

import pm.install
from pm.registry import get_package, walk
from pm.store import Store


def test_arm64_windows_ripgrep_stages_pythons_vc_runtime_beside_rg(tmp_path, monkeypatch):
    store = Store(tmp_path / "store")
    runtime = store.entry("python-1") / "vcruntime140.dll"
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"arm64 vc runtime")
    monkeypatch.setattr(pm.install, "_lockfile", lambda: None)
    monkeypatch.setattr(pm.install, "_installed_location",
                        lambda package, lockfile, target: ({"python": {"entry": "python-1"}}, store))

    staged = tmp_path / "scratch" / "tree"
    wrapped = staged / "ripgrep-15.2.0-aarch64-pc-windows-msvc"
    wrapped.mkdir(parents=True)
    (wrapped / "rg.exe").write_bytes(b"rg")

    get_package("ripgrep").stage(store, staged, "15.2.0", "win32-arm64")

    assert (staged / "rg.exe").is_file()
    assert (staged / "vcruntime140.dll").read_bytes() == runtime.read_bytes()


def test_python_installs_before_ripgrep_so_staging_can_read_it():
    order = [package.name for package in walk(["ripgrep"])]
    assert order.index("python") < order.index("ripgrep")
