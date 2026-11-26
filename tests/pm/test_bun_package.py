"""Behavioral contracts for the optional PM-managed Bun package (Linux x64/arm64)."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from pm import Lockfile, get_package, paths, registry, store
from pm.store import ALL_TARGETS


def test_bun_lockfile_artifacts_match_supported_linux_targets():
    package = get_package("bun")
    lock = Lockfile(paths.lockfile_path())
    version = lock.version("bun")

    assert package.optional is True
    assert package.default is False

    for target in ALL_TARGETS:
        if target in ("linux-x64", "linux-arm64"):
            assert package.missing_reason(target) is None, target
            artifacts = lock.artifacts("bun", target)
            assert [a["url"] for a in artifacts] == package.fetch_urls(version, target)
            assert all(len(a["sha256"]) == 64 for a in artifacts)
        else:
            assert package.missing_reason(target) is not None, target
            assert lock.artifacts("bun", target) == []


@pytest.mark.parametrize("target", ALL_TARGETS)
def test_bun_is_never_in_default_source_install_closure(monkeypatch, target):
    monkeypatch.setattr(store, "current_target", lambda: target)
    names = Lockfile(paths.lockfile_path()).names()
    assert "bun" not in registry.source_install_packages(names)


def test_bun_is_not_copied_to_user_facing_shims():
    from pm.user_facing import COPIED

    assert "bun" not in COPIED
    assert "node" in COPIED
    assert "npm" in COPIED


@pytest.mark.parametrize("launcher", ["bun", "bunx"])
def test_mcp_managed_launcher_resolves_pm_bun_on_supported_targets(monkeypatch, tmp_path, launcher):
    import pm
    from tools import mcp_tool_config

    bin_dir = tmp_path / "bun-bin"
    bin_dir.mkdir()
    exe = bin_dir / (f"{launcher}.exe" if os.name == "nt" else launcher)
    exe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    exe.chmod(0o755)

    ensured: list[str] = []
    monkeypatch.setattr(pm, "current_target", lambda: "linux-x64")
    monkeypatch.setattr(pm, "ensure", lambda pkg, **kw: ensured.append(pkg))
    monkeypatch.setattr(pm, "installed_package", lambda pkg: SimpleNamespace(binary=exe) if pkg == "bun" else None)

    result = mcp_tool_config._managed_launcher(launcher)
    assert result is not None
    resolved, dirs = result
    assert Path(resolved).resolve() == exe.resolve()
    assert dirs == [str(bin_dir)]
    assert ensured == ["bun"]


def test_mcp_managed_launcher_returns_none_for_bun_on_unsupported_targets(monkeypatch):
    import pm
    from tools import mcp_tool_config

    monkeypatch.setattr(pm, "current_target", lambda: "darwin-arm64")
    assert mcp_tool_config._managed_launcher("bun") is None
    assert mcp_tool_config._managed_launcher("bunx") is None
