"""Updating the bridge must install its new dependency pin before pairing."""

from types import SimpleNamespace

import pytest

_BASE_PACKAGE = '{"name": "bridge"}\n'
_BASE_LOCK = "lockfileVersion: '9.0'\n"


def _write_manifests(bridge_dir, *, package_json=_BASE_PACKAGE, lockfile=_BASE_LOCK):
    (bridge_dir / "package.json").write_text(package_json, encoding="utf-8")
    if lockfile is not None:
        (bridge_dir / "pnpm-lock.yaml").write_text(lockfile, encoding="utf-8")


def _stamp(bridge_dir, kind, *, node_modules=True):
    """``match``: the hash of the manifests on disk; ``stale``: a pre-drift hash."""
    if not node_modules:
        return
    from gateway.platforms.whatsapp_common import bridge_manifest_hash

    nm = bridge_dir / "node_modules"
    nm.mkdir(exist_ok=True)
    value = bridge_manifest_hash(bridge_dir) if kind == "match" else "0" * 16
    (nm / ".hermes-pkg-hash").write_text(value, encoding="utf-8")


@pytest.mark.parametrize("package_json,lockfile,stamp,node_modules,refresh", [
    # Fresh stamp over the current manifests: skip the install.
    (_BASE_PACKAGE, _BASE_LOCK, "match", True, False),
    # package.json drift (a bumped bridge build): reinstall.
    ('{"name": "bridge", "version": "2"}\n', _BASE_LOCK, "stale", True, True),
    # pnpm-lock.yaml drift (a bumped dependency pin): reinstall.
    (_BASE_PACKAGE, "lockfileVersion: '9.0'\n# bumped\n", "stale", True, True),
    # A missing lockfile is never a current install.
    (_BASE_PACKAGE, None, "match", True, True),
    # A stale stamp (an interrupted/older install): reinstall.
    (_BASE_PACKAGE, _BASE_LOCK, "stale", True, True),
    # No node_modules at all: install.
    (_BASE_PACKAGE, _BASE_LOCK, "match", False, True),
])
def test_setup_refreshes_stale_dependencies_but_keeps_matching_install(
        tmp_path, monkeypatch, package_json, lockfile, stamp, node_modules, refresh):
    from hermes_cli import main_platform_setup as setup
    import hermes_constants

    _write_manifests(tmp_path, package_json=package_json, lockfile=lockfile)
    _stamp(tmp_path, stamp, node_modules=node_modules)
    calls = []

    def install(command, **kwargs):
        calls.append(command)
        assert kwargs["cwd"] == str(tmp_path)
        assert kwargs["timeout"] == 450
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv("WHATSAPP_PNPM_INSTALL_TIMEOUT", "450")
    monkeypatch.setattr(hermes_constants, "find_node_executable", lambda name: "isolated-pnpm")
    monkeypatch.setattr(hermes_constants, "with_hermes_node_path", dict)
    monkeypatch.setattr(setup.subprocess, "run", install)
    assert setup._whatsapp_install_bridge(tmp_path)
    assert bool(calls) is refresh
