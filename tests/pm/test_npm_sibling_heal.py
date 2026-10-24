"""A managed npm install missing npx is not healthy; a complete pair is kept.

Discovery does not install, and these tests never write the real Hermes home.
"""

import json
from pathlib import Path

import pytest

from pm import paths
from pm.install import _entry_current, _installed_location, ensure
from pm.lock import Facts, Lockfile
from pm.registry import get_package
from pm.store import Store, current_target, tree_digest

_REAL_HERMES_HOME = Path.home() / ".hermes"


def _pin(name: str, version: str, target: str, entry: Path, sha: str) -> None:
    package = get_package(name)
    lock = Lockfile(paths.lockfile_path())
    lock.set_pin(
        name,
        version,
        {target: {"url": f"https://example.invalid/{name}", "sha256": sha}},
    )
    lock.save()
    Facts(paths.facts_path()).record(
        name,
        version,
        entry.name,
        package.env(entry, target),
        paths.store_root(),
        target=target,
        artifacts=[sha],
        digest=tree_digest(entry),
    )


def _script(path: Path, body: str = "#!/bin/sh\necho ok\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def managed_npm(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(home / "tools"))
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    monkeypatch.setattr(paths, "lockfile_path", lambda: tmp_path / "lock.json")
    target = current_target()
    store = paths.store_root()
    assert store.is_relative_to(home)
    assert not store.is_relative_to(_REAL_HERMES_HOME)

    node_pkg = get_package("node")
    node_entry = store / node_pkg.store_entry("1.2.3", target)
    node_bin = node_pkg.binary(node_entry, target)
    assert node_bin is not None
    _script(node_bin)
    _pin("node", "1.2.3", target, node_entry, "a" * 64)

    npm_pkg = get_package("npm")
    npm_entry = store / npm_pkg.store_entry("9.9.9", target)
    npm_bin = npm_pkg.binary(npm_entry, target)
    assert npm_bin is not None
    _script(npm_bin, "#!/bin/sh\necho npm\n")
    sibling = npm_pkg.companion(npm_entry, target)
    assert sibling is not None
    _script(sibling, "#!/bin/sh\necho npx\n")
    _pin("npm", "9.9.9", target, npm_entry, "b" * 64)
    return {
        "home": home,
        "target": target,
        "node_bin": node_bin,
        "npm_bin": npm_bin,
        "npx_bin": sibling,
        "npm_entry": npm_entry,
    }


def test_npm_companion_is_the_npx_sibling():
    package = get_package("npm")
    entry = Path("entry")
    assert package.companion(entry, "linux-x64") == entry / "bin" / "npx"
    assert package.companion(entry, "darwin-arm64") == entry / "bin" / "npx"
    assert package.companion(entry, "win32-arm64") == entry / "npx.cmd"
    assert get_package("node").companion(entry, "linux-x64") is None


def test_npm_verify_rejects_a_missing_npx_sibling(tmp_path):
    package = get_package("npm")
    entry = tmp_path / "npm-entry"
    binary = package.binary(entry, "linux-x64")
    assert binary is not None
    _script(binary)
    assert "npx" in package.verify(entry, "linux-x64")
    sibling = package.companion(entry, "linux-x64")
    assert sibling is not None
    _script(sibling)
    assert package.verify(entry, "linux-x64") == ""


@pytest.mark.platforms("posix")
def test_missing_npx_is_repaired_without_deleting_a_healthy_install(
    managed_npm, monkeypatch
):
    target = managed_npm["target"]
    node_bytes = managed_npm["node_bin"].read_bytes()
    npm_bytes = managed_npm["npm_bin"].read_bytes()
    package = get_package("npm")
    lock = Lockfile(paths.lockfile_path())
    facts = Facts(paths.facts_path())
    store = Store(paths.store_root())
    entry = managed_npm["npm_entry"]
    pin = json.dumps({"target": target, "sha256": ["b" * 64]})
    assert _entry_current(
        package,
        lock,
        facts,
        store,
        entry,
        facts.get("npm"),
        "9.9.9",
        pin,
        target,
    )

    installs: list[str] = []

    def forbid(package, *args, **kwargs):
        installs.append(package.name)
        raise AssertionError(f"healthy install must not be replaced: {package.name}")

    monkeypatch.setattr("pm.install._install", forbid)
    ensure("npm", explicit=True)
    assert installs == []
    assert managed_npm["node_bin"].read_bytes() == node_bytes
    assert managed_npm["npm_bin"].read_bytes() == npm_bytes
    assert managed_npm["npx_bin"].is_file()

    managed_npm["npx_bin"].unlink()
    assert _installed_location(package, lock, target) is None
    assert _installed_location(get_package("node"), lock, target) is not None
    assert not _entry_current(
        package,
        lock,
        facts,
        store,
        entry,
        facts.get("npm"),
        "9.9.9",
        pin,
        target,
    )

    def restore(package, lockfile, facts, store, target, **kwargs):
        installs.append(package.name)
        assert package.name == "npm"
        sibling = package.companion(entry, target)
        assert sibling is not None
        _script(sibling, "#!/bin/sh\necho restored\n")
        return entry

    monkeypatch.setattr("pm.install._install", restore)
    ensure("npm", explicit=True)
    assert installs == ["npm"]
    assert managed_npm["node_bin"].read_bytes() == node_bytes
    assert managed_npm["npm_bin"].read_bytes() == npm_bytes
    assert managed_npm["npx_bin"].read_text(encoding="utf-8").startswith("#!/bin/sh")
    assert paths.store_root().is_relative_to(managed_npm["home"])
    assert not paths.store_root().is_relative_to(_REAL_HERMES_HOME)
