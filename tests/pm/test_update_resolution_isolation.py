"""One upstream resolution outage must not discard independently resolved updates."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pm import cli
from pm.lock import Lockfile
from pm.update import Resolved


def update(*names, check=False, uv=False, npm=False):
    return cli.cmd_update(SimpleNamespace(names=list(names), target=None, termux=False,
                                          check=check, uv=uv, npm=npm))


@pytest.fixture
def update_fixture(tmp_path, monkeypatch):
    lock = Lockfile(tmp_path / "lock.json")
    for name in ("first", "broken", "last"):
        lock.set_pin(name, "1.0", {"fixture": {"url": name + "-old", "sha256": "old"}})
    lock.save()
    packages = {name: SimpleNamespace(
        name=name, internal=False, version_style="semver",
        missing_reason=lambda target: None,
        fetch_urls=lambda version, target, name=name: [name + "-" + version],
        known_sha256=lambda version, url: "fixture-hash",
    ) for name in lock.names()}
    monkeypatch.setattr(cli, "_lockfile", lambda: lock)
    monkeypatch.setattr(cli, "get_package", packages.__getitem__)
    monkeypatch.setattr(cli, "ALL_TARGETS", ["fixture"])
    def resolve(package, *args, **kwargs):
        if package.name == "broken":
            raise OSError("fixture index unavailable")
        return Resolved(package.name, "1.0", "semver", "2.0", per_target={"fixture": "2.0"})
    monkeypatch.setattr(cli, "resolve_package", resolve)
    install = Mock(return_value=0)
    sync = Mock(return_value=True)
    monkeypatch.setattr(cli, "_install_names", install)
    monkeypatch.setattr(cli, "_sync_venv_step", sync)
    return lock, install, sync


def test_apply_keeps_independently_resolved_updates(update_fixture, capsys):
    lock, install, sync = update_fixture
    before = lock.pinned_artifacts("broken")
    assert update("first", "broken", "last") == 1
    reopened = Lockfile(lock.path)
    assert reopened.version("first") == reopened.version("last") == "2.0"
    assert reopened.version("broken") == "1.0"
    assert reopened.pinned_artifacts("broken") == before
    install.assert_called_once_with(["first", "last"])
    sync.assert_called_once_with()
    output = capsys.readouterr().out
    assert "resolution failed for broken" in output
    assert "no changes applied" not in output


def test_check_does_not_apply_successful_siblings(update_fixture):
    lock, install, sync = update_fixture
    before = lock.path.read_bytes()
    assert update("first", "broken", "last", check=True) == 1
    assert lock.path.read_bytes() == before
    install.assert_not_called()
    sync.assert_not_called()


def test_all_resolution_failures_do_not_write_or_install(update_fixture):
    lock, install, sync = update_fixture
    before = lock.path.read_bytes()
    assert update("broken") == 1
    assert lock.path.read_bytes() == before
    install.assert_not_called()
    sync.assert_not_called()


def test_no_resolution_failure_keeps_success_exit(update_fixture):
    lock, install, sync = update_fixture
    assert update("first", "last") == 0
    assert Lockfile(lock.path).version("first") == "2.0"


def test_resolution_failure_still_skips_optional_lock_refresh(update_fixture, monkeypatch):
    lock, _, _ = update_fixture
    uv = Mock(side_effect=AssertionError("optional refresh should stay skipped on failure"))
    npm = Mock(side_effect=AssertionError("optional refresh should stay skipped on failure"))
    monkeypatch.setattr(cli, "_refresh_uv_lock", uv)
    monkeypatch.setattr(cli, "_refresh_npm_lock", npm)
    assert update("first", "broken", uv=True, npm=True) == 1
    assert Lockfile(lock.path).version("first") == "2.0"
    uv.assert_not_called()
    npm.assert_not_called()
