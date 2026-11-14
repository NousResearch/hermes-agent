"""A temp HERMES_HOME whose ``tools`` symlinks to the owner's store must not
publish a launcher baking the temp path (deleted with the temp dir -> rc126)."""
import os
import shutil

import pytest
from hermes_cli import _launchers

from tests.hermes_cli.test_hermes_home_shared_launchers import (
    _isolate,
    _make_home,
    _make_repo,
    _publish,
)

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX symlinked store")


def _launcher_pythons(repo):
    local = repo / ".hermes" / "bin"
    return {name: _launchers._launcher_python(local / name) for name in _launchers.ENTRY_POINTS}


@pytest.mark.platforms("posix")
def test_symlinked_temp_store_does_not_bake_temp_path(tmp_path, monkeypatch):
    owner_home, _ = _make_home(tmp_path, "owner")
    repo = _make_repo(tmp_path)
    _isolate(tmp_path, monkeypatch, owner_home)
    before = _publish(repo)

    temp_home = tmp_path / "sandbox"
    temp_home.mkdir()
    (temp_home / "tools").symlink_to(owner_home / "tools")
    _isolate(tmp_path, monkeypatch, temp_home)
    _launchers.ensure_install_launchers(repo, repo / ".hermes" / "bin")
    shutil.rmtree(temp_home)

    for name, python in _launcher_pythons(repo).items():
        assert python is not None
        assert temp_home not in python.parents
        assert python.is_file()
        assert (repo / ".hermes" / "bin" / name).read_bytes() == before[name]


@pytest.mark.platforms("posix")
def test_real_temp_store_leaves_launcher_untouched(tmp_path, monkeypatch):
    owner_home, _ = _make_home(tmp_path, "owner")
    temp_home, _ = _make_home(tmp_path, "temp")
    repo = _make_repo(tmp_path)
    _isolate(tmp_path, monkeypatch, owner_home)
    before = _publish(repo)

    _isolate(tmp_path, monkeypatch, temp_home)
    _launchers.ensure_install_launchers(repo, repo / ".hermes" / "bin")

    for name, content in before.items():
        assert (repo / ".hermes" / "bin" / name).read_bytes() == content
