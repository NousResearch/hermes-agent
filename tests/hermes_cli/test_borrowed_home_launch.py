"""A launch under a data root that does not own the checkout leaves the checkout to its owner (#123238).

Layout in every test: the user's default root is ``<tmp>/.hermes`` (``Path.home`` is ``<tmp>``) and
the checkout is ``<tmp>/.hermes/hermes-agent``, the ``install.sh`` layout. A root "has state" for the
checkout when ``<root>/installs/<install_key>/facts.json`` exists -- the record a committed
dependency sync leaves.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

from hermes_cli import venv_sync
from pm.environments import install_key, install_state_dir, owning_home_root, store_root

# Every lane, Windows included: the owner check resolves the platform default root, which differs on Windows, so the
# marked-OS lanes (scripts/ci/list_os_marked_tests.py) must collect this file too.
pytestmark = pytest.mark.platforms("any")


@pytest.fixture(autouse=True)
def _no_tool_downloads(monkeypatch):
    import pm.client

    monkeypatch.setattr(pm.client, "ensure_tools_for_sync", lambda: None)


@pytest.fixture
def completion_tail(monkeypatch):
    spawned: list = []
    monkeypatch.setattr(venv_sync.subprocess, "call", lambda command, **kw: spawned.append(command) or 0)
    return spawned


def _checkout(tmp_path, monkeypatch, *, parent: Path | None = None) -> Path:
    import hermes_constants

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # The layout's default root is ``<tmp>/.hermes`` on every OS. On Windows the platform default is
    # ``%LOCALAPPDATA%\hermes`` (``Path.home`` only backs an unset ``LOCALAPPDATA``), and no
    # ``LOCALAPPDATA`` value spells ``<tmp>/.hermes``, so pin the default itself.
    monkeypatch.setattr(hermes_constants, "_get_platform_default_hermes_home", lambda: tmp_path / ".hermes")
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    root = (parent or tmp_path / ".hermes") / "hermes-agent"
    root.mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='example'\n", encoding="utf-8")
    (root / "install-stamp.json").write_text(json.dumps({"updateMechanism": "self", "commit": "abc"}), encoding="utf-8")
    return root


def _state(data_root: Path, checkout: Path) -> Path:
    facts = data_root / "installs" / install_key(checkout) / "facts.json"
    facts.parent.mkdir(parents=True, exist_ok=True)
    facts.write_text(json.dumps({"packages": {"venv": {"stamp": "complete", "extras": ["all"]}}}), encoding="utf-8")
    return facts


def _home(monkeypatch, path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(path))
    return path


def test_a_temporary_home_borrows_the_default_roots_checkout(tmp_path, monkeypatch):
    root = _checkout(tmp_path, monkeypatch)
    owner = tmp_path / ".hermes"
    _state(owner, root)
    _home(monkeypatch, tmp_path / "tmp" / "hermes-test-a")

    assert owning_home_root(root) == owner
    assert store_root(root) == owner / "tools"


@pytest.mark.parametrize("home", ["default", "profile"])
def test_the_owner_and_its_profiles_are_not_borrowers(tmp_path, monkeypatch, home):
    root = _checkout(tmp_path, monkeypatch)
    owner = tmp_path / ".hermes"
    _state(owner, root)
    if home == "profile":
        _home(monkeypatch, owner / "profiles" / "work")
    else:
        monkeypatch.delenv("HERMES_HOME", raising=False)

    assert owning_home_root(root) is None
    assert store_root(root) == owner / "tools"


def test_a_fresh_install_and_a_custom_root_keep_their_own_store(tmp_path, monkeypatch):
    # Nobody has state yet: the launching root is about to become the owner.
    root = _checkout(tmp_path, monkeypatch, parent=tmp_path / "data" / "h")
    _home(monkeypatch, tmp_path / "data" / "h")
    assert owning_home_root(root) is None
    assert store_root(root) == tmp_path / "data" / "h" / "tools"

    # `install.sh --hermes-home /data/h` owns its tree even if the default root once
    # borrowed it and kept state of its own.
    _state(tmp_path / "data" / "h", root)
    _state(tmp_path / ".hermes", root)
    assert owning_home_root(root) is None
    assert store_root(root) == tmp_path / "data" / "h" / "tools"


def test_a_service_home_borrows_the_root_the_checkout_sits_in(tmp_path, monkeypatch):
    """A CI runner whose HOME is its own tool-home runs the host's checkout from PATH: the host's
    root is the checkout's parent, not this process's platform default."""
    host = tmp_path / "host" / ".hermes"
    root = _checkout(tmp_path / "runner", monkeypatch, parent=host)
    _state(host, root)
    _home(monkeypatch, tmp_path / "runner" / ".hermes" / "pytest-1")

    assert owning_home_root(root) == host
    assert store_root(root) == host / "tools"


@pytest.mark.skipif(not hasattr(os, "symlink") or sys.platform == "win32", reason="POSIX symlink")
def test_a_home_whose_installs_link_back_to_the_owner_is_the_owner(tmp_path, monkeypatch):
    root = _checkout(tmp_path, monkeypatch)
    owner = tmp_path / ".hermes"
    _state(owner, root)
    task = _home(monkeypatch, tmp_path / "tasks" / "t1")
    (task / "installs").symlink_to(owner / "installs")

    assert owning_home_root(root) is None


def test_an_explicit_runtime_dir_still_wins(tmp_path, monkeypatch):
    root = _checkout(tmp_path, monkeypatch)
    _state(tmp_path / ".hermes", root)
    _home(monkeypatch, tmp_path / "tmp" / "a")
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "runtime"))

    assert store_root(root) == (tmp_path / "runtime").resolve()


def test_a_borrowing_launch_syncs_its_own_dependencies_and_nothing_of_the_checkout(
    tmp_path, monkeypatch, completion_tail
):
    """The first launch under a temporary home: its own dependency generation, but no tail (products,
    maintenance, install stamp), no launcher publication and no owner marker touched."""
    import pm
    from hermes_cli import _launchers

    root = _checkout(tmp_path, monkeypatch)
    _state(tmp_path / ".hermes", root)
    _home(monkeypatch, tmp_path / "tmp" / "hermes-flash-compat-x" / "a")
    stamp = (root / "install-stamp.json").read_bytes()
    owner_marker = root / ".update-incomplete"
    owner_marker.write_text("owner's update\n", encoding="utf-8")
    borrowed_facts = install_state_dir(root) / "facts.json"
    syncs = []

    def sync(extras=None, **kwargs):
        syncs.append(extras)
        borrowed_facts.parent.mkdir(parents=True, exist_ok=True)
        borrowed_facts.write_text("{}", encoding="utf-8")

    python = tmp_path / ".hermes" / "tools" / "python" / "bin" / "python3"
    monkeypatch.setattr(pm, "venv_is_current", lambda **kw: borrowed_facts.is_file())
    monkeypatch.setattr(pm, "sync_venv", sync)
    monkeypatch.setattr(_launchers, "resolve_store_python", lambda _: python)
    monkeypatch.setattr(venv_sync, "publish_launchers", lambda *a, **kw: pytest.fail("published the owner's launchers"))

    assert venv_sync.prepare_launch(root, ["kanban", "init"]) == python  # relaunch into the synced env
    assert len(syncs) == 1
    assert completion_tail == [], "a borrowing launch ran the checkout's update tail"
    assert not venv_sync.completion_pending_path(root).exists(), "a tail was armed for a borrowing root"
    assert owner_marker.is_file(), "the owner's update marker was removed"
    assert (root / "install-stamp.json").read_bytes() == stamp

    # Its relaunch finds everything current and does nothing further.
    monkeypatch.setattr(sys, "executable", str(python))
    assert venv_sync.prepare_launch(root, ["kanban", "init"]) is None
    assert len(syncs) == 1 and completion_tail == []


def test_a_borrowing_launch_ignores_a_tail_armed_before_the_fix(tmp_path, monkeypatch, completion_tail):
    import pm
    from hermes_cli import _launchers

    root = _checkout(tmp_path, monkeypatch)
    _state(tmp_path / ".hermes", root)
    _home(monkeypatch, tmp_path / "tmp" / "a")
    pending = venv_sync.completion_pending_path(root)
    pending.parent.mkdir(parents=True)
    pending.write_text("source update tail not finished\n", encoding="utf-8")
    monkeypatch.setattr(pm, "venv_is_current", lambda **kw: True)
    monkeypatch.setattr(_launchers, "resolve_store_python", lambda _: Path(sys.executable))

    assert venv_sync.prepare_launch(root, []) is None
    assert completion_tail == []


def test_the_owner_still_owes_and_runs_its_tail(tmp_path, monkeypatch, completion_tail):
    """Unchanged for the owner: a stale environment syncs, arms and runs the tail."""
    import pm
    from hermes_cli import _launchers

    root = _checkout(tmp_path, monkeypatch)
    facts = _state(tmp_path / ".hermes", root)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    facts.unlink()
    monkeypatch.setattr(pm, "venv_is_current", lambda **kw: facts.is_file())
    monkeypatch.setattr(pm, "sync_venv", lambda *a, **kw: facts.write_text("{}", encoding="utf-8"))
    monkeypatch.setattr(_launchers, "resolve_store_python", lambda _: Path(sys.executable))
    published = []
    monkeypatch.setattr(venv_sync, "publish_launchers", lambda r, **kw: published.append(r))

    assert venv_sync.prepare_launch(root, []) == Path(sys.executable)
    assert len(completion_tail) == 1
    assert published == [root]
