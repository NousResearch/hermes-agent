"""Home maintenance and a profile a plugin runs elsewhere.

After an installed revision changes, the first start of each profile runs home maintenance once
(``hermes_cli.boot_bootstrap``): it migrates ``config.yaml``, first copying ``config.yaml`` and
``.env`` beside themselves, drops the home's plugin-catalog cache, and more. Every ``hermes``
command and the gateway process make that call for their own profile, before any command can
refuse. While a plugin runs the profile elsewhere, its home is written by the backend the plugin
runs, so a host process does none of it: the maintenance stays owed, and the first start after the
marker is gone does it.

The steps here are the real config migration over real files, or counters where only the order
of calls matters. Installs are sealed trees (an ``install-stamp.json``), so no git is needed.
"""
from __future__ import annotations

import errno
import json
import logging
import os
from pathlib import Path

import pytest

from hermes_cli import boot_bootstrap, post_update
from hermes_cli import profiles_run_elsewhere as pre

PLUGIN = "sandbox-plugin"
KEYS = "MADE_UP_PROVIDER_KEY=sk-made-up-0123456789\n"


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A Hermes root with the default profile and the named profiles ``ops`` and ``dev``; this
    process's profile is ``ops`` (not marked yet)."""
    home = tmp_path / ".hermes"
    for folder in (home, home / "profiles" / "ops", home / "profiles" / "dev"):
        folder.mkdir(parents=True)
        (folder / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home / "profiles" / "ops"))
    return home


@pytest.fixture
def install(tmp_path):
    """A sealed install whose revision is ``aaaa1111``."""
    tree = tmp_path / "payload"
    tree.mkdir()
    _revision(tree, "aaaa1111")
    return tree


def _revision(tree: Path, commit: str) -> None:
    (tree / "install-stamp.json").write_text(
        json.dumps({"commit": commit, "payload": "full", "updateMechanism": "electron-updater"}), encoding="utf-8")


@pytest.fixture
def steps(monkeypatch):
    """One counting step in place of the registry; ``steps["ran"]`` lists the home of each run."""
    ran: list[str] = []

    def step():
        ran.append(os.environ["HERMES_HOME"])
        return {"ok": True}

    monkeypatch.setattr(post_update, "BOOT_HOME_STEPS", (("counted", step),))
    return {"ran": ran}


def _mark(root: Path, profile: str = "ops") -> None:
    pre.mark_profile_run_elsewhere(profile, PLUGIN, root=root)


def _clear(root: Path, profile: str = "ops") -> None:
    assert pre.clear_profile_run_elsewhere(profile, PLUGIN, root=root)


def _link(link: Path, target: Path) -> None:
    """Make *link* lead to the folder *target*: a symlink, or on Windows a junction when this
    account may not create symlinks. Skips where the OS offers neither."""
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        if os.name != "nt":
            pytest.skip(f"cannot create a directory symlink here: {exc}")
        import _winapi

        _winapi.CreateJunction(str(target), str(link))


# ── the copies of the keys file ───────────────────────────────────────


def _outdated_home(home: Path) -> dict[str, bytes]:
    """Give *home* a config one schema version behind and a keys file; returns their bytes."""
    from hermes_cli.config import DEFAULT_CONFIG

    (home / "config.yaml").write_text(
        f"_config_version: {int(DEFAULT_CONFIG['_config_version']) - 1}\n", encoding="utf-8")
    (home / ".env").write_text(KEYS, encoding="utf-8")
    return {name: (home / name).read_bytes() for name in ("config.yaml", ".env")}


def _real_migration(monkeypatch) -> None:
    monkeypatch.setattr(post_update, "BOOT_HOME_STEPS", (("migrate_config", post_update.step_migrate_config),))


def test_a_marked_home_gets_no_migration_and_no_copy_of_its_keys_file(root, install, monkeypatch):
    """The reproduction: a config behind the installed schema, a keys file, a new revision."""
    ops = root / "profiles" / "ops"
    before = _outdated_home(ops)
    _real_migration(monkeypatch)
    _mark(root)

    assert boot_bootstrap.run_boot_bootstrap(install) == {"home": "runs-elsewhere"}

    assert {name: (ops / name).read_bytes() for name in before} == before
    assert sorted(entry.name for entry in ops.iterdir()) == [".env", "config.yaml"]
    assert not boot_bootstrap.record_path(install).exists()


def test_the_first_start_after_the_marker_is_gone_migrates(root, install, monkeypatch):
    ops = root / "profiles" / "ops"
    before = _outdated_home(ops)
    _real_migration(monkeypatch)
    _mark(root)
    boot_bootstrap.run_boot_bootstrap(install)
    _clear(root)

    done = boot_bootstrap.run_boot_bootstrap(install)["home"]

    assert done["migrate_config"]["ok"] is True and "migrated" in done["migrate_config"]
    assert (ops / "config.yaml").read_bytes() != before["config.yaml"]
    copies = sorted(entry.name for entry in ops.iterdir() if ".bak-" in entry.name)
    assert [name.split(".bak-")[0] for name in copies] == [".env", "config.yaml"]
    assert (ops / copies[0]).read_bytes() == before[".env"]  # what the marked home was spared
    assert boot_bootstrap.read_last_known(boot_bootstrap.record_path(install))["identity"] == "aaaa1111"


# ── owed while marked, done once after ────────────────────────────────


def test_maintenance_stays_owed_while_marked_and_runs_once_when_the_marker_is_gone(root, install, steps):
    _mark(root)
    for _start in range(3):
        assert boot_bootstrap.run_boot_bootstrap(install) == {"home": "runs-elsewhere"}
    assert steps["ran"] == []
    assert not boot_bootstrap.record_path(install).exists()  # nothing says this revision is done
    assert boot_bootstrap.needs_bootstrap(install) == "aaaa1111"

    _clear(root)
    assert boot_bootstrap.run_boot_bootstrap(install) == {"home": {"counted": {"ok": True}}}
    assert boot_bootstrap.run_boot_bootstrap(install) == {"home": "skipped"}
    assert steps["ran"] == [str(root / "profiles" / "ops")]


def test_a_profile_marked_after_its_maintenance_was_done_owes_the_next_revision(root, install, steps):
    boot_bootstrap.run_boot_bootstrap(install)
    _mark(root)
    assert boot_bootstrap.run_boot_bootstrap(install) == {"home": "skipped"}  # nothing due: nothing asked
    _revision(install, "bbbb2222")
    assert boot_bootstrap.run_boot_bootstrap(install) == {"home": "runs-elsewhere"}
    assert boot_bootstrap.read_last_known(boot_bootstrap.record_path(install))["identity"] == "aaaa1111"
    _clear(root)
    boot_bootstrap.run_boot_bootstrap(install)
    assert len(steps["ran"]) == 2
    assert boot_bootstrap.read_last_known(boot_bootstrap.record_path(install))["identity"] == "bbbb2222"


def test_a_step_owed_from_an_earlier_start_is_not_retried_while_marked(root, install, monkeypatch):
    """A step a start left owed (default plugins a failed fetch left uninstalled) is retried on
    every start; not for a home that runs elsewhere, and again once it runs here."""
    outcomes = iter([{"ok": True, "owed": True}, {"ok": True}])
    ran: list[int] = []

    def step():
        ran.append(1)
        return next(outcomes)

    monkeypatch.setattr(post_update, "BOOT_HOME_STEPS", (("install_default_plugins", step),))
    boot_bootstrap.run_boot_bootstrap(install)
    record = boot_bootstrap.record_path(install)
    owed = record.read_bytes()
    _mark(root)

    assert boot_bootstrap.run_boot_bootstrap(install) == {"home": "runs-elsewhere"}
    assert len(ran) == 1 and record.read_bytes() == owed

    _clear(root)
    assert boot_bootstrap.run_boot_bootstrap(install) == {"home": {"install_default_plugins": {"ok": True}}}
    assert len(ran) == 2


# ── who is not affected ───────────────────────────────────────────────


def test_an_unmarked_profile_is_maintained_as_before(root, install, steps):
    _mark(root, "dev")  # another profile of the same root
    assert boot_bootstrap.run_boot_bootstrap(install) == {"home": {"counted": {"ok": True}}}
    assert steps["ran"] == [str(root / "profiles" / "ops")]
    assert boot_bootstrap.record_path(install).name == "ops.json"


def test_the_default_profile_is_maintained_while_a_named_profile_is_marked(root, install, steps, monkeypatch):
    """The default profile's home holds ``profiles/ops``; the marker of ``ops`` is not its own."""
    monkeypatch.setenv("HERMES_HOME", str(root))
    _mark(root)
    assert boot_bootstrap.run_boot_bootstrap(install) == {"home": {"counted": {"ok": True}}}
    assert steps["ran"] == [str(root)]
    assert boot_bootstrap.record_path(install).name == "default.json"


def test_the_backend_the_plugin_runs_is_not_held_back(root, install, steps, monkeypatch):
    """A process that is denied the marker folder is the backend elsewhere, not the host."""
    _mark(root)
    real_listdir = os.listdir

    def listdir(path):
        if Path(path).name == pre.RUN_ELSEWHERE_DIRNAME:
            raise PermissionError(errno.EACCES, "Access is denied")
        return real_listdir(path)

    monkeypatch.setattr(pre.os, "listdir", listdir)
    assert boot_bootstrap.run_boot_bootstrap(install) == {"home": {"counted": {"ok": True}}}


def test_a_marker_folder_that_cannot_be_read_holds_maintenance_back(root, install, steps, monkeypatch):
    """Any other error reading the folder refuses every profile; its home is left alone too."""
    real_listdir = os.listdir

    def listdir(path):
        if Path(path).name == pre.RUN_ELSEWHERE_DIRNAME:
            raise OSError(errno.EIO, "I/O error")
        return real_listdir(path)

    monkeypatch.setattr(pre.os, "listdir", listdir)
    assert boot_bootstrap.run_boot_bootstrap(install) == {"home": "runs-elsewhere"}
    assert steps["ran"] == [] and not boot_bootstrap.record_path(install).exists()


# ── under every name ──────────────────────────────────────────────────


def test_a_marked_home_reached_through_a_link_is_left_alone(root, install, steps, monkeypatch):
    """``profiles/alias`` is a link to the home of ``ops``: the same directory under a second name."""
    alias = root / "profiles" / "alias"
    _link(alias, root / "profiles" / "ops")
    monkeypatch.setenv("HERMES_HOME", str(alias))
    _mark(root)
    assert boot_bootstrap.run_boot_bootstrap(install) == {"home": "runs-elsewhere"}
    assert steps["ran"] == []


def test_the_directory_behind_a_marked_link_is_left_alone(root, install, steps):
    """The marked name is itself the link: the directory it leads to is covered under its own name."""
    _link(root / "profiles" / "alias", root / "profiles" / "ops")
    _mark(root, "alias")
    assert boot_bootstrap.run_boot_bootstrap(install) == {"home": "runs-elsewhere"}
    assert steps["ran"] == []


def test_a_link_from_outside_the_root_is_left_alone(root, install, steps, tmp_path, monkeypatch):
    elsewhere = tmp_path / "shortcut"
    _link(elsewhere, root / "profiles" / "ops")
    monkeypatch.setenv("HERMES_HOME", str(elsewhere))
    _mark(root)
    assert boot_bootstrap.run_boot_bootstrap(install) == {"home": "runs-elsewhere"}
    assert steps["ran"] == []


# ── cost and words ────────────────────────────────────────────────────


def test_nothing_is_asked_when_no_maintenance_is_due(root, install, steps, monkeypatch):
    """The usual start (this revision's maintenance is done, nothing owed) reads its record and
    returns: no look at the marker folder is added to it."""
    boot_bootstrap.run_boot_bootstrap(install)

    def asked(home):
        raise AssertionError(f"asked whether {home} runs elsewhere")

    monkeypatch.setattr(pre, "run_elsewhere_for_home", asked)
    monkeypatch.setattr(pre, "home_runs_elsewhere", asked)
    assert boot_bootstrap.run_boot_bootstrap(install) == {"home": "skipped"}


def test_the_marker_is_asked_once_per_start_and_before_the_lock(root, install, steps, monkeypatch):
    _mark(root)
    asked: list[str] = []
    real = pre.run_elsewhere_for_home

    def counting(home):
        asked.append(str(home))
        return real(home)

    def no_lock(self):
        raise AssertionError("took the record lock for a home that runs elsewhere")

    monkeypatch.setattr(pre, "run_elsewhere_for_home", counting)
    monkeypatch.setattr(boot_bootstrap._RecordLock, "acquire", no_lock)
    assert boot_bootstrap.run_boot_bootstrap(install) == {"home": "runs-elsewhere"}
    assert asked == [str(root / "profiles" / "ops")]


def test_one_plain_log_line_says_why(root, install, steps, caplog):
    _mark(root)
    with caplog.at_level(logging.INFO, logger="hermes_cli.boot_bootstrap"):
        boot_bootstrap.maybe_run_boot_bootstrap(install)
    lines = [record.getMessage() for record in caplog.records if record.name == "hermes_cli.boot_bootstrap"]
    assert lines == [
        "home maintenance: the 'sandbox-plugin' plugin runs profile 'ops' elsewhere, so this host leaves "
        "its home alone; it is owed until the profile runs here"
    ]
