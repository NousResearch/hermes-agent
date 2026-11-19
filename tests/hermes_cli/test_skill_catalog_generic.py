"""Generic local mutators honor live catalog ownership through real OS leases.

Every test calls the real mutator (skill_manage edit/delete/remove, curator archive/restore,
hub replacement) against a live projection; no source-string checks.
"""
import multiprocessing
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.skill_catalog.locks import FileLease
from hermes_cli.skill_catalog.validation import CatalogError
from tests.hermes_cli.test_skill_catalog_projection import entry, projection, record, review, started

BODY = "---\nname: {name}\ndescription: test skill\n---\n\n# Body\n"


@pytest.fixture
def home(tmp_path, monkeypatch):
    root = tmp_path / "home"
    (root / "skills").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    import tools.skill_usage as usage
    monkeypatch.setattr(usage, "_prune_builtins_enabled", lambda: False)
    return root


def skill(home, relative, name="mine"):
    directory = home / "skills" / relative
    (directory / "references").mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(BODY.format(name=name), encoding="utf-8")
    (directory / "references/note.md").write_text("note", encoding="utf-8")
    return directory


def snapshot(directory):
    return {p.relative_to(directory).as_posix(): p.read_bytes() for p in sorted(directory.rglob("*")) if p.is_file()}


def _edit(_home):
    from tools.skill_manager_tool import _edit_skill
    return _edit_skill("mine", BODY.format(name="mine") + "\nchanged\n")["success"]


def _delete(_home):
    from tools.skill_manager_tool import _delete_skill
    return _delete_skill("mine")["success"]


def _remove(_home):
    from tools.skill_manager_tool import _remove_file
    return _remove_file("mine", "references/note.md")["success"]


def _write(_home):
    from tools.skill_manager_tool import _write_file
    return _write_file("mine", "references/new.md", "new")["success"]


def _archive(_home):
    from tools.skill_usage import archive_skill
    return archive_skill("mine")[0]


def _quarantine(home, name="mine"):
    from tools.skills_hub import _quarantine_dir
    staged = _quarantine_dir() / "req" / name
    staged.mkdir(parents=True)
    (staged / "SKILL.md").write_text(BODY.format(name=name) + "\nupstream\n", encoding="utf-8")
    return staged


def _hub_replace(home, category="cat"):
    from tools.skills_hub_install import install_from_quarantine
    bundle = SimpleNamespace(source="github", identifier="o/r/mine", trust_level="community",
                             metadata={}, files={"SKILL.md": ""})
    try:
        install_from_quarantine(_quarantine(home), "mine", category, bundle,
                                SimpleNamespace(verdict="safe", scan_provenance=None))
    except ValueError:
        return False
    return True


MUTATORS = dict(edit=_edit, delete=_delete, remove_file=_remove, write_file=_write, archive=_archive,
                hub_replace=_hub_replace)


@pytest.mark.parametrize("claim", ["cat/mine", "cat", "cat/mine/references"])
@pytest.mark.parametrize("action", sorted(MUTATORS))
def test_live_claim_refuses_every_generic_mutator_with_files_intact(home, action, claim):
    target = skill(home, "cat/mine")
    before = snapshot(target)
    with started(home) as (p, owner):
        p.publish(owner, [record(owner, claim)])
        assert claim in p.read()["targets"]
        assert MUTATORS[action](home) is False
    assert snapshot(target) == before
    assert not (home / "skills/.archive").exists() or not any((home / "skills/.archive").iterdir())


@pytest.mark.parametrize("action", sorted(MUTATORS))
def test_live_reservation_refuses_generic_mutator(home, action):
    target = skill(home, "cat/mine")
    before = snapshot(target)
    with started(home) as (p, owner):
        p.publish(owner, [record(owner, "cat/mine")])
        p.reserve(owner, review(owner, "cat/mine", action="update", installation_revision="receipt-1"))
        p.publish(owner, [])  # the reservation alone must still exclude generic writers
        assert not p.read()["targets"] and p.read()["reservations"]
        assert MUTATORS[action](home) is False
    assert snapshot(target) == before


def test_reconciling_registration_holds_generic_writes_busy(home):
    target = skill(home, "cat/mine")
    before = snapshot(target)
    p, owner = projection(home), entry(home)
    handle = p.start(owner)
    try:
        assert _delete(home) is False
        assert _edit(home) is False
    finally:
        owner.stopped = True
        p.stop(owner)
        handle.close()
    assert snapshot(target) == before


def test_restore_destination_reserved_by_provider_refuses(home):
    from tools.skill_usage import archive_skill, restore_skill
    skill(home, "mine")
    assert archive_skill("mine")[0]
    archived = home / "skills/.archive/mine"
    before = snapshot(archived)
    with started(home) as (p, owner):
        p.reserve(owner, review(owner, "mine"))
        ok, _message = restore_skill("mine")
        assert ok is False
    assert snapshot(archived) == before
    assert not (home / "skills/mine").exists()


def test_hub_install_into_reserved_absent_target_refuses(home):
    with started(home) as (p, owner):
        p.reserve(owner, review(owner, "cat/mine"))
        assert _hub_replace(home) is False
    assert not (home / "skills/cat/mine").exists()


def _probe_gate(home):
    """Inside the effect: the mutation gate and the writer chain must both be held."""
    from hermes_cli.skill_catalog.locks import writer_hold
    directory = home / "cache/skill-catalog/live-v1"
    with pytest.raises(CatalogError):
        FileLease(directory / "mutation.lock", deadline=time.monotonic()).acquire()
    with pytest.raises(CatalogError):
        with writer_hold(home, ["cat/mine"], deadline=time.monotonic()):
            pytest.fail("writer chain absent")


def test_stock_no_provider_mutators_succeed_inside_gate(home, monkeypatch):
    import shutil
    import tools.skill_manager_tool as manager
    import tools.skills_hub_install as hub
    seen = []
    real_write, real_rmtree, real_move = manager.atomic_write_text, shutil.rmtree, shutil.move

    def write(*args, **kwargs):
        _probe_gate(home)
        seen.append("write")
        return real_write(*args, **kwargs)

    def rmtree(path, *args, **kwargs):
        if Path(path).is_relative_to(home / "skills/cat"):
            _probe_gate(home)
            seen.append("rmtree")
        return real_rmtree(path, *args, **kwargs)

    def move(src, dst, *args, **kwargs):
        _probe_gate(home)
        seen.append("move")
        return real_move(src, dst, *args, **kwargs)

    monkeypatch.setattr(manager, "atomic_write_text", write)
    monkeypatch.setattr(manager, "shutil", SimpleNamespace(rmtree=rmtree))
    monkeypatch.setattr(hub, "shutil", SimpleNamespace(rmtree=rmtree, move=move))
    skill(home, "cat/mine")
    assert _edit(home) and _write(home) and _remove(home)
    assert _hub_replace(home)
    assert "upstream" in (home / "skills/cat/mine/SKILL.md").read_text()
    assert _delete(home)
    assert not (home / "skills/cat/mine").exists()
    assert {"write", "rmtree", "move"} <= set(seen)
    assert not (home / "cache/skill-catalog/live-v1/claims.json").exists()


def test_stock_archive_and_restore_hold_both_endpoints(home, monkeypatch):
    from hermes_cli.skill_catalog.locks import writer_hold
    from tools.skill_usage import archive_skill, restore_skill
    skill(home, "mine")
    real_rename = Path.rename
    seen = []

    def rename(self, target):
        directory = home / "cache/skill-catalog/live-v1"
        with pytest.raises(CatalogError):
            FileLease(directory / "mutation.lock", deadline=time.monotonic()).acquire()
        with pytest.raises(CatalogError):
            with writer_hold(home, ["mine"], deadline=time.monotonic()):
                pytest.fail("live endpoint unheld")
        seen.append((Path(self).name, Path(target).parent.name))
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", rename)
    assert archive_skill("mine")[0]
    assert (home / "skills/.archive/mine/SKILL.md").is_file()
    assert restore_skill("mine")[0]
    assert (home / "skills/mine/SKILL.md").is_file()
    assert seen == [("mine", ".archive"), ("mine", "skills")]


def _remote_claim(home, connection):
    import os
    os.environ["HERMES_HOME"] = home
    with started(Path(home)) as (p, owner):
        p.publish(owner, [record(owner, "cat/mine")])
        connection.send("claimed")
        connection.recv()


@contextmanager
def remote_claim(home):
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=_remote_claim, args=(str(home), child))
    process.start()
    child.close()
    try:
        assert parent.poll(30)
        assert parent.recv() == "claimed"
        yield process
    finally:
        if process.is_alive():
            with suppress(BrokenPipeError, EOFError, OSError):
                parent.send("close")
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join(10)
        parent.close()


@pytest.mark.platforms("posix", "windows")
def test_cross_process_live_claim_refuses_then_owner_death_restores_stock(home):
    target = skill(home, "cat/mine")
    before = snapshot(target)
    with remote_claim(home) as process:
        assert _delete(home) is False
        assert _archive(home) is False
        assert snapshot(target) == before
        process.kill()
        process.join(10)
        # OS lease release is the only proof; the stale row confers no ownership.
        assert _edit(home) is True
        assert _delete(home) is True
    assert not target.exists()
