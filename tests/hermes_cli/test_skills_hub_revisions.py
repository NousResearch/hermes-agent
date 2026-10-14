"""CLI real-import/temp-home contracts; scanners/remote transports may be mocked."""
import argparse
import hashlib
from io import StringIO
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from hermes_cli import skills_hub as cli
from hermes_cli.subcommands.skills import build_skills_parser
from tools import skills_hub as hub
from tools.skills_guard import ScanResult, content_hash
from tools.skills_hub_install import check_for_skill_updates, install_from_quarantine, quarantine_bundle
from tools.skills_hub_models import SkillBundle

COMMIT = "a" * 40
TREE = "b" * 40
IDENTIFIER = "sample/bundle/candidates/demo"


def blob_hash(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


@pytest.fixture
def installed(monkeypatch, tmp_path):
    monkeypatch.setattr(hub, "SKILLS_DIR", tmp_path / "skills")
    monkeypatch.setattr(hub, "HUB_DIR", tmp_path / "skills/.hub")
    monkeypatch.setattr(hub, "LOCK_FILE", tmp_path / "skills/.hub/lock.json")
    monkeypatch.setattr(hub, "QUARANTINE_DIR", tmp_path / "skills/.hub/quarantine")
    monkeypatch.setattr(hub, "AUDIT_LOG", tmp_path / "skills/.hub/audit.log")
    files = {"SKILL.md": b"---\nname: demo\ndescription: Example.\n---\nExample.\n", "guide.md": b"Guide\n"}
    bundle = SkillBundle("demo", files, "github", IDENTIFIER, "community", {
        "requested_revision": COMMIT, "resolved_commit": COMMIT, "source_tree": TREE,
        "source_revision": TREE, "source_blob_hashes": {rel: blob_hash(data) for rel, data in files.items()}})
    result = ScanResult(skill_name="demo", source="community", trust_level="community", verdict="safe")
    path = install_from_quarantine(quarantine_bundle(bundle), "demo", "", bundle, result)
    source = MagicMock()
    source.source_id.return_value = "github"
    source.fetch.return_value = bundle
    source.inspect.return_value = None
    monkeypatch.setattr(cli, "_sources", lambda: [source])
    monkeypatch.setattr(cli, "_print_tier1_advisory", lambda *_: None)
    monkeypatch.setattr(cli, "_finish_change", lambda *_: None)
    monkeypatch.setattr(cli, "_announce_blueprint", lambda *_: None)
    monkeypatch.setattr(cli, "_record_skill_install", lambda *_args, **_kw: None)
    return path, bundle, source, Console(file=StringIO(), force_terminal=False)


@pytest.mark.parametrize("action", ["install", "inspect"])
@pytest.mark.parametrize("revision", ["main", "a" * 7, "A" * 40, "", COMMIT + "\n"])
def test_cli_invalid_revision_no_source_resolution(monkeypatch, action, revision):
    monkeypatch.setattr(cli, "_sources", lambda: pytest.fail("source resolution before validation"))
    assert getattr(cli, "do_" + action)(IDENTIFIER, revision=revision,
                                       console=Console(file=StringIO())) is False


@pytest.mark.parametrize("action", ["install", "inspect"])
def test_cli_parser_dispatches_literal_revision(monkeypatch, action):
    parser = argparse.ArgumentParser()
    build_skills_parser(parser.add_subparsers(), cmd_skills=lambda *_: None)
    args = parser.parse_args(["skills", action, IDENTIFIER, "--revision", COMMIT])
    handler = MagicMock()
    monkeypatch.setattr(cli, "do_" + action, handler)
    cli._CLI_ACTIONS[action](args)
    assert handler.call_args.kwargs["revision"] == COMMIT


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("verdict", ["caution", "dangerous"])
def test_update_scanner_refusal_never_overwrites_files_or_lock(installed, monkeypatch, overwrite, verdict):
    path, _bundle, source, console = installed
    # Legacy floating install legitimately has an upstream change; no pin is inferred from its tree SHA.
    entry = hub.HubLockFile().get_installed("demo")
    entry["metadata"] = {"source_revision": TREE}
    lock = hub.HubLockFile()
    data = lock.load()
    data["installed"]["demo"] = entry
    lock.save(data)
    changed = SkillBundle("demo", {"SKILL.md": b"new remote content"}, "github", IDENTIFIER, "community")
    source.fetch.return_value = changed
    source.current_revision.return_value = "c" * 40
    import tools.skills_hub_search as search
    monkeypatch.setattr(search, "create_source_router", lambda **_: [source])
    result = ScanResult(skill_name="demo", source="community", trust_level="community", verdict=verdict)
    monkeypatch.setattr(cli, "_scan_quarantined", lambda *_: result)
    before_hash, before_lock = content_hash(path), hub.LOCK_FILE.read_bytes()
    assert cli.do_update("demo", console=console, force=overwrite) is False
    assert content_hash(path) == before_hash
    assert hub.LOCK_FILE.read_bytes() == before_lock


def test_pin_check_and_two_updates_are_idempotent(installed, monkeypatch):
    path, _bundle, source, console = installed
    import tools.skills_hub_search as search
    monkeypatch.setattr(search, "create_source_router", lambda **_: [source])
    before_lock, before_hash = hub.LOCK_FILE.read_bytes(), content_hash(path)
    for _ in range(2):
        row = check_for_skill_updates(sources=[source])[0]
        assert row["status"] == "up_to_date"
        assert cli.do_update("demo", console=console) is None
    assert hub.LOCK_FILE.read_bytes() == before_lock
    assert content_hash(path) == before_hash
    source.current_revision.assert_not_called()
    assert all(call.kwargs == {"revision": COMMIT} for call in source.fetch.call_args_list)


@pytest.mark.parametrize("drift", ["edit", "add", "delete"])
def test_disk_drift_detected_and_preserved_without_overwrite(installed, monkeypatch, drift):
    path, _, source, console = installed
    if drift == "edit":
        (path / "guide.md").write_bytes(b"local edit")
    elif drift == "add":
        (path / "extra.txt").write_bytes(b"local file")
    else:
        (path / "guide.md").unlink()
    import tools.skills_hub_search as search
    monkeypatch.setattr(search, "create_source_router", lambda **_: [source])
    row = check_for_skill_updates(sources=[source])[0]
    assert row["status"] == "update_available" and row["local_modified"]
    before_hash, before_lock = content_hash(path), hub.LOCK_FILE.read_bytes()
    assert cli.do_update("demo", console=console) is None
    assert content_hash(path) == before_hash
    assert hub.LOCK_FILE.read_bytes() == before_lock


def test_pinned_drift_overwrite_remains_scanner_gated(installed, monkeypatch):
    path, _, source, console = installed
    (path / "guide.md").write_bytes(b"local edit")
    import tools.skills_hub_search as search
    monkeypatch.setattr(search, "create_source_router", lambda **_: [source])
    result = ScanResult(skill_name="demo", source="community", trust_level="community", verdict="caution")
    monkeypatch.setattr(cli, "_scan_quarantined", lambda *_: result)
    before = hub.LOCK_FILE.read_bytes(), content_hash(path)
    assert cli.do_update("demo", console=console, force=True) is False
    assert (hub.LOCK_FILE.read_bytes(), content_hash(path)) == before


def test_real_scanner_install_with_temp_home_records_pin_and_inventory(installed):
    path, bundle, _source, console = installed
    assert cli.do_install(IDENTIFIER, revision=COMMIT, overwrite_existing=True,
                          skip_confirm=True, console=console) is True
    entry = hub.HubLockFile().get_installed("demo")
    assert entry["metadata"]["resolved_commit"] == COMMIT
    assert entry["metadata"]["source_revision"] == TREE
    assert set(entry["files"]) == set(bundle.files)
    assert entry["content_hash"] == content_hash(path)
    assert entry["scan_verdict"] == "safe"
    assert entry["scan_provenance"]["source"] == IDENTIFIER


@pytest.mark.parametrize("change", ["identifier", "source"])
def test_pinned_source_collision_preserves_existing(installed, change):
    path, bundle, _source, console = installed
    before = hub.LOCK_FILE.read_bytes(), content_hash(path)
    setattr(bundle, change, "other/repository/candidates/demo" if change == "identifier" else "skills.sh")
    assert cli.do_install(IDENTIFIER, revision=COMMIT, overwrite_existing=True,
                          skip_confirm=True, console=console) is False
    assert (hub.LOCK_FILE.read_bytes(), content_hash(path)) == before


def test_pinned_local_untracked_collision_is_not_deleted(installed):
    path, bundle, _source, _console = installed
    hub.HubLockFile().record_uninstall("demo")
    before = hub.LOCK_FILE.read_bytes(), content_hash(path)
    scan = ScanResult(skill_name="demo", source="community", trust_level="community", verdict="safe")
    with pytest.raises(ValueError, match="untracked"):
        install_from_quarantine(quarantine_bundle(bundle), "demo", "", bundle, scan)
    assert (hub.LOCK_FILE.read_bytes(), content_hash(path)) == before


def test_publication_error_restores_lock_and_previous_files(installed, monkeypatch):
    path, bundle, _source, _console = installed
    before = hub.LOCK_FILE.read_bytes(), content_hash(path)
    bundle.files = {"SKILL.md": b"replacement"}
    # Transaction faults require a coherent candidate; they do not test pin corruption.
    bundle.metadata["source_blob_hashes"] = {"SKILL.md": blob_hash(b"replacement")}
    scan = ScanResult(skill_name="demo", source="community", trust_level="community", verdict="safe")

    def fail_record(self, **kwargs):
        self.path.write_bytes(b"incomplete lock")
        raise OSError("simulated storage failure")

    monkeypatch.setattr(hub.HubLockFile, "record_install", fail_record)
    with pytest.raises(OSError, match="storage failure"):
        install_from_quarantine(quarantine_bundle(bundle), "demo", "", bundle, scan)
    assert (hub.LOCK_FILE.read_bytes(), content_hash(path)) == before


def test_failed_pinned_fetch_does_not_fall_back_or_mutate(installed, monkeypatch):
    path, _bundle, source, console = installed
    source.fetch.return_value = None
    foreign = MagicMock()
    foreign.source_id.return_value = "clawhub"
    monkeypatch.setattr(cli, "_sources", lambda: [source, foreign])
    before = hub.LOCK_FILE.read_bytes(), content_hash(path)
    assert cli.do_install(IDENTIFIER, revision=COMMIT, overwrite_existing=True,
                          console=console) is False
    foreign.fetch.assert_not_called()
    foreign.inspect.assert_not_called()
    assert (hub.LOCK_FILE.read_bytes(), content_hash(path)) == before


@pytest.mark.parametrize("identifier", ["official/a/b", "clawhub/a/b", "lobehub/a/b", "https://example.test/SKILL.md", "demo"])
def test_revision_unsupported_source_rejected_before_resolution(monkeypatch, identifier):
    monkeypatch.setattr(cli, "_sources", lambda: pytest.fail("unsupported source contacted"))
    assert cli.do_install(identifier, revision=COMMIT, console=Console(file=StringIO())) is False


@pytest.mark.parametrize("field", ["requested_revision", "resolved_commit", "source_tree"])
def test_invalid_or_inconsistent_lock_pin_unavailable_not_floating(installed, monkeypatch, field):
    path, _bundle, source, console = installed
    lock = hub.HubLockFile()
    data = lock.load()
    data["installed"]["demo"]["metadata"][field] = "main" if field == "requested_revision" else "c" * 40
    lock.save(data)
    before = hub.LOCK_FILE.read_bytes(), content_hash(path)
    import tools.skills_hub_search as search
    monkeypatch.setattr(search, "create_source_router", lambda **_: [source])
    assert cli.do_update("demo", console=console) is False
    source.current_revision.assert_not_called()
    assert (hub.LOCK_FILE.read_bytes(), content_hash(path)) == before


def test_safe_explicit_overwrite_restores_pin_and_next_updates_are_noops(installed, monkeypatch):
    path, _bundle, source, console = installed
    (path / "guide.md").write_bytes(b"local edit")
    import tools.skills_hub_search as search
    monkeypatch.setattr(search, "create_source_router", lambda **_: [source])
    assert cli.do_update("demo", console=console, force=True) is None
    assert (path / "guide.md").read_bytes() == b"Guide\n"
    before = hub.LOCK_FILE.read_bytes(), content_hash(path)
    for _ in range(2):
        assert cli.do_update("demo", console=console) is None
        assert (hub.LOCK_FILE.read_bytes(), content_hash(path)) == before
    assert hub.HubLockFile().get_installed("demo")["metadata"]["resolved_commit"] == COMMIT


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("fault", ["bytes", "missing", "extra", "alias", "hardlink", "directory", "symlink", "fifo"])
@pytest.mark.parametrize("phase", ["materialize", "scan", "direct_publication"])
def test_pinned_materialization_integrity_preserves_existing(installed, monkeypatch, fault, phase):
    """Actual quarantine mutations, not an emulation of Windows/macOS filesystems."""
    import copy
    import os
    from pathlib import Path
    path, bundle, source, console = installed
    before = hub.LOCK_FILE.read_bytes(), content_hash(path)
    metadata = copy.deepcopy(bundle.metadata)
    real_write = Path.write_bytes

    def damage(directory):
        guide = directory / "guide.md"
        actions = {
            "bytes": lambda: real_write(guide, b"corrupted materialization"),
            "missing": guide.unlink,
            "extra": lambda: real_write(directory / "extra.md", b"extra"),
            "alias": lambda: guide.rename(directory / "GUIDE.md"),
            "hardlink": lambda: (guide.unlink(), os.link(directory / "SKILL.md", guide)),
            "directory": lambda: (directory / "unexpected").mkdir(),
            "symlink": lambda: (guide.unlink(), guide.symlink_to("SKILL.md")),
            "fifo": lambda: (guide.unlink(), os.mkfifo(guide)),
        }
        actions[fault]()

    scan = ScanResult(skill_name="demo", source="community", trust_level="community", verdict="safe")
    if phase == "direct_publication":
        quarantine = quarantine_bundle(bundle)
        damage(quarantine)
        with pytest.raises(ValueError):
            install_from_quarantine(quarantine, "demo", "", bundle, scan)
    else:
        if phase == "materialize":
            def write(target, body):
                result = real_write(target, body)
                if target == hub.QUARANTINE_DIR / "demo/guide.md":
                    damage(target.parent)
                return result
            monkeypatch.setattr(Path, "write_bytes", write)
            monkeypatch.setattr(cli, "_scan_quarantined", lambda *_: pytest.fail("scan before integrity check"))
        else:
            def scanning(_console, directory, *_args):
                damage(directory)
                return scan
            monkeypatch.setattr(cli, "_scan_quarantined", scanning)
        assert cli.do_install(IDENTIFIER, revision=COMMIT, overwrite_existing=True,
                              skip_confirm=True, console=console) is False
        assert not (hub.QUARANTINE_DIR / "demo").exists()
    assert (hub.LOCK_FILE.read_bytes(), content_hash(path)) == before
    assert bundle.metadata == metadata
    source.current_revision.assert_not_called()


@pytest.mark.parametrize("name", ["demo.", "demo ", "CON", "NUL.txt", "COM¹", "a?b"])
def test_pinned_quarantine_name_rejected_before_write(installed, name):
    path, bundle, _source, _console = installed
    before = hub.LOCK_FILE.read_bytes(), content_hash(path)
    bundle.name = name
    with pytest.raises(ValueError):
        quarantine_bundle(bundle)
    assert (hub.LOCK_FILE.read_bytes(), content_hash(path)) == before
    assert list(hub.QUARANTINE_DIR.iterdir()) == []


@pytest.mark.parametrize("text", ["Line\n資料\n", "Line\r\n資料\r\n", "Line\r\n資料\n"])
@pytest.mark.parametrize("as_text", [False, True])
def test_pinned_materialization_literal_utf8_newlines(installed, text, as_text):
    path, bundle, _source, _console = installed
    raw = text.encode("utf-8")
    bundle.files["guide.md"] = text if as_text else raw
    bundle.metadata["source_blob_hashes"]["guide.md"] = blob_hash(raw)
    scan = ScanResult(skill_name="demo", source="community", trust_level="community", verdict="safe")
    quarantine = quarantine_bundle(bundle)
    assert (quarantine / "guide.md").read_bytes() == raw
    install_from_quarantine(quarantine, "demo", "", bundle, scan)
    assert (path / "guide.md").read_bytes() == raw
    entry = hub.HubLockFile().get_installed("demo")
    assert entry is not None and entry["metadata"]["source_blob_hashes"]["guide.md"] == blob_hash(raw)


def test_pinned_materialization_positive_unicode_inventory(installed):
    path, bundle, _source, _console = installed
    bundle.files["e\u0301/資料.md"] = b"Unicode\n"
    bundle.metadata["source_blob_hashes"]["e\u0301/資料.md"] = blob_hash(b"Unicode\n")
    scan = ScanResult(skill_name="demo", source="community", trust_level="community", verdict="safe")
    install_from_quarantine(quarantine_bundle(bundle), "demo", "", bundle, scan)
    assert (path / "e\u0301/資料.md").read_bytes() == b"Unicode\n"
    entry = hub.HubLockFile().get_installed("demo")
    assert entry is not None
    assert set(entry["files"]) == set(bundle.files)
    assert entry["metadata"] == bundle.metadata


PIN_FIELDS = ("requested_revision", "resolved_commit", "source_tree", "source_blob_hashes")


@pytest.mark.parametrize("field", PIN_FIELDS)
@pytest.mark.parametrize("damage", ["missing", "null", "type"])
@pytest.mark.parametrize("action", ["check", "update"])
def test_partial_pin_refused_before_any_source_operation(installed, monkeypatch, field, damage, action):
    path, _bundle, source, console = installed
    lock = hub.HubLockFile()
    data = lock.load()
    metadata = data["installed"]["demo"]["metadata"]
    if damage == "missing":
        del metadata[field]
    else:
        metadata[field] = None if damage == "null" else ["invalid"]
    lock.save(data)
    before = hub.LOCK_FILE.read_bytes(), content_hash(path)
    import tools.skills_hub_search as search
    monkeypatch.setattr(search, "create_source_router", lambda **_: [source])
    if action == "check":
        row = check_for_skill_updates(sources=[source])[0]
        assert row["status"] == "unavailable" and row.get("error")
    else:
        assert cli.do_update("demo", console=console, force=True) is False
    source.current_revision.assert_not_called()
    source.fetch.assert_not_called()
    source.inspect.assert_not_called()
    assert (hub.LOCK_FILE.read_bytes(), content_hash(path)) == before


@pytest.mark.parametrize("damage", ["blob_value", "blob_inventory", "files", "source_revision"])
@pytest.mark.parametrize("action", ["check", "update"])
def test_registered_pin_corruption_not_replaced_by_fetched_metadata(installed, monkeypatch, damage, action):
    path, _bundle, source, console = installed
    lock = hub.HubLockFile()
    data = lock.load()
    entry = data["installed"]["demo"]
    if damage == "blob_value":
        entry["metadata"]["source_blob_hashes"]["guide.md"] = "c" * 40
    elif damage == "blob_inventory":
        del entry["metadata"]["source_blob_hashes"]["guide.md"]
    elif damage == "files":
        entry["files"] = ["SKILL.md", "SKILL.md"]
    else:
        entry["metadata"]["source_revision"] = COMMIT
    lock.save(data)
    before = hub.LOCK_FILE.read_bytes(), content_hash(path)
    import tools.skills_hub_search as search
    monkeypatch.setattr(search, "create_source_router", lambda **_: [source])
    if action == "check":
        row = check_for_skill_updates(sources=[source])[0]
        assert row["status"] == "unavailable" and row.get("error")
    else:
        assert cli.do_update("demo", console=console, force=True) is False
    source.current_revision.assert_not_called()
    source.inspect.assert_not_called()
    assert (hub.LOCK_FILE.read_bytes(), content_hash(path)) == before


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("floating", [False, True])
@pytest.mark.parametrize("overwrite", [False, True])
def test_real_fifo_never_invisible_or_destroyed(installed, monkeypatch, floating, overwrite):
    import os
    import stat
    from tools.skills_hub_install import installed_skill_matches
    path, bundle, source, console = installed
    if floating:
        lock = hub.HubLockFile()
        data = lock.load()
        data["installed"]["demo"]["metadata"] = {"source_revision": TREE}
        lock.save(data)
        source.fetch.return_value = SkillBundle("demo", {"SKILL.md": b"remote change"},
                                               "github", IDENTIFIER, "community")
        source.current_revision.return_value = "c" * 40
    os.mkfifo(path / "special")
    (path / "guide.md").write_bytes(b"ordinary local edit")
    entry = hub.HubLockFile().get_installed("demo")
    assert not installed_skill_matches(entry, path)
    import tools.skills_hub_search as search
    monkeypatch.setattr(search, "create_source_router", lambda **_: [source])
    before = hub.LOCK_FILE.read_bytes(), (path / "guide.md").read_bytes()
    assert check_for_skill_updates(sources=[source])[0]["status"] == "invalid_install"
    assert cli.do_update("demo", console=console, force=overwrite) is False
    assert stat.S_ISFIFO((path / "special").lstat().st_mode)
    assert (hub.LOCK_FILE.read_bytes(), (path / "guide.md").read_bytes()) == before
    scan = ScanResult(skill_name="demo", source="community", trust_level="community", verdict="safe")
    with pytest.raises(ValueError, match="non-regular"):
        install_from_quarantine(quarantine_bundle(bundle), "demo", "", bundle, scan)
    assert stat.S_ISFIFO((path / "special").lstat().st_mode)


@pytest.mark.parametrize("phase", ["initial_move", "new_move", "record", "audit"])
@pytest.mark.parametrize("after", [False, True])
@pytest.mark.parametrize("interrupt", [False, True])
def test_transaction_fault_restores_exact_previous_state(installed, monkeypatch, phase, after, interrupt):
    import tools.skills_hub_install as install
    path, bundle, _source, _console = installed
    before = hub.LOCK_FILE.read_bytes(), content_hash(path)
    bundle.files = {"SKILL.md": b"replacement"}
    # Transaction faults require a coherent candidate; they do not test pin corruption.
    bundle.metadata["source_blob_hashes"] = {"SKILL.md": blob_hash(b"replacement")}
    quarantine = quarantine_bundle(bundle)
    error = KeyboardInterrupt if interrupt else OSError
    real_move, real_record, real_audit = install.shutil.move, hub.HubLockFile.record_install, hub.append_audit_log
    calls = 0

    def move(src, dest, *args, **kwargs):
        nonlocal calls
        calls += 1
        target = 1 if phase == "initial_move" else 2 if phase == "new_move" else -1
        if calls == target:
            if after:
                real_move(src, dest, *args, **kwargs)
            raise error("publication fault")
        return real_move(src, dest, *args, **kwargs)

    def record(self, **kwargs):
        if phase == "record":
            if after:
                real_record(self, **kwargs)
            raise error("publication fault")
        return real_record(self, **kwargs)

    def audit(*args, **kwargs):
        if phase == "audit":
            if after:
                real_audit(*args, **kwargs)
            raise error("publication fault")
        return real_audit(*args, **kwargs)

    monkeypatch.setattr(install.shutil, "move", move)
    monkeypatch.setattr(hub.HubLockFile, "record_install", record)
    monkeypatch.setattr(hub, "append_audit_log", audit)
    scan = ScanResult(skill_name="demo", source="community", trust_level="community", verdict="safe")
    with pytest.raises(error, match="publication fault"):
        install_from_quarantine(quarantine, "demo", "", bundle, scan)
    assert (hub.LOCK_FILE.read_bytes(), content_hash(path)) == before


@pytest.mark.parametrize("phase", ["cleanup", "restore"])
@pytest.mark.parametrize("interrupt", [False, True])
@pytest.mark.parametrize("after", [False, True])
def test_recovery_failure_retains_previous_and_exact_lock_backup(installed, monkeypatch, phase, interrupt, after):
    from pathlib import Path
    import tools.skills_hub_install as install
    path, bundle, _source, _console = installed
    before_lock, before_hash = hub.LOCK_FILE.read_bytes(), content_hash(path)
    bundle.files = {"SKILL.md": b"replacement"}
    # Transaction faults require a coherent candidate; they do not test pin corruption.
    bundle.metadata["source_blob_hashes"] = {"SKILL.md": blob_hash(b"replacement")}
    quarantine = quarantine_bundle(bundle)
    real_move, real_rmtree = install.shutil.move, install.shutil.rmtree
    error = KeyboardInterrupt if interrupt else OSError

    def fail_record(self, **kwargs):
        self.path.write_bytes(b"partial")
        raise OSError("publication fault")

    def move(src, dest, *args, **kwargs):
        if phase == "restore" and Path(src).name == "previous":
            if after:
                real_move(src, dest, *args, **kwargs)
            raise error("recovery fault")
        return real_move(src, dest, *args, **kwargs)

    def rmtree(target, *args, **kwargs):
        if phase == "cleanup" and Path(target) == path:
            if after:
                real_rmtree(target, *args, **kwargs)
            raise error("recovery fault")
        return real_rmtree(target, *args, **kwargs)

    monkeypatch.setattr(hub.HubLockFile, "record_install", fail_record)
    monkeypatch.setattr(install.shutil, "move", move)
    monkeypatch.setattr(install.shutil, "rmtree", rmtree)
    scan = ScanResult(skill_name="demo", source="community", trust_level="community", verdict="safe")
    with pytest.raises(RuntimeError, match="Recovery incomplete.*publish-"):
        install_from_quarantine(quarantine, "demo", "", bundle, scan)
    snapshots = list(hub.QUARANTINE_DIR.glob("publish-*/lock.before"))
    assert len(snapshots) == 1 and snapshots[0].read_bytes() == before_lock
    previous = path if phase == "restore" and after else snapshots[0].parent / "previous"
    assert content_hash(previous) == before_hash
    assert hub.LOCK_FILE.read_bytes() == before_lock


@pytest.mark.parametrize("bad", [123, True, "invalid", {}, []])
@pytest.mark.parametrize("field", PIN_FIELDS)
@pytest.mark.parametrize("action", ["check", "update"])
def test_pin_malformed_types_have_explicit_errors(installed, monkeypatch, field, bad, action):
    path, _bundle, source, console = installed
    lock = hub.HubLockFile()
    data = lock.load()
    data["installed"]["demo"]["metadata"][field] = bad
    lock.save(data)
    before = hub.LOCK_FILE.read_bytes(), content_hash(path)
    import tools.skills_hub_search as search
    monkeypatch.setattr(search, "create_source_router", lambda **_: [source])
    if action == "check":
        row = check_for_skill_updates(sources=[source])[0]
        assert row["status"] == "unavailable" and row.get("error")
    else:
        assert cli.do_update("demo", console=console, force=True) is False
    source.fetch.assert_not_called()
    source.current_revision.assert_not_called()
    assert (hub.LOCK_FILE.read_bytes(), content_hash(path)) == before


def test_authentic_legacy_has_only_tree_revision_and_remains_floating(installed):
    _path, _bundle, source, _console = installed
    lock = hub.HubLockFile()
    data = lock.load()
    data["installed"]["demo"]["metadata"] = {"source_revision": TREE}
    lock.save(data)
    source.current_revision.return_value = TREE
    assert check_for_skill_updates(sources=[source])[0]["status"] == "up_to_date"
    source.current_revision.assert_called_once_with(IDENTIFIER)
    source.fetch.assert_not_called()


def test_lock_restore_failure_retains_snapshot_even_after_files_restore(installed, monkeypatch):
    from pathlib import Path
    import tools.skills_hub_install as install
    path, bundle, _source, _console = installed
    before_lock, before_hash = hub.LOCK_FILE.read_bytes(), content_hash(path)
    real_write = Path.write_bytes
    bundle.files = {"SKILL.md": b"replacement"}
    # Transaction faults require a coherent candidate; they do not test pin corruption.
    bundle.metadata["source_blob_hashes"] = {"SKILL.md": blob_hash(b"replacement")}
    quarantine = quarantine_bundle(bundle)

    def record(self, **_kwargs):
        real_write(self.path, b"partial")
        raise OSError("publication fault")

    def write(target, body):
        if target == hub.LOCK_FILE and body == before_lock:
            raise OSError("lock restore fault")
        return real_write(target, body)

    monkeypatch.setattr(hub.HubLockFile, "record_install", record)
    monkeypatch.setattr(Path, "write_bytes", write)
    scan = ScanResult(skill_name="demo", source="community", trust_level="community", verdict="safe")
    with pytest.raises(RuntimeError, match="Recovery incomplete"):
        install_from_quarantine(quarantine, "demo", "", bundle, scan)
    assert content_hash(path) == before_hash
    snapshots = list(hub.QUARANTINE_DIR.glob("publish-*/lock.before"))
    assert len(snapshots) == 1 and snapshots[0].read_bytes() == before_lock


def test_postcommit_cleanup_failure_is_not_false_success(installed, monkeypatch):
    from pathlib import Path
    import tools.skills_hub_install as install
    path, bundle, _source, _console = installed
    before_lock, before_hash = hub.LOCK_FILE.read_bytes(), content_hash(path)
    bundle.files = {"SKILL.md": b"replacement"}
    # Transaction faults require a coherent candidate; they do not test pin corruption.
    bundle.metadata["source_blob_hashes"] = {"SKILL.md": blob_hash(b"replacement")}
    quarantine = quarantine_bundle(bundle)
    real_rmtree = install.shutil.rmtree

    def rmtree(target, *args, **kwargs):
        if Path(target).name.startswith("publish-"):
            raise OSError("committed cleanup failure")
        return real_rmtree(target, *args, **kwargs)

    monkeypatch.setattr(install.shutil, "rmtree", rmtree)
    scan = ScanResult(skill_name="demo", source="community", trust_level="community", verdict="safe")
    with pytest.raises(RuntimeError, match="Publication committed.*publish-"):
        install_from_quarantine(quarantine, "demo", "", bundle, scan)
    assert (path / "SKILL.md").read_bytes() == b"replacement"
    assert hub.HubLockFile().get_installed("demo")["content_hash"] == content_hash(path)
    backups = list(hub.QUARANTINE_DIR.glob("publish-*/previous"))
    assert len(backups) == 1 and content_hash(backups[0]) == before_hash
    assert (backups[0].parent / "lock.before").read_bytes() == before_lock
