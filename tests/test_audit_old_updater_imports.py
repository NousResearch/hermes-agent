"""Exercise updater archaeology against real Git DAGs, not today's filenames."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def audit(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "audit_old_updater_history_tests",
        Path(__file__).resolve().parents[1] / "scripts/audit-old-updater-imports.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    git(tmp_path, "init", "-b", "main")
    git(tmp_path, "config", "user.name", "Audit Test")
    git(tmp_path, "config", "user.email", "audit@example.invalid")
    git(tmp_path, "config", "commit.gpgsign", "false")
    return module


def git(root, *args):
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()


def put(root, path, source):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")


def commit(root, message):
    git(root, "add", ".")
    git(root, "commit", "-m", message)
    sha = git(root, "rev-parse", "HEAD")
    git(root, "update-ref", "refs/remotes/origin/main", sha)
    return sha


def test_discovers_entrypoint_renames_extractions_and_deleted_helpers(audit):
    root = audit.REPO_ROOT
    put(root, "hermes_cli/ancient.py", "def cmd_update():\n    from hermes_constants import first\n")
    first = commit(root, "first updater at an unknown address")
    git(root, "mv", "hermes_cli/ancient.py", "hermes_cli/briefly.py")
    renamed = commit(root, "rename without editing the blob")
    put(root, "hermes_cli/briefly.py", "from hermes_cli.gone import phase\ndef cmd_update():\n    phase()\n")
    put(root, "hermes_cli/gone.py", "def phase():\n    from hermes_constants import extracted\n")
    extracted = commit(root, "extract a helper under an unrelated filename")
    git(root, "mv", "hermes_cli/gone.py", "hermes_cli/later.py")
    put(root, "hermes_cli/briefly.py", "from hermes_cli.later import phase\ndef cmd_update():\n    phase()\n")
    commit(root, "rename the extracted helper")
    git(root, "rm", "hermes_cli/briefly.py", "hermes_cli/later.py")
    put(root, "hermes_cli/update_cmd.py", "def cmd_update():\n    from hermes_constants import newest\n")
    commit(root, "delete every historical filename")

    surface = audit.audit_history()
    assert {("hermes_constants", s) for s in ("first", "extracted", "newest")} <= surface.required.keys()
    assert {
        "hermes_cli/ancient.py:cmd_update", "hermes_cli/briefly.py:cmd_update"
    } <= surface.sites[("hermes_constants", "first")]
    assert first[:12] in surface.required[("hermes_constants", "first")]
    assert renamed[:12] in surface.required[("hermes_constants", "first")]
    assert extracted[:12] in surface.required[("hermes_constants", "extracted")]
    assert {
        "hermes_cli/gone.py:phase", "hermes_cli/later.py:phase"
    } <= surface.sites[("hermes_constants", "extracted")]
    assert surface.stats["commits"] == int(git(root, "rev-list", "--count", "origin/main"))
    assert surface.stats["complete_history"] is True
    assert surface.stats["history_ref"] == git(root, "rev-parse", "origin/main")
    assert surface.stats["roots"] == [first]
    assert {"hermes_cli/ancient.py", "hermes_cli/briefly.py"} <= set(surface.stats["entrypoint_paths"])
    assert {"hermes_cli/gone.py", "hermes_cli/later.py"} <= set(surface.stats["files_analyzed"])
    assert not any(root.glob("hermes_cli/*gone*"))
    assert audit.audit_tree().required.keys() == {("hermes_constants", "newest")}


def test_all_merge_parents_and_merge_result_but_not_unmerged_branches(audit):
    root = audit.REPO_ROOT
    path = "hermes_cli/prehistoric.py"
    put(root, path, "def cmd_update():\n    pass\n")
    base = commit(root, "base")
    git(root, "checkout", "-b", "side")
    put(root, path, "def cmd_update():\n    from hermes_constants import side_only\n")
    side = commit(root, "side updater")
    git(root, "checkout", "main")
    put(root, path, "def cmd_update():\n    from hermes_constants import main_only\n")
    commit(root, "main updater")
    # An ours merge is TREESAME to main. Ordinary path-limited git log
    # simplifies away the side parent and its real shipped revision.
    git(root, "merge", "-s", "ours", "--no-ff", "side", "-m", "discard side changes")
    commit_sha = git(root, "rev-parse", "HEAD")
    git(root, "update-ref", "refs/remotes/origin/main", commit_sha)
    assert side[:12] in audit.audit_history().required[("hermes_constants", "side_only")]
    git(root, "checkout", "-b", "other", base)
    put(root, "unrelated.txt", "new branch\n")
    commit(root, "unrelated change")
    git(root, "checkout", "main")
    git(root, "merge", "--no-ff", "--no-commit", "other")
    put(root, path, "def cmd_update():\n    from hermes_constants import merge_only\n")
    merged = commit(root, "updater code created only by merge resolution")
    git(root, "checkout", "-b", "unshipped", base)
    put(root, path, "def cmd_update():\n    from hermes_constants import never_shipped\n")
    commit(root, "unmerged branch")
    git(root, "update-ref", "refs/remotes/origin/main", merged)

    surface = audit.audit_history()
    assert {s for m, s in surface.required if m == "hermes_constants"} == {
        "side_only", "main_only", "merge_only"
    }
    assert merged[:12] in surface.required[("hermes_constants", "merge_only")]
    assert surface.stats["commits"] == int(git(root, "rev-list", "--count", "origin/main"))


def test_deleted_siblings_and_renamed_seed_helper_are_not_tied_to_head(audit):
    root = audit.REPO_ROOT
    put(root, "hermes_cli/update_cmd.py", "def cmd_update():\n    pass\n")
    put(root, "hermes_cli/old_support.py", "def old_phase():\n    from hermes_constants import before_rename\n")
    commit(root, "support before its known name")
    git(root, "mv", "hermes_cli/old_support.py", "hermes_cli/post_update.py")
    put(root, "hermes_cli/update_cmd_removed.py", "def phase():\n    from hermes_constants import deleted_sibling\n")
    commit(root, "known helper and temporary extraction")
    git(root, "rm", "hermes_cli/post_update.py", "hermes_cli/update_cmd_removed.py")
    commit(root, "remove both helpers")
    surface = audit.audit_history()
    assert {("hermes_constants", s) for s in ("before_rename", "deleted_sibling")} <= surface.required.keys()
    assert "hermes_cli/old_support.py:old_phase" in surface.sites[("hermes_constants", "before_rename")]
    assert not audit.audit_tree().required


def test_shallow_clone_refuses_freeze_without_overwriting_and_tree_still_works(audit, tmp_path, monkeypatch, capsys):
    root = audit.REPO_ROOT
    put(root, "hermes_cli/unknown.py", "def cmd_update():\n    from hermes_constants import old\n")
    commit(root, "historical contract")
    put(root, "hermes_cli/unknown.py", "def cmd_update():\n    from hermes_constants import current\n")
    commit(root, "current contract")
    clone = tmp_path / "shallow"
    git(root, "clone", "--depth=1", root.as_uri(), str(clone))
    monkeypatch.setattr(audit, "REPO_ROOT", clone)
    assert git(clone, "rev-parse", "--is-shallow-repository") == "true"
    with pytest.raises(audit.AuditError, match="shallow"):
        audit.audit_history()
    assert set(audit.audit_tree().required) == {("hermes_constants", "current")}
    output = clone / "frozen.json"
    output.write_text("do not truncate this", encoding="utf-8")
    with pytest.raises(SystemExit) as failure:
        audit.main(["--freeze", str(output)])
    assert failure.value.code != 0
    assert "--unshallow" in capsys.readouterr().err
    assert output.read_text() == "do not truncate this"


def test_union_keeps_old_and_uncommitted_loads_and_bare_wins(audit, tmp_path):
    root = audit.REPO_ROOT
    put(root, "hermes_cli/update_cmd.py", """def cmd_update():
    from hermes_constants import old_only, bare_then_guarded
    try:
        from hermes_constants import guarded_then_bare, always_guarded
    except ImportError:
        pass
""")
    commit(root, "not a tag, still a shipped updater")
    put(root, "hermes_cli/update_cmd.py", """def cmd_update():
    from hermes_constants import new_only, guarded_then_bare
    try:
        from hermes_constants import bare_then_guarded, always_guarded
    except ImportError:
        pass
""")
    output = tmp_path / "union.json"
    assert audit.main(["--freeze", str(output)]) == 0
    frozen = json.loads(output.read_text())
    assert set(frozen["bare"]) == {
        "hermes_constants::old_only", "hermes_constants::new_only",
        "hermes_constants::bare_then_guarded", "hermes_constants::guarded_then_bare",
    }
    assert frozen["guarded_only"] == ["hermes_constants::always_guarded"]
    assert frozen["stats"]["mode"] == "union"
    assert frozen["stats"]["history"]["complete_history"] is True


@pytest.mark.parametrize("malformed", [b"def cmd_update(:\n", b"def cmd_update():\n    # \xff\n    pass\n"])
def test_malformed_historical_entrypoint_fails_closed(audit, malformed):
    root = audit.REPO_ROOT
    target = root / "forgotten.py"
    target.write_bytes(malformed)
    commit(root, "broken historical source")
    put(root, "forgotten.py", "def cmd_update():\n    pass\n")
    commit(root, "repaired at tip")
    with pytest.raises(audit.AuditError, match="forgotten.py.*blob"):
        audit.audit_history()


def test_committed_merge_conflict_audits_both_arms_and_records_recovery(audit):
    root = audit.REPO_ROOT
    put(root, "old.py", """<<<<<<< HEAD
def cmd_update():
    from hermes_constants import ours
=======
def cmd_update():
    from hermes_constants import theirs
>>>>>>> incoming
""")
    commit(root, "accidentally committed conflict")
    put(root, "old.py", "def cmd_update():\n    pass\n")
    commit(root, "fix conflict")
    surface = audit.audit_history()
    assert {("hermes_constants", "ours"), ("hermes_constants", "theirs")} <= surface.required.keys()
    assert any("old.py" in recovery for recovery in surface.parse_recoveries)


def test_identical_blobs_have_path_specific_relative_imports_and_seed_roles(audit):
    root = audit.REPO_ROOT
    source = "def cmd_update():\n    from .constants import important\ndef unrelated():\n    from hermes_constants import helper_only\n"
    put(root, "hermes_cli/unknown.py", source)
    put(root, "hermes_cli/post_update.py", source)
    put(root, "pm/unknown.py", source)
    commit(root, "identical source, distinct modules and seed modes")
    surface = audit.audit_history()
    assert ("hermes_cli.constants", "important") in surface.required
    assert ("pm.constants", "important") in surface.required
    assert surface.sites[("hermes_constants", "helper_only")] == {"hermes_cli/post_update.py:unrelated"}


def test_only_swallowed_matching_load_failures_are_guarded(audit):
    root = audit.REPO_ROOT
    put(root, "hermes_cli/update_cmd.py", """def cmd_update():
    import sys
    try:
        from hermes_constants import wrong_exception
    except AttributeError:
        pass
    try:
        from hermes_constants import name_not_module
    except ModuleNotFoundError:
        pass
    try:
        from hermes_constants import reraised
    except ImportError:
        raise
    try:
        module = sys.modules.get("hermes_constants")
        getattr(module, "optional")
    except AttributeError:
        pass
""")
    commit(root, "exception boundaries")
    surface = audit.audit_history()
    assert surface.guarded_only == {("hermes_constants", "optional")}
    assert {("hermes_constants", s) for s in ("wrong_exception", "name_not_module", "reraised")} <= surface.required.keys()


def test_lazy_module_facade_follows_called_helper_not_unrelated_commands(audit):
    root = audit.REPO_ROOT
    put(root, "hermes_cli/entry.py", """def _m():
    from hermes_cli import main
    return main

def cmd_update():
    _m().phase()
""")
    put(root, "hermes_cli/main.py", """from hermes_cli.gone import phase as phase

def unrelated_command():
    from hermes_constants import not_an_updater_load
""")
    put(root, "hermes_cli/gone.py", "def phase():\n    from hermes_constants import actual_helper_load\n")
    commit(root, "lazy facade and re-exported helper")
    surface = audit.audit_history()
    assert ("hermes_constants", "actual_helper_load") in surface.required
    assert ("hermes_constants", "not_an_updater_load") not in surface.required


def test_resolver_requires_a_module_binding_not_a_nested_name(audit):
    root = audit.REPO_ROOT
    put(root, "hermes_constants.py", """def owner():
    from pathlib import Path as local_import
    local_assignment = 1
    def local_function():
        pass

class Owner:
    def method(self):
        pass

if True:
    from pathlib import Path as exported
    value = 1
""")
    for symbol in ("local_import", "local_assignment", "local_function", "method"):
        assert not audit.resolve_in_tree("hermes_constants", symbol, root)[0], symbol
    for symbol in ("owner", "Owner", "exported", "value"):
        assert audit.resolve_in_tree("hermes_constants", symbol, root)[0], symbol


def test_same_named_functions_keep_requirements_from_every_definition(audit):
    result = audit.analyse("""class First:
    def phase(self):
        from hermes_constants import first

class Second:
    def phase(self):
        from hermes_constants import second

if True:
    def platform_phase():
        from hermes_constants import platform_a
else:
    def platform_phase():
        from hermes_constants import platform_b

def cmd_update():
    Second().phase()
    platform_phase()
""", "hermes_cli/update_cmd.py", entrypoints=True)
    assert {r.symbol for r in result.requirements} == {
        "first", "second", "platform_a", "platform_b",
    }


def test_reviewed_dynamic_loads_survive_union_regeneration(audit, monkeypatch):
    root = audit.REPO_ROOT
    put(root, "hermes_cli/update_cmd.py", "def cmd_update():\n    from hermes_constants import anchor\n")
    witness = commit(root, "module-object call manually reviewed")
    monkeypatch.setattr(audit, "REVIEWED_DYNAMIC_LOADS", (
        ("hermes_constants", "dynamic", witness[:12], "hermes_cli/update_cmd.py:cmd_update"),
        ("hermes_constants", "unshipped", "not-a-witness", "other.py:cmd_update"),
    ))
    surface = audit.audit_union()
    assert ("hermes_constants", "dynamic") in surface.required
    assert ("hermes_constants", "dynamic") not in surface.guarded_only
    assert ("hermes_constants", "unshipped") not in surface.required
