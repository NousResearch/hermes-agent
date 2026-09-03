"""Task 30 — multi-project registry and isolation."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from htr import io, paths
from htr.ids import generate_project_id, new_run_id, validate_id
from htr.project_registry import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    ProjectIdentityConflict,
    ProjectInvalidInput,
    ProjectNotRegistered,
    ProjectPathConflict,
    ProjectPathEscape,
    ProjectRegistryCorrupt,
    ProjectRegistryFilesystemError,
    ProjectRegistrySchemaUnsupported,
    assert_path_in_project,
    canonicalize_runs_root,
    ensure_project_registry,
    get_project,
    list_projects,
    lookup_project_by_runs_root,
    register_project,
    resolve_invocation_runs_root,
    resolve_project_runs_root,
    update_project_metadata,
)


def _home(tmp_path: Path) -> Path:
    home = tmp_path / "hermes-home"
    home.mkdir()
    return home


def _runs_dir(tmp_path: Path, name: str = "runs") -> Path:
    root = tmp_path / name
    root.mkdir()
    return root.resolve()


def test_empty_registry_list_and_ensure(tmp_path):
    home = _home(tmp_path)
    assert list_projects(hermes_home=home) == []
    root = ensure_project_registry(hermes_home=home)
    assert root.is_dir()
    assert (root / "projects").is_dir()
    assert list_projects(hermes_home=home) == []


def test_register_first_project_and_read_back(tmp_path):
    home = _home(tmp_path)
    runs = _runs_dir(tmp_path)
    record = register_project(runs, display_name="Alpha", hermes_home=home)
    assert validate_id(record.project_id, "project")
    assert record.runs_root == runs
    assert record.status == "active"
    assert record.display_name == "Alpha"
    assert record.schema_version == SCHEMA_VERSION
    loaded = get_project(record.project_id, hermes_home=home)
    assert loaded == record
    listed = list_projects(hermes_home=home)
    assert [item.project_id for item in listed] == [record.project_id]


def test_register_multiple_projects_are_isolated(tmp_path):
    home = _home(tmp_path)
    runs_a = _runs_dir(tmp_path, "proj-a")
    runs_b = _runs_dir(tmp_path, "proj-b")
    project_a = register_project(runs_a, display_name="A", hermes_home=home)
    project_b = register_project(runs_b, display_name="B", hermes_home=home)
    (runs_a / "only-a.txt").write_text("a", encoding="utf-8")
    (runs_b / "only-b.txt").write_text("b", encoding="utf-8")

    assert resolve_project_runs_root(project_a.project_id, hermes_home=home) == runs_a
    assert resolve_project_runs_root(project_b.project_id, hermes_home=home) == runs_b
    assert assert_path_in_project(
        project_a.project_id, runs_a / "only-a.txt", hermes_home=home
    ) == (runs_a / "only-a.txt").resolve()
    with pytest.raises(ProjectPathEscape) as escaped:
        assert_path_in_project(project_a.project_id, runs_b / "only-b.txt", hermes_home=home)
    assert escaped.value.error_class == "path_escape"
    assert not (resolve_project_runs_root(project_a.project_id, hermes_home=home) / "only-b.txt").exists()


def test_reregister_same_identity_is_idempotent(tmp_path):
    home = _home(tmp_path)
    runs = _runs_dir(tmp_path)
    first = register_project(
        runs, project_id=None, display_name="First", hermes_home=home
    )
    record_path = paths.project_record_path(first.project_id, home)
    original = record_path.read_text(encoding="utf-8")
    replay = register_project(
        runs,
        project_id=first.project_id,
        display_name="Changed label is ignored on replay",
        hermes_home=home,
    )
    assert replay.project_id == first.project_id
    assert replay.project_identity_digest == first.project_identity_digest
    assert replay.display_name == "First"
    assert replay.created_at == first.created_at
    assert record_path.read_text(encoding="utf-8") == original


def test_project_id_conflict_different_path(tmp_path):
    home = _home(tmp_path)
    runs_a = _runs_dir(tmp_path, "a")
    runs_b = _runs_dir(tmp_path, "b")
    first = register_project(runs_a, hermes_home=home)
    with pytest.raises(ProjectIdentityConflict) as exc:
        register_project(runs_b, project_id=first.project_id, hermes_home=home)
    assert exc.value.error_class == "identity_conflict"


def test_project_path_conflict_different_id(tmp_path):
    home = _home(tmp_path)
    runs = _runs_dir(tmp_path)
    register_project(runs, hermes_home=home)
    other_id = generate_project_id()
    with pytest.raises(ProjectPathConflict) as exc:
        register_project(runs, project_id=other_id, hermes_home=home)
    assert exc.value.error_class == "path_conflict"


def test_nested_runs_root_is_path_conflict(tmp_path):
    home = _home(tmp_path)
    parent = _runs_dir(tmp_path, "parent")
    child = parent / "nested"
    child.mkdir()
    register_project(parent, hermes_home=home)
    with pytest.raises(ProjectPathConflict):
        register_project(child, hermes_home=home)


def test_relative_and_missing_paths_fail_closed(tmp_path, monkeypatch):
    home = _home(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()
    with pytest.raises(ProjectInvalidInput) as relative:
        register_project("runs", hermes_home=home)
    assert "absolute" in str(relative.value)
    missing = tmp_path / "does-not-exist"
    with pytest.raises(ProjectInvalidInput) as absent:
        register_project(missing, hermes_home=home)
    assert "does not exist" in str(absent.value)
    file_path = tmp_path / "not-a-dir"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(ProjectInvalidInput):
        register_project(file_path, hermes_home=home)


def test_cwd_is_not_used_for_identity(tmp_path, monkeypatch):
    home = _home(tmp_path)
    runs = _runs_dir(tmp_path, "real-runs")
    other = tmp_path / "other-cwd"
    other.mkdir()
    monkeypatch.chdir(other)
    record = register_project(runs, hermes_home=home)
    assert record.runs_root == runs
    assert record.path_comparison_key != os.path.normcase(str(other.resolve()))


def test_persistence_survives_new_reader(tmp_path):
    home = _home(tmp_path)
    runs = _runs_dir(tmp_path)
    created = register_project(runs, display_name="Persist", hermes_home=home)
    reread = get_project(created.project_id, hermes_home=home)
    assert reread.to_dict() == created.to_dict()
    assert lookup_project_by_runs_root(runs, hermes_home=home).project_id == created.project_id


def test_corrupt_record_fail_closed(tmp_path):
    home = _home(tmp_path)
    runs = _runs_dir(tmp_path)
    created = register_project(runs, hermes_home=home)
    paths.project_record_path(created.project_id, home).write_text("{not-json", encoding="utf-8")
    with pytest.raises(ProjectRegistryCorrupt) as exc:
        get_project(created.project_id, hermes_home=home)
    assert exc.value.error_class == "registry_corrupt"
    with pytest.raises(ProjectRegistryCorrupt):
        list_projects(hermes_home=home)


def test_unsupported_schema_version_fail_closed(tmp_path):
    home = _home(tmp_path)
    runs = _runs_dir(tmp_path)
    created = register_project(runs, hermes_home=home)
    payload = json.loads(paths.project_record_path(created.project_id, home).read_text(encoding="utf-8"))
    payload["schema_version"] = SCHEMA_VERSION + 1
    paths.project_record_path(created.project_id, home).write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ProjectRegistrySchemaUnsupported) as exc:
        get_project(created.project_id, hermes_home=home)
    assert exc.value.error_class == "schema_unsupported"


def test_update_failure_preserves_original_record(tmp_path):
    home = _home(tmp_path)
    runs = _runs_dir(tmp_path)
    created = register_project(runs, display_name="Keep", hermes_home=home)
    record_path = paths.project_record_path(created.project_id, home)
    original = record_path.read_text(encoding="utf-8")

    def _boom(*_args, **_kwargs):
        raise OSError("simulated write failure")

    with mock.patch("htr.project_registry.io.atomic_write_json", side_effect=_boom):
        with pytest.raises(ProjectRegistryFilesystemError) as exc:
            update_project_metadata(
                created.project_id, display_name="New", hermes_home=home
            )
    assert exc.value.error_class == "filesystem_error"
    assert record_path.read_text(encoding="utf-8") == original
    assert get_project(created.project_id, hermes_home=home).display_name == "Keep"


def test_concurrent_register_distinct_projects(tmp_path):
    home = _home(tmp_path)
    runs_a = _runs_dir(tmp_path, "ca")
    runs_b = _runs_dir(tmp_path, "cb")
    barrier = threading.Barrier(2)
    results: list[object] = []
    errors: list[BaseException] = []

    def _worker(root: Path) -> None:
        try:
            barrier.wait(timeout=5)
            results.append(register_project(root, hermes_home=home))
        except BaseException as exc:  # noqa: BLE001 — capture for assertion
            errors.append(exc)

    threads = [
        threading.Thread(target=_worker, args=(runs_a,)),
        threading.Thread(target=_worker, args=(runs_b,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert errors == []
    assert len(results) == 2
    ids = {item.project_id for item in results}
    assert len(ids) == 2
    listed = list_projects(hermes_home=home)
    assert {item.project_id for item in listed} == ids


def test_concurrent_register_same_identity_is_idempotent(tmp_path):
    home = _home(tmp_path)
    runs = _runs_dir(tmp_path)
    project_id = generate_project_id()
    barrier = threading.Barrier(2)
    results: list[object] = []
    errors: list[BaseException] = []

    def _worker() -> None:
        try:
            barrier.wait(timeout=5)
            results.append(
                register_project(runs, project_id=project_id, hermes_home=home)
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert errors == []
    assert len(results) == 2
    assert {item.project_id for item in results} == {project_id}
    assert {item.project_identity_digest for item in results} == {
        results[0].project_identity_digest
    }


def test_unregistered_and_invalid_id(tmp_path):
    home = _home(tmp_path)
    missing = generate_project_id()
    with pytest.raises(ProjectNotRegistered) as exc:
        get_project(missing, hermes_home=home)
    assert exc.value.error_class == "not_registered"
    with pytest.raises(ProjectInvalidInput):
        get_project("not-an-id", hermes_home=home)
    with pytest.raises(ProjectInvalidInput):
        get_project("../escape", hermes_home=home)


def test_legacy_single_project_workflow_without_registry(tmp_path):
    home = _home(tmp_path)
    runs = _runs_dir(tmp_path)
    run_id = new_run_id()
    root = io.create_run_workspace(run_id, base_dir=runs)
    assert root.is_dir()
    assert paths.run_manifest_path(run_id, runs).is_file()
    assert list_projects(hermes_home=home) == []
    assert lookup_project_by_runs_root(runs, hermes_home=home) is None
    assert resolve_invocation_runs_root(hermes_home=home) is None
    assert resolve_invocation_runs_root(runs_root=runs, hermes_home=home) == Path(runs)


def test_invocation_resolver_requires_matching_project_and_path(tmp_path):
    home = _home(tmp_path)
    runs = _runs_dir(tmp_path, "bound")
    other = _runs_dir(tmp_path, "other")
    record = register_project(runs, hermes_home=home)
    assert (
        resolve_invocation_runs_root(project_id=record.project_id, hermes_home=home)
        == runs
    )
    matched = resolve_invocation_runs_root(
        project_id=record.project_id,
        runs_root=runs,
        hermes_home=home,
    )
    assert matched == runs
    with pytest.raises(ProjectIdentityConflict):
        resolve_invocation_runs_root(
            project_id=record.project_id,
            runs_root=other,
            hermes_home=home,
        )


def test_update_metadata_and_archive(tmp_path):
    home = _home(tmp_path)
    runs = _runs_dir(tmp_path)
    created = register_project(runs, display_name="Label", hermes_home=home)
    updated = update_project_metadata(
        created.project_id, display_name="Relabel", hermes_home=home
    )
    assert updated.display_name == "Relabel"
    assert updated.project_identity_digest == created.project_identity_digest
    archived = update_project_metadata(
        created.project_id, status="archived", hermes_home=home
    )
    assert archived.status == "archived"
    assert archived.created_at == created.created_at
    assert archived.runs_root == created.runs_root
    assert archived.project_id == created.project_id
    assert get_project(created.project_id, hermes_home=home).status == "archived"
    assert list_projects(hermes_home=home) == []
    assert list_projects(hermes_home=home, include_archived=True)[0].status == "archived"
    with pytest.raises(ProjectInvalidInput):
        resolve_project_runs_root(created.project_id, hermes_home=home)
    other_id = generate_project_id()
    with pytest.raises(ProjectPathConflict):
        register_project(runs, project_id=other_id, hermes_home=home)


def test_symlink_alias_is_same_project_path(tmp_path):
    real = _runs_dir(tmp_path, "real")
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are not available")
    home = _home(tmp_path)
    first = register_project(alias, hermes_home=home)
    assert first.runs_root == real.resolve()
    with pytest.raises(ProjectPathConflict):
        register_project(real, project_id=generate_project_id(), hermes_home=home)


def test_path_traversal_rejected_in_registry_paths(tmp_path):
    with pytest.raises(ValueError):
        paths.project_record_dir("../escape", tmp_path)


def test_canonicalize_rejects_relative_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()
    with pytest.raises(ProjectInvalidInput):
        canonicalize_runs_root("runs")


def test_cli_project_list_and_register(tmp_path, capsys):
    from hermes_cli.htr import htr_command

    runs = _runs_dir(tmp_path)
    args = SimpleNamespace(
        htr_command="project",
        htr_project_command="register",
        runs_root=str(runs),
        project_id=None,
        display_name="CLI",
    )
    rc = htr_command(args)
    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["ok"] is True
    assert payload["project"]["schema"] == SCHEMA_NAME
    assert payload["project"]["display_name"] == "CLI"

    list_args = SimpleNamespace(
        htr_command="project",
        htr_project_command="list",
        include_archived=False,
    )
    rc = htr_command(list_args)
    listed = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert len(listed["projects"]) == 1


def test_cli_unknown_project_error_class(capsys):
    from hermes_cli.htr import htr_command

    args = SimpleNamespace(
        htr_command="project",
        htr_project_command="show",
        project_id=generate_project_id(),
    )
    rc = htr_command(args)
    payload = json.loads(capsys.readouterr().out)
    assert rc != 0
    assert payload["ok"] is False
    assert payload["error_class"] == "not_registered"


def test_string_prefix_paths_are_not_overlap(tmp_path):
    home = _home(tmp_path)
    shorter = _runs_dir(tmp_path, "proj")
    longer = _runs_dir(tmp_path, "project")
    first = register_project(shorter, hermes_home=home)
    second = register_project(longer, hermes_home=home)
    assert first.project_id != second.project_id
    listed = list_projects(hermes_home=home)
    assert {item.project_id for item in listed} == {first.project_id, second.project_id}


def test_parent_after_child_is_path_conflict(tmp_path):
    home = _home(tmp_path)
    parent = _runs_dir(tmp_path, "outer")
    child = parent / "inner"
    child.mkdir()
    register_project(child, hermes_home=home)
    with pytest.raises(ProjectPathConflict):
        register_project(parent, hermes_home=home)


def test_corrupt_top_level_array_and_field_types(tmp_path):
    home = _home(tmp_path)
    runs = _runs_dir(tmp_path)
    created = register_project(runs, hermes_home=home)
    record_path = paths.project_record_path(created.project_id, home)
    original = json.loads(record_path.read_text(encoding="utf-8"))

    record_path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(ProjectRegistryCorrupt, match="not a JSON object"):
        get_project(created.project_id, hermes_home=home)

    missing = dict(original)
    del missing["created_at"]
    record_path.write_text(json.dumps(missing, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ProjectRegistryCorrupt, match="created_at"):
        get_project(created.project_id, hermes_home=home)

    wrong_name = dict(original)
    wrong_name["display_name"] = 12
    record_path.write_text(json.dumps(wrong_name, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ProjectRegistryCorrupt, match="display_name"):
        list_projects(hermes_home=home)

    relative = dict(original)
    relative["runs_root"] = "relative/runs"
    record_path.write_text(json.dumps(relative, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ProjectRegistryCorrupt, match="not absolute"):
        get_project(created.project_id, hermes_home=home)


def test_update_rejects_invalid_status(tmp_path):
    home = _home(tmp_path)
    runs = _runs_dir(tmp_path)
    created = register_project(runs, hermes_home=home)
    with pytest.raises(ProjectInvalidInput, match="status"):
        update_project_metadata(created.project_id, status="deleted", hermes_home=home)
    assert get_project(created.project_id, hermes_home=home).status == "active"


def test_list_projects_sorts_by_project_id(tmp_path):
    home = _home(tmp_path)
    stamp = "20260824"
    ids = [f"prj_{stamp}_ffffff", f"prj_{stamp}_000000", f"prj_{stamp}_aaaaaa"]
    for index, project_id in enumerate(ids):
        register_project(
            _runs_dir(tmp_path, f"sort-{index}"),
            project_id=project_id,
            hermes_home=home,
        )
    listed = list_projects(hermes_home=home)
    assert [item.project_id for item in listed] == sorted(ids)


def test_concurrent_metadata_updates_leave_valid_record(tmp_path):
    home = _home(tmp_path)
    runs = _runs_dir(tmp_path)
    created = register_project(runs, display_name="Start", hermes_home=home)
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def _worker(label: str) -> None:
        try:
            barrier.wait(timeout=5)
            update_project_metadata(created.project_id, display_name=label, hermes_home=home)
        except BaseException as exc:  # noqa: BLE001 — capture for assertion
            errors.append(exc)

    threads = [
        threading.Thread(target=_worker, args=("Alpha",)),
        threading.Thread(target=_worker, args=("Beta",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert errors == []
    loaded = get_project(created.project_id, hermes_home=home)
    assert loaded.display_name in {"Alpha", "Beta"}
    assert loaded.project_id == created.project_id
    assert loaded.runs_root == created.runs_root
    assert loaded.created_at == created.created_at


def test_cli_update_unregistered_and_observe_project_id(tmp_path, capsys, monkeypatch):
    from hermes_cli import htr as cli_htr

    missing = SimpleNamespace(
        htr_command="project",
        htr_project_command="update",
        project_id=generate_project_id(),
        display_name="Nope",
        clear_display_name=False,
        status=None,
    )
    rc = cli_htr.htr_command(missing)
    payload = json.loads(capsys.readouterr().out)
    assert rc != 0
    assert payload["error_class"] == "not_registered"

    captured: dict[str, object] = {}

    def fake_snapshot(run_id, base_dir=None):
        captured["run_id"] = run_id
        captured["base_dir"] = base_dir
        return {"run_id": run_id, "integrity": {"status": "ok", "error_count": 0}}

    monkeypatch.setattr(cli_htr, "build_run_snapshot", fake_snapshot)
    monkeypatch.setattr(cli_htr, "compute_exit_code", lambda *_args, **_kwargs: 0)

    runs = _runs_dir(tmp_path)
    record = register_project(runs)
    observe_args = SimpleNamespace(
        htr_command="observe",
        run_id="run_20260824_aaaaaa",
        project_id=record.project_id,
        runs_root=None,
        summary=False,
        strict=False,
    )
    assert cli_htr.htr_command(observe_args) == 0
    capsys.readouterr()
    assert captured["base_dir"] == record.runs_root

    other = _runs_dir(tmp_path, "other-root")
    mismatch = SimpleNamespace(
        htr_command="observe",
        run_id="run_20260824_aaaaaa",
        project_id=record.project_id,
        runs_root=str(other),
        summary=False,
        strict=False,
    )
    mismatch_rc = cli_htr.htr_command(mismatch)
    mismatch_payload = json.loads(capsys.readouterr().out)
    assert mismatch_rc != 0
    assert mismatch_payload["error_class"] == "identity_conflict"

    legacy = SimpleNamespace(
        htr_command="observe",
        run_id="run_20260824_aaaaaa",
        project_id=None,
        runs_root=str(runs),
        summary=False,
        strict=False,
    )
    assert cli_htr.htr_command(legacy) == 0
    assert captured["base_dir"] == Path(str(runs))

    plan_captured: dict[str, object] = {}

    def fake_plan(snapshot, intent):
        plan_captured["base_dir"] = intent.htr_runs_root
        return {"ok": True}

    monkeypatch.setattr(cli_htr, "build_action_plan", fake_plan)
    monkeypatch.setattr(cli_htr, "compute_plan_exit_code", lambda *_args, **_kwargs: 0)
    plan_args = SimpleNamespace(
        htr_command="plan",
        run_id="run_20260824_aaaaaa",
        project_id=record.project_id,
        runs_root=None,
        inputs_file=None,
        action=None,
        project_checkpoint=None,
        remediation_intent=False,
        summary=False,
    )
    assert cli_htr.htr_command(plan_args) == 0
    assert plan_captured["base_dir"] == str(record.runs_root)
