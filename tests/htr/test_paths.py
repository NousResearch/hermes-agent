import pytest

from htr.ids import new_attempt_id, new_run_id, new_task_id
from htr import paths


def test_run_root_uses_base_dir_override(tmp_path):
    run_id = new_run_id()
    root = paths.run_root(run_id, base_dir=tmp_path)
    assert root == tmp_path / run_id


def test_attempt_dir_layout(tmp_path):
    run_id = new_run_id()
    task_id = new_task_id()
    attempt_id = new_attempt_id()

    attempt = paths.attempt_dir(run_id, task_id, attempt_id, base_dir=tmp_path)
    assert attempt.name == attempt_id
    assert paths.result_json_path(run_id, task_id, attempt_id, base_dir=tmp_path) == (
        attempt / "output" / "result.json"
    )


@pytest.mark.parametrize(
    "bad_id",
    ["../evil", "run_bad", "run_20260718", "run_20260718_nothex"],
)
def test_invalid_run_id_rejected(tmp_path, bad_id):
    with pytest.raises(ValueError):
        paths.run_root(bad_id, base_dir=tmp_path)


def test_path_traversal_in_task_id_rejected(tmp_path):
    run_id = new_run_id()
    with pytest.raises(ValueError):
        paths.task_dir(run_id, "../escape", base_dir=tmp_path)


def test_project_registry_paths_use_hermes_home(tmp_path):
    from htr.ids import generate_project_id

    project_id = generate_project_id()
    root = paths.project_registry_root(tmp_path)
    assert root == tmp_path / ".htr" / "project_registry"
    assert paths.project_record_path(project_id, tmp_path) == (
        root / "projects" / project_id / "record.json"
    )


def test_project_id_traversal_rejected(tmp_path):
    with pytest.raises(ValueError):
        paths.project_record_dir("../escape", tmp_path)
