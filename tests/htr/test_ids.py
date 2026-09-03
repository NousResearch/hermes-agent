from htr.ids import (
    ID_PREFIXES,
    generate_id,
    new_run_id,
    parse_id,
    validate_id,
)


def test_generate_id_has_expected_prefix_and_shape():
    run_id = generate_id("run")
    assert run_id.startswith("run_")
    parsed = parse_id(run_id)
    assert parsed is not None
    assert parsed[0] == "run"
    assert len(parsed[1]) == 8
    assert len(parsed[2]) == 6


def test_project_id_kind_is_valid():
    from htr.ids import generate_project_id

    project_id = generate_project_id()
    assert project_id.startswith("prj_")
    assert validate_id(project_id, "project")
    assert not validate_id(project_id, "run")


def test_all_prefixes_are_unique():
    prefixes = set(ID_PREFIXES.values())
    assert len(prefixes) == len(ID_PREFIXES)


def test_validate_id_accepts_matching_kind():
    run_id = new_run_id()
    assert validate_id(run_id, "run")
    assert not validate_id(run_id, "task")


def test_generated_ids_are_unique():
    ids = {generate_id("event") for _ in range(100)}
    assert len(ids) == 100
