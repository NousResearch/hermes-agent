from __future__ import annotations

import json
from pathlib import Path

import pytest

from plugins.study_os.application import StudyOSApplication
from plugins.study_os.contract_models import (
    StudyProject,
    StudySchedule,
    study_contract_json_schema,
    study_project_id_json_schema,
    study_project_tool_properties,
    study_schedule_json_schema,
)
from plugins.study_os.dashboard.plugin_api import router
from plugins.study_os.learning import STUDY_ACTIVITY_SCHEMA, STUDY_COACH_SCHEMA
from plugins.study_os.schemas import validate_study_project, validate_study_schedule
from plugins.study_os.scripts.generate_contracts import generated_files
from plugins.study_os.tools import STUDY_PROJECT_SCHEMA, STUDY_SCHEDULE_SCHEMA


FIXTURES_PATH = (
    Path(__file__).resolve().parents[2]
    / "plugins"
    / "study_os"
    / "contracts"
    / "fixtures.json"
)
FIXTURES = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "case",
    FIXTURES["cases"],
    ids=[case["name"] for case in FIXTURES["cases"]],
)
def test_python_contract_matches_shared_fixture(case: dict):
    validator = (
        validate_study_project
        if case["kind"] == "project"
        else validate_study_schedule
    )

    ok, _value_or_errors = validator(case["data"])

    assert ok is case["valid"]


@pytest.mark.parametrize(
    "case",
    FIXTURES["relationships"],
    ids=[case["name"] for case in FIXTURES["relationships"]],
)
def test_application_owns_project_schedule_relationships(case: dict):
    errors = StudyOSApplication.validate_schedule_relationships(
        case["project"], case["schedule"]
    )

    assert (not errors) is case["valid"]


def test_generated_contracts_are_current():
    stale = [
        path
        for path, expected in generated_files().items()
        if not path.exists() or path.read_text(encoding="utf-8") != expected
    ]

    assert stale == []


def test_contract_schema_reuses_project_and_schedule_fields_for_tools():
    schema = study_contract_json_schema()
    project_properties = study_project_tool_properties()
    schedule_schema = study_schedule_json_schema()

    assert "StudyProjectV1" in schema["$defs"]
    assert project_properties["project_id"]["pattern"].startswith("^[a-z0-9]")
    assert schedule_schema["properties"]["events"]["items"]["properties"][
        "duration_minutes"
    ]["maximum"] == 720

    project_tool_properties = STUDY_PROJECT_SCHEMA["parameters"]["properties"]
    schedule_tool_data = STUDY_SCHEDULE_SCHEMA["parameters"]["properties"]["data"]
    assert project_tool_properties["project_id"] == project_properties["project_id"]
    assert schedule_tool_data["properties"]["events"]["items"]["properties"][
        "duration_minutes"
    ]["maximum"] == 720

    project_id_schema = study_project_id_json_schema()
    for model_tool in (STUDY_ACTIVITY_SCHEMA, STUDY_COACH_SCHEMA):
        assert model_tool["parameters"]["properties"]["project_id"] == project_id_schema


def test_fastapi_project_and_schedule_routes_use_canonical_models():
    routes = {getattr(route, "path", ""): route for route in router.routes}

    assert routes["/projects/{project_id}"].response_model is StudyProject
    assert (
        routes["/projects/{project_id}/schedules/{schedule_id}"].response_model
        is StudySchedule
    )
