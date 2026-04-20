"""Tests for hermes_office.models."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from hermes_office.models import (
    ACTIVITY_TO_ZONE,
    Activity,
    AvatarStyle,
    Department,
    Employee,
    Task,
    Zone,
    zone_for,
)


def test_activity_to_zone_total():
    """Every Activity must map to exactly one Zone (no Optional)."""
    for a in Activity:
        assert a in ACTIVITY_TO_ZONE
        assert isinstance(ACTIVITY_TO_ZONE[a], Zone)


def test_zone_for_helper():
    assert zone_for(Activity.WORKING) is Zone.WORK
    assert zone_for(Activity.RESTING) is Zone.REST
    assert zone_for(Activity.OFFLINE) is Zone.REST
    assert zone_for(Activity.LEARNING) is Zone.LEARN
    assert zone_for(Activity.TALKING) is Zone.TALK


def test_avatar_validates_sprite_id():
    with pytest.raises(ValidationError):
        AvatarStyle(sprite_id="cthulhu")
    AvatarStyle(sprite_id="cat", hue=0)
    AvatarStyle(sprite_id="cat", hue=359)
    with pytest.raises(ValidationError):
        AvatarStyle(sprite_id="cat", hue=400)


def test_employee_id_format():
    dept = Department(name="d", color="#ffffff")
    emp = Employee(department_id=dept.id, name="A", model="m")
    assert emp.id.startswith("emp_")
    assert len(emp.id) >= len("emp_") + 6
    with pytest.raises(ValidationError):
        Employee(id="bad-id", department_id=dept.id, name="A", model="m")


def test_department_color_must_be_hex():
    Department(name="ok", color="#abcdef")
    with pytest.raises(ValidationError):
        Department(name="bad", color="orange")


def test_task_default_status():
    t = Task(text="do thing")
    assert t.status == "queued"
    assert t.id.startswith("task_")


def test_employee_round_trip_json():
    dept = Department(name="x", color="#000000")
    emp = Employee(
        department_id=dept.id,
        name="Bob",
        role="Coder",
        model="gemma4-e2b-hermes",
        enabled_toolsets=["web", "file"],
        skills=["research/arxiv"],
    )
    j = emp.model_dump_json()
    parsed = Employee.model_validate(json.loads(j))
    assert parsed == emp
    bumped = emp.with_revision_bumped()
    assert bumped.revision == emp.revision + 1


def test_employee_long_name_rejected():
    dept = Department(name="x", color="#000000")
    with pytest.raises(ValidationError):
        Employee(department_id=dept.id, name="x" * 100, model="m")
