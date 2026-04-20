"""Tests for the persistent JSON store + atomic writes + redaction."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_office.models import (
    ActivityEvent,
    Department,
    Employee,
    Task,
)
from hermes_office.store import (
    Store,
    atomic_write_json,
    atomic_write_text,
    redact_secrets,
)


# ── atomic write ────────────────────────────────────────────────────────────


def test_atomic_write_text_creates_dir(tmp_path: Path):
    target = tmp_path / "deep" / "nest" / "f.txt"
    atomic_write_text(target, "hello")
    assert target.read_text() == "hello"


def test_atomic_write_json_round_trip(tmp_path: Path):
    target = tmp_path / "data.json"
    atomic_write_json(target, {"a": [1, 2, 3]})
    assert json.loads(target.read_text()) == {"a": [1, 2, 3]}


def test_atomic_write_does_not_leave_temp_on_success(tmp_path: Path):
    target = tmp_path / "f.json"
    atomic_write_json(target, {"x": 1})
    leftovers = list(tmp_path.glob("*.tmp.*"))
    assert leftovers == []


def test_atomic_write_does_not_corrupt_on_failure(tmp_path: Path, monkeypatch):
    """If os.replace raises mid-write, the original file must be untouched."""
    target = tmp_path / "f.json"
    atomic_write_text(target, "v1")
    import os as real_os

    def boom(*a, **kw):
        raise OSError("boom")

    monkeypatch.setattr("hermes_office.store.os.replace", boom)
    with pytest.raises(OSError):
        atomic_write_text(target, "v2")
    assert target.read_text() == "v1"


# ── redaction ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expect",
    [
        ("api_key=abcdefghijklmnop1234", "***REDACTED***"),
        ("Bearer Vy1bk7Cz4HjP9MfQ8RtSaB", "***REDACTED***"),
        ("OPENAI_API_KEY: sk-abcdef0123456789xyz", "***REDACTED***"),
    ],
)
def test_redact(raw, expect):
    assert expect in redact_secrets(raw)


def test_redact_passthrough():
    assert redact_secrets("nothing secret here") == "nothing secret here"
    assert redact_secrets("") == ""


# ── store CRUD ──────────────────────────────────────────────────────────────


def test_create_department(store: Store):
    d = Department(name="Marketing", color="#aabbcc")
    out = store.create_department(d)
    assert out.id == d.id
    assert (store.root / "departments" / f"{d.id}.json").exists()
    assert any(x.id == d.id for x in store.list_departments())


def test_create_employee_requires_department(store: Store):
    e = Employee(department_id="dept_doesnotexist", name="X", model="m")
    with pytest.raises(ValueError):
        store.create_employee(e)


def test_employee_lifecycle(store: Store, department):
    emp = Employee(department_id=department.id, name="Joe", model="m")
    created = store.create_employee(emp)
    assert created.revision == 1
    # Department should now know about him.
    dept_after = store.get_department(department.id)
    assert created.id in dept_after.employee_ids

    updated = store.update_employee(created.id, name="Joseph")
    assert updated.revision == 2
    assert updated.name == "Joseph"
    assert updated.updated_at >= created.updated_at

    store.delete_employee(created.id)
    assert store.get_employee(created.id) is None
    dept_after = store.get_department(department.id)
    assert created.id not in dept_after.employee_ids


def test_delete_department_cascades(store: Store, department):
    e1 = store.create_employee(Employee(department_id=department.id, name="A", model="m"))
    e2 = store.create_employee(Employee(department_id=department.id, name="B", model="m"))
    removed = store.delete_department(department.id)
    assert sorted(removed) == sorted([e1.id, e2.id])
    assert store.get_department(department.id) is None
    assert store.get_employee(e1.id) is None
    assert store.get_employee(e2.id) is None


# ── boot / quarantine ───────────────────────────────────────────────────────


def test_boot_quarantines_bad_files(office_root: Path):
    bad = office_root / "departments" / "broken.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("{not json")
    s = Store(office_root)
    counts = s.boot_from_disk()
    assert counts["quarantined"] == 1
    assert not bad.exists()
    assert any(p.name == "broken.json" for p in (office_root / ".quarantine").rglob("*.json"))


def test_orphan_employee_quarantined(office_root: Path):
    # employee referencing missing department
    emp_path = office_root / "employees" / "emp_orphanid.json"
    emp_path.parent.mkdir(parents=True, exist_ok=True)
    emp_path.write_text(json.dumps({
        "id": "emp_orphanid",
        "department_id": "dept_missing0",
        "name": "Orphan",
        "model": "m",
    }))
    s = Store(office_root)
    counts = s.boot_from_disk()
    assert counts["quarantined"] >= 1


# ── activity log + read pagination ──────────────────────────────────────────


def test_activity_append_and_read(store: Store, employee):
    for i in range(7):
        store.append_activity(ActivityEvent(
            employee_id=employee.id,
            department_id=employee.department_id,
            kind="assistant",
            text=f"message {i}",
        ))
    events, cursor = store.read_activity(employee.id, limit=4)
    assert len(events) == 4
    assert events[-1]["text"].endswith("6")
    earlier, _ = store.read_activity(employee.id, limit=4, cursor=cursor)
    assert len(earlier) == 3


def test_activity_redaction_persisted(store: Store, employee):
    store.append_activity(ActivityEvent(
        employee_id=employee.id,
        department_id=employee.department_id,
        kind="tool_call",
        text="api_key=AbCdEfGhIj1234567890",
    ))
    events, _ = store.read_activity(employee.id, limit=10)
    assert "AbCdEfGhIj1234567890" not in events[-1]["text"]


# ── export / import round-trip ──────────────────────────────────────────────


def test_export_import_round_trip(store: Store, department):
    emp = store.create_employee(Employee(department_id=department.id, name="Z", model="m"))
    payload = store.export()
    assert payload["version"] == 1

    fresh_root = store.root.parent / "office2"
    fresh = Store(fresh_root)
    fresh.boot_from_disk()
    counts = fresh.import_(payload)
    assert counts == {"departments": 1, "employees": 1}
    assert fresh.get_employee(emp.id) is not None
    assert fresh.get_department(department.id) is not None
