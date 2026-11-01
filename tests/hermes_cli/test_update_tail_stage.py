"""Tail progress heartbeat: publish/read/clear round-trips against a temp home."""
import json
import time

import pytest

import hermes_constants
from hermes_cli import update_stage
from hermes_cli.update_stage import (
    TAIL_STAGE_STALE_SECONDS,
    clear_tail_stage,
    publish_tail_stage,
    read_tail_stage,
    tail_stage_file,
)


@pytest.fixture
def tail_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: home)
    return home


def test_publish_then_read_reports_a_fresh_stage(tail_home):
    publish_tail_stage("Building update products")
    stage = read_tail_stage()
    assert stage is not None
    assert stage["status"] == "running"
    assert stage["message"] == "Building update products"
    assert stage["age_seconds"] is not None and stage["age_seconds"] < 60


def test_terminal_state_is_not_running(tail_home):
    publish_tail_stage("Tail finished with 1 follow-up(s) owed", status="done")
    stage = read_tail_stage()
    assert stage is not None
    assert stage["status"] == "done"


def test_missing_file_reads_as_no_stage(tail_home):
    assert read_tail_stage() is None


def test_clear_removes_the_file(tail_home):
    publish_tail_stage("Publishing launchers")
    assert tail_stage_file() is not None and tail_stage_file().is_file()
    clear_tail_stage()
    assert read_tail_stage() is None


def test_old_heartbeat_reads_as_stale(tail_home):
    path = tail_stage_file()
    assert path is not None
    old = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(time.time() - TAIL_STAGE_STALE_SECONDS - 60))
    path.write_text(json.dumps({"status": "running", "message": "Building", "at": old}), encoding="utf-8")
    stage = read_tail_stage()
    assert stage is not None
    assert stage["age_seconds"] is not None
    assert stage["age_seconds"] > TAIL_STAGE_STALE_SECONDS


def test_garbage_file_reads_as_no_stage(tail_home):
    path = tail_stage_file()
    assert path is not None
    path.write_text("not json {{", encoding="utf-8")
    assert read_tail_stage() is None


def test_publish_never_raises_when_home_is_unwritable(monkeypatch):
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: None)
    publish_tail_stage("Building update products")  # must not raise
    clear_tail_stage()  # must not raise
