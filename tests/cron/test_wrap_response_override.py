"""Per-job wrap_response: strict storage normaliser, explicit tool reset, CLI reset flag."""
from __future__ import annotations

import argparse
import json

import pytest

from cron.jobs import _normalize_wrap_response, get_job
from tools.cronjob_tools import cronjob


@pytest.fixture(autouse=True)
def _cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")


def _create(**kw) -> str:
    created = json.loads(cronjob(action="create", prompt="Brief", schedule="every 1h", **kw))
    assert created["success"] is True, created
    return created["job_id"]


def test_tool_off_then_inherit_resets_to_global():
    job_id = _create(wrap_response="off")
    assert get_job(job_id)["wrap_response"] is False

    updated = json.loads(cronjob(action="update", job_id=job_id, wrap_response="inherit"))
    assert updated["success"] is True, updated
    assert get_job(job_id).get("wrap_response") is None


def test_tool_omitted_leaves_override_unchanged():
    job_id = _create(wrap_response="on")
    json.loads(cronjob(action="update", job_id=job_id, name="renamed"))
    assert get_job(job_id)["wrap_response"] is True


def test_new_job_inherits_by_default():
    assert get_job(_create()).get("wrap_response") is None


@pytest.mark.parametrize("value,expected", [(True, True), (False, False), (None, None)])
def test_normaliser_accepts_only_bool_or_null(value, expected):
    assert _normalize_wrap_response(value) is expected


@pytest.mark.parametrize("value", ["false", "true", "off", 0, 1])
def test_normaliser_rejects_strings_and_ints(value):
    with pytest.raises(ValueError):
        _normalize_wrap_response(value)


def test_string_false_is_not_stored_as_wrap_on():
    job_id = _create(wrap_response="on")
    result = json.loads(cronjob(action="update", job_id=job_id, wrap_response="false"))
    assert result["success"] is False
    assert get_job(job_id)["wrap_response"] is True


def test_cli_edit_inherit_flag_parses_to_reset():
    from hermes_cli.subcommands.cron import build_cron_parser

    parser = argparse.ArgumentParser()
    build_cron_parser(parser.add_subparsers(dest="command"), cmd_cron=lambda a: None)
    args = parser.parse_args(["cron", "edit", "abc123", "--inherit-wrap-response"])
    assert args.wrap_response == "inherit"
    args = parser.parse_args(["cron", "edit", "abc123", "--no-wrap-response"])
    assert args.wrap_response is False


@pytest.mark.parametrize("job,cfg,expected", [
    ({"wrap_response": False}, {"cron": {"wrap_response": True}}, False),
    ({"wrap_response": True}, {"cron": {"wrap_response": False}}, True),
    ({}, {"cron": {"wrap_response": False}}, False),
    ({}, None, True),                       # config failed to load
    ({}, {"cron": "not-a-mapping"}, True),   # malformed section
])
def test_effective_wrap_response_precedence(job, cfg, expected):
    from cron.scheduler_delivery import _effective_wrap_response

    assert bool(_effective_wrap_response(job, cfg)) is expected
