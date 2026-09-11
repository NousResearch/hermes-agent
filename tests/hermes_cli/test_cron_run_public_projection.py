"""Cron history is public run metadata, never a raw session row."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_cli.web_routers import cron


@pytest.mark.parametrize("profile", [None, "worker"])
def test_run_history_projects_private_session_rows(monkeypatch, profile):
    row = {
        "id": "cron_job_20260907", "started_at": 10, "last_active": 20,
        "ended_at": 30, "end_reason": "completed", "archived": 0,
        "system_prompt": "PRIVATE_PROMPT_SENTINEL", "title": "PRIVATE_TITLE_SENTINEL",
        "preview": "PRIVATE_PREVIEW_SENTINEL", "model": "PRIVATE_MODEL_SENTINEL",
        "future_field": {"secret": "PRIVATE_FUTURE_SENTINEL"},
    }
    db = SimpleNamespace(list_cron_job_runs=Mock(return_value=[row]), close=Mock())
    monkeypatch.setattr(cron, "_find_cron_job_profile", lambda job_id: "worker")
    monkeypatch.setattr(cron, "_call_cron_for_profile", lambda *args: {"id": "job"})
    monkeypatch.setattr(cron, "_open_session_db_for_profile", lambda *args, **kwargs: db)
    result = cron._list_cron_job_runs_sync("job", profile=profile)

    assert result == {"runs": [{
        "id": "cron_job_20260907", "status": "completed", "started_at": 10,
        "ended_at": 30, "last_active": 20, "is_active": False, "archived": False,
    }], "limit": 20}
    assert "PRIVATE_" not in repr(result)
    assert "profile" not in row
    assert row["system_prompt"] == "PRIVATE_PROMPT_SENTINEL"
    db.close.assert_called_once_with()


@pytest.mark.parametrize("value", [True, "PRIVATE_TIME", {}, -1, float("nan"), float("inf"), 10**400])
def test_run_time_rejects_non_public_values(value):
    assert cron._public_cron_run_time(value) is None


def test_run_projection_does_not_coerce_private_objects():
    class PrivateValue:
        def __str__(self):
            raise AssertionError("private value coerced")

        def __bool__(self):
            raise AssertionError("private value tested")

    private = PrivateValue()
    assert cron._public_cron_run({"id": private}, now=40) is None
    row = cron._public_cron_run({
        "id": "cron_job_20260907", "started_at": private, "ended_at": 30,
        "last_active": private, "end_reason": private, "archived": private,
    }, now=40)
    assert row["status"] == "ended"
    assert row["started_at"] is None and row["last_active"] is None
    assert row["is_active"] is False and row["archived"] is False
