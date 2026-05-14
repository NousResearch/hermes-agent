"""Tests for session-scoped cronjob tool visibility."""

import json
import sys
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    """Isolated cron environment with temp HERMES_HOME."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "cron").mkdir()
    (hermes_home / "cron" / "output").mkdir()
    (hermes_home / "scripts").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import cron.jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")

    return hermes_home


@pytest.fixture(autouse=True)
def reset_session_context(monkeypatch):
    """Keep gateway session env/context isolated between tests."""
    session_env_names = [
        "HERMES_SESSION_PLATFORM",
        "HERMES_SESSION_CHAT_ID",
        "HERMES_SESSION_CHAT_NAME",
        "HERMES_SESSION_THREAD_ID",
        "HERMES_SESSION_USER_ID",
        "HERMES_SESSION_USER_NAME",
        "HERMES_SESSION_KEY",
    ]
    for name in session_env_names:
        monkeypatch.delenv(name, raising=False)
    yield
    from gateway.session_context import _UNSET, _VAR_MAP

    for var in _VAR_MAP.values():
        var.set(_UNSET)


def _list_job_names():
    from tools.cronjob_tools import cronjob

    result = json.loads(cronjob(action="list"))
    assert result["success"] is True
    return [job["name"] for job in result["jobs"]]


def test_cronjob_list_without_session_context_lists_all_jobs(cron_env):
    from cron.jobs import create_job

    create_job(
        prompt="Other group job",
        schedule="every 1h",
        name="other group monitor",
        origin={"platform": "telegram", "chat_id": "-1003956282046", "thread_id": None},
    )
    create_job(
        prompt="ChatLFG job",
        schedule="every 1h",
        name="chatlfg monitor",
        origin={"platform": "telegram", "chat_id": "-1001234567890", "thread_id": None},
    )

    assert _list_job_names() == ["other group monitor", "chatlfg monitor"]


def test_cronjob_list_in_group_session_shows_only_current_chat_jobs(cron_env):
    from cron.jobs import create_job
    from gateway.session_context import clear_session_vars, set_session_vars

    create_job(
        prompt="Other group job",
        schedule="every 1h",
        name="other group monitor",
        origin={"platform": "telegram", "chat_id": "-1003956282046", "thread_id": None},
    )
    create_job(
        prompt="ChatLFG job",
        schedule="every 1h",
        name="chatlfg monitor",
        origin={"platform": "telegram", "chat_id": "-1001234567890", "thread_id": None},
    )

    tokens = set_session_vars(platform="telegram", chat_id="-1001234567890")
    try:
        assert _list_job_names() == ["chatlfg monitor"]
    finally:
        clear_session_vars(tokens)


def test_cronjob_list_in_group_session_hides_originless_jobs(cron_env):
    from cron.jobs import create_job
    from gateway.session_context import clear_session_vars, set_session_vars

    create_job(
        prompt="Legacy local job",
        schedule="every 1h",
        name="legacy local monitor",
        origin=None,
    )
    create_job(
        prompt="ChatLFG job",
        schedule="every 1h",
        name="chatlfg monitor",
        origin={"platform": "telegram", "chat_id": "-1001234567890", "thread_id": None},
    )

    tokens = set_session_vars(platform="telegram", chat_id="-1001234567890")
    try:
        assert _list_job_names() == ["chatlfg monitor"]
    finally:
        clear_session_vars(tokens)


def test_cronjob_list_in_thread_session_shows_only_current_thread_jobs(cron_env):
    from cron.jobs import create_job
    from gateway.session_context import clear_session_vars, set_session_vars

    create_job(
        prompt="Same chat different topic",
        schedule="every 1h",
        name="other topic monitor",
        origin={"platform": "telegram", "chat_id": "-1001234567890", "thread_id": "111"},
    )
    create_job(
        prompt="Same chat current topic",
        schedule="every 1h",
        name="current topic monitor",
        origin={"platform": "telegram", "chat_id": "-1001234567890", "thread_id": "222"},
    )

    tokens = set_session_vars(platform="telegram", chat_id="-1001234567890", thread_id="222")
    try:
        assert _list_job_names() == ["current topic monitor"]
    finally:
        clear_session_vars(tokens)


def test_cronjob_management_in_group_session_rejects_other_chat_job_ids(cron_env):
    from cron.jobs import create_job, get_job
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools.cronjob_tools import cronjob

    other_chat_job = create_job(
        prompt="Other group job",
        schedule="every 1h",
        name="other group monitor",
        origin={"platform": "telegram", "chat_id": "-1003956282046", "thread_id": None},
    )

    tokens = set_session_vars(platform="telegram", chat_id="-1001234567890")
    try:
        result = json.loads(cronjob(action="pause", job_id=other_chat_job["id"]))
    finally:
        clear_session_vars(tokens)

    assert result["success"] is False
    assert "not found" in result["error"]
    assert get_job(other_chat_job["id"])["enabled"] is True
