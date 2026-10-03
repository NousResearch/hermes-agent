"""Regression tests for HERMES_CRON_SESSION environment leak into gateway turns (#108121).

Ensures that ambient or process-global HERMES_CRON_SESSION env vars do not bleed
into subsequent Discord/Telegram/Gateway sessions when set_session_vars() is called.
"""

import os
import pytest
from gateway.session_context import (
    clear_session_vars,
    get_session_env,
    reset_session_vars,
    set_session_vars,
)
from tools.approval_context import _is_cron_approval_context


@pytest.fixture(autouse=True)
def _cleanup_env(monkeypatch):
    reset_session_vars()
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    yield
    reset_session_vars()


def test_set_session_vars_defaults_cron_session_to_empty_string(monkeypatch):
    """When os.environ has HERMES_CRON_SESSION=1, set_session_vars without cron_session masks it."""
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")

    # Before binding, process env would fall back to "1"
    assert get_session_env("HERMES_CRON_SESSION") == "1"

    # Gateway message arrives and binds session context without specifying cron_session
    tokens = set_session_vars(platform="discord", chat_id="456", session_key="disc-1")
    try:
        assert get_session_env("HERMES_CRON_SESSION") == ""
        assert _is_cron_approval_context() is False
    finally:
        clear_session_vars(tokens)


def test_cron_run_scope_binds_cron_session(monkeypatch):
    """Cron scope sets cron_session=1 during execution and clears upon exit."""
    from cron.scheduler import _CronRunScope

    scope = _CronRunScope(job={"id": "test_job"}, job_id="test_job", execution_id="exec-1")
    try:
        scope.enter()
        assert get_session_env("HERMES_CRON_SESSION") == "1"
        assert _is_cron_approval_context() is True
    finally:
        scope.exit()

    # After exit, session context is cleared
    assert get_session_env("HERMES_CRON_SESSION") == ""
    assert _is_cron_approval_context() is False
