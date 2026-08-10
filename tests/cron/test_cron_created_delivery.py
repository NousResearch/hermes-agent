"""Create-time delivery resolution for cron-context job creation.

A cron-run agent (gate: cron.allow_agent_scheduling) creating a job must
never produce a job whose deliver field points at the creating run itself:
the run's session is ephemeral, so a stored literal 'origin' would resolve
against a session that no longer exists by fire time. The cronjob tool
therefore resolves 'origin' (or an omitted deliver) AT CREATE TIME to the
creating job's own concrete persistent target, read from the
HERMES_CRON_AUTO_DELIVER_* contextvars that run_job publishes per run.
Non-cron sessions keep the literal 'origin' (live chat sessions have a real
origin to resolve at fire time).
"""

import json

import pytest

from gateway.session_context import _VAR_MAP, clear_session_vars, set_session_vars


@pytest.fixture
def temp_cron_home(tmp_path, monkeypatch):
    from cron import jobs as cron_jobs

    with cron_jobs.use_cron_store(tmp_path):
        cron_jobs.ensure_dirs()
        yield tmp_path


def _enter_cron_context(platform=None, chat_id=None, thread_id=None):
    """Simulate the contextvar state run_job() establishes for a cron run."""
    tokens = set_session_vars(
        platform="",
        chat_id="",
        chat_name="",
        cron_session="1",
    )
    extra = []
    if platform is not None:
        extra.append(
            (_VAR_MAP["HERMES_CRON_AUTO_DELIVER_PLATFORM"],
             _VAR_MAP["HERMES_CRON_AUTO_DELIVER_PLATFORM"].set(platform))
        )
        extra.append(
            (_VAR_MAP["HERMES_CRON_AUTO_DELIVER_CHAT_ID"],
             _VAR_MAP["HERMES_CRON_AUTO_DELIVER_CHAT_ID"].set(str(chat_id)))
        )
        if thread_id is not None:
            extra.append(
                (_VAR_MAP["HERMES_CRON_AUTO_DELIVER_THREAD_ID"],
                 _VAR_MAP["HERMES_CRON_AUTO_DELIVER_THREAD_ID"].set(str(thread_id)))
            )
    return tokens, extra


def _exit_cron_context(tokens, extra):
    for var, token in reversed(extra):
        var.reset(token)
    clear_session_vars(tokens)


def _create(deliver=None):
    from tools.cronjob_tools import cronjob

    return json.loads(
        cronjob(
            action="create",
            schedule="every 4h",
            prompt="check the thing",
            deliver=deliver,
        )
    )


class TestCronContextDeliveryResolution:
    def test_omitted_deliver_resolves_to_creator_target(self, temp_cron_home):
        tokens, extra = _enter_cron_context("telegram", "-100123456", "17")
        try:
            result = _create()
        finally:
            _exit_cron_context(tokens, extra)
        assert result["success"] is True
        assert result["deliver"] == "telegram:-100123456:17"

    def test_literal_origin_resolves_to_creator_target(self, temp_cron_home):
        tokens, extra = _enter_cron_context("telegram", "-100123456", "17")
        try:
            result = _create(deliver="origin")
        finally:
            _exit_cron_context(tokens, extra)
        assert result["deliver"] == "telegram:-100123456:17"

    def test_no_thread_id_omits_thread_segment(self, temp_cron_home):
        tokens, extra = _enter_cron_context("slack", "C0ABC")
        try:
            result = _create(deliver="origin")
        finally:
            _exit_cron_context(tokens, extra)
        assert result["deliver"] == "slack:C0ABC"

    def test_creator_without_delivery_target_falls_back_to_local(self, temp_cron_home):
        # Creator job delivers nowhere concrete (deliver='local' run):
        # AUTO_DELIVER vars unset. Child must store 'local', never 'origin'.
        tokens, extra = _enter_cron_context()
        try:
            result = _create(deliver="origin")
        finally:
            _exit_cron_context(tokens, extra)
        assert result["deliver"] == "local"

    def test_comma_list_resolves_only_origin_element(self, temp_cron_home):
        tokens, extra = _enter_cron_context("telegram", "-100123456", "17")
        try:
            result = _create(deliver="origin,all")
        finally:
            _exit_cron_context(tokens, extra)
        assert result["deliver"] == "telegram:-100123456:17,all"

    def test_explicit_target_passes_through_verbatim(self, temp_cron_home):
        tokens, extra = _enter_cron_context("telegram", "-100999", "3")
        try:
            result = _create(deliver="discord:#engineering")
        finally:
            _exit_cron_context(tokens, extra)
        assert result["deliver"] == "discord:#engineering"

    def test_local_passes_through(self, temp_cron_home):
        tokens, extra = _enter_cron_context("telegram", "-100999")
        try:
            result = _create(deliver="local")
        finally:
            _exit_cron_context(tokens, extra)
        assert result["deliver"] == "local"

    def test_stored_deliver_never_literal_origin_in_cron_context(self, temp_cron_home):
        from cron.jobs import get_job

        tokens, extra = _enter_cron_context("telegram", "-100123456")
        try:
            result = _create(deliver="origin")
        finally:
            _exit_cron_context(tokens, extra)
        job = get_job(result["job_id"])
        stored = str(job.get("deliver", ""))
        assert "origin" not in [p.strip() for p in stored.split(",")]


class TestNonCronContextUnchanged:
    def test_chat_session_create_keeps_literal_origin(self, temp_cron_home):
        # No cron_session var — ordinary chat/CLI create. Existing semantics:
        # 'origin' stays literal and resolves at fire time.
        result = _create(deliver="origin")
        assert result["deliver"] == "origin"

    def test_chat_session_omitted_deliver_unchanged(self, temp_cron_home):
        # Outside cron context the resolution helper must be a no-op so the
        # ordinary chat/CLI create path is byte-identical to before.
        from tools.cronjob_tools import _resolve_cron_context_deliver

        assert _resolve_cron_context_deliver(None) is None
        assert _resolve_cron_context_deliver("origin") == "origin"
        assert _resolve_cron_context_deliver("origin,all") == "origin,all"
