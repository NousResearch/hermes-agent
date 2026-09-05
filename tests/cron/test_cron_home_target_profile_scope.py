"""Regression: multiplex cron delivery destination must follow the OWNING profile.

#94862 / #97909 / #99028: the winning ticker process delivered a profile's cron
job through the EXECUTING process's home channel — `_env_home_target_chat_id`
read `os.getenv` directly, so the default profile's TELEGRAM_HOME_CHANNEL (or
Feishu/Discord equivalent) won over the owning profile's `.env`. The fix reads
through `agent.secret_scope.get_secret`, which resolves the job-owning
profile's scope installed by run_one_job for the whole execute→deliver span.
"""

import pytest

from agent import secret_scope


@pytest.fixture()
def profile_home(tmp_path):
    home = tmp_path / "profiles" / "tecnologia"
    home.mkdir(parents=True)
    (home / ".env").write_text(
        'TELEGRAM_HOME_CHANNEL="111111111"\n'
        'TELEGRAM_HOME_CHANNEL_THREAD_ID="42"\n',
        encoding="utf-8",
    )
    return home


@pytest.fixture()
def owning_profile_scope(profile_home):
    """Install the owning profile's secret scope, as run_one_job does."""
    token = secret_scope.set_secret_scope(
        secret_scope.build_profile_secret_scope(profile_home)
    )
    try:
        yield profile_home
    finally:
        secret_scope.reset_secret_scope(token)


class TestHomeTargetFollowsOwningProfile:
    def test_chat_id_resolves_from_profile_scope_not_process_env(
        self, owning_profile_scope, monkeypatch
    ):
        # The executing process (e.g. the Desktop/default backend that won the
        # tick lock) carries the DEFAULT profile's home channel in os.environ.
        monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "5697433938")

        from cron.scheduler_delivery import _env_home_target_chat_id

        assert _env_home_target_chat_id("telegram") == "111111111"

    def test_thread_id_resolves_from_profile_scope_not_process_env(
        self, owning_profile_scope, monkeypatch
    ):
        monkeypatch.setenv("TELEGRAM_HOME_CHANNEL_THREAD_ID", "7")

        from cron.scheduler_delivery import _get_home_target_thread_id

        assert _get_home_target_thread_id("telegram") == "42"

    def test_no_scope_keeps_legacy_process_env_behavior(self, monkeypatch):
        """Single-profile deployments (no scope installed) read os.environ."""
        monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "5697433938")

        from cron.scheduler_delivery import _env_home_target_chat_id

        assert secret_scope.current_secret_scope() is None
        assert _env_home_target_chat_id("telegram") == "5697433938"
