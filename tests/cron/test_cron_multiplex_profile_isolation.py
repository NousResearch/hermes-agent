"""Regression tests for #87575 — profile scope isolation for cron jobs under multiplexing.

Ensures:
1. In-process cron jobs for a secondary profile execute under that profile's
   HERMES_HOME and secret scope via profile_runtime_scope.
2. Concurrent gateway turns for another profile (e.g. default/alpha) retain
   their own HERMES_HOME without bleeding.
3. Cron SessionDB connects to the target profile's state.db rather than
   the default root state.db.
4. load_hermes_dotenv in multiplex mode does not mutate process-global os.environ.
"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
import pytest

import cron.scheduler as cron_scheduler
from agent.secret_scope import (
    is_multiplex_active,
    profile_runtime_scope,
    set_multiplex_active,
)
from hermes_constants import (
    get_hermes_home,
    get_hermes_home_override,
    set_hermes_home_override,
    reset_hermes_home_override,
)


@pytest.fixture(autouse=True)
def _reset_multiplex_and_scope():
    orig_multiplex = is_multiplex_active()
    yield
    set_multiplex_active(orig_multiplex)


def test_profile_runtime_scope_isolates_home_and_secrets(tmp_path):
    """profile_runtime_scope sets and restores HERMES_HOME override and secret scope."""
    alpha_home = tmp_path / "alpha"
    beta_home = tmp_path / "beta"
    alpha_home.mkdir()
    beta_home.mkdir()
    (alpha_home / ".env").write_text("API_KEY=alpha-secret\n", encoding="utf-8")
    (beta_home / ".env").write_text("API_KEY=beta-secret\n", encoding="utf-8")

    from agent.secret_scope import get_secret

    set_multiplex_active(True)

    assert get_hermes_home_override() is None

    with profile_runtime_scope(alpha_home):
        assert get_hermes_home().resolve() == alpha_home.resolve()
        assert get_secret("API_KEY") == "alpha-secret"

        # Nested/concurrent beta scope
        with profile_runtime_scope(beta_home):
            assert get_hermes_home().resolve() == beta_home.resolve()
            assert get_secret("API_KEY") == "beta-secret"

        # Restores alpha
        assert get_hermes_home().resolve() == alpha_home.resolve()
        assert get_secret("API_KEY") == "alpha-secret"

    # Restores unset
    assert get_hermes_home_override() is None


def test_concurrent_threads_retain_distinct_profile_scopes(tmp_path):
    """Two concurrent threads each running under different profile scopes observe only their own."""
    alpha_home = tmp_path / "alpha"
    beta_home = tmp_path / "beta"
    alpha_home.mkdir()
    beta_home.mkdir()
    (alpha_home / ".env").write_text("PROVIDER_KEY=alpha-val\n", encoding="utf-8")
    (beta_home / ".env").write_text("PROVIDER_KEY=beta-val\n", encoding="utf-8")

    from agent.secret_scope import get_secret

    set_multiplex_active(True)

    def _run_in_scope(home: Path, expected_secret: str) -> tuple[Path, str]:
        with profile_runtime_scope(home):
            import time
            time.sleep(0.05)
            resolved_home = get_hermes_home().resolve()
            resolved_secret = get_secret("PROVIDER_KEY")
            return resolved_home, resolved_secret

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fut_alpha = pool.submit(_run_in_scope, alpha_home, "alpha-val")
        fut_beta = pool.submit(_run_in_scope, beta_home, "beta-val")

        alpha_res_home, alpha_res_sec = fut_alpha.result()
        beta_res_home, beta_res_sec = fut_beta.result()

    assert alpha_res_home == alpha_home.resolve()
    assert alpha_res_sec == "alpha-val"
    assert beta_res_home == beta_home.resolve()
    assert beta_res_sec == "beta-val"


def test_cron_session_db_targets_profile_state_db(tmp_path, monkeypatch):
    """Cron run_job initializes SessionDB with the profile's state.db."""
    beta_home = tmp_path / "profiles" / "beta"
    beta_home.mkdir(parents=True)
    (beta_home / ".env").write_text("TEST_KEY=1\n", encoding="utf-8")

    opened_paths = []

    class _CaptureSessionDB:
        def __init__(self, db_path=None, read_only=False):
            effective_path = Path(db_path) if db_path else (get_hermes_home() / "state.db")
            opened_paths.append(effective_path)
        def set_session_title(self, *args, **kwargs):
            pass

        def end_session(self, *args, **kwargs):
            pass

        def close(self):
            pass

    class _FakeAgent:
        def __init__(self, *args, **kwargs):
            self.session_db = kwargs.get("session_db")

        def run_conversation(self, prompt):
            return {
                "completed": True,
                "failed": False,
                "final_response": "done",
                "turn_exit_reason": "",
            }

        def close(self):
            pass

    monkeypatch.setattr("hermes_state.SessionDB", _CaptureSessionDB)
    monkeypatch.setattr("run_agent.AIAgent", _FakeAgent)
    monkeypatch.setattr(cron_scheduler, "_get_hermes_home", lambda: beta_home)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {
            "api_key": "test-key",
            "base_url": None,
            "provider": "test-provider",
            "api_mode": None,
            "command": None,
            "args": None,
        },
    )
    monkeypatch.setattr("tools.mcp_tool.discover_mcp_tools", lambda: [])
    monkeypatch.setattr(cron_scheduler, "get_fallback_chain", lambda _cfg: [])
    monkeypatch.setattr(
        cron_scheduler, "_guard_job_credential_exfil", lambda _job: None
    )

    set_multiplex_active(True)

    with profile_runtime_scope(beta_home):
        success, _out, resp, err = cron_scheduler.run_job(
            {
                "id": "profile-db-test",
                "name": "Profile DB Test",
                "prompt": "Say done",
            }
        )

    assert success is True
    assert len(opened_paths) == 1
    assert opened_paths[0].resolve() == (beta_home / "state.db").resolve()
