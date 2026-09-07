"""Cron pre-dispatch configuration validation (T1-26).

A job whose configuration cannot possibly produce a successful run — missing
provider API key, unready attached skill (missing required env), unknown
delivery platform — must be blocked BEFORE any agent machinery is constructed:

  - ``last_status`` becomes ``blocked_config`` (not a generic ``error``),
  - exactly ONE alert is delivered (no re-alert every tick — same
    alert-once spirit as the dead-pin auto-pause in #73506),
  - the agent is NEVER constructed, so no LLM call is burned.

``cron.preflight: false`` in config.yaml restores the old behavior (the run
proceeds to resolution and fails loudly every tick).

Related precedent: #27948 (fail-loud for hidden tools — same fail-before-run
spirit, different check) and #44585 (drift guard: skip-run-no-spend shape).
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cron.jobs as cron_jobs
from cron.scheduler import run_job
import cron.scheduler as sched


_RUNTIME = {
    "api_key": "test-key",
    "base_url": "https://example.invalid/v1",
    "provider": "openrouter",
    "api_mode": "chat_completions",
}


def _job(**overrides):
    job = {
        "id": "pf-test",
        "name": "preflight test",
        "prompt": "hello",
        "enabled": True,
        "state": "scheduled",
        "schedule": {"kind": "interval", "minutes": 5, "display": "every 5m"},
        "deliver": "local",
        "model": None,
        "provider": None,
        "base_url": None,
    }
    job.update(overrides)
    return job


class _AuthErrorFactory:
    """Raise a real AuthError from hermes_cli.auth."""

    def __call__(self, **kwargs):
        from hermes_cli.auth import AuthError

        raise AuthError("No API key configured for provider 'openrouter'")


def _run_job_patched(job, tmp_path, *, resolve=None, skill_view=None):
    """Drive run_job with the standard cron-test seams patched.

    Returns (success, output, final_response, error, agent_constructed).
    """
    fake_db = MagicMock()
    patches = [
        patch("cron.scheduler._hermes_home", tmp_path),
        patch("cron.scheduler._resolve_origin", return_value=None),
        patch("hermes_cli.env_loader.load_hermes_dotenv"),
        patch("hermes_cli.env_loader.reset_secret_source_cache"),
        patch("hermes_state.get_shared_session_db", return_value=fake_db),
        patch("tools.mcp_tool.discover_mcp_tools", return_value=[]),
    ]
    if resolve is None:
        patches.append(
            patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                return_value=dict(_RUNTIME),
            )
        )
    else:
        patches.append(
            patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                side_effect=resolve,
            )
        )
    if skill_view is not None:
        patches.append(patch("tools.skills_tool.skill_view", side_effect=skill_view))

    with patch("run_agent.AIAgent") as mock_agent_cls:
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {"final_response": "ok"}
        mock_agent_cls.return_value = mock_agent
        from contextlib import ExitStack

        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            success, output, final_response, error = run_job(job)
        agent_constructed = mock_agent_cls.called
    return success, output, final_response, error, agent_constructed


class TestMissingProviderKeyBlocks:
    def test_missing_key_blocked_config_no_agent(self, tmp_path):
        """Missing provider key (AuthError, no fallback chain) → blocked_config,
        agent never constructed, no LLM run burned."""
        job = _job()
        with cron_jobs.use_cron_store(tmp_path):
            cron_jobs.save_jobs([job])
            success, output, final_response, error, agent_constructed = \
                _run_job_patched(job, tmp_path, resolve=_AuthErrorFactory())

        assert agent_constructed is False
        assert success is False
        assert error is not None
        assert "[blocked_config]" in error
        assert "blocked" in output.lower() or "BLOCKED" in output

    def test_single_alert_across_two_ticks_and_blocked_status(self, tmp_path):
        """Two ticks of a blocked job through run_one_job deliver exactly ONE
        alert and persist last_status='blocked_config'."""
        job = _job()
        deliveries = []

        def fake_deliver(job, content, adapters=None, loop=None, **kwargs):
            deliveries.append(content)
            return None

        with cron_jobs.use_cron_store(tmp_path):
            cron_jobs.save_jobs([job])
            fake_db = MagicMock()
            for _tick in range(2):
                fresh = [j for j in cron_jobs.load_jobs() if j["id"] == job["id"]][0]
                with patch("cron.scheduler._hermes_home", tmp_path), \
                     patch("cron.scheduler._resolve_origin", return_value=None), \
                     patch("hermes_cli.env_loader.load_hermes_dotenv"), \
                     patch("hermes_cli.env_loader.reset_secret_source_cache"), \
                     patch("hermes_state.get_shared_session_db", return_value=fake_db), \
                     patch("tools.mcp_tool.discover_mcp_tools", return_value=[]), \
                     patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                           side_effect=_AuthErrorFactory()), \
                     patch.object(sched, "_deliver_result", side_effect=fake_deliver), \
                     patch("run_agent.AIAgent") as mock_agent_cls:
                    ok = sched.run_one_job(fresh)
                    assert ok is True
                    assert mock_agent_cls.called is False

            stored = [j for j in cron_jobs.load_jobs() if j["id"] == job["id"]][0]

        assert stored["last_status"] == "blocked_config"
        assert len(deliveries) == 1, (
            f"expected exactly one alert across two ticks, got {len(deliveries)}: "
            f"{deliveries!r}"
        )
        assert "blocked" in deliveries[0].lower()

    def test_fallback_chain_rescues_missing_primary_key(self, tmp_path):
        """A configured fallback chain means a missing primary key does NOT
        block — the existing auth-fallback path handles it."""
        (tmp_path / "config.yaml").write_text(
            "fallback_providers:\n"
            "  - provider: openrouter\n"
            "    model: z-ai/glm-5.2\n",
            encoding="utf-8",
        )
        calls = []

        def resolve(**kwargs):
            calls.append(kwargs.get("requested"))
            if kwargs.get("requested") in (None, ""):
                from hermes_cli.auth import AuthError

                raise AuthError("no key")
            return {**_RUNTIME, "provider": "openrouter"}

        job = _job()
        with cron_jobs.use_cron_store(tmp_path):
            cron_jobs.save_jobs([job])
            success, output, final_response, error, agent_constructed = \
                _run_job_patched(job, tmp_path, resolve=resolve)

        assert agent_constructed is True
        assert success is True
        assert error is None


class TestHealthyJobUnaffected:
    def test_healthy_job_runs_normally(self, tmp_path):
        job = _job()
        with cron_jobs.use_cron_store(tmp_path):
            cron_jobs.save_jobs([job])
            success, output, final_response, error, agent_constructed = \
                _run_job_patched(job, tmp_path)

        assert success is True
        assert error is None
        assert final_response == "ok"
        assert agent_constructed is True

    def test_recovery_clears_alert_marker(self, tmp_path):
        """After a blocked tick, a healthy tick clears the alert-dedup marker
        so a FUTURE config break re-alerts."""
        job = _job()
        with cron_jobs.use_cron_store(tmp_path):
            cron_jobs.save_jobs([job])
            # Tick 1: blocked.
            _run_job_patched(job, tmp_path, resolve=_AuthErrorFactory())
            stored = [j for j in cron_jobs.load_jobs() if j["id"] == job["id"]][0]
            assert stored.get("preflight_alerted")
            # Tick 2: key restored → healthy run clears the marker.
            fresh = [j for j in cron_jobs.load_jobs() if j["id"] == job["id"]][0]
            success, *_rest, agent_constructed = _run_job_patched(fresh, tmp_path)
            assert success is True
            assert agent_constructed is True
            stored = [j for j in cron_jobs.load_jobs() if j["id"] == job["id"]][0]
            assert not stored.get("preflight_alerted")


class TestOptOut:
    def test_preflight_false_restores_old_behavior(self, tmp_path):
        """cron.preflight: false → job proceeds to resolution and fails the
        old way (error status, re-alerts every tick, no blocked_config)."""
        (tmp_path / "config.yaml").write_text(
            "cron:\n  preflight: false\n", encoding="utf-8"
        )
        job = _job()
        deliveries = []

        def fake_deliver(job, content, adapters=None, loop=None, **kwargs):
            deliveries.append(content)
            return None

        with cron_jobs.use_cron_store(tmp_path):
            cron_jobs.save_jobs([job])
            fake_db = MagicMock()
            for _tick in range(2):
                fresh = [j for j in cron_jobs.load_jobs() if j["id"] == job["id"]][0]
                with patch("cron.scheduler._hermes_home", tmp_path), \
                     patch("cron.scheduler._resolve_origin", return_value=None), \
                     patch("hermes_cli.env_loader.load_hermes_dotenv"), \
                     patch("hermes_cli.env_loader.reset_secret_source_cache"), \
                     patch("hermes_state.get_shared_session_db", return_value=fake_db), \
                     patch("tools.mcp_tool.discover_mcp_tools", return_value=[]), \
                     patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                           side_effect=_AuthErrorFactory()), \
                     patch.object(sched, "_deliver_result", side_effect=fake_deliver), \
                     patch("run_agent.AIAgent") as mock_agent_cls:
                    sched.run_one_job(fresh)
                    assert mock_agent_cls.called is False

            stored = [j for j in cron_jobs.load_jobs() if j["id"] == job["id"]][0]

        assert stored["last_status"] == "error"
        assert len(deliveries) == 2  # old behavior: alert every tick


class TestSkillReadiness:
    def test_unready_skill_blocks(self, tmp_path):
        """An attached skill whose readiness_status is setup_needed (missing
        required env) blocks the run before the agent is constructed."""
        payload = json.dumps(
            {
                "success": True,
                "content": "# needy skill\nbody",
                "readiness_status": "setup_needed",
                "setup_needed": True,
                "missing_required_environment_variables": ["NEEDY_API_KEY"],
                "missing_required_commands": [],
            }
        )

        def fake_skill_view(name, *args, **kwargs):
            return payload

        job = _job(skills=["needy-skill"])
        with cron_jobs.use_cron_store(tmp_path):
            cron_jobs.save_jobs([job])
            success, output, final_response, error, agent_constructed = \
                _run_job_patched(job, tmp_path, skill_view=fake_skill_view)

        assert agent_constructed is False
        assert success is False
        assert error is not None and "[blocked_config]" in error
        assert "NEEDY_API_KEY" in f"{error} {output}"

    def test_ready_skill_runs(self, tmp_path):
        payload = json.dumps(
            {
                "success": True,
                "content": "# ready skill\nbody",
                "readiness_status": "available",
                "setup_needed": False,
                "missing_required_environment_variables": [],
            }
        )

        def fake_skill_view(name, *args, **kwargs):
            return payload

        job = _job(skills=["ready-skill"])
        with cron_jobs.use_cron_store(tmp_path):
            cron_jobs.save_jobs([job])
            success, output, final_response, error, agent_constructed = \
                _run_job_patched(job, tmp_path, skill_view=fake_skill_view)

        assert success is True
        assert agent_constructed is True


class TestDeliveryPlatform:
    def test_unknown_delivery_platform_blocks(self, tmp_path):
        job = _job(deliver="notaplatform")
        with cron_jobs.use_cron_store(tmp_path):
            cron_jobs.save_jobs([job])
            with patch("cron.scheduler._is_known_delivery_platform",
                       return_value=False):
                success, output, final_response, error, agent_constructed = \
                    _run_job_patched(job, tmp_path)

        assert agent_constructed is False
        assert success is False
        assert error is not None and "[blocked_config]" in error
        assert "notaplatform" in f"{error} {output}"

    def test_local_delivery_never_touches_gateway_config(self, tmp_path):
        """deliver=local jobs must not load gateway config in preflight."""
        job = _job(deliver="local")
        with cron_jobs.use_cron_store(tmp_path):
            cron_jobs.save_jobs([job])
            with patch("gateway.config.load_gateway_config",
                       side_effect=AssertionError("gateway config loaded")):
                success, *_rest, agent_constructed = _run_job_patched(job, tmp_path)

        assert success is True
        assert agent_constructed is True
