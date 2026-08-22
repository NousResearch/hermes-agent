"""Cron pre-dispatch configuration validation (T1-26).

A job whose configuration cannot possibly produce a successful run — missing
provider API key, unready attached skill (missing required env), unknown
delivery platform — must be blocked BEFORE any agent machinery is constructed:

  - ``last_status`` becomes ``blocked_config`` (not a generic ``error``),
  - exactly ONE alert is delivered (no re-alert every tick — same
    alert-once spirit as the dead-pin auto-pause in #73506),
  - the agent is NEVER constructed, so no LLM call is burned.

``cron.preflight: false`` in config.yaml restores the old behavior for the
optional provider/skill/delivery checks. Mandatory authority checks such as
effective toolset resolution remain active.

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


def _run_job_patched(
    job,
    tmp_path,
    *,
    resolve=None,
    skill_view=None,
    trace=None,
):
    """Drive run_job with the standard cron-test seams patched.

    Returns (success, output, final_response, error, agent_constructed).
    When *trace* is a dict, it receives the provider, MCP-discovery, and agent
    constructor mocks so boundary ordering can be asserted without changing
    the stable return shape used by existing tests.
    """
    fake_db = MagicMock()
    mcp_patcher = patch("tools.mcp_tool.discover_mcp_tools", return_value=[])
    patches = [
        patch("cron.scheduler._hermes_home", tmp_path),
        patch("cron.scheduler._resolve_origin", return_value=None),
        patch("hermes_cli.env_loader.load_hermes_dotenv"),
        patch("hermes_cli.env_loader.reset_secret_source_cache"),
        patch("hermes_state.SessionDB", return_value=fake_db),
    ]
    if resolve is None:
        runtime_patcher = patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value=dict(_RUNTIME),
        )
    else:
        runtime_patcher = patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=resolve,
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
            mcp_mock = stack.enter_context(mcp_patcher)
            runtime_mock = stack.enter_context(runtime_patcher)
            success, output, final_response, error = run_job(job)
        agent_constructed = mock_agent_cls.called
        if trace is not None:
            trace.update(
                agent=mock_agent_cls,
                mcp_discovery=mcp_mock,
                runtime_provider=runtime_mock,
            )
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

        def fake_deliver(job, content, adapters=None, loop=None):
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
                     patch("hermes_state.SessionDB", return_value=fake_db), \
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


class TestToolsetResolutionBlocks:
    def test_resolution_failure_blocks_before_provider_mcp_or_agent(self, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "platform_toolsets:\n  cron: web\n", encoding="utf-8"
        )
        job = _job()
        trace = {}

        with cron_jobs.use_cron_store(tmp_path):
            cron_jobs.save_jobs([job])
            success, output, _final_response, error, agent_constructed = (
                _run_job_patched(job, tmp_path, trace=trace)
            )

        assert success is False
        assert agent_constructed is False
        assert error is not None and "[blocked_config]" in error
        assert "toolset resolution failed" in f"{error} {output}".lower()
        trace["runtime_provider"].assert_not_called()
        trace["mcp_discovery"].assert_not_called()

    def test_resolution_failure_stays_mandatory_when_optional_preflight_is_disabled(
        self, tmp_path
    ):
        (tmp_path / "config.yaml").write_text(
            "cron:\n  preflight: false\n"
            "platform_toolsets:\n  - not-a-mapping\n",
            encoding="utf-8",
        )
        job = _job()

        with cron_jobs.use_cron_store(tmp_path):
            cron_jobs.save_jobs([job])
            success, output, _final_response, error, agent_constructed = (
                _run_job_patched(job, tmp_path)
            )

        assert success is False
        assert agent_constructed is False
        assert error is not None and "[blocked_config]" in error
        assert "toolset resolution failed" in f"{error} {output}".lower()
        assert "cron.preflight: false" not in output

    def test_disabled_toolset_resolution_failure_blocks_before_provider(self, tmp_path):
        job = _job()
        trace = {}

        with cron_jobs.use_cron_store(tmp_path):
            cron_jobs.save_jobs([job])
            with patch.object(
                sched,
                "_resolve_cron_disabled_toolsets",
                side_effect=sched.CronToolsetResolutionError(
                    "cron toolset resolution failed; check cron tool configuration"
                ),
            ):
                success, output, _final_response, error, agent_constructed = (
                    _run_job_patched(job, tmp_path, trace=trace)
                )

        assert success is False
        assert agent_constructed is False
        assert error is not None and "[blocked_config]" in error
        assert "toolset resolution failed" in f"{error} {output}".lower()
        trace["runtime_provider"].assert_not_called()
        trace["mcp_discovery"].assert_not_called()

    def test_malformed_agent_policy_uses_blocked_config_before_max_turns(self, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "model:\n  default: test/model\nagent: not-a-mapping\n",
            encoding="utf-8",
        )
        job = _job()
        trace = {}

        with cron_jobs.use_cron_store(tmp_path):
            cron_jobs.save_jobs([job])
            success, output, _final_response, error, agent_constructed = (
                _run_job_patched(job, tmp_path, trace=trace)
            )

        assert success is False
        assert agent_constructed is False
        assert error is not None and "[blocked_config]" in error
        assert "toolset resolution failed" in f"{error} {output}".lower()
        trace["runtime_provider"].assert_not_called()
        trace["mcp_discovery"].assert_not_called()

    def test_real_empty_platform_toolset_reaches_agent_as_empty(self, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "platform_toolsets:\n  cron: []\n",
            encoding="utf-8",
        )
        job = _job()
        trace = {}

        with cron_jobs.use_cron_store(tmp_path):
            cron_jobs.save_jobs([job])
            success, _output, _final_response, error, agent_constructed = (
                _run_job_patched(job, tmp_path, trace=trace)
            )

        assert success is True
        assert error is None
        assert agent_constructed is True
        assert trace["agent"].call_args.kwargs["enabled_toolsets"] == []
        assert trace["agent"].call_args.kwargs["enabled_toolsets"] is not None

    def test_toolset_policy_resolvers_are_called_once(self, tmp_path):
        job = _job()

        with cron_jobs.use_cron_store(tmp_path):
            cron_jobs.save_jobs([job])
            with patch.object(
                sched,
                "_resolve_cron_enabled_toolsets",
                return_value=["file"],
            ) as enabled_resolver, patch.object(
                sched,
                "_resolve_cron_disabled_toolsets",
                return_value=["cronjob", "messaging", "clarify"],
            ) as disabled_resolver:
                success, _output, _final_response, error, agent_constructed = (
                    _run_job_patched(job, tmp_path)
                )

        assert success is True
        assert error is None
        assert agent_constructed is True
        enabled_resolver.assert_called_once()
        disabled_resolver.assert_called_once()

    def test_mandatory_failure_alert_rearms_after_recovery_with_preflight_disabled(
        self, tmp_path
    ):
        (tmp_path / "config.yaml").write_text(
            "cron:\n  preflight: false\n", encoding="utf-8"
        )
        job = _job()
        deliveries = []
        outcomes = iter([False, False, True, False])

        def resolve_toolsets(*_args, **_kwargs):
            if next(outcomes):
                return ["file"]
            raise sched.CronToolsetResolutionError(
                "cron toolset resolution failed; check cron tool configuration"
            )

        def fake_deliver(_job, content, adapters=None, loop=None):
            deliveries.append(content)
            return None

        fake_db = MagicMock()
        with cron_jobs.use_cron_store(tmp_path):
            cron_jobs.save_jobs([job])
            with patch("cron.scheduler._hermes_home", tmp_path), \
                 patch("cron.scheduler._resolve_origin", return_value=None), \
                 patch("hermes_cli.env_loader.load_hermes_dotenv"), \
                 patch("hermes_cli.env_loader.reset_secret_source_cache"), \
                 patch("hermes_state.SessionDB", return_value=fake_db), \
                 patch("tools.mcp_tool.discover_mcp_tools", return_value=[]), \
                 patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                       return_value=dict(_RUNTIME)), \
                 patch.object(sched, "_resolve_cron_enabled_toolsets",
                              side_effect=resolve_toolsets), \
                 patch.object(sched, "_deliver_result", side_effect=fake_deliver), \
                 patch("run_agent.AIAgent") as mock_agent_cls:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "ok"}
                mock_agent_cls.return_value = mock_agent

                for tick in range(4):
                    fresh = [
                        item
                        for item in cron_jobs.load_jobs()
                        if item["id"] == job["id"]
                    ][0]
                    assert sched.run_one_job(fresh) is True
                    stored = [
                        item
                        for item in cron_jobs.load_jobs()
                        if item["id"] == job["id"]
                    ][0]
                    blocked_alerts = [
                        content
                        for content in deliveries
                        if "blocked by configuration" in content.lower()
                    ]
                    if tick == 0:
                        assert stored["last_status"] == "blocked_config"
                        assert stored.get("preflight_alerted")
                        assert len(blocked_alerts) == 1
                    elif tick == 1:
                        assert stored.get("preflight_alerted")
                        assert len(blocked_alerts) == 1
                    elif tick == 2:
                        assert stored["last_status"] == "ok"
                        assert not stored.get("preflight_alerted")
                        assert len(blocked_alerts) == 1
                    else:
                        assert stored["last_status"] == "blocked_config"
                        assert stored.get("preflight_alerted")
                        assert len(blocked_alerts) == 2


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
        """Disabling optional preflight lets provider resolution fail the old
        way (error status, re-alert every tick, no blocked_config)."""
        (tmp_path / "config.yaml").write_text(
            "cron:\n  preflight: false\n", encoding="utf-8"
        )
        job = _job()
        deliveries = []

        def fake_deliver(job, content, adapters=None, loop=None):
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
                     patch("hermes_state.SessionDB", return_value=fake_db), \
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
