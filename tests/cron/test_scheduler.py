"""Tests for cron/scheduler.py — origin resolution, delivery routing, and error logging."""

import json
import logging
import os
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from cron.scheduler import _resolve_origin, _resolve_delivery_target, _deliver_result, run_job, SILENT_MARKER, _build_job_prompt


class TestResolveOrigin:
    def test_full_origin(self):
        job = {
            "origin": {
                "platform": "telegram",
                "chat_id": "123456",
                "chat_name": "Test Chat",
                "thread_id": "42",
            }
        }
        result = _resolve_origin(job)
        assert isinstance(result, dict)
        assert result == job["origin"]
        assert result["platform"] == "telegram"
        assert result["chat_id"] == "123456"
        assert result["chat_name"] == "Test Chat"
        assert result["thread_id"] == "42"

    def test_no_origin(self):
        assert _resolve_origin({}) is None
        assert _resolve_origin({"origin": None}) is None

    def test_missing_platform(self):
        job = {"origin": {"chat_id": "123"}}
        assert _resolve_origin(job) is None

    def test_missing_chat_id(self):
        job = {"origin": {"platform": "telegram"}}
        assert _resolve_origin(job) is None

    def test_empty_origin(self):
        job = {"origin": {}}
        assert _resolve_origin(job) is None


class TestResolveDeliveryTarget:
    def test_origin_delivery_preserves_thread_id(self):
        job = {
            "deliver": "origin",
            "origin": {
                "platform": "telegram",
                "chat_id": "-1001",
                "thread_id": "17585",
            },
        }

        assert _resolve_delivery_target(job) == {
            "platform": "telegram",
            "chat_id": "-1001",
            "thread_id": "17585",
        }

    def test_bare_platform_uses_matching_origin_chat(self):
        job = {
            "deliver": "telegram",
            "origin": {
                "platform": "telegram",
                "chat_id": "-1001",
                "thread_id": "17585",
            },
        }

        assert _resolve_delivery_target(job) == {
            "platform": "telegram",
            "chat_id": "-1001",
            "thread_id": "17585",
        }

    def test_bare_platform_falls_back_to_home_channel(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "-2002")
        job = {
            "deliver": "telegram",
            "origin": {
                "platform": "discord",
                "chat_id": "abc",
            },
        }

        assert _resolve_delivery_target(job) == {
            "platform": "telegram",
            "chat_id": "-2002",
            "thread_id": None,
        }


class TestDeliverResultMirrorLogging:
    """Verify that mirror_to_session failures are logged, not silently swallowed."""

    def test_mirror_failure_is_logged(self, caplog):
        """When mirror_to_session raises, a warning should be logged."""
        from gateway.config import Platform

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})), \
             patch("gateway.mirror.mirror_to_session", side_effect=ConnectionError("network down")):
            job = {
                "id": "test-job",
                "deliver": "origin",
                "origin": {"platform": "telegram", "chat_id": "123"},
            }
            with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
                _deliver_result(job, "Hello!")

        assert any("mirror_to_session failed" in r.message for r in caplog.records), \
            f"Expected 'mirror_to_session failed' warning in logs, got: {[r.message for r in caplog.records]}"

    def test_origin_delivery_preserves_thread_id(self):
        """Origin delivery should forward thread_id to send/mirror helpers."""
        from gateway.config import Platform

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}

        job = {
            "id": "test-job",
            "deliver": "origin",
            "origin": {
                "platform": "telegram",
                "chat_id": "-1001",
                "thread_id": "17585",
            },
        }

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
             patch("gateway.mirror.mirror_to_session") as mirror_mock:
            _deliver_result(job, "hello")

        send_mock.assert_called_once()
        assert send_mock.call_args.kwargs["thread_id"] == "17585"
        mirror_mock.assert_called_once_with(
            "telegram",
            "-1001",
            "hello",
            source_label="cron",
            thread_id="17585",
        )


class TestRunJobSessionPersistence:
    def test_run_job_passes_session_db_and_cron_platform(self, tmp_path):
        job = {
            "id": "test-job",
            "name": "test",
            "prompt": "hello",
        }
        fake_db = MagicMock()

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("dotenv.load_dotenv"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch(
                 "hermes_cli.runtime_provider.resolve_runtime_provider",
                 return_value={
                     "api_key": "test-key",
                     "base_url": "https://example.invalid/v1",
                     "provider": "openrouter",
                     "api_mode": "chat_completions",
                 },
             ), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent

            success, output, final_response, error = run_job(job)

        assert success is True
        assert error is None
        assert final_response == "ok"
        assert "ok" in output

        kwargs = mock_agent_cls.call_args.kwargs
        assert kwargs["session_db"] is fake_db
        assert kwargs["platform"] == "cron"
        assert kwargs["session_id"].startswith("cron_test-job_")
        fake_db.close.assert_called_once()

    def test_run_job_sets_auto_delivery_env_from_dotenv_home_channel(self, tmp_path, monkeypatch):
        job = {
            "id": "test-job",
            "name": "test",
            "prompt": "hello",
            "deliver": "telegram",
        }
        fake_db = MagicMock()
        seen = {}

        (tmp_path / ".env").write_text("TELEGRAM_HOME_CHANNEL=-2002\n")
        monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)
        monkeypatch.delenv("HERMES_CRON_AUTO_DELIVER_PLATFORM", raising=False)
        monkeypatch.delenv("HERMES_CRON_AUTO_DELIVER_CHAT_ID", raising=False)
        monkeypatch.delenv("HERMES_CRON_AUTO_DELIVER_THREAD_ID", raising=False)

        class FakeAgent:
            def __init__(self, *args, **kwargs):
                pass

            def run_conversation(self, *args, **kwargs):
                seen["platform"] = os.getenv("HERMES_CRON_AUTO_DELIVER_PLATFORM")
                seen["chat_id"] = os.getenv("HERMES_CRON_AUTO_DELIVER_CHAT_ID")
                seen["thread_id"] = os.getenv("HERMES_CRON_AUTO_DELIVER_THREAD_ID")
                return {"final_response": "ok"}

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch(
                 "hermes_cli.runtime_provider.resolve_runtime_provider",
                 return_value={
                     "api_key": "***",
                     "base_url": "https://example.invalid/v1",
                     "provider": "openrouter",
                     "api_mode": "chat_completions",
                 },
             ), \
             patch("run_agent.AIAgent", FakeAgent):
            success, output, final_response, error = run_job(job)

        assert success is True
        assert error is None
        assert final_response == "ok"
        assert "ok" in output
        assert seen == {
            "platform": "telegram",
            "chat_id": "-2002",
            "thread_id": None,
        }
        assert os.getenv("HERMES_CRON_AUTO_DELIVER_PLATFORM") is None
        assert os.getenv("HERMES_CRON_AUTO_DELIVER_CHAT_ID") is None
        assert os.getenv("HERMES_CRON_AUTO_DELIVER_THREAD_ID") is None
        fake_db.close.assert_called_once()


class TestRunJobConfigLogging:
    """Verify that config.yaml parse failures are logged, not silently swallowed."""

    def test_bad_config_yaml_is_logged(self, caplog, tmp_path):
        """When config.yaml is malformed, a warning should be logged."""
        bad_yaml = tmp_path / "config.yaml"
        bad_yaml.write_text("invalid: yaml: [[[bad")

        job = {
            "id": "test-job",
            "name": "test",
            "prompt": "hello",
        }

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("dotenv.load_dotenv"), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent

            with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
                run_job(job)

        assert any("failed to load config.yaml" in r.message for r in caplog.records), \
            f"Expected 'failed to load config.yaml' warning in logs, got: {[r.message for r in caplog.records]}"

    def test_bad_prefill_messages_is_logged(self, caplog, tmp_path):
        """When the prefill messages file contains invalid JSON, a warning should be logged."""
        # Valid config.yaml that points to a bad prefill file
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text("prefill_messages_file: prefill.json\n")

        bad_prefill = tmp_path / "prefill.json"
        bad_prefill.write_text("{not valid json!!!")

        job = {
            "id": "test-job",
            "name": "test",
            "prompt": "hello",
        }

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("dotenv.load_dotenv"), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent

            with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
                run_job(job)

        assert any("failed to parse prefill messages" in r.message for r in caplog.records), \
            f"Expected 'failed to parse prefill messages' warning in logs, got: {[r.message for r in caplog.records]}"


class TestRunJobPerJobOverrides:
    def test_job_level_model_provider_and_base_url_overrides_are_used(self, tmp_path):
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text(
            "model:\n"
            "  default: gpt-5.4\n"
            "  provider: openai-codex\n"
            "  base_url: https://chatgpt.com/backend-api/codex\n"
        )

        job = {
            "id": "briefing-job",
            "name": "briefing",
            "prompt": "hello",
            "model": "perplexity/sonar-pro",
            "provider": "custom",
            "base_url": "http://127.0.0.1:4000/v1",
        }

        fake_db = MagicMock()
        fake_runtime = {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "http://127.0.0.1:4000/v1",
            "api_key": "***",
        }

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("dotenv.load_dotenv"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=fake_runtime) as runtime_mock, \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent

            success, output, final_response, error = run_job(job)

        assert success is True
        assert error is None
        assert final_response == "ok"
        assert "ok" in output
        runtime_mock.assert_called_once_with(
            requested="custom",
            explicit_base_url="http://127.0.0.1:4000/v1",
        )
        assert mock_agent_cls.call_args.kwargs["model"] == "perplexity/sonar-pro"
        fake_db.close.assert_called_once()


class TestRunJobSkillBacked:
    def test_run_job_loads_skill_and_disables_recursive_cron_tools(self, tmp_path):
        job = {
            "id": "skill-job",
            "name": "skill test",
            "prompt": "Check the feeds and summarize anything new.",
            "skill": "blogwatcher",
        }

        fake_db = MagicMock()

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("dotenv.load_dotenv"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch(
                 "hermes_cli.runtime_provider.resolve_runtime_provider",
                 return_value={
                     "api_key": "***",
                     "base_url": "https://example.invalid/v1",
                     "provider": "openrouter",
                     "api_mode": "chat_completions",
                 },
             ), \
             patch("tools.skills_tool.skill_view", return_value=json.dumps({"success": True, "content": "# Blogwatcher\nFollow this skill."})), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent

            success, output, final_response, error = run_job(job)

        assert success is True
        assert error is None
        assert final_response == "ok"

        kwargs = mock_agent_cls.call_args.kwargs
        assert "cronjob" in (kwargs["disabled_toolsets"] or [])

        prompt_arg = mock_agent.run_conversation.call_args.args[0]
        assert "blogwatcher" in prompt_arg
        assert "Follow this skill" in prompt_arg
        assert "Check the feeds and summarize anything new." in prompt_arg

    def test_run_job_loads_multiple_skills_in_order(self, tmp_path):
        job = {
            "id": "multi-skill-job",
            "name": "multi skill test",
            "prompt": "Combine the results.",
            "skills": ["blogwatcher", "find-nearby"],
        }

        fake_db = MagicMock()

        def _skill_view(name):
            return json.dumps({"success": True, "content": f"# {name}\nInstructions for {name}."})

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("dotenv.load_dotenv"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch(
                 "hermes_cli.runtime_provider.resolve_runtime_provider",
                 return_value={
                     "api_key": "***",
                     "base_url": "https://example.invalid/v1",
                     "provider": "openrouter",
                     "api_mode": "chat_completions",
                 },
             ), \
             patch("tools.skills_tool.skill_view", side_effect=_skill_view) as skill_view_mock, \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent

            success, output, final_response, error = run_job(job)

        assert success is True
        assert error is None
        assert final_response == "ok"
        assert skill_view_mock.call_count == 2
        assert [call.args[0] for call in skill_view_mock.call_args_list] == ["blogwatcher", "find-nearby"]

        prompt_arg = mock_agent.run_conversation.call_args.args[0]
        assert prompt_arg.index("blogwatcher") < prompt_arg.index("find-nearby")
        assert "Instructions for blogwatcher." in prompt_arg
        assert "Instructions for find-nearby." in prompt_arg
        assert "Combine the results." in prompt_arg


class TestSilentDelivery:
    """Verify that [SILENT] responses suppress delivery while still saving output."""

    def _make_job(self):
        return {
            "id": "monitor-job",
            "name": "monitor",
            "deliver": "origin",
            "origin": {"platform": "telegram", "chat_id": "123"},
        }

    def test_normal_response_delivers(self):
        with patch("cron.scheduler.get_due_jobs", return_value=[self._make_job()]), \
             patch("cron.scheduler.run_job", return_value=(True, "# output", "Results here", None)), \
             patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"), \
             patch("cron.scheduler._deliver_result") as deliver_mock, \
             patch("cron.scheduler.mark_job_run"):
            from cron.scheduler import tick
            tick(verbose=False)
        deliver_mock.assert_called_once()

    def test_silent_response_suppresses_delivery(self, caplog):
        with patch("cron.scheduler.get_due_jobs", return_value=[self._make_job()]), \
             patch("cron.scheduler.run_job", return_value=(True, "# output", "[SILENT]", None)), \
             patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"), \
             patch("cron.scheduler._deliver_result") as deliver_mock, \
             patch("cron.scheduler.mark_job_run"):
            from cron.scheduler import tick
            with caplog.at_level(logging.INFO, logger="cron.scheduler"):
                tick(verbose=False)
        deliver_mock.assert_not_called()
        assert any(SILENT_MARKER in r.message for r in caplog.records)

    def test_silent_with_note_suppresses_delivery(self):
        with patch("cron.scheduler.get_due_jobs", return_value=[self._make_job()]), \
             patch("cron.scheduler.run_job", return_value=(True, "# output", "[SILENT] No changes detected", None)), \
             patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"), \
             patch("cron.scheduler._deliver_result") as deliver_mock, \
             patch("cron.scheduler.mark_job_run"):
            from cron.scheduler import tick
            tick(verbose=False)
        deliver_mock.assert_not_called()

    def test_silent_is_case_insensitive(self):
        with patch("cron.scheduler.get_due_jobs", return_value=[self._make_job()]), \
             patch("cron.scheduler.run_job", return_value=(True, "# output", "[silent] nothing new", None)), \
             patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"), \
             patch("cron.scheduler._deliver_result") as deliver_mock, \
             patch("cron.scheduler.mark_job_run"):
            from cron.scheduler import tick
            tick(verbose=False)
        deliver_mock.assert_not_called()

    def test_failed_job_always_delivers(self):
        """Failed jobs deliver regardless of [SILENT] in output."""
        with patch("cron.scheduler.get_due_jobs", return_value=[self._make_job()]), \
             patch("cron.scheduler.run_job", return_value=(False, "# output", "", "some error")), \
             patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"), \
             patch("cron.scheduler._deliver_result") as deliver_mock, \
             patch("cron.scheduler.mark_job_run"):
            from cron.scheduler import tick
            tick(verbose=False)
        deliver_mock.assert_called_once()

    def test_output_saved_even_when_delivery_suppressed(self):
        with patch("cron.scheduler.get_due_jobs", return_value=[self._make_job()]), \
             patch("cron.scheduler.run_job", return_value=(True, "# full output", "[SILENT]", None)), \
             patch("cron.scheduler.save_job_output") as save_mock, \
             patch("cron.scheduler._deliver_result") as deliver_mock, \
             patch("cron.scheduler.mark_job_run"):
            save_mock.return_value = "/tmp/out.md"
            from cron.scheduler import tick
            tick(verbose=False)
        save_mock.assert_called_once_with("monitor-job", "# full output")
        deliver_mock.assert_not_called()


class TestBuildJobPromptSilentHint:
    """Verify _build_job_prompt always injects [SILENT] guidance."""

    def test_hint_always_present(self):
        job = {"prompt": "Check for updates"}
        result = _build_job_prompt(job)
        assert "[SILENT]" in result
        assert "Check for updates" in result

    def test_hint_present_even_without_prompt(self):
        job = {"prompt": ""}
        result = _build_job_prompt(job)
        assert "[SILENT]" in result


class TestBuildJobPromptMissingSkill:
    """Verify that a missing skill logs a warning and does not crash the job."""

    def _missing_skill_view(self, name: str) -> str:
        return json.dumps({"success": False, "error": f"Skill '{name}' not found."})

    def test_missing_skill_does_not_raise(self):
        """Job should run even when a referenced skill is not installed."""
        with patch("tools.skills_tool.skill_view", side_effect=self._missing_skill_view):
            result = _build_job_prompt({"skills": ["ghost-skill"], "prompt": "do something"})
        # prompt is preserved even though skill was skipped
        assert "do something" in result

    def test_missing_skill_injects_user_notice_into_prompt(self):
        """A system notice about the missing skill is injected into the prompt."""
        with patch("tools.skills_tool.skill_view", side_effect=self._missing_skill_view):
            result = _build_job_prompt({"skills": ["ghost-skill"], "prompt": "do something"})
        assert "ghost-skill" in result
        assert "not found" in result.lower() or "skipped" in result.lower()

    def test_missing_skill_logs_warning(self, caplog):
        """A warning is logged when a skill cannot be found."""
        with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
            with patch("tools.skills_tool.skill_view", side_effect=self._missing_skill_view):
                _build_job_prompt({"name": "My Job", "skills": ["ghost-skill"], "prompt": "do something"})
        assert any("ghost-skill" in record.message for record in caplog.records)

    def test_valid_skill_loaded_alongside_missing(self):
        """A valid skill is still loaded when another skill in the list is missing."""

        def _mixed_skill_view(name: str) -> str:
            if name == "real-skill":
                return json.dumps({"success": True, "content": "Real skill content."})
            return json.dumps({"success": False, "error": f"Skill '{name}' not found."})

        with patch("tools.skills_tool.skill_view", side_effect=_mixed_skill_view):
            result = _build_job_prompt({"skills": ["ghost-skill", "real-skill"], "prompt": "go"})
        assert "Real skill content." in result
        assert "go" in result
