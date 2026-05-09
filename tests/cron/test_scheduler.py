"""Tests for cron/scheduler.py — origin resolution, delivery routing, and error logging."""

import json
import logging
import os
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from cron.scheduler import (
    _resolve_origin,
    _resolve_delivery_target,
    _deliver_result,
    run_job,
    SILENT_MARKER,
    _build_job_prompt,
    classify_cron_delivery,
    sanitize_slack_cron_response,
)
from hadto_patches.recurring_routines import (
    SLACK_CRON_MAX_REPORT_CHARS,
    SLACK_CRON_MAX_REPORT_LINES,
    SLACK_CRON_TRIM_NOTE,
)


@pytest.fixture(autouse=True)
def _isolate_scheduler_lock(tmp_path, monkeypatch):
    """Give each test its own tick lock so xdist workers do not contend."""
    lock_dir = tmp_path / "cron-lock"
    monkeypatch.setattr("cron.scheduler._LOCK_DIR", lock_dir)
    monkeypatch.setattr("cron.scheduler._LOCK_FILE", lock_dir / ".tick.lock")


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

    def test_explicit_telegram_topic_target_with_thread_id(self):
        """deliver: 'telegram:chat_id:thread_id' parses correctly."""
        job = {
            "deliver": "telegram:-1003724596514:17",
        }
        assert _resolve_delivery_target(job) == {
            "platform": "telegram",
            "chat_id": "-1003724596514",
            "thread_id": "17",
        }

    def test_explicit_telegram_chat_id_without_thread_id(self):
        """deliver: 'telegram:chat_id' sets thread_id to None."""
        job = {
            "deliver": "telegram:-1003724596514",
        }
        assert _resolve_delivery_target(job) == {
            "platform": "telegram",
            "chat_id": "-1003724596514",
            "thread_id": None,
        }

    def test_human_friendly_label_resolved_via_channel_directory(self):
        """deliver: 'whatsapp:Alice (dm)' resolves to the real JID."""
        job = {"deliver": "whatsapp:Alice (dm)"}
        with patch(
            "gateway.channel_directory.resolve_channel_name",
            return_value="12345678901234@lid",
        ):
            result = _resolve_delivery_target(job)
        assert result == {
            "platform": "whatsapp",
            "chat_id": "12345678901234@lid",
            "thread_id": None,
        }

    def test_human_friendly_label_without_suffix_resolved(self):
        """deliver: 'telegram:My Group' resolves without display suffix."""
        job = {"deliver": "telegram:My Group"}
        with patch(
            "gateway.channel_directory.resolve_channel_name",
            return_value="-1009999",
        ):
            result = _resolve_delivery_target(job)
        assert result == {
            "platform": "telegram",
            "chat_id": "-1009999",
            "thread_id": None,
        }

    def test_raw_id_not_mangled_when_directory_returns_none(self):
        """deliver: 'whatsapp:12345@lid' passes through when directory has no match."""
        job = {"deliver": "whatsapp:12345@lid"}
        with patch(
            "gateway.channel_directory.resolve_channel_name",
            return_value=None,
        ):
            result = _resolve_delivery_target(job)
        assert result == {
            "platform": "whatsapp",
            "chat_id": "12345@lid",
            "thread_id": None,
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


class TestDeliverResultWrapping:
    """Verify that cron deliveries are wrapped with header/footer and no longer mirrored."""

    def test_delivery_wraps_content_with_header_and_footer(self):
        """Delivered content should include task name header and agent-invisible note."""
        from gateway.config import Platform

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock:
            job = {
                "id": "test-job",
                "name": "daily-report",
                "deliver": "origin",
                "origin": {"platform": "telegram", "chat_id": "123"},
            }
            _deliver_result(job, "Here is today's summary.")

        send_mock.assert_called_once()
        sent_content = send_mock.call_args.kwargs.get("content") or send_mock.call_args[0][-1]
        assert "Cronjob Response: daily-report" in sent_content
        assert "-------------" in sent_content
        assert "Here is today's summary." in sent_content
        assert "The agent cannot see this message" in sent_content

    def test_delivery_uses_job_id_when_no_name(self):
        """When a job has no name, the wrapper should fall back to job id."""
        from gateway.config import Platform

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock:
            job = {
                "id": "abc-123",
                "deliver": "origin",
                "origin": {"platform": "telegram", "chat_id": "123"},
            }
            _deliver_result(job, "Output.")

        sent_content = send_mock.call_args.kwargs.get("content") or send_mock.call_args[0][-1]
        assert "Cronjob Response: abc-123" in sent_content

    def test_delivery_skips_wrapping_when_config_disabled(self):
        """When cron.wrap_response is false, deliver raw content without header/footer."""
        from gateway.config import Platform

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}):
            job = {
                "id": "test-job",
                "name": "daily-report",
                "deliver": "origin",
                "origin": {"platform": "telegram", "chat_id": "123"},
            }
            _deliver_result(job, "Clean output only.")

        send_mock.assert_called_once()
        sent_content = send_mock.call_args.kwargs.get("content") or send_mock.call_args[0][-1]
        assert sent_content == "Clean output only."
        assert "Cronjob Response" not in sent_content
        assert "The agent cannot see" not in sent_content

    def test_no_mirror_to_session_call(self):
        """Cron deliveries should NOT mirror into the gateway session."""
        from gateway.config import Platform

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})), \
             patch("gateway.mirror.mirror_to_session") as mirror_mock:
            job = {
                "id": "test-job",
                "deliver": "origin",
                "origin": {"platform": "telegram", "chat_id": "123"},
            }
            _deliver_result(job, "Hello!")

        mirror_mock.assert_not_called()

    def test_origin_delivery_preserves_thread_id(self):
        """Origin delivery should forward thread_id to the send helper."""
        from gateway.config import Platform

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}

        job = {
            "id": "test-job",
            "name": "topic-job",
            "deliver": "origin",
            "origin": {
                "platform": "telegram",
                "chat_id": "-1001",
                "thread_id": "17585",
            },
        }

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock:
            _deliver_result(job, "hello")

        send_mock.assert_called_once()
        assert send_mock.call_args.kwargs["thread_id"] == "17585"


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
        fake_db.end_session.assert_called_once()
        call_args = fake_db.end_session.call_args
        assert call_args[0][0].startswith("cron_test-job_")
        assert call_args[0][1] == "cron_complete"
        fake_db.close.assert_called_once()

    def test_run_job_empty_response_returns_empty_not_placeholder(self, tmp_path):
        """Empty final_response should stay empty for delivery logic (issue #2234).
        
        The placeholder '(No response generated)' should only appear in the
        output log, not in the returned final_response that's used for delivery.
        """
        job = {
            "id": "silent-job",
            "name": "silent test",
            "prompt": "do work via tools only",
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
            # Agent did work via tools but returned no text
            mock_agent.run_conversation.return_value = {"final_response": ""}
            mock_agent_cls.return_value = mock_agent

            success, output, final_response, error = run_job(job)

        assert success is True
        assert error is None
        # final_response should be empty for delivery logic to skip
        assert final_response == ""
        # But the output log should show the placeholder
        assert "(No response generated)" in output

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

    def test_repeated_no_change_hadto_heartbeat_suppresses_delivery(self):
        """A repeated no-worker/no-proof HAD-445-style heartbeat is Slack noise."""
        job = {
            "id": "6342b03b3de2",
            "name": "hadto-hourly-heartbeat",
            "deliver": "slack:C0APDPNRGVB",
            "role": "report",
            "scope": "global",
            "schedule": {"kind": "cron", "expr": "0 * * * *"},
        }
        previous = "Heartbeat: active 0, selected HAD-445, stale 0, blocked 0, cron failures 0, next no-proof."
        current = "Heartbeat: active 0, selected HAD-445, stale 0, blocked 0, cron failures 0, next no-proof."

        with patch("cron.scheduler.get_due_jobs", return_value=[job]), \
             patch("cron.scheduler.load_recent_job_responses", return_value=[previous]), \
             patch("cron.scheduler.advance_next_run"), \
             patch("cron.scheduler.run_job", return_value=(True, "# output", current, None)), \
             patch("cron.scheduler.save_job_output", return_value="/tmp/out.md") as save_mock, \
             patch("cron.scheduler._deliver_result") as deliver_mock, \
             patch("cron.scheduler.mark_job_run"):
            from cron.scheduler import tick
            tick(verbose=False)

        save_mock.assert_called_once_with("6342b03b3de2", "# output")
        deliver_mock.assert_not_called()

    def test_heartbeat_material_trigger_delivers(self):
        job = {
            "id": "6342b03b3de2",
            "name": "hadto-hourly-heartbeat",
            "deliver": "slack:C0APDPNRGVB",
            "role": "report",
            "scope": "global",
            "schedule": {"kind": "cron", "expr": "0 * * * *"},
        }
        current = (
            "Routine Delivery Gate: Material Trigger=live worker count changed; Emit=yes; "
            "Durable Artifact=codex_delegate run list; Circulation=Slack heartbeat; "
            "Productive Fallback=implementation; Systematic Defect Action=none.\n"
            "Heartbeat: active 1, selected HAD-445, stale 0, blocked 0, cron failures 0, next worker running."
        )

        with patch("cron.scheduler.get_due_jobs", return_value=[job]), \
             patch("cron.scheduler.load_recent_job_responses", return_value=[]), \
             patch("cron.scheduler.advance_next_run"), \
             patch("cron.scheduler.run_job", return_value=(True, "# output", current, None)), \
             patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"), \
             patch("cron.scheduler._deliver_result") as deliver_mock, \
             patch("cron.scheduler.mark_job_run"):
            from cron.scheduler import tick
            tick(verbose=False)

        deliver_mock.assert_called_once()

    def test_profile_interview_no_fresh_material_suppresses_delivery(self):
        job = {
            "id": "d683949c21a5",
            "name": "operator-profile-interview-slack",
            "deliver": "slack",
            "role": "study",
            "scope": "hermes",
            "schedule": {"kind": "interval", "minutes": 60},
        }
        current = "Profile interview: no new packet. The reporting preference is already captured; same enforcement miss."

        with patch("cron.scheduler.get_due_jobs", return_value=[job]), \
             patch("cron.scheduler.load_recent_job_responses", return_value=[]), \
             patch("cron.scheduler.advance_next_run"), \
             patch("cron.scheduler.run_job", return_value=(True, "# output", current, None)), \
             patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"), \
             patch("cron.scheduler._deliver_result") as deliver_mock, \
             patch("cron.scheduler.mark_job_run"):
            from cron.scheduler import tick
            tick(verbose=False)

        deliver_mock.assert_not_called()

    def test_slack_self_audit_sections_stripped_before_delivery(self):
        job = {
            "id": "had-655",
            "name": "hadto-governance-noise",
            "deliver": "slack:C0APDPNRGVB",
            "role": "coordinate",
            "scope": "global",
            "schedule": {"kind": "interval", "minutes": 60},
        }
        useful = "HAD-655: Slack cron delivery strips governance noise. Next: watch the next scheduled run."
        noisy = (
            f"{useful}\n\n"
            "Trust Contract\n"
            "Commitment=prove this run was useful\n"
            "Artifact=cron/output/had-655\n"
            "Verification=local audit\n"
            "Outcome=kept\n\n"
            "Persistence Ratchet\n"
            "Evidence=same recurring audit text\n"
            "Decisions=repeat the same governance frame\n"
            "Artifacts=none\n"
            "Carry-forward=repeat again\n\n"
            "Coverage Completion\n"
            "Evidence Coverage=operator attention consumed\n"
            "Contradictions=none\n"
            "Closure Basis=status restatement\n\n"
            "Value Surfaces\n"
            "Durable Store=none\n"
            "Circulation=Slack post\n"
            "Closure Rule=do not count chatter\n\n"
            "Attention Budget\n"
            "Attention Cost=high\n"
            "Decision Value=low\n"
            "Focus Effect=report spam\n\n"
            "Aggregate Stewardship\n"
            "Shared Provider Concentration=unchanged\n"
            "Portfolio State=unchanged\n\n"
            "First Proof Point\n"
            "Seed Surface=Slack cron\n"
            "Success Signal=short report\n\n"
            "Geometry Shaping\n"
            "Default Changed=none\n"
            "Policy-vs-Path=restated doctrine"
        )

        with patch("cron.scheduler.get_due_jobs", return_value=[job]), \
             patch("cron.scheduler.load_recent_job_responses", return_value=[]), \
             patch("cron.scheduler.advance_next_run"), \
             patch("cron.scheduler.run_job", return_value=(True, "# durable output", noisy, None)), \
             patch("cron.scheduler.save_job_output", return_value="/tmp/out.md") as save_mock, \
             patch("cron.scheduler._deliver_result") as deliver_mock, \
             patch("cron.scheduler.mark_job_run"):
            from cron.scheduler import tick
            tick(verbose=False)

        save_mock.assert_called_once_with("had-655", "# durable output")
        deliver_mock.assert_called_once_with(job, useful)

    def test_slack_self_audit_only_suppresses_delivery(self):
        job = {
            "id": "had-655",
            "name": "hadto-governance-noise",
            "deliver": "slack:C0APDPNRGVB",
            "role": "coordinate",
            "scope": "global",
            "schedule": {"kind": "interval", "minutes": 60},
        }
        governance_only = (
            "Trust Contract\n"
            "Commitment=repeat the control frame\n"
            "Artifact=cron/output/had-655\n"
            "Verification=self-audit\n\n"
            "Attention Budget\n"
            "Attention Cost=high\n"
            "Decision Value=none\n"
            "Focus Effect=low-yield alerting\n\n"
            "Geometry Shaping\n"
            "Default Changed=none\n"
            "Policy-vs-Path=doctrine restated"
        )

        with patch("cron.scheduler.get_due_jobs", return_value=[job]), \
             patch("cron.scheduler.load_recent_job_responses", return_value=[]), \
             patch("cron.scheduler.advance_next_run"), \
             patch("cron.scheduler.run_job", return_value=(True, "# durable output", governance_only, None)), \
             patch("cron.scheduler.save_job_output", return_value="/tmp/out.md") as save_mock, \
             patch("cron.scheduler._deliver_result") as deliver_mock, \
             patch("cron.scheduler.mark_job_run"):
            from cron.scheduler import tick
            tick(verbose=False)

        save_mock.assert_called_once_with("had-655", "# durable output")
        deliver_mock.assert_not_called()

    def test_slack_cron_report_is_hard_capped(self):
        job = {
            "id": "had-655",
            "name": "hadto-long-report",
            "deliver": "slack:C0APDPNRGVB",
            "role": "report",
            "scope": "global",
            "schedule": {"kind": "interval", "minutes": 60},
        }
        long_report = "Update: " + ("HAD-655 dispatch boundary verified. " * 80)

        decision = sanitize_slack_cron_response(job, long_report)

        assert decision["suppress"] is False
        assert decision["capped"] is True
        assert len(decision["content"]) <= SLACK_CRON_MAX_REPORT_CHARS
        assert SLACK_CRON_TRIM_NOTE in decision["content"]

    def test_slack_cron_report_shape_is_hard_capped(self):
        job = {
            "id": "had-655",
            "name": "hadto-many-lines",
            "deliver": "slack:C0APDPNRGVB",
            "role": "report",
            "scope": "global",
            "schedule": {"kind": "interval", "minutes": 60},
        }
        many_lines = "\n".join(f"Update: HAD-655 dispatch line {i}" for i in range(30))

        decision = sanitize_slack_cron_response(job, many_lines)

        assert decision["suppress"] is False
        assert decision["capped"] is True
        assert len(decision["content"].splitlines()) <= SLACK_CRON_MAX_REPORT_LINES
        assert SLACK_CRON_TRIM_NOTE in decision["content"]

    def test_non_slack_delivery_keeps_original_governance_output(self):
        job = {
            "id": "had-655",
            "name": "telegram-governance-output",
            "deliver": "telegram:123",
            "role": "coordinate",
            "scope": "global",
            "schedule": {"kind": "interval", "minutes": 60},
        }
        noisy = (
            "HAD-655: Telegram delivery is not Slack-gated.\n\n"
            "Trust Contract\n"
            "Commitment=preserve non-Slack behavior"
        )

        with patch("cron.scheduler.get_due_jobs", return_value=[job]), \
             patch("cron.scheduler.load_recent_job_responses", return_value=[]), \
             patch("cron.scheduler.advance_next_run"), \
             patch("cron.scheduler.run_job", return_value=(True, "# durable output", noisy, None)), \
             patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"), \
             patch("cron.scheduler._deliver_result") as deliver_mock, \
             patch("cron.scheduler.mark_job_run"):
            from cron.scheduler import tick
            tick(verbose=False)

        deliver_mock.assert_called_once_with(job, noisy)

    def test_classify_outreach_blocked_without_fallback_is_suppressed(self):
        job = {
            "id": "outreach",
            "name": "hadto-outreach",
            "deliver": "slack",
            "role": "report",
            "scope": "global",
            "schedule": {"kind": "interval", "minutes": 60},
            "prompt": "Run cold outreach when approval is available.",
        }
        decision = classify_cron_delivery(
            job,
            "Outreach remains blocked by approval/public-touch constraints, so I audited the policy again.",
            previous_responses=[],
        )
        assert decision["suppress"] is True
        assert decision["reason"] == "outreach_blocked_without_productive_fallback"


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

    def test_study_role_injects_execution_guidance(self):
        job = {"prompt": "Study Chapter 7", "role": "study"}
        result = _build_job_prompt(job)
        assert "classified as role=study" in result
        assert "execution loop, not a passive summary" in result
        assert "self_improvement_pipeline" in result
        assert "prevents rediscovering the same gap" in result

    def test_non_study_role_does_not_inject_study_guidance(self):
        job = {"prompt": "Send report", "role": "report"}
        result = _build_job_prompt(job)
        assert "classified as role=study" not in result

    def test_recurring_control_loop_injects_persistence_ratchet_guidance(self):
        job = {
            "id": "coord123",
            "prompt": "Coordinate backlog",
            "role": "coordinate",
            "scope": "global",
            "schedule": {"kind": "interval", "minutes": 60},
        }
        result = _build_job_prompt(job)
        assert "bounded persistence-ratchet check" in result
        assert "Persistence Ratchet" in result
        assert "operator-value, anti-make-work, and leading-indicator" in result
        assert "cron/output/coord123" in result
        assert "Trust Contract" in result
        assert "one_shot_disconnected|repeated_trust_bearing" in result
        assert "discovery|execution|bridge" in result
        assert "Dignity=<how this run preserved operator agency" in result
        assert "Capability=<how this run compounded operator capability" in result
        assert "Viability=<how this run kept the surrounding system stable" in result
        assert "Fast Loop=<what this run may change immediately>" in result
        assert "Slow Loop=<what requires slower governance or rule revision>" in result
        assert "Escalate When=<the concrete checkpoint" in result
        assert "First Proof Point=<one bounded protected seed" in result
        assert "Seed Surface=<one concrete issue/job/repo path/operator workflow" in result
        assert "Protection Assumptions=<what keeps this seed bounded" in result
        assert "Success Signal=<the observable result that proves the seed worked>" in result
        assert "Imitation Path=<what another site would copy only after the signal holds>" in result
        assert "Why First=<why this seed is the right first nucleation site>" in result
        assert "Geometry Shaping=<the concrete path-shaping moves this run made" in result
        assert "Default Changed=<which default or default path changed>" in result
        assert "Channel Opened=<which channel, route, or evidence surface opened>" in result
        assert "Friction Changed=<which friction was added or removed>" in result
        assert "Stale Path Pruned=<which stale branch, route, or behavior was pruned>" in result
        assert "Policy-vs-Path=<how this changed the path itself rather than only restating doctrine>" in result
        assert "Coverage Completion" in result
        assert "Evidence Coverage=<what evidence or behavior is now covered by this run>" in result
        assert "Contradictions=<none or the exact conflict that still breaks confidence>" in result
        assert "Uncovered Region=<the next behavior, claim, or surface that still lacks enough coverage>" in result
        assert "Closure Basis=<why this run counts as real progress rather than a status restatement>" in result
        assert "Value Surfaces" in result
        assert "Durable Store=<the durable artifact" in result
        assert "Circulation=<the cheap coordination outputs" in result
        assert "Closure Rule=<why circulation-only output cannot count as closure>" in result
        assert "Attention Budget" in result
        assert "Attention Cost=<how much operator attention this run consumed or asked for>" in result
        assert "Decision Value=<what decision, judgment update, or durable state change this attention bought>" in result
        assert "Focus Effect=<whether this shaped useful focus or drifted into low-yield alerting/report spam>" in result
        assert "Aggregate Stewardship" in result
        assert "Shared Provider Concentration=<which providers/models/base URLs or auth surfaces concentrate risk across jobs>" in result
        assert "Dependency Choke Points=<which shared artifacts, routes, queues, repos, or operator surfaces many jobs depend on>" in result
        assert "Verification Debt=<which jobs or claims still lack enough verification and how much of the portfolio that debt touches>" in result
        assert "Synchronized Failure Risk=<what could fail many loops at once>" in result
        assert "Portfolio State=<whether the portfolio is locally green but globally fragile, or why it is healthy>" in result
        assert "Shared Artifact=<the durable shared artifact that carries this portfolio view across runs" in result
        assert "Ownership Decisions" in result
        assert "Selection=<selected|denied|skipped reason=..." in result
        assert "Execution=<started|skipped reason=live_execution|planning_only|repo_unresolved>" in result
        assert "Delegate=<delegated|denied|skipped reason=delegate_allowed|delegate_denied|owner_label_missing|owner_label_mismatch|owner_label_conflict|human_owned|already_undelegated|writeback_skipped>" in result
        assert "workspace-orchestrator:HAD-" in result
        assert "owner_label_missing, owner_label_mismatch, owner_label_conflict" in result

    def test_hadto_heartbeat_injects_routine_delivery_gate(self):
        job = {
            "id": "6342b03b3de2",
            "name": "hadto-hourly-heartbeat",
            "prompt": "Post David's hourly Hadto heartbeat.",
            "role": "report",
            "scope": "global",
            "deliver": "slack:C0APDPNRGVB",
            "schedule": {"kind": "cron", "expr": "0 * * * *"},
        }
        result = _build_job_prompt(job)
        assert "Recurring Routine Governance" in result
        assert "Routine Delivery Gate" in result
        assert "selected issue changed" in result
        assert "A no-worker/no-proof/no-change heartbeat must be [SILENT]" in result
        assert "Durable Artifact=<file/issue/PR/comment/state that compounds>" in result

    def test_profile_interview_injects_fresh_material_silence_guard(self):
        job = {
            "id": "d683949c21a5",
            "name": "operator-profile-interview-slack",
            "prompt": "Run David's Slack-native operator-profile interview routine.",
            "role": "study",
            "scope": "hermes",
            "deliver": "slack",
            "schedule": {"kind": "interval", "minutes": 60},
        }
        result = _build_job_prompt(job)
        assert "Profile-interview-specific rule" in result
        assert "fresh material question or tagged answer" in result
        assert "Do not report the same enforcement miss to Slack" in result

    def test_outreach_blocked_injects_productive_fallback_lanes(self):
        job = {
            "id": "outreach",
            "name": "hadto-outreach",
            "prompt": "Cold outreach is blocked on David approval and public-touch constraints.",
            "role": "report",
            "scope": "global",
            "deliver": "slack",
            "schedule": {"kind": "interval", "minutes": 60},
        }
        result = _build_job_prompt(job)
        assert "Outreach appears approval/public-touch blocked" in result
        assert "vertical opportunity research" in result
        assert "source discovery" in result
        assert "ontology enrichment" in result
        assert "lead research" in result

    def test_one_shot_job_does_not_inject_persistence_ratchet_guidance(self):
        job = {
            "id": "oneshot123",
            "prompt": "Send report",
            "role": "coordinate",
            "scope": "global",
            "schedule": {"kind": "once", "run_at": "2030-01-01T00:00:00+00:00"},
        }
        result = _build_job_prompt(job)
        assert "bounded persistence-ratchet check" not in result
        assert "Trust Contract" not in result
        assert "Coverage Completion" not in result
        assert "Attention Budget" not in result
        assert "Ownership Decisions" not in result


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


class TestTickAdvanceBeforeRun:
    """Verify that tick() calls advance_next_run before run_job for crash safety."""

    def test_advance_called_before_run_job(self, tmp_path):
        """advance_next_run must be called before run_job to prevent crash-loop re-fires."""
        call_order = []

        def fake_advance(job_id):
            call_order.append(("advance", job_id))
            return True

        def fake_run_job(job):
            call_order.append(("run", job["id"]))
            return True, "output", "response", None

        fake_job = {
            "id": "test-advance",
            "name": "test",
            "prompt": "hello",
            "enabled": True,
            "schedule": {"kind": "cron", "expr": "15 6 * * *"},
        }

        with patch("cron.scheduler.get_due_jobs", return_value=[fake_job]), \
             patch("cron.scheduler.advance_next_run", side_effect=fake_advance) as adv_mock, \
             patch("cron.scheduler.run_job", side_effect=fake_run_job), \
             patch("cron.scheduler.save_job_output", return_value=tmp_path / "out.md"), \
             patch("cron.scheduler.mark_job_run"), \
             patch("cron.scheduler._deliver_result"):
            from cron.scheduler import tick
            executed = tick(verbose=False)

        assert executed == 1
        adv_mock.assert_called_once_with("test-advance")
        # advance must happen before run
        assert call_order == [("advance", "test-advance"), ("run", "test-advance")]
