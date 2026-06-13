"""Tests for cron job model resolution fallback (Issue #43899).

This module tests that cron jobs correctly resolve the model from config.yaml
when no explicit model override is set on the job, matching the behavior of
gateway and CLI sessions.
"""

import os
from unittest.mock import patch, MagicMock

import pytest

# Mark as cron tests for the test runner
pytestmark = pytest.mark.cron


def _make_base_patches(tmp_path):
    """Create common patches needed for run_job tests."""
    fake_db = MagicMock()
    return [
        patch("cron.scheduler._hermes_home", tmp_path),
        patch("cron.scheduler._resolve_origin", return_value=None),
        patch("dotenv.load_dotenv"),
        patch("hermes_state.SessionDB", return_value=fake_db),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "api_key": "test-key",
                "base_url": "https://example.invalid/v1",
                "provider": "openrouter",
                "api_mode": "chat_completions",
            },
        ),
    ]


class TestCronModelResolution:
    """Tests for model resolution in cron jobs."""

    def test_model_from_job_override(self, tmp_path):
        """Model explicitly set on the job should be used."""
        from cron.scheduler import run_job

        job = {
            "id": "override-job",
            "name": "test",
            "prompt": "hello",
            "model": "claude-sonnet-4-20250514",
        }
        patches = _make_base_patches(tmp_path)
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent
            run_job(job)

        kwargs = mock_agent_cls.call_args.kwargs
        assert kwargs["model"] == "claude-sonnet-4-20250514"

    def test_model_from_env_var(self, tmp_path):
        """Model from HERMES_MODEL env var should be used when job has none."""
        from cron.scheduler import run_job

        job = {
            "id": "env-job",
            "name": "test",
            "prompt": "hello",
        }
        patches = _make_base_patches(tmp_path)
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patch("run_agent.AIAgent") as mock_agent_cls, \
             patch.dict(os.environ, {"HERMES_MODEL": "gpt-5.4-mini"}):
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent
            run_job(job)

        kwargs = mock_agent_cls.call_args.kwargs
        assert kwargs["model"] == "gpt-5.4-mini"

    def test_model_from_config_yaml_default(self, tmp_path):
        """Model should fall back to config.yaml model.default."""
        from cron.scheduler import run_job

        # Write config.yaml with model.default
        config_dir = tmp_path / ".hermes"
        config_dir.mkdir(exist_ok=True)
        (tmp_path / "config.yaml").write_text(
            "model:\n"
            "  default: kimi-k2.6\n"
            "  provider: kimi-for-coding\n"
        )

        job = {
            "id": "config-job",
            "name": "test",
            "prompt": "hello",
        }
        patches = _make_base_patches(tmp_path)
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent
            run_job(job)

        kwargs = mock_agent_cls.call_args.kwargs
        assert kwargs["model"] == "kimi-k2.6"

    def test_model_from_config_yaml_string(self, tmp_path):
        """Model should work when config.yaml model is a plain string."""
        from cron.scheduler import run_job

        (tmp_path / "config.yaml").write_text("model: gpt-5.4-mini\n")

        job = {
            "id": "string-config-job",
            "name": "test",
            "prompt": "hello",
        }
        patches = _make_base_patches(tmp_path)
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent
            run_job(job)

        kwargs = mock_agent_cls.call_args.kwargs
        assert kwargs["model"] == "gpt-5.4-mini"

    def test_no_model_configured_raises_clear_error(self, tmp_path):
        """When no model is configured anywhere, raise RuntimeError with clear message."""
        from cron.scheduler import run_job

        job = {
            "id": "no-model-job",
            "name": "test",
            "prompt": "hello",
        }
        patches = _make_base_patches(tmp_path)
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent_cls.return_value = mock_agent
            with pytest.raises(RuntimeError) as exc_info:
                run_job(job)

        error_msg = str(exc_info.value)
        assert "No model configured" in error_msg
        assert "model.default" in error_msg
        assert "no-model-job" in error_msg

    def test_model_default_empty_dict_value(self, tmp_path):
        """When model dict has no 'default' key, should raise clear error."""
        from cron.scheduler import run_job

        # config.yaml with model as dict but no 'default' key
        (tmp_path / "config.yaml").write_text(
            "model:\n"
            "  provider: openrouter\n"
            "  base_url: https://openrouter.ai/api/v1\n"
        )

        job = {
            "id": "no-default-job",
            "name": "test",
            "prompt": "hello",
        }
        patches = _make_base_patches(tmp_path)
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent_cls.return_value = mock_agent
            with pytest.raises(RuntimeError) as exc_info:
                run_job(job)

        error_msg = str(exc_info.value)
        assert "No model configured" in error_msg

    def test_model_default_is_empty_string(self, tmp_path):
        """When model.default is empty string, should raise clear error."""
        from cron.scheduler import run_job

        (tmp_path / "config.yaml").write_text(
            "model:\n"
            "  default: \"\"\n"
        )

        job = {
            "id": "empty-default-job",
            "name": "test",
            "prompt": "hello",
        }
        patches = _make_base_patches(tmp_path)
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent_cls.return_value = mock_agent
            with pytest.raises(RuntimeError) as exc_info:
                run_job(job)

        error_msg = str(exc_info.value)
        assert "No model configured" in error_msg

    def test_job_model_overrides_config(self, tmp_path):
        """Job model should take precedence over config.yaml."""
        from cron.scheduler import run_job

        (tmp_path / "config.yaml").write_text(
            "model:\n"
            "  default: kimi-k2.6\n"
        )

        job = {
            "id": "override-job",
            "name": "test",
            "prompt": "hello",
            "model": "claude-opus-4-20250514",
        }
        patches = _make_base_patches(tmp_path)
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent
            run_job(job)

        kwargs = mock_agent_cls.call_args.kwargs
        assert kwargs["model"] == "claude-opus-4-20250514"

    def test_no_config_file_uses_job_model(self, tmp_path):
        """When no config.yaml exists, job model should be used."""
        from cron.scheduler import run_job

        # No config.yaml file

        job = {
            "id": "no-config-job",
            "name": "test",
            "prompt": "hello",
            "model": "deepseek-chat",
        }
        patches = _make_base_patches(tmp_path)
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent
            run_job(job)

        kwargs = mock_agent_cls.call_args.kwargs
        assert kwargs["model"] == "deepseek-chat"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
