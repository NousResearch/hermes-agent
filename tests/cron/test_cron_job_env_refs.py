"""Cron job runtime overrides expand ``${VAR}`` refs at fire time (#107157).

Background: ``config.yaml`` values are expanded by ``hermes_cli.config._expand_env_vars`` but
``jobs.json`` string fields were never in that path — a job pinned with
``"model": "${GB10_LLM_DEFAULT}"`` shipped the literal placeholder to the provider (HTTP 404
"model does not exist"). The per-run dotenv reload already runs before model resolution, so the
values are available at the seam; what was missing is the expansion itself.

Contract (mirrors the per-run ``api_key`` expansion, #9682):
  - ``run_job()`` expands ``${VAR}`` / ``${env:VAR}`` in the job's runtime override fields
    (``model``, ``provider``, ``base_url``) on a shallow copy; the stored job keeps the templates
    so a later env change applies next tick.
  - A ref that is still unresolved after expansion blocks in preflight with the field named,
    instead of an opaque provider 404.

These tests exercise the full ``run_job`` path (real imports, mocked AIAgent +
resolve_runtime_provider against a temp HERMES_HOME).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is importable.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cron.scheduler import run_job


def _base_job(**overrides):
    job = {
        "id": "envref-test",
        "name": "envref test",
        "prompt": "hello",
        "model": None,
        "provider": None,
        "provider_snapshot": None,
        "model_snapshot": None,
        "base_url": None,
    }
    job.update(overrides)
    return job


def _run(job, tmp_path, monkeypatch=None):
    """Drive run_job against a temp config.yaml; returns ``(success, error, resolve_kwargs)``."""
    (tmp_path / "config.yaml").write_text(
        "model:\n  default: fallback-model\n  provider: openrouter\n"
    )

    resolve_kwargs = {}

    def _resolve(**kwargs):
        resolve_kwargs.update(kwargs)
        return {
            "api_key": "test-key",
            "base_url": "https://example.invalid/v1",
            "provider": kwargs.get("requested") or "openrouter",
            "api_mode": "chat_completions",
        }

    fake_db = MagicMock()
    with (
        patch("cron.scheduler._hermes_home", tmp_path),
        patch("cron.scheduler._get_hermes_home", return_value=tmp_path),
        patch("cron.scheduler_delivery._resolve_origin", return_value=None),
        patch("hermes_cli.env_loader.load_hermes_dotenv"),
        patch("hermes_cli.env_loader.reset_secret_source_cache"),
        patch("hermes_state_registry.acquire", return_value=fake_db),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=_resolve
        ),
        patch("run_agent.AIAgent") as mock_agent_cls,
    ):
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {"final_response": "ok"}
        mock_agent_cls.return_value = mock_agent
        success, _output, _final, error = run_job(job)
    return success, error, (resolve_kwargs or None)


class TestJobEnvRefExpansion:
    def test_model_and_provider_refs_expand_before_provider_resolution(self, tmp_path):
        job = _base_job(model="${ENVREF_MODEL}", provider="${ENVREF_PROVIDER}")
        import os

        os.environ["ENVREF_MODEL"] = "glm-4.7"
        os.environ["ENVREF_PROVIDER"] = "zai"
        try:
            success, error, resolve_kwargs = _run(job, tmp_path)
        finally:
            del os.environ["ENVREF_MODEL"]
            del os.environ["ENVREF_PROVIDER"]
        assert success is True, error
        assert resolve_kwargs is not None
        assert resolve_kwargs["target_model"] == "glm-4.7"
        assert resolve_kwargs["requested"] == "zai"

    def test_env_colon_ref_form_expands(self, tmp_path):
        job = _base_job(model="${env:ENVREF_MODEL_V2}")
        import os

        os.environ["ENVREF_MODEL_V2"] = "kimi-k2"
        try:
            success, error, resolve_kwargs = _run(job, tmp_path)
        finally:
            del os.environ["ENVREF_MODEL_V2"]
        assert success is True, error
        assert resolve_kwargs is not None
        assert resolve_kwargs["target_model"] == "kimi-k2"

    def test_stored_job_template_is_not_mutated(self, tmp_path):
        job = _base_job(model="${ENVREF_MODEL}")
        import os

        os.environ["ENVREF_MODEL"] = "glm-4.7"
        try:
            success, error, _kwargs = _run(job, tmp_path)
        finally:
            del os.environ["ENVREF_MODEL"]
        assert success is True, error
        assert job["model"] == "${ENVREF_MODEL}", (
            "run_job must expand on a copy: persisting the resolved value would "
            "freeze the job on one env value and leak secrets into jobs.json"
        )

    def test_unresolved_ref_blocks_with_named_field(self, tmp_path):
        import os

        job = _base_job(model="${ENVREF_DEFINITELY_UNSET_VAR_107157}")
        os.environ.pop("ENVREF_DEFINITELY_UNSET_VAR_107157", None)
        success, error, resolve_kwargs = _run(job, tmp_path)
        assert success is False
        assert error is not None
        assert "[blocked_config]" in error
        assert "unresolved env reference in field 'model'" in error
        assert resolve_kwargs is None, (
            "no provider resolution may happen for a blocked job"
        )

    def test_plain_values_are_untouched(self, tmp_path):
        job = _base_job(model="plain-model-name", provider="openrouter")
        success, error, resolve_kwargs = _run(job, tmp_path)
        assert success is True, error
        assert resolve_kwargs is not None
        assert resolve_kwargs["target_model"] == "plain-model-name"
        assert resolve_kwargs["requested"] == "openrouter"
