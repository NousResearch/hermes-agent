"""Cron stamps a hard job --model pin onto the agent so the runtime ladder cannot clobber it."""

from unittest.mock import MagicMock, patch

from cron.scheduler import run_job


class _FakeAgent:
    def __init__(self, **kwargs):
        self.model = kwargs.get("model")
        self.provider = kwargs.get("provider")
        self.run_conversation = MagicMock(return_value={"final_response": "ok"})


_RUNTIME = {
    "api_key": "test-key",
    "base_url": "https://example.invalid/v1",
    "provider": "pinnedlocal",
    "requested_provider": "pinnedlocal",
    "api_mode": "chat_completions",
}


def _run(tmp_path, job, runtime=None):
    fake_db = MagicMock()
    agents = []

    def _agent(**kwargs):
        inst = _FakeAgent(**kwargs)
        agents.append(inst)
        return inst

    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler_delivery._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state_registry.acquire", return_value=fake_db), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider",
               return_value=runtime or _RUNTIME), \
         patch("tools.mcp_tool_discovery.discover_mcp_tools", return_value=[]), \
         patch("run_agent.AIAgent", side_effect=_agent):
        success, _, _, error = run_job(job)
    return success, error, agents


def _write_cfg(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "model:\n"
        "  default: default-model\n"
        "  provider: defaultok\n"
        "fallback_providers:\n"
        "  - provider: sinkok\n"
        "    model: sink-model-y\n",
        encoding="utf-8",
    )


def test_pinned_cron_job_stamps_fallback_pin_model(tmp_path):
    _write_cfg(tmp_path)
    success, error, agents = _run(
        tmp_path,
        {
            "id": "pin-runtime",
            "name": "pin runtime",
            "prompt": "hi",
            "provider": "pinnedlocal",
            "model": "pinned-model-x",
        },
    )
    assert success is True, error
    assert agents[0]._fallback_pin_model == "pinned-model-x"


def test_unpinned_cron_job_does_not_stamp_fallback_pin_model(tmp_path):
    _write_cfg(tmp_path)
    success, error, agents = _run(
        tmp_path,
        {
            "id": "unpinned-runtime",
            "name": "unpinned runtime",
            "prompt": "hi",
            "provider_snapshot": "defaultok",
            "model_snapshot": "default-model",
        },
    )
    assert success is True, error
    assert not hasattr(agents[0], "_fallback_pin_model")
