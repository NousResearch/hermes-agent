"""Regression test: cron jobs must forward provider_routing.data_collection
and provider_routing.require_parameters to AIAgent, the same way they already
forward only/ignore/order/sort.

Found 2026-07-12: cron/scheduler.py::run_job built the per-job AIAgent with
providers_allowed/providers_ignored/providers_order/provider_sort taken from
config.yaml's provider_routing block, but silently dropped data_collection and
require_parameters — both fully supported by AIAgent (agent/agent_init.py) and
already correctly forwarded to delegated subagents (tools/delegate_tool.py).

Net effect before the fix: a user who set provider_routing.data_collection:
"deny" to stop OpenRouter/Nous Portal from routing to data-training providers
got that protection on their main chat and on delegate_task subagents, but
NOT on any scheduled cron job — every cron run silently ignored the setting.
"""
from unittest.mock import MagicMock, patch

from cron.scheduler import run_job


def _base_job(**overrides):
    job = {
        "id": "dc-test",
        "name": "data collection test",
        "prompt": "hello",
        "model": None,
        "provider": None,
        "provider_snapshot": "openrouter",
        "base_url": None,
    }
    job.update(overrides)
    return job


def _run_job_capturing_agent_kwargs(job, provider_routing_cfg, tmp_path):
    """Drive run_job with a given provider_routing config block and capture
    the exact kwargs AIAgent was constructed with."""
    fake_db = MagicMock()
    fake_cfg = {"provider_routing": provider_routing_cfg} if provider_routing_cfg is not None else {}

    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
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

        # scheduler.py loads config.yaml straight off disk (yaml.safe_load),
        # so write a real file into the patched _hermes_home rather than
        # mocking the loader.
        import yaml
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(fake_cfg))

        success, output, final_response, error = run_job(job)
        assert mock_agent_cls.called, f"AIAgent never constructed: {error}\n{output}"
        _, kwargs = mock_agent_cls.call_args

    return kwargs


class TestCronForwardsDataCollectionAndRequireParameters:
    def test_data_collection_deny_is_forwarded(self, tmp_path):
        job = _base_job()
        kwargs = _run_job_capturing_agent_kwargs(
            job, {"data_collection": "deny"}, tmp_path
        )
        assert kwargs.get("provider_data_collection") == "deny", (
            "provider_routing.data_collection was not forwarded to AIAgent — "
            "cron jobs would silently ignore the user's data-collection privacy setting"
        )

    def test_require_parameters_true_is_forwarded(self, tmp_path):
        job = _base_job()
        kwargs = _run_job_capturing_agent_kwargs(
            job, {"require_parameters": True}, tmp_path
        )
        assert kwargs.get("provider_require_parameters") is True

    def test_no_provider_routing_block_defaults_safely(self, tmp_path):
        """Back-compat: no provider_routing section at all must not crash and
        must default to no data-collection restriction (None) and
        require_parameters=False, matching pre-fix behavior for users who
        never configured this."""
        job = _base_job()
        kwargs = _run_job_capturing_agent_kwargs(job, None, tmp_path)
        assert kwargs.get("provider_data_collection") is None
        assert kwargs.get("provider_require_parameters") is False

    def test_existing_only_ignore_order_sort_still_forwarded(self, tmp_path):
        """Guard against regressing the four routing keys that already worked
        while adding the two that didn't."""
        job = _base_job()
        kwargs = _run_job_capturing_agent_kwargs(
            job,
            {
                "only": ["anthropic"],
                "ignore": ["together"],
                "order": ["anthropic", "google"],
                "sort": "price",
                "data_collection": "deny",
            },
            tmp_path,
        )
        assert kwargs.get("providers_allowed") == ["anthropic"]
        assert kwargs.get("providers_ignored") == ["together"]
        assert kwargs.get("providers_order") == ["anthropic", "google"]
        assert kwargs.get("provider_sort") == "price"
        assert kwargs.get("provider_data_collection") == "deny"
