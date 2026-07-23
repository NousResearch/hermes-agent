"""Provider-routing parity for cron jobs.

Scheduled jobs load the same ``provider_routing`` section of config.yaml as
the CLI, messaging gateway and desktop/TUI, but the cron builder forwarded
only ``only`` / ``ignore`` / ``order`` / ``sort`` — silently dropping
``require_parameters`` and ``data_collection``.

The ``data_collection`` drop is the one that bites: a user who sets
``data_collection: "deny"`` had it honored on every interactive path and
quietly ignored on every scheduled run, sending exactly the prompts they
asked to protect to data-retaining providers.

These tests pin the full forwarding set so cron cannot drift from the other
construction paths again.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

# Ensure project root is importable.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cron.scheduler import run_job


def _job():
    return {
        "id": "routing-test",
        "name": "routing test",
        "prompt": "hello",
        "model": None,
        "provider": None,
        "provider_snapshot": "openrouter",
        "base_url": None,
    }


def _run_with_config(cfg, tmp_path):
    """Drive run_job against a temp HERMES_HOME holding ``cfg``.

    Returns the kwargs AIAgent was constructed with.
    """
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    fake_db = MagicMock()
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

        run_job(_job())

        assert mock_agent_cls.called, "AIAgent was never constructed"
        return mock_agent_cls.call_args.kwargs


def test_cron_forwards_all_provider_routing_keys(tmp_path):
    """Every provider_routing key reaches the scheduled agent."""
    kwargs = _run_with_config({
        "provider_routing": {
            "only": ["anthropic", "google"],
            "ignore": ["deepinfra"],
            "order": ["anthropic", "together"],
            "sort": "throughput",
            "require_parameters": True,
            "data_collection": "deny",
        },
    }, tmp_path)

    assert kwargs["providers_allowed"] == ["anthropic", "google"]
    assert kwargs["providers_ignored"] == ["deepinfra"]
    assert kwargs["providers_order"] == ["anthropic", "together"]
    assert kwargs["provider_sort"] == "throughput"
    assert kwargs["provider_require_parameters"] is True
    assert kwargs["provider_data_collection"] == "deny"


def test_cron_provider_routing_defaults_when_unset(tmp_path):
    """No provider_routing section → defaults, so behavior is unchanged for
    users who never configured it."""
    kwargs = _run_with_config({}, tmp_path)

    assert kwargs["providers_allowed"] is None
    assert kwargs["providers_ignored"] is None
    assert kwargs["providers_order"] is None
    assert kwargs["provider_sort"] is None
    assert kwargs["provider_require_parameters"] is False
    assert kwargs["provider_data_collection"] is None
