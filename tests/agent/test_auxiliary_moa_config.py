"""Contracts for task-scoped MoA auxiliary routing and retries."""

from unittest.mock import MagicMock, patch

from agent.auxiliary_client import resolve_provider_client


def test_moa_reference_can_override_global_free_only_without_widening_other_tasks(monkeypatch):
    """Only the configured MoA-reference task may use a paid OpenRouter SKU."""
    config = {"auxiliary": {"free_only": True, "moa_reference": {"free_only": False}}}
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    with (
        patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)),
        patch("hermes_cli.config.load_config_readonly", return_value=config),
        patch(
            "agent.auxiliary_client._create_openai_client",
            return_value=MagicMock(name="openrouter_client"),
        ) as build,
    ):
        client, model = resolve_provider_client(
            "openrouter", model="openai/gpt-5.5", task="moa_reference"
        )

    assert client is build.return_value
    assert model == "openai/gpt-5.5"

    with (
        patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)),
        patch("hermes_cli.config.load_config_readonly", return_value=config),
        patch("agent.auxiliary_client._create_openai_client") as build,
    ):
        client, model = resolve_provider_client(
            "openrouter", model="openai/gpt-5.5", task="title_generation"
        )

    assert client is None
    assert model is None
    build.assert_not_called()


def test_real_config_scopes_paid_lane_and_retries_to_moa_tasks(tmp_path, monkeypatch):
    """Real YAML loading preserves the MoA-only exception and retry limits."""
    import hermes_yaml as yaml

    from agent.auxiliary_client import _transient_retry_count

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "auxiliary": {
                    "free_only": True,
                    "transient_retries": 2,
                    "moa_reference": {"free_only": False, "transient_retries": 0},
                    "moa_aggregator": {"transient_retries": 0},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

    with (
        patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)),
        patch(
            "agent.auxiliary_client._create_openai_client",
            return_value=MagicMock(name="openrouter_client"),
        ) as build,
    ):
        client, model = resolve_provider_client(
            "openrouter", model="openai/gpt-5.5", task="moa_reference"
        )

    assert client is build.return_value
    assert model == "openai/gpt-5.5"

    with (
        patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)),
        patch("agent.auxiliary_client._create_openai_client") as build,
    ):
        client, model = resolve_provider_client(
            "openrouter", model="openai/gpt-5.5", task="title_generation"
        )

    assert client is None
    assert model is None
    build.assert_not_called()
    assert _transient_retry_count("moa_reference") == 0
    assert _transient_retry_count("moa_aggregator") == 0
    assert _transient_retry_count("title_generation") == 2


def test_task_scoped_transient_retry_override():
    from agent.auxiliary_client import _transient_retry_count

    config = {
        "auxiliary": {
            "transient_retries": 2,
            "moa_reference": {"transient_retries": 0},
            "moa_aggregator": {"transient_retries": 0},
        }
    }
    with patch("hermes_cli.config.load_config", return_value=config):
        assert _transient_retry_count("moa_reference") == 0
        assert _transient_retry_count("moa_aggregator") == 0
        assert _transient_retry_count("title_generation") == 2
