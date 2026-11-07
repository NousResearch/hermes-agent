"""The auxiliary Codex client picks its pool entry for the model it will call.

An unscoped pick honours every model cooldown on an entry, so a bench on one slug (a Codex
entitlement bench lasts a year) would otherwise hide the credential from every other model.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def test_codex_client_picks_the_pool_entry_for_its_model():
    entry = SimpleNamespace(
        runtime_api_key="codex-pool-token",
        runtime_base_url="https://chatgpt.com/backend-api/codex",
    )
    with (
        patch("agent.auxiliary_client._select_pool_entry", return_value=(True, entry)) as pick,
        patch("agent.auxiliary_client.OpenAI") as mock_openai,
    ):
        mock_openai.return_value = MagicMock()
        from agent.auxiliary_client import _build_codex_client

        client, _ = _build_codex_client("gpt-5.4")

    assert client is not None
    pick.assert_called_once_with("openai-codex", model="gpt-5.4")


def test_an_unscoped_pick_stays_unscoped():
    """A caller that names no model gets the pool's plain select, as before."""
    pool = MagicMock()
    pool.has_credentials.return_value = True
    with patch("agent.auxiliary_client._load_pool_with_credentials", return_value=pool):
        from agent.auxiliary_client import _select_pool_entry

        _select_pool_entry("openai-codex")
        _select_pool_entry("openai-codex", model="gpt-5.4")

    assert [c.kwargs for c in pool.select.call_args_list] == [{}, {"model": "gpt-5.4"}]
