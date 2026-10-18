"""Under the ``balanced`` pool strategy a cached gateway agent keeps its credential.

The gateway resolves the runtime, and so selects a pool credential, on every turn. With a key
in the cache signature, a different pick rebuilds the agent on another account and rewrites the
session's whole prompt cache. A balanced agent owns its credential instead: it rotates on
failure and re-places itself between turns (``agent.turn_credential_placement``).
"""

from types import SimpleNamespace

from gateway.run import GatewayRunner


def _signature(api_key: str, pool_balanced: bool) -> str:
    runtime = {"provider": "anthropic", "api_key": api_key,
               "credential_pool": SimpleNamespace(balanced=pool_balanced)}
    return GatewayRunner._agent_config_signature("claude-opus-4-7", runtime, ["hermes-cli"], "")


def test_a_balanced_pick_keeps_the_cached_agent_and_other_strategies_still_rebuild():
    assert _signature("key-a", True) == _signature("key-b", True)
    assert _signature("key-a", False) != _signature("key-b", False)
