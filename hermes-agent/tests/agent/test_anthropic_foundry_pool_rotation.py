"""A Foundry session must not rotate onto the Claude Code OAuth row after a 429 (#154242), even when
the attached pool is the whole ``anthropic`` pool (``_rebind_primary_credential_pool`` reloads it
with ``load_pool()`` after a fallback turn). Real config, real pool seeding and the real recovery
path; only the Anthropic client builder is stubbed. Adapted from @BearHuddleston's review patch.
"""

from __future__ import annotations

import json
import time
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

FOUNDRY_URL = "https://myres.services.ai.azure.com/anthropic"
ANTHROPIC_URL = "https://api.anthropic.com"
OAUTH_TOKEN = "sk-ant-" + "oat01-" + "A" * 40  # concatenated: push protection rejects token-shaped literals


@pytest.mark.parametrize(("base_url", "after_429"), [(FOUNDRY_URL, "azure-key"), (ANTHROPIC_URL, OAUTH_TOKEN)])
def test_rate_limit_rotation_takes_the_oauth_row_only_off_foundry(tmp_path, monkeypatch, base_url, after_429):
    from hermes_constants import get_hermes_home
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "AZURE_ANTHROPIC_KEY"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".credentials.json").write_text(json.dumps({"claudeAiOauth": {
        "accessToken": OAUTH_TOKEN, "refreshToken": "r" * 30, "expiresAt": int((time.time() + 86400) * 1000)}}))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    (get_hermes_home() / "config.yaml").write_text(
        f"model:\n  provider: anthropic\n  base_url: {base_url}\n  default: claude-sonnet-4-5\n")
    from agent.agent_runtime_helpers import recover_with_credential_pool
    from agent.client_lifecycle import ClientLifecycleMixin
    from agent.credential_pool import AUTH_TYPE_API_KEY, SOURCE_MANUAL, PooledCredential, load_pool
    from agent.error_classifier import FailoverReason
    load_pool("anthropic").add_entry(PooledCredential(
        provider="anthropic", id="azure1", label="azure", auth_type=AUTH_TYPE_API_KEY, priority=0,
        source=SOURCE_MANUAL, access_token="azure-key", base_url=base_url))
    pool = load_pool("anthropic")  # the whole pool, as a restore after a fallback turn attaches it
    assert {e.access_token for e in pool.entries()} >= {"azure-key", OAUTH_TOKEN}
    agent = SimpleNamespace(
        provider="anthropic", model="claude-sonnet-4-5", api_mode="anthropic_messages", base_url=base_url,
        api_key="azure-key", _anthropic_api_key="azure-key", _anthropic_base_url=base_url,
        _anthropic_client=MagicMock(), _build_direct_anthropic_client=lambda *_a, **_k: MagicMock(),
        _anthropic_oauth_flag=lambda _key: False, _credential_pool=pool, _credential_pool_entry_id="azure1",
        _credential_pool_revert_id=None)
    agent._swap_credential = partial(ClientLifecycleMixin._swap_credential, agent)  # type: ignore[arg-type]

    recover_with_credential_pool(agent, status_code=429, has_retried_429=True,
                                 classified_reason=FailoverReason.rate_limit)

    assert agent.api_key == agent._anthropic_api_key == after_429
    assert agent.base_url == base_url
