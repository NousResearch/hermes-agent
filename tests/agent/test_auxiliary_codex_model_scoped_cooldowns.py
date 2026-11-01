"""Reproduce goal-judge Codex pool starvation from a cooldown on another model.

A cooldown on ``gpt-5.3-codex`` says nothing about ``gpt-6-luna`` on the same
ChatGPT account. The goal-judge route must select credentials using its resolved
model rather than treating any model-specific cooldown as a provider-wide ban.

Uses the real ``agent.credential_pool`` against a disposable auth store; only the
final OpenAI constructor is mocked.
"""
import json
import time
from unittest.mock import MagicMock, patch

import pytest

BENCHED_MODEL = "gpt-5.3-codex"
JUDGE_MODEL = "gpt-6-luna"
TOKENS = ("tok-account-a", "tok-account-b")


@pytest.fixture
def codex_pool_with_cooldowns(tmp_path, monkeypatch):
    """Factory: write a disposable two-entry Codex pool cooled down for ``models``."""
    root = tmp_path / "hermes-root"
    root.mkdir()
    fake_home = tmp_path / "fakehome"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("HERMES_CODEX_BASE_URL", raising=False)
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)

    def _make(*models):
        cooldown_until = time.time() + 3600
        (root / "auth.json").write_text(json.dumps({"credential_pool": {"openai-codex": [
            {"id": f"cred-{i}", "label": f"acct-{i}", "auth_type": "oauth", "priority": i, "source": "manual",
             "access_token": tok, "refresh_token": f"rt-{i}", "expires_at_ms": 4_000_000_000_000,
             "model_cooldowns": {m: cooldown_until for m in models}}
            for i, tok in enumerate(TOKENS)
        ]}}))
        from agent.credential_pool import load_pool
        return load_pool("openai-codex")

    return _make


@pytest.fixture
def benched_codex_pool(codex_pool_with_cooldowns):
    return codex_pool_with_cooldowns(BENCHED_MODEL)


def test_explicit_model_client_ignores_sibling_model_cooldown(benched_codex_pool):
    from agent.auxiliary_client import CodexAuxiliaryClient, _build_codex_client

    # Prove the fixture models the actual asymmetry: both entries are benched for
    # the main Codex model but eligible for the configured goal-judge model.
    assert benched_codex_pool.select(model=BENCHED_MODEL) is None
    assert benched_codex_pool.select(model=JUDGE_MODEL) is not None

    with patch("agent.auxiliary_client._openai_http_client_kwargs", return_value={}), \
         patch("agent.auxiliary_client.OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        client, model = _build_codex_client(JUDGE_MODEL)

    assert isinstance(client, CodexAuxiliaryClient)
    assert model == JUDGE_MODEL
    mock_openai.assert_called_once()
    assert mock_openai.call_args.kwargs["api_key"] in TOKENS


def test_goal_judge_task_route_ignores_sibling_model_cooldown(benched_codex_pool):
    """``auxiliary.goal_judge`` config -> ``get_text_auxiliary_client`` -> Codex builder.

    Only the task config lookup is stubbed (to the shape of the trader profile's
    ``auxiliary.goal_judge``); provider routing, model resolution and the pool
    selector are the real code.
    """
    from agent.auxiliary_client import CodexAuxiliaryClient, get_text_auxiliary_client

    task_config = {"provider": "openai-codex", "model": JUDGE_MODEL}
    with patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=task_config), \
         patch("agent.auxiliary_client._openai_http_client_kwargs", return_value={}), \
         patch("agent.auxiliary_client.OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        client, _model = get_text_auxiliary_client("goal_judge")

    assert isinstance(client, CodexAuxiliaryClient)
    mock_openai.assert_called_once()
    assert mock_openai.call_args.kwargs["api_key"] in TOKENS


def test_explicit_model_client_still_blocked_by_its_own_cooldown(codex_pool_with_cooldowns, monkeypatch):
    from agent import auxiliary_client
    from agent.auxiliary_client import _build_codex_client

    pool = codex_pool_with_cooldowns(JUDGE_MODEL)
    assert pool.select(model=JUDGE_MODEL) is None
    # Keep the auth.json singleton fallback out of the picture; this asserts the pool gate.
    monkeypatch.setattr(auxiliary_client, "_read_codex_singleton_token", lambda: None)

    with patch("agent.auxiliary_client.OpenAI") as mock_openai:
        client, model = _build_codex_client(JUDGE_MODEL)

    assert (client, model) == (None, None)
    mock_openai.assert_not_called()


def test_modelless_codex_credential_resolution_stays_conservative(benched_codex_pool, monkeypatch):
    """Without a model any active model cooldown keeps blocking pool entries."""
    from agent import auxiliary_client

    monkeypatch.setattr(auxiliary_client, "_read_codex_singleton_token", lambda: None)

    assert benched_codex_pool.select() is None
    token, _base_url = auxiliary_client._resolve_codex_credential_and_base()
    assert token is None
    assert auxiliary_client._resolve_codex_credential_and_base(JUDGE_MODEL)[0] in TOKENS
