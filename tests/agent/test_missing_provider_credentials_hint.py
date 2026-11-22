"""The "no credentials" error for an explicit provider names a remedy that can actually work.

Deriving the env var from the provider id (``f"{id.upper()}_API_KEY"``) invents names nothing reads:
``MINIMAX-OAUTH_API_KEY`` for OAuth ids (#114405, #78996) and ``ALIBABA_API_KEY`` where the registry
reads ``DASHSCOPE_API_KEY``. Both the auxiliary ladder and main-agent init share one helper.
"""
import pytest
import hermes_yaml as yaml

from agent.auxiliary_unavailable import missing_provider_credentials_message
from hermes_cli.auth import PROVIDER_REGISTRY


@pytest.mark.parametrize("provider, expected, forbidden", [
    ("minimax-oauth", "hermes auth add minimax-oauth", "MINIMAX-OAUTH_API_KEY"),
    ("alibaba", "Set the DASHSCOPE_API_KEY environment variable", "ALIBABA_API_KEY"),
])
def test_aux_ladder_names_registry_remedy_for_explicit_provider(tmp_path, monkeypatch, provider, expected, forbidden):
    """Real call_llm → ladder with the compression provider pinned and no credentials anywhere."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for var in ("MINIMAX_API_KEY", "DASHSCOPE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": provider, "default": "test-model"},
        "auxiliary": {"compression": {"provider": provider, "model": "test-model"}},
    }), encoding="utf-8")
    from agent.auxiliary_client import call_llm
    from agent.auxiliary_unavailable import AuxiliaryClientUnavailable

    with pytest.raises(AuxiliaryClientUnavailable) as excinfo:
        call_llm(task="compression", messages=[{"role": "user", "content": "hi"}], max_tokens=5)
    assert expected in str(excinfo.value)
    assert forbidden not in str(excinfo.value)


def test_main_init_shares_helper_and_no_registry_provider_gets_an_invented_env_var(tmp_path, monkeypatch):
    """agent_init raises the same text as the helper; no registry id yields a made-up ``<ID>_API_KEY``."""
    from types import SimpleNamespace

    from agent.agent_init import _routed_client_kwargs

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = SimpleNamespace(provider="minimax-oauth", model="m", base_url=None, api_key=None,
                            _fallback_activated=False, _explicit_provider="minimax-oauth")
    with pytest.raises(RuntimeError, match=r"hermes auth add minimax-oauth"):
        _routed_client_kwargs(agent, None, 60)

    for pid, pconfig in PROVIDER_REGISTRY.items():
        invented = f"{pid.upper()}_API_KEY"
        message = missing_provider_credentials_message(pid)
        if invented in message:
            assert invented in pconfig.api_key_env_vars, (pid, message)
        if not pconfig.api_key_env_vars:
            assert "_API_KEY environment variable" not in message, (pid, message)


def _write_exhausted_codex_pool(home, *, count: int, reset_at: float):
    """Persist a Codex OAuth pool the way ``mark_exhausted`` leaves it after a 429."""
    import json
    import time

    from agent.credential_pool import PooledCredential

    entries = [PooledCredential(
        provider="openai-codex", id=f"codex-{i}", label=f"acct{i}", auth_type="oauth", priority=i,
        source="manual", access_token=f"eyJ.fake.{i}", refresh_token="rt", last_status="exhausted",
        last_status_at=time.time() - 60, last_error_code=429, last_error_reason="usage_limit_reached",
        last_error_reset_at=reset_at).to_dict() for i in range(count)]
    (home / "auth.json").write_text(json.dumps({"version": 1, "credential_pool": {"openai-codex": entries}}))


def test_exhausted_oauth_pool_reports_cooldown_not_missing_credentials(tmp_path, monkeypatch):
    """#56810: a valid OAuth grant in 429 cooldown is not "no credentials … hermes auth add".

    Both raise sites (main-agent init and the auxiliary ladder) render the pool state: how many
    credentials are benched and when the next one resets.
    """
    import time
    from types import SimpleNamespace

    from agent.agent_init import _routed_client_kwargs

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))  # never adopt the host's Codex CLI tokens
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    reset_at = time.time() + 3 * 3600
    _write_exhausted_codex_pool(tmp_path, count=2, reset_at=reset_at)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "openai-codex", "default": "test-model"},
        "auxiliary": {"compression": {"provider": "openai-codex", "model": "test-model"}},
    }), encoding="utf-8")
    expected_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(reset_at))

    agent = SimpleNamespace(provider="openai-codex", model="test-model", base_url=None, api_key=None,
                            _fallback_activated=False, _explicit_provider="openai-codex")
    with pytest.raises(RuntimeError) as excinfo:
        _routed_client_kwargs(agent, None, 60)
    message = str(excinfo.value)
    assert "all 2 credentials are cooling down" in message
    assert expected_time in message
    assert "no credentials were found" not in message

    from agent.auxiliary_client import call_llm
    from agent.auxiliary_unavailable import AuxiliaryClientUnavailable

    with pytest.raises(AuxiliaryClientUnavailable) as aux_exc:
        call_llm(task="compression", messages=[{"role": "user", "content": "hi"}], max_tokens=5)
    assert "cooling down" in str(aux_exc.value)
    assert "no credentials were found" not in str(aux_exc.value)


def test_elapsed_cooldown_or_empty_pool_keeps_missing_credentials_text(tmp_path, monkeypatch):
    """Control: a pool whose cooldown already elapsed (or no pool at all) is not called "cooling down"."""
    import time

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert "cooling down" not in missing_provider_credentials_message("openai-codex")
    _write_exhausted_codex_pool(tmp_path, count=1, reset_at=time.time() - 60)
    assert "cooling down" not in missing_provider_credentials_message("openai-codex")
