"""A cron job with its own provider/model/base_url pin never walks the global fallback chain (#100437).

Two vectors, one rule (``cron.scheduler._job_fallback_chain``, shared with delegate_task's pinned
children via ``hermes_cli.fallback_config.scoped_fallback_chain``):

- credential resolution: an AuthError / DNS blip on the pinned primary must not resolve a
  ``fallback_providers`` entry instead;
- mid-run: the chain handed to ``AIAgent(fallback_model=...)`` is what the conversation loop's
  provider ladder walks on a 5xx/429, so a pinned job must get none.

Unpinned jobs (which store no provider/model since jobs follow the main model) keep inheriting
the chain at both points. Drives the real ``run_job`` with AIAgent and
``resolve_runtime_provider`` mocked.
"""

from unittest.mock import MagicMock, patch

import pytest

from cron import scheduler
from cron.scheduler import _CronJobConfig, _resolve_job_runtime, run_job
from hermes_cli.auth import AuthError

_CONFIG = (
    "model:\n"
    "  default: gpt-5.6-sol\n"
    "  provider: openai-codex\n"
    "fallback_providers:\n"
    "  - provider: openrouter\n"
    "    model: z-ai/glm-5.2\n"
)
_CHAIN = [{"provider": "openrouter", "model": "z-ai/glm-5.2"}]


def _runtime(provider):
    return {"api_key": "k", "base_url": "https://example.invalid/v1", "provider": provider,
            "api_mode": "chat_completions"}


def _run(tmp_path, job, *, primary_error=None):
    """Run *job*; the primary resolve raises *primary_error* (if any), fallback entries succeed.
    Returns ``(success, error, requested providers, AIAgent kwargs — {} when never built)``."""
    (tmp_path / "config.yaml").write_text(_CONFIG, encoding="utf-8")
    requested = []

    def resolve(**kwargs):
        requested.append(kwargs.get("requested"))
        if kwargs.get("requested") == "openrouter":
            return _runtime("openrouter")
        if primary_error is not None:
            raise primary_error
        return _runtime(kwargs.get("requested") or "openai-codex")

    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler._get_hermes_home", return_value=tmp_path), \
         patch("cron.scheduler_delivery._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state_registry.acquire", return_value=MagicMock()), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=resolve), \
         patch("tools.mcp_tool_discovery.discover_mcp_tools", return_value=[]), \
         patch("run_agent.AIAgent") as agent_cls:
        agent_cls.return_value.run_conversation.return_value = {"final_response": "ok"}
        success, _output, _final, error = run_job(dict(job))
    kwargs = agent_cls.call_args.kwargs if agent_cls.called else {}
    return success, error, requested, kwargs


def _job(**pin):
    return {"id": "pin-job", "name": "pin job", "prompt": "hi", "model": None, "provider": None,
            "base_url": None, **pin}


PINS = [
    pytest.param({"provider": "anthropic", "model": "claude-sonnet-5"}, id="provider+model"),
    pytest.param({"model": "claude-sonnet-5"}, id="model-only"),
    pytest.param({"provider": "anthropic"}, id="provider-only"),
    pytest.param({"provider": "custom", "base_url": "http://127.0.0.1:8080/v1", "model": "local"},
                 id="endpoint"),
]


@pytest.mark.parametrize("pin", PINS)
def test_pinned_job_primary_auth_failure_does_not_land_on_the_global_chain(tmp_path, pin):
    success, error, requested, agent_kwargs = _run(
        tmp_path, _job(**pin), primary_error=AuthError("No credentials stored"))
    assert success is False
    assert "No credentials stored" in (error or "")
    assert "openrouter" not in requested
    assert agent_kwargs == {}  # the job failed; nothing ran on a substituted route


def test_pinned_job_dns_blip_does_not_land_on_the_global_chain(tmp_path):
    import httpx

    success, _error, requested, agent_kwargs = _run(
        tmp_path, _job(provider="xai-oauth", model="grok-4.5"),
        primary_error=httpx.ConnectError("[Errno 8] nodename nor servname provided, or not known"))
    assert success is False
    assert set(requested) == {"xai-oauth"}  # preflight probe + real resolve; no chain entry
    assert agent_kwargs == {}


@pytest.mark.parametrize("pin", PINS)
def test_resolve_job_runtime_does_not_walk_the_chain_for_a_pinned_job(pin):
    """The resolver itself, below preflight (which can be off: ``cron.preflight: false``)."""
    jc = _CronJobConfig(cfg={"fallback_providers": list(_CHAIN)}, model=pin.get("model") or "gpt-5.6-sol",
                        model_cfg={}, cron_default_provider="")
    resolve = MagicMock(side_effect=AuthError("No credentials stored"))
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", resolve):
        with pytest.raises(RuntimeError, match="No credentials stored"):
            _resolve_job_runtime(_job(**pin), "pin-job", jc)
    assert resolve.call_count == 1


def test_resolve_job_runtime_walks_the_chain_for_an_unpinned_job():
    jc = _CronJobConfig(cfg={"fallback_providers": list(_CHAIN)}, model="gpt-5.6-sol",
                        model_cfg={}, cron_default_provider="")

    def resolve(**kwargs):
        if kwargs.get("requested") == "openrouter":
            return _runtime("openrouter")
        raise AuthError("No credentials stored")

    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=resolve):
        runtime, model = _resolve_job_runtime(_job(), "free-job", jc)
    assert (runtime["provider"], model) == ("openrouter", "z-ai/glm-5.2")


@pytest.mark.parametrize("pin", PINS)
def test_pinned_job_agent_gets_no_runtime_fallback_ladder(tmp_path, pin):
    success, error, _requested, agent_kwargs = _run(tmp_path, _job(**pin))
    assert (success, error) == (True, None)
    assert agent_kwargs["fallback_model"] is None


def test_unpinned_job_still_walks_the_global_chain_at_resolve_time(tmp_path):
    success, error, requested, agent_kwargs = _run(
        tmp_path, _job(), primary_error=AuthError("No Codex credentials stored"))
    assert (success, error) == (True, None)
    assert requested == [None, "openrouter"]
    assert (agent_kwargs["provider"], agent_kwargs["model"]) == ("openrouter", "z-ai/glm-5.2")


def test_unpinned_job_agent_inherits_the_global_chain_mid_run(tmp_path):
    success, error, _requested, agent_kwargs = _run(tmp_path, _job())
    assert (success, error) == (True, None)
    assert agent_kwargs["fallback_model"] == _CHAIN


def test_legacy_snapshot_keys_are_not_a_pin(tmp_path):
    """Records written before jobs followed the main model carry *_snapshot keys; they follow the
    main model now, so they keep the chain too."""
    job = _job(provider_snapshot="openai-codex", model_snapshot="gpt-5.6-sol")
    success, error, _requested, agent_kwargs = _run(tmp_path, job)
    assert (success, error) == (True, None)
    assert agent_kwargs["fallback_model"] == _CHAIN


def test_pinned_job_same_provider_credential_pool_still_loads(tmp_path):
    """Credential-pool rotation stays on the pinned provider; it is not the fallback chain."""
    pool = MagicMock()
    pool.has_credentials.return_value = True
    with patch("agent.credential_pool.load_pool", return_value=pool) as load_pool:
        success, error, _requested, agent_kwargs = _run(
            tmp_path, _job(provider="anthropic", model="claude-sonnet-5"))
    assert (success, error) == (True, None)
    load_pool.assert_called_once_with("anthropic")
    assert agent_kwargs["credential_pool"] is pool
    assert agent_kwargs["fallback_model"] is None


def test_pinned_job_failure_notice_does_not_promise_a_backup(monkeypatch):
    monkeypatch.setattr(scheduler, "load_config", lambda: {"fallback_providers": list(_CHAIN)})
    phrase = scheduler._fallback_chain_phrase(_job(provider="anthropic", model="claude-sonnet-5"))
    assert "pinned" in phrase and "--unpin" in phrase
    assert "succeeded" not in phrase
    assert scheduler._fallback_chain_phrase(_job()) == "No backup provider succeeded either."
