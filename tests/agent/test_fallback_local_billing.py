"""A background run does not switch to a local model after a billing refusal.

Contract (``agent/fallback_local_billing.py``, applied in ``try_activate_fallback`` on the
resolved endpoint, so it holds for whichever chain the run walks): a cron run, a subagent of a
cron run, or a Kanban worker process leaving a billing-refused provider skips a loopback entry
and takes the next cloud entry. An interactive session, a non-billing reason, and the
``fallback.background_local_when_billing_blocked`` opt-in (read from the profile's config.yaml)
keep the local switch.
"""

import os
import weakref
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.error_classifier import FailoverReason
from run_agent import AIAgent

_LOCAL = {"provider": "custom", "model": "gemma-4-e4b", "base_url": "http://127.0.0.1:1234/v1"}
_CLOUD = {"provider": "openrouter", "model": "z-ai/glm-5.2", "base_url": "https://openrouter.ai/api/v1"}


class _CronParent:
    platform = "cron"


def _agent(platform):
    with patch("model_tools.get_tool_definitions", return_value=[]), \
         patch("model_tools.check_toolset_requirements", return_value={}), \
         patch("agent.process_bootstrap.OpenAI"):
        agent = AIAgent(
            api_key="k", base_url="https://api.x.ai/v1", provider="xai", model="grok-4.20",
            quiet_mode=True, skip_context_files=True, skip_memory=True, platform=platform,
            fallback_model=[dict(_LOCAL), dict(_CLOUD)],
        )
    agent.client = MagicMock()
    return agent


def _client(_provider, *, explicit_base_url=None, **_kw):
    client = MagicMock()
    client.base_url, client.api_key = explicit_base_url, "k"
    return client, _kw.get("model")


@pytest.mark.parametrize("who, reason, opt_in, expected", [
    ("cron", FailoverReason.billing, False, _CLOUD["model"]),
    ("cron-subagent", FailoverReason.billing, False, _CLOUD["model"]),
    ("kanban-worker", FailoverReason.billing, False, _CLOUD["model"]),
    ("cli", FailoverReason.billing, False, _LOCAL["model"]),
    ("cron", FailoverReason.rate_limit, False, _LOCAL["model"]),
    ("cron", FailoverReason.billing, True, _LOCAL["model"]),
])
def test_background_billing_refusal_skips_local_fallback(monkeypatch, who, reason, opt_in, expected):
    if opt_in:
        (Path(os.environ["HERMES_HOME"]) / "config.yaml").write_text(
            "fallback:\n  background_local_when_billing_blocked: true\n", encoding="utf-8")
    if who == "kanban-worker":
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_1")
    agent = _agent({"cron": "cron", "cron-subagent": "subagent"}.get(who, "cli"))
    parent = _CronParent()
    if who == "cron-subagent":
        agent._delegate_parent_ref = weakref.ref(parent)

    with patch("agent.auxiliary_client.resolve_provider_client", side_effect=_client), \
         patch.object(AIAgent, "_ensure_lmstudio_runtime_loaded", lambda self: None):
        assert agent._try_activate_fallback(reason=reason) is True

    assert agent.model == expected
