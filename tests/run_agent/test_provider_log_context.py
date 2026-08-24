"""Regression tests: canonical provider/base_url/model fragment on API log lines.

Several conversation-loop log lines report which provider an API call went
to. Historically each site formatted these fields by hand: some printed only
``model=``, some printed ``model=/provider=`` in varying orders, and none of
the empty-retry / fallback / success lines carried ``base_url``. With custom
providers the bare ``provider=custom`` value is ambiguous — only ``base_url``
identifies the actual gateway — so omitting it makes failure triage
guesswork.

The contract under test: every one of these lines embeds the shared
``provider=<p> base_url=<u> model=<m>`` fragment produced by
``agent.log_context.model_provider_fields`` (mirroring the existing
conventions of ``_client_log_context()`` and ``agent/stream_diag.py``).
"""
import logging
import re
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.log_context import model_provider_fields
from run_agent import AIAgent


def _tool_defs(*names):
    """Helper: create minimal tool definitions for given names."""
    return [
        {
            "type": "function", "function": {
                "name": name,
                "description": "test tool",
                "parameters": {"type": "object", "properties": {}},
            }
        }
        for name in names
    ]


def _response(*, content, finish_reason, tool_calls=None, usage=None):
    """Helper: create a minimal API response object."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=usage)


_FRAGMENT_RE = re.compile(r"provider=(\S+) base_url=(\S+) model=(\S+)")


def _make_custom_gateway_agent(provider="custom",
                               base_url="https://api.kilo.ai/api/gateway/v1/",
                               model="stealth/ox-alpha"):
    """Build a bare agent whose identity looks like an ambiguous custom gateway."""
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs("todo")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://api.kilo.ai/api/gateway/v1/",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.provider = provider
    agent.base_url = base_url
    agent.model = model
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.valid_tool_names = {"todo"}
    return agent


class TestModelProviderFields:
    def test_fragment_shape_and_order(self):
        agent = SimpleNamespace(
            provider="custom:kilo",
            base_url="https://api.kilo.ai/api/gateway/v1/",
            model="stealth/ox-alpha",
        )
        assert model_provider_fields(agent) == (
            "provider=custom:kilo "
            "base_url=https://api.kilo.ai/api/gateway/v1/ "
            "model=stealth/ox-alpha"
        )

    def test_missing_attributes_render_as_unknown_but_stay_greppable(self):
        class _Bare:
            pass

        fields = model_provider_fields(_Bare())
        assert fields == "provider=unknown base_url=unknown model=unknown"
        for key in ("provider=", "base_url=", "model="):
            assert key in fields

    def test_whitespace_padded_values_are_stripped(self):
        """Trailing/config whitespace must not corrupt the key=value shape."""
        agent = SimpleNamespace(
            provider="custom ",
            base_url="https://api.kilo.ai/api/gateway/v1/ ",
            model="stealth/ox-alpha",
        )
        assert model_provider_fields(agent) == (
            "provider=custom "
            "base_url=https://api.kilo.ai/api/gateway/v1/ "
            "model=stealth/ox-alpha"
        )

    def test_empty_strings_render_as_unknown(self):
        agent = SimpleNamespace(provider="", base_url="", model="")
        assert model_provider_fields(agent) == (
            "provider=unknown base_url=unknown model=unknown"
        )


def test_successful_api_call_log_carries_provider_base_url_and_model(caplog):
    """The success summary line must include the full routing context.

    Before the fix this line printed ``model=… provider=…`` (in that order,
    with no ``base_url``), so a session routed through several custom gateways
    produced indistinguishable lines like ``provider=custom``.
    """
    agent = _make_custom_gateway_agent()
    agent.client = MagicMock()
    usage = SimpleNamespace(
        prompt_tokens=120, completion_tokens=8, total_tokens=128,
    )
    agent.client.chat.completions.create.side_effect = [
        _response(content="All done.", finish_reason="stop", usage=usage),
    ]

    with (
        caplog.at_level(logging.INFO, logger="agent.conversation_loop"),
        patch("run_agent.handle_function_call", return_value="ok"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        agent.run_conversation("say done")

    summaries = [r for r in caplog.records if "API call #" in r.getMessage()]
    assert summaries, "expected the per-call success summary log line"
    matched = [_FRAGMENT_RE.search(r.getMessage()) for r in summaries]
    # Contract: EVERY summary line carries the complete fragment — a partial
    # fragment must fail loudly here rather than be filtered out silently.
    assert matched and all(matched), (
        "some API call summary line lacks the complete canonical "
        "provider=/base_url=/model= fragment; got: "
        f"{[r.getMessage() for r in summaries]}"
    )
    provider, base_url, model = matched[-1].groups()
    assert provider == "custom"
    assert base_url == "https://api.kilo.ai/api/gateway/v1/"
    assert model == "stealth/ox-alpha"


def test_empty_response_retry_lines_identify_the_full_route(caplog):
    """Empty-retry warnings must name provider AND base_url, not just model.

    Real-world symptom: a gateway returning intermittent empty completions
    logged ``Empty response (no content or reasoning) — retry N/M
    (model=…)`` with zero routing context, while sibling lines in the same
    incident said ``provider=custom``, making it impossible to tell WHICH
    custom gateway was failing.
    """
    agent = _make_custom_gateway_agent()
    agent.client = MagicMock()
    empty = _response(content="", finish_reason="stop")
    # One empty per retry-backoff plus the final one that trips the
    # exhausted-without-fallback branch.
    agent.client.chat.completions.create.side_effect = [empty] * 4

    with (
        caplog.at_level(logging.WARNING, logger="agent.conversation_loop"),
        patch("agent.conversation_loop.jittered_backoff", return_value=0.0),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        agent.run_conversation("do the task")

    retry_records = [
        r for r in caplog.records
        if "Empty response (no content or reasoning)" in r.getMessage()
    ]
    assert retry_records, "expected empty-response retry warning lines"

    exhausted = [r for r in caplog.records if "No fallback available." in r.getMessage()]
    assert exhausted, "expected the retries-exhausted warning line"

    for record in retry_records + exhausted:
        match = _FRAGMENT_RE.search(record.getMessage())
        assert match, (
            f"log line lacks the provider/base_url/model fragment: "
            f"{record.getMessage()}"
        )
        provider, base_url, model = match.groups()
        assert provider == "custom"
        assert base_url == "https://api.kilo.ai/api/gateway/v1/"
        assert model == "stealth/ox-alpha"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"finish_reason": "stop"},
        {
            # Narration-only: finish_reason signals tool calls but the
            # array arrived empty.
            "finish_reason": "tool_calls",
            "tool_calls": [],
        },
    ],
    ids=["plain-empty", "narration-only-tool-turn"],
)
def test_dropped_or_missing_content_lines_never_lose_routing_context(
    caplog, kwargs,
):
    """Both empty-flavors must keep the full route visible across retries.

    Guards the narration-only variant (``finish_reason=tool_calls`` with an
    empty tool_calls array) alongside the plain empty response — both re-prompt
    paths previously formatted model/provider by hand.
    """
    agent = _make_custom_gateway_agent()
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = [
        _response(content="", **kwargs),
        _response(content="Recovered.", finish_reason="stop"),
    ]

    with (
        caplog.at_level(logging.WARNING, logger="agent.conversation_loop"),
        patch("run_agent.handle_function_call", return_value="ok"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        agent.run_conversation("do the task")

    warnings_with_route = [
        r for r in caplog.records
        if _FRAGMENT_RE.search(r.getMessage())
    ]
    assert warnings_with_route, (
        "expected at least one warning carrying the routing fragment; got: "
        f"{[r.getMessage() for r in caplog.records]}"
    )
