"""Forks that run provider-less inherit the memory provider under ``compression.checkpoint_required``.

A fork built with ``skip_memory=True`` arms the gate from the same config as its parent and
would raise ``CompressionCheckpointUnavailable`` at its first compaction. Under the armed gate
the fork sites load the provider, scoped as ``agent_context="subagent"``; gate off they stay
provider-less. The detached review and ``/btw`` forks persist nothing and are exempt upstream
(``_pre_compress_memory_context``), so they are not fork sites here.
"""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.memory_provider import PRE_COMPRESS_CHECKPOINT_API_VERSION, MemoryProvider


def _fork_config(checkpoint_required):
    """Minimal config shape the spawn sites read through."""
    return {"compression": {"checkpoint_required": checkpoint_required}, "agent": {}, "memory": {}, "delegation": {}}


class _ConstructorStop(BaseException):
    """Abort at the child constructor. BaseException, because the sites catch ``except Exception``
    and would swallow an ordinary failure into a green test."""


def _fork_site(site):
    """``(module holding the AIAgent binding, spawn thunk, extra patches)``, resolved lazily.
    batch_runner binds AIAgent at import, so its patch lands on its own global."""
    if site == "delegate":
        import run_agent
        from tools import delegate_tool

        parent = SimpleNamespace(
            model="m", provider="openrouter", api_key="k", base_url="https://openrouter.ai/api/v1",
            api_mode="chat_completions", session_id="parent", enabled_toolsets=None, disabled_toolsets=None,
            request_overrides={}, prefill_messages=None, _delegate_depth=0,
        )
        return run_agent, (lambda: delegate_tool._build_child_agent(
            task_index=0, goal="g", context=None, toolsets=None, model=None, max_iterations=3, task_count=1,
            parent_agent=parent,
        )), ()
    if site == "curator":
        from agent import curator
        import run_agent

        return run_agent, (lambda: curator._run_llm_review("prompt")), ()
    if site == "batch_runner":
        import batch_runner

        return batch_runner, (lambda: batch_runner._process_single_prompt(
            0, {"prompt": "p"}, 0, {"model": "m", "max_iterations": 3, "distribution": {}},
        )), (patch.object(batch_runner, "sample_toolsets_from_distribution", return_value=["terminal"]),)
    if site == "feishu_comment":
        from plugins.platforms.feishu import feishu_comment
        import run_agent

        return run_agent, (lambda: feishu_comment._run_comment_agent("p", client=object(), session_key="")), (
            patch.object(feishu_comment, "_resolve_model_and_runtime", return_value=("m", {})),
            patch.object(feishu_comment, "_load_session_history", return_value=[]),
        )
    raise AssertionError(f"unknown fork site: {site}")


def _capture_fork_kwargs(site, *, checkpoint_required):
    """Drive one real fork site and return the kwargs it passed to AIAgent."""
    binding, spawn, extra_patches = _fork_site(site)
    captured = {}

    def _recorder(**kwargs):
        captured.update(kwargs)
        raise _ConstructorStop()

    cfg = _fork_config(checkpoint_required)
    with ExitStack() as stack:
        stack.enter_context(patch("hermes_cli.config.load_config", return_value=cfg))
        stack.enter_context(patch("hermes_cli.config.load_config_readonly", return_value=cfg))
        stack.enter_context(patch.object(binding, "AIAgent", _recorder))
        for extra in extra_patches:
            stack.enter_context(extra)
        with pytest.raises(_ConstructorStop):
            spawn()
    assert captured, f"{site} never reached the child constructor"
    return captured


_FORK_SITES = ("delegate", "curator", "batch_runner", "feishu_comment")


@pytest.mark.parametrize("site", _FORK_SITES)
def test_fork_inherits_memory_provider_when_checkpoint_required(site):
    captured = _capture_fork_kwargs(site, checkpoint_required=True)

    assert captured["skip_memory"] is False
    # The value agent/memory_provider.py names; a fresh per-site string would fall out of the
    # providers' non-primary write gates and the host's fork scope.
    assert captured["memory_agent_context"] == "subagent"


@pytest.mark.parametrize("site", _FORK_SITES)
def test_fork_stays_provider_less_when_gate_is_off(site):
    """The control against "always inherit"."""
    assert _capture_fork_kwargs(site, checkpoint_required=False)["skip_memory"] is True


def test_checkpoint_required_from_config_reads_the_armed_key():
    from agent.conversation_compression_checkpoint import checkpoint_required_from_config

    def _with(cfg):
        with patch("hermes_cli.config.load_config_readonly", return_value=cfg):
            return checkpoint_required_from_config()

    assert _with({"compression": {"checkpoint_required": True}}) is True
    assert _with({"compression": {"checkpoint_required": "yes"}}) is True
    assert _with({"compression": {"checkpoint_required": False}}) is False
    assert _with({"compression": {}}) is False
    assert _with({}) is False
    # A section of the wrong shape must not raise out of a spawn path.
    assert _with({"compression": "nonsense"}) is False
    assert _with(None) is False
    # An unreadable config reads as "not armed", matching init_agent.
    with patch("hermes_cli.config.load_config_readonly", side_effect=RuntimeError("config unreadable")):
        assert checkpoint_required_from_config() is False


class _RecordingProvider(MemoryProvider):
    """Records the ``initialize()`` kwargs (one shared instance per test via ``seen``)."""

    pre_compress_checkpoint_api_version = PRE_COMPRESS_CHECKPOINT_API_VERSION

    def __init__(self, seen):
        self.seen = seen

    @property
    def name(self):
        return "recording"

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self.seen.clear()
        self.seen.update(kwargs)

    def get_tool_schemas(self):
        return []


def _build_agent_with_recording_provider(seen, **extra):
    """A real AIAgent init with a recording provider; returns the agent."""
    cfg = {"compression": {}, "agent": {}, "memory": {"provider": "recording", "memory_enabled": False,
                                                      "user_profile_enabled": False}}
    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("plugins.memory.load_memory_provider", return_value=_RecordingProvider(seen)),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        from run_agent import AIAgent

        return AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                       skip_context_files=True, skip_memory=False, **extra)


def test_memory_agent_context_reaches_the_provider_initialize():
    """The label survives the whole constructor path to ``initialize()``; unset, the platform
    rule still decides."""
    seen = {}
    _build_agent_with_recording_provider(seen, memory_agent_context="subagent", platform="curator")
    assert seen["agent_context"] == "subagent"
    assert seen["platform"] == "curator"

    _build_agent_with_recording_provider(seen, platform="curator")
    assert seen["agent_context"] == "primary"
    _build_agent_with_recording_provider(seen, platform="subagent")
    assert seen["agent_context"] == "subagent"
