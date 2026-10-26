"""Streaming is a provider capability, not an authentication mechanism."""
import threading
from types import SimpleNamespace

import pytest

from agent.turn_api_call import _should_stream
from providers.base import ProviderProfile


def test_streaming_capability_preserves_positional_plugin_construction():
    from dataclasses import dataclass, fields
    from typing import Any

    @dataclass
    class PluginProfile(ProviderProfile):
        plugin_option: str = "default"

    # All pre-existing base and subclass fields retain their positional slots.
    legacy_fields = [f for f in fields(PluginProfile) if f.name != "supports_streaming"]
    values: list[Any] = [object() for _ in legacy_fields]
    profile = PluginProfile(*values, supports_streaming=True)
    for field, value in zip(legacy_fields, values):
        assert getattr(profile, field.name) is value
    assert profile.supports_streaming is True


@pytest.mark.parametrize("auth,capability,url,disabled,consumers,expected", [
    ("external_process", True, "process://fixture", False, True, True),
    ("external_process", True, "process://fixture", False, False, True),
    ("external_process", None, "process://fixture", False, True, False),
    ("external_process", None, "https://proxy.invalid/v1", False, True, False),
    ("external_process", False, "process://fixture", False, True, False),
    ("external_process", True, "process://fixture", True, True, False),
    ("external_process", True, "acp://fixture", False, True, False),
    ("external_process", True, "acp+tcp://127.0.0.1:9000", False, True, False),
    ("api_key", None, "https://provider.invalid/v1", False, True, True),
    ("api_key", False, "https://provider.invalid/v1", False, True, False),
])
def test_external_process_can_opt_into_streaming(monkeypatch, auth, capability, url, disabled, consumers, expected):
    profile = ProviderProfile(name="streaming-process", auth_type=auth, supports_streaming=capability)
    monkeypatch.setattr("providers.get_provider_profile", lambda name: profile)
    agent = SimpleNamespace(
        provider=profile.name, base_url=url,
        _disable_streaming=disabled, _has_stream_consumers=lambda: consumers,
    )
    assert _should_stream(agent) is expected


def test_discovered_process_provider_delivers_progress_before_completion(tmp_path, monkeypatch):
    """A buffered call deadlocks this handshake; the real streaming consumer releases it.

    Load a process profile through the user-plugin registry in an isolated home,
    then cross the normal dispatch and streaming accumulator with a fixture client.
    Only native I/O is replaced, not routing, callbacks, accumulation or cleanup.
    """
    from hermes_constants import get_hermes_home
    from providers import get_provider_profile
    from run_agent import AIAgent
    from agent.turn_api_call import perform_api_call

    plugin = get_hermes_home() / "plugins" / "streaming-fixture"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        "name: streaming-fixture\nkind: model-provider\nversion: 0.1.0\ndescription: test fixture\n")
    (plugin / "__init__.py").write_text(
        "from providers import register_provider\n"
        "from providers.base import ProviderProfile\n"
        "register_provider(ProviderProfile(name='streaming-fixture', "
        "auth_type='external_process', base_url='process://fixture', supports_streaming=True))\n")
    profile = get_provider_profile("streaming-fixture")
    assert profile is not None
    progress_seen = threading.Event()
    closed = []
    requests = []
    carrier = [{"type": "streaming-fixture.native_assistant", "value": "fixture-signature"}]

    def chunk(delta=None, finish=None, usage=None):
        return SimpleNamespace(
            model="fixture-model", usage=usage,
            choices=[] if delta is None else [SimpleNamespace(
                index=0, delta=SimpleNamespace(**delta), finish_reason=finish)],
        )

    def chunks():
        yield chunk({"reasoning_content": "fixture reasoning"})
        assert progress_seen.wait(5), "Host buffered progress until completion"
        yield chunk({"tool_calls": [SimpleNamespace(
            index=0, id="fixture-call", type="function",
            function=SimpleNamespace(name="read_file", arguments='{"path":"fixture.txt"}'))],
            "reasoning_details": carrier}, finish="tool_calls")
        yield chunk(usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10))

    class FixtureClient:
        HERMES_SKIP_TRANSPORT_WRAP = True
        HERMES_SKIP_ASYNC_WRAP = True

        def __init__(self):
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            requests.append(kwargs)
            if kwargs.get("stream"):
                return chunks()
            # Mirror a process client buffering its native events until completion.
            list(chunks())
            raise AssertionError("Buffered path must not be selected")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(profile, "create_client", lambda **kwargs: FixtureClient())
    agent = AIAgent(
        provider=profile.name, model="fixture-model", api_key="fixture", base_url=profile.base_url,
        enabled_toolsets=[], quiet_mode=True, skip_context_files=True, skip_memory=True,
        save_trajectories=False,
    )
    agent.reasoning_callback = lambda text: progress_seen.set()
    kwargs = {"model": agent.model, "messages": [{"role": "user", "content": "fixture"}]}
    try:
        verdict = perform_api_call(
            agent, api_kwargs=kwargs, _original_api_kwargs=kwargs, _llm_middleware_trace=[],
            _moa_prepared_request=None, _retry=SimpleNamespace(), thinking_spinner=None,
            retry_count=0, api_call_count=0, api_request_id="fixture-request",
            effective_task_id=None, turn_id="fixture-turn", interrupted=False,
        )
        assert verdict.action == "fallthrough"
        response = verdict.response
        assert requests[0]["stream"] is True
        assert progress_seen.is_set()
        assert response.choices[0].message.tool_calls[0].function.arguments == '{"path":"fixture.txt"}'
        assert response.choices[0].message.reasoning_content == "fixture reasoning"
        assert response.choices[0].message.reasoning_details == carrier
        assert response.usage.completion_tokens == 3
        assert response.choices[0].finish_reason == "tool_calls"
    finally:
        agent.close()
    assert closed
