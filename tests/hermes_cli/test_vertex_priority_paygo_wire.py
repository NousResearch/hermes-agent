"""End-to-end proof that the Priority PayGo header reaches the outgoing HTTP request.

The hook test proves the profile returns the header; this proves the request the provider
actually sends carries it. It runs the real transport (build_kwargs -> create()) against a
recording stand-in client, so a header that is dropped between the profile hook and the
wire would fail here.
"""

from __future__ import annotations

import pytest


class _RecordingCompletions:
    def __init__(self, sink):
        self._sink = sink

    def create(self, **kwargs):
        self._sink.append(kwargs)
        raise RuntimeError("request captured; no network in this test")


class _RecordingClient:
    def __init__(self, sink):
        self.chat = type("_Chat", (), {"completions": _RecordingCompletions(sink)})()


@pytest.fixture
def vertex_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _write_config(home, priority_paygo: bool) -> None:
    (home / "config.yaml").write_text(
        "vertex:\n"
        "  project_id: proj-abc\n"
        "  region: global\n"
        f"  priority_paygo: {str(priority_paygo).lower()}\n"
    )


def _capture_request(vertex_home, priority_paygo: bool) -> dict:
    """Build the request kwargs the Vertex provider would send, and record them."""
    from agent.transports.chat_completions import ChatCompletionsTransport
    from providers import get_provider_profile

    profile = get_provider_profile("vertex")
    assert profile is not None

    sink: list[dict] = []
    client = _RecordingClient(sink)
    transport = ChatCompletionsTransport()
    api_kwargs = transport.build_kwargs(
        provider_profile=profile,
        model="google/gemini-4-argon",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        base_url="https://aiplatform.googleapis.com/v1beta1/projects/proj-abc/locations/global/endpoints/openapi",
    )
    with pytest.raises(RuntimeError):
        client.chat.completions.create(**api_kwargs)
    assert sink, "the request was never issued"
    return sink[0]


def test_request_carries_the_priority_header_when_enabled(vertex_home):
    _write_config(vertex_home, True)

    request = _capture_request(vertex_home, True)

    headers = request.get("extra_headers") or {}
    assert headers.get("X-Vertex-AI-LLM-Shared-Request-Type") == "priority", (
        f"the Priority PayGo header did not reach the request: {sorted(request)}")


def test_request_carries_no_priority_header_by_default(vertex_home):
    _write_config(vertex_home, False)

    request = _capture_request(vertex_home, False)

    headers = request.get("extra_headers") or {}
    assert "X-Vertex-AI-LLM-Shared-Request-Type" not in headers, (
        f"the Priority PayGo header reached the request unasked: {headers}")
