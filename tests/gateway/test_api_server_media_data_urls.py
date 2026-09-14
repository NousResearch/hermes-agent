"""Provider-backed MEDIA rendering for the API server."""

import base64
import asyncio
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("aiohttp")

from aiohttp import web  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from gateway.config import PlatformConfig  # noqa: E402
from gateway.outbound_files import (  # noqa: E402
    Base64OutboundFileProvider,
    OmittedOutboundFileProvider,
    OutboundFileExporter,
    OutboundFilesConfigError,
    create_outbound_file_provider,
    OutboundFilesConfig,
)
from gateway.platforms.api_server import APIServerAdapter  # noqa: E402


def _openai_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    return app


def _agent_result(text: str) -> tuple[dict, dict]:
    return (
        {"final_response": text, "messages": [], "api_calls": 1},
        {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    )


_PROVIDER_OUTPUTS = [
    pytest.param("base64", "data:image/png;base64,", id="base64"),
    pytest.param("none", "[IMAGE OMITTED]", id="none"),
]


def _adapter_for_provider(provider: str) -> APIServerAdapter:
    return APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"outbound_files": {"provider": provider}},
        )
    )


@pytest.mark.asyncio
async def test_media_tag_is_inlined_as_a_data_url(tmp_path):
    path = tmp_path / "shot.png"
    contents = b"png"
    path.write_bytes(contents)
    adapter = object.__new__(APIServerAdapter)
    adapter._outbound_files = OutboundFileExporter.from_dict(None)

    rendered = await adapter._render_outbound_text(f"Here: MEDIA:{path}")

    encoded = base64.b64encode(contents).decode("ascii")
    assert rendered == f"Here: ![image](data:image/png;base64,{encoded})"


@pytest.mark.asyncio
async def test_non_image_is_omitted_and_missing_image_is_left_untouched(tmp_path):
    document = tmp_path / "report.pdf"
    document.write_bytes(b"pdf")
    missing = tmp_path / "missing.png"
    adapter = object.__new__(APIServerAdapter)
    adapter._outbound_files = OutboundFileExporter.from_dict(None)

    document_text = f"MEDIA:{document}"
    missing_text = f"MEDIA:{missing}"
    assert await adapter._render_outbound_text(document_text) == "[FILE OMITTED]"
    assert await adapter._render_outbound_text(missing_text) == missing_text


@pytest.mark.asyncio
async def test_base64_provider_enforces_configured_size_limit(tmp_path):
    image = tmp_path / "large.png"
    document = tmp_path / "large.pdf"
    image.write_bytes(b"12345")
    document.write_bytes(b"12345")
    provider = Base64OutboundFileProvider.from_options(
        {"max_image_size_bytes": 4}, provider_name="base64"
    )

    assert await provider.render(image) is None
    assert await provider.render(document) == "[FILE OMITTED]"


def test_base64_provider_preserves_legacy_default_limit():
    provider = Base64OutboundFileProvider.from_options({}, provider_name="base64")

    assert provider.max_image_size_bytes == 5 * 1024 * 1024


def test_unknown_provider_is_rejected():
    config = OutboundFilesConfig.from_dict({"provider": "unknown"})

    with pytest.raises(OutboundFilesConfigError, match="unsupported.*unknown"):
        create_outbound_file_provider(config)


@pytest.mark.parametrize("provider", [None, "none", "NONE"])
def test_none_provider_selects_omitted_file_renderer(provider):
    config = OutboundFilesConfig.from_dict({"provider": provider})

    assert isinstance(create_outbound_file_provider(config), OmittedOutboundFileProvider)


@pytest.mark.asyncio
async def test_none_provider_omits_images_and_other_files_without_exposing_paths():
    exporter = OutboundFileExporter.from_dict({"provider": None})

    assert await exporter.export_media_path("/missing/private.png") == "[IMAGE OMITTED]"
    assert await exporter.export_media_path("/missing/private.pdf") == "[FILE OMITTED]"


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "1024"])
def test_base64_provider_rejects_invalid_size_limit(value):
    with pytest.raises(OutboundFilesConfigError, match="positive integer"):
        Base64OutboundFileProvider.from_options(
            {"max_image_size_bytes": value}, provider_name="base64"
        )


@pytest.mark.parametrize("provider", ["base64", "none"])
def test_provider_rejects_options_not_declared_by_its_dataclass(provider):
    config = OutboundFilesConfig.from_dict(
        {"provider": provider, "provider_options": {"unknown": True}}
    )

    with pytest.raises(OutboundFilesConfigError, match=f"{provider}.*unknown"):
        create_outbound_file_provider(config)


def test_provider_options_must_be_nested():
    with pytest.raises(OutboundFilesConfigError, match="unsupported.*max_image"):
        OutboundFilesConfig.from_dict(
            {"provider": "base64", "max_image_size_bytes": 1024}
        )


def test_outbound_system_prompt_matches_provider():
    base64_exporter = OutboundFileExporter.from_dict(
        {
            "provider": "base64",
            "provider_options": {
                "max_image_size_bytes": 1024,
            },
        }
    )
    none_exporter = OutboundFileExporter.from_dict({"provider": None})

    assert "Images up to 1024 bytes" in base64_exporter.system_prompt_hint()
    assert "replaced with [FILE OMITTED]" in base64_exporter.system_prompt_hint()
    assert "delivery is disabled" in none_exporter.system_prompt_hint()


def test_adapter_appends_provider_hint_to_request_system_prompt():
    adapter = object.__new__(APIServerAdapter)
    adapter._outbound_files = OutboundFileExporter.from_dict({"provider": "none"})

    prompt = adapter._with_outbound_files_prompt("Client instruction")

    assert prompt.startswith("Client instruction\n\n")
    assert "delivery is disabled" in prompt


@pytest.mark.asyncio
async def test_streaming_processor_holds_only_media_candidate(tmp_path):
    path = tmp_path / "shot.png"
    path.write_bytes(b"png")
    adapter = object.__new__(APIServerAdapter)
    adapter._outbound_files = OutboundFileExporter.from_dict(None)
    processor = adapter._new_media_response_processor()

    assert await processor.feed("Ready\nMED") == "Ready\n"
    assert await processor.feed(f"IA:{str(path)[:-4]}") == ""
    rendered = await processor.feed(".png\nDone")

    assert rendered.startswith("![image](data:image/png;base64,")
    assert rendered.endswith("\nDone")


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider", "expected"), _PROVIDER_OUTPUTS)
async def test_chat_completions_non_streaming_renders_media(
    tmp_path, provider, expected
):
    path = tmp_path / "completion.png"
    path.write_bytes(b"png")
    raw = f"Ready\nMEDIA:{path}\nDone"
    adapter = _adapter_for_provider(provider)

    with patch.object(
        adapter, "_run_agent", new=AsyncMock(return_value=_agent_result(raw))
    ):
        async with TestClient(TestServer(_openai_app(adapter))) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "hermes-agent",
                    "messages": [{"role": "user", "content": "create an image"}],
                },
            )
            payload = await response.json()

    content = payload["choices"][0]["message"]["content"]
    assert response.status == 200
    assert str(path) not in content
    assert expected in content


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider", "expected"), _PROVIDER_OUTPUTS)
async def test_chat_completions_streaming_renders_split_media(
    tmp_path, provider, expected
):
    path = tmp_path / "completion-stream.png"
    path.write_bytes(b"png")
    raw = f"Ready\nMEDIA:{path}\nDone"
    adapter = _adapter_for_provider(provider)

    async def run_agent(**kwargs):
        callback = kwargs["stream_delta_callback"]
        callback("Ready\nMED")
        callback(f"IA:{str(path)[:-4]}")
        callback(".png\nDone")
        return _agent_result(raw)

    with patch.object(adapter, "_run_agent", side_effect=run_agent):
        async with TestClient(TestServer(_openai_app(adapter))) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "hermes-agent",
                    "messages": [{"role": "user", "content": "create an image"}],
                    "stream": True,
                },
            )
            wire_output = await response.text()

    assert response.status == 200
    assert str(path) not in wire_output
    assert expected in wire_output


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider", "expected"), _PROVIDER_OUTPUTS)
async def test_responses_non_streaming_renders_media(tmp_path, provider, expected):
    path = tmp_path / "response.png"
    path.write_bytes(b"png")
    raw = f"Ready\nMEDIA:{path}\nDone"
    adapter = _adapter_for_provider(provider)

    with patch.object(
        adapter, "_run_agent", new=AsyncMock(return_value=_agent_result(raw))
    ):
        async with TestClient(TestServer(_openai_app(adapter))) as client:
            response = await client.post(
                "/v1/responses",
                json={"model": "hermes-agent", "input": "create an image"},
            )
            payload = await response.json()

    content = payload["output"][-1]["content"][0]["text"]
    assert response.status == 200
    assert str(path) not in content
    assert expected in content


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider", "expected"), _PROVIDER_OUTPUTS)
async def test_responses_streaming_endpoint_renders_split_media(
    tmp_path, provider, expected
):
    path = tmp_path / "response-stream.png"
    path.write_bytes(b"png")
    raw = f"Ready\nMEDIA:{path}\nDone"
    adapter = _adapter_for_provider(provider)

    async def run_agent(**kwargs):
        callback = kwargs["stream_delta_callback"]
        callback("Ready\nMED")
        callback(f"IA:{str(path)[:-4]}")
        callback(".png\nDone")
        return _agent_result(raw)

    with patch.object(adapter, "_run_agent", side_effect=run_agent):
        async with TestClient(TestServer(_openai_app(adapter))) as client:
            response = await client.post(
                "/v1/responses",
                json={
                    "model": "hermes-agent",
                    "input": "create an image",
                    "stream": True,
                },
            )
            wire_output = await response.text()

    assert response.status == 200
    assert str(path) not in wire_output
    assert expected in wire_output


@pytest.mark.asyncio
async def test_responses_stream_replaces_split_media_before_writing(tmp_path):
    import gateway.platforms.api_server as api_server
    from gateway.config import PlatformConfig

    path = tmp_path / "chart.png"
    path.write_bytes(b"png")
    raw = f"Ready\nMEDIA:{path}\nDone"
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "test"}))
    written = []

    class FakeStreamResponse:
        async def prepare(self, _request):
            return None

        async def write(self, payload):
            written.append(payload)

    stream_q = api_server.ThreadSafeAsyncQueue()
    stream_q.put_nowait("Ready\nMED")
    stream_q.put_nowait(f"IA:{str(path)[:-4]}")
    stream_q.put_nowait(".png\nDone")
    stream_q.put_nowait(None)

    async def agent_result():
        return (
            {"final_response": raw, "messages": []},
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )

    request = MagicMock()
    request.headers = {}
    with patch.object(
        api_server.web,
        "StreamResponse",
        return_value=FakeStreamResponse(),
    ):
        await adapter._write_sse_responses(
            request=request,
            response_id=f"resp_{uuid.uuid4().hex[:28]}",
            model="hermes-agent",
            created_at=int(time.time()),
            stream_q=stream_q,
            agent_task=asyncio.create_task(agent_result()),
            agent_ref=[None],
            conversation_history=[],
            user_message="create a chart",
            instructions=None,
            conversation=None,
            store=False,
            session_id="session-1",
        )

    wire_output = b"".join(written).decode("utf-8")
    assert str(path) not in wire_output
    assert "data:image/png;base64," in wire_output


@pytest.mark.asyncio
async def test_runs_stream_replaces_split_media_before_writing(tmp_path):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    from gateway.config import PlatformConfig

    path = tmp_path / "chart.png"
    path.write_bytes(b"png")
    raw = f"Ready\nMEDIA:{path}"
    adapter = APIServerAdapter(PlatformConfig(enabled=True))

    app = web.Application()
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)

    def create_agent(**kwargs):
        agent = MagicMock()

        def run_conversation(**_run_kwargs):
            kwargs["stream_delta_callback"]("Ready\nMED")
            kwargs["stream_delta_callback"](f"IA:{path}")
            return {"final_response": raw}

        agent.run_conversation.side_effect = run_conversation
        agent.session_prompt_tokens = 1
        agent.session_completion_tokens = 1
        agent.session_total_tokens = 2
        return agent

    with patch.object(adapter, "_create_agent", side_effect=create_agent):
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/v1/runs", json={"input": "make chart"})
            run_id = (await response.json())["run_id"]
            for _ in range(40):
                status = await (await client.get(f"/v1/runs/{run_id}")).json()
                if status["status"] == "completed":
                    break
                await asyncio.sleep(0.05)
            events = await client.get(f"/v1/runs/{run_id}/events")
            wire_output = await events.text()

    assert str(path) not in wire_output
    assert str(path) not in status["output"]
    assert "data:image/png;base64," in wire_output
    assert "data:image/png;base64," in status["output"]
