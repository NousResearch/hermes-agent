import asyncio
import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import feed_images


class FakeResponse:
    def __init__(self, status_code=200, payload=None, content=b""):
        self.status_code = status_code
        self._payload = payload or {}
        self.content = content
        self.text = json.dumps(self._payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}: {self.text}")

    def json(self):
        return self._payload


@pytest.mark.asyncio
async def test_post_discord_feed_image_sends_multipart_with_jwt_and_metadata(monkeypatch, tmp_path):
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"png-bytes")
    calls = []

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def post(self, url, **kwargs):
            calls.append((url, kwargs))
            return FakeResponse(payload={"id": 123, "status": "queued"})

    monkeypatch.setattr(feed_images.httpx, "AsyncClient", FakeAsyncClient)

    result = await feed_images.post_discord_feed_image(
        image_path=str(image_path),
        message_text="make it cute",
        source={
            "platform": "discord",
            "chat_id": "chan-1",
            "message_id": "msg-1",
            "user_id": "user-1",
        },
        api_base_url="https://dev-api.fanhearts.com",
        jwt="jwt-token",
    )

    assert result == {"id": 123, "status": "queued"}
    assert len(calls) == 1
    url, kwargs = calls[0]
    assert url == "https://dev-api.fanhearts.com/feed_images"
    assert kwargs["headers"]["Authorization"] == "Bearer jwt-token"
    assert "image" in kwargs["files"]
    assert json.loads(kwargs["data"]["metadata"])["discord_message_id"] == "msg-1"
    assert kwargs["data"]["prompt"] == "make it cute"


def test_build_transform_prompt_uses_required_synthesis_prompt(tmp_path):
    prompt = feed_images.build_transform_prompt(tmp_path / "source.png", "ignored user text")

    assert "첨부한 사진을 기반으로 자연스러운 합성 이미지를 만들어줘." in prompt
    assert "사진 속 인물의 얼굴 생김새, 표정, 시선, 포즈, 체형, 신체 비율은 최대한 그대로 유지해줘." in prompt
    assert "사진을 들고 있는 손이나 원래의 종이 사진 느낌은 제거하고" in prompt
    assert "네거티브 프롬프트" in prompt
    assert "다른 사람처럼 변형된 얼굴" in prompt
    assert "ignored user text" not in prompt


def test_generate_with_codex_oauth_image_uses_input_image_and_writes_result(monkeypatch, tmp_path):
    input_image = tmp_path / "source.jpg"
    input_image.write_bytes(b"source-image-bytes")
    output_b64 = base64.b64encode(b"generated-image-bytes").decode("ascii")
    captured = {}

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def __iter__(self):
            item = SimpleNamespace(type="image_generation_call", result=output_b64, status="completed", revised_prompt="revised")
            yield SimpleNamespace(type="response.output_item.done", item=item)

        def get_final_response(self):
            return SimpleNamespace(id="resp_1", usage=SimpleNamespace(total_tokens=1), output=[])

    class FakeResponses:
        def stream(self, **kwargs):
            captured["payload"] = kwargs
            return FakeStream()

    class FakeClient:
        responses = FakeResponses()

    monkeypatch.setattr(feed_images, "_build_codex_oauth_client", lambda: FakeClient())

    output_path, output_url = feed_images.generate_with_codex_oauth_image(input_image, "generated prompt", tmp_path / "job")

    assert output_url is None
    assert output_path == tmp_path / "job" / "completed_image.png"
    assert output_path.read_bytes() == b"generated-image-bytes"
    payload = captured["payload"]
    assert payload["model"] == "gpt-5.4"
    assert payload["tools"] == [{"type": "image_generation", "model": "gpt-image-2", "size": "1024x1024", "quality": "medium", "output_format": "png", "background": "auto"}]
    content = payload["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "generated prompt"}
    assert content[1]["type"] == "input_image"
    assert set(content[1]) == {"type", "image_url"}
    assert content[1]["image_url"].startswith("data:image/jpeg;base64,")
    assert base64.b64decode(content[1]["image_url"].split(",", 1)[1]) == b"source-image-bytes"
    metadata = json.loads((tmp_path / "job" / "completed_image.png.json").read_text())
    assert metadata["model"] == "gpt-5.4"
    assert metadata["image_model"] == "gpt-image-2"
    assert metadata["revised_prompt"] == "revised"


@pytest.mark.asyncio
async def test_process_queued_feed_images_claims_generates_and_updates(monkeypatch, tmp_path):
    input_image = tmp_path / "source.png"
    output_image = tmp_path / "generated.png"
    input_image.write_bytes(b"source")
    output_image.write_bytes(b"generated")
    calls = []

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def get(self, url, **kwargs):
            calls.append(("GET", url, kwargs))
            return FakeResponse(payload={"feed_images": [{"id": "img-1", "image_url": "https://cdn/input.png", "prompt": "turn into fan art"}]})

        async def post(self, url, **kwargs):
            calls.append(("POST", url, kwargs))
            if url.endswith("/claim"):
                return FakeResponse(payload={"ok": True})
            raise AssertionError(url)

        async def put(self, url, **kwargs):
            calls.append(("PUT", url, kwargs))
            return FakeResponse(payload={"ok": True})

    async def fake_download(client, url, workdir):
        assert url == "https://cdn/input.png"
        return input_image

    def fake_generate_prompt(image_path, user_prompt):
        assert image_path == input_image
        return "generated prompt"

    def fake_generate_image(image_path, prompt, output_dir):
        assert image_path == input_image
        assert prompt == "generated prompt"
        return output_image, None

    monkeypatch.setattr(feed_images.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(feed_images, "_download_image", fake_download)
    monkeypatch.setattr(feed_images, "build_transform_prompt", fake_generate_prompt)
    monkeypatch.setattr(feed_images, "generate_with_codex_oauth_image", fake_generate_image)

    summary = await feed_images.process_queued_feed_images(
        api_base_url="https://dev-api.fanhearts.com",
        jwt="jwt-token",
        limit=1,
        workdir=tmp_path,
    )

    assert summary["completed"] == 1
    assert summary["failed"] == 0
    put_calls = [call for call in calls if call[0] == "PUT"]
    assert len(put_calls) == 1
    _, url, kwargs = put_calls[0]
    assert url == "https://dev-api.fanhearts.com/feed_images/img-1"
    assert kwargs["headers"]["Authorization"] == "Bearer jwt-token"
    assert kwargs["data"]["status"] == "completed"
    assert kwargs["data"]["transform_prompt"] == "generated prompt"
    assert kwargs["data"]["model"] == "codex-oauth/gpt-5.4-image_generation"
    assert kwargs["data"]["output_image_url"] == ""
    assert "completed_image" in kwargs["files"]
