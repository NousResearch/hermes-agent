import base64
import importlib
from types import SimpleNamespace


mod = importlib.import_module("plugins.image_gen.openai-codex")


class _FakeStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        item = SimpleNamespace(type="image_generation_call", result="ZmFrZS1wbmc=")
        yield SimpleNamespace(type="response.output_item.done", item=item)

    def get_final_response(self):
        return SimpleNamespace(output=[])


class _FakeResponses:
    def __init__(self):
        self.last_stream_kwargs = None

    def stream(self, **kwargs):
        self.last_stream_kwargs = kwargs
        return _FakeStream(**kwargs)


class _FakeClient:
    def __init__(self):
        self.responses = _FakeResponses()


def test_build_input_content_encodes_local_reference_image(tmp_path):
    ref = tmp_path / "source.png"
    ref.write_bytes(b"fake image bytes")

    content = mod._build_input_content("pixelate this", [str(ref)])

    assert content[0] == {"type": "input_text", "text": "pixelate this"}
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/png;base64,")
    encoded = content[1]["image_url"].split(",", 1)[1]
    assert base64.b64decode(encoded) == b"fake image bytes"


def test_build_input_content_preserves_http_and_data_urls():
    data_url = "data:image/png;base64,abcd"
    http_url = "https://example.com/source.png"

    content = mod._build_input_content("use refs", [data_url, http_url])

    assert content[1] == {"type": "input_image", "image_url": data_url}
    assert content[2] == {"type": "input_image", "image_url": http_url}


def test_collect_image_b64_sends_reference_images_to_responses_input(tmp_path):
    ref = tmp_path / "source.png"
    ref.write_bytes(b"fake image bytes")
    client = _FakeClient()

    result = mod._collect_image_b64(
        client,
        prompt="convert exactly",
        size="1024x1024",
        quality="low",
        reference_images=[str(ref)],
    )

    assert result == "ZmFrZS1wbmc="
    message = client.responses.last_stream_kwargs["input"][0]
    assert message["content"][0]["type"] == "input_text"
    assert message["content"][1]["type"] == "input_image"
    assert message["content"][1]["image_url"].startswith("data:image/png;base64,")


def test_provider_generate_forwards_reference_images(monkeypatch, tmp_path):
    captured = {}
    saved = tmp_path / "out.png"

    monkeypatch.setattr(mod, "_read_codex_access_token", lambda: "token")
    monkeypatch.setattr(mod, "_build_codex_client", lambda: object())
    monkeypatch.setattr(mod, "save_b64_image", lambda *args, **kwargs: saved)

    def fake_collect(client, *, prompt, size, quality, reference_images=None):
        captured["reference_images"] = reference_images
        return "ZmFrZS1wbmc="

    monkeypatch.setattr(mod, "_collect_image_b64", fake_collect)

    result = mod.OpenAICodexImageGenProvider().generate(
        "convert exactly",
        aspect_ratio="square",
        reference_images=["/tmp/ref.png"],
    )

    assert result["success"] is True
    assert result["reference_image_count"] == 1
    assert captured["reference_images"] == ["/tmp/ref.png"]
