import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "plugins" / "image_gen" / "openai-codex" / "__init__.py"


def load_plugin_module():
    spec = importlib.util.spec_from_file_location("openai_codex_image_gen_plugin", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeStream:
    def __init__(self, final):
        self._final = final

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        return iter(())

    def get_final_response(self):
        return self._final


class _FinalResponse:
    output = []


class _FakeResponses:
    def __init__(self):
        self.kwargs = None

    def stream(self, **kwargs):
        self.kwargs = kwargs
        return _FakeStream(_FinalResponse())


class _FakeClient:
    def __init__(self):
        self.responses = _FakeResponses()


def test_codex_image_generation_uses_latest_responses_host_and_full_image_only():
    module = load_plugin_module()
    client = _FakeClient()

    module._collect_image_b64(client, prompt="draw precise French text", size="2160x3840", quality="high")

    kwargs = client.responses.kwargs
    assert kwargs["model"] == "gpt-5.5"
    tool = kwargs["tools"][0]
    assert tool["type"] == "image_generation"
    assert tool["model"] == "gpt-image-2"
    assert tool["quality"] == "high"
    assert tool["size"] == "2160x3840"
    assert tool["action"] == "generate"
    assert tool["partial_images"] == 0


def test_codex_image_generation_keeps_default_aspect_sizes_upstream_compatible():
    module = load_plugin_module()

    assert module._SIZES["portrait"] == "1024x1536"
    assert module._SIZES["landscape"] == "1536x1024"
    assert module._SIZES["square"] == "1024x1024"
