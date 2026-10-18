"""A final fallback error must not weaken conversation preservation (#72637)."""

import json
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI

from agent import auxiliary_client as aux
from agent.context_compressor import ContextCompressor
from agent.conversation_compression_diagnostics import _emit_compression_auth_hint


@pytest.mark.parametrize("primary_status,fallback_status", [(401, 429), (429, 401)])
def test_exhausted_fallback_preserves_transcript_and_reports_final_route(
    tmp_path, monkeypatch, primary_status, fallback_status,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "primary-key")
    (tmp_path / "config.yaml").write_text(
        "auxiliary:\n  compression:\n    provider: openrouter\n    model: main-B\n"
        "    fallback_chain:\n      - provider: custom\n        model: main-B\n"
        "        base_url: https://fallback.example/v1\n        api_key: fallback-key\n"
    )
    wire = []

    def handle(request):
        wire.append((request.url.host, json.loads(request.content)["model"]))
        status = primary_status if request.url.host == "openrouter.ai" else fallback_status
        return httpx.Response(status, request=request, json={
            "error": {"message": "invalid credential" if status == 401 else "rate limit exceeded"},
        })

    class TransportClient(OpenAI):
        def __init__(self, **kwargs):
            kwargs["http_client"] = httpx.Client(transport=httpx.MockTransport(handle))
            kwargs["max_retries"] = 0
            super().__init__(**kwargs)

    monkeypatch.setattr(aux, "OpenAI", TransportClient)
    aux._client_cache.clear()
    comp = ContextCompressor(
        model="main-B", provider="custom", base_url="https://main.example/v1", api_key="main-key",
        quiet_mode=True, protect_first_n=2, protect_last_n=2, abort_on_summary_failure=False,
    )
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}: " + "detail " * 200}
                for i in range(20)]
    try:
        compressed = comp.compress(messages, current_tokens=120_000, force=True)
        assert list(dict.fromkeys(wire)) == [("openrouter.ai", "main-B"), ("fallback.example", "main-B")]
        assert compressed == messages
        assert comp._last_compress_aborted
        assert comp.compression_count == 0
        assert comp._last_attempt_failure_class == ("auth" if fallback_status == 401 else "other")
        warnings = []
        _emit_compression_auth_hint(SimpleNamespace(context_compressor=comp, _emit_warning=warnings.append))
        rendered = "\n".join(warnings)
        assert "fallback.example" in rendered and "main-B" in rendered
        assert "openrouter.ai" not in rendered
        if fallback_status == 429:
            assert "auth/permission" not in rendered
    finally:
        aux._client_cache.clear()
