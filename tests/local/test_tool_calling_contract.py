from __future__ import annotations

from unittest.mock import patch

from agent.local_secretary.llama_contract import looks_like_plaintext_tool_call, run_llama_contract_checks


def test_tool_calling_contract_requires_structured_tool_calls():
    models_payload = {"data": [{"id": "qwen35-9b-secretary"}]}
    props_payload = {"default_generation_settings": {"n_ctx": 65536}}

    def fake_get(url: str, timeout: float = 15.0):
        if url.endswith("/v1/models"):
            return 200, models_payload
        if url.endswith("/props"):
            return 200, props_payload
        raise AssertionError(url)

    plaintext = {"choices": [{"message": {"content": "tool_call get_weather(city='Tokyo')"}}]}
    structured = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"Tokyo"}'},
                        }
                    ]
                }
            }
        ]
    }

    with patch("agent.local_secretary.llama_contract._get_json", side_effect=fake_get), patch(
        "agent.local_secretary.llama_contract._post_json",
        side_effect=[
            (200, {"choices": [{"message": {"content": "pong"}}]}),
            (200, plaintext),
        ],
    ):
        bad = run_llama_contract_checks("http://127.0.0.1:8080")

    assert bad["ok"] is False
    assert "jinja" in bad["checks"]["tool_calling"]["error"].lower()
    assert looks_like_plaintext_tool_call("tool_call get_weather") is True

    with patch("agent.local_secretary.llama_contract._get_json", side_effect=fake_get), patch(
        "agent.local_secretary.llama_contract._post_json",
        side_effect=[
            (200, {"choices": [{"message": {"content": "pong"}}]}),
            (200, structured),
        ],
    ):
        good = run_llama_contract_checks("http://127.0.0.1:8080")

    assert good["checks"]["tool_calling"]["ok"] is True
