"""An auxiliary-model Anthropic 429 benches (key, aux model) only, like the main loop does.

Anthropic rate limits are per model: a 429 on the cheap aux model says nothing about the same
key's standing for the main model, so aux recovery must not take the key out of rotation for it.
"""
from types import SimpleNamespace

import httpx
from openai import RateLimitError

from agent import auxiliary_client as aux
from agent.credential_pool import PooledCredential, load_pool

KEY_A, KEY_B = "sk-ant-api03-" + "a" * 40, "sk-ant-api03-" + "b" * 40
AUX_MODEL, MAIN_MODEL = "claude-haiku-4-5", "claude-opus-4-8"


def test_aux_model_429_leaves_the_key_available_for_the_main_model(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    pool = load_pool("anthropic")
    for label, key in (("key-a", KEY_A), ("key-b", KEY_B)):
        pool.add_entry(PooledCredential.from_dict("anthropic", {"label": label, "access_token": key, "source": "manual"}))
    assert load_pool("anthropic").select(model=MAIN_MODEL).runtime_api_key == KEY_A

    rate_limited = RateLimitError(
        f"rate_limit_error: This request would exceed the rate limit of 50 requests per minute ({AUX_MODEL})",
        response=httpx.Response(429, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")),
        body={})
    route = SimpleNamespace(
        client=SimpleNamespace(api_key=KEY_A, base_url="https://api.anthropic.com"), task="compression", tag="",
        resolved_provider="anthropic", base_info="https://api.anthropic.com", resolved_model=AUX_MODEL,
        final_model=AUX_MODEL, main_runtime=None)
    ladder = aux._ladder_credential_rungs(rate_limited, route, {"model": AUX_MODEL}, False)
    assert next(ladder).kind == "call"  # a rate limit is retried once on the same key first
    assert ladder.throw(rate_limited).kind == "retry_same_provider"  # then the pool rotated
    ladder.close()

    after = load_pool("anthropic")
    assert after.select(model=AUX_MODEL).runtime_api_key == KEY_B
    assert after.select(model=MAIN_MODEL).runtime_api_key == KEY_A
