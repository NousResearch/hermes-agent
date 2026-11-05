"""A llama.cpp router lists each model's trained window as ``context_length`` and answers the bare /props for
itself (``role: router``, n_ctx 0). Hermes must plan with the window the child is launched with: a 256K-trained
model served at --ctx-size 65536 was planned at 262,144, so it never compressed before the server's limit."""

from __future__ import annotations

import httpx
import pytest

from tests.agent.metadata_transport import metadata_transport  # fixture, requested by name

_ROUTER_OWN_PROPS = {"role": "router", "model_alias": "llama-server",
                     "default_generation_settings": {"params": None, "n_ctx": 0}}


@pytest.fixture(autouse=True)
def _clear_endpoint_caches():
    from agent import model_metadata as mm

    mm._endpoint_model_metadata_cache.clear()
    mm._endpoint_model_metadata_cache_time.clear()
    yield
    mm._endpoint_model_metadata_cache.clear()
    mm._endpoint_model_metadata_cache_time.clear()


def _router_models(state: str, ctx_args: list[str], trained: int = 262144):
    return httpx.Response(200, json={"data": [{
        "id": "qwen", "owned_by": "llamacpp", "context_length": trained,
        "status": {"value": state, "args": ["llama-server", "--alias", "qwen", *ctx_args]},
    }]})


@pytest.mark.parametrize("ctx_args, planned", [
    (["--ctx-size", "65536", "--cache-ram", "0"], 65536),
    (["--ctx-size", "524288"], 262144),  # llama-server caps a slot at the trained window
])
def test_resolver_plans_an_unloaded_router_model_at_its_launch_window(request, ctx_args, planned):
    from agent.model_metadata import get_model_context_length

    responses, requests = request.getfixturevalue("metadata_transport")
    responses.extend([
        _router_models("unloaded", ctx_args),
        httpx.Response(404, text="not found"),  # /v1/props
        httpx.Response(200, json=_ROUTER_OWN_PROPS),  # /props
        _router_models("unloaded", ctx_args),  # native /models
    ])

    assert get_model_context_length("qwen", base_url="http://localhost:8000/v1", provider="custom") == planned
    assert not [r for r in requests if r.url.params.get("model")]  # never probes (or autoloads) the child


@pytest.mark.parametrize("bare_props", [
    httpx.Response(200, json=_ROUTER_OWN_PROPS),  # current routers answer for themselves
    httpx.Response(400, json={"error": {"message": "model is not loaded"}}),  # older routers refuse
])
def test_loaded_router_child_reports_its_allocated_window(request, bare_props):
    from agent import model_metadata as mm

    responses, requests = request.getfixturevalue("metadata_transport")
    responses.extend([
        _router_models("loaded", ["--ctx-size", "131072"]),
        httpx.Response(404, text="not found"),  # /v1/props
        bare_props,  # /props
        _router_models("loaded", ["--ctx-size", "131072"]),  # native /models
        httpx.Response(404, text="not found"),  # /v1/props?model=qwen
        httpx.Response(200, json={"default_generation_settings": {"n_ctx": 65536}, "model_alias": "qwen"}),
    ])
    result = mm.fetch_endpoint_model_metadata("http://localhost:8000/v1", force_refresh=True)

    assert result["qwen"]["context_length"] == 65536  # the slot the child allocated, not its launch args
    child_reads = [r for r in requests if r.url.params.get("model") == "qwen"]
    assert child_reads and all(r.url.params.get("autoload") == "false" for r in child_reads)
