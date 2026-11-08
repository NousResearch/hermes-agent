"""The OpenAI-compatible doors report the model that served a turn, never an echo (#118254).

``/v1/chat/completions`` and ``/v1/responses`` used to echo the request's ``model`` back even
when Hermes ran another one (an ignored bare model, a ``model_routes`` alias, a fallback), which
made clients believe a model pin worked. A requested model the provider rejects is an OpenAI
``404 model_not_found`` instead of a 200 whose content is the failure.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from tests.gateway.test_api_server import _create_app, _make_routing_adapter


_ENDPOINTS = [
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
    ("/v1/responses", {"input": "hi"}),
]


def _terminal_model(endpoint, body_text):
    frames = [json.loads(line[6:]) for line in body_text.splitlines() if line.startswith("data: {")]
    pick = (lambda f: f) if endpoint.endswith("chat/completions") else (lambda f: f["response"])
    return pick(frames[0])["model"], pick(frames[-1])["model"], frames


class TestResponseModelReportsServedModel:
    """The response ``model`` names the model that served the turn: a route alias, an ignored
    bare model and a session /model override are never echoed back as if they had run."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("endpoint", "payload"), _ENDPOINTS)
    async def test_nonstreaming_endpoints_honor_bare_model_and_report_runtime(
        self, endpoint, payload,
    ):
        """Regression for #118254: an alias must not be echoed as the served model."""
        adapter = _make_routing_adapter(
            {"client-route": {"model": "provider/served-model"}}
        )
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "ok", "messages": [], "api_calls": 1,
                     "runtime": {"model": "provider/served-model"}},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )
                resp = await cli.post(
                    endpoint, json={"model": "client-route", **payload},
                )
                response_data = await resp.json()
        assert resp.status == 200
        assert mock_run.call_args.kwargs["route"] == {"model": "provider/served-model"}
        assert mock_run.call_args.kwargs["route_source"] == "model_routes"
        assert response_data["model"] == "provider/served-model"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("endpoint", "payload"), _ENDPOINTS)
    async def test_streaming_endpoints_complete_with_effective_routed_model(self, endpoint, payload):
        adapter = _make_routing_adapter(
            {"client-route": {"model": "provider/served-model"}}
        )
        app = _create_app(adapter)

        async def _run(**kwargs):
            callback = kwargs.get("stream_delta_callback")
            if callback:
                callback("ok")
            return (
                {"final_response": "ok", "messages": [], "api_calls": 1,
                 "runtime": {"model": "provider/served-model"}},
                {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            )

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", side_effect=_run) as mock_run:
                resp = await cli.post(endpoint, json={"model": "client-route", "stream": True, **payload})
                body = await resp.text()
        initial, terminal, _ = _terminal_model(endpoint, body)
        assert initial == terminal == "provider/served-model"
        assert mock_run.call_args.kwargs["route"] == {"model": "provider/served-model"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize(("endpoint", "payload"), _ENDPOINTS)
    async def test_ignored_bare_model_reports_the_default_that_ran(self, endpoint, payload, stream):
        """direct_model_requests is off: the bare model is not honored, so the response names the
        model that served the turn instead of echoing the request (#118254). Before the turn runs
        (the first SSE frame) that is the advertised agent name, never the unhonored value."""
        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        app = _create_app(adapter)

        async def _run(**kwargs):
            if kwargs.get("stream_delta_callback"):
                kwargs["stream_delta_callback"]("ok")
            return ({"final_response": "ok", "messages": [], "api_calls": 1,
                     "runtime": {"model": "default-model", "route_source": "global"}},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", side_effect=_run) as mock_run:
                resp = await cli.post(endpoint, json={"model": "SomeOtherModel", "stream": stream, **payload})
                body = await resp.text()
        assert resp.status == 200
        kwargs = mock_run.call_args.kwargs
        assert "requested_model" not in kwargs
        # The turn records what served it, with the ignored ask on record as not applied.
        assert kwargs["route_source"] == "global"
        assert kwargs["requested_runtime"]["model"] == "SomeOtherModel"
        if stream:
            initial, terminal, _ = _terminal_model(endpoint, body)
            assert (initial, terminal) == (adapter._model_name, "default-model")
        else:
            assert json.loads(body)["model"] == "default-model"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("endpoint", "payload"), _ENDPOINTS)
    async def test_virtual_model_keeps_the_advertised_name(self, endpoint, payload):
        """Asking for the advertised agent name is not a model selection: no requested runtime
        is recorded and the response keeps that name, not the backing model id."""
        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "ok", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )
                resp = await cli.post(endpoint, json={"model": adapter._model_name, **payload})
                data = await resp.json()
        assert "requested_runtime" not in mock_run.call_args.kwargs
        assert data["model"] == adapter._model_name

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("endpoint", "payload"), _ENDPOINTS)
    async def test_unknown_requested_model_is_a_404_model_not_found(self, endpoint, payload):
        """A requested model the provider rejects is an OpenAI ``404 model_not_found`` error,
        not a 200 whose assistant content is the failure text."""
        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "Model 'no-such-model' isn't available.", "messages": [],
                     "api_calls": 1, "failed": True, "completed": False,
                     "error": "HTTP 404: no such model", "failure_reason": "model_not_found"},
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post(
                    endpoint, json={"model": "no-such-model", "provider": "custom", **payload})
                data = await resp.json()
        assert resp.status == 404
        assert data["error"]["code"] == "model_not_found"
        assert data["error"]["param"] == "model"
        assert "no-such-model" in data["error"]["message"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("endpoint", "payload"), _ENDPOINTS)
    async def test_unknown_requested_model_streams_a_model_not_found_error(self, endpoint, payload):
        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        app = _create_app(adapter)

        async def _run(**kwargs):
            return ({"final_response": "", "messages": [], "api_calls": 1, "failed": True,
                     "completed": False, "error": "HTTP 404: no such model",
                     "failure_reason": "model_not_found"},
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", side_effect=_run):
                resp = await cli.post(endpoint, json={
                    "model": "no-such-model", "provider": "custom", "stream": True, **payload})
                body = await resp.text()
        _, _, frames = _terminal_model(endpoint, body)
        last = frames[-1] if endpoint.endswith("chat/completions") else frames[-1]["response"]
        assert last["error"]["code"] == "model_not_found"
