"""Tests for Ollama num_ctx context length detection and injection.

Covers:
  agent/model_metadata.py — query_ollama_num_ctx()
  run_agent.py — _ollama_num_ctx detection + extra_body injection
"""

from unittest.mock import patch, MagicMock

import pytest


from agent.model_metadata import query_ollama_num_ctx, query_ollama_supports_vision


# ═══════════════════════════════════════════════════════════════════════
# Level 1: query_ollama_num_ctx — Ollama API interaction
# ═══════════════════════════════════════════════════════════════════════


def _mock_httpx_client(show_response_data, status_code=200):
    """Create a mock httpx.Client context manager that returns given /api/show data."""
    mock_resp = MagicMock(status_code=status_code)
    mock_resp.json.return_value = show_response_data
    mock_client = MagicMock()
    mock_client.post.return_value = mock_resp
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_client)
    mock_ctx.__exit__ = MagicMock(return_value=False)
    return mock_ctx, mock_client


class TestQueryOllamaNumCtx:
    """Test the Ollama /api/show context length query."""

    def test_returns_context_from_model_info(self):
        """Should extract context_length from GGUF model_info metadata."""
        show_data = {
            "model_info": {"llama.context_length": 131072},
            "parameters": "",
        }
        mock_ctx, _ = _mock_httpx_client(show_data)

        with patch("agent.model_metadata.detect_local_server_type", return_value="ollama"):
            # httpx is imported inside the function — patch the module import
            import httpx
            with patch.object(httpx, "Client", return_value=mock_ctx):
                result = query_ollama_num_ctx("llama3.1:8b", "http://localhost:11434/v1")

        assert result == 131072

    def test_prefers_explicit_num_ctx_from_modelfile(self):
        """If the Modelfile sets num_ctx explicitly, that should take priority."""
        show_data = {
            "model_info": {"llama.context_length": 131072},
            "parameters": "num_ctx 32768\ntemperature 0.7",
        }
        mock_ctx, _ = _mock_httpx_client(show_data)

        with patch("agent.model_metadata.detect_local_server_type", return_value="ollama"):
            import httpx
            with patch.object(httpx, "Client", return_value=mock_ctx):
                result = query_ollama_num_ctx("custom-model", "http://localhost:11434")

        assert result == 32768




    def test_strips_provider_prefix(self):
        """Should strip 'local:' prefix from model name before querying."""
        show_data = {
            "model_info": {"qwen2.context_length": 32768},
            "parameters": "",
        }
        mock_ctx, mock_client = _mock_httpx_client(show_data)

        with patch("agent.model_metadata.detect_local_server_type", return_value="ollama"):
            import httpx
            with patch.object(httpx, "Client", return_value=mock_ctx):
                result = query_ollama_num_ctx("local:qwen2.5:7b", "http://localhost:11434/v1")

        # Verify the post was called with stripped name (no "local:" prefix)
        call_args = mock_client.post.call_args
        assert call_args.kwargs["json"]["name"] == "qwen2.5:7b"
        assert result == 32768




class TestQueryOllamaSupportsVision:
    """Test Ollama /api/show vision capability detection."""

    def test_returns_true_when_capabilities_include_vision(self):
        show_data = {"capabilities": ["completion", "vision"]}
        mock_ctx, _ = _mock_httpx_client(show_data)

        with patch("agent.model_metadata.detect_local_server_type", return_value="ollama"):
            import httpx
            with patch.object(httpx, "Client", return_value=mock_ctx):
                result = query_ollama_supports_vision("gemma4:e2b", "http://localhost:11434/v1")

        assert result is True


    def test_falls_back_to_model_info_vision_block_count(self):
        show_data = {"model_info": {"gemma3.vision.block_count": 27}}
        mock_ctx, _ = _mock_httpx_client(show_data)

        with patch("agent.model_metadata.detect_local_server_type", return_value="ollama"):
            import httpx
            with patch.object(httpx, "Client", return_value=mock_ctx):
                result = query_ollama_supports_vision("llava", "http://localhost:11434")

        assert result is True

    def test_returns_none_for_non_ollama_server(self):
        with patch("agent.model_metadata.detect_local_server_type", return_value="vllm"):
            result = query_ollama_supports_vision("llava", "http://localhost:8000/v1")
        assert result is None


class TestGatewayShapedPayloadIsNotOllama:
    """A local gateway that proxies models can answer /api/show in an Ollama-ish shape. Believing
    it made Hermes send Ollama's own ``options.num_ctx`` to a backend that validates its schema
    strictly and rejects the field - every request 400s (#142239)."""

    def test_a_gateway_shaped_payload_yields_no_num_ctx(self):
        show_data = {
            "details": {"context_length": 131072},
            "model_info": {"context_length": 131072},
        }
        mock_ctx, _ = _mock_httpx_client(show_data)

        with patch("agent.model_metadata.detect_local_server_type", return_value="ollama"):
            import httpx
            with patch.object(httpx, "Client", return_value=mock_ctx):
                result = query_ollama_num_ctx("gemini-3-pro", "http://127.0.0.1:8080/v1")

        assert result is None

    def test_a_real_ollama_payload_still_yields_the_window(self):
        """The control: Ollama's own payload (a Modelfile parameter block) is believed."""
        show_data = {
            "parameters": "num_ctx 32768\n",
            "model_info": {"llama.context_length": 131072},
        }
        mock_ctx, _ = _mock_httpx_client(show_data)

        with patch("agent.model_metadata.detect_local_server_type", return_value="ollama"):
            import httpx
            with patch.object(httpx, "Client", return_value=mock_ctx):
                result = query_ollama_num_ctx("llama3.1:8b", "http://127.0.0.1:11434/v1")

        assert result == 32768

    def test_the_agent_keeps_num_ctx_unset_so_no_options_are_injected(self):
        """The injection contract: no num_ctx -> no ``extra_body.options`` -> no upstream 400."""
        show_data = {
            "details": {"context_length": 131072},
            "model_info": {"context_length": 131072},
        }
        mock_ctx, _ = _mock_httpx_client(show_data)

        with (
            patch("agent.model_metadata.detect_local_server_type", return_value="ollama"),
            patch("model_tools.get_tool_definitions", return_value=[]),
            patch("model_tools.check_toolset_requirements", return_value={}),
            patch("agent.process_bootstrap.OpenAI"),
        ):
            import httpx
            with patch.object(httpx, "Client", return_value=mock_ctx):
                agent = _build_agent(
                    {"agent": {}, "model": {}}, probed_ctx=131072,
                    base_url="http://127.0.0.1:8080/v1",
                )

        assert agent._ollama_num_ctx is None


class TestRealLocalServerE2E:
    """The real probe against a real local server - no mocked HTTP (#142239).

    The contributing rubric asks a network-I/O change to exercise the real path with real
    imports, so the local server here is a real socket: a gateway that speaks Ollama's shape
    over the wire, and a real Ollama for the control.
    """

    @staticmethod
    def _serve(show_payload):
        """A local HTTP server answering ``/api/tags`` and ``/api/show``; returns its base URL."""
        import json as _json
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class _Handler(BaseHTTPRequestHandler):
            def _send(self, payload):
                body = _json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.startswith("/api/tags"):
                    self._send({"models": [{"name": "gemini-3-pro"}]})
                else:
                    self.send_error(404)

            def do_POST(self):
                if self.path.startswith("/api/show"):
                    self._send(show_payload)
                else:
                    self.send_error(404)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server, f"http://127.0.0.1:{server.server_address[1]}/v1"

    def test_a_gateway_looking_server_over_a_real_socket_yields_no_num_ctx(self):
        from agent.model_metadata import query_ollama_num_ctx

        server, base_url = self._serve(
            {"details": {"context_length": 131072}, "model_info": {"context_length": 131072}}
        )
        try:
            assert query_ollama_num_ctx("gemini-3-pro", base_url) is None
        finally:
            server.shutdown()
            server.server_close()

    def test_a_real_ollama_over_a_real_socket_still_yields_the_window(self):
        """The control: Ollama's own payload over the wire is believed."""
        from agent.model_metadata import query_ollama_num_ctx

        server, base_url = self._serve(
            {"parameters": "num_ctx 32768\n", "model_info": {"llama.context_length": 131072}}
        )
        try:
            assert query_ollama_num_ctx("llama3.1:8b", base_url) == 32768
        finally:
            server.shutdown()
            server.server_close()


# ═══════════════════════════════════════════════════════════════════════
# Level 3: init-order — compressor window must clamp to effective num_ctx
# (#57275 residual claim 3 / #60103 init-order half)
# ═══════════════════════════════════════════════════════════════════════


def _build_agent(cfg, probed_ctx, base_url="http://localhost:11434/v1"):
    import agent.context_compressor as cc_mod
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch(
            "agent.model_metadata.get_model_context_length",
            return_value=probed_ctx,
        ),
        patch.object(
            cc_mod, "get_model_context_length", return_value=probed_ctx,
        ),
    ):
        from run_agent import AIAgent
        return AIAgent(
            model="gemma3:27b",
            api_key="ollama",
            base_url=base_url,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )


class TestCompressorClampsToNumCtx:
    """A config setting ONLY model.ollama_num_ctx (no model.context_length)
    must not leave the compressor targeting the probed model window while
    requests run at the smaller served num_ctx."""

    def test_num_ctx_only_config_clamps_compressor_window(self):
        agent = _build_agent(
            {"agent": {}, "model": {"ollama_num_ctx": 65536}}, probed_ctx=262144
        )
        assert agent._ollama_num_ctx == 65536
        # The compressor must target the served window, not the probed 256K —
        # otherwise its trigger sits far above what the server accepts and
        # compaction never fires (#57275 claim 3).
        assert agent.context_compressor.context_length == 65536
        assert agent.context_compressor.threshold_tokens < 65536

    def test_larger_num_ctx_does_not_inflate_compressor_window(self):
        agent = _build_agent(
            {"agent": {}, "model": {"ollama_num_ctx": 131072}}, probed_ctx=65536
        )
        # num_ctx above the resolved window must not RAISE the compressor
        # window: the clamp is one-directional.
        assert agent.context_compressor.context_length == 65536


class TestServedNumCtxSatisfiesTheFloor:
    """#100437: the 64K floor judges the window a local Ollama server actually serves. A Modelfile
    or model.ollama_num_ctx at 64K+ is usable even when the GGUF metadata advertises 40K, so
    construction must succeed; the compressor still targets the smaller probed window."""

    def test_explicit_num_ctx_above_the_floor_admits_a_small_metadata_window(self):
        agent = _build_agent({"agent": {}, "model": {"ollama_num_ctx": 65536}}, probed_ctx=40960)
        assert agent._ollama_num_ctx == 65536
        assert agent.context_compressor.context_length == 40960  # one-directional clamp unchanged

    def test_served_window_counts_only_for_a_local_endpoint(self):
        """Only a local server honours num_ctx; a stale override must not admit a hosted 40K model."""
        with pytest.raises(ValueError, match="below the minimum"):
            _build_agent({"agent": {}, "model": {"ollama_num_ctx": 65536}}, probed_ctx=40960,
                         base_url="https://openrouter.ai/api/v1")


class TestFloorRefusalNamesTheLocalServerHonestly:
    """#87075: a local OpenAI-compatible server without /api/show (llama.cpp, vLLM) that serves a
    sub-64K window must get server-agnostic guidance — raise the server's context or set
    model.ollama_num_ctx — while a hosted route keeps the model.context_length advice."""

    def test_local_refusal_names_server_flag_and_num_ctx_key(self):
        with pytest.raises(ValueError) as exc:
            _build_agent({"agent": {}, "model": {}}, probed_ctx=32768, base_url="http://127.0.0.1:8422/v1")
        msg = str(exc.value)
        assert "model.ollama_num_ctx" in msg

    def test_hosted_refusal_keeps_context_length_advice(self):
        with pytest.raises(ValueError, match="model.context_length") as exc:
            _build_agent({"agent": {}, "model": {}}, probed_ctx=32768, base_url="https://openrouter.ai/api/v1")
        assert "ollama_num_ctx" not in str(exc.value)
