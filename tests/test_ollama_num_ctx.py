"""Tests for Ollama num_ctx context length detection and injection.

Covers:
  agent/model_metadata.py — query_ollama_num_ctx()
  run_agent.py — _ollama_num_ctx detection + extra_body injection
"""

from unittest.mock import patch, MagicMock


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
        assert call_args[1]["json"]["name"] == "qwen2.5:7b" or call_args[0][1] is not None
        assert result == 32768

    def test_handles_qwen2_architecture_key(self):
        """Different model architectures use different key prefixes in model_info."""
        show_data = {
            "model_info": {"qwen2.context_length": 65536},
            "parameters": "",
        }
        mock_ctx, _ = _mock_httpx_client(show_data)

        with patch("agent.model_metadata.detect_local_server_type", return_value="ollama"):
            import httpx
            with patch.object(httpx, "Client", return_value=mock_ctx):
                result = query_ollama_num_ctx("qwen2.5:32b", "http://localhost:11434")

        assert result == 65536



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


# ═══════════════════════════════════════════════════════════════════════
# Level 3: init-order — compressor window must clamp to effective num_ctx
# (#57275 residual claim 3 / #60103 init-order half)
# ═══════════════════════════════════════════════════════════════════════


class TestCompressorClampsToNumCtx:
    """A config setting ONLY model.ollama_num_ctx (no model.context_length)
    must not leave the compressor targeting the probed model window while
    requests run at the smaller served num_ctx."""

    def _build_agent(self, cfg, probed_ctx):
        import agent.context_compressor as cc_mod
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
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
                base_url="http://localhost:11434/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

    def test_num_ctx_only_config_clamps_compressor_window(self):
        agent = self._build_agent(
            {"agent": {}, "model": {"ollama_num_ctx": 65536}}, probed_ctx=262144
        )
        assert agent._ollama_num_ctx == 65536
        # The compressor must target the served window, not the probed 256K —
        # otherwise its trigger sits far above what the server accepts and
        # compaction never fires (#57275 claim 3).
        assert agent.context_compressor.context_length == 65536
        assert agent.context_compressor.threshold_tokens < 65536

    def test_larger_num_ctx_does_not_inflate_compressor_window(self):
        agent = self._build_agent(
            {"agent": {}, "model": {"ollama_num_ctx": 131072}}, probed_ctx=65536
        )
        # num_ctx above the resolved window must not RAISE the compressor
        # window: the clamp is one-directional.
        assert agent.context_compressor.context_length == 65536

    def _build_cloud_agent(self, cfg, probed_ctx):
        import agent.context_compressor as cc_mod
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
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
                model="glm-5.3-flash",
                api_key="sk-test",
                base_url="https://api.z.ai/api/paas/v4",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

    def test_cloud_endpoint_ignores_ollama_num_ctx_override(self):
        """A stale model.ollama_num_ctx (set to cap VRAM on a local Ollama
        box) must not clamp a cloud session's compressor to that value —
        ollama_num_ctx only steers local endpoints, matching the
        auto-detection branch."""
        agent = self._build_cloud_agent(
            {"agent": {}, "model": {"ollama_num_ctx": 65536}},
            probed_ctx=1000000,
        )
        # The override is still recorded (it applies when the user points
        # the session back at the local server)…
        assert agent._ollama_num_ctx == 65536
        # …but the compressor keeps the provider-resolved 1M window.
        assert agent.context_compressor.context_length == 1000000
