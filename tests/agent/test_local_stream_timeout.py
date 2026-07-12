"""Tests for local provider stream read timeout auto-detection.

When a local LLM provider is detected (Ollama, llama.cpp, vLLM, etc.),
the httpx stream read timeout should be automatically increased from the
default 60s to HERMES_API_TIMEOUT (1800s) to avoid premature connection
kills during long prefill phases.
"""

import os
import pytest
from unittest.mock import patch

from agent.model_metadata import is_local_endpoint
from agent.chat_completion_helpers import (
    _dflash_local_first_chunk_timeout,
    _dflash_local_stale_timeout,
    _is_dflash_like_model,
    interruptible_streaming_api_call,
    resolve_dflash_local_first_chunk_timeout,
    resolve_stream_stale_timeout,
)


class TestLocalStreamReadTimeout:
    """Verify stream read timeout auto-detection logic."""

    @pytest.mark.parametrize("base_url", [
        "http://localhost:11434",
        "http://127.0.0.1:8080",
        "http://0.0.0.0:5000",
        "http://192.168.1.100:8000",
        "http://10.0.0.5:1234",
        "http://host.docker.internal:11434",
        "http://host.containers.internal:11434",
        "http://host.lima.internal:11434",
    ])
    def test_local_endpoint_bumps_read_timeout(self, base_url):
        """Local endpoint + default timeout -> bumps to base_timeout."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_STREAM_READ_TIMEOUT", None)
            _base_timeout = float(os.getenv("HERMES_API_TIMEOUT", 1800.0))
            _stream_read_timeout = float(os.getenv("HERMES_STREAM_READ_TIMEOUT", 120.0))
            if _stream_read_timeout == 120.0 and base_url and is_local_endpoint(base_url):
                _stream_read_timeout = _base_timeout
            assert _stream_read_timeout == 1800.0

    def test_user_override_respected_for_local(self):
        """User sets HERMES_STREAM_READ_TIMEOUT -> keep their value even for local."""
        with patch.dict(os.environ, {"HERMES_STREAM_READ_TIMEOUT": "300"}, clear=False):
            _base_timeout = float(os.getenv("HERMES_API_TIMEOUT", 1800.0))
            _stream_read_timeout = float(os.getenv("HERMES_STREAM_READ_TIMEOUT", 120.0))
            base_url = "http://localhost:11434"
            if _stream_read_timeout == 120.0 and base_url and is_local_endpoint(base_url):
                _stream_read_timeout = _base_timeout
            assert _stream_read_timeout == 300.0

    @pytest.mark.parametrize("base_url", [
        "https://api.openai.com",
        "https://openrouter.ai/api",
        "https://api.anthropic.com",
    ])
    def test_remote_endpoint_keeps_default(self, base_url):
        """Remote endpoint -> keep 120s default."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_STREAM_READ_TIMEOUT", None)
            _base_timeout = float(os.getenv("HERMES_API_TIMEOUT", 1800.0))
            _stream_read_timeout = float(os.getenv("HERMES_STREAM_READ_TIMEOUT", 120.0))
            if _stream_read_timeout == 120.0 and base_url and is_local_endpoint(base_url):
                _stream_read_timeout = _base_timeout
            assert _stream_read_timeout == 120.0

    def test_empty_base_url_keeps_default(self):
        """No base_url set -> keep 120s default."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_STREAM_READ_TIMEOUT", None)
            _base_timeout = float(os.getenv("HERMES_API_TIMEOUT", 1800.0))
            _stream_read_timeout = float(os.getenv("HERMES_STREAM_READ_TIMEOUT", 120.0))
            base_url = ""
            if _stream_read_timeout == 120.0 and base_url and is_local_endpoint(base_url):
                _stream_read_timeout = _base_timeout
            assert _stream_read_timeout == 120.0


class TestLocalDflashStaleTimeout:
    """dflash is local, but must not be allowed to wait forever with no chunks."""

    @staticmethod
    def _payload_for_estimated_tokens(tokens: int) -> dict[str, list[str]]:
        return {"messages": ["x" * (tokens * 4)]}

    def _make_agent(self, *, model="dflash", base_url="http://10.10.20.211:8080/v1"):
        from run_agent import AIAgent

        with patch("agent.context_compressor.get_model_context_length", return_value=256_000):
            return AIAgent(
                api_key="sk-dummy",
                base_url=base_url,
                provider="taro",
                model=model,
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                platform="cli",
            )

    @pytest.mark.parametrize(
        "model",
        [
            "dflash",
            "deepseek-v4-flash-w2",
            "deepseek-v4-flash-iq3xxs",
        ],
    )
    def test_default_local_dflash_stream_stale_timeout_is_bounded(
        self,
        monkeypatch,
        tmp_path,
        model,
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / ".env").write_text("", encoding="utf-8")
        monkeypatch.delenv("HERMES_STREAM_STALE_TIMEOUT", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_STALE_TIMEOUT", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_STREAM_STALE_TIMEOUT", raising=False)

        agent = self._make_agent(model=model)

        timeout = resolve_stream_stale_timeout(
            agent,
            {"model": model, "messages": [{"role": "user", "content": "hi"}]},
        )

        assert timeout == 180.0

    @pytest.mark.parametrize(
        "model",
        [
            "dflash",
            "taro/dflash-w2",
            "deepseek-v4-flash-w2",
            "deepseek-v4-flash-iq3xxs",
            "custom/deepseek_v4_flash:iq3xxs",
        ],
    )
    def test_dflash_family_recognizes_aliases_and_production_ids(self, model):
        assert _is_dflash_like_model(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            None,
            "",
            "qwen3.6-27b",
            "deepseek-v4",
            "deepseek-v3-flash-w2",
            "notdeepseek-v4-flash-w2",
            "dflashish",
        ],
    )
    def test_dflash_family_rejects_unrelated_model_ids(self, model):
        assert _is_dflash_like_model(model) is False

    @pytest.mark.parametrize(
        ("estimated_tokens", "expected_timeout"),
        [
            (50_000, 180.0),
            (50_001, 240.0),
            (100_000, 240.0),
            (100_001, 300.0),
            (120_000, 300.0),
        ],
    )
    def test_default_dflash_first_chunk_timeout_scales_with_context(
        self,
        monkeypatch,
        estimated_tokens,
        expected_timeout,
    ):
        monkeypatch.delenv("HERMES_DFLASH_FIRST_CHUNK_TIMEOUT", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_TTFB_TIMEOUT", raising=False)

        timeout = _dflash_local_first_chunk_timeout(
            self._payload_for_estimated_tokens(estimated_tokens),
            "dflash",
        )

        assert timeout == expected_timeout

    @pytest.mark.parametrize(
        "model",
        ["deepseek-v4-flash-w2", "deepseek-v4-flash-iq3xxs"],
    )
    def test_production_dflash_first_chunk_default_allows_observed_cold_start(
        self,
        monkeypatch,
        model,
    ):
        monkeypatch.delenv("HERMES_DFLASH_FIRST_CHUNK_TIMEOUT", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_TTFB_TIMEOUT", raising=False)

        timeout = _dflash_local_first_chunk_timeout({"messages": []}, model)

        assert timeout == 180.0
        assert timeout > 116.0

    def test_production_w2_no_first_chunk_wait_exits_at_family_deadline(
        self,
        monkeypatch,
    ):
        """Exercise the poll-loop guard without sleeping for the real deadline."""
        monkeypatch.delenv("HERMES_STREAM_STALE_TIMEOUT", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_STALE_TIMEOUT", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_STREAM_STALE_TIMEOUT", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_FIRST_CHUNK_TIMEOUT", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_TTFB_TIMEOUT", raising=False)

        class Clock:
            now = 0.0

            @classmethod
            def time(cls):
                return cls.now

        class NoResponseThread:
            def __init__(self, *, target, daemon):
                self.target = target
                self.daemon = daemon

            def start(self):
                return None

            def is_alive(self):
                return True

            def join(self, timeout=None):
                if timeout == 0.3:
                    Clock.now = 181.0

        statuses = []
        replacements = []

        class Agent:
            api_mode = "chat_completions"
            base_url = "http://10.10.20.211:8080/v1"
            provider = "taro"
            model = "deepseek-v4-flash-w2"
            _interrupt_requested = False
            _consecutive_stale_streams = 0

            @staticmethod
            def _touch_activity(_message):
                return None

            @staticmethod
            def _buffer_status(message):
                statuses.append(message)

            @staticmethod
            def _replace_primary_openai_client(*, reason):
                replacements.append(reason)

        monkeypatch.setattr(
            "agent.chat_completion_helpers.threading.Thread",
            NoResponseThread,
        )
        monkeypatch.setattr("agent.chat_completion_helpers.time.time", Clock.time)

        with pytest.raises(
            TimeoutError,
            match=r"Local dflash stream produced no first chunk after 181s .*180s",
        ):
            interruptible_streaming_api_call(
                Agent(),
                {
                    "model": "deepseek-v4-flash-w2",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

        assert len(statuses) == 1
        assert "No first stream chunk from local dflash for 181s" in statuses[0]
        assert statuses[0].endswith("Reconnecting...")
        assert replacements == ["dflash_first_chunk_pool_cleanup"]

    def test_dflash_first_chunk_timeout_has_independent_env(self, monkeypatch):
        monkeypatch.setenv("HERMES_DFLASH_FIRST_CHUNK_TIMEOUT", "45")

        assert _dflash_local_first_chunk_timeout({"messages": []}, "dflash") == 45.0

    def test_configured_model_stale_timeout_is_canonical_for_both_stream_phases(
        self,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / ".env").write_text("", encoding="utf-8")
        (tmp_path / "config.yaml").write_text(
            """providers:
  taro:
    stale_timeout_seconds: 320
    models:
      deepseek-v4-flash-w2:
        stale_timeout_seconds: 360
""",
            encoding="utf-8",
        )
        # Config is canonical even when every legacy environment surface is
        # present. The payload model must also win over the agent default.
        monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "12")
        monkeypatch.setenv("HERMES_DFLASH_STALE_TIMEOUT", "45")
        monkeypatch.setenv("HERMES_DFLASH_STREAM_STALE_TIMEOUT", "50")
        monkeypatch.setenv("HERMES_DFLASH_FIRST_CHUNK_TIMEOUT", "55")
        monkeypatch.setenv("HERMES_DFLASH_TTFB_TIMEOUT", "60")

        agent = self._make_agent(model="deepseek-v4-flash-iq3xxs")
        payload = {
            "model": "deepseek-v4-flash-w2",
            **self._payload_for_estimated_tokens(120_000),
        }
        stream_timeout = resolve_stream_stale_timeout(agent, payload)
        first_chunk_timeout = resolve_dflash_local_first_chunk_timeout(
            agent,
            payload,
            resolved_stream_stale_timeout=stream_timeout,
        )

        assert stream_timeout == 360.0
        assert first_chunk_timeout == 360.0

        provider_payload = {
            "model": "deepseek-v4-flash-iq3xxs",
            **self._payload_for_estimated_tokens(120_000),
        }
        provider_stream_timeout = resolve_stream_stale_timeout(
            agent,
            provider_payload,
        )
        provider_first_chunk_timeout = resolve_dflash_local_first_chunk_timeout(
            agent,
            provider_payload,
            resolved_stream_stale_timeout=provider_stream_timeout,
        )

        assert provider_stream_timeout == 320.0
        assert provider_first_chunk_timeout == 320.0

    def test_generic_local_stream_stale_timeout_still_disables_by_default(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / ".env").write_text("", encoding="utf-8")
        monkeypatch.delenv("HERMES_STREAM_STALE_TIMEOUT", raising=False)

        agent = self._make_agent(model="qwen3.6-27b")

        timeout = resolve_stream_stale_timeout(
            agent,
            {"model": "qwen3.6-27b", "messages": [{"role": "user", "content": "hi"}]},
        )

        assert timeout == float("inf")

    def test_default_local_dflash_non_stream_stale_timeout_is_bounded(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / ".env").write_text("", encoding="utf-8")
        monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_STALE_TIMEOUT", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_STREAM_STALE_TIMEOUT", raising=False)

        agent = self._make_agent(model="deepseek-v4-flash-w2")

        assert agent._compute_non_stream_stale_timeout({"messages": []}) == 180.0

    def test_explicit_stream_stale_timeout_still_wins_for_dflash(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / ".env").write_text("", encoding="utf-8")
        monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "12")
        monkeypatch.setenv("HERMES_DFLASH_STALE_TIMEOUT", "75")

        agent = self._make_agent(model="deepseek-v4-flash-w2")

        assert resolve_stream_stale_timeout(
            agent,
            {"model": "deepseek-v4-flash-w2", "messages": []},
        ) == 12.0

    @pytest.mark.parametrize(
        ("estimated_tokens", "expected_timeout"),
        [
            (10_000, 180.0),
            (10_001, 180.0),
            (25_000, 180.0),
            (25_001, 180.0),
            (50_000, 180.0),
            (50_001, 240.0),
            (100_000, 240.0),
            (100_001, 300.0),
        ],
    )
    def test_default_dflash_stale_timeout_threshold_boundaries(
        self,
        monkeypatch,
        estimated_tokens,
        expected_timeout,
    ):
        monkeypatch.delenv("HERMES_DFLASH_STALE_TIMEOUT", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_STREAM_STALE_TIMEOUT", raising=False)

        timeout = _dflash_local_stale_timeout(
            self._payload_for_estimated_tokens(estimated_tokens),
            "deepseek-v4-flash-iq3xxs",
        )

        assert timeout == expected_timeout

    def test_non_positive_dflash_stale_timeout_disables_watchdog(self, monkeypatch):
        monkeypatch.setenv("HERMES_DFLASH_STALE_TIMEOUT", "0")

        timeout = _dflash_local_stale_timeout(
            self._payload_for_estimated_tokens(1),
            "deepseek-v4-flash-w2",
        )

        assert timeout == float("inf")


class TestIsLocalEndpoint:
    """Direct unit tests for is_local_endpoint."""

    @pytest.mark.parametrize("url", [
        "http://localhost:11434",
        "http://127.0.0.1:8080",
        "http://0.0.0.0:5000",
        "http://[::1]:11434",
        "http://192.168.1.100:8000",
        "http://10.0.0.5:1234",
        "http://172.17.0.1:11434",
    ])
    def test_classic_local_addresses(self, url):
        assert is_local_endpoint(url) is True

    @pytest.mark.parametrize("url", [
        "http://host.docker.internal:11434",
        "http://host.docker.internal:8080/v1",
        "http://gateway.docker.internal:11434",
        "http://host.containers.internal:11434",
        "http://host.lima.internal:11434",
    ])
    def test_container_dns_names(self, url):
        assert is_local_endpoint(url) is True

    @pytest.mark.parametrize("url", [
        "http://ollama:11434",
        "http://litellm:4000/v1",
        "http://hermes-litellm:8080",
        "http://vllm:8000",
    ])
    def test_unqualified_docker_hostnames(self, url):
        """Unqualified hostnames (no dots) are local — Docker Compose, /etc/hosts, etc."""
        assert is_local_endpoint(url) is True

    @pytest.mark.parametrize("url", [
        "https://api.openai.com",
        "https://openrouter.ai/api",
        "https://api.anthropic.com",
        "https://evil.docker.internal.example.com",
    ])
    def test_remote_endpoints(self, url):
        assert is_local_endpoint(url) is False

    @pytest.mark.parametrize("url", [
        "http://100.64.0.0:11434",            # lower bound of CGNAT block
        "http://100.64.0.1:11434/v1",         # lower bound +1
        "http://100.77.243.5:11434",          # representative Tailscale host
        "https://100.100.100.100:443",        # Tailscale MagicDNS anchor
        "https://100.127.255.254:443",        # upper bound -1
        "http://100.127.255.255:11434",       # upper bound of CGNAT block
    ])
    def test_tailscale_cgnat_is_local(self, url):
        """Tailscale 100.64.0.0/10 should be treated as local for timeout bumps."""
        assert is_local_endpoint(url) is True

    @pytest.mark.parametrize("url", [
        "http://100.63.255.255:11434",        # just below CGNAT block
        "http://100.128.0.1:11434",           # just above CGNAT block
        "http://100.200.0.1:11434",           # well outside CGNAT
        "http://99.64.0.1:11434",             # first octet wrong
    ])
    def test_near_but_not_cgnat_is_remote(self, url):
        """Hosts adjacent to but outside 100.64.0.0/10 must not match."""
        assert is_local_endpoint(url) is False
