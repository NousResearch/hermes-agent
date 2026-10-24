"""The summarizer's own client carries the user's ``model.default_headers``.

Companion to ``tests/agent/test_auxiliary_user_default_headers.py`` (#40033). That
file covers the routed auxiliary path (``resolve_provider_client``), whose docstring
already counts context compression among the auxiliary client's jobs. The compressor
does not go through it when its endpoint matches no known provider: it builds its own
OpenAI client from ``base_url`` + ``api_key_env``, and that client used to carry the
OpenAI SDK's own identifying headers whatever the user configured. So a custom
endpoint behind a gateway or WAF answered the main turn and the other auxiliary calls
while rejecting (or misattributing) the summary call sent to the same endpoint.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

_API_KEY_ENV = "HERMES_TEST_SUMMARY_KEY"


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Redirect HERMES_HOME so load_config() reads our test config.yaml."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv(_API_KEY_ENV, "test-key")
    (hermes_home / "config.yaml").write_text("model:\n  default: test-model\n")


def _write_config(tmp_path, config_dict):
    import hermes_yaml as yaml

    (tmp_path / ".hermes" / "config.yaml").write_text(yaml.safe_dump(config_dict))


def _compressor():
    """A compressor with everything _init_summarizer() reads, and no tokenizer."""
    from trajectory_compressor import TrajectoryCompressor

    compressor = TrajectoryCompressor.__new__(TrajectoryCompressor)
    # A loopback base_url, like a local gateway: it matches no known provider, so
    # _detect_provider() answers "" and the raw-client branch is the one taken.
    compressor.config = SimpleNamespace(
        base_url="http://127.0.0.1:3425/v1",
        api_key_env=_API_KEY_ENV,
        summarization_model="test-model",
        max_concurrent_requests=1,
    )
    return compressor


def test_summarizer_client_carries_user_default_headers(tmp_path):
    _write_config(tmp_path, {
        "model": {"default": "m", "default_headers": {"User-Agent": "hermes-agent", "X-Waf": "1"}},
    })

    compressor = _compressor()
    with patch("openai.OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        compressor._init_summarizer()

    assert mock_openai.called
    headers = mock_openai.call_args.kwargs.get("default_headers") or {}
    assert headers.get("User-Agent") == "hermes-agent"
    assert headers.get("X-Waf") == "1"


def test_summarizer_client_without_config_keeps_sdk_defaults(tmp_path):
    """No ``model.default_headers``: the client is built exactly as before."""
    compressor = _compressor()
    with patch("openai.OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        compressor._init_summarizer()

    assert mock_openai.called
    assert not mock_openai.call_args.kwargs.get("default_headers")


def test_summarizer_client_honors_extra_headers_alias(tmp_path):
    """``model.extra_headers`` wins over ``model.default_headers``, as on the routed path."""
    _write_config(tmp_path, {
        "model": {
            "default": "m",
            "default_headers": {"User-Agent": "curl/8.7.1", "X-Keep": "1"},
            "extra_headers": {"User-Agent": "hermes-agent"},
        },
    })

    compressor = _compressor()
    with patch("openai.OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        compressor._init_summarizer()

    headers = mock_openai.call_args.kwargs.get("default_headers") or {}
    assert headers.get("User-Agent") == "hermes-agent"
    assert headers.get("X-Keep") == "1"


def test_async_summarizer_client_carries_the_same_headers(tmp_path):
    """The lazily built async client names itself the same way the sync one does."""
    _write_config(tmp_path, {
        "model": {"default": "m", "default_headers": {"User-Agent": "hermes-agent"}},
    })

    compressor = _compressor()
    with patch("openai.OpenAI") as mock_openai, patch("openai.AsyncOpenAI") as mock_async:
        mock_openai.return_value = MagicMock()
        mock_async.return_value = MagicMock()
        compressor._init_summarizer()
        compressor._get_async_client()

    assert mock_async.called
    headers = mock_async.call_args.kwargs.get("default_headers") or {}
    assert headers.get("User-Agent") == "hermes-agent"
