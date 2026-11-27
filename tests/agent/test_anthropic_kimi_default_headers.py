"""User ``model.default_headers`` must reach Kimi /coding Anthropic-wire clients.

The OpenAI-wire client builders merge ``model.default_headers`` (main agent and
auxiliary alike), but the Anthropic-wire ``build_anthropic_client`` never saw the
profile, so the kimi branch sent only the built-in attribution headers and any
user override — e.g. a custom ``User-Agent`` an endpoint requires for client
attribution — was silently dropped on title/compression/vision aux calls.
"""

import pytest

from agent.anthropic_adapter import build_anthropic_client

KIMI_CODING_URL = "https://api.kimi.com/coding"


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Redirect HERMES_HOME so config loads read our test config.yaml."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / "config.yaml").write_text("model:\n  default: test-model\n")


def _write_config(tmp_path, config_dict):
    import yaml
    (tmp_path / ".hermes" / "config.yaml").write_text(yaml.dump(config_dict))


def _wire_headers(client) -> dict:
    """Headers the SDK would put on a /v1/messages POST."""
    from anthropic._models import FinalRequestOptions

    return dict(client._build_headers(FinalRequestOptions(method="post", url="/v1/messages", json_data={})))


def test_kimi_client_merges_user_default_headers(tmp_path):
    """User headers reach the wire and win over built-in attribution."""
    pytest.importorskip("anthropic")
    _write_config(tmp_path, {
        "model": {
            "default": "m",
            "default_headers": {"User-Agent": "user-ua/1.0", "X-Extra": "1", "X-Drop": None},
        },
    })
    client = build_anthropic_client("sk-test", base_url=KIMI_CODING_URL)
    headers = _wire_headers(client)
    assert headers.get("user-agent") == "user-ua/1.0"  # user wins over attribution
    assert headers.get("x-extra") == "1"
    assert "x-drop" not in headers  # None values skipped


def test_kimi_client_without_config_keeps_attribution(tmp_path):
    """No configured default_headers: behavior unchanged (attribution UA only)."""
    pytest.importorskip("anthropic")
    client = build_anthropic_client("sk-test", base_url=KIMI_CODING_URL)
    headers = _wire_headers(client)
    assert headers.get("user-agent", "").startswith("HermesAgent/")
    assert headers.get("x-title") == "Hermes Agent"
