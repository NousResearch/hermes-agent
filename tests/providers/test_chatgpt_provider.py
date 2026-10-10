"""The ChatGPT subscription route has an account catalog, separate from Codex OAuth."""

import io
import json

from providers import get_provider_profile


def test_chatgpt_catalog_preserves_account_order_and_never_reuses_codex(monkeypatch):
    profile = get_provider_profile("openai-chatgpt")
    assert profile is not None
    requests = []

    def open_catalog(request, **kwargs):
        requests.append(request)
        return io.BytesIO(json.dumps({"models": [
            {"slug": "account-second", "display_name": "Second", "visibility": "list"},
            {"slug": "hidden", "display_name": "Hidden", "visibility": "hide"},
            {"slug": "account-first", "display_name": "First", "visibility": "list"},
        ]}).encode())

    monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", open_catalog)
    assert profile.fetch_models(api_key="account-bearer") == ["account-second", "account-first"]
    assert requests[0].full_url == "https://api.openai.com/v1/models"
    assert requests[0].get_header("Authorization") == "Bearer account-bearer"
    assert profile.fallback_models == ()
    assert profile.api_mode == "codex_responses"
    from hermes_cli.auth import resolve_provider
    assert resolve_provider("chatgpt") == "openai-codex"
    assert get_provider_profile("openai-codex").base_url != profile.base_url


def test_chatgpt_catalog_requires_its_account_and_keeps_tokens_on_official_origin(monkeypatch):
    profile = get_provider_profile("openai-chatgpt")
    assert profile is not None
    requests = []
    monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", lambda *a, **k: requests.append(a))
    assert profile.fetch_models() is None
    assert profile.fetch_models(api_key="account-bearer", base_url="https://other.example/v1") is None
    assert requests == []


def test_chatgpt_forces_streaming_without_changing_the_public_api_key_route():
    from types import SimpleNamespace
    from agent.model_metadata import _infer_provider_from_url
    from agent.turn_api_call import _should_stream

    agent = SimpleNamespace(provider="openai-chatgpt", _disable_streaming=True,
                            base_url="https://api.openai.com/v1")
    assert _should_stream(agent)
    agent.provider = "openai"
    assert not _should_stream(agent)
    assert _infer_provider_from_url(agent.base_url) == "openai"


def test_chatgpt_setup_displays_names_from_the_selected_account(monkeypatch):
    from types import SimpleNamespace
    from hermes_cli.model_setup_flows import _plugin_flow_live_rows

    profile = get_provider_profile("openai-chatgpt")
    selected = SimpleNamespace(runtime_api_key="selected-bearer", runtime_base_url=profile.base_url)
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: SimpleNamespace(select=lambda: selected))
    monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", lambda request, **kw: io.BytesIO(json.dumps({
        "models": [{"slug": "chosen-model", "display_name": "Account model display name", "visibility": "list"}],
    }).encode()))
    ids, notes = _plugin_flow_live_rows(profile, "selected-bearer", profile.base_url)
    assert ids == ["chosen-model"]
    assert notes == {"chosen-model": "Account model display name"}
