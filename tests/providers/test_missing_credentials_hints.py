"""Missing-credential guidance seams: ``ProviderProfile.missing_credentials_hint`` on every "No usable
credentials found" surface and ``ACPClient.auth_required_hint`` on ACP's auth-required error."""

import sys

import pytest

from agent.acp_client import ACPClient
from providers.base import ProviderProfile

_HINT = "Run `fake-cli login` to sign in."


class _HintedProfile(ProviderProfile):
    def missing_credentials_hint(self):
        return _HINT


@pytest.fixture
def hinted_provider(tmp_path, monkeypatch):
    """A registry api-key provider (a plugin profile) whose key is unset, with or without a hint."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("FAKE_HINTED_KEY", raising=False)
    import providers
    from hermes_cli.auth import PROVIDER_REGISTRY, ProviderConfig

    def install(profile_cls):
        profile = profile_cls(name="fake-hinted", auth_type="api_key", env_vars=("FAKE_HINTED_KEY",),
                              base_url="https://api.fake-hinted.test/v1")
        monkeypatch.setattr(providers, "get_provider_profile",
                            lambda name: profile if name == "fake-hinted" else None)
        monkeypatch.setitem(PROVIDER_REGISTRY, "fake-hinted", ProviderConfig(
            id="fake-hinted", name="Fake", auth_type="api_key", inference_base_url=profile.base_url,
            api_key_env_vars=("FAKE_HINTED_KEY",)))
        return profile
    return install


@pytest.mark.parametrize("profile_cls, suffix", [(_HintedProfile, f" {_HINT}"), (ProviderProfile, "")])
def test_runtime_resolution_appends_the_profile_hint(hinted_provider, profile_cls, suffix):
    hinted_provider(profile_cls)
    from hermes_cli.auth import AuthError
    from hermes_cli.runtime_provider import resolve_runtime_provider

    with pytest.raises(AuthError) as excinfo:
        resolve_runtime_provider(requested="fake-hinted")
    assert str(excinfo.value) == f"No usable credentials found for provider 'fake-hinted'. Set FAKE_HINTED_KEY.{suffix}"


@pytest.mark.parametrize("profile_cls, suffix", [(_HintedProfile, f" {_HINT}"), (ProviderProfile, "")])
def test_readiness_check_appends_the_profile_hint(hinted_provider, profile_cls, suffix):
    hinted_provider(profile_cls)
    from hermes_cli.runtime_readiness import check_runtime_readiness

    verdict = check_runtime_readiness(
        "fake-hinted", resolve=lambda: ("m", {"provider": "fake-hinted", "api_key": "", "source": "none"}))
    assert verdict["ok"] is False
    assert verdict["error"] == f"No usable credentials found for fake-hinted.{suffix}"


class _RaisingHintProfile(ProviderProfile):
    def missing_credentials_hint(self):
        raise RuntimeError("plugin host is down")


class _ListHintProfile(ProviderProfile):
    def missing_credentials_hint(self):
        return ["not", "a", "string"]


@pytest.mark.parametrize("profile_cls", [_RaisingHintProfile, _ListHintProfile])
def test_a_broken_hint_keeps_the_auth_error_and_the_fallback_chain(hinted_provider, monkeypatch, caplog, profile_cls):
    hinted_provider(profile_cls)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")
    from hermes_cli import runtime_readiness
    from hermes_cli.auth import AuthError
    from hermes_cli.runtime_provider import resolve_runtime_provider, resolve_runtime_with_fallback

    monkeypatch.setattr(runtime_readiness, "_HINT_FAILURES_LOGGED", set(), raising=False)
    for _ in range(2):
        with pytest.raises(AuthError) as excinfo:
            resolve_runtime_provider(requested="fake-hinted")
        assert str(excinfo.value) == "No usable credentials found for provider 'fake-hinted'. Set FAKE_HINTED_KEY."
    runtime, entry = resolve_runtime_with_fallback(
        {"fallback_providers": [{"provider": "openrouter", "model": "acme/model"}]}, requested="fake-hinted")
    assert (runtime["provider"], entry["provider"]) == ("openrouter", "openrouter")
    warned = [r for r in caplog.records if "missing_credentials_hint failed" in r.getMessage()]
    assert len(warned) == (1 if profile_cls is _RaisingHintProfile else 0)  # once per provider, not per call


_AUTH_REQUIRED_AGENT = """import json, sys
for line in sys.stdin:
    request = json.loads(line)
    if request.get("method") == "initialize":
        reply = {"result": {"protocolVersion": 1, "authMethods": [{"id": "fake", "name": "Fake login"}]}}
    else:
        reply = {"error": {"code": -32000, "message": "Authentication required"}}
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], **reply}), flush=True)
"""


class _FakeLoginClient(ACPClient):
    label = "Fake ACP"
    auth_required_hint = _HINT


@pytest.mark.parametrize("client_cls, suffix", [(_FakeLoginClient, f" {_HINT}"), (ACPClient, "")])
def test_acp_auth_required_error_carries_the_client_hint(tmp_path, client_cls, suffix):
    agent = tmp_path / "auth_required_agent.py"
    agent.write_text(_AUTH_REQUIRED_AGENT, encoding="utf-8")
    client = client_cls(command=sys.executable, args=[str(agent)], acp_cwd=str(tmp_path))
    with pytest.raises(RuntimeError) as excinfo:
        with client._session(30):
            pass
    assert str(excinfo.value) == f"{client_cls.label} session/new failed: Authentication required{suffix}"
