"""``build_anthropic_client`` asks the profile serving the host for its auth style
(``anthropic_auth_style``) and client headers (``client_headers``), through fake relay profiles.

Contract: a claimed https host gets the profile's style and headers (core's anthropic-beta set, the
credential, transport and hop-by-hop headers, and ``custom_providers`` extra_headers, in any letter case,
still win); unclaimed hosts, hosts whose auth core owns (native Anthropic, OAuth, Azure, MiniMax, Kimi,
Nous, ...), non-ASCII hosts, cleartext URLs that are not strictly local, callable credentials and profiles
that only answer ``client_headers`` keep core's generic behavior, and ``client_headers`` never sees the key
there; a raising hook degrades to it. A profile refresh on ``anthropic_messages`` rebuilds the Anthropic
client, moves every token and base-URL holder, and rolls back completely on failure, on a malformed answer,
or on a base URL that changes the origin other than to an https host the named profile claims (a
host-matched owner's token is adopted on the current route instead).
"""

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.client_lifecycle import ClientLifecycleMixin
from providers.base import ProviderProfile

RELAY = "https://relay.example"
OTHER = "https://other.example"


class _RelayProfile(ProviderProfile):
    style = "bearer"

    def anthropic_auth_style(self, *, base_url):
        return self.style

    def client_headers(self, *, base_url, api_key="", has_images=False):
        return {"Editor-Version": "relay/1", "User-Agent": "relay-agent", "anthropic-beta": "evil"}


class _RecordingProfile(_RelayProfile):
    """Records every hook call (and the key ``client_headers`` received)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    def anthropic_auth_style(self, *, base_url):
        self.calls.append(("anthropic_auth_style", base_url))
        return self.style

    def client_headers(self, *, base_url, api_key="", has_images=False):
        self.calls.append(("client_headers", api_key))
        return {"Editor-Version": "relay/1"}


class _HeadersOnlyProfile(ProviderProfile):
    def client_headers(self, *, base_url, api_key="", has_images=False):
        return {"Editor-Version": "openai-wire/1"}


class _BrokenProfile(ProviderProfile):
    def anthropic_auth_style(self, *, base_url):
        raise RuntimeError("boom")


class _HeaderProfile(_RelayProfile):
    headers: dict = {}

    def client_headers(self, *, base_url, api_key="", has_images=False):
        return dict(self.headers)


def _register(request, profile):
    import providers

    providers.list_providers()  # discovery first, so the fake survives it
    providers.register_provider(profile)

    def _cleanup():
        import hermes_cli.auth

        providers._REGISTRY.pop(profile.name, None)
        providers._SOURCES.pop(profile.name, None)
        for alias in profile.aliases:
            providers._ALIASES.pop(alias, None)
            hermes_cli.auth.PROVIDER_REGISTRY.pop(alias, None)
        providers._PROVIDER_LIST_CACHE = None
        hermes_cli.auth.PROVIDER_REGISTRY.pop(profile.name, None)

    request.addfinalizer(_cleanup)
    return profile


@pytest.fixture
def relay(request):
    return _register(request, _RelayProfile(name="relay-fake", hostnames=("relay.example",)))


def _client_kwargs(base_url, key="tok-123"):
    from agent.anthropic_adapter import build_anthropic_client

    with patch("agent.anthropic_adapter._anthropic_sdk") as sdk, \
            patch("agent.anthropic_adapter._get_claude_code_version", return_value="9.9.9"):
        build_anthropic_client(key, base_url=base_url)
        return sdk.Anthropic.call_args[1]


def _str_headers(kw):
    return {k: v for k, v in kw.get("default_headers", {}).items() if isinstance(v, str)}


def test_claimed_host_uses_profile_bearer_and_headers(relay):
    kw = _client_kwargs(RELAY)
    assert kw["auth_token"] == "tok-123"
    assert "api_key" not in kw
    headers = kw["default_headers"]
    assert headers["Editor-Version"] == "relay/1"
    assert headers["User-Agent"] == "relay-agent"
    assert "evil" not in headers.get("anthropic-beta", "")  # core owns the beta set


def test_claimed_host_can_choose_x_api_key(request):
    profile = _RelayProfile(name="relay-fake", hostnames=("relay.example",))
    profile.style = "api_key"
    _register(request, profile)
    kw = _client_kwargs(RELAY)
    assert kw["api_key"] == "tok-123"
    assert "auth_token" not in kw
    assert kw["default_headers"]["Editor-Version"] == "relay/1"


def test_unclaimed_host_keeps_generic_auth(relay):
    kw = _client_kwargs(OTHER)
    assert kw["api_key"] == "tok-123"
    assert "auth_token" not in kw
    assert "Editor-Version" not in kw.get("default_headers", {})


def test_headers_only_profile_does_not_leak_onto_anthropic_client(request):
    _register(request, _HeadersOnlyProfile(name="headers-fake", hostnames=("relay.example",)))
    kw = _client_kwargs(RELAY)
    assert kw["api_key"] == "tok-123"
    assert "Editor-Version" not in kw.get("default_headers", {})


def test_raising_hook_degrades_to_generic(request):
    _register(request, _BrokenProfile(name="broken-fake", hostnames=("relay.example",)))
    kw = _client_kwargs(RELAY)
    assert kw["api_key"] == "tok-123"
    assert "auth_token" not in kw


def test_custom_provider_extra_headers_still_win(relay):
    with patch("agent.anthropic_adapter._custom_provider_extra_headers", return_value={"Editor-Version": "user/9"}):
        kw = _client_kwargs(RELAY)
    assert kw["default_headers"]["Editor-Version"] == "user/9"


def test_custom_provider_extra_headers_win_in_any_letter_case(relay):
    """A user ``editor-version`` replaces the profile's ``Editor-Version``: one header line, the user's value."""
    with patch("agent.anthropic_adapter._custom_provider_extra_headers", return_value={"editor-version": "user/9"}):
        kw = _client_kwargs(RELAY)
    editor = {k: v for k, v in _str_headers(kw).items() if k.lower() == "editor-version"}
    assert editor == {"editor-version": "user/9"}
    assert "X-Api-Key" in kw["default_headers"]  # core's Omit() credential guard is never removed


@pytest.mark.parametrize("user_name", ["authorization", "AUTHORIZATION", "Authorization"])
def test_user_credential_header_takes_the_sdk_casing(relay, user_name):
    """A user ``authorization`` extra header replaces the SDK's ``Authorization`` line instead of being sent
    as a second credential line next to it (the SDK spells its own credential headers in canonical case)."""
    with patch("agent.anthropic_adapter._custom_provider_extra_headers", return_value={user_name: "Bearer user"}):
        kw = _client_kwargs(RELAY)
    auth = {k: v for k, v in kw["default_headers"].items() if k.lower() == "authorization"}
    assert auth == {"Authorization": "Bearer user"}
    assert "X-Api-Key" in kw["default_headers"]  # the other credential stays guarded


def test_base_profile_hook_is_noop():
    assert ProviderProfile(name="x").anthropic_auth_style(base_url=RELAY) is None


@pytest.mark.parametrize("url, host", [
    ("https://api.anthropic.com", "anthropic.com"),
    ("https://my-res.services.ai.azure.com/anthropic", "azure.com"),
    ("https://api.minimax.io/anthropic", "minimax.io"),
    ("https://api.kimi.com/coding", "kimi.com"),
    ("https://api.moonshot.ai/anthropic", "moonshot.ai"),
    ("https://inference-api.nousresearch.com/v1", "nousresearch.com"),
    ("https://example.palantirfoundry.com/anthropic", "palantirfoundry.com"),
])
def test_profile_claiming_a_core_owned_host_is_never_asked(request, url, host):
    """Core owns these hosts' auth: a plugin claiming one changes nothing and never receives the key."""
    from agent.anthropic_adapter import _route_profile_client_overrides

    profile = _register(request, _RecordingProfile(name="claim-fake", hostnames=(host,)))
    profile.style = "api_key"
    kw = _client_kwargs(url, key="core-owned-secret")
    assert "Editor-Version" not in _str_headers(kw)
    assert _route_profile_client_overrides(url, "core-owned-secret") == (None, {})
    assert profile.calls == []  # client_headers never saw the key


def test_core_bearer_host_keeps_bearer_when_a_profile_answers_api_key(request):
    profile = _register(request, _RecordingProfile(name="claim-fake", hostnames=("api.commandcode.ai",)))
    profile.style = "api_key"
    kw = _client_kwargs("https://api.commandcode.ai/anthropic")
    assert kw["auth_token"] == "tok-123" and "api_key" not in kw
    assert profile.calls == []


@pytest.mark.parametrize("base_url", [None, "https://api.anthropic.com"])
def test_native_anthropic_oauth_unchanged_with_a_profile_registered(request, base_url):
    profile = _register(request, _RecordingProfile(name="claim-fake", hostnames=("anthropic.com",)))
    profile.style = "api_key"
    kw = _client_kwargs(base_url, key="sk-ant-oat01-native-oauth")
    assert kw["auth_token"] == "sk-ant-oat01-native-oauth" and "api_key" not in kw
    headers = _str_headers(kw)
    assert headers["user-agent"].startswith("claude-code/")
    assert "Editor-Version" not in headers
    assert profile.calls == []


def test_oauth_or_kimi_core_style_is_never_overridden(request):
    from agent.anthropic_adapter import _route_profile_client_overrides

    profile = _register(request, _RecordingProfile(name="relay-fake", hostnames=("relay.example",)))
    for core_style in ("oauth", "kimi"):
        assert _route_profile_client_overrides(RELAY, "tok", core_style=core_style) == (None, {})
    assert profile.calls == []


def test_callable_credential_skips_the_profile(request):
    from agent.anthropic_adapter import build_anthropic_client

    profile = _register(request, _RecordingProfile(name="relay-fake", hostnames=("relay.example",)))
    with patch("agent.anthropic_adapter._anthropic_sdk") as sdk:
        build_anthropic_client(lambda: "minted-jwt", base_url=RELAY)
    kw = sdk.Anthropic.call_args[1]
    assert kw["auth_token"] == "entra-id-bearer-via-http-hook"
    assert "Editor-Version" not in _str_headers(kw)
    assert profile.calls == []


def test_cleartext_non_local_url_is_refused_but_local_http_is_allowed(request):
    profile = _register(request, _RecordingProfile(name="relay-fake", hostnames=("relay.example", "localhost")))
    kw = _client_kwargs("http://relay.example")
    assert kw["api_key"] == "tok-123" and "auth_token" not in kw
    assert profile.calls == []
    kw = _client_kwargs("http://localhost:8080")
    assert kw["auth_token"] == "tok-123"
    assert kw["default_headers"]["Editor-Version"] == "relay/1"


@pytest.mark.parametrize("url, ok", [
    ("http://localhost:8080", True), ("http://127.0.0.1", True), ("http://[::1]:8080", True),
    ("http://10.0.0.5", True), ("http://192.168.1.2:11434", True), ("http://169.254.1.1", True),
    ("http://host.docker.internal:8080", True),
    ("http://3627730435", False),  # integer IPv4: the resolver maps it to a public address
    ("http://0xd83ad603", False),  # hex IPv4, same
    ("http://relay", False),  # dotless names resolve through the DNS search suffix
    ("http://relay.local", False), ("http://8.8.8.8", False), ("http://relay.example", False),
    ("https://[x]", False), ("", False),
    ("http://[::ffff:8.8.8.8]", False),  # IPv4-mapped: judged by 8.8.8.8 on every Python version
    ("http://[::ffff:10.0.0.1]", True),
    ("http://[2002:808:808::1]", False),  # 6to4 tunnels to 8.8.8.8 although is_private says True
    ("http://198.18.0.1", False),  # benchmarking range, also is_private
    ("http://10.0.0.1.", False),  # trailing dot: base_url_hostname strips it, httpx does not
    ("http://127.0.0.1.:8080", False), ("http://localhost.:8080", True),
    ("http://localhost..", False), ("http://host.docker.internal..", False),  # empty DNS label
    ("http://.docker.internal", False),
    ("http://[fd00::1]", True), ("http://[fe80::1]", True), ("http://0.0.0.0:8080", True),
])
def test_cleartext_gate_is_strictly_local(url, ok):
    from agent.anthropic_endpoints import _cleartext_endpoint_ok

    assert _cleartext_endpoint_ok(url) is ok


def test_integer_ipv4_http_url_is_never_handed_to_a_profile(request):
    profile = _register(request, _RecordingProfile(name="relay-fake", hostnames=("3627730435",)))
    kw = _client_kwargs("http://3627730435", key="live-secret")
    assert kw["api_key"] == "live-secret" and "auth_token" not in kw
    assert profile.calls == []


def test_profile_cannot_set_credential_or_transport_headers(request):
    profile = _HeaderProfile(name="cred-fake", hostnames=("relay.example",))
    profile.headers = {
        "Authorization": "Bearer evil", "X-Api-Key": "evil", "x-api-key": "evil", "ANTHROPIC-BETA": "evil",
        "anthropic-version": "evil", "Host": "evil.example", "Content-Length": "1", "Transfer-Encoding": "evil",
        "Connection": "evil", "Proxy-Authorization": "evil", "Cookie": "evil", "Editor-Version": "relay/1",
    }
    _register(request, profile)
    kw = _client_kwargs(RELAY)
    assert kw["auth_token"] == "tok-123"  # the SDK's single credential stays the only one
    headers = kw["default_headers"]
    # Only core's Omit() guard may occupy the other credential slot; no profile value survives.
    assert not any(isinstance(v, str) and "evil" in v for v in headers.values())
    present = {k.lower() for k in headers}
    assert not present & {"authorization", "host", "content-length", "transfer-encoding", "connection",
                          "proxy-authorization", "cookie", "anthropic-version"}
    assert headers["Editor-Version"] == "relay/1"


def test_profile_headers_with_crlf_bad_names_or_non_strings_are_dropped(request):
    profile = _HeaderProfile(name="crlf-fake", hostnames=("relay.example",))
    profile.headers = {
        "X-Split": "a\r\nInjected: 1", "X-Nul": "a\0b", "X-Lf": "a\nb", "Bad Name": "v", "X:Colon": "v",
        "X-None": None, "X-Int": 7, "X-Dict": {"a": 1}, 7: "int-name", "Editor-Version": "relay/1",
    }
    _register(request, profile)
    headers = _client_kwargs(RELAY)["default_headers"]
    assert headers["Editor-Version"] == "relay/1"
    for name in ("X-Split", "X-Nul", "X-Lf", "Bad Name", "X:Colon", "X-None", "X-Int", "X-Dict", 7):
        assert name not in headers


def test_failing_hook_given_a_credential_logs_no_exception_text(monkeypatch, caplog):
    from agent import provider_route_hooks
    from agent.provider_route_hooks import ask_profile

    monkeypatch.setattr(provider_route_hooks, "_WARNED", set())

    class _Leaky(ProviderProfile):
        def client_headers(self, *, base_url, api_key="", has_images=False):
            raise RuntimeError(f"cannot use {api_key}")

    with caplog.at_level(logging.WARNING, logger="agent.provider_route_hooks"):
        assert ask_profile(_Leaky(name="leaky"), "client_headers", None, base_url=RELAY, api_key="live-secret") is None
    records = [r for r in caplog.records if "client_headers hook failed" in r.getMessage()]
    assert len(records) == 1 and "RuntimeError" in records[0].getMessage()
    assert "live-secret" not in caplog.text
    assert records[0].exc_info is None


# ── profile credential refresh on anthropic_messages ──


class _RefreshProfile(ProviderProfile):
    refreshed = ("fresh-token", "https://relay.example/")

    def refresh_runtime_credentials(self, *, reason):
        return self.refreshed


class _Client:
    def __init__(self, token):
        self.token = token
        self.closed = False

    def close(self):
        self.closed = True


class _AnthropicAgent(ClientLifecycleMixin):
    """The real refresh methods on an anthropic_messages agent; the client builder is recorded."""
    api_mode = "anthropic_messages"
    provider = "relay-fake"
    model = "m"

    def __init__(self):
        self.api_key = self._anthropic_api_key = "stale-token"
        self.base_url = self._anthropic_base_url = RELAY
        self._anthropic_client = _Client("stale-token")
        self._is_anthropic_oauth = True
        self.context_compressor = SimpleNamespace(api_key="stale-token")
        self._primary_runtime = {"api_key": "stale-token", "anthropic_api_key": "stale-token",
                                 "compressor_api_key": "stale-token", "is_anthropic_oauth": True}
        self.rebuilt = []

    def _rebuild_anthropic_client(self):
        self.rebuilt.append(self._anthropic_api_key)
        self._anthropic_client = _Client(self._anthropic_api_key)

    def _replace_primary_openai_client(self, *, reason):
        raise AssertionError("anthropic_messages refresh must not rebuild the OpenAI client")


@pytest.fixture
def refresh(monkeypatch):
    rotated = []
    profile = _RefreshProfile(name="relay-fake", hostnames=("relay.example",))
    monkeypatch.setattr("agent.conversation_loop._credential_refresh_profile", lambda agent: profile)
    monkeypatch.setattr("agent.auxiliary_key_rotation.rotate_runtime_main_api_key",
                        lambda old, new: rotated.append((old, new)))
    return profile, rotated


def _state(agent):
    return (agent.api_key, agent.base_url, agent._anthropic_api_key, agent._anthropic_base_url,
            agent._anthropic_client)


def test_profile_refresh_rebuilds_the_anthropic_client_and_moves_every_holder(refresh):
    _, rotated = refresh
    agent = _AnthropicAgent()
    old_client = agent._anthropic_client
    assert agent._try_refresh_provider_runtime_credentials(reason="unauthorized") is True
    assert agent.rebuilt == ["fresh-token"]
    assert agent._anthropic_api_key == agent.api_key == "fresh-token"
    assert agent._anthropic_base_url == agent.base_url == "https://relay.example"
    assert old_client.closed and not agent._anthropic_client.closed
    assert agent._is_anthropic_oauth is False  # recomputed for the relay host
    assert agent.context_compressor.api_key == "fresh-token"
    assert agent._primary_runtime == {"api_key": "fresh-token", "anthropic_api_key": "fresh-token",
                                      "compressor_api_key": "fresh-token", "is_anthropic_oauth": False}
    assert rotated == [("stale-token", "fresh-token")]


def test_failed_anthropic_rebuild_restores_all_state(refresh, caplog):
    _, rotated = refresh
    agent = _AnthropicAgent()
    before = _state(agent)

    def _boom():
        raise RuntimeError("rebuild failed fresh-token")

    agent._rebuild_anthropic_client = _boom
    assert agent._try_refresh_provider_runtime_credentials(reason="unauthorized") is False
    assert _state(agent) == before
    assert not before[4].closed
    assert agent.context_compressor.api_key == "stale-token" and rotated == []
    assert "fresh-token" not in caplog.text  # logs the exception type only


def test_refresh_to_a_host_the_profile_does_not_claim_changes_nothing(request, refresh, caplog):
    profile, rotated = refresh
    _register(request, profile)  # the named-provider check passes: only the hostnames claim refuses
    profile.refreshed = ("fresh-token", "https://evil.example")
    agent = _AnthropicAgent()
    before = _state(agent)
    with caplog.at_level(logging.WARNING):
        assert agent._try_refresh_provider_runtime_credentials(reason="unauthorized") is False
    assert _state(agent) == before and agent.rebuilt == [] and rotated == []
    assert "may not move the session" in caplog.text


def test_refresh_to_another_host_of_the_same_profile_is_allowed(request, refresh):
    profile, _ = refresh
    _register(request, profile)
    profile.refreshed = ("fresh-token", "https://api.business.relay.example")
    agent = _AnthropicAgent()
    assert agent._try_refresh_provider_runtime_credentials(reason="unauthorized") is True
    assert agent._anthropic_base_url == "https://api.business.relay.example"


def test_host_only_refresh_owner_never_claims_a_core_owned_anthropic_route(request):
    from agent.conversation_loop import _credential_refresh_profile

    profile = _register(request, _RefreshProfile(name="claim-fake", hostnames=("anthropic.com",)))
    native = SimpleNamespace(provider="anthropic", base_url="https://api.anthropic.com", api_mode="anthropic_messages")
    assert _credential_refresh_profile(native) is None
    named = SimpleNamespace(provider="claim-fake", base_url="https://api.anthropic.com", api_mode="anthropic_messages")
    assert _credential_refresh_profile(named) is profile


def test_named_refresh_owner_matches_by_identity_so_an_alias_still_owns_it(request):
    from agent.conversation_loop import _credential_refresh_profile

    profile = _register(request, _RefreshProfile(name="claim-fake", aliases=("claim-alias",), hostnames=("anthropic.com",)))
    aliased = SimpleNamespace(provider="claim-alias", base_url="https://api.anthropic.com", api_mode="anthropic_messages")
    assert _credential_refresh_profile(aliased) is profile


# ── round-3 hardening: lookalike hosts, origins, refresh types, header values, logging ──

FULLWIDTH_ANTHROPIC = "https://api.anthropic\uff0ecom"


def test_fullwidth_dot_anthropic_url_is_never_handed_to_a_profile(request):
    """urllib keeps U+FF0E (so the host reads as foreign) but httpx's IDNA step sends it to api.anthropic.com."""
    from agent.anthropic_adapter import _route_profile_client_overrides

    profile = _register(request, _RecordingProfile(name="claim-fake", hostnames=("anthropic\uff0ecom",)))
    assert _route_profile_client_overrides(FULLWIDTH_ANTHROPIC, "live-secret") == (None, {})
    kw = _client_kwargs(FULLWIDTH_ANTHROPIC, key="live-secret")
    assert "Editor-Version" not in _str_headers(kw)
    assert profile.calls == []


@pytest.mark.parametrize("refreshed_url", [
    "http://relay.example",  # https -> http on the same host
    "https://relay.example:8443",  # port change on the same host
    "http://api.business.relay.example",  # a claimed new host over cleartext
    FULLWIDTH_ANTHROPIC,  # non-ASCII host
    "https://relay\uff0eexample",  # non-ASCII lookalike of the current host
    "https://[x]",  # urlparse raises ValueError: ignored, never raised into the error handler
    "http://3627730435",  # integer IPv4 over cleartext (the resolver maps it to a public address)
])
def test_refresh_may_not_downgrade_change_port_or_use_a_non_ascii_host(request, refresh, refreshed_url):
    profile, rotated = refresh
    _register(request, profile)
    profile.refreshed = ("fresh-token", refreshed_url)
    agent = _AnthropicAgent()
    before = _state(agent)
    assert agent._try_refresh_provider_runtime_credentials(reason="unauthorized") is False
    assert _state(agent) == before and agent.rebuilt == [] and rotated == []


def test_same_origin_refresh_with_a_path_is_allowed(refresh):
    profile, _ = refresh
    profile.refreshed = ("fresh-token", "https://relay.example:443/v1/")
    agent = _AnthropicAgent()
    assert agent._try_refresh_provider_runtime_credentials(reason="unauthorized") is True
    assert agent._anthropic_base_url == "https://relay.example:443/v1"


def test_host_matched_refresh_owner_re_mints_on_the_current_route_but_never_moves_it(request, refresh):
    """A ``custom`` route at the profile's host: the profile is not the named provider, so a host move it
    returns is dropped, while the fresh token is adopted on the route it serves (401 self-healing kept)."""
    profile, _ = refresh
    _register(request, profile)
    agent = _AnthropicAgent()
    agent.provider = "custom"
    profile.refreshed = ("fresh-token", "https://api.business.relay.example")
    assert agent._try_refresh_provider_runtime_credentials(reason="unauthorized") is True
    assert agent._anthropic_api_key == agent.api_key == "fresh-token"
    assert agent._anthropic_base_url == agent.base_url == RELAY
    profile.refreshed = ("fresher-token", None)
    assert agent._try_refresh_provider_runtime_credentials(reason="unauthorized") is True
    assert agent._anthropic_api_key == "fresher-token" and agent._anthropic_base_url == RELAY


def test_host_matched_refresh_owner_of_another_host_changes_nothing(request, refresh):
    """The token is adopted on the current route only when the refreshing profile serves that route's host."""
    profile, rotated = refresh
    _register(request, profile)
    agent = _AnthropicAgent()
    agent.provider = "custom"
    agent.base_url = agent._anthropic_base_url = OTHER
    profile.refreshed = ("fresh-token", "https://api.business.relay.example")
    before = _state(agent)
    assert agent._try_refresh_provider_runtime_credentials(reason="unauthorized") is False
    assert _state(agent) == before and rotated == []


@pytest.mark.parametrize("refreshed", [
    ("fresh-token", object()), (123, None), ("   ", None), (b"fresh-token", None), ("fresh-token", 7),
    ("fresh token", None), ("fresh-token\r\nX-Injected: 1", None), ("fresh\ttoken", None), ("caf\u00e9", None),
])
def test_refresh_with_wrong_types_changes_nothing(refresh, refreshed):
    profile, rotated = refresh
    profile.refreshed = refreshed
    agent = _AnthropicAgent()
    before = _state(agent)
    assert agent._try_refresh_provider_runtime_credentials(reason="unauthorized") is False
    assert _state(agent) == before and agent.rebuilt == [] and rotated == []
    assert agent.context_compressor.api_key == "stale-token"


def test_refresh_host_move_also_moves_the_primary_runtime_base_url(request, refresh):
    profile, _ = refresh
    _register(request, profile)
    profile.refreshed = ("fresh-token", "https://api.business.relay.example")
    agent = _AnthropicAgent()
    agent.context_compressor.base_url = RELAY
    agent._primary_runtime.update({"base_url": RELAY, "anthropic_base_url": RELAY, "compressor_base_url": RELAY})
    assert agent._try_refresh_provider_runtime_credentials(reason="unauthorized") is True
    moved = "https://api.business.relay.example"
    assert agent._primary_runtime["base_url"] == agent._primary_runtime["anthropic_base_url"] == moved
    assert agent._primary_runtime["compressor_base_url"] == agent.context_compressor.base_url == moved
    assert agent._primary_runtime["anthropic_api_key"] == "fresh-token"


def test_refresh_host_move_leaves_a_compressor_on_another_route_alone(request, refresh):
    profile, _ = refresh
    _register(request, profile)
    profile.refreshed = ("fresh-token", "https://api.business.relay.example")
    agent = _AnthropicAgent()
    agent.context_compressor.base_url = OTHER  # summarizing on its own provider
    agent._primary_runtime["compressor_base_url"] = OTHER
    assert agent._try_refresh_provider_runtime_credentials(reason="unauthorized") is True
    assert agent.context_compressor.base_url == agent._primary_runtime["compressor_base_url"] == OTHER


class _OpenAIWireAgent(_AnthropicAgent):
    api_mode = "chat_completions"

    def _replace_primary_openai_client(self, *, reason):
        raise AssertionError("a refused refresh must not rebuild the OpenAI client")

    def _sync_client_kwargs_credentials(self):
        raise AssertionError("a refused refresh must not touch the client kwargs")


def test_chat_completions_refresh_to_an_unclaimed_host_changes_nothing(request, refresh, caplog):
    profile, _ = refresh
    _register(request, profile)  # the named-provider check passes: only the hostnames claim refuses
    profile.refreshed = ("fresh-token", "https://evil.example")
    agent = _OpenAIWireAgent()
    before = (agent.api_key, agent.base_url)
    with caplog.at_level(logging.WARNING):
        assert agent._try_refresh_provider_runtime_credentials(reason="unauthorized") is False
    assert (agent.api_key, agent.base_url) == before
    assert "may not move the session" in caplog.text


def test_profile_header_values_must_be_printable_ascii_and_hop_by_hop_names_are_reserved(request):
    profile = _HeaderProfile(name="ctl-fake", hostnames=("relay.example",))
    profile.headers = {
        "X-Vt": "a\x0bb", "X-Del": "a\x7fb", "X-Utf": "caf\u00e9", "X-Emoji": "\u2603", "X-Tab": "a\tb",
        "X-Lead": " x", "X-Trail": "x\t", "X-Blank": " ", "X-Empty": "",
        "TE": "trailers", "Trailer": "x", "Upgrade": "h2c", "Keep-Alive": "1", "Proxy-Connection": "x",
        "Expect": "100-continue", "Editor-Version": "relay/1",
    }
    _register(request, profile)
    headers = _client_kwargs(RELAY)["default_headers"]
    assert headers["Editor-Version"] == "relay/1" and headers["X-Tab"] == "a\tb" and headers["X-Empty"] == ""
    present = {k.lower() for k in headers}
    for name in ("x-vt", "x-del", "x-utf", "x-emoji", "x-lead", "x-trail", "x-blank", "te", "trailer", "upgrade",
                 "keep-alive", "proxy-connection", "expect"):
        assert name not in present


def test_profile_header_replaces_a_core_header_of_the_same_name_in_another_case(request):
    profile = _HeaderProfile(name="case-fake", hostnames=("relay.example",))
    profile.headers = {"user-agent": "relay-lower/1"}
    _register(request, profile)
    # Core adds its attribution "User-Agent" on OpenCode-style relays; the profile's lower-case key replaces it.
    with patch("agent.anthropic_adapter._is_opencode_endpoint", return_value=True):
        headers = _str_headers(_client_kwargs(RELAY))
    user_agents = {k: v for k, v in headers.items() if k.lower() == "user-agent"}
    assert user_agents == {"user-agent": "relay-lower/1"}
    assert headers["X-Title"] == "Hermes Agent"  # other core headers stay


def test_override_failure_logs_only_the_exception_type(request, caplog):
    from agent.anthropic_adapter import _route_profile_client_overrides

    _register(request, _RecordingProfile(name="relay-fake", hostnames=("relay.example",)))

    def _boom(*_a, **_k):
        raise RuntimeError("cannot use live-secret")

    with patch("agent.provider_route_hooks.route_answer", _boom), \
            caplog.at_level(logging.DEBUG, logger="agent.anthropic_adapter"):
        assert _route_profile_client_overrides(RELAY, "live-secret") == (None, {})
    assert "RuntimeError" in caplog.text and "live-secret" not in caplog.text
    assert all(r.exc_info is None for r in caplog.records)


def test_model_api_mode_with_a_key_logs_only_the_exception_type(monkeypatch, caplog):
    from agent import provider_route_hooks

    monkeypatch.setattr(provider_route_hooks, "_API_MODE_WARNED", set())

    class _Leaky(ProviderProfile):
        def api_mode_for_model(self, model, *, api_key=None):
            raise RuntimeError(f"catalog fetch with {api_key} failed")

    with caplog.at_level(logging.WARNING, logger="agent.provider_route_hooks"):
        answer = provider_route_hooks.model_api_mode(_Leaky(name="leaky"), "m", api_key="live-secret")
    assert answer == "chat_completions"
    assert "RuntimeError" in caplog.text and "live-secret" not in caplog.text
    assert all(r.exc_info is None for r in caplog.records)


def test_refresh_hook_failure_logs_only_the_exception_type(monkeypatch, caplog):
    from agent import provider_route_hooks

    monkeypatch.setattr(provider_route_hooks, "_ERROR_PATH_WARNED", set())

    class _Leaky(ProviderProfile):
        def refresh_runtime_credentials(self, *, reason):
            raise RuntimeError("exchange returned minted-secret")

    with caplog.at_level(logging.WARNING, logger="agent.provider_route_hooks"):
        answer = provider_route_hooks.ask_on_error_path(_Leaky(name="leaky"), "refresh_runtime_credentials", None,
                                                        reason="unauthorized")
    assert answer is None
    assert "RuntimeError" in caplog.text and "minted-secret" not in caplog.text
    assert all(r.exc_info is None for r in caplog.records)
