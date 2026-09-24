"""Per-model capability overrides must survive to the wire on every consumer.

``anthropic_oauth_proxy`` is resolved per provider AND per model
(``runtime_provider_custom._lift_model_capabilities``), so two models on one relay may legitimately
disagree. The value chooses Bearer vs ``x-api-key``, the Claude Code transforms and the
session-affinity header, so a consumer that qualifies the decision by provider + endpoint only
would let one model's ``true`` lend wire authority to a model declared ``false`` (and lose a
model-level ``true`` behind a provider-level ``false``) right before the wire policy is chosen.

One endpoint, two models with opposing declarations, exercised in both directions.
"""

import json

import httpx
import pytest
import yaml

URL = "https://relay.example.com"
KEY = "opaque-relay-key"
# On this one relay: the provider level says yes, and TRUSTLESS_MODEL says no for itself.
TRUSTED_MODEL = "claude-sonnet-4-6"
TRUSTLESS_MODEL = "claude-haiku-4-6"


@pytest.fixture
def relay(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TEST_RELAY_KEY", KEY)
    for stale in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_TOKEN"):
        monkeypatch.delenv(stale, raising=False)
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({
            "model": {"provider": "custom:relay", "default": TRUSTED_MODEL},
            "providers": {
                "relay": {
                    "api": URL,
                    "key_env": "TEST_RELAY_KEY",
                    "transport": "anthropic_messages",
                    "capabilities": {"anthropic_oauth_proxy": True},
                    "models": {
                        TRUSTLESS_MODEL: {"anthropic_oauth_proxy": False},
                    },
                },
                # The mirror image: provider-level deny, one model opting itself in.
                "inverse": {
                    "api": URL,
                    "key_env": "TEST_RELAY_KEY",
                    "transport": "anthropic_messages",
                    "capabilities": {"anthropic_oauth_proxy": False},
                    "models": {
                        TRUSTED_MODEL: {"anthropic_oauth_proxy": True},
                    },
                },
            },
        }),
        encoding="utf-8",
    )
    requests = []

    def send(client, request, **kwargs):
        requests.append(request)
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(200, request=request, json={
                "id": "cc_test", "object": "chat.completion", "created": 0, "model": TRUSTED_MODEL,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })
        message = {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": TRUSTED_MODEL,
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        if json.loads(request.content or b"{}").get("stream"):
            events = [
                {"type": "message_start", "message": message},
                {"type": "message_stop"},
            ]
            data = "".join(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
            )
            return httpx.Response(
                200, request=request, content=data,
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, request=request, json=message)

    monkeypatch.setattr(httpx.Client, "send", send)
    return requests


def assert_oauth_wire(request, expected: bool) -> None:
    """Bearer + OAuth beta when the route's own model declares the capability, else x-api-key."""
    assert request.url.host == "relay.example.com"
    assert (request.headers.get("authorization") == f"Bearer {KEY}") is expected
    assert (request.headers.get("x-api-key") is None) is expected
    assert ("oauth-2025-04-20" in request.headers.get("anthropic-beta", "")) is expected


# ── the resolver itself ──────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "provider,model,expected",
    [
        ("custom:relay", TRUSTED_MODEL, True),      # provider-level true, model silent
        ("custom:relay", TRUSTLESS_MODEL, False),   # model-level false overrides true
        ("custom:inverse", TRUSTED_MODEL, True),    # model-level true overrides false
        ("custom:inverse", TRUSTLESS_MODEL, False),  # provider-level false, model silent
    ],
)
def test_declared_policy_is_model_qualified(relay, provider, model, expected):
    from agent.auxiliary_oauth import declared_oauth_proxy

    assert declared_oauth_proxy(provider, model) is expected


def test_main_runtime_is_not_inherited_across_models_on_one_endpoint(relay):
    """The live main runtime is authority for ITS model only.

    A main session on the trusted model must not lend its ``true`` to an auxiliary call routed to
    a model whose own declaration is ``false`` — same provider, same endpoint, different route.
    """
    from agent.auxiliary_oauth import runtime_oauth_proxy

    main = {
        "provider": "custom",
        "requested_provider": "custom:relay",
        "base_url": URL,
        "model": TRUSTED_MODEL,
        "capabilities": {"anthropic_oauth_proxy": True},
    }
    assert runtime_oauth_proxy(main, "custom:relay", URL, TRUSTED_MODEL) is True
    assert runtime_oauth_proxy(main, "custom:relay", URL, TRUSTLESS_MODEL) is False


def test_a_model_level_true_is_not_lost_behind_a_main_runtime_false(relay):
    """The inverse: a ``false`` main session must not mask a model that declares itself ``true``."""
    from agent.auxiliary_oauth import runtime_oauth_proxy

    main = {
        "provider": "custom",
        "requested_provider": "custom:inverse",
        "base_url": URL,
        "model": TRUSTLESS_MODEL,
        "capabilities": {"anthropic_oauth_proxy": False},
    }
    assert runtime_oauth_proxy(main, "custom:inverse", URL, TRUSTLESS_MODEL) is False
    assert runtime_oauth_proxy(main, "custom:inverse", URL, TRUSTED_MODEL) is True


# ── the same decision through the cached auxiliary client ────────────────────

@pytest.mark.parametrize(
    "provider,main_model,aux_model,expected",
    [
        # true → false: the main session's authority must not reach the denying model.
        ("relay", TRUSTED_MODEL, TRUSTLESS_MODEL, False),
        ("relay", TRUSTED_MODEL, TRUSTED_MODEL, True),
        # false → true: the denying main session must not mask the model's own opt-in.
        ("inverse", TRUSTLESS_MODEL, TRUSTED_MODEL, True),
        ("inverse", TRUSTLESS_MODEL, TRUSTLESS_MODEL, False),
    ],
)
def test_cached_auxiliary_client_carries_the_target_models_policy(
    relay, provider, main_model, aux_model, expected
):
    from agent.auxiliary_client import _get_cached_client

    main = {
        "provider": "custom",
        "requested_provider": f"custom:{provider}",
        "base_url": URL,
        "api_mode": "anthropic_messages",
        "model": main_model,
        "capabilities": {"anthropic_oauth_proxy": provider == "relay"},
    }
    client, model = _get_cached_client(f"custom:{provider}", aux_model, main_runtime=main)
    assert client is not None
    client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "hello"}], max_tokens=32,
    )
    assert_oauth_wire(relay[-1], expected)


def test_cached_clients_do_not_leak_policy_between_two_models(relay):
    """Both models in one process, alternating: the cache must key the decision per model."""
    from agent.auxiliary_client import _get_cached_client

    main = {
        "provider": "custom",
        "requested_provider": "custom:relay",
        "base_url": URL,
        "api_mode": "anthropic_messages",
        "model": TRUSTED_MODEL,
        "capabilities": {"anthropic_oauth_proxy": True},
    }
    for aux_model, expected in (
        (TRUSTED_MODEL, True), (TRUSTLESS_MODEL, False),
        (TRUSTED_MODEL, True), (TRUSTLESS_MODEL, False),
    ):
        client, model = _get_cached_client("custom:relay", aux_model, main_runtime=main)
        client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "hello"}], max_tokens=32,
        )
        assert_oauth_wire(relay[-1], expected)


def test_session_affinity_header_follows_the_model_not_the_endpoint(relay):
    """``x-claude-code-session-id`` rides only on routes whose own model declares the capability."""
    from agent import auxiliary_client as aux
    from agent.claude_code_session import CLAUDE_CODE_SESSION_HEADER

    main = {
        "provider": "custom",
        "requested_provider": "custom:relay",
        "base_url": URL,
        "api_mode": "anthropic_messages",
        "model": TRUSTED_MODEL,
        "session_id": "20260922_120000_relay",
        "capabilities": {"anthropic_oauth_proxy": True},
    }

    def header_for(model):
        with aux.scoped_runtime_main(main):
            kwargs = aux._build_call_kwargs(
                "custom:relay", model, [{"role": "user", "content": "hi"}], base_url=URL,
            )
        return (kwargs.get("extra_headers") or {}).get(CLAUDE_CODE_SESSION_HEADER)

    # The main model's own route declares the capability; the sibling model declares it off, so no
    # header — the relay must not be told this call belongs to an OAuth-pinned conversation.
    assert header_for(TRUSTED_MODEL)
    assert header_for(TRUSTLESS_MODEL) is None


# ── a slot that arrives with its own resolved endpoint (MoA) ─────────────────

@pytest.mark.parametrize(
    "provider,model,expected",
    [
        ("relay", TRUSTED_MODEL, True),
        ("relay", TRUSTLESS_MODEL, False),
        ("custom:relay", TRUSTED_MODEL, True),
        ("inverse", TRUSTED_MODEL, True),
        ("inverse", TRUSTLESS_MODEL, False),
    ],
)
def test_moa_slot_with_resolved_endpoint_keeps_its_named_providers_policy(relay, provider, model, expected):
    """A MoA slot is sent with the base_url/api_key/api_mode its provider resolved to.

    The explicit endpoint must not flatten the named provider into anonymous ``custom``: the
    policy is looked up by provider name, so the flattened call went out without the OAuth wire
    and a relay answered 429 on every reference and aggregator call while the main session on
    the same relay and model kept working.
    """
    from agent.auxiliary_client import call_llm
    from agent.moa_loop import _slot_runtime

    runtime = _slot_runtime({"provider": provider, "model": model})
    assert runtime.get("base_url") == URL
    call_llm(
        task="moa_aggregator", messages=[{"role": "user", "content": "hello"}], max_tokens=32,
        main_runtime={"provider": "moa", "base_url": "moa://local", "model": "simple"}, **runtime,
    )
    assert_oauth_wire(relay[-1], expected)


def _rewrite_config(_replace=False, **sections):
    """Merge (or with ``_replace`` overwrite) *sections* in the fixture's config.yaml."""
    import os
    from pathlib import Path

    path = Path(os.environ["HERMES_HOME"]) / "config.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key, value in sections.items():
        if not _replace and isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = value
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def assert_no_oauth_identity(request) -> None:
    """A request that carries none of the relay's declared OAuth identity: not its key as a
    Bearer, no OAuth / Claude Code betas, no conversation header."""
    assert request.headers.get("authorization") != f"Bearer {KEY}"
    # NOT asserted: ``x-api-key`` — a name + foreign URL still sends the entry's key as x-api-key.
    # That predates this work (reviews 2 and 3, finding 6) and is a separate decision.
    assert "oauth-2025-04-20" not in request.headers.get("anthropic-beta", "")
    assert "claude-code" not in request.headers.get("anthropic-beta", "")
    assert "x-claude-code-session-id" not in request.headers


@pytest.mark.parametrize(
    "explicit_base,kept",
    [
        (URL, "relay"),
        (f"{URL}/", "relay"),
        # A resolver may hand the endpoint back with /v1 added (OpenCode-family routing): same origin.
        (f"{URL}/v1", "relay"),
        ("https://relay.example.com:443", "relay"),
        # Same host, another origin: a different trust boundary, a different route.
        ("https://relay.example.com:8443", "custom"),
        ("http://relay.example.com", "custom"),
        ("https://elsewhere.example.com", "custom"),
    ],
)
def test_explicit_endpoint_keeps_the_name_only_at_the_providers_own_origin(relay, explicit_base, kept):
    from agent.auxiliary_client import _resolve_task_provider_model

    assert _resolve_task_provider_model(None, "relay", TRUSTED_MODEL, explicit_base, KEY)[0] == kept


def test_kept_identity_is_the_canonical_name(relay):
    """One provider, one identity: spelling must not split the client cache or the logs."""
    from agent.auxiliary_client import _resolve_task_provider_model

    for spelling in ("ReLaY", " relay "):
        assert _resolve_task_provider_model(None, spelling, TRUSTED_MODEL, URL, KEY)[0] == "relay"
    # ``custom:`` is the explicit escape from a built-in of the same name (``custom:anthropic``):
    # it is kept verbatim, never stripped back into the built-in's namespace.
    assert _resolve_task_provider_model(None, "custom:relay", TRUSTED_MODEL, URL, KEY)[0] == "custom:relay"
    # A display name with a space reaches the same entry the lookup normalizes it to.
    _rewrite_config(providers={"my-relay": {"name": "My Relay", "api": URL, "transport": "anthropic_messages"}})
    for spelling in ("My Relay", "my-relay"):
        assert _resolve_task_provider_model(None, spelling, TRUSTED_MODEL, URL, KEY)[0] == "my-relay"


def test_ownership_check_reads_no_credential(relay, monkeypatch):
    """Route ownership runs on every aux call; it must not resolve the provider's secret."""
    from hermes_cli import runtime_provider_custom
    from hermes_cli.route_identity import named_provider_owns_endpoint

    def refuse(*_args, **_kwargs):
        raise AssertionError("ownership check read a credential")

    monkeypatch.setattr(runtime_provider_custom, "get_secret_str", refuse)
    assert named_provider_owns_endpoint("relay", URL) is True
    assert named_provider_owns_endpoint("relay", "https://elsewhere.example.com") is False


def test_a_builtin_name_in_providers_does_not_take_over_the_builtin(relay):
    """``providers.anthropic`` pointing at the relay must not make the built-in ``anthropic`` a
    named provider: the canonical name keeps its catalog branch, the entry stays unreachable by it."""
    from hermes_cli.route_identity import named_provider_owns_endpoint
    from hermes_cli.runtime_provider_custom import _get_named_custom_provider

    _rewrite_config(providers={"anthropic": {"api": URL, "transport": "anthropic_messages"}})
    assert _get_named_custom_provider("anthropic") is None
    assert named_provider_owns_endpoint("anthropic", URL) is False
    # An entry merely matching an alias (``kimi`` → ``kimi-coding``) is still the user's target.
    _rewrite_config(providers={"kimi": {"api": URL, "transport": "anthropic_messages"}})
    assert named_provider_owns_endpoint("kimi", URL) is True


def test_a_malformed_entry_is_not_an_owner_and_does_not_raise(relay):
    """A broken sibling entry must not turn every aux call with a base_url into an exception."""
    from agent.auxiliary_client import _resolve_task_provider_model

    _rewrite_config(providers={"broken": {"api": 12345, "transport": "anthropic_messages"}})
    assert _resolve_task_provider_model(None, "broken", TRUSTED_MODEL, URL, KEY)[0] == "custom"
    assert _resolve_task_provider_model(None, "relay", TRUSTED_MODEL, URL, KEY)[0] == "relay"


@pytest.mark.parametrize("with_key", [True, False])
def test_auxiliary_task_block_on_the_providers_own_endpoint_keeps_its_policy(relay, with_key):
    """``auxiliary.<task>: {provider, base_url[, api_key]}`` on the relay's own URL is the relay.

    With a key in the block the resolver used to flatten the name to ``custom`` (only local-server
    aliases survived), so the task went out without the OAuth wire the relay declares.
    """
    from agent.auxiliary_client import _resolve_task_provider_model, call_llm

    block = {"provider": "relay", "model": TRUSTED_MODEL, "base_url": URL}
    if with_key:
        # A key distinct from the entry's: the block's own key must be the one that goes out.
        block["api_key"] = "block-own-key"
    _rewrite_config(auxiliary={"title_generation": block})
    assert _resolve_task_provider_model("title_generation")[0] == "relay"
    call_llm(
        task="title_generation", messages=[{"role": "user", "content": "hello"}], max_tokens=32,
        main_runtime={"provider": "moa", "base_url": "moa://local", "model": "simple"},
    )
    request = relay[-1]
    assert request.headers.get("authorization") == f"Bearer {'block-own-key' if with_key else KEY}"
    assert "oauth-2025-04-20" in request.headers.get("anthropic-beta", "")


# ── the relay's declaration never travels to another origin ──────────────────

FOREIGN = "https://elsewhere.example.com"


def test_explicit_foreign_endpoint_under_the_relays_name_gets_no_oauth_identity(relay):
    from agent.auxiliary_client import _client_cache, resolve_provider_client

    for name in ("relay", "custom:relay"):
        _client_cache.clear()
        client, model = resolve_provider_client(name, TRUSTED_MODEL, explicit_base_url=FOREIGN)
        client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "hello"}], max_tokens=32,
        )
        assert relay[-1].url.host == "elsewhere.example.com"
        assert_no_oauth_identity(relay[-1])


def test_fallback_chain_entry_at_a_foreign_endpoint_gets_no_oauth_identity(relay):
    """``fallback_chain`` bypasses the task resolver and goes straight to the client builder."""
    from agent.auxiliary_client import _resolve_fallback_entry

    client, model = _resolve_fallback_entry(
        {"provider": "relay", "model": TRUSTED_MODEL, "base_url": FOREIGN},
    )
    client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "hello"}], max_tokens=32,
    )
    assert relay[-1].url.host == "elsewhere.example.com"
    assert_no_oauth_identity(relay[-1])


def test_auxiliary_task_block_at_a_foreign_endpoint_gets_no_oauth_identity(relay):
    """The keyless block keeps the provider name (it resolves its key from the entry), so the
    decision must be made where the wire policy is chosen, not only in the task resolver."""
    from agent.auxiliary_client import call_llm

    _rewrite_config(auxiliary={"title_generation": {
        "provider": "relay", "model": TRUSTED_MODEL, "base_url": FOREIGN,
    }})
    call_llm(
        task="title_generation", messages=[{"role": "user", "content": "hello"}], max_tokens=32,
        main_runtime={"provider": "moa", "base_url": "moa://local", "model": "simple"},
    )
    assert relay[-1].url.host == "elsewhere.example.com"
    assert_no_oauth_identity(relay[-1])


def test_an_unknown_model_keeps_the_provider_level_policy(relay):
    """A caller that resolves no per-model entry still gets the provider's declared map."""
    from agent.auxiliary_oauth import declared_oauth_proxy

    assert declared_oauth_proxy("custom:relay", "some-unlisted-model") is True
    assert declared_oauth_proxy("custom:inverse", "some-unlisted-model") is False
    assert declared_oauth_proxy("custom:absent", TRUSTED_MODEL) is None


def test_vendor_prefixed_model_ids_compare_by_bare_name(relay):
    """``anthropic/claude-haiku-4-6`` and ``claude-haiku-4-6`` are one route, not two."""
    from agent.auxiliary_oauth import runtime_oauth_proxy

    main = {
        "provider": "custom",
        "requested_provider": "custom:relay",
        "base_url": URL,
        "model": TRUSTED_MODEL,
        "capabilities": {"anthropic_oauth_proxy": True},
    }
    assert runtime_oauth_proxy(main, "custom:relay", URL, f"anthropic/{TRUSTLESS_MODEL}") is False
    assert runtime_oauth_proxy(main, "custom:relay", URL, f"anthropic/{TRUSTED_MODEL}") is True


# ── one host, many tenants: the path is part of the endpoint ─────────────────

GATEWAY = "https://gw.example.com"


def _tenant_config():
    _rewrite_config(providers={"tenant": {
        "api": f"{GATEWAY}/tenant-a", "key_env": "TEST_RELAY_KEY", "transport": "anthropic_messages",
        "extra_headers": {"X-Tenant-Token": "tenant-a-secret"},
        "capabilities": {"anthropic_oauth_proxy": True},
    }})


@pytest.mark.parametrize(
    "own,target,same",
    [
        ("https://h/a", "https://h/a", True),
        ("https://h/a", "https://h/a/", True),
        ("https://h/a", "https://h/a/v1", True),
        ("https://h/a/v1", "https://h/a", True),
        ("https://h", "https://h/v1", True),
        ("https://h", "https://h:443", True),
        ("https://H", "https://h", True),
        ("https://h/a", "https://h/b", False),
        ("https://h/a", "https://h/a/b", False),
        ("https://h/a", "https://h", False),
        ("https://h", "https://h/a", False),
        ("https://h/a", "https://h/A", False),
        ("https://h", "http://h", False),
        ("https://h", "https://h:8443", False),
        ("https://h", "", False),
        ("", "https://h", False),
        ("https://h", "https://h:notaport", False),
        # The query is part of the route: it can select a tenant. Adding, dropping or changing it
        # is another route — a query-less URL is not a wildcard in either direction.
        ("https://h/t?team=a", "https://h/t", False),
        ("https://h/t", "https://h/t?team=other", False),
        ("https://h/t?team=a", "https://h/t?team=a", True),
        ("https://h/t?team=a", "https://h/t?team=b", False),
        # Key order is not identity; values, blanks and repeats are.
        ("https://h/t?a=1&b=2", "https://h/t?b=2&a=1", True),
        ("https://h/t?team=a&x=", "https://h/t?team=a", False),
        ("https://h/t?team=a&team=b", "https://h/t?team=a", False),
        ("https://h/t?team=a&team=b", "https://h/t?team=b&team=a", False),
        ("https://h/t?team=a", "https://h/t?team=", False),
        ("https://h/t?team=", "https://h/t?team=", True),
    ],
)
def test_same_provider_endpoint_is_origin_plus_path_modulo_v1(own, target, same):
    from hermes_cli.route_identity import same_provider_endpoint

    assert same_provider_endpoint(own, target) is same


def test_a_sibling_tenant_path_is_not_the_relay(relay):
    """Finding 1 (review 2): same origin, another path — another tenant behind one gateway."""
    from agent.auxiliary_client import _resolve_task_provider_model, call_llm

    _tenant_config()
    assert _resolve_task_provider_model(None, "tenant", TRUSTED_MODEL, f"{GATEWAY}/tenant-a", KEY)[0] == "tenant"
    assert _resolve_task_provider_model(None, "tenant", TRUSTED_MODEL, f"{GATEWAY}/tenant-a/v1", KEY)[0] == "tenant"
    assert _resolve_task_provider_model(None, "tenant", TRUSTED_MODEL, f"{GATEWAY}/tenant-b", KEY)[0] == "custom"
    # The MoA-slot / explicit-caller shape the review reproduced: name + another tenant's URL.
    call_llm(
        provider="tenant", model=TRUSTED_MODEL, base_url=f"{GATEWAY}/tenant-b",
        messages=[{"role": "user", "content": "hello"}], max_tokens=32,
        main_runtime={"provider": "moa", "base_url": "moa://local", "model": "simple"},
    )
    request = relay[-1]
    assert request.url.path.startswith("/tenant-b")
    assert request.headers.get("x-tenant-token") is None
    assert request.headers.get("authorization") != f"Bearer {KEY}"
    assert "x-claude-code-session-id" not in request.headers
    assert_no_oauth_identity(request)


# ── the query is part of the route: a tenant choice, never a wildcard ─────────

def _query_tenant_config():
    """Two relays on one path: one pinned to a tenant by query, one declared without a query."""
    _rewrite_config(providers={
        "qtenant": {"api": f"{URL}/t?team=a", "key_env": "TEST_RELAY_KEY", "transport": "anthropic_messages",
                    "capabilities": {"anthropic_oauth_proxy": True}},
        "qplain": {"api": f"{URL}/t", "key_env": "TEST_RELAY_KEY", "transport": "anthropic_messages",
                   "capabilities": {"anthropic_oauth_proxy": True}},
    })


# (provider, explicit URL, keeps the declaration): the entry's own query — or its own absence of
# one — keeps it; adding, dropping or changing the query is another tenant.
QUERY_ROUTES = [
    ("qtenant", f"{URL}/t?team=a", True),
    ("qtenant", f"{URL}/t", False),
    ("qtenant", f"{URL}/t?team=b", False),
    ("qtenant", f"{URL}/t?team=a&team=b", False),
    ("qplain", f"{URL}/t", True),
    ("qplain", f"{URL}/t?team=other", False),
]


@pytest.mark.parametrize("provider,explicit,kept", QUERY_ROUTES)
def test_explicit_base_url_with_another_query_drops_the_declaration(relay, provider, explicit, kept):
    """andrexibiza review, P1: ``--base-url`` / ``/model`` / CLI fallback under the relay's name."""
    from hermes_cli.runtime_provider import resolve_runtime_provider

    _query_tenant_config()
    runtime = resolve_runtime_provider(requested=provider, target_model=TRUSTED_MODEL, explicit_base_url=explicit)
    assert bool((runtime.get("capabilities") or {}).get("anthropic_oauth_proxy")) is kept


@pytest.mark.parametrize("provider,explicit,kept", QUERY_ROUTES)
def test_delegation_base_url_with_another_query_drops_the_declaration(relay, provider, explicit, kept):
    from tools.delegate_tool_config import _resolve_delegation_credentials

    _query_tenant_config()
    creds = _resolve_delegation_credentials({"provider": provider, "base_url": explicit, "model": TRUSTED_MODEL}, None)
    assert bool((creds["capabilities"] or {}).get("anthropic_oauth_proxy")) is kept


@pytest.mark.parametrize("provider,explicit,kept", QUERY_ROUTES)
def test_auxiliary_call_with_another_query_gets_no_oauth_identity(relay, provider, explicit, kept):
    """The MoA-slot / explicit auxiliary shape: name + URL, main session elsewhere."""
    from agent.auxiliary_client import call_llm

    _query_tenant_config()
    call_llm(
        provider=provider, model=TRUSTED_MODEL, base_url=explicit,
        messages=[{"role": "user", "content": "hello"}], max_tokens=32,
        main_runtime={"provider": "moa", "base_url": "moa://local", "model": "simple"},
    )
    request = relay[-1]
    assert request.url.path.startswith("/t")
    if kept:
        assert_oauth_wire(request, True)
    else:
        assert_no_oauth_identity(request)


def test_client_route_url_puts_the_sdk_split_query_back(relay):
    """The OpenAI SDK keeps ``…/t?team=a`` as a clean base_url plus a default query; the identity
    decision must see the query again, and must not see one that was never there."""
    from openai import OpenAI

    from agent.auxiliary_client import _client_route_url, _extract_url_query_params

    clean, query = _extract_url_query_params(f"{URL}/t?team=a")
    client = OpenAI(api_key="k", base_url=clean, default_query=query)
    assert _client_route_url(client, client.base_url) == f"{URL}/t?team=a"
    bare = OpenAI(api_key="k", base_url=f"{URL}/t")
    assert _client_route_url(bare, bare.base_url) == f"{URL}/t"


def _chat_query_relay():
    _rewrite_config(providers={"qchat": {
        "api": f"{URL}/t?team=a", "key_env": "TEST_RELAY_KEY", "transport": "chat_completions",
        "capabilities": {"anthropic_oauth_proxy": True},
    }})


def test_chat_wire_call_on_a_query_bearing_relay_keeps_its_conversation_header(relay):
    """End to end through ``call_llm``: the OpenAI client built for ``…/t?team=a`` reports the clean
    ``…/t``; the relay's own route must still be recognised, so the conversation header goes out."""
    from agent.auxiliary_client import call_llm
    from agent.claude_code_session import CLAUDE_CODE_SESSION_HEADER

    _chat_query_relay()
    call_llm(
        provider="qchat", model=TRUSTED_MODEL, messages=[{"role": "user", "content": "hello"}], max_tokens=32,
        main_runtime={"provider": "moa", "base_url": "moa://local", "model": "simple",
                      "session_id": "20260924_120000_query"},
    )
    request = relay[-1]
    assert request.url.path.endswith("/chat/completions")
    assert request.url.params.get("team") == "a"
    assert request.headers.get(CLAUDE_CODE_SESSION_HEADER)


def test_fallback_destination_keeps_the_clients_query(relay):
    """A fallback built from a label or a chain entry without ``base_url`` reads the client; its
    route must carry the SDK-split query, or the relay's own fallback loses its policy."""
    from openai import OpenAI

    from agent.auxiliary_client import _fallback_destination, _fallback_destination_from_entry

    client = OpenAI(api_key="k", base_url=f"{URL}/t", default_query={"team": "a"})
    from_entry = _fallback_destination_from_entry({"provider": "qchat"}, client, TRUSTED_MODEL)
    assert from_entry.route_url == f"{URL}/t?team=a"
    from_label = _fallback_destination(None, client, TRUSTED_MODEL, "qchat")
    assert from_label.route_url == f"{URL}/t?team=a"


def test_published_main_runtime_carries_the_client_query(relay):
    """The main agent's ``base_url`` is the SDK-clean half; the runtime published to auxiliary
    routing must carry the tenant query from ``_client_kwargs['default_query']``."""
    from types import SimpleNamespace

    from agent.auxiliary_oauth import runtime_oauth_proxy
    from agent.turn_context import live_route_base_url

    _query_tenant_config()
    agent = SimpleNamespace(base_url=f"{URL}/t", _client_kwargs={"default_query": {"team": "a"}})
    assert live_route_base_url(agent) == f"{URL}/t?team=a"
    assert live_route_base_url(SimpleNamespace(base_url=f"{URL}/t", _client_kwargs={})) == f"{URL}/t"
    main = {"provider": "custom", "requested_provider": "qtenant", "base_url": live_route_base_url(agent),
            "model": TRUSTED_MODEL, "capabilities": {"anthropic_oauth_proxy": True}}
    assert runtime_oauth_proxy(main, "custom", f"{URL}/t?team=a", TRUSTED_MODEL) is True
    assert not runtime_oauth_proxy(main, "custom", f"{URL}/t", TRUSTED_MODEL)


# ── the main runtime and delegation carry the declaration only to its own endpoint ──

def test_runtime_resolution_drops_the_declaration_at_another_endpoint(relay):
    """Finding 2 (review 2): ``--base-url``, a stored ``/model`` URL and CLI fallback all resolve
    through ``resolve_runtime_provider(explicit_base_url=…)``; the main agent and delegation read
    ``capabilities`` off that runtime without any further gate."""
    from hermes_cli.runtime_provider import resolve_runtime_provider

    _tenant_config()
    own = resolve_runtime_provider(requested="tenant", target_model=TRUSTED_MODEL)
    assert own["capabilities"] == {"anthropic_oauth_proxy": True}
    same = resolve_runtime_provider(requested="tenant", target_model=TRUSTED_MODEL,
                                    explicit_base_url=f"{GATEWAY}/tenant-a/v1")
    assert same["capabilities"] == {"anthropic_oauth_proxy": True}
    for foreign in (f"{GATEWAY}/tenant-b", "https://elsewhere.example.com"):
        runtime = resolve_runtime_provider(requested="tenant", target_model=TRUSTED_MODEL, explicit_base_url=foreign)
        assert runtime["base_url"] == foreign
        assert not runtime.get("capabilities")


def test_inherited_branch_cannot_carry_the_declaration_to_another_endpoint(relay):
    """Finding 2 (review 2): ``_inherited_oauth_proxy`` runs before the declaration lookup, so a
    main runtime pointed elsewhere must not already hold the relay's capability."""
    from agent.auxiliary_oauth import runtime_oauth_proxy
    from hermes_cli.runtime_provider import resolve_runtime_provider

    main = resolve_runtime_provider(requested="relay", target_model=TRUSTED_MODEL, explicit_base_url=FOREIGN)
    for provider in ("custom", "auto", "main", "relay"):
        assert runtime_oauth_proxy(main, provider, FOREIGN, TRUSTED_MODEL) is None


def test_delegation_base_url_under_the_relays_name_gets_no_declaration(relay):
    """Finding 2 (review 2): ``delegation: {provider: relay, base_url: <elsewhere>}``."""
    from tools.delegate_tool_config import _resolve_delegation_credentials

    own = _resolve_delegation_credentials({"provider": "relay", "base_url": URL, "model": TRUSTED_MODEL}, None)
    assert own["capabilities"] == {"anthropic_oauth_proxy": True}
    foreign = _resolve_delegation_credentials({"provider": "relay", "base_url": FOREIGN, "model": TRUSTED_MODEL}, None)
    assert not foreign["capabilities"]


@pytest.mark.parametrize(
    "parent_url,child_model,expected",
    [
        # Unpinned: the child is the parent's exact route and inherits the parent's live map.
        (URL, None, {"anthropic_oauth_proxy": True}),
        # Model-only pin at the relay's own endpoint: the child model's OWN declaration.
        (URL, TRUSTED_MODEL, {"anthropic_oauth_proxy": True}),
        (URL, TRUSTLESS_MODEL, {"anthropic_oauth_proxy": False}),
        # Parent already on another endpoint under the relay's name: nothing to look up.
        (FOREIGN, TRUSTED_MODEL, {}),
    ],
)
def test_model_only_child_pin_takes_its_models_declaration_at_the_parents_endpoint(
    relay, parent_url, child_model, expected,
):
    """Finding 3 (review 3): production calls ``_child_route_capabilities`` with
    ``effective_provider=parent.provider``, which is ``custom`` for every named entry — built here
    through the real ``_resolve_child_runtime`` from a real resolved runtime."""
    from types import SimpleNamespace

    from hermes_cli.runtime_provider import resolve_runtime_provider
    from tools.delegate_tool_config import _resolve_child_runtime

    runtime = resolve_runtime_provider(requested="relay", target_model="claude-opus-5")
    assert runtime["provider"] == "custom"
    parent = SimpleNamespace(
        model="claude-opus-5", base_url=parent_url, api_key="k", provider=runtime["provider"],
        requested_provider=runtime.get("requested_provider"),
        capabilities=dict(runtime.get("capabilities") or {}) if parent_url == URL else {},
        api_mode="anthropic_messages", _client_kwargs={"base_url": parent_url, "api_key": "k"}, client=None,
        acp_command=None, acp_args=[], reasoning_config=None, _fallback_chain=None,
    )
    kwargs = _resolve_child_runtime(
        parent, {}, "k", model=child_model, override_provider=None, override_base_url=None,
        override_api_key=None, override_api_mode=None, override_acp_command=None, override_acp_args=None,
    )
    assert (kwargs["capabilities"] or {}) == expected


def _child_capabilities(parent_url, live_url, child_model, requested="relay", default_query=None):
    """Capabilities of a model-only child, through the real ``_resolve_child_runtime``."""
    return _child_runtime(parent_url, live_url, child_model, requested, default_query)["capabilities"] or {}


def _child_runtime(parent_url, live_url, child_model, requested="relay", default_query=None):
    from types import SimpleNamespace

    from tools.delegate_tool_config import _resolve_child_runtime

    parent = SimpleNamespace(
        model="claude-opus-5", base_url=parent_url, api_key="k", provider="custom",
        requested_provider=requested, capabilities={"anthropic_oauth_proxy": True},
        api_mode="chat_completions", client=None,
        _client_kwargs={"base_url": live_url, "api_key": "k",
                        **({"default_query": default_query} if default_query else {})},
        acp_command=None, acp_args=[], reasoning_config=None, _fallback_chain=None,
    )
    kwargs = _resolve_child_runtime(
        parent, {}, "k", model=child_model, override_provider=None, override_base_url=None,
        override_api_key=None, override_api_mode=None, override_acp_command=None, override_acp_args=None,
    )
    return kwargs


def test_child_pin_on_a_query_bearing_entry_keeps_its_declaration(relay):
    """Finding 1 (review 4): the OpenAI-wire client moves ``?team=a`` into ``default_query``, so the
    live URL has no query while the entry's URL does. Same server — and the child must call the
    same tenant (review 5, finding 2): the parent's ``default_query`` goes back into its URL."""
    _rewrite_config(providers={"relay": {
        "api": f"{URL}/t?team=a", "key_env": "TEST_RELAY_KEY", "transport": "chat_completions",
        "capabilities": {"anthropic_oauth_proxy": True},
    }})
    child = _child_runtime(f"{URL}/t?team=a", f"{URL}/t", TRUSTLESS_MODEL, default_query={"team": "a"})
    assert child["base_url"] == f"{URL}/t?team=a"
    assert child["capabilities"] == {"anthropic_oauth_proxy": True}
    # A live URL that already carries its query (pool rotation onto a query-bearing entry) is not
    # given the query a second time.
    child = _child_runtime(f"{URL}/t?team=a", f"{URL}/t?team=a", TRUSTLESS_MODEL, default_query={"team": "a"})
    assert child["base_url"] == f"{URL}/t?team=a"


def test_child_pin_off_the_live_client_keeps_its_query(relay):
    """No ``_client_kwargs`` URL: the child reads the live OpenAI client, whose ``base_url`` is the
    SDK-clean half — its default query must come back with it, or the child loses the tenant."""
    from types import SimpleNamespace

    from openai import OpenAI

    from tools.delegate_tool_config import _resolve_child_runtime

    _rewrite_config(providers={"relay": {
        "api": f"{URL}/t?team=a", "key_env": "TEST_RELAY_KEY", "transport": "chat_completions",
        "capabilities": {"anthropic_oauth_proxy": True},
    }})
    parent = SimpleNamespace(
        model="claude-opus-5", base_url=f"{URL}/t", api_key="k", provider="custom",
        requested_provider="relay", capabilities={"anthropic_oauth_proxy": True}, api_mode="chat_completions",
        client=OpenAI(api_key="k", base_url=f"{URL}/t", default_query={"team": "a"}), _client_kwargs={},
        acp_command=None, acp_args=[], reasoning_config=None, _fallback_chain=None,
    )
    child = _resolve_child_runtime(
        parent, {}, "k", model=TRUSTLESS_MODEL, override_provider=None, override_base_url=None,
        override_api_key=None, override_api_mode=None, override_acp_command=None, override_acp_args=None,
    )
    assert child["base_url"] == f"{URL}/t?team=a"
    assert child["capabilities"] == {"anthropic_oauth_proxy": True}


def test_child_pin_takes_the_live_endpoint_not_the_lagging_surface(relay):
    """Finding 2 (review 4): the child calls the parent's LIVE endpoint (#90009). A surface
    ``base_url`` still on the relay must not lend its declaration to where the child really goes."""
    assert _child_capabilities(URL, URL, TRUSTED_MODEL) == {"anthropic_oauth_proxy": True}
    assert _child_capabilities(URL, FOREIGN, TRUSTED_MODEL) == {}


def test_an_entry_named_custom_does_not_take_over_another_entrys_child_pin(relay):
    """Review 5, finding 1: the runtime provider is ``custom`` for every named entry. When an entry
    literally named ``custom`` exists too, the pin must still read the entry the parent resolved
    from — in both directions."""
    for relay_trust, custom_trust in ((False, True), (True, False)):
        _rewrite_config(providers={
            "relay": {"api": URL, "key_env": "TEST_RELAY_KEY", "transport": "anthropic_messages",
                      "capabilities": {"anthropic_oauth_proxy": relay_trust}},
            "custom": {"api": f"{URL}/v1", "key_env": "TEST_RELAY_KEY", "transport": "anthropic_messages",
                       "capabilities": {"anthropic_oauth_proxy": custom_trust}},
        })
        caps = _child_capabilities(URL, URL, TRUSTED_MODEL)
        assert bool(caps.get("anthropic_oauth_proxy")) is relay_trust, (relay_trust, caps)


def test_child_pin_on_an_entry_literally_named_custom(relay):
    """Finding 3 (review 4): ``_shadowed_by_builtin`` lets a user name an entry ``custom``."""
    # Without such an entry, ``custom`` (the runtime provider) and an unknown requested name name
    # nothing: default-deny, not a lookup under some other provider.
    assert _child_capabilities(URL, URL, TRUSTED_MODEL, requested="nonexistent") == {}
    _rewrite_config(providers={"custom": {
        "api": URL, "key_env": "TEST_RELAY_KEY", "transport": "anthropic_messages",
        "capabilities": {"anthropic_oauth_proxy": True},
    }})
    assert _child_capabilities(URL, URL, TRUSTED_MODEL, requested="custom") == {"anthropic_oauth_proxy": True}


def test_pooled_runtime_drops_the_declaration_at_another_endpoint(relay, monkeypatch):
    """Finding 2 (review 3): the credential-pool exit of ``_resolve_named_custom_runtime`` returns
    before the final scoping; a pool keyed by the provider's name serves any explicit URL."""
    from types import SimpleNamespace

    from hermes_cli import runtime_provider

    _tenant_config()
    entry = SimpleNamespace(access_token="POOL-KEY", runtime_api_key="POOL-KEY", base_url=f"{GATEWAY}/tenant-a")
    pool = SimpleNamespace(has_credentials=lambda: True, select=lambda: entry)
    monkeypatch.setattr(runtime_provider, "load_pool", lambda key: pool)
    own = runtime_provider.resolve_runtime_provider(requested="tenant", target_model=TRUSTED_MODEL)
    assert own["source"] != "custom_provider:tenant"  # the pool path, not the key_env path
    assert own["capabilities"] == {"anthropic_oauth_proxy": True}
    foreign = runtime_provider.resolve_runtime_provider(
        requested="tenant", target_model=TRUSTED_MODEL, explicit_base_url=f"{GATEWAY}/tenant-b")
    assert foreign["source"] == own["source"]
    assert not foreign.get("capabilities")


def test_a_spaced_name_whose_dashed_form_is_a_builtin_alias_is_not_dashed(relay):
    """Finding 1 (review 3): ``Claude Code`` dashed is ``claude-code``, the ``anthropic`` alias;
    the downstream resolver would route it to the built-in and drop the relay's wire policy."""
    from agent.auxiliary_client import _resolve_task_provider_model, call_llm
    from hermes_cli.auth import known_provider_id

    assert known_provider_id("claude-code") is not None
    _rewrite_config(providers={"my-relay": {
        "name": "Claude Code", "api": URL, "key_env": "TEST_RELAY_KEY", "transport": "anthropic_messages",
        "capabilities": {"anthropic_oauth_proxy": True},
    }})
    assert _resolve_task_provider_model(None, "Claude Code", TRUSTED_MODEL, URL, None)[0] == "claude code"
    call_llm(
        provider="Claude Code", model=TRUSTED_MODEL, base_url=URL,
        messages=[{"role": "user", "content": "hello"}], max_tokens=32,
        main_runtime={"provider": "moa", "base_url": "moa://local", "model": "simple"},
    )
    assert_oauth_wire(relay[-1], True)


# ── a named entry whose name is a built-in alias keeps its own authority ─────

@pytest.mark.parametrize(
    "entry_url,wire",
    [
        # api_mode declared: the named branch builds the Anthropic client itself.
        (URL, {"transport": "anthropic_messages"}),
        # No api_mode, an Anthropic-shaped URL: the named branch hands off to _wrap_transport.
        (f"{URL}/anthropic", {}),
    ],
    ids=["named-branch", "wrap-transport"],
)
@pytest.mark.parametrize("aux_model,expected", [(TRUSTED_MODEL, True), (TRUSTLESS_MODEL, False)])
def test_an_alias_named_relay_keeps_its_policy_through_the_cached_client(
    relay, entry_url, wire, aux_model, expected,
):
    """``custom:claude`` is selected from the raw name (``claude`` is the ``anthropic`` alias); the
    wire policy must be looked up under that same identity, not the alias-normalized built-in,
    which owns no custom entry and would answer ``false`` for a relay declaring ``true``."""
    from agent.auxiliary_client import _get_cached_client
    from hermes_cli.auth import known_provider_id

    assert known_provider_id("claude") == "anthropic"
    _rewrite_config(providers={"claude": {
        "api": entry_url, "key_env": "TEST_RELAY_KEY", **wire,
        "capabilities": {"anthropic_oauth_proxy": True},
        "models": {TRUSTLESS_MODEL: {"anthropic_oauth_proxy": False}},
    }})
    # A main session on another route, so nothing can be inherited: only the entry decides.
    main = {"provider": "moa", "base_url": "moa://local", "model": "simple"}
    client, model = _get_cached_client("custom:claude", aux_model, main_runtime=main)
    client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "hello"}], max_tokens=32,
    )
    assert_oauth_wire(relay[-1], expected)
    assert ("Claude Code" in json.dumps(json.loads(relay[-1].content).get("system", []))) is expected


def test_an_unflagged_alias_named_relay_does_not_turn_oauth_on_its_key_shape(relay, monkeypatch):
    """The reverse of the same identity loss: normalized to ``anthropic``, an unflagged relay
    named ``claude`` holding an OAuth-shaped key was classified as the native provider, so the
    payload got the Claude Code transforms while the client itself sent ``x-api-key``."""
    from agent.auxiliary_client import _get_cached_client

    oauth_shaped = "sk-ant-oat01-" + "x" * 40
    monkeypatch.setenv("TEST_RELAY_KEY", oauth_shaped)
    _rewrite_config(providers={"claude": {
        "api": URL, "key_env": "TEST_RELAY_KEY", "transport": "anthropic_messages",
    }})
    main = {"provider": "moa", "base_url": "moa://local", "model": "simple"}
    client, model = _get_cached_client("custom:claude", TRUSTED_MODEL, main_runtime=main)
    client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "hello"}], max_tokens=32,
    )
    request = relay[-1]
    assert request.headers.get("x-api-key") == oauth_shaped
    assert "oauth-2025-04-20" not in request.headers.get("anthropic-beta", "")
    assert "Claude Code" not in json.dumps(json.loads(request.content).get("system", []))


# ── the cheap lookup picks exactly the entry the full lookup picks ────────────

@pytest.mark.parametrize(
    "config,name,expected",
    [
        # enabled: false is invisible, the next match wins
        ({"providers": {"a": {"name": "dup", "api": "https://off.example.com", "enabled": False},
                        "b": {"name": "dup", "api": "https://on.example.com"}}}, "dup", "https://on.example.com"),
        # name: differs from the key — both spellings reach it
        ({"providers": {"key-name": {"name": "Display Name", "api": "https://d.example.com"}}},
         "display-name", "https://d.example.com"),
        ({"providers": {"key-name": {"name": "Display Name", "api": "https://d.example.com"}}},
         "key-name", "https://d.example.com"),
        # legacy list only
        ({"custom_providers": [{"name": "legacy", "base_url": "https://l.example.com"}]},
         "legacy", "https://l.example.com"),
        # both lists: providers: wins
        ({"providers": {"both": {"api": "https://new.example.com"}},
          "custom_providers": [{"name": "both", "base_url": "https://old.example.com"}]},
         "both", "https://new.example.com"),
        # a dict-shaped custom_providers is malformed: nothing
        ({"custom_providers": {"x": {"base_url": "https://x.example.com"}}}, "x", ""),
    ],
)
def test_endpoint_lookup_matches_the_full_lookup(relay, config, name, expected):
    """Finding 3b (review 2): the read-only matcher duplicates ``_get_named_custom_provider``."""
    from hermes_cli.runtime_provider_custom import _get_named_custom_provider, named_custom_provider_endpoint

    _rewrite_config(_replace=True, **{"providers": {}, "custom_providers": [], **config})
    assert named_custom_provider_endpoint(name) == expected
    assert ((_get_named_custom_provider(name) or {}).get("base_url") or "") == expected


def test_ownership_check_follows_a_patched_runtime_config(relay, monkeypatch):
    """Finding 6 (review 2): tests patch ``runtime_provider.load_config``; the ownership check must
    answer from that same config, not from the file underneath it."""
    from hermes_cli import runtime_provider
    from hermes_cli.route_identity import named_provider_owns_endpoint

    patched = {"providers": {"patched": {"api": "https://patched.example.com"}}}
    monkeypatch.setattr(runtime_provider, "load_config", lambda: patched)
    assert named_provider_owns_endpoint("patched", "https://patched.example.com") is True
    assert named_provider_owns_endpoint("relay", URL) is False


def test_wire_policy_path_reads_no_credential(relay, monkeypatch):
    """Finding 4 (review 2): the chokepoint itself — not just the gate — must not resolve a secret."""
    from agent.auxiliary_oauth import runtime_oauth_proxy
    from hermes_cli import runtime_provider_custom

    def refuse(*_args, **_kwargs):
        raise AssertionError("wire-policy lookup read a credential")

    monkeypatch.setattr(runtime_provider_custom, "get_secret_str", refuse)
    assert runtime_oauth_proxy(None, "relay", URL, TRUSTED_MODEL) is True
    assert runtime_oauth_proxy(None, "relay", "", TRUSTLESS_MODEL) is False
