"""DeepInfra profile puts the reasoning switch on the wire (#111872).

DeepInfra's OpenAI-compatible endpoint reads one top-level ``reasoning_effort`` field validated
against a gateway-wide enum (``none``..``max``; Hermes-internal ``ultra`` is rejected). The
profile is the ONLY source of that field on the transport's profile path, and the core
``_supports_reasoning_extra_body`` allowlist passes ``supports_reasoning=False`` for this host,
so the profile must emit without gating on it.

The same profile also answers ``/usage`` for DeepInfra's prepaid credit through the provider-profile
hook (``fetch_account_usage``) — those cases sit at the bottom of this file.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def deepinfra_profile():
    import model_tools  # noqa: F401  (plugin discovery registers the profile)
    import providers

    profile = providers.get_provider_profile("deepinfra")
    assert profile is not None, "deepinfra provider profile must be registered"
    return profile


@pytest.mark.parametrize(
    "reasoning_config, expected_top_level",
    [
        ({"enabled": True, "effort": "high"}, {"reasoning_effort": "high"}),
        ({"enabled": True, "effort": "xhigh"}, {"reasoning_effort": "xhigh"}),  # native, never folded into max
        ({"enabled": True, "effort": "ultra"}, {"reasoning_effort": "max"}),  # Hermes-internal tier clamps
        ({"enabled": False}, {"reasoning_effort": "none"}),  # the only off switch for default-on models
        ({"enabled": True, "effort": "none"}, {"reasoning_effort": "none"}),
        (None, {}),  # nothing requested → keep DeepInfra's per-model default
        ({"enabled": True}, {}),
        ({"enabled": True, "effort": "future-tier"}, {}),  # unknown level omitted rather than 422
    ],
)
def test_profile_translates_reasoning_config_to_top_level_effort(deepinfra_profile, reasoning_config, expected_top_level):
    extra_body, top_level = deepinfra_profile.build_api_kwargs_extras(
        reasoning_config=reasoning_config, supports_reasoning=False, model="deepseek-ai/DeepSeek-V4.1-Flash",
    )
    assert extra_body == {}
    assert top_level == expected_top_level


def test_transport_main_turn_carries_reasoning_effort_without_capability_gate(deepinfra_profile):
    """The main turn builds through ``_build_kwargs_from_profile`` with ``supports_reasoning=False``
    (core allowlist excludes this host) — the field must still reach the request."""
    from agent.transports.chat_completions import ChatCompletionsTransport

    build = ChatCompletionsTransport().build_kwargs
    on = build(
        model="deepseek-ai/DeepSeek-V4.1-Flash", messages=[{"role": "user", "content": "ping"}], tools=None,
        provider_profile=deepinfra_profile, provider_name="deepinfra",
        reasoning_config={"enabled": True, "effort": "high"}, supports_reasoning=False,
    )
    off = build(
        model="zai-org/GLM-4.6", messages=[{"role": "user", "content": "ping"}], tools=None,
        provider_profile=deepinfra_profile, provider_name="deepinfra",
        reasoning_config={"enabled": False}, supports_reasoning=False,
    )
    assert on["reasoning_effort"] == "high"
    assert off["reasoning_effort"] == "none"
    assert "reasoning" not in (on.get("extra_body") or {})


# ── /usage: prepaid credit, held as a negative Stripe balance ───────────────────────────────

# The checklist response as it really arrives: the figures we read, plus the billing PII
# (holder name, address, email flag) that must never reach a surface.
CHECKLIST = {
    "stripe_balance": -5.0,  # credit is held as a negative balance
    "recent": 0.04,
    "limit": None,
    "suspended": False,
    "billing_type": "prepaid",
    "email": "holder@example.com",
    "name": "Holder Example",
    "billing_address_info": {"line1": "1 Example St", "postal_code": "2000"},
}
PII = ("Holder Example", "holder@example.com", "1 Example St", "2000")


class _Response:
    def __init__(self, payload, status_code):
        self.payload, self.status_code = payload, status_code

    def json(self):
        return self.payload


class _FakeHTTP:
    """Stand-in for ``httpx.Client``: records the request, replays a canned response."""

    def __init__(self, payload=None, status_code=200, error=None):
        self.payload, self.status_code, self.error = payload, status_code, error
        self.requests: list[tuple[str, dict]] = []
        self.timeout = None

    def install(self, monkeypatch):
        fake = self

        class _Client:
            def __init__(self, timeout=None):
                fake.timeout = timeout

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get(self, url, headers=None):
                fake.requests.append((url, dict(headers or {})))
                if fake.error is not None:
                    raise fake.error
                return _Response(fake.payload, fake.status_code)

        monkeypatch.setattr("httpx.Client", _Client)
        return fake


@pytest.fixture
def runtime(monkeypatch):
    """Credential resolution, returning the inference route the profile actually carries."""

    def _install(base_url="https://api.deepinfra.com/v1/openai", api_key="sk-deepinfra-test"):
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda requested, explicit_base_url=None, explicit_api_key=None: {
                "provider": requested, "base_url": base_url, "api_key": api_key,
            },
        )

    return _install


def test_credit_and_recent_spend_reach_usage(deepinfra_profile, runtime, monkeypatch):
    """The hook is what /usage calls for a provider with no built-in fetcher."""
    from agent.account_usage import fetch_account_usage, render_account_usage_lines

    runtime()
    _FakeHTTP(payload=CHECKLIST).install(monkeypatch)

    snapshot = fetch_account_usage("deepinfra")

    assert snapshot is not None
    assert (snapshot.provider, snapshot.source) == ("deepinfra", "payment_checklist")
    assert snapshot.details == ("Credits balance: $5.00", "Recent spend: $0.04")
    assert snapshot.windows == ()
    assert "Credits balance: $5.00" in "\n".join(render_account_usage_lines(snapshot))


def test_checklist_is_requested_at_the_api_origin_not_the_inference_route(
    deepinfra_profile, runtime, monkeypatch
):
    """``/v1/openai`` is the inference route; the checklist lives beside it, at the origin."""
    from agent.account_usage import fetch_account_usage

    runtime(base_url="https://api.deepinfra.com/v1/openai")
    fake = _FakeHTTP(payload=CHECKLIST).install(monkeypatch)

    fetch_account_usage("deepinfra")

    assert [url for url, _ in fake.requests] == [
        "https://api.deepinfra.com/payment/checklist?compute_owed=true"
    ]
    assert fake.requests[0][1]["Authorization"] == "Bearer sk-deepinfra-test"


def test_billing_pii_never_enters_the_snapshot(deepinfra_profile, runtime, monkeypatch):
    from agent.account_usage import fetch_account_usage, render_account_usage_lines

    runtime()
    _FakeHTTP(payload=CHECKLIST).install(monkeypatch)

    snapshot = fetch_account_usage("deepinfra")

    assert snapshot is not None
    rendered = "\n".join(render_account_usage_lines(snapshot)) + repr(snapshot.windows)
    assert not [leak for leak in PII if leak in rendered]
    assert snapshot.raw is None  # raw would carry the whole body, PII included


def test_spend_cap_renders_as_a_window_with_remaining_dollars(
    deepinfra_profile, runtime, monkeypatch
):
    from agent.account_usage import fetch_account_usage, render_account_usage_lines

    runtime()
    _FakeHTTP(payload={**CHECKLIST, "limit": 20.0}).install(monkeypatch)

    snapshot = fetch_account_usage("deepinfra")

    assert snapshot is not None
    assert [(w.label, w.used_percent, w.detail) for w in snapshot.windows] == [
        ("Spending limit", 75.0, "$5.00 of $20.00 remaining")
    ]
    rendered = "\n".join(render_account_usage_lines(snapshot))
    assert "Spending limit: 25% remaining (75% used)" in rendered
    assert "$5.00 of $20.00 remaining" in rendered


@pytest.mark.parametrize(
    "payload",
    [
        {"recent": 0.04},
        {"stripe_balance": None},
        {"stripe_balance": "not-a-number"},
    ],
)
def test_payload_without_a_usable_credit_figure_is_soft(
    deepinfra_profile, runtime, monkeypatch, payload
):
    from agent.account_usage import fetch_account_usage

    runtime()
    _FakeHTTP(payload=payload).install(monkeypatch)

    assert fetch_account_usage("deepinfra") is None


def test_non_200_is_soft(deepinfra_profile, runtime, monkeypatch):
    from agent.account_usage import fetch_account_usage

    runtime()
    _FakeHTTP(payload=CHECKLIST, status_code=503).install(monkeypatch)

    assert fetch_account_usage("deepinfra") is None


def test_transport_error_is_soft(deepinfra_profile, runtime, monkeypatch):
    import httpx

    from agent.account_usage import fetch_account_usage

    runtime()
    _FakeHTTP(error=httpx.ConnectError("boom")).install(monkeypatch)

    assert fetch_account_usage("deepinfra") is None


def test_no_credential_makes_no_request(deepinfra_profile, runtime, monkeypatch):
    from agent.account_usage import fetch_account_usage

    runtime(api_key="")
    fake = _FakeHTTP(payload=CHECKLIST).install(monkeypatch)

    assert fetch_account_usage("deepinfra") is None
    assert fake.requests == []


def test_depleted_and_suspended_accounts_are_named(deepinfra_profile, runtime, monkeypatch):
    from agent.account_usage import fetch_account_usage

    runtime()
    _FakeHTTP(payload={**CHECKLIST, "stripe_balance": 0.0}).install(monkeypatch)
    depleted = fetch_account_usage("deepinfra")
    assert depleted is not None
    assert depleted.details[0] == "Credits balance: $0.00"
    assert depleted.details[-1] == "Status: access depleted — top up to restore"

    _FakeHTTP(payload={**CHECKLIST, "suspended": True}).install(monkeypatch)
    suspended = fetch_account_usage("deepinfra")
    assert suspended is not None
    assert suspended.details[-1] == "Status: suspended — billing action required"


def test_hook_timeout_stays_under_the_shared_deadline(deepinfra_profile, runtime, monkeypatch):
    """Core bounds the hook at 10 s and renders nothing past it, so ours must be shorter."""
    from agent.account_usage import PLUGIN_USAGE_HOOK_DEADLINE_S, fetch_account_usage

    runtime()
    fake = _FakeHTTP(payload=CHECKLIST).install(monkeypatch)

    fetch_account_usage("deepinfra")

    assert fake.timeout is not None and fake.timeout < PLUGIN_USAGE_HOOK_DEADLINE_S


# ── edges the first cut got wrong: hostile payloads, unresolvable origins, refused keys ──────


def _profile_module():
    """The plugin module under the name the loader gave it.

    Importing it by path (``plugins.model_providers.deepinfra``) is unreliable: a same-named
    regular package on ``sys.path`` shadows the namespace directory and the import fails. The
    registry already holds the loaded module, so read it from there instead.
    """
    import sys

    import model_tools  # noqa: F401  (plugin discovery registers the profile)
    import providers

    profile = providers.get_provider_profile("deepinfra")
    assert profile is not None, "deepinfra provider profile must be registered"
    return sys.modules[type(profile).__module__]


@pytest.mark.parametrize(
    "value, expected",
    [
        (-5.0, -5.0),
        (0, 0.0),
        ("5.00", 5.0),  # providers disagree on number-vs-string
        (" 5 ", 5.0),
        ("-3.5", -3.5),
        (True, None),  # bool subclasses int, and a boolean is not a balance
        (False, None),
        (None, None),
        ("", None),
        ("not-a-number", None),
        ({}, None),
        ([1, 2], None),
        (float("inf"), None),  # a non-finite figure would render as $inf
        (float("nan"), None),
        (10**400, None),  # OverflowError is contained, never raised
    ],
)
def test_json_number_coercion_is_total(value, expected):
    """Every shape the checklist can send coerces or yields None — it must never raise."""
    deepinfra_module = _profile_module()

    assert deepinfra_module._json_number(value) == expected


@pytest.mark.parametrize(
    "base_url, expected",
    [
        ("https://api.deepinfra.com/v1/openai", "https://api.deepinfra.com"),
        ("https://api.deepinfra.com/v1/openai/", "https://api.deepinfra.com"),
        ("https://api.deepinfra.com/v1", "https://api.deepinfra.com"),  # anthropic-compatible route
        ("https://api.deepinfra.com:8443/v1", "https://api.deepinfra.com:8443"),
        ("  https://api.deepinfra.com/v1  ", "https://api.deepinfra.com"),
        # a path-routed proxy keeps its prefix; the inference suffix is what gets dropped
        ("https://proxy.example.com/deepinfra/v1", "https://proxy.example.com/deepinfra"),
        ("localhost:8000", None),  # configured but unusable → never guess production
        ("", "https://api.deepinfra.com"),  # absent → the profile's own host
        (None, "https://api.deepinfra.com"),
    ],
)
def test_api_origin_resolution(base_url, expected):
    deepinfra_module = _profile_module()

    assert deepinfra_module._api_origin(base_url) == expected


def test_unusable_base_url_sends_the_key_nowhere(deepinfra_profile, runtime, monkeypatch):
    """A base_url we cannot parse must not fall back to production: the credential in hand may
    belong to whatever that value points at, so no request may be made at all."""
    from agent.account_usage import fetch_account_usage

    runtime(base_url="localhost:8000")
    fake = _FakeHTTP(payload=CHECKLIST).install(monkeypatch)

    assert fetch_account_usage("deepinfra") is None
    assert fake.requests == []


@pytest.mark.parametrize("payload", [[1, 2, 3], "checklist", 42])
def test_non_object_payload_is_soft(deepinfra_profile, runtime, monkeypatch, payload):
    """A JSON array or scalar body is not a checklist: it must not raise AttributeError."""
    from agent.account_usage import fetch_account_usage

    runtime()
    _FakeHTTP(payload=payload).install(monkeypatch)

    assert fetch_account_usage("deepinfra") is None


def test_overflowing_credit_figure_is_soft_not_a_traceback(deepinfra_profile, runtime, monkeypatch):
    from agent.account_usage import fetch_account_usage

    runtime()
    _FakeHTTP(payload={**CHECKLIST, "stripe_balance": 10**400}).install(monkeypatch)

    assert fetch_account_usage("deepinfra") is None


@pytest.mark.parametrize("status_code", [401, 403])
def test_rejected_credential_is_reported_not_hidden(
    deepinfra_profile, runtime, monkeypatch, status_code
):
    """A refused key is the user's to fix, so it surfaces through the contract's own
    ``unavailable_reason`` field rather than rendering nothing."""
    from agent.account_usage import fetch_account_usage, render_account_usage_lines

    runtime()
    _FakeHTTP(payload={"error": "unauthorized"}, status_code=status_code).install(monkeypatch)

    snapshot = fetch_account_usage("deepinfra")

    assert snapshot is not None
    assert snapshot.available is False
    assert f"HTTP {status_code}" in (snapshot.unavailable_reason or "")
    assert "Unavailable:" in "\n".join(render_account_usage_lines(snapshot))


def test_overdrawn_account_reports_the_credit_floor(deepinfra_profile, runtime, monkeypatch):
    """A positive Stripe balance means no credit is left. /usage reports the floor rather than a
    debt figure, because the endpoint does not document the units of an amount owed."""
    from agent.account_usage import fetch_account_usage

    runtime()
    _FakeHTTP(payload={**CHECKLIST, "stripe_balance": 50.0}).install(monkeypatch)

    snapshot = fetch_account_usage("deepinfra")

    assert snapshot is not None
    assert snapshot.details[0] == "Credits balance: $0.00"  # the credit floor, not a debt figure
    assert snapshot.details[-1] == "Status: access depleted — top up to restore"
