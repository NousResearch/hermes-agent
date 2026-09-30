"""Picker availability follows credential state, not leftover catalog metadata.

Behaviour contracts:

* A ``providers.<built-in>`` block must not resurrect a provider whose key was removed
  (residual ``base_url`` / ``key_env`` / a stale env var inherited by a long-lived process),
  while genuinely local keyless endpoints keep working.
* An explicit endpoint override that authenticates header-only (``Authorization`` /
  ``X-Api-Key`` in ``extra_headers``) is a real configuration, not removal residue — but a
  non-auth header (``User-Agent``), an empty value, a placeholder, or a ``Bearer`` scheme
  wrapping a placeholder/empty token never authenticates.
* A pool whose only credential is in real 429 cooldown stays selectable on both the
  Desktop (``explicit_only``) and TUI (``include_unconfigured``) surfaces, with or without an
  ``auth.json.providers`` registration. The cooldown fixture must establish a real deadline,
  or the reloaded pool reports ``has_available() is True`` and the test passes vacuously.
* DEAD / empty OAuth husks are not advertised as authenticated; an access token known to be
  expired with no refresh material is neither usable nor recoverable, while an expired token
  with live refresh material, a not-yet-expired token, an unknown-expiry token, and any live
  credential in a multi-credential pool still are.
* Placeholder strings in provider state (``"none"`` / ``"placeholder"``) are not a login,
  and picker visibility never changes the endpoint-probe budget of a normal options open.

Everything below drives the real credential lifecycle (``save/remove_provider_env_credential``),
the real pool loader and the real options entry point; only catalog/network boundaries are
replaced. Credential values are synthetic.
"""

import json
import time
from pathlib import Path

import pytest

_PROVIDER = "openai-codex"
_SYNTHETIC_KEY = "test-" + "a" * 24


@pytest.fixture
def picker_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli.config import invalidate_env_cache
    invalidate_env_cache()
    # Only catalog/network boundaries are replaced; auth, pools, removal and inventory are real.
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda *a, **k: {
        "openrouter": {"env": ["OPENROUTER_API_KEY"], "models": {}},
        "zai": {"env": ["ZAI_API_KEY"], "models": {}},
        "openai": {"env": ["OPENAI_API_KEY"], "models": {}},
    })
    def offline(*args, **kwargs):
        raise OSError("network disabled for picker regression")
    monkeypatch.setattr("socket.socket.connect", offline)
    monkeypatch.setattr("hermes_cli.models.cached_provider_model_ids", lambda *a, **k: ["test-model"])
    monkeypatch.setattr("hermes_cli.models.get_curated_nous_model_ids", lambda *a, **k: [])
    monkeypatch.setattr("hermes_cli.models.fetch_ollama_cloud_models", lambda *a, **k: [])
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", lambda *a, **k: [])
    return tmp_path


def _row(payload, slug):
    return next((p for p in payload["providers"] if p["slug"] == slug), None)


def _register_provider(provider=_PROVIDER, **state):
    """Write an ``auth.json.providers`` registration (the chatgpt-auth shape).

    Must run BEFORE any pool seeding: ``_save_auth_store`` replaces the whole file, so saving
    it afterwards would erase the pool entry under test.
    """
    from hermes_cli.auth import _save_auth_store
    _save_auth_store({"version": 1, "providers": {provider: {"auth_mode": "chatgpt", **state}}})


def _options(*, current_provider="opencode-free", explicit_only=False, include_unconfigured=False):
    """Build the real model.options payload the Desktop/TUI pickers consume."""
    from hermes_cli.inventory import ConfigContext, build_model_options_payload
    ctx = ConfigContext(current_provider, "test-model", "", {}, [])
    return build_model_options_payload(
        ctx, explicit_only=explicit_only, include_unconfigured=include_unconfigured)


def _add_pool_entry(provider, **fields):
    from agent.credential_pool import PooledCredential, load_pool
    entry = PooledCredential.from_dict(provider, {"source": "manual", "auth_type": "oauth", **fields})
    load_pool(provider).add_entry(entry)
    return entry


def _canonical_id(provider):
    """Canonical provider id (``glm`` -> ``zai``): pools, auth state and suppression records are
    keyed by it. Uses the pre-existing public helper so the file collects on the unfixed baseline
    too — a red must come from a behaviour assertion, not an ImportError."""
    from hermes_cli.providers import normalize_provider
    return normalize_provider(provider)


def _canonical_suppression_recorded(provider, env):
    """A removal records suppression under the provider's canonical id (``glm`` -> ``zai``)."""
    from hermes_cli.auth import is_source_suppressed
    return is_source_suppressed(_canonical_id(provider), "env:" + env)


def _cooldown_entry(**overrides):
    """A credential in a *real* 429 cooldown: future reset deadline, usable token."""
    now = time.time()
    base = {
        "id": "rate_limited_cred", "source": "manual", "auth_type": "oauth",
        "access_token": "test-token-12345678",
        "last_status": "exhausted", "last_status_at": now,
        "last_error_code": 429, "last_error_reason": "usage_limit_reached",
        "last_error_reset_at": now + 3600,
    }
    base.update(overrides)
    return base


# ─── Removal → residue → re-add ─────────────────────────────────────────


@pytest.mark.parametrize("provider,env_var,models_only,lingering_env", [
    ("zai", "ZAI_API_KEY", False, False),
    ("opencode-go", "OPENCODE_GO_API_KEY", True, False),
    ("opencode-go", "OPENCODE_GO_API_KEY", False, True),
    ("openrouter", "OPENROUTER_API_KEY", False, True),
])
def test_picker_removal_and_readd(picker_home, monkeypatch, provider, env_var, models_only, lingering_env):
    from agent.credential_pool import load_pool
    from hermes_cli.auth import _save_auth_store
    from hermes_cli.credential_lifecycle import remove_provider_env_credential, save_provider_env_credential
    from hermes_cli.inventory import build_models_payload, load_picker_context

    config = {"model": {"provider": provider, "default": "test-model"}}
    if models_only:
        config["providers"] = {provider: {"models": {"test-model": {}}, "stale_timeout_seconds": 60}}
    picker_home.joinpath("config.yaml").write_text(json.dumps(config), encoding="utf-8")
    metadata = {provider: {"detected_endpoint": "https://example.invalid/v1"}} if provider == "zai" else {}
    _save_auth_store({"version": 1, "providers": metadata})
    # A base override keeps Z.ai region detection offline.
    monkeypatch.setenv("ZAI_BASE_URL", "https://example.invalid/v1")
    key = "test-" + "a" * 24

    def row():
        payload = build_models_payload(load_picker_context(), explicit_only=True, picker_hints=True,
                                       probe_custom_providers=False, for_picker=True)
        return next(p for p in payload["providers"] if p["slug"] == provider)

    save_provider_env_credential(env_var, key)
    assert load_pool(provider).has_credentials()
    assert row()["authenticated"] is True
    remove_provider_env_credential(env_var)
    if lingering_env:
        monkeypatch.setenv(env_var, key)  # another process still inherited the removed key
    assert not load_pool(provider).has_credentials()
    removed = row()
    assert removed["authenticated"] is False
    assert removed["source"] == "configured-current"
    assert removed["warning"]
    unavailable = build_models_payload(
        load_picker_context().with_overrides(current_provider="opencode-free"),
        probe_custom_providers=False, for_picker=True)["providers"]
    assert provider not in {p["slug"] for p in unavailable}
    assert removed["models"] == ["test-model"]  # saved selection, never the old catalog
    save_provider_env_credential(env_var, key)
    assert load_pool(provider).has_credentials()
    assert row()["authenticated"] is True


@pytest.mark.parametrize("provider,env,override", [
    ("opencode-go", "OPENCODE_GO_API_KEY", "base_url"),
    ("opencode-go", "OPENCODE_GO_API_KEY", "key_env"),
    # An alias-named block must resolve the same suppression record (keyed by the canonical id).
    ("glm", "GLM_API_KEY", "key_env"),
])
def test_removed_key_cannot_reappear_via_builtin_overrides(picker_home, monkeypatch, provider, env, override):
    """A ``providers.<built-in>`` block is an override of that provider, not a fresh endpoint:
    once the key is removed, neither a residual ``base_url`` nor a ``key_env`` still inherited by
    an old process may report the row as authenticated, and re-adding the key restores it."""
    from agent.credential_pool import load_pool
    from hermes_cli.credential_lifecycle import remove_provider_env_credential, save_provider_env_credential
    from hermes_cli.inventory import build_model_options_payload, load_picker_context

    pool_id = _canonical_id(provider)  # pools/auth state are keyed by the canonical id
    entry = {"models": {"test-model": {}}, "stale_timeout_seconds": 60}
    if override == "base_url":
        entry["base_url"] = "https://opencode.ai/zen/go/v1"
    else:
        entry["key_env"] = env
    cfg = {"model": {"provider": "opencode-free", "default": "test-model"}, "providers": {provider: entry}}
    picker_home.joinpath("config.yaml").write_text(json.dumps(cfg), encoding="utf-8")

    save_provider_env_credential(env, _SYNTHETIC_KEY)
    remove_provider_env_credential(env)
    if override == "key_env":
        monkeypatch.setenv(env, _SYNTHETIC_KEY)  # a long-lived process still holds the old key
    assert load_pool(pool_id).has_credentials() is False
    assert _canonical_suppression_recorded(provider, env)

    def rows(**kwargs):
        return {p["slug"]: p for p in
                build_model_options_payload(load_picker_context(), **kwargs)["providers"]}

    for surface in ({"explicit_only": True}, {"include_unconfigured": True}):
        row = rows(**surface).get(provider)
        print(f"REMOVED_OVERRIDE {provider}/{override} {surface} -> {row}")
        assert row is None or row.get("authenticated") is False, (provider, override, surface, row)

    save_provider_env_credential(env, _SYNTHETIC_KEY)
    assert load_pool(pool_id).has_credentials()
    assert rows(explicit_only=True)[provider]["authenticated"] is True


@pytest.mark.parametrize("name", ["lmstudio", "local-test"])
def test_keyless_local_endpoints_survive_builtin_and_custom_names(picker_home, name):
    """The residue guard must not eat real keyless endpoints: a loopback URL is legitimate under a
    built-in provider name (LM Studio on its own port) exactly as it is under a custom name."""
    from hermes_cli.inventory import ConfigContext, build_models_payload

    local = {name: {"base_url": "http://127.0.0.1:9999/v1", "models": ["local-model"]}}
    rows = {r["slug"]: r for r in build_models_payload(
        ConfigContext("", "", "", local, []), for_picker=True, picker_hints=True,
        probe_custom_providers=False)["providers"]}
    assert rows[name]["models"] == ["local-model"]
    assert rows[name]["authenticated"] is True


# ─── Header-only authentication on explicit endpoint overrides ──────────


@pytest.mark.parametrize("headers", [
    {"Authorization": "Bearer test-header-only"},
    {"X-Api-Key": "test-header-only"},
    # HTTP auth semantics are case-insensitive on the scheme word.
    {"Authorization": "bearer test-header-only"},
])
def test_builtin_named_override_with_header_auth_is_kept(picker_home, headers):
    """An explicit ``providers.<built-in>`` endpoint override that authenticates through
    ``extra_headers`` (``Authorization`` / ``X-Api-Key``) is a real configuration: the removal
    residue guard must not hide it. This is how the rest of the codebase already treats a
    caller-supplied ``Authorization`` header (validation and local-model paths skip injecting
    their own), and ``X-Api-Key`` is the other header the runtime authenticates with."""
    from hermes_cli.inventory import ConfigContext, build_models_payload
    from hermes_cli.model_switch import _extra_headers_from_config

    entry = {"base_url": "https://gateway.example.invalid/v1",
             "models": ["header-model"], "extra_headers": headers}
    assert _extra_headers_from_config(entry) == headers
    ctx = ConfigContext("", "", "", {"openai": entry}, [])
    payload = build_models_payload(ctx, for_picker=True, picker_hints=True,
                                   probe_custom_providers=False)
    row = next((r for r in payload["providers"] if r["slug"] == "openai"), None)
    assert row is not None, "header-only auth must not read as removal residue"
    assert row["authenticated"] is True
    assert row["models"] == ["header-model"]
    assert row["source"] == "user-config"


@pytest.mark.parametrize("headers", [
    {"User-Agent": "hermes-test/1.0"},
    {"Authorization": "placeholder"},
    {"X-Api-Key": "none"},
    {"Authorization": "  "},
    # The scheme word is not the credential: a Bearer wrapper around a placeholder or an
    # empty token must not authenticate either.
    {"Authorization": "Bearer none"},
    {"Authorization": "Bearer placeholder"},
    {"Authorization": "Bearer   "},
    {"Authorization": "Bearer"},
])
def test_non_auth_or_placeholder_headers_do_not_resurrect_a_builtin(picker_home, headers):
    """The counterweight: a non-auth header, a placeholder value, an empty value, or a
    scheme-only/placeholder-token ``Bearer`` wrapper is not a credential, so a built-in named
    override carrying only those stays hidden — blanket ``bool(extra_headers)`` admission, or
    judging the whole ``Authorization`` string instead of its token part, would wrongly
    resurrect every one of these rows."""
    from hermes_cli.inventory import ConfigContext, build_models_payload

    entry = {"base_url": "https://gateway.example.invalid/v1",
             "models": ["header-model"], "extra_headers": headers}
    ctx = ConfigContext("", "", "", {"openai": entry}, [])
    payload = build_models_payload(ctx, for_picker=True, picker_hints=True,
                                   probe_custom_providers=False)
    row = next((r for r in payload["providers"] if r["slug"] == "openai"), None)
    assert row is None or row.get("authenticated") is False, (headers, row)


# ─── 429 cooldown visibility ────────────────────────────────────────────


def test_picker_keeps_rate_limited_exhausted_provider(picker_home):
    """A pool entirely in 429 cooldown must stay in model.options with its models.

    The fixture must establish a real cooldown deadline and assert that first: with only
    ``last_status``/reason written (no ``last_status_at``, no future reset) the reloaded pool is
    immediately available again and every assertion below would pass without exercising cooldown.
    """
    from agent.credential_pool import PooledCredential, load_pool

    _register_provider()
    load_pool(_PROVIDER).add_entry(PooledCredential.from_dict(_PROVIDER, _cooldown_entry()))

    reloaded = load_pool(_PROVIDER)  # what the picker sees on the next open
    assert reloaded.has_credentials() is True, "the cooldown entry must still be in the pool"
    assert reloaded.has_available() is False, "fixture must establish a live cooldown"

    row = _row(_options(explicit_only=True), _PROVIDER)
    assert row is not None, "a credential in cooldown must not vanish from the picker"
    assert row["authenticated"] is True
    assert row["models"], "the cooled-down provider keeps its selectable model list"


@pytest.mark.parametrize("surface,kwargs", [
    ("desktop", {"explicit_only": True}),
    ("tui", {"include_unconfigured": True}),
])
@pytest.mark.parametrize("registered", [False, True])
def test_429_cooldown_stays_selectable_on_both_surfaces(picker_home, surface, kwargs, registered):
    """Cooldown visibility is not a function of the ``auth.json.providers`` registration.

    ``registered=True`` covers the patch's original path (registration + pool). ``registered=False``
    is the ``hermes auth add`` shape: a self-contained ``manual:*`` pool entry with no singleton
    shadow, which the options payload used to drop/hide entirely.
    """
    if registered:
        _register_provider()
    _add_pool_entry(_PROVIDER, **_cooldown_entry())
    payload = _options(**kwargs)
    row = _row(payload, _PROVIDER)
    print(f"COOLDOWN registered={registered} surface={surface} row={row}")
    assert row is not None, (registered, surface, [p["slug"] for p in payload["providers"]])
    assert row["authenticated"] is True, (registered, surface, row)


# ─── DEAD / empty / recoverable / multi-credential ──────────────────────


@pytest.mark.parametrize("status,token,reason", [
    ("dead", "test-audit-token", "invalid_grant"),
    ("ok", "", ""),
])
def test_irrecoverable_or_empty_oauth_is_not_advertised_authenticated(picker_home, status, token, reason):
    """Pool *presence* is not authentication: a DEAD manual entry and a token-less entry without
    refresh material must be hidden or shown as needing re-auth, never as authenticated."""
    _register_provider()
    _add_pool_entry(_PROVIDER, id="unusable", access_token=token, last_status=status,
                    last_status_at=time.time(), last_error_code=401 if reason else None,
                    last_error_reason=reason or None)
    for kwargs in ({"explicit_only": True}, {"include_unconfigured": True}):
        payload = _options(**kwargs)
        row = _row(payload, _PROVIDER)
        print(f"UNUSABLE {status!r} {kwargs} -> {row}")
        assert row is None or row.get("authenticated") is False, (status, kwargs, row)


def test_irrecoverable_pool_exposes_a_reauth_affordance_on_tui(picker_home):
    """Hidden is acceptable; silently missing is not. On the TUI surface a canonical provider whose
    only credential is DEAD still shows up as needing configuration rather than as authenticated."""
    _register_provider()
    _add_pool_entry(_PROVIDER, id="husk", access_token="", last_status="dead",
                    last_status_at=time.time(), last_error_code=401, last_error_reason="invalid_grant")
    row = _row(_options(include_unconfigured=True), _PROVIDER)
    assert row is not None, "a canonical provider stays listed as needing re-auth"
    assert row["authenticated"] is False
    assert row.get("warning"), row


def test_recoverable_and_multi_credential_pools_stay_authenticated(picker_home):
    """The counterweight to the test above, so the material check cannot over-tighten: an expired
    access token with live refresh material is recoverable, and a pool holding one DEAD husk plus
    one live credential is still an authenticated provider."""
    _register_provider()
    _add_pool_entry(_PROVIDER, id="recoverable", access_token="",
                    refresh_token="test-refresh-material")
    row = _row(_options(explicit_only=True), _PROVIDER)
    assert row is not None and row["authenticated"] is True, row

    _add_pool_entry(_PROVIDER, id="husk", access_token="", last_status="dead",
                    last_status_at=time.time(), last_error_code=401,
                    last_error_reason="invalid_grant")
    row = _row(_options(explicit_only=True), _PROVIDER)
    assert row is not None and row["authenticated"] is True, row
    assert row["models"], row


def test_cooldown_and_dead_entries_coexist_without_hiding_the_provider(picker_home):
    """A DEAD husk must not mask a recoverable (cooling) credential in the same pool."""
    _register_provider()
    _add_pool_entry(_PROVIDER, id="husk", access_token="", last_status="dead",
                    last_status_at=time.time(), last_error_code=401,
                    last_error_reason="invalid_grant")
    _add_pool_entry(_PROVIDER, **_cooldown_entry(id="cooling"))
    row = _row(_options(explicit_only=True), _PROVIDER)
    assert row is not None and row["authenticated"] is True, row


# ─── OAuth expiry vs recoverability ─────────────────────────────────────


@pytest.mark.parametrize("expires_in,refresh,authenticated", [
    (-3600, "", False),                       # known-expired, no refresh material
    (-3600, "test-refresh-material", True),   # expired but recoverable via refresh
    (3600, "", True),                         # not yet expired — the refresh window is not expiry
    (None, "", True),                         # unknown expiry is not expiry
])
def test_oauth_expiry_gates_recoverability(picker_home, expires_in, refresh, authenticated):
    """A token *known* to be expired with no refresh material is neither a usable nor a
    recoverable login on either picker surface; every other expiry shape stays authenticated.
    The manual-pool shape (no ``auth.json.providers`` registration) matches ``hermes auth add``."""
    from agent.credential_pool import load_pool

    fields = {"access_token": "test-oauth-material", "refresh_token": refresh}
    if expires_in is not None:
        fields["expires_at_ms"] = int((time.time() + expires_in) * 1000)
    _add_pool_entry(_PROVIDER, id="oauth", **fields)
    if expires_in is not None and expires_in < 0:
        loaded = load_pool(_PROVIDER).entries()
        assert loaded and loaded[0].expires_at_ms < time.time() * 1000, \
            "fixture precondition: the token must be really expired"
    for kwargs in ({"explicit_only": True}, {"include_unconfigured": True}):
        row = _row(_options(**kwargs), _PROVIDER)
        print(f"OAUTH_EXPIRY expires_in={expires_in} refresh={bool(refresh)} {kwargs} -> {row}")
        if authenticated:
            assert row is not None and row["authenticated"] is True, (expires_in, kwargs, row)
        else:
            assert row is None or row.get("authenticated") is False, (expires_in, kwargs, row)


def test_expired_unrefreshable_pool_exposes_a_reauth_affordance_on_tui(picker_home):
    """Same affordance as the DEAD case: hidden or needs-auth, never authenticated. On the TUI
    surface a canonical provider whose only credential expired beyond recovery still shows up
    as needing configuration."""
    _register_provider()
    _add_pool_entry(_PROVIDER, id="expired", access_token="test-expired-material",
                    refresh_token="", expires_at_ms=int((time.time() - 3600) * 1000))
    row = _row(_options(include_unconfigured=True), _PROVIDER)
    assert row is not None, "a canonical provider stays listed as needing re-auth"
    assert row["authenticated"] is False
    assert row.get("warning"), row


def test_expired_husk_does_not_hide_a_live_pool_credential(picker_home):
    """A mixed pool keeps its login while any entry is valid or recoverable."""
    _add_pool_entry(_PROVIDER, id="expired", access_token="test-expired-material",
                    refresh_token="", expires_at_ms=int((time.time() - 3600) * 1000))
    _add_pool_entry(_PROVIDER, id="live", access_token="test-live-material")
    row = _row(_options(explicit_only=True), _PROVIDER)
    assert row is not None and row["authenticated"] is True, row


@pytest.mark.parametrize("state,authenticated", [
    ({"detected_endpoint": "https://example.invalid/v1"}, False),
    ({"base_url": "https://example.invalid/v1", "auth_mode": "chatgpt"}, False),
    ({"access_token": "test-provider-state-token"}, True),
    # Placeholder strings fail the same has_usable_secret gate as every other credential read.
    ({"api_key": "none"}, False),
    ({"access_token": "placeholder"}, False),
    ({}, False),
])
def test_auth_store_login_requires_credential_material(picker_home, state, authenticated):
    """The registration check keeps its declared contract: endpoint metadata is not a login, while
    a stored credential still is (a provider whose pool row has not been seeded yet stays usable)."""
    from hermes_cli.auth import _save_auth_store
    from hermes_cli.model_switch_providers import _auth_store_has_provider

    _save_auth_store({"version": 1, "providers": {"zai": state}})
    assert _auth_store_has_provider("zai") is authenticated


def test_dead_pool_is_not_masked_by_a_stale_provider_state_token(picker_home):
    """The pool is authoritative once it holds rows: a stale token left in provider state must not
    re-authenticate a credential the pool has marked DEAD."""
    from hermes_cli.model_switch_providers import _auth_store_has_provider

    _register_provider(access_token="test-stale-state-token")
    _add_pool_entry(_PROVIDER, id="husk", access_token="", last_status="dead",
                    last_status_at=time.time(), last_error_code=401, last_error_reason="invalid_grant")
    assert _auth_store_has_provider(_PROVIDER) is False
    row = _row(_options(explicit_only=True), _PROVIDER)
    assert row is None or row.get("authenticated") is False, row


# ─── Profile scope ──────────────────────────────────────────────────────


def test_picker_credentials_do_not_leak_across_profiles(picker_home, monkeypatch):
    """A→B→A: each profile's picker resolves its own ``.env`` credential, and coming back to A
    restores A's view instead of inheriting B's."""
    from hermes_cli.config import invalidate_env_cache
    from hermes_cli.inventory import ConfigContext, build_models_payload

    homes = {}
    for name, env_var in (("a", "ZAI_API_KEY"), ("b", "OPENROUTER_API_KEY")):
        home = picker_home / f"profile-{name}"
        home.mkdir()
        value = "test-" + (name * 12)
        (home / ".env").write_text(f"{env_var}={value}\n", encoding="utf-8")
        homes[name] = home

    def slugs(home):
        monkeypatch.setenv("HERMES_HOME", str(home))
        invalidate_env_cache()
        payload = build_models_payload(ConfigContext("", "", "", {}, []), for_picker=True)
        return {p["slug"] for p in payload["providers"]}

    in_a = slugs(homes["a"])
    assert "zai" in in_a and "openrouter" not in in_a, in_a
    in_b = slugs(homes["b"])
    assert "openrouter" in in_b and "zai" not in in_b, in_b
    back_to_a = slugs(homes["a"])
    assert "zai" in back_to_a and "openrouter" not in back_to_a, back_to_a


# ─── Existing contracts kept ────────────────────────────────────────────


@pytest.mark.parametrize("provider", ["opencode-go", "openai-codex"])
def test_picker_keeps_remaining_credentials_and_keyless_endpoints(picker_home, provider):
    from agent.credential_pool import load_pool, PooledCredential
    from hermes_cli.inventory import build_models_payload, ConfigContext

    pool = load_pool(provider)
    for ident in ("first", "second"):
        pool.add_entry(PooledCredential.from_dict(provider, {
            "id": ident, "source": "manual", "auth_type": "oauth" if provider == "openai-codex" else "api_key",
            "access_token": "test-" + ident * 8,
        }))
    local = {"local-test": {"base_url": "http://127.0.0.1:9999/v1", "models": ["local-model"]}}
    ctx = ConfigContext("", "", "", local, [])

    def rows():
        return {r["slug"]: r for r in build_models_payload(
            ctx, for_picker=True, picker_hints=True, probe_custom_providers=False)["providers"]}

    pool.remove_index(1)
    assert load_pool(provider).has_credentials()
    assert rows()[provider]["authenticated"] is True
    pool.remove_index(1)
    remaining = rows()
    assert provider not in remaining
    # The explicit keyless local endpoint survives credential churn.
    assert remaining["local-test"]["models"] == ["local-model"]
    assert remaining["local-test"]["authenticated"] is True


# ─── Probe budget is decoupled from picker visibility ───────────────────


def _probe_budget_seen(picker_home, monkeypatch):
    """Capture the probe-budget flag the endpoint probe is actually invoked with.

    ``for_picker`` is the flag's pre-decoupling name at the same call site, so the capture
    reads both — a red must come from a behaviour assertion, never from a renamed kwarg."""
    seen = []

    def fake_discover(api_key, api_url, native_catalog_provider, has_explicit_models, **kwargs):
        seen.append(bool(kwargs.get("interactive_probe", kwargs.get("for_picker"))))
        return None, False

    monkeypatch.setattr("hermes_cli.model_switch_providers._discover_endpoint_models", fake_discover)
    return seen


def test_options_payload_keeps_standard_probe_budget(picker_home, monkeypatch):
    """A normal options open declares picker visibility (``for_picker``) but must keep the
    standard endpoint-probe budget — only the interactive CLI picker gets the short one."""
    from hermes_cli.inventory import ConfigContext, build_model_options_payload

    seen = _probe_budget_seen(picker_home, monkeypatch)
    ctx = ConfigContext("", "", "", {"local-gw": {"base_url": "https://gw.example.invalid/v1",
                                                "models": ["m"]}}, [])
    build_model_options_payload(ctx)
    assert seen, "the current custom endpoint probe must run"
    assert seen == [False], seen


def test_cli_picker_uses_short_probe_budget_and_refresh_keeps_long_one(picker_home, monkeypatch):
    """The call-chain contract in both directions: the interactive picker (``for_picker``,
    no override) drives the probe with the short budget; an explicit refresh never shortens it."""
    from hermes_cli.inventory import ConfigContext, build_model_options_payload, build_models_payload

    ctx = ConfigContext("", "", "", {"local-gw": {"base_url": "https://gw.example.invalid/v1",
                                                "models": ["m"]}}, [])
    seen = _probe_budget_seen(picker_home, monkeypatch)
    build_models_payload(ctx, for_picker=True, probe_custom_providers=False,
                         probe_current_custom_provider=True)
    assert seen and seen[-1] is True, seen
    build_model_options_payload(ctx, refresh=True)
    assert seen[-1] is False, seen


def test_timed_out_probe_falls_back_to_declared_models(picker_home, monkeypatch):
    """A slow endpoint (probe raises/returns nothing) must not take the row down: the picker
    falls back to the models the entry declares. The failure is synthesized — no real sleep,
    no real request."""
    from hermes_cli.inventory import ConfigContext, build_models_payload

    attempted = []

    def slow_probe(*args, **kwargs):
        attempted.append(True)
        raise TimeoutError("synthetic slow endpoint")

    monkeypatch.setattr("hermes_cli.model_switch_providers._fetch_picker_live_models", slow_probe)
    # An inline key forces the live-probe branch (a keyless entry with declared models skips it).
    entry = {"base_url": "https://slow.example.invalid/v1", "api_key": "test-slow-probe-key",
             "models": ["declared-model"]}
    ctx = ConfigContext("", "", "", {"local-gw": entry}, [])
    rows = {r["slug"]: r for r in build_models_payload(
        ctx, for_picker=True, picker_hints=True, probe_custom_providers=True)["providers"]}
    assert attempted, "the live probe must actually have been attempted"
    assert rows["local-gw"]["models"] == ["declared-model"]
