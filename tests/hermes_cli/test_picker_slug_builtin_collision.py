"""A ``providers:`` key that collides with a built-in provider id must not be offered bare.

``runtime_provider_custom._shadowed_by_builtin`` deliberately defers a raw name that IS a
canonical built-in id, so the picker row's ``slug`` — documented as "the ``--provider`` value" —
persisted that bare key and the first turn resolved to the BUILT-IN provider, failing on its own
missing credential (#150823). The ``custom:`` spelling is exempt there and is one of the entry's
own aliases, so a shadowed key is offered with it; a key no built-in claims keeps the bare
spelling clients already store.

The same shadowing also swallowed a base_url override: ``providers.lmstudio: {base_url: …}``
resolved to the built-in's OWN endpoint, so its section-3 picker row (the one
``test_lmstudio_providers_block_with_explicit_endpoint_still_uses_section3`` pins) is now offered
as ``custom:lmstudio`` — that test's subject, "the row comes from section 3, not the built-in live
probe", is unchanged.
"""

import hermes_cli.providers as providers_mod
from hermes_cli.model_switch import list_authenticated_providers
from hermes_cli.runtime_provider import resolve_runtime_provider

COLLIDING_KEY = "ai-gateway"  # a real built-in provider id in hermes_cli.auth.PROVIDER_REGISTRY
ENDPOINT = "https://gateway.example.com/v1"


def _rows(monkeypatch, user_providers, *, current_provider="openrouter"):
    """Picker rows for *user_providers*, with every live catalog probe stubbed out."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", dict)
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", lambda *_a, **_kw: [])
    monkeypatch.setattr("hermes_cli.models.cached_fetch_api_models", lambda *_a, **_kw: None)
    return list_authenticated_providers(
        current_provider=current_provider, current_base_url="", current_model="gpt-4o",
        user_providers=user_providers, custom_providers=[], max_models=50,
        probe_custom_providers=False)


def _entry():
    return {"name": "My Gateway", "api": ENDPOINT, "key_cmd": "echo sk-test"}


def _row(rows, name):
    return next(p for p in rows if p.get("name") == name)


def test_colliding_key_is_offered_as_a_custom_identity(monkeypatch):
    rows = _rows(monkeypatch, {COLLIDING_KEY: _entry()})

    # The bare key IS the built-in provider id; offering it makes every client persist a
    # provider whose runtime resolution skips the user's entry entirely.
    assert _row(rows, "My Gateway")["slug"] == f"custom:{COLLIDING_KEY}"


def test_colliding_row_slug_reaches_the_users_endpoint(monkeypatch):
    config = {"providers": {COLLIDING_KEY: _entry()}}
    monkeypatch.setattr("hermes_cli.runtime_provider.load_config", lambda *_a, **_kw: config)
    rows = _rows(monkeypatch, config["providers"])
    slug = _row(rows, "My Gateway")["slug"]

    runtime = resolve_runtime_provider(requested=slug, target_model="m1")

    assert runtime["provider"] == "custom"
    assert str(runtime["base_url"]).rstrip("/") == ENDPOINT


def test_base_url_override_of_a_builtin_row_reaches_that_endpoint(monkeypatch):
    """A shadowed entry that only overrides ``base_url`` used to resolve to the built-in's own URL."""
    override = "http://remote-box:1234/v1"
    config = {"providers": {"lmstudio": {"base_url": override, "discover_models": True}}}
    monkeypatch.setattr("hermes_cli.runtime_provider.load_config", lambda *_a, **_kw: config)
    monkeypatch.setattr("hermes_cli.model_switch_providers._discover_endpoint_models",
                        lambda *_a, **_kw: (["remote-model"], False))
    rows = _rows(monkeypatch, config["providers"])
    row = next(p for p in rows if p.get("is_user_defined"))

    assert row["slug"] == "custom:lmstudio"
    assert resolve_runtime_provider(requested=row["slug"], target_model="m1")["base_url"].rstrip("/") == override


def test_non_colliding_key_keeps_the_bare_slug(monkeypatch):
    """The fix is scoped to names a built-in claims: every other key keeps the stored spelling."""
    rows = _rows(monkeypatch, {"my-router": {"name": "My Router", "api": ENDPOINT,
                                             "key_cmd": "echo sk-test"}})

    assert _row(rows, "My Router")["slug"] == "my-router"


def test_stale_bare_builtin_session_does_not_mark_the_custom_row_current(monkeypatch):
    """A session that stored the bare colliding key still routes to the BUILT-IN, not this row.

    ``providers:`` entries used to be persisted under the bare key, so a session saved before this
    fix carries ``model.provider: ai-gateway``; the resolver defers that spelling to the built-in,
    so the ``custom:ai-gateway`` row must not claim to be the current identity.
    """
    rows = _rows(monkeypatch, {COLLIDING_KEY: _entry()}, current_provider=COLLIDING_KEY)
    row = _row(rows, "My Gateway")

    assert row["slug"] == f"custom:{COLLIDING_KEY}"
    assert row["is_current"] is False


def test_grouped_row_drops_a_shadowed_member_alias_from_is_current(monkeypatch):
    """A shadowed key that only joined an existing row's group is not that row's identity either."""
    rows = _rows(monkeypatch, {
        "my-router": {"name": "My Router", "api": ENDPOINT, "api_key": "sk-test"},
        COLLIDING_KEY: {"name": "My Gateway", "api": ENDPOINT, "api_key": "sk-test"},
    }, current_provider=COLLIDING_KEY)

    assert _row(rows, "My Router")["slug"] == "my-router"
    assert _row(rows, "My Router")["is_current"] is False
