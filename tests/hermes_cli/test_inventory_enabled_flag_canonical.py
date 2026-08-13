"""Tests for ``attach_provider_enabled_flags`` reading the canonical
``providers.<name>.enabled`` schema.

PR #74297 review: the Provider Manager's enablement must be driven by the
canonical keyed ``providers:`` schema (``providers.<name>.enabled``), the same
source the runtime resolver and model picker honour via ``is_provider_enabled``
— NOT the legacy ``custom_providers[].enabled`` flag or ``model.disabled_providers``.
A catalog row sourced from ``providers.<name>`` carries ``slug == <name>``, so the
annotation matches rows to their config block by slug.
"""

from hermes_cli.inventory import attach_provider_enabled_flags


def _payload(*slugs: str) -> dict:
    return {"providers": [{"slug": s, "name": s, "models": []} for s in slugs]}


def test_enabled_false_from_canonical_providers():
    """A ``providers.<name>.enabled: false`` entry marks the matching row disabled."""
    cfg = {"providers": {"demo": {"base_url": "http://x/v1", "enabled": False}}}

    out = attach_provider_enabled_flags(_payload("demo"), cfg)

    assert out["providers"][0]["enabled"] is False


def test_enabled_default_true_when_key_absent():
    """A ``providers`` entry without an ``enabled`` key is enabled by default."""
    cfg = {"providers": {"demo": {"base_url": "http://x/v1"}}}

    out = attach_provider_enabled_flags(_payload("demo"), cfg)

    assert out["providers"][0]["enabled"] is True


def test_enabled_string_false_is_disabled():
    """YAML-quoted ``enabled: "false"`` is honoured (is_provider_enabled coercion)."""
    cfg = {"providers": {"demo": {"base_url": "http://x/v1", "enabled": "false"}}}

    out = attach_provider_enabled_flags(_payload("demo"), cfg)

    assert out["providers"][0]["enabled"] is False


def test_enabled_explicit_true():
    cfg = {"providers": {"demo": {"base_url": "http://x/v1", "enabled": True}}}

    out = attach_provider_enabled_flags(_payload("demo"), cfg)

    assert out["providers"][0]["enabled"] is True


def test_builtin_row_without_providers_entry_defaults_enabled():
    """A row with no matching ``providers`` block (built-in) is enabled."""
    cfg = {"providers": {"demo": {"base_url": "http://x/v1", "enabled": False}}}

    out = attach_provider_enabled_flags(_payload("openai", "demo"), cfg)

    by_slug = {r["slug"]: r for r in out["providers"]}
    assert by_slug["openai"]["enabled"] is True
    assert by_slug["demo"]["enabled"] is False


def test_legacy_custom_providers_flag_is_not_used():
    """The legacy ``custom_providers[].enabled`` flag must NOT drive the flag.

    A legacy entry naming ``qux`` with ``enabled: false`` but NO ``providers``
    block must not force the ``custom:qux`` row disabled — enablement is
    canonical-only now.
    """
    cfg = {"custom_providers": [{"name": "qux", "enabled": False}]}

    out = attach_provider_enabled_flags(_payload("custom:qux"), cfg)

    assert out["providers"][0]["enabled"] is True


def test_disabled_providers_list_is_not_used():
    """``model.disabled_providers`` is not the supported control path anymore."""
    cfg = {
        "model": {"disabled_providers": ["openai"]},
        "providers": {},
    }

    out = attach_provider_enabled_flags(_payload("openai"), cfg)

    assert out["providers"][0]["enabled"] is True


def test_non_dict_providers_is_safe():
    """A malformed ``providers`` value does not raise; rows default to enabled."""
    cfg = {"providers": ["not", "a", "dict"]}

    out = attach_provider_enabled_flags(_payload("openai"), cfg)

    assert out["providers"][0]["enabled"] is True


def test_returns_same_payload_object():
    """The function mutates + returns the payload for call-site convenience."""
    payload = _payload("demo")
    cfg = {"providers": {"demo": {"enabled": False}}}

    out = attach_provider_enabled_flags(payload, cfg)

    assert out is payload
