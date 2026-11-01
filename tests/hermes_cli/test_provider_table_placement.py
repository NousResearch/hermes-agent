"""Plugin profiles place themselves in the core provider tables through generic fields.

``picker_after`` / ``picker_group`` / ``catalog_aliases`` / ``aggregator`` feed the picker catalog
(``sync_plugin_provider_catalog``); ``provider_overlay`` feeds ``hermes_cli.providers``;
``models_dev_id`` feeds the models.dev lookups at lookup time. Exercised with fake profiles only.
"""

import pytest

from providers.base import ProviderProfile


@pytest.fixture
def catalog(monkeypatch):
    """The picker tables as private copies, so admissions stay inside the test."""
    import hermes_cli.models_catalog_static as cat

    monkeypatch.setattr(cat, "CANONICAL_PROVIDERS", list(cat.CANONICAL_PROVIDERS))
    monkeypatch.setattr(cat, "_canonical_slugs", set(cat._canonical_slugs))
    monkeypatch.setattr(cat, "_PROVIDER_LABELS", dict(cat._PROVIDER_LABELS))
    monkeypatch.setattr(cat, "PROVIDER_GROUPS", {k: (lbl, d, list(m)) for k, (lbl, d, m) in cat.PROVIDER_GROUPS.items()})
    monkeypatch.setattr(cat, "_SLUG_TO_GROUP", dict(cat._SLUG_TO_GROUP))
    monkeypatch.setattr(cat, "_AGGREGATOR_PROVIDERS", set(cat._AGGREGATOR_PROVIDERS))
    return cat


def _use_profiles(monkeypatch, *profiles):
    import providers

    monkeypatch.setattr(providers, "list_providers", lambda: list(profiles))
    # The new profile list stands for a registration, which moves the generation.
    monkeypatch.setattr(providers, "_REGISTRY_GENERATION", providers._REGISTRY_GENERATION + 1)


def test_picker_fields_place_plugin_rows(catalog, monkeypatch):
    group = ("zz-vendor", "ZZ Vendor", "API or local agent")
    # Registered child-first: the chained anchor must still land the child right after its parent.
    child = ProviderProfile(name="zz-agent", display_name="ZZ Agent", picker_after="zz-api", picker_group=group)
    parent = ProviderProfile(
        name="zz-api", aliases=("zz", "zz-cloud"), display_name="ZZ API", description="ZZ API (direct)",
        picker_after="nvidia", picker_group=group, catalog_aliases=True, aggregator=True)
    _use_profiles(monkeypatch, child, parent)

    assert catalog.sync_plugin_provider_catalog() == 2

    slugs = [p.slug for p in catalog.CANONICAL_PROVIDERS]
    at = slugs.index("nvidia")
    assert slugs[at + 1:at + 3] == ["zz-api", "zz-agent"]
    assert catalog.CANONICAL_PROVIDERS[at + 1].label == "ZZ API"
    assert catalog.CANONICAL_PROVIDERS[at + 1].tui_desc == "ZZ API (direct)"
    row = next(r for r in catalog.group_providers(slugs) if r.get("group_id") == "zz-vendor")
    assert (row["label"], row["description"], row["members"]) == ("ZZ Vendor", "API or local agent", ["zz-api", "zz-agent"])
    assert catalog._PROVIDER_ALIASES["zz-cloud"] == "zz-api"
    assert "zz-api" in catalog._AGGREGATOR_PROVIDERS and "zz-agent" not in catalog._AGGREGATOR_PROVIDERS
    assert catalog.sync_plugin_provider_catalog() == 0  # idempotent


def test_unplaced_profile_keeps_generic_admission(catalog, monkeypatch):
    plain = ProviderProfile(name="zz-plain", aliases=("zzp",))
    _use_profiles(monkeypatch, plain)

    catalog.sync_plugin_provider_catalog()

    assert catalog.CANONICAL_PROVIDERS[-1].slug == "zz-plain"
    assert catalog.provider_group_for_slug("zz-plain") == ""
    assert "zzp" not in catalog._PROVIDER_ALIASES  # aliases stay out of the model catalog by default
    assert "zz-plain" not in catalog._AGGREGATOR_PROVIDERS


def test_provider_overlay_defines_the_models_dev_keyed_provider(monkeypatch):
    from hermes_cli import providers as prov

    overlaid = ProviderProfile(
        name="zz-acp", aliases=("zz-acp-agent", "zz-acp-cli"), display_name="ZZ ACP", base_url="acp://zz",
        auth_type="external_process", overlay_aliases=("zz-acp-agent",),
        provider_overlay={"transport": "codex_responses", "auth_type": "external_process",
                          "base_url_override": "acp://zz", "base_url_env_var": "ZZ_ACP_BASE_URL"})
    plain = ProviderProfile(name="zz-plain", aliases=("zzp",), base_url="https://zz.test/v1")
    _use_profiles(monkeypatch, overlaid, plain)
    monkeypatch.setattr(prov, "_models_dev_info", lambda canonical, allow_network=True: None)

    assert prov.normalize_provider("ZZ-ACP-Agent") == "zz-acp"
    assert prov.normalize_provider("zzp") == "zzp"  # no overlay: the providers layer leaves it alone
    assert prov.normalize_provider("zz-acp-cli") == "zz-acp-cli"  # only overlay_aliases name the overlay
    pdef = prov.get_provider("zz-acp-agent")
    assert (pdef.id, pdef.name, pdef.transport, pdef.base_url, pdef.base_url_env_var, pdef.source) == (
        "zz-acp", "ZZ ACP", "codex_responses", "acp://zz", "ZZ_ACP_BASE_URL", "hermes")
    assert prov.determine_api_mode("zz-acp") == "codex_responses"
    assert "zz-acp" in prov.hermes_overlays() and "zz-plain" not in prov.hermes_overlays()


def test_mirrored_alias_keys_keep_the_overlay_route(monkeypatch):
    """An overlay-declaring profile's own alias keys in the auth registry share its one row: resolving
    an alias goes through the overlay (transport, base-URL env var), not the lossy-collapse registry rung."""
    from types import SimpleNamespace

    from hermes_cli import auth
    from hermes_cli import providers as prov

    overlaid = ProviderProfile(
        name="zz-acp", aliases=("zz-acp-agent",), display_name="ZZ ACP", base_url="acp://zz",
        overlay_aliases=("zz-acp-agent",), provider_overlay={"transport": "codex_responses",
                                                             "base_url_override": "acp://zz"})
    _use_profiles(monkeypatch, overlaid)
    monkeypatch.setattr(prov, "_models_dev_info", lambda canonical, allow_network=True: None)
    row = SimpleNamespace(id="zz-acp", name="ZZ ACP", api_key_env_vars=(), inference_base_url="acp://zz")
    monkeypatch.setattr(auth, "PROVIDER_REGISTRY", {"zz-acp": row, "zz-acp-agent": row})

    pdef = prov.resolve_provider_full("zz-acp-agent")

    assert (pdef.id, pdef.transport, pdef.source) == ("zz-acp", "codex_responses", "hermes")


def test_profile_models_dev_id_resolves_at_lookup_time(monkeypatch):
    import agent.models_dev as md

    _use_profiles(monkeypatch, ProviderProfile(name="zz-hosted", aliases=("zzh",), models_dev_id="zz-catalog"))
    monkeypatch.setattr(md, "PROVIDER_TO_MODELS_DEV", {})
    monkeypatch.setattr(md, "_MODELS_DEV_TO_PROVIDER", None)

    assert md.models_dev_id_for("zz-hosted") == "zz-catalog"
    assert md.models_dev_id_for("zzh") is None  # exact Hermes ids only, like the static map
    assert md._models_dev_to_hermes_ids("zz-catalog") == ["zz-hosted"]
    assert md.models_dev_provider_map() == {"zz-hosted": "zz-catalog"}
    assert md.PROVIDER_TO_MODELS_DEV == {}  # never merged into the static dict


def test_models_dev_keyed_overlay_row_keeps_the_profile_slug_and_label_offline(monkeypatch):
    """A profile whose overlay is keyed by its models.dev id lists under its own name and display_name
    when no models.dev catalog is at hand (offline, blocked): not under the catalog id."""
    import agent.models_dev as md
    from hermes_cli import providers as prov
    from hermes_cli.model_switch import list_authenticated_providers

    hosted = ProviderProfile(
        name="zz-hosted", aliases=("zz-hosted-alias",), display_name="ZZ Hosted", models_dev_id="zz-catalog",
        overlay_aliases=("zzh",), provider_overlay={"extra_env_vars": ("ZZ_HOSTED_TOKEN",)})
    _use_profiles(monkeypatch, hosted)
    monkeypatch.setattr(md, "_MODELS_DEV_TO_PROVIDER", None)
    monkeypatch.setattr(md, "fetch_models_dev", lambda *a, **k: {})
    monkeypatch.setattr(prov, "_models_dev_info", lambda canonical, allow_network=True: None)
    monkeypatch.setenv("ZZ_HOSTED_TOKEN", "zz-test-token")

    assert prov.get_label("zz-hosted") == prov.get_label("zzh") == "ZZ Hosted"
    rows = [r for r in list_authenticated_providers(current_provider="zz-catalog") if "zz" in r["slug"]]
    assert [(r["slug"], r["name"], r["is_current"]) for r in rows] == [("zz-hosted", "ZZ Hosted", True)]
