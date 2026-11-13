"""Inheritance pins selection identity while leaving route credentials as references."""
from copy import deepcopy
from pathlib import Path
from threading import Thread
from typing import Any
from unittest.mock import Mock

import hermes_yaml
import pytest

from hermes_agent.hermes_cli import config, models_validate, profiles
from hermes_agent.hermes_cli.config_backend import CONFIG_MUTATION_LOCK
from hermes_agent.hermes_cli.profile_provision import (
    ProfileCreateRequest,
    inherit_launch_model,
    provision_profile,
)
from hermes_constants import get_hermes_home
from hermes_agent.tui_gateway.launch_profile_policy import profile_config_scope


@pytest.fixture
def launch(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("SEED_PROVIDER", "custom:lab")
    monkeypatch.setenv("SEED_MODEL", "private-model")
    monkeypatch.setenv("SEED_URL", "http://127.0.0.1:18080/v1")
    monkeypatch.setenv("SEED_TOKEN", "synthetic-only")
    monkeypatch.setattr(profiles, "seed_profile_skills", lambda *a, **k: {"copied": []})
    return home


@pytest.fixture
def validation(monkeypatch):
    validate = Mock(wraps=models_validate.validate_requested_model)
    probe = Mock(return_value={"models": ["private-model"]})
    monkeypatch.setattr(models_validate, "validate_requested_model", validate)
    monkeypatch.setattr("hermes_agent.hermes_cli.models.probe_api_models", probe)
    monkeypatch.setattr("hermes_agent.hermes_cli.model_switch.get_model_info", lambda *a, **k: None)
    monkeypatch.setattr("hermes_agent.hermes_cli.model_switch.get_model_capabilities", lambda *a, **k: None)
    return validate, probe


@pytest.mark.parametrize("schema", ["bare", "providers", "custom_providers", "builtin"])
def test_inherit_copies_raw_route_without_validation(launch, monkeypatch, validation, schema):
    # A named launch scope must win over the process/default profile's config.
    config.atomic_config_write(launch / "config.yaml", {"model": {"provider": "nous", "default": "root"}})
    source = launch / "profiles" / "source"
    source.mkdir(parents=True)
    (source / ".env").write_text(
        "SEED_PROVIDER=custom:lab\nSEED_MODEL=private-model\n"
        "SEED_URL=http://127.0.0.1:18080/v1\nSEED_TOKEN=synthetic-only\n"
    )
    model = {
        "provider": "custom" if schema == "bare" else "${SEED_PROVIDER}",
        "default": "${SEED_MODEL}",
        "base_url": "${SEED_URL}",
        "api_key": "${SEED_TOKEN}",
        "api_mode": "anthropic_messages",
        "context_length": 12345,
        "custom_model": {"extra_body": {"opaque": True}},
    }
    endpoint = {"name": "Lab", "base_url": "${SEED_URL}", "api_key": "${SEED_TOKEN}",
                "models": {"private-model": {"context_length": 12345}}}
    raw: dict = {"model": model}
    expected: dict = {"model": model}
    retained = {"base_url": "http://127.0.0.1:18081/v1"}
    destination: dict = {"tts": {"provider": "edge"}, "providers": {"retained": retained}}
    if schema == "providers":
        raw["providers"] = {"lab": endpoint, "neighbor": retained}
        expected["providers"] = {"lab": endpoint, "retained": retained}
        destination["custom_providers"] = [{"name": "Lab", "base_url": "http://stale.invalid/v1"}]
        expected["custom_providers"] = []
    elif schema == "custom_providers":
        raw[schema] = [endpoint, {"name": "Neighbor", **retained}]
        expected[schema] = [endpoint]
        destination["providers"]["stale"] = {"name": "Lab", "base_url": "http://stale.invalid/v1"}
    elif schema == "builtin":
        model["provider"] = "openrouter"
        raw["providers"] = {"openrouter": {"request_timeout_seconds": 150}, "neighbor": retained}
        expected["providers"] = {"openrouter": raw["providers"]["openrouter"], "retained": retained}
    config.atomic_config_write(source / "config.yaml", raw)
    target = launch / "profiles" / "target"
    target.mkdir()
    config.atomic_config_write(target / "config.yaml", destination)
    before = {p: p.read_bytes() for p in (source / "config.yaml", launch / "config.yaml")}
    real_save = config.save_config
    saves = []

    def save(cfg, **kwargs):
        assert get_hermes_home() == target
        acquired = []

        def contend():
            locked = CONFIG_MUTATION_LOCK.acquire(blocking=False)
            acquired.append(locked)
            if locked:
                CONFIG_MUTATION_LOCK.release()

        contender = Thread(target=contend)
        contender.start()
        contender.join(timeout=2)
        assert acquired == [False]
        saves.append(cfg)
        return real_save(cfg, **kwargs)

    monkeypatch.setattr(config, "save_config", save)
    with profile_config_scope(source):
        cached = config.load_config_readonly()
        snapshot = deepcopy(cached)
        assert inherit_launch_model(target) is True
        assert get_hermes_home() == source
        assert cached == snapshot
    assert get_hermes_home() == launch
    assert {p: p.read_bytes() for p in before} == before
    actual = config.read_user_config_raw(target / "config.yaml")
    carried = {k: v for k, v in model.items() if k not in {"api_key", "context_length"}}
    assert actual["model"] == {**carried, "provider": model["provider"] if schema in {"bare", "builtin"} else "custom:lab",
                               "default": "private-model"}
    assert actual["tts"] == destination["tts"]
    assert actual["providers"] == expected.get("providers", {"retained": retained})
    if "custom_providers" in expected:
        assert actual["custom_providers"] == expected["custom_providers"]
    assert len(saves) == 1
    validate, probe = validation
    validate.assert_not_called()
    probe.assert_not_called()
    pinned = (target / "config.yaml").read_bytes()
    with profile_config_scope(source):
        assert inherit_launch_model(target) is False
    assert (target / "config.yaml").read_bytes() == pinned


@pytest.mark.parametrize("legacy", [False, "default", "name"])
def test_inherit_source_selection_destination_secret_refs(launch, validation, legacy):
    from hermes_agent.hermes_cli.config_effective import load_user_config_effective
    from hermes_agent.hermes_cli.runtime_provider import resolve_runtime_provider

    source = launch / "profiles" / "source"
    target = launch / "profiles" / "target"
    source.mkdir(parents=True)
    target.mkdir()
    for home, selector, model, port in ((source, "lab", "source-model", 18080),
                                       (target, "other", "destination-model", 18081)):
        (home / ".env").write_text(
            f"SELECTOR_PROVIDER=custom:{selector}\nSELECTOR_MODEL={model}\n"
            f"ROUTE_URL=http://127.0.0.1:{port}/v1\nROUTE_KEY=synthetic-{selector}\n")
    model = {"default": "${SELECTOR_MODEL}", "provider": "${SELECTOR_PROVIDER}",
             "base_url": "${ROUTE_URL}", "api_key": "${ROUTE_KEY}"}
    raw: dict[str, Any] = {
        "model": model, "providers": {"lab": {"base_url": "${ROUTE_URL}", "api_key": "${ROUTE_KEY}"}}}
    if legacy:
        raw.update(provider=model.pop("provider"), base_url=model.pop("base_url"), context_length=12345)
        model[legacy] = model.pop("default")
    config.atomic_config_write(source / "config.yaml", raw)
    # The writer canonicalizes too: use the raw backend fixture boundary for legacy input.
    if legacy:
        (source / "config.yaml").write_text(hermes_yaml.safe_dump(raw))
    config.atomic_config_write(target / "config.yaml", {"model": {"temperature": 0.3, "custom_model": {"extra_body": {"keep": True}}}})
    before = (source / "config.yaml").read_bytes()
    with profile_config_scope(source):
        assert inherit_launch_model(target)
        assert get_hermes_home() == source
    inherited = config.read_user_config_raw(target / "config.yaml")
    assert inherited["model"]["provider"] == "custom:lab"
    assert inherited["model"]["default"] == "source-model"
    assert inherited["model"]["base_url"] == "${ROUTE_URL}"
    # The destination reaches another endpoint: the source's model-level key stays behind and
    # the endpoint entry's own ref supplies the destination key.
    assert "api_key" not in inherited["model"]
    assert inherited["model"]["temperature"] == 0.3
    assert inherited["model"]["custom_model"] == {"extra_body": {"keep": True}}
    assert "lab" in inherited["providers"]
    with profile_config_scope(target):
        effective = load_user_config_effective()
        assert effective["model"]["default"] == "source-model"
        runtime = resolve_runtime_provider(requested=effective["model"]["provider"])
        assert runtime["provider"] == "custom"
        assert runtime["base_url"] == "http://127.0.0.1:18081/v1"
        assert runtime["api_key"] == "synthetic-other"
    assert (source / "config.yaml").read_bytes() == before
    validation[0].assert_not_called()
    validation[1].assert_not_called()


@pytest.mark.parametrize("layout", ["root", "api_base", "complete"])
def test_inherit_destination_legacy_route(launch, validation, layout):
    from hermes_agent.hermes_cli.config_effective import load_user_config_effective
    from hermes_agent.hermes_cli.runtime_provider import resolve_runtime_provider

    config.atomic_config_write(launch / "config.yaml", {
        "model": {"provider": "openrouter", "default": "source-model"}})
    target = launch / "profiles" / "target"
    target.mkdir(parents=True)
    (target / ".env").write_text("OPENROUTER_API_KEY=synthetic-destination\n")
    model: dict[str, Any] = {"temperature": 0.3}
    destination: dict[str, Any] = {"model": model, "context_length": 9999,
                   "base_url": "http://stale.invalid/v1", "tts": {"provider": "edge"}}
    if layout == "api_base":
        model["api_base"] = destination.pop("base_url")
    if layout == "complete":
        destination["provider"] = "custom"
        model["name"] = "keep-model"
    (target / "config.yaml").write_text(hermes_yaml.safe_dump(destination))
    before = (target / "config.yaml").read_bytes()
    assert inherit_launch_model(target) is (layout != "complete")
    if layout == "complete":
        assert (target / "config.yaml").read_bytes() == before
    else:
        raw = config.read_user_config_raw(target / "config.yaml")
        assert raw["tts"] == destination["tts"]
        assert raw["model"]["temperature"] == 0.3
        with profile_config_scope(target):
            effective = load_user_config_effective()
            assert not effective["model"].get("context_length")
            runtime = resolve_runtime_provider(requested=effective["model"]["provider"])
            assert runtime["base_url"] == "https://openrouter.ai/api/v1"
            assert runtime["api_key"] == "synthetic-destination"
    validation[0].assert_not_called()
    validation[1].assert_not_called()


@pytest.mark.parametrize("schema", ["providers", "custom_providers"])
@pytest.mark.parametrize("identity", ["literal", "name", "provider_key"])
def test_inherit_endpoint_identity_scopes(launch, validation, schema, identity):
    from hermes_agent.hermes_cli.config_effective import load_user_config_effective
    from hermes_agent.hermes_cli.runtime_provider import resolve_runtime_provider

    source = launch / "profiles" / "source"
    target = launch / "profiles" / "target"
    source.mkdir(parents=True)
    target.mkdir()
    for home, alias, enabled, port in ((source, "Lab", "true", 18080), (target, "Other", "false", 18081)):
        (home / ".env").write_text(
            f"ALIAS={alias}\nENABLED={enabled}\nSELECTOR=custom:{alias.lower()}\n"
            f"ROUTE_URL=http://127.0.0.1:{port}/v1\nROUTE_KEY=synthetic-{alias}\n"
            f"COLLISION={'Other' if home == source else 'Lab'}\n")
    endpoint = {"name": "Lab", "base_url": "${ROUTE_URL}", "api_key": "${ROUTE_KEY}"}
    if identity != "literal":
        endpoint[identity] = "${ALIAS}"
        if identity == "provider_key" and schema == "providers":
            endpoint["name"] = "Display"
    if schema == "providers":
        # Keyed entries route by key/name, not an embedded provider_key.
        source_key = "lab" if identity == "provider_key" else "stable"
        seed = {schema: {source_key: {**endpoint, "enabled": "${ENABLED}"}}}
        collisions = {"custom_providers": [{"name": "${COLLISION}", "base_url": "http://stale.invalid/v1"}]}
    else:
        seed = {schema: [endpoint]}
        collisions = {"providers": {"stale": {"name": "${COLLISION}", "base_url": "http://stale.invalid/v1",
                                              "api_key": "synthetic-stale"}}}
    config.atomic_config_write(source / "config.yaml", {
        "model": {"provider": "${SELECTOR}", "default": "source-model"}, **seed})
    config.atomic_config_write(target / "config.yaml", {**collisions, "tts": {"provider": "edge"}})
    before = (source / "config.yaml").read_bytes()
    with profile_config_scope(source):
        cached = config.load_config_readonly()
        snapshot = deepcopy(cached)
        assert inherit_launch_model(target)
        assert get_hermes_home() == source
        assert cached == snapshot
    raw = config.read_user_config_raw(target / "config.yaml")
    copied = next(iter(raw[schema].values())) if schema == "providers" else raw[schema][0]
    assert copied["base_url"] == "${ROUTE_URL}"
    assert copied["api_key"] == "${ROUTE_KEY}"
    if schema == "providers":
        assert raw["custom_providers"] == []
    else:
        assert "stale" not in raw.get("providers", {})
    with profile_config_scope(target):
        effective = load_user_config_effective()
        assert effective["model"]["default"] == "source-model"
        runtime = resolve_runtime_provider(requested=effective["model"]["provider"])
        assert runtime["base_url"] == "http://127.0.0.1:18081/v1"
        assert runtime["api_key"] == "synthetic-Other"
    assert (source / "config.yaml").read_bytes() == before
    assert get_hermes_home() == launch
    validation[0].assert_not_called()
    validation[1].assert_not_called()


@pytest.mark.parametrize("locked", [True, False])
def test_inherit_keeps_remote_route_refusal(launch, monkeypatch, locked):
    from hermes_agent.hermes_cli import config_backend

    config.atomic_config_write(launch / "config.yaml", {"model": {"provider": "custom", "default": "source",
                                                               "base_url": "${SEED_URL}"}})
    target = launch / "profiles" / "target"
    target.mkdir(parents=True)
    config.atomic_config_write(target / "config.yaml", {"model": {"api_key": "${OLD_KEY}", "key_env": "OLD_KEY"}})
    before = (target / "config.yaml").read_bytes()
    backend = config_backend.get_config_backend()
    monkeypatch.setattr(config_backend, "supports_file_tooling", lambda: False)
    monkeypatch.setattr(backend, "locked", lambda home, key: "tenant" if locked and key == "model.base_url" else None)
    saves = []

    def save(cfg):
        assert get_hermes_home() == target
        assert {"model", "providers", "custom_providers"} <= set(config_backend.refused_locked_omission_prefixes())
        assert cfg["model"]["api_key"] is None
        assert cfg["model"]["key_env"] is None
        saves.append(cfg)

    monkeypatch.setattr(config, "save_config", save)
    if locked:
        with pytest.raises(config_backend.ConfigLockedError):
            inherit_launch_model(target)
        assert saves == []
    else:
        assert inherit_launch_model(target)
        assert len(saves) == 1
    assert get_hermes_home() == launch
    assert config_backend.refused_locked_omission_prefixes() == ()
    assert (target / "config.yaml").read_bytes() == before


@pytest.mark.parametrize("kind", ["fresh", "clone", "clone-all", "clone-unpinned", "explicit", "rejected"])
def test_provision_only_validates_explicit_selections(launch, validation, kind, monkeypatch):
    raw = {"model": {"provider": "custom:lab", "default": "private-model"},
           "providers": {"lab": {"base_url": "http://127.0.0.1:18080/v1", "api_key": "synthetic-only"}}}
    config.atomic_config_write(launch / "config.yaml", raw)
    empty = launch / "profiles" / "empty"
    empty.mkdir(parents=True)
    config.atomic_config_write(empty / "config.yaml", {"tts": {"provider": "edge"}})
    before = (launch / "config.yaml").read_bytes()
    request = {
        "fresh": ProfileCreateRequest(name="target", alias=False),
        "clone": ProfileCreateRequest(name="target", alias=False, clone_config=True),
        "clone-all": ProfileCreateRequest(name="target", alias=False, clone_all=True),
        "clone-unpinned": ProfileCreateRequest(name="target", alias=False, clone_config=True, clone_from="empty"),
        "explicit": ProfileCreateRequest(name="target", alias=False, provider="custom:lab", model="private-model"),
        "rejected": ProfileCreateRequest(name="target", alias=False, provider="custom:lab", model="private-model"),
    }[kind]
    validate, probe = validation
    rejection = Mock(return_value={
        "accepted": False, "persist": False, "recognized": False, "message": "test rejection"})
    if kind == "rejected":
        monkeypatch.setattr(models_validate, "validate_requested_model", rejection)
    result = provision_profile(request)
    assert result.warnings == []
    assert (launch / "config.yaml").read_bytes() == before
    assert get_hermes_home() == launch
    if kind == "rejected":
        rejection.assert_called_once()
        assert result.model_set is False
        assert result.model_error == "test rejection"
        assert config.read_user_config_raw(result.path / "config.yaml")["model"] == raw["model"]
    elif kind == "explicit":
        validate.assert_called_once()
        probe.assert_called_once()
        assert result.model_set is True
    else:
        validate.assert_not_called()
        probe.assert_not_called()
        assert config.read_user_config_raw(result.path / "config.yaml")["model"] == raw["model"]
        assert result.mirrored["model_inherited"] is (kind == "clone-unpinned")


@pytest.mark.parametrize("route", ["same-url", "other-url", "environment"])
def test_inherit_leaves_endpoint_bound_keys_behind(launch, monkeypatch, validation, route):
    # The new profile's environment decides where the route goes, so the source's literal key and
    # context pin never travel with it, whichever endpoint it reaches.
    source = launch / "profiles" / "source"
    target = launch / "profiles" / "target"
    source.mkdir(parents=True)
    target.mkdir()
    url = {"same-url": "http://127.0.0.1:18080/v1", "other-url": "https://other-host.invalid/v1"}.get(route)
    model: dict[str, Any] = {"provider": "custom", "default": "m1", "api_key": "synthetic-source-literal",
                             "key_env": "SOURCE_KEY", "context_length": 12345, "temperature": 0.2}
    if url:
        model["base_url"] = "${ROUTE_URL}"
        (source / ".env").write_text("ROUTE_URL=http://127.0.0.1:18080/v1\n")
        (target / ".env").write_text(f"ROUTE_URL={url}\n")
    else:
        monkeypatch.setenv("CUSTOM_BASE_URL", "https://a.invalid/v1")
        (target / ".env").write_text("CUSTOM_BASE_URL=https://b.invalid/v1\n")
    config.atomic_config_write(source / "config.yaml", {"model": model})
    config.atomic_config_write(target / "config.yaml", {"tts": {"provider": "edge"}})
    with profile_config_scope(source):
        assert inherit_launch_model(target)
    inherited = config.read_user_config_raw(target / "config.yaml")["model"]
    assert inherited == {"provider": "custom", "default": "m1", "temperature": 0.2,
                         **({"base_url": "${ROUTE_URL}"} if url else {})}
    validation[0].assert_not_called()


def test_inherit_normalizes_the_model_for_its_provider(launch, validation):
    config.atomic_config_write(launch / "config.yaml", {
        "model": {"provider": "openrouter", "default": "claude-sonnet-4.6"}})
    target = launch / "profiles" / "target"
    target.mkdir(parents=True)
    assert inherit_launch_model(target)
    model = config.read_user_config_raw(target / "config.yaml")["model"]
    assert (model["provider"], model["default"]) == ("openrouter", "anthropic/claude-sonnet-4.6")
    validation[0].assert_not_called()
    validation[1].assert_not_called()


def test_inherit_keeps_a_bare_custom_provider(launch, validation):
    # Provider recovery would pick the first declared endpoint for bare ``custom``.
    config.atomic_config_write(launch / "config.yaml", {
        "model": {"provider": "custom", "default": "m1", "base_url": "https://a.invalid/v1"},
        "providers": {"other": {"name": "Other", "base_url": "https://b.invalid/v1"}}})
    target = launch / "profiles" / "target"
    target.mkdir(parents=True)
    assert inherit_launch_model(target)
    model = config.read_user_config_raw(target / "config.yaml")["model"]
    assert (model["provider"], model["base_url"]) == ("custom", "https://a.invalid/v1")


@pytest.mark.parametrize("dest_url", ["http://127.0.0.1:18080/v1", "http://127.0.0.1:18081/v1", ""])
def test_inherit_keeps_destination_keys_for_the_same_endpoint(launch, validation, dest_url):
    # Routes compare as the new profile resolves them, not by how each file spells the URL; two
    # empty expansions prove nothing, so the keys are cleared.
    source = launch / "profiles" / "source"
    target = launch / "profiles" / "target"
    source.mkdir(parents=True)
    target.mkdir()
    (source / ".env").write_text("SOURCE_URL=http://127.0.0.1:18080/v1\n")
    source_url = "http://127.0.0.1:18080/v1" if dest_url else ""
    (target / ".env").write_text(f"SOURCE_URL={source_url}\nDEST_URL={dest_url}\n")
    config.atomic_config_write(source / "config.yaml", {
        "model": {"provider": "custom", "default": "m1", "base_url": "${SOURCE_URL}"}})
    config.atomic_config_write(target / "config.yaml", {
        "model": {"provider": "custom", "base_url": "${DEST_URL}", "key_env": "DEST_KEY"}})
    with profile_config_scope(source):
        assert inherit_launch_model(target)
    model = config.read_user_config_raw(target / "config.yaml")["model"]
    assert model["base_url"] == "${SOURCE_URL}"
    assert ("key_env" in model) is dest_url.endswith(":18080/v1")


def test_inherit_clears_destination_keys_when_the_endpoint_entry_is_replaced(launch, validation):
    # Neither model block names a URL: the endpoint comes from the replaced ``providers.lab``.
    config.atomic_config_write(launch / "config.yaml", {
        "model": {"provider": "custom:lab", "default": "m1"},
        "providers": {"lab": {"name": "Lab", "base_url": "https://new.invalid/v1"}}})
    target = launch / "profiles" / "target"
    target.mkdir(parents=True)
    config.atomic_config_write(target / "config.yaml", {
        "model": {"provider": "custom:lab", "key_env": "OLD_KEY", "temperature": 0.3},
        "providers": {"lab": {"name": "Lab", "base_url": "https://old.invalid/v1"}}})
    assert inherit_launch_model(target)
    raw = config.read_user_config_raw(target / "config.yaml")
    assert raw["providers"]["lab"]["base_url"] == "https://new.invalid/v1"
    assert raw["model"] == {"provider": "custom:lab", "default": "m1", "temperature": 0.3}
