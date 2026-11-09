"""Fresh profiles retain the selected custom endpoint, without copying its neighbors."""

from pathlib import Path

import pytest

from hermes_cli.config import (
    DEFAULT_CONFIG,
    atomic_config_write,
    get_compatible_custom_providers,
    read_user_config_raw,
)
from hermes_cli.profiles import create_profile
from hermes_cli.profile_model_seed import launch_model_seed, merge_endpoint_seed
from hermes_cli import runtime_provider


@pytest.fixture
def create_from_launch(tmp_path, monkeypatch):
    """Write *source* as the launch config, create a fresh profile, return its raw config."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    def _create(source: dict) -> dict:
        atomic_config_write(
            home / "config.yaml",
            {**source, "_config_version": DEFAULT_CONFIG["_config_version"]},
        )
        profile = create_profile("sample-bot", no_alias=True, no_skills=True)
        return read_user_config_raw(profile / "config.yaml")

    return _create


@pytest.mark.parametrize(
    "schema,provider,url_key",
    [
        ("providers", "lab", "base_url"),
        ("providers", "custom:lab", "baseUrl"),
        ("custom_providers", "custom:display-name", "base_url"),
    ],
)
def test_fresh_profile_retains_resolvable_custom_endpoint(
    create_from_launch, monkeypatch, schema, provider, url_key
):
    endpoint = {
        "name": "Display Name",
        url_key: "http://127.0.0.1:11434/v1",
        "api_key": "${LAB_TOKEN}",
    }
    unrelated = {"name": "Unrelated", "base_url": "http://127.0.0.1:11435/v1"}
    catalog = (
        {"lab": endpoint, "other": unrelated}
        if schema == "providers"
        else [endpoint, unrelated]
    )
    source = {"model": {"provider": provider, "default": "synthetic-model"}, schema: catalog}

    created = create_from_launch(source)

    assert created["model"] == source["model"]
    copied = get_compatible_custom_providers(created)
    assert [entry["base_url"] for entry in copied] == [endpoint[url_key]]
    assert copied[0]["api_key"] == "${LAB_TOKEN}"
    monkeypatch.setattr(runtime_provider, "load_config", lambda: created)
    resolved = runtime_provider._get_named_custom_provider(provider)
    assert resolved is not None
    assert resolved["base_url"] == endpoint[url_key]


@pytest.mark.parametrize(
    "schema,provider,copied",
    [
        ("providers", "openrouter", False),
        ("custom_providers", "custom:openrouter", True),
    ],
)
def test_builtin_name_collision_only_copies_explicit_custom_selection(
    create_from_launch, schema, provider, copied
):
    endpoint = {"name": "OpenRouter", "base_url": "http://127.0.0.1:11434/v1"}
    catalog = {"other-endpoint": endpoint} if schema == "providers" else [endpoint]

    created = create_from_launch(
        {"model": {"provider": provider, "default": "synthetic-model"}, schema: catalog}
    )

    assert bool(get_compatible_custom_providers(created)) is copied


@pytest.mark.parametrize(
    "model",
    [
        {"provider": "openrouter", "default": "synthetic-model"},
        {
            "provider": "custom",
            "default": "synthetic-model",
            "base_url": "http://127.0.0.1:11434/v1",
        },
    ],
)
def test_fresh_profile_keeps_exact_key_provider_tuning(create_from_launch, model):
    selected = {model["provider"]: {"request_timeout_seconds": 150}}
    neighbor = {"neighbor": {"request_timeout_seconds": 5}}

    created = create_from_launch({"model": model, "providers": {**selected, **neighbor}})

    assert created["providers"] == selected


@pytest.mark.parametrize(
    "providers,legacy",
    [
        (
            {"lab": {"request_timeout_seconds": 150}},
            [{"name": "Lab", "base_url": "http://127.0.0.1:11434/v1"}],
        ),
        (
            {
                "other": {"name": "Lab", "base_url": "http://127.0.0.1:11434/v1"},
                "lab": {"base_url": "http://127.0.0.1:11435/v1"},
            },
            [],
        ),
    ],
)
def test_model_seed_keeps_runtime_endpoint_selection(monkeypatch, providers, legacy):
    source = {
        "model": {"provider": "lab", "default": "synthetic-model"},
        "providers": providers,
        "custom_providers": legacy,
    }
    monkeypatch.setattr(runtime_provider, "load_config", lambda: source)
    selected = runtime_provider._get_named_custom_provider("lab")
    assert selected is not None

    seed = launch_model_seed(source)

    assert len(seed.get("providers", {})) + len(seed.get("custom_providers", [])) == 1
    monkeypatch.setattr(runtime_provider, "load_config", lambda: seed)
    inherited = runtime_provider._get_named_custom_provider("lab")
    assert inherited is not None
    assert inherited["base_url"] == selected["base_url"]


def test_model_seed_selects_env_referenced_provider(monkeypatch):
    monkeypatch.setenv("LAB_PROVIDER", "custom:lab")
    endpoint = {"base_url": "http://127.0.0.1:11434/v1"}
    source = {
        "model": {"provider": "${LAB_PROVIDER}", "default": "synthetic-model"},
        "providers": {"lab": endpoint, "other": {"base_url": "http://127.0.0.1:1/v1"}},
    }

    seed = launch_model_seed(source)

    assert seed == {"model": source["model"], "providers": {"lab": endpoint}}


@pytest.mark.parametrize(
    "schema,endpoint",
    [
        ("providers", {"name": "${LAB_NAME}", "base_url": "http://127.0.0.1:11434/v1"}),
        ("custom_providers", {"name": "${LAB_NAME}", "base_url": "http://127.0.0.1:11434/v1"}),
        ("custom_providers", {"name": "Lab", "base_url": "${LAB_URL}"}),
    ],
)
def test_model_seed_matches_env_referenced_endpoint_names(monkeypatch, schema, endpoint):
    monkeypatch.setenv("LAB_NAME", "Lab")
    monkeypatch.setenv("LAB_URL", "http://127.0.0.1:11434/v1")
    catalog = {"gateway": endpoint} if schema == "providers" else [endpoint]
    source = {"model": {"provider": "custom:lab", "default": "synthetic-model"}, schema: catalog}

    seed = launch_model_seed(source)

    assert seed == source
    dst = {"providers": {"stale": {"name": "lab", "base_url": "http://127.0.0.1:1/v1"}}}
    assert "stale" not in merge_endpoint_seed(dst, seed).get("providers", {})


def test_model_seed_uses_supplied_config_without_reading_credentials(monkeypatch):
    import hermes_cli.runtime_provider_custom as custom_runtime

    def unexpected_read(*args, **kwargs):
        pytest.fail(
            "model seed must use the supplied config without resolving credentials"
        )

    monkeypatch.setattr(runtime_provider, "load_config", unexpected_read)
    monkeypatch.setattr(custom_runtime, "get_secret_str", unexpected_read)
    source = {
        "model": {"provider": "custom:lab", "default": "synthetic-model"},
        "providers": {
            "lab": {
                "api": "http://127.0.0.1:11434/v1",
                "key_env": "LAB_TOKEN",
                "api_key": "${LAB_TOKEN}",
            }
        },
    }

    seed = launch_model_seed(source)

    assert seed == source


URL = "http://127.0.0.1:11434/v1"
LAB = {"provider": "custom:lab", "default": "synthetic-model"}


@pytest.mark.parametrize(
    "seed,dst,expected",
    [
        # Cross-schema: a seeded legacy entry evicts a same-named keyed entry (keyed wins at runtime).
        (
            {"custom_providers": [{"name": "Lab", "base_url": URL}]},
            {"providers": {"stale": {"name": "lab", "base_url": URL + "x"}}},
            {"providers": {}, "custom_providers": [{"name": "Lab", "base_url": URL}]},
        ),
        # The same key claims the identity whatever its URL shape.
        (
            {"custom_providers": [{"name": "Lab", "base_url": URL}]},
            {"providers": {"lab": {"base_url": "localhost:9999/v1"}}},
            {"providers": {}, "custom_providers": [{"name": "Lab", "base_url": URL}]},
        ),
        # A camelCase ``baseUrl`` seed resolves, so it claims its identities.
        (
            {"custom_providers": [{"name": "Lab", "baseUrl": URL}]},
            {"providers": {"lab": {"base_url": URL + "x"}}},
            {"providers": {}, "custom_providers": [{"name": "Lab", "baseUrl": URL}]},
        ),
        # A disabled legacy seed is still live at runtime and evicts a keyed shadow.
        (
            {"custom_providers": [{"name": "Lab", "base_url": URL, "enabled": False}]},
            {"providers": {"lab": {"base_url": URL + "x"}}},
            {
                "providers": {},
                "custom_providers": [{"name": "Lab", "base_url": URL, "enabled": False}],
            },
        ),
        # Same-identity destination entries go even when they could not win on their own:
        # URL-less tuning and disabled entries (the pin step resolves names with other rules).
        (
            {"custom_providers": [{"name": "Lab", "base_url": URL}]},
            {
                "providers": {
                    "lab": {"request_timeout_seconds": 9},
                    "old": {"name": "lab", "base_url": URL + "x", "enabled": False},
                }
            },
            {"providers": {}, "custom_providers": [{"name": "Lab", "base_url": URL}]},
        ),
        # Unrelated destination entries are kept.
        (
            {"custom_providers": [{"name": "Lab", "base_url": URL}]},
            {"providers": {"other": {"base_url": URL + "x"}}},
            {
                "providers": {"other": {"base_url": URL + "x"}},
                "custom_providers": [{"name": "Lab", "base_url": URL}],
            },
        ),
    ],
)
def test_merge_endpoint_seed_drops_same_identity_entries(seed, dst, expected):
    merged = merge_endpoint_seed(dst, {"model": LAB, **seed})

    assert merged == expected
    resolved = runtime_provider._get_named_custom_provider(
        "custom:lab", config=merged, resolve_credentials=False
    )
    assert resolved["base_url"] == URL


def test_merge_endpoint_seed_tuning_seed_replaces_only_its_key():
    seed = {
        "model": {"provider": "custom", "default": "synthetic-model", "base_url": URL},
        "providers": {"custom": {"request_timeout_seconds": 5}},
    }
    dst = {"providers": {"local": {"name": "custom", "base_url": URL + "x"}}}

    assert merge_endpoint_seed(dst, seed)["providers"] == {
        "custom": {"request_timeout_seconds": 5},
        "local": {"name": "custom", "base_url": URL + "x"},
    }
