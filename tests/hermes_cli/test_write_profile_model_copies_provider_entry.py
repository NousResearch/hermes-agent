"""Profile builder must copy a config-map provider into the named profile.

``_write_profile_model`` writes ``model.provider`` / ``model.default`` inside
the target profile's HERMES_HOME. Custom providers such as ``scnet`` live only
in the caller's (outer) ``providers:`` map; without copying that entry, agent
init raises ``Unknown provider 'scnet'``.

Regression for #106643.
"""

from __future__ import annotations

import yaml

from hermes_constants import get_hermes_home
from hermes_cli.web_routers.profiles import _write_profile_model

_SCNET_ENTRY = {
    "name": "scnet",
    "base_url": "https://api.scnet.cn/api/llm/v1",
    "key_env": "HERMES_CUSTOM_SCNET_API_KEY",
    "discover_models": True,
}


def _write_yaml(path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _read_yaml(path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def _empty_profile_dir(outer):
    profile_dir = outer / "profiles" / "worker"
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "config.yaml").write_text("{}\n", encoding="utf-8")
    return profile_dir


def test_copies_config_map_provider_from_outer_home(_isolate_hermes_home):
    """Outer ``providers.scnet`` must land in the target profile config."""
    outer = get_hermes_home()
    _write_yaml(
        outer / "config.yaml",
        {
            "providers": {
                "scnet": dict(_SCNET_ENTRY),
                "other": {
                    "name": "other",
                    "base_url": "https://example.invalid/v1",
                    "key_env": "HERMES_CUSTOM_OTHER_API_KEY",
                },
            }
        },
    )
    profile_dir = _empty_profile_dir(outer)

    _write_profile_model(profile_dir, "scnet", "GLM-5.3-Flash")

    written = _read_yaml(profile_dir / "config.yaml")
    scnet = (written.get("providers") or {}).get("scnet")
    assert isinstance(scnet, dict), "target profile must contain providers.scnet"
    assert scnet.get("base_url") == _SCNET_ENTRY["base_url"]
    assert scnet.get("key_env") == _SCNET_ENTRY["key_env"]
    model = written.get("model") or {}
    assert isinstance(model, dict)
    assert model.get("provider") == "scnet"
    # Only the assigned provider is copied — not the rest of the outer map.
    assert "other" not in (written.get("providers") or {})


def test_copies_literal_inline_api_key_without_key_env(_isolate_hermes_home):
    """A raw inline ``api_key`` (no key_env) is a documented config-map form; copy it as-is."""
    outer = get_hermes_home()
    _write_yaml(
        outer / "config.yaml",
        {
            "providers": {
                "scnet": {
                    "name": "scnet",
                    "base_url": "https://api.scnet.cn/api/llm/v1",
                    "api_key": "sk-test-literal",
                    "discover_models": True,
                }
            }
        },
    )
    profile_dir = _empty_profile_dir(outer)

    _write_profile_model(profile_dir, "scnet", "GLM-5.3-Flash")

    written = _read_yaml(profile_dir / "config.yaml")
    scnet = (written.get("providers") or {}).get("scnet")
    assert isinstance(scnet, dict), "target profile must contain providers.scnet"
    assert scnet.get("api_key") == "sk-test-literal"
    assert "key_env" not in scnet
    assert scnet.get("base_url") == "https://api.scnet.cn/api/llm/v1"
    model = written.get("model") or {}
    assert isinstance(model, dict)
    assert model.get("provider") == "scnet"


def test_builtin_assignment_does_not_invent_providers_block(_isolate_hermes_home):
    """Built-in registry providers have no config-map entry; do not invent one."""
    outer = get_hermes_home()
    _write_yaml(outer / "config.yaml", {"model": {"provider": "openrouter", "default": "x"}})
    profile_dir = _empty_profile_dir(outer)

    _write_profile_model(profile_dir, "openrouter", "anthropic/claude-sonnet-4.6")

    written = _read_yaml(profile_dir / "config.yaml")
    providers = written.get("providers")
    assert not providers, f"built-in assignment must not create providers, got {providers!r}"


def test_existing_target_provider_entry_is_not_overwritten(_isolate_hermes_home):
    """A named profile's local providers.scnet customization must be kept."""
    outer = get_hermes_home()
    _write_yaml(outer / "config.yaml", {"providers": {"scnet": dict(_SCNET_ENTRY)}})
    profile_dir = _empty_profile_dir(outer)
    _write_yaml(
        profile_dir / "config.yaml",
        {
            "providers": {
                "scnet": {
                    "name": "keep-local-name",
                    "base_url": "https://local.example/v1",
                    "key_env": "LOCAL_SCNET_KEY",
                    "models": ["kept-model"],
                }
            }
        },
    )

    _write_profile_model(profile_dir, "scnet", "GLM-5.3-Flash")

    written = _read_yaml(profile_dir / "config.yaml")
    scnet = (written.get("providers") or {}).get("scnet") or {}
    assert scnet.get("name") == "keep-local-name"
    assert scnet.get("models") == ["kept-model"]
    assert scnet.get("base_url") == "https://local.example/v1"
    model = written.get("model") or {}
    assert isinstance(model, dict)
    assert model.get("provider") == "scnet"
