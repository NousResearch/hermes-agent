"""Sol plugin registration and lazy construction boundaries."""

from pathlib import Path

import pytest

import plugins.sol_food as sol_food
from plugins.sol_food.hook import SolFoodHook
from plugins.sol_food.legacy_guard import LegacyHelperPresent


class _Context:
    def __init__(self) -> None:
        self.profile = None
        self.factory = None
        self.llm = object()

    def register_topic_hook_factory(self, profile, factory) -> None:
        self.profile = profile
        self.factory = factory


def test_register_is_inert(monkeypatch) -> None:
    context = _Context()
    monkeypatch.delenv("HEALTH_FOOD_COMMIT_URL", raising=False)
    monkeypatch.delenv("HEALTH_FOOD_COMMIT_TOKEN", raising=False)
    sol_food.register(context)
    assert context.profile == "sol"
    assert callable(context.factory)


def test_plugin_manager_loads_without_touching_health_configuration(
    monkeypatch, tmp_path: Path
) -> None:
    from hermes_cli.plugins import PluginManager, PluginManifest

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HEALTH_FOOD_COMMIT_URL", raising=False)
    monkeypatch.delenv("HEALTH_FOOD_COMMIT_TOKEN", raising=False)
    manager = PluginManager()
    manifest = PluginManifest(
        name="sol-food",
        key="sol-food",
        kind="backend",
        source="bundled",
        path=Path(sol_food.__file__).parent,
    )
    manager._load_plugin(manifest)
    loaded = manager._plugins["sol-food"]
    assert loaded.error is None
    assert not (tmp_path / "state" / "sol-food").exists()


def test_explicit_construction_requires_health_configuration(monkeypatch) -> None:
    monkeypatch.delenv("HEALTH_FOOD_COMMIT_URL", raising=False)
    monkeypatch.delenv("HEALTH_FOOD_COMMIT_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="sol_food_configuration_missing"):
        sol_food._build_topic_hook()


def test_explicit_construction_uses_owner_private_profile_state(
    monkeypatch, tmp_path: Path
) -> None:
    # Canonical base64url encoding of 32 zero bytes (synthetic only).
    token = "A" * 43
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HEALTH_FOOD_COMMIT_URL", "http://127.0.0.1:8765/food")
    monkeypatch.setenv("HEALTH_FOOD_COMMIT_TOKEN", token)
    hook = sol_food._build_topic_hook()
    assert isinstance(hook, SolFoodHook)
    state_dir = tmp_path / "profiles" / "sol" / "state" / "sol-food"
    assert state_dir.is_dir()
    assert state_dir.stat().st_mode & 0o777 == 0o700
    assert not (tmp_path / "state" / "sol-food").exists()


def test_explicit_construction_checks_legacy_helper_in_routed_sol_profile(
    monkeypatch, tmp_path: Path
) -> None:
    token = "A" * 43
    sol_home = tmp_path / "profiles" / "sol"
    legacy = sol_home / "scripts" / "food_log_commit.py"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("legacy", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HEALTH_FOOD_COMMIT_URL", "http://127.0.0.1:8765/food")
    monkeypatch.setenv("HEALTH_FOOD_COMMIT_TOKEN", token)

    with pytest.raises(LegacyHelperPresent, match="sol_food_legacy_helper_present"):
        sol_food._build_topic_hook()


def test_explicit_construction_checks_legacy_helper_in_default_root(
    monkeypatch, tmp_path: Path
) -> None:
    token = "A" * 43
    legacy = tmp_path / "scripts" / "food_log_commit.py"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("legacy", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HEALTH_FOOD_COMMIT_URL", "http://127.0.0.1:8765/food")
    monkeypatch.setenv("HEALTH_FOOD_COMMIT_TOKEN", token)

    with pytest.raises(LegacyHelperPresent, match="sol_food_legacy_helper_present"):
        sol_food._build_topic_hook()


def test_invalid_secret_fails_without_echo(monkeypatch, tmp_path: Path) -> None:
    secret = "not-a-valid-secret"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HEALTH_FOOD_COMMIT_URL", "http://127.0.0.1:8765/food")
    monkeypatch.setenv("HEALTH_FOOD_COMMIT_TOKEN", secret)
    with pytest.raises(RuntimeError) as raised:
        sol_food._build_topic_hook()
    assert str(raised.value) == "health_client_bad_token"
    assert secret not in str(raised.value)


@pytest.mark.asyncio
async def test_host_owned_parser_returns_bounded_candidates() -> None:
    class _Llm:
        async def acomplete_structured(self, **kwargs):
            assert kwargs.get("provider") is None
            assert kwargs.get("model") is None
            assert kwargs.get("profile") is None
            return type(
                "Result",
                (),
                {
                    "parsed": {
                        "candidates": [
                            {
                                "label": "Synthetic option",
                                "items": [
                                    {"plant_key": "synthetic", "is_plant": True}
                                ],
                            }
                        ]
                    }
                },
            )()

    parser = sol_food._parser_for(_Llm())
    candidates = await parser("synthetic meal", None)
    assert len(candidates) == 1
    assert candidates[0].items[0]["plant_key"] == "synthetic"
