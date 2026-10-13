"""``onboarding.model`` in the user's profile pins the desktop setup chat's model on create and on reset; a reset
after the override is removed puts the setup profile back on the user's own model."""

from pathlib import Path

import pytest

from hermes_cli import setup_profile
from hermes_cli.config import atomic_config_replace, read_user_config_raw

OVERRIDE = {"provider": "anthropic", "default": "claude-sonnet-4-5"}
OWN_MODELS = pytest.mark.parametrize("own", [{}, {"model": {"provider": "openrouter", "default": "some/model"}}])


@pytest.fixture
def root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    return root


def _model(home: Path):
    return read_user_config_raw(home / "config.yaml").get("model")


@OWN_MODELS
def test_create_pins_the_override_and_leaves_the_user_profile_alone(root, own):
    atomic_config_replace(root / "config.yaml", {**own, "onboarding": {"model": OVERRIDE}})

    assert _model(setup_profile.ensure_setup_profile().path) == OVERRIDE
    assert _model(root) == own.get("model")


@OWN_MODELS
def test_reset_follows_the_override_both_ways(root, own):
    atomic_config_replace(root / "config.yaml", own)
    setup = setup_profile.ensure_setup_profile().path
    assert _model(setup) == own.get("model")

    atomic_config_replace(root / "config.yaml", {**own, "onboarding": {"model": OVERRIDE}})
    setup_profile.reset_setup_profile(root)
    assert _model(setup) == OVERRIDE

    atomic_config_replace(root / "config.yaml", own)
    setup_profile.reset_setup_profile(root)
    assert _model(setup) == own.get("model")
