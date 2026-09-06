"""`hermes profile create` output tells the operator which runtime the new profile got, and
names a start command that actually works for it."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest


@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    """Isolated home, mirroring tests/hermes_cli/test_profiles.py's fixture."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    return default_home


def _args(**overrides):
    base = dict(profile_name="coder", clone=False, clone_all=False, clone_from=None,
                no_alias=True, no_skills=True, description=None, runtime=None)
    base.update(overrides)
    return Namespace(**base)


def _create(capsys, **overrides):
    from hermes_cli.profile_cmd import _profile_create
    _profile_create(_args(**overrides))
    return capsys.readouterr().out


def test_native_profile_keeps_the_plain_start_hint(profile_env, capsys):
    out = _create(capsys, runtime="native")

    assert "Runtime: native" in out
    assert "coder gateway start" in out
    assert "docker exec" not in out


def test_container_profile_hands_off_to_the_container(profile_env, capsys):
    """The host cannot register an s6 slot, so the only command that works is the one that
    runs inside the container — and the host command must be warned against, not printed."""
    out = _create(capsys, runtime="container")

    assert "Runtime: container" in out
    assert "docker exec hermes hermes -p coder gateway start" in out
    assert "⚠ Do not run 'coder gateway start' on this host" in out


def test_container_name_comes_from_config(profile_env, capsys, monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config",
                        lambda: {"profiles": {"container_name": "hermes-prod"}})

    out = _create(capsys, runtime="container")

    assert "docker exec hermes-prod hermes -p coder gateway start" in out


def test_auto_follows_a_containerized_serving_gateway(profile_env, capsys):
    """The shape this feature exists for: created from a host, but the gateway serving the
    active home declares a container, so the new profile is treated as containerized."""
    (profile_env / "gateway_state.json").write_text(
        json.dumps({"runtime_kind": "container"}), encoding="utf-8")

    out = _create(capsys)

    assert "Runtime: container" in out
    assert "docker exec" in out


def test_auto_is_todays_behaviour_when_nothing_declares_a_container(profile_env, capsys):
    out = _create(capsys)

    assert "Runtime: native" in out
    assert "coder gateway start" in out
    assert "docker exec" not in out


def test_a_bad_runtime_value_fails_before_the_profile_is_created(profile_env, capsys):
    """Resolution happens BEFORE create_profile, so a typo leaves no half-made profile."""
    with pytest.raises(SystemExit):
        _create(capsys, runtime="sideways")

    assert not (profile_env / "profiles" / "coder").exists()
