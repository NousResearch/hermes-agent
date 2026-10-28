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


@pytest.mark.parametrize(
    ("runtime", "config", "declared", "expected_kind", "start_hint"),
    [
        ("native", {}, None, "native", "coder gateway start"),
        ("container", {}, None, "container", "docker exec hermes hermes -p coder gateway start"),
        ("container", {"profiles": {"container_name": "hermes-prod"}}, None, "container", "docker exec hermes-prod hermes -p coder gateway start"),
        # the shape this feature exists for: created from a host, the serving gateway declares a container
        (None, {}, "container", "container", "docker exec hermes hermes -p coder gateway start"),
        (None, {}, None, "native", "coder gateway start"),
    ],
    ids=["native", "container", "container-name-from-config", "auto-follows-container", "auto-is-todays-behaviour"],
)
def test_create_names_the_runtime_and_a_start_command_that_works_for_it(profile_env, capsys, monkeypatch, runtime, config, declared, expected_kind, start_hint):
    """The host cannot register an s6 slot, so a containerized profile's only working start command
    is the one that runs inside the container, and the host command is warned against."""
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    if declared:
        (profile_env / "gateway_state.json").write_text(json.dumps({"runtime_kind": declared}), encoding="utf-8")

    out = _create(capsys, runtime=runtime)

    assert f"Runtime: {expected_kind}" in out
    assert start_hint in out
    assert ("docker exec" in out) is (expected_kind == "container")
    assert ("⚠ Do not run 'coder gateway start' on this host" in out) is (expected_kind == "container")


def test_a_bad_runtime_value_fails_before_the_profile_is_created(profile_env, capsys):
    """Resolution happens BEFORE create_profile, so a typo leaves no half-made profile."""
    with pytest.raises(SystemExit):
        _create(capsys, runtime="sideways")

    assert not (profile_env / "profiles" / "coder").exists()
