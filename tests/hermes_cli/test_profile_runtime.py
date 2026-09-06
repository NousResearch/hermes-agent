"""Where a new profile's gateway will be supervised (`profiles.runtime`).

The interesting case is a host whose ``~/.hermes`` is bind-mounted into a container: the
creating process is native, the serving gateway is containerized, and only the gateway's own
declaration can tell them apart.
"""

from __future__ import annotations

import json

import pytest

from hermes_cli.profile_runtime import RUNTIME_VALUES, resolve_profile_runtime


@pytest.fixture
def home(tmp_path, monkeypatch):
    """An isolated HERMES_HOME with no config on disk, so `profiles.runtime` defaults."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    return tmp_path


def _declare(home, kind):
    (home / "gateway_state.json").write_text(json.dumps({"runtime_kind": kind}), encoding="utf-8")


def test_values_are_the_documented_three():
    assert RUNTIME_VALUES == ("auto", "container", "native")


def test_explicit_flag_beats_config_and_the_gateway_declaration(home, monkeypatch):
    """An operator who names a runtime gets it, even when everything else disagrees."""
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"profiles": {"runtime": "container"}})
    _declare(home, "container")

    kind, reason = resolve_profile_runtime("native")

    assert kind == "native"
    assert "--runtime" in reason


def test_config_is_used_when_no_flag_is_given(home, monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"profiles": {"runtime": "container"}})

    kind, reason = resolve_profile_runtime(None)

    assert kind == "container"
    assert "profiles.runtime" in reason


def test_auto_follows_a_containerized_serving_gateway(home):
    """The whole point: the creating process is native, the gateway says container, and the new
    profile follows the gateway."""
    _declare(home, "container")

    kind, reason = resolve_profile_runtime("auto")

    assert kind == "container"
    assert "container" in reason


def test_auto_ignores_liveness(home):
    """A stopped containerized gateway still OWNS the profile, so a dead PID must not flip the
    answer — the resolver must not consult liveness at all."""
    (home / "gateway_state.json").write_text(
        json.dumps({"runtime_kind": "container", "gateway_state": "stopped", "pid": 999999}),
        encoding="utf-8",
    )

    assert resolve_profile_runtime("auto")[0] == "container"


@pytest.mark.parametrize("payload", [
    {"runtime_kind": "native"},
    {},                                  # a gateway older than the field
    {"runtime_kind": "docker"},          # an unrecognised value must never read as container
    {"runtime_kind": None},
])
def test_auto_resolves_native_without_a_container_declaration(home, payload):
    """Byte-for-byte today's behaviour for every plain host install and every older gateway."""
    (home / "gateway_state.json").write_text(json.dumps(payload), encoding="utf-8")

    assert resolve_profile_runtime("auto")[0] == "native"


def test_auto_resolves_native_when_no_state_file_exists(home):
    assert resolve_profile_runtime("auto")[0] == "native"


def test_auto_survives_a_corrupt_state_file(home):
    (home / "gateway_state.json").write_text("{not json", encoding="utf-8")

    assert resolve_profile_runtime("auto")[0] == "native"


@pytest.mark.parametrize("bad", ["sideways", "docker", "", "Container"])
def test_an_unrecognised_value_raises_naming_the_choices(home, monkeypatch, bad):
    """A typo must not silently pick a runtime."""
    with pytest.raises(ValueError) as excinfo:
        resolve_profile_runtime(bad)

    message = str(excinfo.value)
    for choice in RUNTIME_VALUES:
        assert choice in message


def test_an_unrecognised_config_value_raises_too(home, monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"profiles": {"runtime": "sideways"}})

    with pytest.raises(ValueError, match="profiles.runtime"):
        resolve_profile_runtime(None)
