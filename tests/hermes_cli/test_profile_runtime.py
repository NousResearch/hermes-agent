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


@pytest.mark.parametrize(
    ("flag", "configured", "state_file", "expected", "reason_names"),
    [
        ("native", "container", {"runtime_kind": "container"}, "native", "--runtime"),
        (None, "container", None, "container", "profiles.runtime"),
        ("auto", None, {"runtime_kind": "container"}, "container", "container"),
        # a stopped containerized gateway still OWNS the profile: liveness is never consulted
        ("auto", None, {"runtime_kind": "container", "gateway_state": "stopped", "pid": 999999}, "container", "container"),
        # byte-for-byte today's behaviour for every plain host install and every older gateway
        ("auto", None, {"runtime_kind": "native"}, "native", "auto"),
        ("auto", None, {}, "native", "auto"),
        ("auto", None, {"runtime_kind": "docker"}, "native", "auto"),
        ("auto", None, {"runtime_kind": None}, "native", "auto"),
        ("auto", None, None, "native", "auto"),
        ("auto", None, "{not json", "native", "auto"),
    ],
    ids=["flag-beats-all", "config-when-no-flag", "auto-follows-container", "auto-ignores-liveness",
         "auto-native", "auto-older-gateway", "auto-unknown-value", "auto-null-value", "auto-no-file", "auto-corrupt-file"],
)
def test_resolution_prefers_the_flag_then_config_then_the_serving_gateway(home, monkeypatch, flag, configured, state_file, expected, reason_names):
    if configured is not None:
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"profiles": {"runtime": configured}})
    if state_file is not None:
        text = state_file if isinstance(state_file, str) else json.dumps(state_file)
        (home / "gateway_state.json").write_text(text, encoding="utf-8")

    kind, reason = resolve_profile_runtime(flag)

    assert (kind, reason_names in reason) == (expected, True)


@pytest.mark.parametrize("flag, configured", [("sideways", None), ("docker", None), ("", None), ("Container", None), (None, "sideways")],
                         ids=["typo", "docker", "empty", "wrong-case", "config-typo"])
def test_an_unrecognised_value_raises_naming_the_choices(home, monkeypatch, flag, configured):
    """A typo must not silently pick a runtime, from the flag or from config."""
    if configured is not None:
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"profiles": {"runtime": configured}})

    with pytest.raises(ValueError, match="profiles.runtime" if configured else "--runtime") as excinfo:
        resolve_profile_runtime(flag)

    assert all(choice in str(excinfo.value) for choice in RUNTIME_VALUES)
