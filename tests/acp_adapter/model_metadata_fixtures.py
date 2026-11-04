"""Named-provider catalog fixtures shared by ACP metadata tests."""

import json

import pytest

MODEL_A = "route-a/nested:variant/model"
MODEL_B = "route-b/same-name"
MODEL_OFF = "route-b/plain"
PROVIDER = "custom:relay:one"
CHOICE_A = "custom:relay:one:route-a/nested:variant/model"
CHOICE_B = "custom:relay:one:route-b/same-name"
CHOICE_OFF = "custom:relay:one:route-b/plain"


@pytest.fixture
def declared_models(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    config = {
        "model": {"provider": PROVIDER, "default": MODEL_A},
        "agent": {"reasoning_effort": "low"},
        "providers": {
            "relay:one": {
                "name": "Relay one",
                "api": "https://one.example.invalid/v1",
                "api_key": "fixture-only",
                "discover_models": False,
                "models": {
                    MODEL_A: {"name": "Shared", "group": "upstream-a", "reasoning_efforts": ["low", "high"],
                              "context_length": 64000, "description": "First route"},
                    MODEL_B: {"name": "Shared", "group": "upstream-b", "reasoning_efforts": ["medium"]},
                    MODEL_OFF: {"name": "Plain", "group": "upstream-b", "reasoning_efforts": []},
                    "legacy": {},
                },
            },
            "relay-two": {
                "name": "Relay two",
                "api": "https://two.example.invalid/v1",
                "discover_models": False,
                "models": {MODEL_A: {"name": "Shared", "group": "upstream-a", "reasoning_efforts": ["none", "low"]}},
            },
        },
    }

    def write():
        (home / "config.yaml").write_text(json.dumps(config))

    write()
    return home, config, write
