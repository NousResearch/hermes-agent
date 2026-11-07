"""``passive_rooms`` (config.yaml) / ``MATRIX_PASSIVE_ROOMS`` (env) parse like
``free_response_rooms``: CSV lists, env-over-YAML precedence, and the YAML→env
bridge propagates the key to the env var exactly as the sibling does."""

import pytest

from gateway.config import PlatformConfig


def _csv(extra):
    from plugins.platforms.matrix.adapter import _extra_csv_set

    return _extra_csv_set(
        PlatformConfig(enabled=True, extra=extra), "passive_rooms", "MATRIX_PASSIVE_ROOMS")


def test_passive_rooms_csv_parses_whitespace_and_empty_entries(monkeypatch):
    monkeypatch.setenv("MATRIX_PASSIVE_ROOMS", " !a:example.org , ,!b:example.org\t,  ")

    assert _csv({}) == {"!a:example.org", "!b:example.org"}


def test_passive_rooms_env_fallback_when_config_absent_or_blank(monkeypatch):
    monkeypatch.setenv("MATRIX_PASSIVE_ROOMS", "!home:example.org")

    for extra in ({}, {"passive_rooms": ""}, {"passive_rooms": "  "}):
        assert _csv(extra) == {"!home:example.org"}, extra


def test_passive_rooms_explicit_env_beats_yaml_and_blank_env_falls_to_yaml(monkeypatch):
    monkeypatch.setenv("MATRIX_PASSIVE_ROOMS", "!env:example.org")
    extra = {"passive_rooms": ["!a:example.org", " !b:example.org "]}

    assert _csv(extra) == {"!env:example.org"}

    monkeypatch.setenv("MATRIX_PASSIVE_ROOMS", "  ")  # blank env is unset
    assert _csv(extra) == {"!a:example.org", "!b:example.org"}

    monkeypatch.delenv("MATRIX_PASSIVE_ROOMS")
    assert _csv(extra) == {"!a:example.org", "!b:example.org"}
    assert _csv({}) == set()


@pytest.mark.parametrize("raw", [
    "!one:example.org,!two:example.org",
    " !one:example.org , ,!two:example.org ",
    "!solo:example.org",
])
def test_passive_rooms_parity_with_free_response_rooms_for_identical_inputs(monkeypatch, raw):
    from plugins.platforms.matrix.adapter import _extra_csv_set

    monkeypatch.setenv("MATRIX_PASSIVE_ROOMS", raw)
    monkeypatch.setenv("MATRIX_FREE_RESPONSE_ROOMS", raw)
    env_config = PlatformConfig(enabled=True, extra={})
    assert _extra_csv_set(env_config, "passive_rooms", "MATRIX_PASSIVE_ROOMS") == \
        _extra_csv_set(env_config, "free_response_rooms", "MATRIX_FREE_RESPONSE_ROOMS")

    # Identical YAML lists give identical sets, too.
    rooms = [entry.strip() for entry in raw.split(",") if entry.strip()]
    yaml_config = PlatformConfig(
        enabled=True, extra={"passive_rooms": rooms, "free_response_rooms": rooms})
    assert _extra_csv_set(yaml_config, "passive_rooms", "MATRIX_PASSIVE_ROOMS") == \
        _extra_csv_set(yaml_config, "free_response_rooms", "MATRIX_FREE_RESPONSE_ROOMS")


def test_adapter_loads_passive_rooms_from_env_and_yaml(monkeypatch):
    from plugins.platforms.matrix.adapter import MatrixAdapter

    monkeypatch.setenv("MATRIX_PASSIVE_ROOMS", "!a:example.org, !b:example.org")
    adapter = MatrixAdapter(PlatformConfig(enabled=True, token="***"))
    assert adapter._passive_rooms == {"!a:example.org", "!b:example.org"}

    monkeypatch.delenv("MATRIX_PASSIVE_ROOMS")
    adapter = MatrixAdapter(
        PlatformConfig(enabled=True, token="***", extra={"passive_rooms": ["!c:example.org"]}))
    assert adapter._passive_rooms == {"!c:example.org"}


@pytest.mark.parametrize("existing", [None, "operator"])
@pytest.mark.parametrize("secondary", [False, True])
def test_yaml_bridge_propagates_passive_rooms_to_env(monkeypatch, existing, secondary):
    """Word-for-word shape of the free_response_rooms bridge test: explicit env wins,
    the YAML list is comma-joined, and under a multiplexed secondary scope the env write is
    skipped (first-writer-wins poisoning guard)."""
    import os

    from agent.secret_scope import reset_secret_scope, set_multiplex_active, set_secret_scope
    from plugins.platforms.matrix.adapter import _apply_yaml_config

    expected = {"MATRIX_PASSIVE_ROOMS": "!one:example.org,!two:example.org",
                "MATRIX_FREE_RESPONSE_ROOMS": "!free:example.org"}
    config = {"passive_rooms": ["!one:example.org", "!two:example.org"],
              "free_response_rooms": ["!free:example.org"]}
    for key in expected:
        monkeypatch.delenv(key, raising=False)
        if existing is not None:
            monkeypatch.setenv(key, existing)
    token = set_secret_scope({}) if secondary else None
    set_multiplex_active(secondary)
    try:
        assert _apply_yaml_config({"matrix": config}, config) == config
        assert {key: os.environ.get(key) for key in expected} == (
            dict.fromkeys(expected, existing) if secondary or existing else expected
        )
    finally:
        if token is not None:
            reset_secret_scope(token)
        set_multiplex_active(False)
