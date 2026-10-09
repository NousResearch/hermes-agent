"""``hermes update`` / ``hermes migrate relay``: legacy ``HERMES_NEMO_RELAY_ATIF_*``/``ATOF_*`` vars
become a validated ``relay-plugins.toml`` per profile home, selected from ``.env``."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from hermes_cli.relay_plugin_migrate import (
    RELAY_PLUGINS_TOML_NAME, migrate_all_profile_relay_envs, migrate_profile_relay_env,
    validate_relay_plugin_payload)
from hermes_cli.relay_plugin_cutover import RELAY_PLUGINS_CONFIG_ENV, configured_legacy_relay_env_vars

nemo_relay = pytest.importorskip("nemo_relay")

LEGACY_ENV = """OPENAI_API_KEY=sk-test
HERMES_NEMO_RELAY_ATOF_ENABLED=1
HERMES_NEMO_RELAY_ATOF_OUTPUT_DIRECTORY={home}/telemetry/atof
HERMES_NEMO_RELAY_ATOF_FILENAME=hermes-atof.jsonl
HERMES_NEMO_RELAY_ATOF_MODE=append
HERMES_NEMO_RELAY_ATIF_ENABLED=1
HERMES_NEMO_RELAY_ATIF_OUTPUT_DIRECTORY={home}/telemetry/atif
HERMES_NEMO_RELAY_ATIF_FILENAME_TEMPLATE=trajectory-{{session_id}}.json
HERMES_NEMO_RELAY_ATIF_SUBAGENT_EXPORT_MODE=all
"""


def _parse_env(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip().strip('"')
    return out


@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_legacy_env_becomes_validated_toml_selected_from_env(profile_env):
    (profile_env / ".env").write_text(LEGACY_ENV.format(home=profile_env), encoding="utf-8")

    result = migrate_profile_relay_env(profile_env)

    assert result.migrated, (result.skipped_reason, result.validation_error)
    assert result.diagnostics == []
    toml_path = profile_env / RELAY_PLUGINS_TOML_NAME
    document = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    sink = document["components"][0]["config"]["atof"]["sinks"][0]
    assert sink["type"] == "file" and sink["filename"] == "hermes-atof.jsonl"
    assert document["components"][0]["config"]["atif"]["filename_template"] == "trajectory-{session_id}.json"
    # Relay accepts the complete document generated from the legacy settings.
    report = nemo_relay.plugin.validate_exact(document)
    assert report["config"]["diagnostics"] == []
    # .env now selects the file; the legacy lines survive as comments (not deleted), so the
    # runtime warning + doctor finding go quiet and a second run is a no-op.
    env = _parse_env(profile_env / ".env")
    assert env[RELAY_PLUGINS_CONFIG_ENV] == str(toml_path)
    assert env["OPENAI_API_KEY"] == "sk-test"
    assert configured_legacy_relay_env_vars(env) == ()
    assert "# migrated to relay-plugins.toml: HERMES_NEMO_RELAY_ATOF_ENABLED=1" in (profile_env / ".env").read_text(encoding="utf-8")
    assert migrate_profile_relay_env(profile_env).skipped_reason == "no legacy exporter variables"


def test_update_migrates_every_profile_home_separately(profile_env):
    """Multiplex: each profile keeps its own TOML; a profile without legacy vars is untouched."""
    (profile_env / ".env").write_text("OPENAI_API_KEY=x\n", encoding="utf-8")
    work = profile_env / "profiles" / "work"
    idle = profile_env / "profiles" / "idle"
    for p in (work, idle):
        p.mkdir(parents=True)
    (work / ".env").write_text(LEGACY_ENV.format(home=work), encoding="utf-8")
    (idle / ".env").write_text("SLACK_BOT_TOKEN=y\n", encoding="utf-8")

    results = {r.home: r for r in migrate_all_profile_relay_envs(validate=False)}

    assert results[work].migrated and results[work].toml_path == work / RELAY_PLUGINS_TOML_NAME
    assert not results[idle].migrated and not (idle / RELAY_PLUGINS_TOML_NAME).exists()
    assert not results[profile_env].migrated and not (profile_env / RELAY_PLUGINS_TOML_NAME).exists()
    assert _parse_env(work / ".env")[RELAY_PLUGINS_CONFIG_ENV] == str(work / RELAY_PLUGINS_TOML_NAME)
    assert (idle / ".env").read_text(encoding="utf-8") == "SLACK_BOT_TOKEN=y\n"


def _relay_report(*diagnostics):
    return {"config": {"diagnostics": list(diagnostics)}, "config_paths": [], "dynamic_plugins": []}


def test_error_diagnostics_reject_the_payload_like_relay_0_8_did(monkeypatch):
    monkeypatch.setattr(
        nemo_relay.plugin,
        "validate_exact",
        lambda _payload: _relay_report(
            {"level": "warning", "code": "unknown_field", "message": "unknown field 'x'"},
            {"level": "error", "code": "unsupported_value", "message": "atof.mode 'nope' is unsupported"},
        ),
    )

    with pytest.raises(ValueError, match="atof.mode 'nope' is unsupported"):
        validate_relay_plugin_payload({"version": 1})


def test_warning_diagnostics_are_returned_not_raised(monkeypatch):
    warning = {"level": "warning", "code": "unknown_field", "message": "unknown field 'x'"}
    monkeypatch.setattr(nemo_relay.plugin, "validate_exact", lambda _payload: _relay_report(warning))

    assert validate_relay_plugin_payload({"version": 1}) == [warning]


def test_error_diagnostics_leave_env_untouched(profile_env, monkeypatch):
    (profile_env / ".env").write_text(LEGACY_ENV.format(home=profile_env), encoding="utf-8")
    before = (profile_env / ".env").read_text(encoding="utf-8")
    monkeypatch.setattr(
        nemo_relay.plugin,
        "validate_exact",
        lambda _payload: _relay_report({"level": "error", "code": "bad", "message": "rejected"}),
    )

    result = migrate_profile_relay_env(profile_env)

    assert not result.migrated
    assert result.validation_error == "ValueError: rejected"
    assert (profile_env / ".env").read_text(encoding="utf-8") == before
    assert not (profile_env / RELAY_PLUGINS_TOML_NAME).exists()
