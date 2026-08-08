"""Tests for ``hermes debug reproduce`` (local reproduction capsule)."""

from __future__ import annotations

import json
import zipfile

import pytest

from hermes_cli.repro_bundle import (
    _plugin_inventory,
    _redacted_config_yaml,
    _sanitized_session_export,
    build_repro_bundle,
    inspect_bundle,
    run_debug_reproduce,
)


@pytest.fixture
def hermes_home(monkeypatch, tmp_path):
    home = tmp_path / "cred_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(tmp_path / "bundled_plugins"))
    (tmp_path / "bundled_plugins").mkdir()
    return home


def test_plugin_inventory_reads_real_manifests(hermes_home, tmp_path):
    plugin_dir = tmp_path / "bundled_plugins" / "acme"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        "name: acme\nversion: 1.2.3\nkind: standalone\n", encoding="utf-8"
    )
    entries = _plugin_inventory()
    assert {"name": "acme", "version": "1.2.3", "kind": "standalone", "source": "bundled"} in entries


def test_redacted_config_yaml_never_leaks_secret_shaped_value(hermes_home):
    hermes_home.joinpath("config.yaml").write_text(
        "model:\n  api_key: sk-super-secret-value-99999\n  provider: openai-api\n",
        encoding="utf-8",
    )
    text = _redacted_config_yaml()
    assert "sk-super-secret-value-99999" not in text
    assert "openai-api" in text  # non-secret fields survive


def test_sanitized_session_export_redacts_string_content(monkeypatch):
    class FakeDB:
        def __init__(self, *a, **kw):
            pass

        def get_messages_as_conversation(self, session_id):
            return [
                {"role": "user", "content": "my key is sk-fake-leaked-1234567890"},
                {"role": "assistant", "content": "ok"},
            ]

    import hermes_state

    monkeypatch.setattr(hermes_state, "SessionDB", FakeDB)
    messages = _sanitized_session_export("some-session")
    assert messages is not None
    assert "sk-fake-leaked-1234567890" not in json.dumps(messages)


def test_sanitized_session_export_returns_none_for_empty_session(monkeypatch):
    class FakeDB:
        def __init__(self, *a, **kw):
            pass

        def get_messages_as_conversation(self, session_id):
            return []

    import hermes_state

    monkeypatch.setattr(hermes_state, "SessionDB", FakeDB)
    assert _sanitized_session_export("missing") is None


def test_build_repro_bundle_contains_expected_files_no_session(hermes_home, tmp_path):
    output = tmp_path / "out.zip"
    summary = build_repro_bundle(output, session_id=None, log_lines=10, redact=True)
    assert summary["manifest"]["session_included"] is False
    with zipfile.ZipFile(output) as zf:
        names = set(zf.namelist())
    assert "manifest.json" in names
    assert "report.txt" in names
    assert "plugins.json" in names
    assert "config_redacted.yaml" in names
    assert "README.md" in names
    assert "session.json" not in names


def test_build_repro_bundle_includes_session_when_opted_in(hermes_home, tmp_path, monkeypatch):
    class FakeDB:
        def __init__(self, *a, **kw):
            pass

        def get_messages_as_conversation(self, session_id):
            return [{"role": "user", "content": "hello"}]

    import hermes_state

    monkeypatch.setattr(hermes_state, "SessionDB", FakeDB)

    output = tmp_path / "out.zip"
    summary = build_repro_bundle(output, session_id="abc", log_lines=10, redact=True)
    assert summary["manifest"]["session_included"] is True
    with zipfile.ZipFile(output) as zf:
        assert "session.json" in zf.namelist()


def test_inspect_bundle_lists_files_without_executing_anything(hermes_home, tmp_path):
    output = tmp_path / "out.zip"
    build_repro_bundle(output, session_id=None, log_lines=10, redact=True)
    info = inspect_bundle(output)
    assert "manifest.json" in info["files"]
    assert info["manifest"]["format"] == "hermes-repro/1"


def test_run_debug_reproduce_build_and_inspect_cli(hermes_home, tmp_path, capsys):
    import argparse

    output = tmp_path / "cli.zip"
    build_args = argparse.Namespace(
        reproduce_action="build", output=str(output), session=None,
        lines=10, no_redact=False,
    )
    run_debug_reproduce(build_args)
    assert output.exists()
    capsys.readouterr()

    inspect_args = argparse.Namespace(reproduce_action="inspect", bundle=str(output))
    run_debug_reproduce(inspect_args)
    out = capsys.readouterr().out
    assert "format: hermes-repro/1" in out
    assert "session included: False" in out


def test_run_debug_reproduce_inspect_missing_bundle_exits_nonzero(tmp_path, capsys):
    import argparse

    args = argparse.Namespace(reproduce_action="inspect", bundle=str(tmp_path / "nope.zip"))
    with pytest.raises(SystemExit) as exc_info:
        run_debug_reproduce(args)
    assert exc_info.value.code == 1
    assert "not found" in capsys.readouterr().err
