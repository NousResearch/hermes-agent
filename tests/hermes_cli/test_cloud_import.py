"""E2E tests for the first-boot cloud migration importer.

Runs ``cloud_import.maybe_run`` against a synthetic HERMES_HOME and a
migration bundle served over a ``file://`` URL (the same urllib transport
the prod path uses, without a network hop).
"""

from __future__ import annotations

import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

from hermes_cli import cloud_import
from hermes_cli.cloud_import import (
    IMPORT_MARKER_NAME,
    MIGRATION_BUNDLE_URL_ENV,
    MIGRATION_SCHEMA_VERSION,
    maybe_run,
)

CONFIG = "_config_version: 4\nmodel:\n  provider: nous\n"


def _make_bundle(
    path: Path,
    *,
    schema_version: int = MIGRATION_SCHEMA_VERSION,
    manifest: bool = True,
    secrets: bool = True,
) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("config.yaml", CONFIG)
        zf.writestr("SOUL.md", "be concise")
        zf.writestr("skills/custom/SKILL.md", "---\nname: custom\n---\n# b\n")
        if secrets:
            zf.writestr(".env", "ANTHROPIC_API_KEY=sk-ant-secret\n")
            zf.writestr("auth.json", '{"token": "x"}')
        if manifest:
            zf.writestr(
                "migration-manifest.json",
                json.dumps(
                    {
                        "schema_version": schema_version,
                        "tool_version": "0.0.0-test",
                        "config_version": 4,
                        "secrets_included": secrets,
                        "secrets_excluded": [],
                    }
                ),
            )


@pytest.fixture()
def fresh_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_no_env_var_is_a_noop(fresh_home, capsys):
    assert maybe_run() is False
    assert "cloud_import" not in capsys.readouterr().out


def test_imports_bundle_and_writes_marker(fresh_home, tmp_path, monkeypatch):
    bundle = tmp_path / "bundle.zip"
    _make_bundle(bundle)
    monkeypatch.setenv(MIGRATION_BUNDLE_URL_ENV, bundle.as_uri())

    assert maybe_run() is True

    assert (fresh_home / "config.yaml").read_text() == CONFIG
    assert (fresh_home / "SOUL.md").exists()
    assert (fresh_home / "skills" / "custom" / "SKILL.md").exists()
    # Secrets dropped even though the bundle carried them.
    assert not (fresh_home / ".env").exists()
    assert not (fresh_home / "auth.json").exists()
    # Marker written, staged copy deleted (the source bundle is not ours).
    marker = json.loads((fresh_home / IMPORT_MARKER_NAME).read_text())
    assert marker["manifest"]["schema_version"] == MIGRATION_SCHEMA_VERSION
    assert marker["skipped_secret_files"] == [".env", "auth.json"]
    assert not list(fresh_home.glob(".cloud-migration-bundle.*"))


def test_marker_present_skips(fresh_home, tmp_path, monkeypatch):
    _make_bundle(tmp_path / "bundle.zip")
    (fresh_home / IMPORT_MARKER_NAME).write_text('{"imported_at": "x"}')
    monkeypatch.setenv(MIGRATION_BUNDLE_URL_ENV, (tmp_path / "bundle.zip").as_uri())

    assert maybe_run() is False
    assert not (fresh_home / "config.yaml").exists()


def test_non_fresh_home_skips(fresh_home, tmp_path, monkeypatch):
    # Instance already has session state: importing would clobber live data.
    conn = sqlite3.connect(fresh_home / "state.db")
    conn.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()

    _make_bundle(tmp_path / "bundle.zip")
    monkeypatch.setenv(MIGRATION_BUNDLE_URL_ENV, (tmp_path / "bundle.zip").as_uri())

    assert maybe_run() is False
    assert not (fresh_home / "config.yaml").exists()
    assert not (fresh_home / IMPORT_MARKER_NAME).exists()


def test_newer_schema_refuses_before_mutation(fresh_home, tmp_path, monkeypatch):
    bundle = tmp_path / "bundle.zip"
    _make_bundle(bundle, schema_version=MIGRATION_SCHEMA_VERSION + 1)
    monkeypatch.setenv(MIGRATION_BUNDLE_URL_ENV, bundle.as_uri())

    assert maybe_run() is True  # attempted
    assert not (fresh_home / "config.yaml").exists()
    assert not (fresh_home / IMPORT_MARKER_NAME).exists()
    assert not list(fresh_home.glob(".cloud-migration-bundle.*"))  # staged copy gone


def test_unreachable_url_logs_and_continues(fresh_home, monkeypatch):
    monkeypatch.setenv(MIGRATION_BUNDLE_URL_ENV, "https://127.0.0.1:1/nope.zip")
    assert maybe_run() is True  # attempted, transport failure swallowed
    assert not (fresh_home / "config.yaml").exists()
    assert not (fresh_home / IMPORT_MARKER_NAME).exists()


def test_plain_backup_zip_without_manifest_imports(fresh_home, tmp_path, monkeypatch):
    bundle = tmp_path / "bundle.zip"
    _make_bundle(bundle, manifest=False)
    monkeypatch.setenv(MIGRATION_BUNDLE_URL_ENV, bundle.as_uri())

    assert maybe_run() is True
    assert (fresh_home / "config.yaml").exists()


def test_schema_constant_matches_exporter_when_present():
    cloud_migrate = pytest.importorskip("hermes_cli.cloud_migrate")
    assert cloud_migrate.MIGRATION_SCHEMA_VERSION == MIGRATION_SCHEMA_VERSION


def test_failed_extract_leaves_no_marker_and_no_partial_copy(
    fresh_home, tmp_path, monkeypatch, capsys
):
    # A zip that is not actually a zip: schema gate refuses it, staged copy
    # is removed, no marker is left behind.
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"this is not a zip")
    monkeypatch.setenv(MIGRATION_BUNDLE_URL_ENV, bundle.as_uri())

    assert maybe_run() is True
    assert not (fresh_home / "config.yaml").exists()
    assert not (fresh_home / IMPORT_MARKER_NAME).exists()
    assert not list(fresh_home.glob(".cloud-migration-bundle.*"))