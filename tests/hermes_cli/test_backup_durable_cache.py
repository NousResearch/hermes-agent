"""Regression tests for portable durable cache members during import."""

from argparse import Namespace
from pathlib import Path

from hermes_cli.backup import _import_member_rel, _import_skipped


def test_durable_cache_members_are_not_classified_as_runtime_state():
    """Durable citation/media caches must be restored, while regenerable cache is skipped."""
    assert not _import_skipped("cache/citations/ledger.json")
    assert not _import_skipped("profiles/coder/cache/images/delivered.png")
    assert _import_skipped("cache/model_catalog.json")
    assert _import_skipped("profiles/coder/cache/model_catalog.json")
    assert not _import_skipped("cache/citations/ledger.json-wal")


def test_import_preflight_accepts_durable_cache_members():
    """Preflight must classify the same durable entries as the publish path."""
    assert _import_member_rel("archive/cache/citations/ledger.json", "archive/") == (
        "cache/citations/ledger.json", False)
    assert _import_member_rel("archive/profiles/coder/cache/images/delivered.png", "archive/") == (
        "profiles/coder/cache/images/delivered.png", False)


def test_backup_import_round_trip_restores_root_and_profile_durable_cache(tmp_path, monkeypatch):
    """Import restores durable cache content that backup intentionally preserved."""
    from hermes_cli.backup import run_backup, run_import

    source_home = tmp_path / "source" / ".hermes"
    (source_home / "cache" / "citations").mkdir(parents=True)
    (source_home / "profiles" / "coder" / "cache" / "images").mkdir(parents=True)
    (source_home / "config.yaml").write_text("model: test\n", encoding="utf-8")
    (source_home / "profiles" / "coder" / "config.yaml").write_text(
        "model: profile-test\n", encoding="utf-8"
    )
    (source_home / "cache" / "citations" / "ledger.json").write_text(
        '{"citation": "durable"}\n', encoding="utf-8"
    )
    (source_home / "profiles" / "coder" / "cache" / "images" / "delivered.png").write_bytes(
        b"durable-media"
    )

    archive = tmp_path / "backup.zip"
    monkeypatch.setenv("HERMES_HOME", str(source_home))
    monkeypatch.setattr(Path, "home", lambda: source_home.parent)
    assert run_backup(Namespace(output=str(archive)))

    restored_home = tmp_path / "restored" / ".hermes"
    restored_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(restored_home))
    monkeypatch.setattr(Path, "home", lambda: restored_home.parent)
    run_import(Namespace(zipfile=str(archive), force=True))

    assert (restored_home / "cache" / "citations" / "ledger.json").read_text(
        encoding="utf-8"
    ) == '{"citation": "durable"}\n'
    assert (restored_home / "profiles" / "coder" / "cache" / "images" / "delivered.png").read_bytes() == (
        b"durable-media"
    )