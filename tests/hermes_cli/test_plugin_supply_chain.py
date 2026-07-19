from __future__ import annotations

import dataclasses
import json
import os
import stat
from pathlib import Path

import pytest

from hermes_cli.plugin_supply_chain import (
    LOCK_FILENAME,
    MAX_LOCK_BYTES,
    PluginCapabilityReport,
    PluginProvenance,
    build_capability_report,
    read_provenance_lock,
    validate_full_commit_sha,
    validate_source_url,
    write_provenance_lock,
)


SHA = "0123456789abcdef0123456789abcdef01234567"


@pytest.mark.parametrize(
    "source_url",
    [
        "https://github.com/owner/repo.git",
        "git@github.com:owner/repo.git",
        "ssh://git@github.com/owner/repo.git",
        "file:///tmp/plugin-repo",
    ],
)
def test_validate_source_url_preserves_supported_clone_urls(source_url: str) -> None:
    assert validate_source_url(source_url) == source_url


@pytest.mark.parametrize(
    "source_url",
    [
        "https://user:secret@github.com/owner/repo.git",
        "https://github.com/owner/repo.git?token=secret",
    ],
)
def test_validate_source_url_rejects_credentials_and_query(source_url: str) -> None:
    with pytest.raises(ValueError, match="credentials|query"):
        validate_source_url(source_url)


def _provenance(**overrides: str | None) -> PluginProvenance:
    values: dict[str, str | None] = {
        "source_url": "https://github.com/owner/repo.git",
        "subdir": "plugins/example",
        "resolved_commit": SHA,
        "requested_ref": "main",
        "inspected_at": "2026-07-19T12:00:00Z",
    }
    values.update(overrides)
    return PluginProvenance(**values)


def test_models_are_frozen_dataclasses() -> None:
    provenance = _provenance()
    report = PluginCapabilityReport((), (), (), False, False, ())

    with pytest.raises(dataclasses.FrozenInstanceError):
        provenance.source_url = "changed"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.has_dashboard = True  # type: ignore[misc]


@pytest.mark.parametrize(
    "value",
    [
        "a" * 39,
        "a" * 41,
        "A" * 40,
        "g" * 40,
        "0" * 39 + " ",
        "",
        None,
        123,
    ],
)
def test_validate_full_commit_sha_rejects_noncanonical_values(value: object) -> None:
    with pytest.raises(ValueError, match="40 lowercase hexadecimal"):
        validate_full_commit_sha(value)


def test_validate_full_commit_sha_accepts_exact_lowercase_sha() -> None:
    assert validate_full_commit_sha(SHA) == SHA


def test_build_capability_report_uses_manifest_schema_without_executing_code(
    tmp_path: Path,
) -> None:
    (tmp_path / "dashboard").mkdir()
    (tmp_path / "dashboard" / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "after-install.md").write_text("instructions", encoding="utf-8")
    (tmp_path / "plugin.py").write_text("raise RuntimeError('must not execute')")
    manifest = {
        "hooks": ["post_tool_call", "pre_tool_call", "post_tool_call"],
        "provides_tools": ["z_tool", "a_tool", "z_tool"],
        "requires_env": [
            {"name": "Z_TOKEN", "secret": True},
            "A_TOKEN",
            {"name": "A_TOKEN"},
            {"description": "ignored without a name"},
        ],
        "tools": ["unsupported_field"],
        "required_env": ["unsupported_field"],
    }

    report = build_capability_report(tmp_path, manifest)

    assert report.hooks == ("post_tool_call", "pre_tool_call")
    assert report.tools == ("a_tool", "z_tool")
    assert report.required_env == ("A_TOKEN", "Z_TOKEN")
    assert report.has_dashboard is True
    assert report.has_after_install is True
    assert report.warnings == ("CAPABILITY_REPORT_IS_NOT_SECURITY_AUDIT",)


def test_build_capability_report_requires_actual_presence_markers(tmp_path: Path) -> None:
    (tmp_path / "dashboard").mkdir()
    (tmp_path / "after_install.py").write_text("raise RuntimeError")

    report = build_capability_report(tmp_path, {})

    assert report.has_dashboard is False
    assert report.has_after_install is False


def test_provenance_lock_round_trips_as_deterministic_json(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    provenance = _provenance()

    lock_path = write_provenance_lock(plugin_dir, provenance)

    assert lock_path == plugin_dir / LOCK_FILENAME
    assert lock_path.name == ".hermes-plugin-lock.json"
    expected = json.dumps(dataclasses.asdict(provenance), sort_keys=True, indent=2) + "\n"
    assert lock_path.read_text(encoding="utf-8") == expected
    assert read_provenance_lock(plugin_dir) == provenance


def test_read_provenance_lock_returns_none_when_absent(tmp_path: Path) -> None:
    assert read_provenance_lock(tmp_path) is None


@pytest.mark.parametrize(
    "change",
    [
        {"resolved_commit": "abc123"},
        {"subdir": "../escape"},
        {"subdir": "/absolute"},
        {"source_url": "https://user:secret@example.com/repo.git"},
        {"unexpected": "field"},
    ],
)
def test_read_provenance_lock_fails_closed_on_unsafe_content(
    tmp_path: Path, change: dict[str, str]
) -> None:
    data = dataclasses.asdict(_provenance())
    data.update(change)
    (tmp_path / LOCK_FILENAME).write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError):
        read_provenance_lock(tmp_path)


@pytest.mark.parametrize(
    "source_url",
    [
        "https://github.com/owner/repo.git?token=secret",
        "https://github.com/owner/repo.git#access_token=secret",
    ],
)
def test_read_provenance_lock_rejects_source_url_query_or_fragment(
    tmp_path: Path, source_url: str
) -> None:
    data = dataclasses.asdict(_provenance(source_url=source_url))
    (tmp_path / LOCK_FILENAME).write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="query or fragment"):
        read_provenance_lock(tmp_path)


@pytest.mark.parametrize(
    "source_url",
    [
        "https://github.com/owner/repo.git?token=secret",
        "https://github.com/owner/repo.git#access_token=secret",
    ],
)
def test_write_provenance_lock_rejects_source_url_query_or_fragment(
    tmp_path: Path, source_url: str
) -> None:
    with pytest.raises(ValueError, match="query or fragment"):
        write_provenance_lock(tmp_path, _provenance(source_url=source_url))


def test_read_provenance_lock_fails_closed_on_malformed_json(tmp_path: Path) -> None:
    (tmp_path / LOCK_FILENAME).write_text("not json", encoding="utf-8")

    with pytest.raises(ValueError):
        read_provenance_lock(tmp_path)


def test_write_provenance_lock_rejects_credentials_and_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_provenance_lock(
            tmp_path, _provenance(source_url="https://token@example.com/repo.git")
        )
    with pytest.raises(ValueError):
        write_provenance_lock(tmp_path, _provenance(subdir="plugin/../secret"))


@pytest.mark.parametrize(
    "source_url",
    [
        "git@github.com:owner/repo.git",
        "ssh://git@github.com/owner/repo.git",
    ],
)
def test_provenance_lock_accepts_conventional_git_transport_identity(
    tmp_path: Path, source_url: str
) -> None:
    provenance = _provenance(source_url=source_url)

    write_provenance_lock(tmp_path, provenance)

    assert read_provenance_lock(tmp_path) == provenance


@pytest.mark.parametrize(
    "source_url",
    [
        "TOKEN@github.com:owner/repo.git",
        "oauth2:SECRET@gitlab.example.com:owner/repo.git",
        "git@@github.com:owner/repo.git",
        "git@:owner/repo.git",
        "git@github.com:",
        "git@github.com/owner/repo.git",
        "git@github.com:owner/repo.git?token=secret",
        "git@github.com:owner/repo.git#secret",
        "git@github.com:owner/repo git",
        "https://token@github.com/owner/repo.git",
        "ssh://other@github.com/owner/repo.git",
        "ssh://git:secret@github.com/owner/repo.git",
    ],
)
def test_write_provenance_lock_rejects_credential_or_malformed_remote(
    tmp_path: Path, source_url: str
) -> None:
    with pytest.raises(ValueError, match="source_url"):
        write_provenance_lock(tmp_path, _provenance(source_url=source_url))


@pytest.mark.parametrize(
    "source_url",
    [
        "TOKEN@github.com:owner/repo.git",
        "oauth2:SECRET@gitlab.example.com:owner/repo.git",
        "git@@github.com:owner/repo.git",
        "git@:owner/repo.git",
        "git@github.com:",
    ],
)
def test_read_provenance_lock_rejects_credential_or_malformed_scp_remote(
    tmp_path: Path, source_url: str
) -> None:
    data = dataclasses.asdict(_provenance(source_url=source_url))
    (tmp_path / LOCK_FILENAME).write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="source_url"):
        read_provenance_lock(tmp_path)


def test_read_provenance_lock_rejects_oversized_real_file(tmp_path: Path) -> None:
    (tmp_path / LOCK_FILENAME).write_bytes(b" " * (MAX_LOCK_BYTES + 1))

    with pytest.raises(ValueError, match="malformed or unreadable"):
        read_provenance_lock(tmp_path)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits only")
def test_write_provenance_lock_is_owner_only(tmp_path: Path) -> None:
    lock_path = write_provenance_lock(tmp_path, _provenance())

    assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600
