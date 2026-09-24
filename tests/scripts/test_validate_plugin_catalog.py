"""Behavior tests for scripts/validate_plugin_catalog.py.

The script is the no-install structural validator used by the plugin-catalog
admission CI: it must run with only stdlib + pyyaml, take file paths or a
directory, exit 0/1, and support --json machine output. These tests exercise
the CLI contract via subprocess (the same way CI invokes it).
"""

import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "validate_plugin_catalog.py"

VALID_ENTRY = {
    "name": "example-plugin",
    "repo": "https://github.com/NousResearch/hermes-example-plugins",
    "sha": "38fe0fb53eff98d477f807432e965429e665ca33",
    "subdir": "",
    "description": "One-line description.",
    "maintainer": "NousResearch",
    "tier": "official",
    "requires_hermes": ">=0.19",
    "docs_url": "",
    "platforms": [],
    "capabilities": {
        "provides_tools": ["example_tool"],
        "provides_hooks": [],
        "provides_middleware": [],
        "requires_env": [],
    },
}


def write_entry(tmp_path: Path, data: dict, filename: str | None = None) -> Path:
    name = filename or f"{data.get('name', 'entry')}.yaml"
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def run_validator(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
    )


# ── valid input ────────────────────────────────────────────────────────


def test_valid_entry_passes(tmp_path):
    path = write_entry(tmp_path, VALID_ENTRY)
    result = run_validator(str(path))
    assert result.returncode == 0, result.stdout + result.stderr


def test_valid_entry_without_optional_fields_passes(tmp_path):
    entry = {
        "name": "minimal-plugin",
        "repo": "https://github.com/example/minimal",
        "sha": "a" * 40,
        "description": "Minimal.",
        "maintainer": "someone",
    }
    path = write_entry(tmp_path, entry)
    result = run_validator(str(path))
    assert result.returncode == 0, result.stdout + result.stderr


# ── each malformed field fails with a pointed error ────────────────────


def _expect_error(tmp_path, mutation: dict, expected_substring: str, drop: str = ""):
    entry = {**VALID_ENTRY, **mutation}
    if drop:
        entry.pop(drop, None)
    path = write_entry(tmp_path, entry, filename="entry.yaml")
    result = run_validator(str(path))
    combined = result.stdout + result.stderr
    assert result.returncode == 1, combined
    assert expected_substring in combined, combined
    assert "entry.yaml" in combined, combined


def test_bad_name_fails(tmp_path):
    _expect_error(tmp_path, {"name": "Bad Name!"}, "name")


def test_name_too_long_fails(tmp_path):
    _expect_error(tmp_path, {"name": "x" * 65}, "name")


def test_non_https_repo_fails(tmp_path):
    _expect_error(tmp_path, {"repo": "git@github.com:evil/x.git"}, "repo")


def test_short_sha_fails(tmp_path):
    _expect_error(tmp_path, {"sha": "abc123"}, "sha")


def test_non_hex_sha_fails(tmp_path):
    _expect_error(tmp_path, {"sha": "z" * 40}, "sha")


def test_bad_tier_fails(tmp_path):
    _expect_error(tmp_path, {"tier": "platinum"}, "tier")


def test_empty_description_fails(tmp_path):
    _expect_error(tmp_path, {"description": ""}, "description")


def test_empty_maintainer_fails(tmp_path):
    _expect_error(tmp_path, {"maintainer": ""}, "maintainer")


def test_missing_required_field_fails(tmp_path):
    _expect_error(tmp_path, {}, "sha", drop="sha")


def test_capabilities_value_not_a_list_fails(tmp_path):
    _expect_error(
        tmp_path,
        {"capabilities": {"provides_tools": "not-a-list"}},
        "provides_tools",
    )


def test_capabilities_list_of_non_strings_fails(tmp_path):
    _expect_error(
        tmp_path,
        {"capabilities": {"requires_env": [1, 2]}},
        "requires_env",
    )


def test_bad_requires_hermes_spec_fails(tmp_path):
    _expect_error(tmp_path, {"requires_hermes": "banana"}, "requires_hermes")


def test_comma_separated_requires_hermes_passes(tmp_path):
    entry = {**VALID_ENTRY, "requires_hermes": ">=0.19, <2.0"}
    path = write_entry(tmp_path, entry)
    result = run_validator(str(path))
    assert result.returncode == 0, result.stdout + result.stderr


def test_unknown_platform_fails(tmp_path):
    _expect_error(tmp_path, {"platforms": ["linux", "amiga"]}, "platforms")


def test_entry_not_a_mapping_fails(tmp_path):
    path = tmp_path / "entry.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    result = run_validator(str(path))
    assert result.returncode == 1
    assert "mapping" in (result.stdout + result.stderr)


# ── unknown top-level keys warn but do not fail ────────────────────────


def test_unknown_key_warns_but_passes(tmp_path):
    entry = {**VALID_ENTRY, "future_field": "hello"}
    path = write_entry(tmp_path, entry)
    result = run_validator(str(path))
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "future_field" in combined
    assert "warning" in combined.lower()


# ── removed.yaml shape ─────────────────────────────────────────────────


def test_valid_removed_yaml_passes(tmp_path):
    path = tmp_path / "removed.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "removed": [
                    {
                        "name": "some-plugin",
                        "repo": "https://github.com/evil/some-plugin",
                        "reason": "Exfiltrated env vars",
                        "date": "2026-07-02",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    result = run_validator(str(path))
    assert result.returncode == 0, result.stdout + result.stderr


def test_removed_yaml_not_a_list_fails(tmp_path):
    path = tmp_path / "removed.yaml"
    path.write_text(yaml.safe_dump({"removed": "nope"}), encoding="utf-8")
    result = run_validator(str(path))
    assert result.returncode == 1
    assert "removed" in (result.stdout + result.stderr)


def test_removed_item_missing_name_fails(tmp_path):
    path = tmp_path / "removed.yaml"
    path.write_text(
        yaml.safe_dump({"removed": [{"reason": "bad", "date": "2026-01-01"}]}),
        encoding="utf-8",
    )
    result = run_validator(str(path))
    assert result.returncode == 1
    assert "name" in (result.stdout + result.stderr)


# ── --json machine output ──────────────────────────────────────────────


def test_json_output_shape_on_failure(tmp_path):
    bad = write_entry(tmp_path, {**VALID_ENTRY, "sha": "short"}, filename="bad.yaml")
    result = run_validator("--json", str(bad))
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert isinstance(payload["files"], list)
    entry = next(f for f in payload["files"] if f["path"].endswith("bad.yaml"))
    assert entry["ok"] is False
    assert any("sha" in e for e in entry["errors"])


def test_json_output_shape_on_success_with_warning(tmp_path):
    good = write_entry(tmp_path, {**VALID_ENTRY, "future_field": 1})
    result = run_validator("--json", str(good))
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    (entry,) = payload["files"]
    assert entry["ok"] is True
    assert entry["errors"] == []
    assert any("future_field" in w for w in entry["warnings"])


# ── directory mode ─────────────────────────────────────────────────────


def test_directory_mode_validates_all_entries_and_removed(tmp_path):
    write_entry(tmp_path, VALID_ENTRY)
    write_entry(tmp_path, {**VALID_ENTRY, "name": "bad-one", "sha": "nope"})
    (tmp_path / "removed.yaml").write_text(
        yaml.safe_dump({"removed": [{"name": "gone", "reason": "test"}]}),
        encoding="utf-8",
    )
    result = run_validator(str(tmp_path))
    combined = result.stdout + result.stderr
    assert result.returncode == 1
    assert "bad-one.yaml" in combined
    # the valid entry and removed.yaml must not produce errors
    assert combined.count("ERROR") == combined.count("bad-one.yaml: ERROR")


def test_directory_mode_all_valid_exits_zero(tmp_path):
    write_entry(tmp_path, VALID_ENTRY)
    (tmp_path / "removed.yaml").write_text(
        yaml.safe_dump({"removed": []}), encoding="utf-8"
    )
    result = run_validator(str(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
