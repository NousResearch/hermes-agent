"""Durable last-known-good recovery for profile-local config.yaml."""

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import yaml

from hermes_cli.config import atomic_config_write, load_config, save_config


LKG_NAME = "config.validated.yaml"


def _run_load(repo: Path, home: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = str(repo)
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; from hermes_cli.config import load_config; "
            "print(json.dumps(load_config()))",
        ],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def test_successful_load_atomically_maintains_restrictive_profile_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "model:\n  default: test/secure\napprovals:\n  deny:\n    - 'curl*evil*'\n",
        encoding="utf-8",
    )

    loaded = load_config()

    snapshot = tmp_path / LKG_NAME
    assert loaded["approvals"]["deny"] == ["curl*evil*"]
    assert yaml.safe_load(snapshot.read_text(encoding="utf-8"))["approvals"]["deny"] == [
        "curl*evil*"
    ]
    if os.name == "posix":
        assert stat.S_IMODE(snapshot.stat().st_mode) == 0o600
    assert not list(tmp_path.glob(f".{snapshot.stem}_*.tmp"))


def test_fresh_process_uses_durable_snapshot_and_leaves_broken_config_untouched(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    config_path = tmp_path / "config.yaml"
    good = "model:\n  default: test/secure\napprovals:\n  deny:\n    - 'curl*evil*'\n"
    config_path.write_text(good, encoding="utf-8")
    first = _run_load(repo, tmp_path)
    assert json.loads(first.stdout)["approvals"]["deny"] == ["curl*evil*"]

    broken = "approvals:\n  deny: [unclosed\n"
    config_path.write_text(broken, encoding="utf-8")
    second = _run_load(repo, tmp_path)

    recovered = json.loads(second.stdout)
    assert recovered["model"]["default"] == "test/secure"
    assert recovered["approvals"]["deny"] == ["curl*evil*"]
    assert config_path.read_text(encoding="utf-8") == broken
    assert "DURABLE LAST-KNOWN-GOOD" in second.stderr
    assert "security policy" in second.stderr


def test_absent_or_corrupt_snapshot_falls_back_loudly_without_touching_config(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    broken = "model: [unclosed\n"
    (tmp_path / "config.yaml").write_text(broken, encoding="utf-8")

    absent = _run_load(repo, tmp_path)
    assert "model" in json.loads(absent.stdout)
    assert "NO VALID LAST-KNOWN-GOOD" in absent.stderr
    assert (tmp_path / "config.yaml").read_text(encoding="utf-8") == broken

    (tmp_path / LKG_NAME).write_text("also: [broken\n", encoding="utf-8")
    corrupt = _run_load(repo, tmp_path)
    assert "model" in json.loads(corrupt.stdout)
    assert "last-known-good snapshot is unusable" in corrupt.stderr
    assert (tmp_path / LKG_NAME).read_text(encoding="utf-8") == "also: [broken\n"


def test_save_and_shared_atomic_writer_refresh_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    save_config({"model": {"default": "test/saved"}, "approvals": {"deny": ["rm *"]}})
    snapshot = yaml.safe_load((tmp_path / LKG_NAME).read_text(encoding="utf-8"))
    assert snapshot["model"]["default"] == "test/saved"
    assert snapshot["approvals"]["deny"] == ["rm *"]

    atomic_config_write(
        tmp_path / "config.yaml",
        {"model": {"default": "test/atomic"}, "approvals": {"deny": ["sudo *"]}},
    )
    snapshot = yaml.safe_load((tmp_path / LKG_NAME).read_text(encoding="utf-8"))
    assert snapshot["model"]["default"] == "test/atomic"
    assert snapshot["approvals"]["deny"] == ["sudo *"]


def test_snapshot_is_profile_local(tmp_path, monkeypatch):
    one = tmp_path / "profiles" / "one"
    two = tmp_path / "profiles" / "two"
    one.mkdir(parents=True)
    two.mkdir(parents=True)

    monkeypatch.setenv("HERMES_HOME", str(one))
    save_config({"model": {"default": "test/one"}})
    monkeypatch.setenv("HERMES_HOME", str(two))
    save_config({"model": {"default": "test/two"}})

    assert yaml.safe_load((one / LKG_NAME).read_text())["model"]["default"] == "test/one"
    assert yaml.safe_load((two / LKG_NAME).read_text())["model"]["default"] == "test/two"


def test_snapshot_preserves_env_reference_instead_of_expanded_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("PRIVATE_TOKEN", "expanded-secret-must-not-be-persisted")
    (tmp_path / "config.yaml").write_text(
        "custom_providers:\n"
        "  - name: private\n"
        "    base_url: https://example.invalid/v1\n"
        "    api_key: ${PRIVATE_TOKEN}\n",
        encoding="utf-8",
    )

    loaded = load_config()

    assert loaded["custom_providers"][0]["api_key"] == "expanded-secret-must-not-be-persisted"
    snapshot_text = (tmp_path / LKG_NAME).read_text(encoding="utf-8")
    assert "${PRIVATE_TOKEN}" in snapshot_text
    assert "expanded-secret-must-not-be-persisted" not in snapshot_text


def test_structurally_invalid_config_does_not_replace_validated_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  default: test/good\n", encoding="utf-8")
    load_config()
    before = (tmp_path / LKG_NAME).read_text(encoding="utf-8")

    # Parses as YAML, but validation classifies this shape as an error.
    config_path.write_text("custom_providers:\n  name: broken-shape\n", encoding="utf-8")
    load_config()

    assert (tmp_path / LKG_NAME).read_text(encoding="utf-8") == before
