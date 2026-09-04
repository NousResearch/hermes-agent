from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_policy(
    tmp_path: Path, yaml_text: str | None = None
) -> subprocess.CompletedProcess[str]:
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    if yaml_text is not None:
        (hermes_home / "config.yaml").write_text(
            textwrap.dedent(yaml_text).lstrip(), encoding="utf-8"
        )

    return _run_policy_home(hermes_home)


def _run_policy_home(hermes_home: Path) -> subprocess.CompletedProcess[str]:

    env = os.environ.copy()
    env["HERMES_HOME"] = str(hermes_home)
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.update_preflight_policy"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


def _payload(result: subprocess.CompletedProcess[str]) -> dict[str, object]:
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout.count("\n") == 1
    return json.loads(result.stdout)


def test_policy_process_is_zero_write_with_missing_or_existing_config(tmp_path: Path):
    missing_home = tmp_path / "missing-home"
    missing_home.mkdir()
    assert _payload(_run_policy_home(missing_home))["mode"] == "quick"
    assert list(missing_home.iterdir()) == []

    existing_home = tmp_path / "existing-home"
    existing_home.mkdir()
    config_path = existing_home / "config.yaml"
    config_path.write_text(
        "updates:\n  pre_update_backup: false\n  backup_keep: 1\n",
        encoding="utf-8",
    )
    before = (config_path.read_bytes(), config_path.stat().st_mtime_ns)
    assert _payload(_run_policy_home(existing_home))["mode"] == "off"
    assert [path.name for path in existing_home.iterdir()] == ["config.yaml"]
    assert (config_path.read_bytes(), config_path.stat().st_mtime_ns) == before


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("off", "off"),
        ("quick", "quick"),
        ("full", "full"),
        ("false", "off"),
        ("true", "full"),
    ],
)
def test_policy_process_resolves_modes_and_legacy_booleans(
    tmp_path: Path, configured: str, expected: str
):
    result = _run_policy(
        tmp_path,
        f"""
        updates:
          pre_update_backup: {configured}
          backup_keep: 7
        """,
    )

    assert _payload(result) == {
        "mode": expected,
        "backup_keep": 7,
        "quick_keep": 1,
        "quick_max_file_size": 1 << 30,
    }


def test_policy_process_uses_canonical_defaults_when_keys_are_missing(tmp_path: Path):
    result = _run_policy(tmp_path, "updates: {}\n")

    assert _payload(result) == {
        "mode": "quick",
        "backup_keep": 5,
        "quick_keep": 1,
        "quick_max_file_size": 1 << 30,
    }


@pytest.mark.parametrize("configured", [0, -5])
def test_policy_process_floors_backup_keep_to_one(tmp_path: Path, configured: int):
    result = _run_policy(
        tmp_path,
        f"""
        updates:
          pre_update_backup: full
          backup_keep: {configured}
        """,
    )

    assert _payload(result)["backup_keep"] == 1


@pytest.mark.parametrize(
    "yaml_text",
    [
        """
        updates:
          pre_update_backup: sometimes
          backup_keep: 5
        """,
        """
        updates:
          pre_update_backup: full
          backup_keep: many
        """,
        """
        updates:
          pre_update_backup: null
          backup_keep: 5
        """,
        "updates: []\n",
    ],
)
def test_policy_process_fails_closed_on_ambiguous_or_malformed_config(
    tmp_path: Path, yaml_text: str
):
    result = _run_policy(tmp_path, yaml_text)

    assert result.returncode == 2
    assert result.stdout == ""
    assert "pre-update policy resolution failed" in result.stderr


def test_cli_resolver_preserves_the_legacy_unknown_value_fallback(monkeypatch, caplog):
    from argparse import Namespace

    import hermes_cli.config as config_module
    from hermes_cli.update_cmd import (
        _normalize_pre_update_backup_keep,
        _resolve_pre_update_backup_mode,
    )

    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda: {"updates": {"pre_update_backup": "surprise"}},
    )

    assert (
        _resolve_pre_update_backup_mode(Namespace(no_backup=False, backup=False))
        == "quick"
    )
    assert "using 'quick'" in caplog.text
    assert _normalize_pre_update_backup_keep(0) == 1
    with pytest.raises(ValueError):
        _normalize_pre_update_backup_keep("many")
