"""``gateway.bot_loop_guard.reset_all_on_human`` is read from the install-root config.yaml."""

from __future__ import annotations

from pathlib import Path

import pytest

from gateway.bot_loop_reset import reset_all_on_human


def _write(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def test_missing_file_is_false(tmp_path: Path) -> None:
    missing = tmp_path / "absent"
    assert not (missing / "config.yaml").exists()
    assert reset_all_on_human(missing) is False


def test_missing_key_is_false(tmp_path: Path) -> None:
    _write(tmp_path / "config.yaml", "gateway:\n  bot_loop_guard:\n    enabled: true\n")
    assert reset_all_on_human(tmp_path) is False


def test_sibling_key_does_not_enable(tmp_path: Path) -> None:
    _write(tmp_path / "config.yaml", "gateway:\n  bot_loop_reset_on_human: true\n")
    assert reset_all_on_human(tmp_path) is False


def test_nested_true_enables(tmp_path: Path) -> None:
    _write(
        tmp_path / "config.yaml",
        "gateway:\n  bot_loop_guard:\n    reset_all_on_human: true\n",
    )
    assert reset_all_on_human(tmp_path) is True


def test_nested_false_is_off(tmp_path: Path) -> None:
    _write(
        tmp_path / "config.yaml",
        "gateway:\n  bot_loop_guard:\n    reset_all_on_human: false\n",
    )
    assert reset_all_on_human(tmp_path) is False


@pytest.mark.parametrize(
    ("literal", "expected"),
    [('"yes"', True), ("'yes'", True), ('"no"', False), ("'no'", False)],
)
def test_string_literals_use_as_bool(tmp_path: Path, literal: str, expected: bool) -> None:
    _write(
        tmp_path / "config.yaml",
        f"gateway:\n  bot_loop_guard:\n    reset_all_on_human: {literal}\n",
    )
    assert reset_all_on_human(tmp_path) is expected


def test_reads_root_config_yaml_not_load_settings(tmp_path: Path, monkeypatch) -> None:
    seen: dict[str, object] = {}

    def _raw(config_path=None):
        seen["path"] = config_path
        return {"gateway": {"bot_loop_guard": {"reset_all_on_human": "yes"}}}

    def _boom(*_args, **_kwargs):
        raise AssertionError("load_config/load_settings must not be consulted")

    monkeypatch.setattr("hermes_cli.config.read_user_config_raw", _raw)
    monkeypatch.setattr("hermes_cli.config.load_config", _boom)
    monkeypatch.setattr("gateway.bot_loop_guard.load_settings", _boom)
    assert reset_all_on_human(tmp_path) is True
    assert seen["path"] == tmp_path / "config.yaml"


def test_install_root_wins_over_profile_config(tmp_path: Path, monkeypatch) -> None:
    """HERMES_HOME is the profile; the flag is read from the install root only."""
    root = tmp_path / "install"
    profile = root / "profiles" / "grid-lead"
    profile.mkdir(parents=True)
    _write(
        root / "config.yaml",
        "gateway:\n  bot_loop_guard:\n    reset_all_on_human: true\n",
    )
    _write(
        profile / "config.yaml",
        "gateway:\n  bot_loop_guard:\n    reset_all_on_human: false\n",
    )
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setattr("gateway.bot_loop_reset.get_hermes_root", lambda: root)
    assert reset_all_on_human() is True
