"""``gateway.bot_loop_reset_on_human`` is on only when the raw value is boolean True."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli.config import bot_loop_reset_on_human_enabled


def _write_yaml(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_missing_file_is_off(tmp_path: Path) -> None:
    missing = tmp_path / "absent" / "config.yaml"
    assert not missing.exists()
    assert bot_loop_reset_on_human_enabled(missing) is False


def test_missing_key_is_off(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path / "config.yaml",
        "gateway:\n  bot_loop_guard:\n    enabled: true\n",
    )
    assert bot_loop_reset_on_human_enabled(path) is False


def test_empty_file_is_off(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path / "config.yaml", "")
    assert bot_loop_reset_on_human_enabled(path) is False


def test_false_is_off(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path / "config.yaml",
        "gateway:\n  bot_loop_reset_on_human: false\n",
    )
    assert bot_loop_reset_on_human_enabled(path) is False


def test_true_is_on(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path / "config.yaml",
        "gateway:\n  bot_loop_reset_on_human: true\n",
    )
    assert bot_loop_reset_on_human_enabled(path) is True


@pytest.mark.parametrize("literal", ['"true"', "'true'", "1", "null"])
def test_non_bool_is_off(tmp_path: Path, literal: str) -> None:
    path = _write_yaml(
        tmp_path / "config.yaml",
        f"gateway:\n  bot_loop_reset_on_human: {literal}\n",
    )
    assert bot_loop_reset_on_human_enabled(path) is False


def test_top_level_true_is_not_the_gateway_flag(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path / "config.yaml", "bot_loop_reset_on_human: true\n")
    assert bot_loop_reset_on_human_enabled(path) is False


def test_reads_user_config_raw(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def _raw(config_path=None):
        seen["path"] = config_path
        return {"gateway": {"bot_loop_reset_on_human": True}}

    monkeypatch.setattr("hermes_cli.config.read_user_config_raw", _raw)
    path = tmp_path / "config.yaml"
    assert bot_loop_reset_on_human_enabled(path) is True
    assert seen["path"] == path
