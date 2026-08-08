"""Regression coverage for fail-closed non-interactive config startup."""

from __future__ import annotations

import logging
import os
import sys
import types
from argparse import Namespace

import pytest


def _args(**overrides) -> Namespace:
    values = {
        "command": "chat",
        "ignore_user_config": False,
        "oneshot": None,
        "query": "hello",
        "quiet": False,
        "safe_mode": False,
        "yolo": False,
    }
    values.update(overrides)
    return Namespace(**values)


@pytest.fixture(autouse=True)
def _isolated_config_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    yield
    os.environ.pop("HERMES_IGNORE_USER_CONFIG", None)


@pytest.mark.parametrize(
    "args",
    [
        _args(query="hello"),
        _args(command=None, query=None, oneshot="hello"),
        _args(query=None, quiet=True),
    ],
    ids=["single-query", "oneshot", "quiet"],
)
def test_noninteractive_guard_rejects_malformed_yaml(args, tmp_path, caplog, capsys):
    from hermes_cli import main as main_mod

    broken = "model: [unterminated\n"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(broken, encoding="utf-8")

    with caplog.at_level(logging.ERROR, logger="hermes_cli.config"):
        with pytest.raises(SystemExit) as exc_info:
            main_mod._guard_noninteractive_user_config(args)

    assert exc_info.value.code == 2
    assert "Refusing non-interactive startup" in capsys.readouterr().err
    assert any(
        record.levelno == logging.ERROR
        and "Refusing non-interactive startup" in record.getMessage()
        for record in caplog.records
    )
    assert config_path.read_text(encoding="utf-8") == broken
    backups = list(tmp_path.glob("config.yaml.corrupt.*.bak"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == broken


def test_prepare_rejects_bad_config_before_plugin_discovery(monkeypatch, tmp_path):
    from hermes_cli import main as main_mod

    (tmp_path / "config.yaml").write_text("model: [unterminated\n")
    discovery_calls = []
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(
            discover_plugins=lambda: discovery_calls.append("plugins")
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        main_mod._prepare_agent_startup(_args())

    assert exc_info.value.code == 2
    assert discovery_calls == []


@pytest.mark.parametrize(
    "content", [None, "", "{}\n", "model:\n  default: local/test\n"]
)
def test_noninteractive_guard_accepts_missing_empty_and_mapping_configs(
    content, tmp_path
):
    from hermes_cli import main as main_mod

    if content is not None:
        (tmp_path / "config.yaml").write_text(content, encoding="utf-8")
    args = _args()

    main_mod._guard_noninteractive_user_config(args)

    assert args._noninteractive_config_validated is True


def test_noninteractive_guard_rejects_non_mapping_yaml(tmp_path, capsys):
    from hermes_cli import main as main_mod

    (tmp_path / "config.yaml").write_text("- model\n- provider\n")

    with pytest.raises(SystemExit) as exc_info:
        main_mod._guard_noninteractive_user_config(_args())

    assert exc_info.value.code == 2
    assert "top-level YAML value must be a mapping" in capsys.readouterr().err


@pytest.mark.parametrize(
    "args",
    [
        _args(ignore_user_config=True),
        _args(safe_mode=True),
    ],
    ids=["ignore-user-config", "safe-mode"],
)
def test_explicit_config_bypasses_allow_noninteractive_recovery(args, tmp_path):
    from hermes_cli import main as main_mod

    (tmp_path / "config.yaml").write_text("model: [unterminated\n")

    main_mod._guard_noninteractive_user_config(args)

    assert args._noninteractive_config_validated is True
    assert list(tmp_path.glob("config.yaml.corrupt.*.bak")) == []


def test_interactive_chat_keeps_existing_repair_behavior(tmp_path):
    from hermes_cli import main as main_mod

    (tmp_path / "config.yaml").write_text("model: [unterminated\n")
    args = _args(query=None)

    main_mod._guard_noninteractive_user_config(args)

    assert not hasattr(args, "_noninteractive_config_validated")
    assert list(tmp_path.glob("config.yaml.corrupt.*.bak")) == []


def test_ignore_user_config_is_applied_before_oneshot_startup(monkeypatch):
    from hermes_cli import main as main_mod

    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)

    main_mod._apply_user_config_bypass(_args(ignore_user_config=True))

    assert os.environ["HERMES_IGNORE_USER_CONFIG"] == "1"
