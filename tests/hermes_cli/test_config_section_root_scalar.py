"""A scalar written to an *unset* mapping-section root must be refused.

`hermes config set terminal foo` (with `terminal:` absent from config.yaml) used to store
`terminal: foo` — a string in a slot whose every reader (`terminal_tool` → `terminal.cwd` /
`terminal.backend`, and siblings for `agent` / `gateway` / `display`) expects a mapping. The
runtime silently ignored it while `config get terminal` echoed it back as if live.

`_expected_container_type` resolved an unset single-segment mapping-section root to ``None``,
so `_refuse_container_type_mismatch` had no expected type to check and let the scalar through.
Now an unset section root stays ``"mapping"``; `--force` keeps its documented meaning (replace a
whole section); nested leaves and the bare `model` shorthand are unaffected.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import hermes_yaml as yaml

from hermes_cli.config import config_command


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _set(config_path: Path, key: str, value: str, *, force: bool = False):
    with patch.dict(os.environ, {"HERMES_HOME": str(config_path.parent)}):
        return config_command(_Args(config_command="set", key=key, value=value, force=force))


def test_set_scalar_on_unset_section_root_is_refused(tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("display:\n  compact: true\n", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        _set(config_path, "terminal", "foo")

    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "must be a mapping" in err and "nothing was written" in err
    assert config_path.read_text(encoding="utf-8") == "display:\n  compact: true\n"


def test_set_scalar_with_force_still_replaces_section(tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("display:\n  compact: true\n", encoding="utf-8")

    _set(config_path, "display", "scalar", force=True)

    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved["display"] == "scalar"


def test_set_nested_leaf_under_section_root_is_unchanged(tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("terminal:\n  backend: shell\n", encoding="utf-8")

    _set(config_path, "terminal.cwd", "/tmp")

    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved["terminal"] == {"backend": "shell", "cwd": "/tmp"}


def test_bare_model_shorthand_still_redirects(tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  provider: nous\n", encoding="utf-8")

    _set(config_path, "model", "gpt-4o")

    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    # the bare model shorthand preserves siblings and writes model.default
    assert saved["model"]["provider"] == "nous"
    assert saved["model"]["default"] == "gpt-4o"


def test_existing_section_scalar_still_refused_by_section_guard(tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("terminal:\n  backend: shell\n", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        _set(config_path, "terminal", "foo")

    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "configuration section" in err  # _guard_section_overwrite's copy
    assert config_path.read_text(encoding="utf-8") == "terminal:\n  backend: shell\n"