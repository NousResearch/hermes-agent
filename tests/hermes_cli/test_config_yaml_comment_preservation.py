"""Regression guard for #92554: every config.yaml writer preserves user comments and key order.

Each test seeds a hand-commented config.yaml, runs one production write path against it and
asserts that every comment survives, top-level key order is unchanged, and the written value
landed. A new writer that reaches for PyYAML instead of ``atomic_config_write`` fails here or in
``scripts/check_config_yaml_writers.py`` (also exercised below).
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import hermes_yaml as yaml

REPO = Path(__file__).resolve().parents[2]

COMMENTED = """\
# TOP COMMENT — rationale for this whole file must survive
_config_version: 12
model:
  provider: test   # pinned for eval reproducibility
  default: some-model

plugins:
  # rationale for the enabled list
  enabled:
    - alpha  # kept for the demo account
approvals:
  mode: "off"   # quoted on purpose: PyYAML reads bare off as False
hooks:
  pre_tool_call:
    # why this hook exists
    - command: /bin/echo
      timeout: 10
"""

COMMENTS = [
    "# TOP COMMENT — rationale for this whole file must survive",
    "# pinned for eval reproducibility",
    "# rationale for the enabled list",
    "# kept for the demo account",
    "# quoted on purpose: PyYAML reads bare off as False",
    "# why this hook exists",
]
KEY_ORDER = ["_config_version", "model", "plugins", "approvals", "hooks"]


@pytest.fixture
def home(tmp_path):
    (tmp_path / ".env").touch()
    (tmp_path / "config.yaml").write_text(COMMENTED, encoding="utf-8")
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        from hermes_cli import config as config_mod
        config_mod._RAW_CONFIG_CACHE.clear()
        yield tmp_path


def _assert_preserved(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    missing = [c for c in COMMENTS if c not in text]
    assert not missing, f"comments destroyed by the write: {missing}\n---\n{text}"
    data = yaml.safe_load(text)
    top = [k for k in data if k in KEY_ORDER]
    assert top == KEY_ORDER, f"key order changed: {top}\n---\n{text}"
    assert data["approvals"]["mode"] == "off"  # not coerced to False
    assert "── Security ──" not in text, "stock boilerplate was appended to an existing file"
    return data


class TestEveryWriterPreservesComments:
    def test_config_set(self, home):
        from hermes_cli.config import set_config_value

        set_config_value("streaming.enabled", "true")
        data = _assert_preserved(home / "config.yaml")
        assert data["streaming"]["enabled"] is True

    def test_config_unset(self, home):
        from hermes_cli.config import unset_config_value

        unset_config_value("model.default")
        data = _assert_preserved(home / "config.yaml")
        assert "default" not in data["model"]

    def test_save_config_plugin_enable_and_memory_provider(self, home):
        """The bulk writer behind ``plugins enable``, ``memory setup``, the wizard and the dashboard."""
        from hermes_cli.config import load_config, save_config

        cfg = load_config()
        cfg["plugins"]["enabled"].append("beta")
        cfg.setdefault("memory", {})["provider"] = "honcho"
        save_config(cfg)
        data = _assert_preserved(home / "config.yaml")
        assert data["plugins"]["enabled"] == ["alpha", "beta"]
        assert data["memory"]["provider"] == "honcho"

    def test_migration_version_bump(self, home):
        from hermes_cli.config import check_config_version, migrate_config

        _, latest = check_config_version()
        migrate_config(interactive=False, quiet=True)
        data = _assert_preserved(home / "config.yaml")
        assert data["_config_version"] == latest

    def test_atomic_config_write_direct(self, home):
        """Direct callers (auth provider reset, gateway slash commands, telegram, doctor)."""
        from hermes_cli.config import atomic_config_write, read_user_config_raw

        raw = read_user_config_raw(home / "config.yaml")
        raw["model"]["provider"] = "auto"
        atomic_config_write(home / "config.yaml", raw)
        data = _assert_preserved(home / "config.yaml")
        assert data["model"]["provider"] == "auto"

    def test_boilerplate_only_on_create(self, home):
        from hermes_cli.config import save_config

        (home / "config.yaml").unlink()
        save_config({"model": {"provider": "test"}})
        text = (home / "config.yaml").read_text(encoding="utf-8")
        assert "── Security ──" in text  # fresh file gets the commented examples once
        save_config({"model": {"provider": "test", "default": "m"}})
        assert (home / "config.yaml").read_text(encoding="utf-8").count("── Security ──") == 1


class TestStaticGuard:

    def test_guard_flags_pyyaml_dump_of_config_path(self, tmp_path):
        sys.path.insert(0, str(REPO / "scripts"))
        try:
            import check_config_yaml_writers as guard
        finally:
            sys.path.pop(0)
        bad = tmp_path / "hermes_cli" / "bad_writer.py"
        bad.parent.mkdir()
        bad.write_text(
            "import yaml\nfrom utils import atomic_yaml_write\n"
            "def a(config_path, cfg):\n    atomic_yaml_write(config_path, cfg)\n"
            "def b(cfg_path, cfg):\n    cfg_path.write_text(yaml.safe_dump(cfg))\n"
            "def c(other_path, cfg):\n    atomic_yaml_write(other_path, cfg)\n",
            encoding="utf-8")
        with patch.object(guard, "ROOT", tmp_path):
            problems = guard.scan_file(bad)
        assert [p.split(":")[1] for p in problems] == ["4", "6"], problems
