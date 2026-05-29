"""Tests for utils.atomic_yaml_write — crash-safe YAML file writes."""

import os
import stat
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from utils import atomic_yaml_write


class TestAtomicYamlWrite:
    def test_writes_valid_yaml(self, tmp_path):
        target = tmp_path / "data.yaml"
        data = {"key": "value", "nested": {"a": 1}}

        atomic_yaml_write(target, data)

        assert yaml.safe_load(target.read_text(encoding="utf-8")) == data

    def test_cleans_up_temp_file_on_baseexception(self, tmp_path):
        class SimulatedAbort(BaseException):
            pass

        target = tmp_path / "data.yaml"
        original = {"preserved": True}
        target.write_text(yaml.safe_dump(original), encoding="utf-8")

        with patch("utils.yaml.dump", side_effect=SimulatedAbort):
            with pytest.raises(SimulatedAbort):
                atomic_yaml_write(target, {"new": True})

        tmp_files = [f for f in tmp_path.iterdir() if ".tmp" in f.name]
        assert len(tmp_files) == 0
        assert yaml.safe_load(target.read_text(encoding="utf-8")) == original

    def test_appends_extra_content(self, tmp_path):
        target = tmp_path / "data.yaml"

        atomic_yaml_write(target, {"key": "value"}, extra_content="\n# comment\n")

        text = target.read_text(encoding="utf-8")
        assert "key: value" in text
        assert "# comment" in text

    @pytest.mark.skipif(os.name != "posix", reason="POSIX-only mode assertion")
    def test_first_create_keeps_default_temp_permissions_without_create_mode(self, tmp_path):
        target = tmp_path / "default_mode.yaml"

        atomic_yaml_write(target, {"ok": True})

        mode = stat.S_IMODE(target.stat().st_mode)
        assert mode == 0o600

    @pytest.mark.skipif(os.name != "posix", reason="POSIX-only mode assertion")
    def test_first_create_applies_explicit_create_mode(self, tmp_path):
        target = tmp_path / "explicit_mode.yaml"

        atomic_yaml_write(target, {"ok": True}, create_mode=0o644)

        mode = stat.S_IMODE(target.stat().st_mode)
        assert mode == 0o644
