"""The ConfigBackend seam (config-config design §4.1, D10/D11) and its reader gate."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import config_backend as cb

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.delenv(cb.BACKEND_ENV, raising=False)
    h = tmp_path / "home"
    h.mkdir()
    return h


class TestSelection:

    def test_default_is_file(self, home):
        assert cb.get_config_backend().name == "file"
        assert cb.supports_file_tooling() is True

    @pytest.mark.parametrize("value", ["remote", "bogus"])
    def test_unavailable_backend_fails_closed(self, home, monkeypatch, value):
        # No config value can select the backend, and an unknown one never falls back to the file.
        monkeypatch.setenv(cb.BACKEND_ENV, value)
        with pytest.raises(cb.ConfigBackendUnavailable):
            cb.get_config_backend()
        with pytest.raises(cb.ConfigBackendUnavailable):
            cb.read_config_doc(home / "config.yaml")

    def test_explicit_non_config_file_ignores_backend(self, home, monkeypatch):
        other = home / "import-source.yaml"
        other.write_text("a: 1\n", encoding="utf-8")
        monkeypatch.setenv(cb.BACKEND_ENV, "remote")
        assert cb.read_config_doc(other) == {"a": 1}


class TestFileBackend:

    def test_missing(self, home):
        path = home / "config.yaml"
        assert cb.config_exists(path) is False
        with pytest.raises(FileNotFoundError):
            cb.config_version(path)
        with pytest.raises(FileNotFoundError):
            cb.read_config_doc(path)

    def test_read_version_and_bom(self, home):
        path = home / "config.yaml"
        path.write_bytes("\ufeffmodel:\n  default: x\n".encode("utf-8"))
        assert cb.config_exists(path)
        assert cb.read_config_doc(path) == {"model": {"default": "x"}}
        assert cb.read_config_doc_readonly(path) == {"model": {"default": "x"}}
        v1 = cb.config_version(path)
        path.write_text("model:\n  default: longer-value\n", encoding="utf-8")
        assert cb.config_version(path) != v1

    def test_writes_preserve_comments(self, home):
        path = home / "config.yaml"
        path.write_text("# keep me\nmodel:\n  default: a  # inline\n", encoding="utf-8")
        cb.write_config_key(path, "model.default", "b")
        cb.write_config_key(path, "display.personality", "kawaii")
        text = path.read_text(encoding="utf-8")
        assert "# keep me" in text and "# inline" in text
        assert cb.read_config_doc(path) == {"model": {"default": "b"}, "display": {"personality": "kawaii"}}
        cb.write_config_document(path, {"model": {"default": "c"}})
        assert "# keep me" in path.read_text(encoding="utf-8")
        assert cb.read_config_doc(path) == {"model": {"default": "c"}}


class TestReaderGate:

    def _guard(self):
        sys.path.insert(0, str(REPO / "scripts"))
        try:
            import check_config_yaml_readers as guard
        finally:
            sys.path.pop(0)
        return guard

    def test_flags_direct_readers(self, tmp_path):
        guard = self._guard()
        bad = tmp_path / "hermes_cli" / "bad_reader.py"
        bad.parent.mkdir()
        bad.write_text(
            "import yaml\n"
            "def a(home):\n    p = home / 'config.yaml'\n    return yaml.safe_load(p.read_text())\n"
            "def b(config_path):\n    return config_path.exists()\n"
            "def c(other):\n    return other.read_text()\n"
            "def d(cfg_path):\n    return cfg_path.stat()  # config-reader: ok — test\n"
            "def e():\n    from utils import atomic_roundtrip_yaml_update\n"
            "    atomic_roundtrip_yaml_update(get_config_path(), 'a', 1)\n",
            encoding="utf-8")
        with patch.object(guard, "ROOT", tmp_path):
            problems = guard.scan_file(bad)
        assert sorted(int(p.split(":")[1]) for p in problems) == [4, 6, 13], problems

    def test_backend_calls_are_clean(self, tmp_path):
        guard = self._guard()
        good = tmp_path / "hermes_cli" / "good_reader.py"
        good.parent.mkdir()
        good.write_text(
            "from hermes_cli.config_backend import config_exists, read_config_doc\n"
            "def a(home):\n    p = home / 'config.yaml'\n"
            "    return read_config_doc(p) if config_exists(p) else {}\n",
            encoding="utf-8")
        with patch.object(guard, "ROOT", tmp_path):
            assert guard.scan_file(good) == []
