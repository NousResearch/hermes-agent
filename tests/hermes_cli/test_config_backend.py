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

    def test_unknown_backend_fails_closed(self, home, monkeypatch):
        # No config value can select the backend, and an unknown one never falls back to the file.
        monkeypatch.setenv(cb.BACKEND_ENV, "bogus")
        with pytest.raises(cb.ConfigBackendUnavailable):
            cb.get_config_backend()
        with pytest.raises(cb.ConfigBackendUnavailable):
            cb.read_config_doc(home / "config.yaml")

    def test_unconfigured_remote_backend_fails_closed_on_read(self, home, monkeypatch):
        # Selected from the env alone; without its plane identity the first read exits, and the
        # local config.yaml is never used as a fallback (D2).
        from plugins.config_backends import remote
        (home / "config.yaml").write_text("a: 1\n", encoding="utf-8")
        monkeypatch.setenv(cb.BACKEND_ENV, "remote")
        monkeypatch.delenv("HERMES_CONFIG_INSTANCE_ID", raising=False)
        remote._reset_for_tests()
        try:
            assert cb.get_config_backend().name == "remote"
            with pytest.raises(cb.ConfigBackendUnavailable, match="Remote Config"):
                cb.read_config_doc(home / "config.yaml")
        finally:
            remote._reset_for_tests()

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


class _RecordingBackend(cb.FileBackend):
    """The file backend, recording which operations each pipeline asks of it."""

    name = "recording"

    def __init__(self):
        self.calls = []

    def _rec(self, op, home):
        self.calls.append((op, Path(home)))

    def read_user_layer(self, home):
        self._rec("read", home)
        return super().read_user_layer(home)

    def read_user_doc_readonly(self, home):
        self._rec("read", home)
        return super().read_user_doc_readonly(home)

    def version(self, home):
        self._rec("version", home)
        return super().version(home)

    def exists(self, home):
        self._rec("exists", home)
        return super().exists(home)

    def write_changes(self, home, changes):
        self._rec("write", home)
        return super().write_changes(home, changes)

    def ops(self, home):
        return {op for op, h in self.calls if h == Path(home)}


@pytest.fixture
def recording(monkeypatch, tmp_path):
    """A fresh HERMES_HOME with a config.yaml and every config read/write recorded."""
    from hermes_cli import config as config_mod
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "# hand-tuned\nmodel:\n  default: my-model\nsecrets:\n  sources: []\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv(cb.BACKEND_ENV, raising=False)
    for cache in (config_mod._LOAD_CONFIG_CACHE, config_mod._RAW_CONFIG_CACHE):
        cache.clear()
    backend = _RecordingBackend()
    monkeypatch.setattr(cb, "get_config_backend", lambda: backend)
    return home, backend


class TestPipelinesConsultTheBackend:
    """Every read pipeline and writer reaches the selected backend, keyed by the hermes home."""

    def test_load_config(self, recording):
        from hermes_cli.config import load_config
        home, backend = recording
        assert load_config()["model"]["default"] == "my-model"
        assert {"version", "read"} <= backend.ops(home)

    def test_raw_primitives(self, recording):
        from hermes_cli.config import read_raw_config, read_user_config_raw
        home, backend = recording
        assert read_raw_config()["model"]["default"] == "my-model"
        assert read_user_config_raw()["model"]["default"] == "my-model"
        assert {"version", "read"} <= backend.ops(home)

    def test_effective_loader(self, recording):
        from hermes_cli import config_effective
        home, backend = recording
        config_effective._EFFECTIVE_CACHE.clear()
        assert config_effective.load_user_config_effective()["model"]["default"] == "my-model"
        assert {"version", "read"} <= backend.ops(home)

    def test_cli_loader(self, recording):
        from hermes_cli.cli_config_load import load_cli_config
        home, backend = recording
        load_cli_config()
        assert {"exists", "read"} <= backend.ops(home)

    def test_gateway_yaml_layers(self, recording):
        from gateway.config_loader import read_yaml_layers
        home, backend = recording
        assert read_yaml_layers(home)["model"]["default"] == "my-model"
        assert "read" in backend.ops(home)

    def test_dotenv_secrets_read(self, recording):
        from hermes_cli.env_loader import _load_secrets_config
        home, backend = recording
        _load_secrets_config(home)
        assert "read" in backend.ops(home)

    def test_writers(self, recording):
        from hermes_cli.config import load_config, save_config
        from hermes_cli.personality import persist_personality
        home, backend = recording
        cfg = load_config()
        cfg["model"]["default"] = "other-model"
        save_config(cfg)
        assert persist_personality("kawaii")
        assert [op for op, h in backend.calls if op == "write" and h == home] == ["write", "write"]
        text = (home / "config.yaml").read_text(encoding="utf-8")
        assert "# hand-tuned" in text and "other-model" in text and "kawaii" in text


class TestFileTooling:

    def test_file_backend_allows_it(self, home):
        cb.require_file_tooling("x")

    def test_backend_without_a_file_refuses_it(self, home, monkeypatch):
        class NoFile(cb.FileBackend):
            name = "nofile"

            def supports_file_tooling(self):
                return False
        monkeypatch.setattr(cb, "get_config_backend", NoFile)
        from hermes_cli.config_backups import backup_config
        (home / "config.yaml").write_text("a: 1\n", encoding="utf-8")
        assert backup_config(home / "config.yaml", "test") is None
        with pytest.raises(cb.ConfigBackendUnavailable, match="Profile clone"):
            from hermes_cli.profiles import _resolve_clone_source
            _resolve_clone_source(None)


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

    @pytest.mark.parametrize("call", [
        "yaml.full_load(config_path)", "yaml.unsafe_load(config_path)", "yaml.compose(config_path)",
        "yaml.load_all(config_path)", "os.path.getsize(config_path)",
        "config_path.stat()  # config-reader: ok",  # an escape without a reason does not suppress
    ])
    def test_every_read_form_is_flagged(self, tmp_path, call):
        guard = self._guard()
        bad = tmp_path / "hermes_cli" / "bad_reader.py"
        bad.parent.mkdir()
        bad.write_text(f"import os, yaml\ndef a(config_path):\n    return {call}\n", encoding="utf-8")
        with patch.object(guard, "ROOT", tmp_path):
            assert len(guard.scan_file(bad)) == 1

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
