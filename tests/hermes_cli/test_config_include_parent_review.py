"""Independent regression cases for config includes."""
import os
from unittest.mock import patch


def test_atomic_replacement_same_size_and_mtime_invalidates_include(tmp_path):
    import hermes_cli.config as c
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        inc = tmp_path / "included.yaml"
        inc.write_text("model:\n  default: old/model\n")
        (tmp_path / "config.yaml").write_text("include: included.yaml\n")
        c._LOAD_CONFIG_CACHE.clear()
        c._RAW_CONFIG_CACHE.clear()
        assert c.load_config()["model"]["default"] == "old/model"
        stat = inc.stat()
        replacement = tmp_path / "replacement.yaml"
        replacement.write_text("model:\n  default: new/model\n")
        os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        replacement.replace(inc)
        assert inc.stat().st_size == stat.st_size
        assert inc.stat().st_mtime_ns == stat.st_mtime_ns
        assert c.load_config()["model"]["default"] == "new/model"


def test_nested_relative_include_cycle_and_cache_hit(tmp_path, caplog):
    import hermes_cli.config as c
    from hermes_cli import config_includes
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "config.yaml").write_text("include: sub/child.yaml\nmodel:\n  max_tokens: 300\n")
        (sub / "child.yaml").write_text("include: ../leaf.yaml\nmodel:\n  default: child/model\n")
        (tmp_path / "leaf.yaml").write_text("include: sub/child.yaml\nmodel:\n  default: leaf/model\n  max_tokens: 100\n")
        c._LOAD_CONFIG_CACHE.clear()
        c._RAW_CONFIG_CACHE.clear()
        config_includes.INCLUDE_SIGNATURES.clear()
        loaded = c.load_config()
        assert loaded["model"]["default"] == "child/model"
        assert loaded["model"]["max_tokens"] == 300
        assert "Skipping recursive config include" in caplog.text
        with patch.object(config_includes, "read_config_yaml", side_effect=AssertionError("unexpected reparse")):
            assert c.load_config() == loaded
