"""Regression: config cache must detect file *replacements*.

A replacement that preserves ``st_mtime_ns`` and ``st_size`` (``cp -p``,
``rsync -t``, timestamp-pinning scripts, sync clients) was treated as
unchanged, so long-lived processes kept serving stale config until restart.
The signature now also covers ``st_ino`` (fresh inode on atomic replace) and
``st_ctime_ns`` (cannot be backdated via ``os.utime``).

See #111105.
"""
import os
import shutil
from unittest.mock import patch

from hermes_cli import config as config_mod
from hermes_cli.config import load_config

A = "model:\n  provider: opencode-go\n  default: aaaa-route\n"
B = "model:\n  provider: opencode-go\n  default: bbbb-route\n"


def test_load_config_detects_replacement_with_pinned_mtime(tmp_path):
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        config_mod._LOAD_CONFIG_CACHE.clear()
        config_mod._RAW_CONFIG_CACHE.clear()
        cfg = tmp_path / "config.yaml"
        cfg.write_text(A, encoding="utf-8")
        assert load_config()["model"]["default"] == "aaaa-route"
        # Replace content, keep st_mtime_ns + st_size identical.
        before = cfg.stat()
        other = tmp_path / "other.yaml"
        other.write_text(B, encoding="utf-8")
        shutil.copy2(other, cfg)
        os.utime(cfg, ns=(before.st_atime_ns, before.st_mtime_ns))
        after = cfg.stat()
        assert after.st_mtime_ns == before.st_mtime_ns
        assert after.st_size == before.st_size
        assert cfg.read_text(encoding="utf-8").split("default: ")[1].strip() == "bbbb-route"
        assert load_config()["model"]["default"] == "bbbb-route"
