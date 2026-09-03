"""Regression test for config parser crash on null ``extra`` field.

YAML ``extra:`` with no value parses to ``None`` in Python. The dict-unpacking
``**plat_block.get('extra', {})`` returns ``None`` (key exists, default unused)
and raises ``TypeError: argument of type 'NoneType' is not iterable``. The
exception is caught by the outer try/except, silently dropping the platform
registration — api_server never starts after unclean shutdown.

This suite locks the fix: ``or {}`` coerces None before unpacking.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from gateway.config import Platform, load_gateway_config


def _write_config(tmp_path: Path, yaml_body: str) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(textwrap.dedent(yaml_body), encoding="utf-8")
    return cfg


class TestNullExtraDoesNotCrash:
    """``extra:`` (null) must not abort platform registration."""

    def test_api_server_extra_null(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _write_config(
            tmp_path,
            """\
            platforms:
              api_server:
                enabled: true
                extra:
                key: test-key-1234
                host: 0.0.0.0
                port: 8642
            """,
        )

        # Must not raise TypeError.
        cfg = load_gateway_config()

        # api_server platform must be registered with the bridged key.
        ps = cfg.platforms
        assert Platform.API_SERVER in ps, "api_server platform silently dropped"
        assert ps[Platform.API_SERVER].enabled is True

    def test_webhook_extra_null(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _write_config(
            tmp_path,
            """\
            platforms:
              webhook:
                enabled: true
                extra:
                port: 8649
            """,
        )

        cfg = load_gateway_config()
        assert Platform.WEBHOOK in cfg.platforms

    def test_extra_dict_still_merges(self, tmp_path, monkeypatch):
        """When extra is a real dict, merge must still work."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _write_config(
            tmp_path,
            """\
            platforms:
              api_server:
                enabled: true
                extra:
                  key: my-key
                  custom_field: custom_value
            """,
        )

        cfg = load_gateway_config()
        ps = cfg.platforms
        assert Platform.API_SERVER in ps
        extra = ps[Platform.API_SERVER].extra
        assert extra is not None
        assert extra.get("key") == "my-key"
        assert extra.get("custom_field") == "custom_value"
