"""Accessibility forcing must be opt-in for both desktop launch paths (#133652)."""

import pytest

from hermes_cli.config import load_config
from hermes_cli.main_desktop import _desktop_launch_options


@pytest.mark.parametrize("desktop_yaml,expected", [("{}", False), ("{renderer_accessibility: true}", True)])
def test_desktop_accessibility_requires_explicit_opt_in(tmp_path, monkeypatch, desktop_yaml, expected):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(f"desktop: {desktop_yaml}\n", encoding="utf-8")

    assert load_config()["desktop"]["renderer_accessibility"] is expected
    assert _desktop_launch_options()[-1] is expected
