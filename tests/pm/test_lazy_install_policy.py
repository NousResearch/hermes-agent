"""PM reads boolean install policy from real config, without CLI output."""

import pytest

import pm
from hermes_cli.config import get_config_path


@pytest.mark.parametrize("content,expected", [
    ("{}\n", True),
    ("security:\n  allow_lazy_installs: true\n", True),
    ("security:\n  allow_lazy_installs: false\n", False),
    ("security:\n  allow_lazy_installs: 'false'\n", False),
    ("security: [\n", False),
])
def test_real_policy_is_boolean_and_fails_closed(monkeypatch, capsys, content, expected):
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    config = get_config_path()
    config.write_text(content, encoding="utf-8")

    assert pm.lazy_installs_allowed() is expected
    assert capsys.readouterr().out == ""
    assert config.read_text(encoding="utf-8") == content


def test_internal_disable_overrides_enabled_config(monkeypatch):
    get_config_path().write_text("security:\n  allow_lazy_installs: true\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")

    assert not pm.lazy_installs_allowed()
