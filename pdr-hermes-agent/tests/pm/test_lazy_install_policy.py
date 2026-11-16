"""PM reads boolean install policy from real config, without CLI output."""

import sys

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


class _Missing:
    """Make importing ``target`` fail as if module ``name`` were not installed."""

    def __init__(self, target: str, name: str):
        self.target, self.name = target, name

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.target:
            raise ModuleNotFoundError(f"No module named {self.name!r}", name=self.name)


@pytest.mark.parametrize("missing,allowed", [("hermes_cli", True), ("hermes_cli.config", True), ("yaml", False)])
def test_policy_without_the_config_package_allows_only_when_it_is_absent(missing, allowed, monkeypatch):
    """A bootstrap without hermes_cli may install; a hermes_cli.config that fails on a dependency may not."""
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    monkeypatch.delitem(sys.modules, "hermes_cli.config", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_Missing("hermes_cli.config", missing), *sys.meta_path])
    assert pm.install.lazy_installs_allowed() is allowed
