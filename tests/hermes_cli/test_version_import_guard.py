"""Old-updater version lookup distinguishes absent PM from a broken PM import."""

import sys

import pytest

from hermes_agent import hermes_cli


class _MissingPaths:
    def __init__(self, error):
        self.error = error

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pm.paths":
            raise self.error


def _fail_paths_import(monkeypatch, error):
    monkeypatch.delitem(sys.modules, "pm.paths", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_MissingPaths(error), *sys.meta_path])


@pytest.mark.parametrize("missing", ["pm", "pm.paths"])
def test_version_uses_checkout_stamp_when_pm_is_absent(monkeypatch, tmp_path, missing):
    package = tmp_path / "hermes_agent" / "hermes_cli"
    package.mkdir(parents=True)
    monkeypatch.setattr(hermes_cli, "__file__", str(package / "__init__.py"))
    (tmp_path / "install-stamp.json").write_text('{"baseVersion": "1.2.3"}', encoding="utf-8")
    _fail_paths_import(monkeypatch, ModuleNotFoundError(name=missing))

    assert hermes_cli.__getattr__("__version__") == "1.2.3"


@pytest.mark.parametrize("missing", ["pm.other", "pm.paths.dependency", "pm.paths_extra", "yaml", None])
def test_version_propagates_pm_import_faults(monkeypatch, missing):
    error = ModuleNotFoundError(name=missing)
    _fail_paths_import(monkeypatch, error)

    with pytest.raises(ModuleNotFoundError) as raised:
        hermes_cli.__getattr__("__version__")
    assert raised.value is error
