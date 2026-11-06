"""PluginContext's ``register_<category>_provider`` methods are written out with ``def`` (greppable,
visible to type checkers) while their data and docstrings stay in the registrar table."""

import ast
import inspect
import sys

from hermes_cli.plugins import PluginContext, PluginManifest
from hermes_cli.plugins_registrars import _SCOPED_PROVIDER_REGISTRARS


def _class_defs() -> dict[str, ast.FunctionDef]:
    tree = ast.parse(inspect.getsource(sys.modules[PluginContext.__module__]))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PluginContext")
    return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}


def test_every_scoped_provider_registrar_is_an_explicit_def():
    defs = _class_defs()
    missing = [row[0] for row in _SCOPED_PROVIDER_REGISTRARS if row[0] not in defs]
    assert missing == []


def test_explicit_registrars_keep_the_row_docstring_and_parameter():
    for name, *_rest, doc, options in _SCOPED_PROVIDER_REGISTRARS:
        method = getattr(PluginContext, name)
        param = options.get("param", "provider")
        assert method.__doc__ == doc, name
        assert method.__qualname__ == f"PluginContext.{name}"
        assert str(inspect.signature(method)) == f"(self, {param})", name


def test_explicit_registrars_still_ignore_calls_after_an_abandoned_load(caplog):
    ctx = PluginContext.__new__(PluginContext)
    ctx._load_abandoned = True
    ctx.manifest = PluginManifest(name="late")
    for name, *_ in _SCOPED_PROVIDER_REGISTRARS:
        assert getattr(ctx, name)(object()) is None
    assert sum("after its load timed out" in r.getMessage() for r in caplog.records) == len(
        _SCOPED_PROVIDER_REGISTRARS
    )
