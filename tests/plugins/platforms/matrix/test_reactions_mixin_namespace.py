"""The reactions mixin stays in the adapter's namespace and logs through the adapter's logger.

The plugin loader imports a bundled platform package as ``hermes_plugins.<slug>``
(``PluginManager._load_directory_module``), where the adapter's own
``logging.getLogger(__name__)`` is ``hermes_plugins.<slug>.adapter``. The methods
lifted into ``MatrixReactionsMixin`` keep logging through that same logger only if
the adapter imports the mixin relatively and the mixin derives its logger from
``__package__``; an absolute import or a hard-coded name sends their records to
``plugins.platforms.matrix.adapter`` instead.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

MATRIX_DIR = Path(__file__).resolve().parents[4] / "plugins" / "platforms" / "matrix"


@pytest.fixture
def namespaced_matrix():
    """Import the matrix package as ``hermes_plugins.<slug>``, the way the plugin loader does."""
    created_parent = "hermes_plugins" not in sys.modules
    if created_parent:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    name = "hermes_plugins.platforms__matrix_reactions_nstest"
    spec = importlib.util.spec_from_file_location(
        name, MATRIX_DIR / "__init__.py", submodule_search_locations=[str(MATRIX_DIR)])
    package = importlib.util.module_from_spec(spec)
    package.__package__ = name
    package.__path__ = [str(MATRIX_DIR)]
    sys.modules[name] = package
    try:
        spec.loader.exec_module(package)
        yield name
    finally:
        for loaded in [m for m in sys.modules if m == name or m.startswith(f"{name}.")]:
            del sys.modules[loaded]
        if created_parent:
            sys.modules.pop("hermes_plugins", None)


def test_namespaced_adapter_shares_its_logger_with_the_mixin(namespaced_matrix):
    adapter = sys.modules[f"{namespaced_matrix}.adapter"]
    mixin_module = sys.modules[adapter.MatrixReactionsMixin.__module__]
    assert mixin_module.__name__ == f"{namespaced_matrix}.reactions_mixin"
    assert adapter.logger.name == f"{namespaced_matrix}.adapter"
    assert mixin_module.logger is adapter.logger


def test_canonical_adapter_shares_its_logger_with_the_mixin():
    from plugins.platforms.matrix import adapter, reactions_mixin

    assert adapter.MatrixReactionsMixin is reactions_mixin.MatrixReactionsMixin
    assert reactions_mixin.logger is adapter.logger
