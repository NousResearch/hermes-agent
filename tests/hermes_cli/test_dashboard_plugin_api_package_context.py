"""Dashboard plugin API files must load with their plugin's package context (#134408).

Both dashboard loaders (``plugin_host_child.py::op_asgi`` and
``web_server_dashboard.py::_mount_plugin_api_routes``) create the plugin API module from a
bare file location, so ``__package__`` is empty and a relative import inside a plugin's
``dashboard/plugin_api.py`` dies with ``ImportError: attempted relative import with no known
parent package`` — every ``/api/plugins/<name>/`` route 500s, while the same plugin's
agent-side tools (imported as ``hermes_plugins.<slug>``) work.

These tests pin that both load paths hand the registered plugin package to the API module:
the package is registered exactly as the agent-side loader leaves it (``hermes_plugins.<slug>``
with the plugin dir on ``__path__``).
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys

from hermes_cli.plugin_host_child import _import_plugin
from hermes_cli.plugin_host_wire import decode

# The reported shape: a dashboard API that reaches back into its own plugin package.
API_BODY = (
    "from fastapi import APIRouter\n"
    "from ..team_manager import list_teams\n"
    "\n"
    "router = APIRouter()\n"
    "\n"
    "@router.get('/teams')\n"
    "def teams():\n"
    "    return list_teams()\n"
)


def _make_plugin(root, slug):
    """A directory plugin whose dashboard API uses a relative import (the #134408 shape)."""
    plugin_dir = root / "plugins" / slug
    (plugin_dir / "dashboard").mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text("", encoding="utf-8")
    (plugin_dir / "team_manager.py").write_text(
        "def list_teams():\n    return {'teams': ['alpha']}\n", encoding="utf-8")
    (plugin_dir / "dashboard" / "plugin_api.py").write_text(API_BODY, encoding="utf-8")
    (plugin_dir / "dashboard" / "manifest.json").write_text(
        json.dumps({"name": slug, "label": slug, "api": "plugin_api.py"}), encoding="utf-8")
    return plugin_dir


def _register_package(plugin_dir, slug):
    """Register the plugin package the way agent-side discovery leaves it before the dashboard
    loads: ``hermes_plugins.<slug>`` with the plugin dir on ``__path__``."""
    _import_plugin({"path": str(plugin_dir), "module_name": f"hermes_plugins.{slug}", "name": slug})


def _cleanup(slug):
    sys.modules.pop(f"hermes_dashboard_plugin_{slug}", None)
    sys.modules.pop(f"hermes_plugins.{slug}", None)


def test_hosted_asgi_serves_an_api_with_relative_imports(tmp_path):
    """``plugins.isolation: host`` — op_asgi loads the API file inside the plugin host, where
    the plugin package is already registered (op_load ran first)."""
    from hermes_cli.plugin_host_child import HostRuntime

    plugin_dir = _make_plugin(tmp_path, "ctxplug")
    _register_package(plugin_dir, "ctxplug")
    # The channel streams are unused: op_asgi loads the module and runs the ASGI call in-process.
    runtime = HostRuntime(io.BytesIO(), io.BytesIO())
    try:
        result = runtime.op_asgi({
            "plugin": "ctxplug",
            "dashboard_dir": str(plugin_dir / "dashboard"),
            "api_file": "plugin_api.py",
            "method": "GET", "path": "/teams", "query": "", "headers": [], "body": None,
        })
        assert result["status"] == 200
        assert json.loads(decode(result["body"])) == {"teams": ["alpha"]}
    finally:
        _cleanup("ctxplug")


def test_inprocess_mount_serves_an_api_with_relative_imports(tmp_path):
    """Default ``plugins.isolation: in_process`` — _mount_plugin_api_routes imports the API
    file in the dashboard process, where agent-side discovery registered the package."""
    from hermes_constants import get_hermes_home
    from hermes_cli import web_server, web_server_dashboard

    home = get_hermes_home()
    plugin_dir = _make_plugin(home, "ctxmount")
    (home / "config.yaml").write_text("plugins:\n  enabled:\n    - ctxmount\n", encoding="utf-8")
    _register_package(plugin_dir, "ctxmount")

    app = web_server.app
    original_routes = list(app.router.routes)
    web_server._dashboard_plugins_cache = None
    try:
        web_server_dashboard._mount_plugin_api_routes()
        module = sys.modules.get("hermes_dashboard_plugin_ctxmount")
        assert module is not None, "the dashboard API module must stay loaded"
        assert module.router is not None
        routes = [getattr(r, "path", "") for r in app.router.routes]
        assert "/api/plugins/ctxmount/teams" in routes
    finally:
        app.router.routes[:] = original_routes
        web_server._dashboard_plugins_cache = None
        _cleanup("ctxmount")


def test_module_package_context_uses_the_registered_package_dir(tmp_path):
    """The context comes from the registered package's ``__path__``, not a name assembled from
    the plugin id (``_directory_module_name`` may append a ``__home_<digest>`` suffix)."""
    from hermes_cli.plugins_loader import give_module_package_context

    plugin_dir = _make_plugin(tmp_path, "ctxunit")
    _register_package(plugin_dir, "ctxunit")
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_ctxunit", plugin_dir / "dashboard" / "plugin_api.py")
    module = importlib.util.module_from_spec(spec)
    try:
        give_module_package_context(module, spec, plugin_dir)
        assert module.__package__ == "hermes_plugins.ctxunit.dashboard"
        # ``__name__`` stays as the caller registered it; only the spec is aligned, which keeps
        # 3.14+ from warning that ``__package__ != __spec__.parent`` on every relative import.
        assert module.__name__ == "hermes_dashboard_plugin_ctxunit"
        assert spec.parent == "hermes_plugins.ctxunit.dashboard"
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)  # the relative import resolves
        assert module.router is not None
    finally:
        sys.modules.pop(spec.name, None)
        _cleanup("ctxunit")


def test_module_package_context_is_a_noop_without_a_registered_package(tmp_path):
    """No registered package is not an error: the module keeps its bare loader context (and its
    caller's error, if any) instead of importing under an invented package name."""
    from hermes_cli.plugins_loader import give_module_package_context

    plugin_dir = _make_plugin(tmp_path, "ctxabsent")
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_ctxabsent", plugin_dir / "dashboard" / "plugin_api.py")
    module = importlib.util.module_from_spec(spec)
    before = module.__package__
    give_module_package_context(module, spec, plugin_dir)
    assert module.__package__ == before
