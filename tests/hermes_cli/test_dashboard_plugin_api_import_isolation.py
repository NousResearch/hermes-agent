"""How ``_mount_one_plugin_api`` imports a dashboard plugin's backend ``api`` file.

Under ``plugins.isolation: host`` a user plugin's ``api`` file must never execute in the dashboard
process (its routes are served from the plugin host instead). Bundled APIs and in_process user APIs
execute in-process, registered in ``sys.modules`` while the body runs and removed if it raises.
"""

import sys
from unittest.mock import patch

import hermes_agent.hermes_yaml as yaml
import pytest
from fastapi import FastAPI

from hermes_agent.hermes_cli import web_server_dashboard as dashboard
from hermes_agent.hermes_constants import get_hermes_home

NAME = "isoprobe"
MODULE = f"hermes_dashboard_plugin_{NAME}"
# The body records whether it ran registered in sys.modules; a 'raise' file next to it makes it fail.
API = (
    "import sys\n"
    "from pathlib import Path\n"
    "from fastapi import APIRouter\n"
    "here = Path(__file__)\n"
    "here.with_name('imported').write_text(str(sys.modules.get(__name__) is not None))\n"
    "if here.with_name('raise').exists():\n"
    "    raise RuntimeError('api body failed')\n"
    "router = APIRouter()\n"
    "router.add_api_route('/ping', lambda: 'pong')\n"
)


def _plugin(tmp_path, isolation, source):
    (get_hermes_home() / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"isolation": isolation, "enabled": [NAME]}}), encoding="utf-8")
    dash = tmp_path / NAME / "dashboard"
    dash.mkdir(parents=True)
    (dash / "plugin_api.py").write_text(API, encoding="utf-8")
    return dash, {"name": NAME, "source": source, "_dir": str(dash), "_api_file": "plugin_api.py"}


def _mount(plugin):
    app = FastAPI()
    with patch.object(dashboard, "_mount_hosted_plugin_api") as hosted:
        mounted = dashboard._mount_one_plugin_api(app, plugin, {NAME}, set())
    return app, mounted, hosted


def test_host_isolation_keeps_user_dashboard_api_out_of_the_process(tmp_path):
    dash, plugin = _plugin(tmp_path, "host", "user")
    sys.modules.pop(MODULE, None)

    _, mounted, hosted = _mount(plugin)

    assert mounted is True
    hosted.assert_called_once()  # served from the plugin host
    assert not (dash / "imported").exists()
    assert MODULE not in sys.modules


@pytest.mark.parametrize(("isolation", "source", "raises"),
                         [("in_process", "user", False), ("host", "bundled", True)])
def test_in_process_dashboard_api_executes_registered_in_sys_modules(tmp_path, isolation, source, raises):
    dash, plugin = _plugin(tmp_path, isolation, source)
    if raises:
        (dash / "raise").write_text("", encoding="utf-8")
    try:
        app, mounted, hosted = _mount(plugin)

        hosted.assert_not_called()
        # Executed in this process, registered under its module name while the body ran.
        assert (dash / "imported").read_text() == "True"
        if raises:
            assert mounted is False
            assert MODULE not in sys.modules
            assert not dashboard._plugin_api_mounted(app, NAME)
        else:
            assert mounted is True
            assert sys.modules[MODULE].__file__ == str(dash / "plugin_api.py")
            assert dashboard._plugin_api_mounted(app, NAME)
    finally:
        sys.modules.pop(MODULE, None)


def test_non_python_api_file_is_skipped_without_a_warning(tmp_path, caplog):
    dash, plugin = _plugin(tmp_path, "in_process", "user")
    (dash / "plugin_api.py").rename(dash / "plugin_api.txt")
    plugin["_api_file"] = "plugin_api.txt"

    with caplog.at_level("DEBUG", logger="hermes_agent.hermes_cli"):
        app, mounted, hosted = _mount(plugin)

    assert mounted is False
    hosted.assert_not_called()
    assert not (dash / "imported").exists()
    assert not dashboard._plugin_api_mounted(app, NAME)
    assert [r.getMessage() for r in caplog.records if r.levelname == "WARNING"] == []
