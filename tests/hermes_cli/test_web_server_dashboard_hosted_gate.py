"""A hosted process never imports a home plugin's dashboard api file.

``_mount_one_plugin_api`` is the seam the web server's import sweep and the hot-mount
fallback share; the process-wide code fence the hosted boot installs refuses the api
module's import when its file resolves outside the deployment image — a hosted
deployment's dashboard serves only the plugins its image bundles, and a local install
mounts exactly as before.
"""

from __future__ import annotations

import sys

import pytest

fastapi = pytest.importorskip("fastapi")


@pytest.fixture
def hosted(monkeypatch):
    """A frozen hosted verdict with the process-wide code fence installed."""
    from hermes_cli import hosted_mode
    from hermes_cli.hosted_code_fence import install_hosted_code_fence

    monkeypatch.setitem(hosted_mode._VERDICT, "hosted", True)
    install_hosted_code_fence()


def test_a_home_plugin_api_is_refused_in_a_hosted_process(tmp_path, monkeypatch, hosted):
    from hermes_cli import web_server_dashboard as dashboard
    marker = tmp_path / "api-imported"
    plugin_dir = tmp_path / "probe-plugin"
    plugin_dir.mkdir()
    (plugin_dir / "plugin_api.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('x')\n"
        "from fastapi import APIRouter\n"
        "router = APIRouter()\n",
        encoding="utf-8",
    )
    app = fastapi.FastAPI()
    plugin = {"name": "probe-plugin", "source": "user", "_dir": str(plugin_dir),
              "_api_file": "plugin_api.py"}

    assert dashboard._mount_one_plugin_api(app, plugin, {"probe-plugin"}, set()) is False
    assert not marker.exists()  # the api module never ran
    assert not [m for m in sys.modules if m.startswith("hermes_dashboard_plugin_")]
