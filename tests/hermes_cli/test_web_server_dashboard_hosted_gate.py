"""A hosted process never imports a home plugin's dashboard api file.

``_mount_one_plugin_api`` is the seam the web server's import sweep and the hot-mount
fallback share; the one hosted provenance rule
(:func:`plugins.plugin_loader.hosted_inprocess_refusal`) answers it before the api
module can run — a hosted deployment's dashboard serves only the plugins its image
bundles, and a local install mounts exactly as before.
"""

from __future__ import annotations

import sys

import pytest

fastapi = pytest.importorskip("fastapi")


def test_a_home_plugin_api_is_refused_in_a_hosted_process(tmp_path, monkeypatch):
    from hermes_cli import hosted_mode
    from hermes_cli import web_server_dashboard as dashboard

    monkeypatch.setattr(hosted_mode, "is_hosted", lambda: True)
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
