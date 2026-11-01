"""Missing SDK guidance follows the running PM generation (regression for #140620)."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

from pm import environments, paths


@pytest.mark.parametrize("surface", ["connect", "standalone"])
@pytest.mark.parametrize("state", ["installed_later", "still_missing", "current", "foreign"])
def test_missing_feishu_sdk_reports_when_a_restart_can_load_it(
    tmp_path, monkeypatch, caplog, surface, state,
):
    from gateway.config import PlatformConfig
    from plugins.platforms.feishu import adapter

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    root = tmp_path / "repo"
    monkeypatch.setattr(paths, "repo_root", lambda: root)
    install = environments.install_state_dir(root)
    first = install / "environments" / "first" / "venv"
    second = install / "environments" / "second" / "venv"
    for environment in (first, second):
        environments.site_packages(environment).mkdir(parents=True)
        (environment / "pyvenv.cfg").write_text("home = fixture\n", encoding="utf-8")
    if state != "still_missing":
        (environments.site_packages(second) / "lark_oapi.py").write_text(
            "raise AssertionError('diagnostics must never import a new generation')\n",
            encoding="utf-8",
        )
    if state != "foreign":
        monkeypatch.syspath_prepend(str(environments.site_packages(first)))
    selected = first if state == "current" else second
    environments.runtime_facts_path(root).write_text(json.dumps({
        "packages": {"venv": {"environment": str(selected)}},
    }), encoding="utf-8")
    # The failure seam is the SDK import; selection and restart detection stay real.
    monkeypatch.setattr(adapter, "_load_lark_oapi", lambda: False)
    config = PlatformConfig(extra={"app_id": "fixture", "app_secret": "fixture"})
    original_path = list(sys.path)

    if surface == "connect":
        assert asyncio.run(adapter.FeishuAdapter(config).connect()) is False
        message = caplog.text
    else:
        result = asyncio.run(adapter._standalone_send(config, "fixture", "message"))
        message = result["error"]

    assert ("restart" in message.lower()) is (state == "installed_later"), message
    if state == "installed_later":
        assert "hermes setup" not in message
    assert sys.path == original_path
