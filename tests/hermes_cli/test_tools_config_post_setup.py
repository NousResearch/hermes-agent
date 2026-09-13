"""Provider setup delegates package policy to PM and external server owners."""

import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import ModuleType
from unittest.mock import patch

import pm
import pytest

from hermes_cli.tools_config_post_setup import _run_post_setup


def test_cloud_browser_explicit_setup_ensures_managed_driver():
    with (
        patch("tools.browser_use_cli.install_cli", return_value=(True, "ready")),
        patch("tools.browser_tool_install._find_agent_browser", return_value="/external/agent-browser"),
        patch("pm.ensure") as ensure,
        patch("subprocess.run") as run,
    ):
        _run_post_setup("browserbase")

    ensure.assert_called_once_with("agent-browser", explicit=True)
    run.assert_not_called()


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("playwright_present", [True, False])
def test_local_setup_uses_pm_dependency_union_and_gives_system_library_guidance(
    capsys, playwright_present,
):
    with (
        patch("tools.browser_use_cli.install_cli", return_value=(True, "ready")),
        patch("tools.browser_tool_install._running_in_docker", return_value=False),
        patch("tools.browser_tool_install._chromium_installed", return_value=False),
        patch("importlib.util.find_spec", return_value=object() if playwright_present else None),
        patch("shutil.which", return_value="/usr/bin/apt-get"),
        patch("pm.ensure") as ensure,
        patch("subprocess.run") as run,
    ):
        before = dict(os.environ)
        _run_post_setup("agent_browser")
        assert dict(os.environ) == before

    ensure.assert_called_once_with("agent-browser", explicit=True)
    run.assert_not_called()
    output = capsys.readouterr().out
    assert "system" in output
    if playwright_present:
        assert sys.executable in output
        assert "-m playwright install-deps chromium" in output
    else:
        assert "https://playwright.dev/python/docs/browsers#install-system-dependencies" in output
    assert "npx" not in output


@pytest.mark.parametrize("status", [200, 503])
def test_camofox_setup_leaves_external_server_and_config_owned_by_user(monkeypatch, capsys, status):
    from hermes_cli.config import get_hermes_home
    from hermes_cli.tools_config_post_setup import _POST_SETUP_READY

    class Health(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == "/health"
            self.send_response(status)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, format, *args):
            pass

    config_path = get_hermes_home() / "config.yaml"
    config_path.write_text("browser:\n  cloud_provider: camofox\n", encoding="utf-8")
    before = config_path.read_bytes()
    with ThreadingHTTPServer(("127.0.0.1", 0), Health) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        monkeypatch.setenv("CAMOFOX_URL", f"http://127.0.0.1:{server.server_port}")
        try:
            with (
                patch("hermes_constants.find_node_executable", return_value="/external/npm"),
                patch("subprocess.run") as run,
                patch("pm.ensure") as ensure,
            ):
                _run_post_setup("camofox")
                run.assert_not_called()
                ensure.assert_not_called()
                assert _POST_SETUP_READY["camofox"]() is (status == 200)
        finally:
            server.shutdown()
            thread.join(timeout=5)

    assert config_path.read_bytes() == before
    output = capsys.readouterr().out
    assert "externally managed" in output
    assert ("reachable" if status == 200 else "not reachable") in output


@pytest.mark.parametrize("key,extra", [
    ("ddgs", "ddgs"), ("faster_whisper", "stt-whisper"),
    ("kittentts", "kittentts"), ("piper", "piper"),
])
@pytest.mark.parametrize("refused", [False, True])
def test_importable_sdk_does_not_bypass_pm_constraints(monkeypatch, capsys, key, extra, refused):
    sdk = ModuleType(key)
    sdk.__version__ = "0.0.0"
    monkeypatch.setitem(sys.modules, key, sdk)
    error = pm.InstallError("venv", "outside frozen feature set") if refused else None
    with patch("pm.sync_venv", side_effect=error) as sync:
        _run_post_setup(key)

    sync.assert_called_once_with([extra], explicit=True)
    output = capsys.readouterr().out
    if refused:
        assert "outside frozen feature set" in output
        assert "Retry with: hermes tools" in output
        assert "Restart Hermes" not in output
    else:
        assert "Restart Hermes" in output


@pytest.mark.parametrize("failure", [None, "sdk", "admission"])
def test_langfuse_setup_uses_plugin_admission_and_preserves_config_on_refusal(
    monkeypatch, tmp_path, capsys, failure,
):
    from hermes_cli.config import get_hermes_home, read_raw_config

    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setitem(sys.modules, "langfuse", ModuleType("langfuse"))
    config_path = get_hermes_home() / "config.yaml"
    config_path.write_text(
        "plugins:\n  enabled: [other]\n  disabled: [langfuse, observability/langfuse]\n",
        encoding="utf-8",
    )
    before = config_path.read_bytes()

    def resolve_candidate(**kwargs):
        assert kwargs["explicit"] is True
        kwargs["plugin_dirs"]()
        if failure == "admission":
            raise pm.InstallError("venv", "candidate refused")
        # The real admission publisher must commit both lists, not a second UI writer.
        publication = kwargs["before_publish"]()
        publication.finish()

    with (
        patch("pm.sync_venv", side_effect=pm.InstallError("venv", "SDK refused") if failure == "sdk" else None) as sdk,
        patch("pm.client.sync_venv", side_effect=resolve_candidate) as admission,
    ):
        _run_post_setup("langfuse")

    sdk.assert_called_once_with(["langfuse"], explicit=True)
    if failure == "sdk":
        admission.assert_not_called()
    else:
        admission.assert_called_once()
    if failure:
        assert config_path.read_bytes() == before
        output = capsys.readouterr().out
        assert "refused" in output
        if failure == "sdk":
            assert "Retry with: hermes tools" in output
    else:
        plugin_config = read_raw_config()["plugins"]
        assert set(plugin_config["enabled"]) == {"other", "observability/langfuse"}
        assert plugin_config["disabled"] == []
