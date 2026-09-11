"""A one-commit artifact cannot enter any self-update path."""

import json
from unittest.mock import Mock

import pytest

from hermes_cli.update_contract import evaluate_update_admission

MESSAGE = "This build doesn't get updates. Ask the developer who gave it to you for a new build."


@pytest.fixture
def commit_build(tmp_path, monkeypatch):
    root = tmp_path / "app"
    root.mkdir()
    (root / "install-stamp.json").write_text(json.dumps({
        "source": "commit-build", "distribution": "desktop-app", "payload": "bundled",
        "updateMechanism": "external", "commit": "a" * 40,
        "displayVersion": "1.2.3+gabcdef12", "baseVersion": "1.2.3", "channel": "canary",
    }))
    monkeypatch.setattr("hermes_cli.image_provenance.IMAGE_PROVENANCE_PATH", tmp_path / "absent")
    monkeypatch.setattr("hermes_cli.config.get_project_root", lambda: root)
    monkeypatch.setattr("hermes_cli.version_info._CODE_ROOT", root)
    monkeypatch.delenv("HERMES_INSTALL_ROOT", raising=False)
    from hermes_cli.version_info import _reset_version_info_cache
    _reset_version_info_cache()
    yield root
    _reset_version_info_cache()


@pytest.mark.parametrize("git_present", [False, True])
def test_commit_build_refuses_without_gui_advice(commit_build, git_present):
    if git_present:
        (commit_build / ".git").mkdir()
    refusal = evaluate_update_admission(commit_build)
    assert refusal is not None
    assert refusal.code == "commit-build"
    assert refusal.message == MESSAGE
    assert refusal.update_command == ""


@pytest.mark.parametrize("passive", [False, True])
def test_commit_build_never_checks_upstream_or_reuses_source_cache(commit_build, monkeypatch, passive):
    from hermes_cli import banner

    # Even a shared home's source-checkout cache and an embedded SHA cannot turn this into an update.
    monkeypatch.setenv("HERMES_REVISION", "b" * 40)
    check = Mock(return_value=5)
    monkeypatch.setattr(banner, "_check_via_rev", check)
    cache = Mock(return_value={"behind": 5})
    monkeypatch.setattr(banner, "_read_json", cache)
    assert banner.check_for_updates(passive=passive) is None
    check.assert_not_called()
    cache.assert_not_called()


def test_commit_version_banner_uses_stamp_not_shared_checkout(commit_build, monkeypatch):
    from hermes_cli import banner
    from hermes_cli.version_info import get_version_info

    git = Mock(side_effect=AssertionError("version must not inspect another checkout"))
    monkeypatch.setattr(banner, "get_git_banner_state", git)
    label = banner.format_banner_version_label()
    assert "1.2.3+gabcdef12" in label
    assert "commit-build" in label
    assert get_version_info().source == "commit-build"
    git.assert_not_called()


@pytest.mark.parametrize("managed", [False, True])
def test_commit_backend_update_routes_refuse_before_checks_or_spawns(commit_build, monkeypatch, tmp_path, managed):
    from starlette.testclient import TestClient
    import hermes_cli.web_server as server
    import hermes_cli.web_server_gateway as gateway
    from hermes_cli import banner
    from hermes_constants import get_hermes_home

    monkeypatch.setattr(server, "PROJECT_ROOT", commit_build)
    monkeypatch.setattr(gateway, "_ACTION_LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr("hermes_cli.web_server_files._dashboard_local_update_managed_externally", lambda: managed)
    spawn = Mock(side_effect=AssertionError("no updater process"))
    monkeypatch.setattr(gateway, "_spawn_hermes_action", spawn)
    check = Mock(side_effect=AssertionError("no update check"))
    monkeypatch.setattr(banner, "check_for_updates", check)
    cache = get_hermes_home() / ".update_check"
    cache.write_text("source checkout cache")
    client = TestClient(server.app)
    client.headers[server._SESSION_HEADER_NAME] = server._SESSION_TOKEN
    result = client.get("/api/hermes/update/check?force=true")
    assert result.status_code == 200
    assert result.json()["message"] == MESSAGE
    assert result.json()["can_apply"] is False
    assert result.json()["current_version"] == "1.2.3+gabcdef12"
    assert client.post("/api/hermes/update").json()["message"] == MESSAGE
    assert cache.read_text() == "source checkout cache"
    check.assert_not_called()
    spawn.assert_not_called()
