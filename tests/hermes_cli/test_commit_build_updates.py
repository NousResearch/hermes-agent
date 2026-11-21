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
    monkeypatch.setattr("pm.paths.repo_root", lambda: root)
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


@pytest.mark.platforms("posix")
def test_root_cannot_update_user_owned_install(tmp_path, monkeypatch):
    from hermes_cli.update_contract import ROOT_UPDATE_MESSAGE

    class UserStat:
        st_uid = 1000

    monkeypatch.setattr("hermes_cli.update_contract.os.geteuid", lambda: 0)
    monkeypatch.setattr(type(tmp_path), "stat", lambda self: UserStat())
    _neutralize_host_install_markers(monkeypatch, tmp_path)
    refusal = evaluate_update_admission(tmp_path)

    assert refusal is not None
    assert refusal.code == "root-owned-install"
    assert refusal.message == ROOT_UPDATE_MESSAGE


@pytest.mark.platforms("posix")
def test_root_can_update_root_owned_install(tmp_path, monkeypatch):
    class RootStat:
        st_uid = 0

    monkeypatch.setattr("hermes_cli.update_contract.os.geteuid", lambda: 0)
    monkeypatch.setattr(type(tmp_path), "stat", lambda self: RootStat())
    _neutralize_host_install_markers(monkeypatch, tmp_path)

    assert evaluate_update_admission(tmp_path) is None


def _neutralize_host_install_markers(monkeypatch, tmp_path) -> None:
    """Pin the ownership-admission tests to Layer 0 only.

    These tests exercise the root/install-owner gate, but ``evaluate_update_admission``
    also consults host state below it: the baked ``/etc/hermes/image-provenance.json``
    marker (a verifier host that itself runs an image-managed install flips the
    admissible case into an unrelated ``image-marker`` refusal) and the
    ``HERMES_MANAGED`` signal. Both describe the HOST, not the fixture, so they are
    neutralized here.
    """
    monkeypatch.setattr("hermes_cli.image_provenance.IMAGE_PROVENANCE_PATH", tmp_path / "absent")
    monkeypatch.delenv("HERMES_MANAGED", raising=False)


@pytest.mark.platforms("posix")
def test_root_owned_admission_yields_to_baked_image_marker(tmp_path, monkeypatch):
    """A valid baked image marker outranks root-owned admission.

    Inter-layer contract: the ownership gate (Layer 0) admits a root-owned
    install, but an image-managed host (Layer 1) still refuses in-place
    updates — verifier hosts that are themselves image-managed installs see
    ``image-marker``, not ``None``, here.
    """
    marker = tmp_path / "image-provenance.json"
    marker.write_text(json.dumps({
        "schema": 1, "deployment_kind": "image", "manager": "docker", "image": "hermes:test",
    }))
    monkeypatch.setattr("hermes_cli.image_provenance.IMAGE_PROVENANCE_PATH", marker)

    class RootStat:
        st_uid = 0

    monkeypatch.setattr("hermes_cli.update_contract.os.geteuid", lambda: 0)
    monkeypatch.setattr(type(tmp_path), "stat", lambda self: RootStat())
    monkeypatch.delenv("HERMES_MANAGED", raising=False)

    refusal = evaluate_update_admission(tmp_path)
    assert refusal is not None
    assert refusal.code == "image-marker"


@pytest.mark.parametrize("passive", [False, True])
def test_commit_build_never_checks_upstream_or_reuses_source_cache(commit_build, monkeypatch, passive):
    from hermes_cli import banner, source_check

    # Even a shared home's source-checkout cache and an embedded SHA cannot turn this into an update.
    monkeypatch.setenv("HERMES_REVISION", "b" * 40)
    check = Mock(side_effect=AssertionError("must not probe"))
    monkeypatch.setattr(source_check, "_branch_tip", check)
    assert source_check.check_for_updates(passive=passive)["behind"] is None
    check.assert_not_called()


def test_commit_version_banner_uses_stamp_not_shared_checkout(commit_build, monkeypatch):
    from hermes_cli import banner, source_check
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
    from hermes_cli import banner, source_check
    from hermes_constants import get_hermes_home

    monkeypatch.setattr(server, "PROJECT_ROOT", commit_build)
    monkeypatch.setattr(gateway, "_ACTION_LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr("hermes_cli.web_server_files._dashboard_local_update_managed_externally", lambda: managed)
    spawn = Mock(side_effect=AssertionError("no updater process"))
    monkeypatch.setattr(gateway, "_spawn_hermes_action", spawn)
    check = Mock(side_effect=AssertionError("no update check"))
    monkeypatch.setattr(source_check, "check_for_updates", check)
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
