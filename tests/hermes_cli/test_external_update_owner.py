"""A checkout stamped ``external`` hands its updates to another tool: every update path skips it."""

import subprocess
from unittest.mock import Mock

import pytest

from hermes_cli.update_contract import evaluate_update_admission

COMMAND = "~/bin/rebuild-hermes --sync"


def _git(root, *args):
    subprocess.run(
        ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
         "-c", "commit.gpgsign=false", *args],
        cwd=root, check=True, capture_output=True,
    )


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """A git checkout on a local branch with an unmerged commit, stamped external."""
    root = tmp_path / "checkout"
    root.mkdir()
    _git(root, "init", "-b", "main")
    _git(root, "commit", "--allow-empty", "-m", "base")
    _git(root, "checkout", "-b", "personal-build")
    _git(root, "commit", "--allow-empty", "-m", "local work")
    (root / ".install_method").write_text("external\n")
    (root / ".update_command").write_text(f"{COMMAND}\n")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    monkeypatch.delenv("HERMES_REVISION", raising=False)
    monkeypatch.setattr("hermes_cli.image_provenance.IMAGE_PROVENANCE_PATH", tmp_path / "absent")
    monkeypatch.setattr("hermes_cli.config.get_project_root", lambda: root)
    return root


def test_admission_skips_with_the_named_command_and_exit_zero(checkout):
    refusal = evaluate_update_admission(checkout)
    assert refusal is not None
    assert refusal.code == "external"
    assert refusal.exit_code == 0
    assert refusal.update_command == COMMAND
    assert refusal.message == f"This checkout's updates are managed externally — run: {COMMAND}"


def test_admission_without_a_command_still_explains_the_skip(checkout):
    (checkout / ".update_command").unlink()
    refusal = evaluate_update_admission(checkout)
    assert refusal is not None and refusal.code == "external"
    assert refusal.update_command == ""
    assert "managed externally" in refusal.message


def test_an_unstamped_checkout_still_updates_in_place(checkout):
    (checkout / ".install_method").unlink()
    assert evaluate_update_admission(checkout) is None


def test_a_shared_home_stamp_does_not_hand_off_other_installs(checkout, tmp_path):
    from hermes_cli.config import detect_install_method

    (checkout / ".install_method").unlink()
    (tmp_path / "home" / ".install_method").write_text("external\n")
    assert detect_install_method(checkout) == "git"
    assert evaluate_update_admission(checkout) is None


def test_cmd_update_exits_zero_and_touches_nothing(checkout, monkeypatch, capsys):
    import argparse

    from hermes_cli import main

    monkeypatch.setattr(main, "PROJECT_ROOT", checkout)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=checkout, capture_output=True, text=True).stdout
    args = argparse.Namespace(check=False, plan=False, yes=True, branch=None, channel=None)
    with pytest.raises(SystemExit) as exit_info:
        main.cmd_update(args)
    assert exit_info.value.code == 0
    assert f"run: {COMMAND}" in capsys.readouterr().out
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=checkout, capture_output=True, text=True)
    assert branch.stdout.strip() == "personal-build"
    assert subprocess.run(["git", "rev-parse", "HEAD"], cwd=checkout, capture_output=True, text=True).stdout == head
    assert subprocess.run(["git", "stash", "list"], cwd=checkout, capture_output=True, text=True).stdout == ""


def test_update_plan_reports_not_updatable_with_the_command(checkout):
    from hermes_cli.update_inventory import collect_runtime_inventory

    plan = collect_runtime_inventory()
    assert plan.install_method == "external"
    assert plan.updatable_in_place is False
    assert plan.update_mechanism == COMMAND


def test_desktop_source_check_reports_the_skip_without_probing(checkout, monkeypatch):
    from hermes_cli import source_check

    probe = Mock(side_effect=AssertionError("must not probe upstream"))
    monkeypatch.setattr(source_check, "_branch_tip", probe)
    status = source_check.check_for_updates(install_root=checkout, home=checkout.parent / "home")
    assert status["supported"] is False
    assert status["reason"] == "external"
    assert status["advice"] == COMMAND
    probe.assert_not_called()


def test_backend_update_routes_skip_before_checks_or_spawns(checkout, monkeypatch, tmp_path):
    from starlette.testclient import TestClient

    import hermes_cli.web_server as server
    import hermes_cli.web_server_gateway as gateway
    from hermes_cli import source_check

    monkeypatch.setattr(server, "PROJECT_ROOT", checkout)
    monkeypatch.setattr(gateway, "_ACTION_LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr("hermes_cli.web_server_files._dashboard_local_update_managed_externally", lambda: False)
    spawn = Mock(side_effect=AssertionError("no updater process"))
    monkeypatch.setattr(gateway, "_spawn_hermes_action", spawn)
    monkeypatch.setattr(source_check, "check_for_updates", Mock(side_effect=AssertionError("no update check")))
    client = TestClient(server.app)
    client.headers[server._SESSION_HEADER_NAME] = server._SESSION_TOKEN

    check = client.get("/api/hermes/update/check?force=true").json()
    assert check["can_apply"] is False
    assert check["update_command"] == COMMAND
    assert "managed externally" in check["message"]

    applied = client.post("/api/hermes/update").json()
    assert applied["ok"] is False
    assert applied["error"] == "update_managed_externally"
    assert applied["update_command"] == COMMAND
    spawn.assert_not_called()
