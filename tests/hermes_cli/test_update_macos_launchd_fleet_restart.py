"""macOS update restart coverage for launchd-managed profile gateways."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest


_DEFAULT_LAUNCHD_JOBS = {
    "ai.hermes.gateway": 101,
    "ai.hermes.gateway-builder": 201,
    "ai.hermes.gateway-writer": 202,
    "ai.hermes.gateway-dormant": None,
    "com.example.unrelated": 303,
}


def _run_mocked_macos_update(
    tmp_path,
    monkeypatch,
    capsys,
    *,
    launchd_jobs: dict[str, int | None] | None = None,
    launchctl_list_failure: str | None = None,
    restart_failure_label: str | None = "ai.hermes.gateway-writer",
):
    from hermes_cli import gateway as gateway_cli
    from hermes_cli import main as hm
    import hermes_cli.config as config_cli
    import hermes_cli.managed_uv as managed_uv
    import hermes_cli.profiles as profiles_cli
    import tools.skills_sync as skills_sync

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").mkdir()
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    active_plist = tmp_path / "ai.hermes.gateway.plist"
    active_plist.write_text("<plist />", encoding="utf-8")

    jobs = dict(_DEFAULT_LAUNCHD_JOBS if launchd_jobs is None else launchd_jobs)
    restart_calls: list[str] = []

    def fake_run(cmd, **kwargs):
        if cmd[:3] == ["git", "fetch", "origin"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:3] == ["git", "rev-parse", "--abbrev-ref"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="main\n", stderr="")
        if cmd[:2] == ["git", "rev-list"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="1\n", stderr="")
        if cmd[:3] == ["git", "merge", "--ff-only"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:2] == ["launchctl", "list"] and len(cmd) == 2:
            if launchctl_list_failure == "nonzero":
                return subprocess.CompletedProcess(
                    cmd,
                    1,
                    stdout="",
                    stderr="launchctl could not enumerate jobs\n",
                )
            if launchctl_list_failure == "timeout":
                raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 5))
            if launchctl_list_failure == "missing":
                raise FileNotFoundError("launchctl")
            rows = [
                f"{pid if pid is not None else '-'}\t0\t{label}"
                for label, pid in jobs.items()
            ]
            if "ai.hermes.gateway-writer" in jobs:
                rows.append("202\t0\tai.hermes.gateway-writer")
            return subprocess.CompletedProcess(
                cmd, 0, stdout="\n".join(rows), stderr=""
            )
        if cmd[:2] == ["launchctl", "list"] and len(cmd) == 3:
            pid = jobs.get(cmd[2])
            if pid is None:
                return subprocess.CompletedProcess(
                    cmd, 0, stdout='"PID" = -1;\n', stderr=""
                )
            return subprocess.CompletedProcess(
                cmd, 0, stdout=f'"PID" = {pid};\n', stderr=""
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    def fake_launchd_restart(label: str | None = None):
        restarted_label = label or gateway_cli.get_launchd_label()
        restart_calls.append(restarted_label)
        if restarted_label == restart_failure_label:
            raise subprocess.CalledProcessError(
                75,
                ["launchctl", "kickstart", "-k", f"gui/501/{restarted_label}"],
                stderr=f"{restarted_label} failed to restart",
            )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(hm, "PROJECT_ROOT", checkout)
    monkeypatch.setattr(config_cli, "PROJECT_ROOT", checkout, raising=False)
    monkeypatch.setattr(hm, "_run_pre_update_backup", lambda args: None)
    monkeypatch.setattr(hm, "_pause_windows_gateways_for_update", lambda: None)
    monkeypatch.setattr(hm, "_resume_windows_gateways_after_update", lambda token: None)
    monkeypatch.setattr(hm, "_is_windows", lambda: False)
    monkeypatch.setattr(
        hm,
        "_get_origin_url",
        lambda git_cmd, cwd: "git@github.com:NousResearch/hermes-agent.git",
    )
    monkeypatch.setattr(hm.shutil, "which", lambda command: None)
    monkeypatch.setattr(hm, "_stash_local_changes_if_needed", lambda git_cmd, cwd: None)
    monkeypatch.setattr(hm, "_capture_head_sha", lambda git_cmd, cwd: "pre-update-sha")
    monkeypatch.setattr(
        hm, "_validate_critical_files_syntax", lambda root: (True, None, None)
    )
    monkeypatch.setattr(hm, "_clear_bytecode_cache", lambda root: 0)
    monkeypatch.setattr(hm, "_record_bytecode_fingerprint", lambda: None)
    monkeypatch.setattr(hm, "_reload_updated_runtime_modules", lambda: None)
    monkeypatch.setattr(
        hm, "_upgrade_pip_before_lazy_refresh", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        hm, "_refresh_active_lazy_features", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(hm, "_clear_lazy_refresh_incomplete_marker", lambda: None)
    monkeypatch.setattr(hm, "_refresh_active_memory_provider_dependencies", lambda: None)
    monkeypatch.setattr(hm, "_update_node_dependencies", lambda: [])
    monkeypatch.setattr(hm, "_build_web_ui", lambda path: None)
    monkeypatch.setattr(hm, "_desktop_packaged_executable", lambda path: None)
    monkeypatch.setattr(hm, "_desktop_dist_exists", lambda path: False)
    monkeypatch.setattr(hm, "_finish_dashboard_update_cleanup", lambda failures: None)
    monkeypatch.setattr(hm, "_ensure_fhs_path_guard", lambda: None)
    monkeypatch.setattr(hm, "_ensure_acp_launcher", lambda: None)
    monkeypatch.setattr(hm, "_print_fts_optimize_available_notice", lambda: None)
    monkeypatch.setattr(hm, "_print_curator_first_run_notice", lambda: None)
    monkeypatch.setattr(hm, "_print_curator_recent_run_notice", lambda: None)
    monkeypatch.setattr(gateway_cli, "is_macos", lambda: True)
    monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: False)
    monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: active_plist)
    monkeypatch.setattr(gateway_cli, "launchd_restart", fake_launchd_restart)
    monkeypatch.setattr(
        gateway_cli,
        "_get_service_pids",
        lambda: {pid for pid in jobs.values() if pid is not None},
    )
    monkeypatch.setattr(
        gateway_cli,
        "find_gateway_pids",
        lambda exclude_pids=None, all_profiles=False: [],
    )
    monkeypatch.setattr(
        gateway_cli, "find_profile_gateway_processes", lambda exclude_pids=None: []
    )
    monkeypatch.setattr(gateway_cli, "has_legacy_hermes_units", lambda: False)
    monkeypatch.setattr(managed_uv, "ensure_uv", lambda **kwargs: None)
    monkeypatch.setattr(managed_uv, "update_managed_uv", lambda **kwargs: None)
    monkeypatch.setattr(
        skills_sync,
        "sync_skills",
        lambda quiet=True: {
            "copied": [],
            "updated": [],
            "user_modified": [],
            "cleaned": [],
        },
    )
    monkeypatch.setattr(profiles_cli, "list_profiles", lambda: [])
    monkeypatch.setattr(profiles_cli, "backfill_profile_envs", lambda quiet=True: [])
    monkeypatch.setattr(hm.subprocess, "run", fake_run)
    monkeypatch.setattr(
        "hermes_cli.config.get_missing_env_vars", lambda required_only=True: []
    )
    monkeypatch.setattr("hermes_cli.config.get_missing_config_fields", lambda: [])
    monkeypatch.setattr("hermes_cli.config.check_config_version", lambda: (1, 1))

    def run_update():
        return hm._cmd_update_impl(
            SimpleNamespace(yes=True, force=False, force_venv=False),
            gateway_mode=False,
        )

    return run_update, restart_calls


def test_macos_update_restarts_running_launchd_gateway_fleet_once(
    tmp_path, monkeypatch, capsys
):
    """Post-update macOS refresh must cover every running Hermes launchd gateway."""
    run_update, restart_calls = _run_mocked_macos_update(
        tmp_path, monkeypatch, capsys
    )

    with pytest.raises(SystemExit) as excinfo:
        run_update()

    assert excinfo.value.code == 1
    assert restart_calls == [
        "ai.hermes.gateway",
        "ai.hermes.gateway-builder",
        "ai.hermes.gateway-writer",
    ]
    assert restart_calls.count("ai.hermes.gateway-writer") == 1
    assert "ai.hermes.gateway-dormant" not in restart_calls
    assert "com.example.unrelated" not in restart_calls

    out = capsys.readouterr().out
    assert "ai.hermes.gateway-writer" in out
    assert "Update incomplete" in out


@pytest.mark.parametrize("launchctl_list_failure", ["nonzero", "timeout", "missing"])
def test_macos_update_fails_closed_when_launchd_fleet_discovery_fails(
    tmp_path, monkeypatch, capsys, launchctl_list_failure
):
    run_update, restart_calls = _run_mocked_macos_update(
        tmp_path,
        monkeypatch,
        capsys,
        launchctl_list_failure=launchctl_list_failure,
        restart_failure_label=None,
    )

    with pytest.raises(SystemExit) as excinfo:
        run_update()

    assert excinfo.value.code == 1
    assert restart_calls == []
    out = capsys.readouterr().out
    assert "macOS launchd gateway discovery" in out
    assert "Update incomplete" in out
    assert "\u2713 Update complete!" not in out


def test_macos_update_treats_successful_empty_launchd_discovery_as_complete(
    tmp_path, monkeypatch, capsys
):
    run_update, restart_calls = _run_mocked_macos_update(
        tmp_path,
        monkeypatch,
        capsys,
        launchd_jobs={},
        restart_failure_label=None,
    )

    run_update()

    assert restart_calls == []
    out = capsys.readouterr().out
    assert "macOS launchd gateway discovery" not in out
    assert "\u2713 Update complete!" in out
