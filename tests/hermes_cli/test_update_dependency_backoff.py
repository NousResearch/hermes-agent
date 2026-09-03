"""Dependency-install fallback tests for ``hermes update``."""

import subprocess
from types import SimpleNamespace

import pytest

from hermes_cli import update_cmd


def _completed(cmd, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)


def test_daily_backoff_candidates_select_one_upstream_commit_per_day(monkeypatch, tmp_path):
    responses = {
        "@1727913600": "day1\n",
        "@1727827200": "day2\n",
        "@1727740800": "day2\n",  # no commit that day: deduplicated
        "@1727654400": "too-old\n",
    }

    def fake_run(cmd, **_kwargs):
        joined = " ".join(cmd)
        if "show -s --format=%ct tip" in joined:
            return _completed(cmd, stdout="1728000000\n")
        if "rev-list -1 --first-parent --before=" in joined:
            stamp = next(key for key in responses if key in joined)
            return _completed(cmd, stdout=responses[stamp])
        if "merge-base --is-ancestor base" in joined:
            return _completed(cmd, returncode=1 if cmd[-1] == "too-old" else 0)
        raise AssertionError(cmd)

    monkeypatch.setattr(update_cmd.subprocess, "run", fake_run)

    assert update_cmd._daily_dependency_backoff_candidates(
        ["git"], tmp_path, tip_sha="tip", lower_bound_sha="base", max_days=4
    ) == ["day1", "day2"]


def test_dependency_attempt_backs_off_and_keeps_successful_upstream_target(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(
        update_cmd,
        "_daily_dependency_backoff_candidates",
        lambda *_args, **_kwargs: ["older-1", "older-2"],
    )
    checkouts = []

    def fake_run(cmd, **_kwargs):
        if "checkout" in cmd:
            checkouts.append(cmd[-1])
        return _completed(cmd)

    monkeypatch.setattr(update_cmd.subprocess, "run", fake_run)
    attempts = []

    def install(candidate):
        attempts.append(candidate)
        if candidate in {"tip", "older-1"}:
            raise subprocess.CalledProcessError(1, ["uv", "pip", "install"])
        return {"candidate": candidate}

    selected, result = update_cmd._run_dependency_attempts_with_daily_backoff(
        ["git"],
        tmp_path,
        tip_sha="tip",
        lower_bound_sha="base",
        install_attempt=install,
        max_days=8,
    )

    assert selected == "older-2"
    assert result == {"candidate": "older-2"}
    assert attempts == ["tip", "older-1", "older-2"]
    assert checkouts == ["older-1", "older-2"]
    assert "trying the update from 2 day(s) earlier" in capsys.readouterr().out


def test_dependency_attempt_treats_node_failure_as_candidate_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        update_cmd,
        "_daily_dependency_backoff_candidates",
        lambda *_args, **_kwargs: ["older"],
    )
    monkeypatch.setattr(
        update_cmd.subprocess,
        "run",
        lambda cmd, **_kwargs: _completed(cmd),
    )

    def install(candidate):
        if candidate == "tip":
            raise update_cmd.UpdateDependencyInstallError(
                "Node.js dependency install failed"
            )
        return "ok"

    assert update_cmd._run_dependency_attempts_with_daily_backoff(
        ["git"],
        tmp_path,
        tip_sha="tip",
        lower_bound_sha="base",
        install_attempt=install,
    ) == ("older", "ok")


def test_dependency_attempt_restores_original_source_when_every_candidate_fails(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        update_cmd,
        "_daily_dependency_backoff_candidates",
        lambda *_args, **_kwargs: ["older"],
    )
    checkouts = []

    def fake_run(cmd, **_kwargs):
        if "checkout" in cmd:
            checkouts.append(cmd[-1])
        return _completed(cmd)

    monkeypatch.setattr(update_cmd.subprocess, "run", fake_run)

    def install(_candidate):
        raise subprocess.CalledProcessError(1, ["uv", "pip", "install"])

    failed_callbacks = []
    with pytest.raises(update_cmd.UpdateDependencyInstallError, match="all update targets"):
        update_cmd._run_dependency_attempts_with_daily_backoff(
            ["git"],
            tmp_path,
            tip_sha="tip",
            lower_bound_sha="base",
            install_attempt=install,
            on_all_failed=lambda: failed_callbacks.append(True),
        )

    assert checkouts == ["older", "base"]
    assert failed_callbacks == [True]


def test_dependency_backoff_refuses_checkout_when_files_appear_mid_update(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        update_cmd,
        "_daily_dependency_backoff_candidates",
        lambda *_args, **_kwargs: ["older"],
    )
    monkeypatch.setattr(
        update_cmd,
        "_working_tree_clean_for_dependency_backoff",
        lambda *_args, **_kwargs: False,
    )
    checkouts = []

    def fake_run(cmd, **_kwargs):
        if "merge-base" in cmd:
            return _completed(cmd)
        if "checkout" in cmd:
            checkouts.append(cmd[-1])
        return _completed(cmd)

    monkeypatch.setattr(update_cmd.subprocess, "run", fake_run)

    with pytest.raises(
        update_cmd.UpdateDependencyInstallError, match="working tree became dirty"
    ):
        update_cmd._run_dependency_attempts_with_daily_backoff(
            ["git"],
            tmp_path,
            tip_sha="tip",
            lower_bound_sha="base",
            install_attempt=lambda _candidate: (_ for _ in ()).throw(
                subprocess.CalledProcessError(1, ["uv", "pip", "install"])
            ),
        )

    assert checkouts == []


def test_checkout_toctou_preserves_edit_created_after_clean_probe(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        update_cmd,
        "_daily_dependency_backoff_candidates",
        lambda *_args, **_kwargs: ["older"],
    )
    cleanliness = iter([True, False])
    monkeypatch.setattr(
        update_cmd,
        "_working_tree_clean_for_dependency_backoff",
        lambda *_args, **_kwargs: next(cleanliness),
    )
    checkouts = []

    def fake_run(cmd, **_kwargs):
        if "merge-base" in cmd:
            return _completed(cmd)
        if "checkout" in cmd:
            checkouts.append(cmd[-1])
            # Real non-forced checkout refuses when the racing edit conflicts.
            return _completed(cmd, returncode=1, stderr="local changes would be overwritten")
        return _completed(cmd)

    monkeypatch.setattr(update_cmd.subprocess, "run", fake_run)

    with pytest.raises(
        update_cmd.UpdateDependencyInstallError,
        match="checkout refused without overwriting",
    ):
        update_cmd._run_dependency_attempts_with_daily_backoff(
            ["git"],
            tmp_path,
            tip_sha="tip",
            lower_bound_sha="base",
            install_attempt=lambda _candidate: (_ for _ in ()).throw(
                subprocess.CalledProcessError(1, ["uv", "pip", "install"])
            ),
        )

    assert checkouts == ["older"]


def test_git_checkout_B_is_transactional_when_local_edit_would_be_lost(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    assert git("init", "-b", "main").returncode == 0
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Hermes Test")
    tracked = repo / "tracked.txt"
    tracked.write_text("base\n", encoding="utf-8")
    git("add", "tracked.txt")
    assert git("commit", "-m", "base").returncode == 0
    base = git("rev-parse", "HEAD").stdout.strip()

    tracked.write_text("tip\n", encoding="utf-8")
    git("add", "tracked.txt")
    assert git("commit", "-m", "tip").returncode == 0
    tip = git("rev-parse", "HEAD").stdout.strip()
    tracked.write_text("user edit\n", encoding="utf-8")

    checkout = git("checkout", "-B", "main", base)

    assert checkout.returncode != 0
    assert git("rev-parse", "HEAD").stdout.strip() == tip
    assert tracked.read_text(encoding="utf-8") == "user edit\n"


def test_dependency_backoff_never_attempts_tip_below_pre_update_head(
    monkeypatch, tmp_path
):
    installs = []
    checkouts = []

    def fake_run(cmd, **_kwargs):
        if "merge-base" in cmd:
            return _completed(cmd, returncode=1)
        if "checkout" in cmd:
            checkouts.append(cmd[-1])
        return _completed(cmd)

    monkeypatch.setattr(update_cmd.subprocess, "run", fake_run)
    monkeypatch.setattr(
        update_cmd, "_working_tree_clean_for_dependency_backoff", lambda *_args: True
    )

    with pytest.raises(
        update_cmd.UpdateDependencyInstallError, match="does not contain the pre-update HEAD"
    ):
        update_cmd._run_dependency_attempts_with_daily_backoff(
            ["git"],
            tmp_path,
            tip_sha="tip",
            lower_bound_sha="base",
            install_attempt=lambda candidate: installs.append(candidate),
        )

    assert installs == []
    assert checkouts == ["base"]


def test_fast_dependency_network_env_disables_package_manager_retries():
    env = update_cmd._fast_update_dependency_env({"KEEP": "yes"})

    assert env["KEEP"] == "yes"
    assert env["UV_HTTP_RETRIES"] == "0"
    assert env["UV_HTTP_TIMEOUT"] == "15"
    assert env["PIP_RETRIES"] == "0"
    assert env["PIP_DEFAULT_TIMEOUT"] == "15"
    assert env["npm_config_fetch_retries"] == "0"
    assert env["npm_config_fetch_timeout"] == "15000"


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("npm ERR! code EUSAGE\nnpm ci can only install packages when package.json and package-lock.json are in sync", True),
        ("Unknown command: ci", True),
        ("npm ERR! 404 Not Found - GET https://registry/pkg", False),
        ("npm ERR! network request timed out", False),
    ],
)
def test_npm_ci_only_falls_back_to_install_for_lockfile_or_old_npm_errors(
    output, expected
):
    from hermes_cli import main as hm

    assert hm._npm_ci_failure_allows_install_fallback(output) is expected


def test_install_update_candidate_uses_fast_env_and_requires_node_success(
    monkeypatch, tmp_path
):
    installs = []
    fake_main = SimpleNamespace(
        PROJECT_ROOT=tmp_path,
        sys=SimpleNamespace(executable="python"),
        _is_windows=lambda: False,
        _is_termux_env=lambda _env=None: False,
        _install_python_dependencies_with_optional_fallback=lambda prefix, **kwargs: installs.append(
            (prefix, kwargs)
        ),
        _verify_core_dependencies_installed=lambda *_args, **_kwargs: None,
        _verify_console_scripts_installed=lambda *_args, **_kwargs: None,
        _clear_update_incomplete_marker=lambda: None,
    )
    monkeypatch.setattr(update_cmd, "_m", lambda: fake_main)
    monkeypatch.setattr(update_cmd, "_editable_install_is_current", lambda *_args: False)
    node_forces = []
    monkeypatch.setattr(
        update_cmd,
        "_update_node_dependencies",
        lambda *, force=False: node_forces.append(force) or [],
    )
    monkeypatch.setattr("hermes_cli.managed_uv.update_managed_uv", lambda: None)
    monkeypatch.setattr("hermes_cli.managed_uv.ensure_uv", lambda: "uv")
    monkeypatch.setattr(
        "hermes_cli.managed_uv.managed_python_env", lambda: {"KEEP": "yes"}
    )

    state = update_cmd._install_update_candidate_dependencies(
        ["git"], pre_pull_sha="base"
    )

    assert installs[0][0] == ["uv", "pip"]
    install_env = installs[0][1]["env"]
    assert install_env["KEEP"] == "yes"
    assert install_env["UV_HTTP_RETRIES"] == "0"
    assert install_env["npm_config_fetch_retries"] == "0"
    assert installs[0][1]["require_all_extras"] is True
    assert state["install_prefix"] == ["uv", "pip"]
    assert state["node_failures"] == []
    assert node_forces == [False]


def test_install_update_candidate_raises_when_node_dependencies_fail(
    monkeypatch, tmp_path
):
    fake_main = SimpleNamespace(
        PROJECT_ROOT=tmp_path,
        sys=SimpleNamespace(executable="python"),
        _is_windows=lambda: False,
        _is_termux_env=lambda _env=None: False,
        _install_python_dependencies_with_optional_fallback=lambda *_args, **_kwargs: None,
        _verify_core_dependencies_installed=lambda *_args, **_kwargs: None,
        _verify_console_scripts_installed=lambda *_args, **_kwargs: None,
        _clear_update_incomplete_marker=lambda: None,
    )
    monkeypatch.setattr(update_cmd, "_m", lambda: fake_main)
    monkeypatch.setattr(update_cmd, "_editable_install_is_current", lambda *_args: False)
    monkeypatch.setattr(
        update_cmd,
        "_update_node_dependencies",
        lambda *, force=False: ["ui-tui, web workspaces"],
    )
    monkeypatch.setattr("hermes_cli.managed_uv.update_managed_uv", lambda: None)
    monkeypatch.setattr("hermes_cli.managed_uv.ensure_uv", lambda: "uv")
    monkeypatch.setattr("hermes_cli.managed_uv.managed_python_env", lambda: {})

    with pytest.raises(
        update_cmd.UpdateDependencyInstallError, match="Node.js dependency install failed"
    ):
        update_cmd._install_update_candidate_dependencies(
            ["git"], pre_pull_sha="base"
        )


def test_older_candidate_forces_clean_python_and_node_reinstall(monkeypatch, tmp_path):
    installs = []
    node_forces = []
    fake_main = SimpleNamespace(
        PROJECT_ROOT=tmp_path,
        sys=SimpleNamespace(executable="python"),
        _is_windows=lambda: False,
        _is_termux_env=lambda _env=None: False,
        _install_python_dependencies_with_optional_fallback=lambda prefix, **kwargs: installs.append(
            (prefix, kwargs)
        ),
        _verify_core_dependencies_installed=lambda *_args, **_kwargs: None,
        _verify_console_scripts_installed=lambda *_args, **_kwargs: None,
        _clear_update_incomplete_marker=lambda: None,
    )
    monkeypatch.setattr(update_cmd, "_m", lambda: fake_main)
    monkeypatch.setattr(update_cmd, "_editable_install_is_current", lambda *_args: True)
    monkeypatch.setattr(
        update_cmd,
        "_update_node_dependencies",
        lambda *, force=False: node_forces.append(force) or [],
    )
    monkeypatch.setattr("hermes_cli.managed_uv.update_managed_uv", lambda: None)
    monkeypatch.setattr("hermes_cli.managed_uv.ensure_uv", lambda: "uv")
    monkeypatch.setattr("hermes_cli.managed_uv.managed_python_env", lambda: {})

    update_cmd._install_update_candidate_dependencies(
        ["git"], pre_pull_sha="base", force_reinstall=True
    )

    assert len(installs) == 1
    assert installs[0][1]["require_all_extras"] is True
    assert node_forces == [True]


def test_older_candidate_requires_uv_for_exact_environment(monkeypatch, tmp_path):
    fake_main = SimpleNamespace(
        PROJECT_ROOT=tmp_path,
        sys=SimpleNamespace(executable="python"),
        _is_windows=lambda: False,
        _is_termux_env=lambda _env=None: False,
    )
    monkeypatch.setattr(update_cmd, "_m", lambda: fake_main)
    monkeypatch.setattr(update_cmd, "_editable_install_is_current", lambda *_args: True)
    monkeypatch.setattr(update_cmd, "_ensure_uv_for_termux", lambda _cmd: None)
    monkeypatch.setattr("hermes_cli.managed_uv.update_managed_uv", lambda: None)
    monkeypatch.setattr("hermes_cli.managed_uv.ensure_uv", lambda: None)

    with pytest.raises(
        update_cmd.UpdateDependencyInstallError, match="requires uv exact-sync"
    ):
        update_cmd._install_update_candidate_dependencies(
            ["git"], pre_pull_sha="base", force_reinstall=True
        )


def test_all_fail_raises_when_source_rollback_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(
        update_cmd, "_daily_dependency_backoff_candidates", lambda *_a, **_k: []
    )
    monkeypatch.setattr(
        update_cmd, "_working_tree_clean_for_dependency_backoff", lambda *_a: True
    )

    def fake_run(cmd, **_kwargs):
        if "merge-base" in cmd:
            return _completed(cmd)
        if "checkout" in cmd:
            return _completed(cmd, returncode=1, stderr="cannot checkout")
        return _completed(cmd)

    monkeypatch.setattr(update_cmd.subprocess, "run", fake_run)

    with pytest.raises(
        update_cmd.UpdateDependencyInstallError, match="failed to restore pre-update source"
    ):
        update_cmd._run_dependency_attempts_with_daily_backoff(
            ["git"],
            tmp_path,
            tip_sha="tip",
            lower_bound_sha="base",
            install_attempt=lambda _candidate: (_ for _ in ()).throw(
                subprocess.CalledProcessError(1, ["uv", "pip", "install"])
            ),
        )


def test_deterministic_npm_does_not_retry_registry_failure_as_install(
    monkeypatch, tmp_path
):
    from hermes_cli import main as hm

    (tmp_path / "package-lock.json").write_text("{}", encoding="utf-8")
    calls = []

    def fail_registry(cmd, **_kwargs):
        calls.append(cmd)
        return _completed(cmd, 1, stderr="npm ERR! 404 Not Found")

    monkeypatch.setattr(hm, "_run_npm_watching_for_engine_failure", fail_registry)
    monkeypatch.setattr("hermes_cli.npm_engine.maybe_repair_npm_engine", lambda *_args: None)

    result = hm._run_npm_install_deterministic("npm", tmp_path)

    assert result.returncode == 1
    assert [cmd[1] for cmd in calls] == ["ci"]


def test_deterministic_npm_retries_lockfile_drift_as_install(monkeypatch, tmp_path):
    from hermes_cli import main as hm

    (tmp_path / "package-lock.json").write_text("{}", encoding="utf-8")
    calls = []

    def run(cmd, **_kwargs):
        calls.append(cmd)
        if cmd[1] == "ci":
            return _completed(cmd, 1, stderr="npm ERR! code EUSAGE")
        return _completed(cmd)

    monkeypatch.setattr(hm, "_run_npm_watching_for_engine_failure", run)

    result = hm._run_npm_install_deterministic("npm", tmp_path)

    assert result.returncode == 0
    assert [cmd[1] for cmd in calls] == ["ci", "install"]


def test_strict_python_candidate_fails_after_one_exact_uv_attempt(monkeypatch):
    from hermes_cli import main as hm

    calls = []

    def fail(cmd, **_kwargs):
        calls.append(cmd)
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(hm, "_is_windows", lambda: False)
    monkeypatch.setattr(hm, "_run_quarantined_install", fail)

    with pytest.raises(subprocess.CalledProcessError):
        hm._install_python_dependencies_with_optional_fallback(
            ["uv", "pip"], require_all_extras=True
        )

    assert len(calls) == 1
    assert calls[0] == ["uv", "pip", "install", "--exact", "-e", ".[all]"]


def test_forced_node_candidate_bypasses_success_hash_cache(monkeypatch, tmp_path):
    from hermes_cli import main as hm

    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    (tmp_path / "package-lock.json").write_text("{}", encoding="utf-8")
    npm_calls = []

    monkeypatch.setattr(hm, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(hm, "_resolve_node_runtime_npm", lambda: "npm")
    monkeypatch.setattr(hm, "_npm_lockfile_changed", lambda _root: False)
    monkeypatch.setattr(
        hm,
        "_run_npm_install_deterministic",
        lambda *args, **kwargs: npm_calls.append((args, kwargs))
        or _completed(["npm", "ci"]),
    )
    monkeypatch.setattr(
        "tools.browser_tool.warm_agent_browser_npx_cache", lambda: True
    )

    assert update_cmd._update_node_dependencies(force=True) == []
    assert len(npm_calls) == 1
