"""A source install that still owes a step never ends on ``✓ Install complete!`` (#135210).

``install.sh`` / ``install.ps1`` run ``hermes_cli/source_completion.py`` for the products stage:
launchers, then the builds, then the post-build maintenance. A failed step is owed (printed as
``⚠ Update follow-up '<step>' did not finish``) and the stage exits 1, but the maintenance step
used to print ``✓ Install complete!`` anyway, so a failed bootstrap log ended on a success line.
An install has no commit point (C3 is the update's contract), so while a step is owed its
completion line names the owed steps instead. A committed update keeps its banner: the Desktop
hand-off reads ``Update complete!`` as the commit signal.
"""

import os
import subprocess
from pathlib import Path

import pytest

from hermes_cli import source_completion


def _checkout(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "hermes_cli").mkdir(parents=True)
    (root / "hermes_cli" / "source_completion.py").write_text("", encoding="utf-8")
    env = {"HOME": str(tmp_path), "PATH": os.environ["PATH"]}

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=root, env=env, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.name", "Hermes Test")
    git("config", "user.email", "hermes@example.invalid")
    git("add", "-A")
    git("commit", "-qm", "release")
    git("tag", "v0.21.4")
    return root


@pytest.fixture
def products_stage(tmp_path, monkeypatch, capsys):
    """Run the installer's products stage (``source_completion.main``, prepared phase) with the
    REAL completion tail and maintenance; only the slow or host-touching leaves are stubbed.
    ``run(owe=step)`` makes that step fail and returns ``(exit code, stdout)``."""
    from hermes_cli import update_cmd
    from hermes_cli import update_cmd_maint as maint

    root = _checkout(tmp_path)
    monkeypatch.setattr("hermes_cli.main.PROJECT_ROOT", root)
    monkeypatch.setattr("hermes_cli.update_lock.update_marker_path", lambda: tmp_path / ".update-marker")
    monkeypatch.setattr("pm.environments.activate_dependencies", lambda root: None)
    monkeypatch.setattr("hermes_cli._subprocess_compat.expose_pm_git", lambda root: None)
    monkeypatch.setattr("hermes_cli.macos_tcc_anchor.ensure_tcc_anchor", lambda: None)
    monkeypatch.setattr("hermes_cli.gitlock.fetch_full_commit_graph", lambda *a, **kw: False)
    monkeypatch.setattr("hermes_cli.model_catalog.seed_cache_from_checkout", lambda root: False)
    monkeypatch.setattr(update_cmd, "_post_update_sqlite_runtime_status", lambda: (True, None))
    monkeypatch.setattr(update_cmd, "_branch_head_suffix", lambda *a, **kw: "")
    for name in ("_verify_and_restore_state_dbs_post_update", "_invalidate_live_plugin_catalog_caches",
                 "_print_bundled_skills_sync_report", "_print_post_update_notices_and_self_heals"):
        monkeypatch.setattr(maint, name, lambda: None)

    def run(*, owe: str | None = None, update: bool = False) -> tuple[int, str]:
        def leaf(step: str, reason: str):
            def call(*_args, **_kwargs):
                if step == owe:
                    raise RuntimeError(reason)
            return call

        monkeypatch.setattr("hermes_cli.venv_sync.publish_launchers", leaf("launchers", "source launcher publication failed"))
        monkeypatch.setattr("hermes_cli.source_build.build_update_products", leaf("build", "web UI build: npm exited 1"))
        monkeypatch.setattr(maint, "_sync_profiles_after_update", leaf("profile_sync", "profile sync crashed"))
        monkeypatch.setattr(update_cmd, "_check_and_apply_config_migration", leaf("config_migration", "migration crashed"))
        argv = ["--source", str(root), *(["--finish-update"] if update else []), "--prepared"]
        code = source_completion.main(argv)
        return code, capsys.readouterr().out

    return run


@pytest.mark.parametrize("step", ["launchers", "build", "profile_sync", "config_migration"])
def test_install_with_an_owed_step_names_it_instead_of_claiming_success(products_stage, step):
    code, out = products_stage(owe=step)

    assert code == 1
    assert "✓ Install complete!" not in out, out
    assert f"⚠ Update follow-up '{step}' did not finish" in out
    assert f"⚠ Install did not finish; still owed: {step}" in out
    # The withheld ✓ line is not a SQLite verdict: no bogus unsafe-runtime follow-up.
    assert "sqlite_runtime" not in out


def test_finished_install_still_reports_success(products_stage):
    code, out = products_stage()

    assert code == 0
    assert "✓ Install complete!" in out
    assert "did not finish" not in out


def test_committed_update_keeps_its_completion_banner_while_a_step_is_owed(products_stage):
    # C3: the update's code is committed; the Desktop hand-off reads "Update complete!" as that
    # signal and the owed step as a follow-up, so only the install wording changes.
    code, out = products_stage(owe="build", update=True)

    assert code == 1
    assert "✓ Update complete!" in out
    assert "⚠ Update follow-up 'build' did not finish" in out
    assert "Install did not finish" not in out
