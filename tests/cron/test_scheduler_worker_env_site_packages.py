"""Regression for #125999: the cron external worker's PYTHONPATH pin restored the repo
root but not the runtime's own site-packages, so any third-party import on the worker's
bootstrap path (``ruamel.yaml`` via ``hermes_yaml``, ``dotenv`` via
``hermes_cli.env_loader``) died with ModuleNotFoundError before the worker could send its
ownership acknowledgement (#112729 follow-up)."""

from cron.scheduler_worker_env import pin_hermes_tree_on_pythonpath


def test_pin_restores_site_packages_alongside_repo_root(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    site_packages = tmp_path / "venv" / "lib" / "python3.14" / "site-packages"
    site_packages.mkdir(parents=True)

    monkeypatch.setattr(
        "tools.environments.local_pythonpath._get_hermes_site_packages",
        lambda env: [site_packages],
    )
    monkeypatch.setattr("cron.scheduler_worker_env._installed_purelib", lambda: None)

    env = pin_hermes_tree_on_pythonpath({"PYTHONPATH": "/existing/user/path"}, repo_root)

    entries = env["PYTHONPATH"].split(":")
    assert str(repo_root) in entries
    assert str(site_packages) in entries
    assert "/existing/user/path" in entries


def test_pin_is_a_noop_under_a_purelib_install(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr("cron.scheduler_worker_env._installed_purelib", lambda: repo_root)

    env = pin_hermes_tree_on_pythonpath({}, repo_root)

    assert "PYTHONPATH" not in env
