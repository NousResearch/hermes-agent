"""Regression coverage for #102023.

``hermes status`` used to print ``.env file: ✗ not found`` for the very
checkout whose project-root ``.env`` it had just read credentials from, while
``hermes doctor`` reported the same file as present.  Both commands re-derived
the path themselves instead of asking the loader, so the two answers drifted.

These tests pin the shared resolver, the invariant that keeps it aligned with
``load_hermes_dotenv``, and the status line that consumes it.
"""

import os
from types import SimpleNamespace

from hermes_cli.env_loader import load_hermes_dotenv, resolve_env_sources


def _write_env(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def test_resolve_env_sources_empty_when_nothing_exists(tmp_path):
    home = tmp_path / "hermes"
    home.mkdir()

    assert resolve_env_sources(hermes_home=home, project_env=tmp_path / "repo" / ".env") == []


def test_resolve_env_sources_finds_user_env(tmp_path):
    home = tmp_path / "hermes"
    user_env = _write_env(home / ".env", "USER_ONLY=1\n")

    assert resolve_env_sources(hermes_home=home, project_env=tmp_path / "repo" / ".env") == [user_env]


def test_resolve_env_sources_finds_project_env(tmp_path):
    """The #102023 case: only the project-root .env that setup-hermes.sh creates."""
    home = tmp_path / "hermes"
    home.mkdir()
    project_env = _write_env(tmp_path / "repo" / ".env", "PROJECT_ONLY=1\n")

    assert resolve_env_sources(hermes_home=home, project_env=project_env) == [project_env]


def test_resolve_env_sources_orders_user_before_project(tmp_path):
    home = tmp_path / "hermes"
    user_env = _write_env(home / ".env", "BOTH=user\n")
    project_env = _write_env(tmp_path / "repo" / ".env", "BOTH=project\n")

    assert resolve_env_sources(hermes_home=home, project_env=project_env) == [user_env, project_env]


def test_resolve_env_sources_dedupes_when_home_is_the_checkout(tmp_path):
    """A checkout used as its own HERMES_HOME must not list one file twice."""
    home = tmp_path / "repo"
    env_file = _write_env(home / ".env", "SHARED=1\n")

    assert resolve_env_sources(hermes_home=home, project_env=env_file) == [env_file]


def test_loader_return_matches_resolver(tmp_path, monkeypatch):
    """The anti-drift invariant: what the loader reports loading is what the
    resolver advertises.  Reporting commands read the resolver, so any future
    change to loading order has to move both together or fail here."""
    home = tmp_path / "hermes"
    _write_env(home / ".env", "DRIFT_USER=user\n")
    project_env = _write_env(tmp_path / "repo" / ".env", "DRIFT_PROJECT=project\n")
    monkeypatch.delenv("DRIFT_USER", raising=False)
    monkeypatch.delenv("DRIFT_PROJECT", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home, project_env=project_env)

    assert loaded == resolve_env_sources(hermes_home=home, project_env=project_env)


def test_project_env_still_loads_as_fallback(tmp_path, monkeypatch):
    """Behavior preserved: a lone project .env is loaded, not merely reported."""
    home = tmp_path / "hermes"
    home.mkdir()
    project_env = _write_env(tmp_path / "repo" / ".env", "FALLBACK_ONLY_KEY=from-project\n")
    monkeypatch.delenv("FALLBACK_ONLY_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home, project_env=project_env)

    assert loaded == [project_env]
    assert os.environ["FALLBACK_ONLY_KEY"] == "from-project"


def test_user_env_still_wins_over_project_env(tmp_path, monkeypatch):
    """Precedence preserved: the project .env only fills gaps when a user .env exists."""
    home = tmp_path / "hermes"
    _write_env(home / ".env", "PRECEDENCE_KEY=from-user\n")
    project_env = _write_env(
        tmp_path / "repo" / ".env",
        "PRECEDENCE_KEY=from-project\nPROJECT_GAP_KEY=gap-filled\n",
    )
    monkeypatch.delenv("PRECEDENCE_KEY", raising=False)
    monkeypatch.delenv("PROJECT_GAP_KEY", raising=False)

    load_hermes_dotenv(hermes_home=home, project_env=project_env)

    assert os.environ["PRECEDENCE_KEY"] == "from-user"
    assert os.environ["PROJECT_GAP_KEY"] == "gap-filled"


def test_status_reports_project_env_instead_of_not_found(tmp_path, monkeypatch, capsys):
    """#102023: status claimed 'not found' for a project .env it was reading."""
    from hermes_cli import status as status_mod

    home = tmp_path / "hermes"
    home.mkdir()
    repo = tmp_path / "repo"
    _write_env(repo / ".env", "STATUS_PROJECT_KEY=1\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(status_mod, "PROJECT_ROOT", repo)

    status_mod.show_status(SimpleNamespace(all=False, deep=False))

    output = capsys.readouterr().out
    assert ".env file:" in output
    assert "exists (in project directory)" in output
    assert "not found" not in output.split("Model:")[0]


def test_status_reports_not_found_when_no_env_exists(tmp_path, monkeypatch, capsys):
    from hermes_cli import status as status_mod

    home = tmp_path / "hermes"
    home.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(status_mod, "PROJECT_ROOT", repo)

    status_mod.show_status(SimpleNamespace(all=False, deep=False))

    output = capsys.readouterr().out
    assert "not found" in output.split("Model:")[0]
