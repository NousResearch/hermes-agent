"""A profile .env that exists but cannot be read, as a sandbox launcher leaves it.

The launcher withholds the profile's .env, passes the keys it allows in as environment and pins
``terminal.backend`` through managed scope. Under that pin the file loads as absent, and neither a
project .env nor ``/reload`` may replace or remove the launcher's keys. Without the pin the read error
still stops the load.
"""

import builtins
import io
import logging
import os
from pathlib import Path

import pytest

from hermes_cli import managed_scope
from hermes_cli.env_loader import load_hermes_dotenv


def _deny_reads(monkeypatch, path: Path) -> None:
    """Make *path* exist but refuse every read, as Windows does for a file a sandbox denies."""
    denied = os.path.abspath(path)
    real_open = io.open

    def guarded_open(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and os.path.abspath(file) == denied:
            raise PermissionError(13, "Access is denied", os.fspath(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(io, "open", guarded_open)
    monkeypatch.setattr(builtins, "open", guarded_open)


@pytest.fixture
def managed_dir(tmp_path, monkeypatch):
    md = tmp_path / "managed"
    md.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(md))
    managed_scope.invalidate_managed_cache()
    yield md
    managed_scope.invalidate_managed_cache()


def _write_managed(md: Path, body: str) -> None:
    (md / "config.yaml").write_text(body, encoding="utf-8")
    managed_scope.invalidate_managed_cache()


@pytest.fixture
def pinned(managed_dir):
    _write_managed(managed_dir, "terminal:\n  backend: busybox\n")
    return managed_dir


def _profile(tmp_path: Path) -> tuple[Path, Path]:
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    env_file.write_text("OPENAI_API_KEY=from-file\n", encoding="utf-8")
    return home, env_file


def test_unreadable_user_env_loads_as_absent_under_a_pinned_backend(tmp_path, monkeypatch, caplog, pinned):
    home, env_file = _profile(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "from-launcher")
    monkeypatch.setenv("HERMES_ACP_AUTH_METHOD", "from-launcher")  # a readable .env clears it when unset there
    _deny_reads(monkeypatch, env_file)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.env_loader"):
        for _ in range(3):  # gateways reload the .env every turn
            assert load_hermes_dotenv(hermes_home=home) == []

    assert os.environ["OPENAI_API_KEY"] == "from-launcher"
    assert os.environ["HERMES_ACP_AUTH_METHOD"] == "from-launcher"
    warnings = [r for r in caplog.records if str(env_file) in r.getMessage()]
    assert len(warnings) == 1


@pytest.mark.parametrize("managed_yaml", [
    None,
    "terminal:\n  timeout: 5\n",
    "terminal:\n  backend:\n",
    "terminal:\n  backend: ''\n",
    "terminal:\n  backend: '   '\n",
])
def test_unreadable_user_env_raises_without_a_pinned_backend(tmp_path, monkeypatch, managed_dir, managed_yaml):
    """The file may be the only place the backend is set; skipping it would run commands on the host."""
    if managed_yaml is not None:
        _write_managed(managed_dir, managed_yaml)
    home, env_file = _profile(tmp_path)
    _deny_reads(monkeypatch, env_file)

    with pytest.raises(PermissionError):
        load_hermes_dotenv(hermes_home=home)


def test_project_env_does_not_replace_launcher_keys(tmp_path, monkeypatch, pinned):
    home, env_file = _profile(tmp_path)
    project_env = tmp_path / "project.env"
    project_env.write_text("OPENAI_API_KEY=from-project\nLAB_PROJECT_ONLY=from-project\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "from-launcher")
    monkeypatch.setenv("LAB_PROJECT_ONLY", "unset at teardown")
    monkeypatch.delenv("LAB_PROJECT_ONLY")
    _deny_reads(monkeypatch, env_file)

    assert load_hermes_dotenv(hermes_home=home, project_env=project_env) == [project_env]

    assert os.environ["OPENAI_API_KEY"] == "from-launcher"
    assert os.environ["LAB_PROJECT_ONLY"] == "from-project"


def test_reload_keeps_known_keys_when_user_env_is_unreadable(monkeypatch):
    from hermes_cli.config import get_env_path, reload_env

    env_file = get_env_path()
    env_file.write_text("OPENAI_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "from-launcher")
    _deny_reads(monkeypatch, env_file)

    assert reload_env() == 0
    assert os.environ["OPENAI_API_KEY"] == "from-launcher"


def test_reload_still_removes_keys_a_readable_env_dropped(monkeypatch):
    from hermes_cli.config import get_env_path, reload_env

    get_env_path().write_text("ANTHROPIC_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "stale")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "stale")

    assert reload_env() == 2
    assert "OPENAI_API_KEY" not in os.environ
    assert os.environ["ANTHROPIC_API_KEY"] == "from-file"
