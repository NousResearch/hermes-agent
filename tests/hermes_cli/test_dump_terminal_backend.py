"""`hermes debug` must report the EFFECTIVE terminal backend.

``terminal.backend`` in config.yaml is bridged to the ``TERMINAL_ENV`` env var,
but a ``TERMINAL_ENV`` set in .env / the shell overrides config and is what
``terminal_tool`` actually uses.  The dump used to print only the config value,
which hid the override and made users believe the agent was running ``local``
while it was really jailed in a docker/podman sandbox (and vice-versa).
"""

from pathlib import Path
from types import SimpleNamespace


def _terminal_line(out: str) -> str:
    for line in out.splitlines():
        if line.startswith("terminal:"):
            return line
    raise AssertionError(f"no 'terminal:' line in dump output:\n{out}")


def _seed(home: Path, *, config_yaml: str, env_text: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(config_yaml)
    (home / ".env").write_text(env_text)


def test_dump_surfaces_terminal_env_override(monkeypatch, capsys, tmp_path):
    """A TERMINAL_ENV override that survives dotenv loading is surfaced.

    Since the #29186 fix, config.yaml's explicit terminal keys are re-applied
    after every .env load, so a stale .env value can no longer produce this
    mismatch. The dump's override line remains as defense-in-depth for env
    values set AFTER load (in-process mutation, config-bridge failure), so
    pin it with a config.yaml that has no explicit backend key.
    """
    from hermes_cli import dump
    from hermes_cli.config import get_hermes_home

    monkeypatch.setenv("TERMINAL_ENV", "docker")
    # Keep run_dump's project-.env fallback from touching the real repo.
    monkeypatch.setattr(dump, "get_project_root", lambda: tmp_path / "noproject")

    home = get_hermes_home()
    # No explicit terminal.backend in config.yaml — merged default is 'local',
    # the env override is what actually runs.
    _seed(home, config_yaml="display:\n  streaming: true\n", env_text="")

    dump.run_dump(SimpleNamespace(show_keys=False))

    line = _terminal_line(capsys.readouterr().out)
    # Effective backend (docker) is what actually runs, not the config 'local'.
    assert "docker" in line
    assert "overrides config.yaml" in line
    # The shadowed config value is still shown so the mismatch is obvious.
    assert "terminal.backend=local" in line


def test_dump_reports_config_backend_winning_over_stale_env(monkeypatch, capsys, tmp_path):
    """Regression for #29186: a stale TERMINAL_ENV=docker in .env no longer
    beats config.yaml terminal.backend=local — load_hermes_dotenv re-applies
    the explicit config keys, so the dump reports local with no override
    warning."""
    from hermes_cli import dump
    from hermes_cli.config import get_hermes_home

    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.setattr(dump, "get_project_root", lambda: tmp_path / "noproject")

    home = get_hermes_home()
    _seed(home, config_yaml="terminal:\n  backend: local\n", env_text="TERMINAL_ENV=docker\n")

    dump.run_dump(SimpleNamespace(show_keys=False))

    line = _terminal_line(capsys.readouterr().out)
    assert "local" in line
    assert "overrides" not in line


def test_dump_reports_config_backend_when_no_override(monkeypatch, capsys, tmp_path):
    from hermes_cli import dump
    from hermes_cli.config import get_hermes_home

    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.setattr(dump, "get_project_root", lambda: tmp_path / "noproject")

    home = get_hermes_home()
    _seed(home, config_yaml="terminal:\n  backend: docker\n", env_text="")

    dump.run_dump(SimpleNamespace(show_keys=False))

    line = _terminal_line(capsys.readouterr().out)
    assert "docker" in line
    assert "overrides" not in line


