"""`hermes config show` must report the EFFECTIVE terminal backend.

``terminal.backend`` in config.yaml is bridged to the ``TERMINAL_ENV`` env var,
but a ``TERMINAL_ENV`` set in .env / the shell overrides config and is what
``terminal_tool`` actually uses.  ``config show`` used to print only the config
value, which hid the override and made users believe the agent was running
``local`` while it was really jailed in a docker/podman sandbox (and vice-versa).
``hermes dump`` and ``hermes status`` already report the effective value; this
aligns the third command in the trio.
"""

from pathlib import Path


def _backend_line(out: str) -> str:
    for line in out.splitlines():
        if line.strip().startswith("Backend:"):
            return line
    raise AssertionError(f"no 'Backend:' line in config show output:\n{out}")


def _seed(home: Path, *, config_yaml: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(config_yaml)


def test_config_show_surfaces_terminal_env_override(monkeypatch, capsys):
    from hermes_cli import config as config_mod

    monkeypatch.setenv("TERMINAL_ENV", "docker")

    home = config_mod.get_hermes_home()
    _seed(home, config_yaml="terminal:\n  backend: local\n")

    config_mod.show_config()

    line = _backend_line(capsys.readouterr().out)
    # Effective backend (docker) is what actually runs, not the config 'local'.
    assert "docker" in line
    assert "overrides config.yaml" in line
    # The shadowed config value is still shown so the mismatch is obvious.
    assert "terminal.backend=local" in line


def test_config_show_reports_config_backend_when_no_override(monkeypatch, capsys):
    from hermes_cli import config as config_mod

    monkeypatch.delenv("TERMINAL_ENV", raising=False)

    home = config_mod.get_hermes_home()
    _seed(home, config_yaml="terminal:\n  backend: docker\n")

    config_mod.show_config()

    line = _backend_line(capsys.readouterr().out)
    assert "docker" in line
    assert "overrides" not in line


def test_config_show_no_override_when_env_matches_config(monkeypatch, capsys):
    from hermes_cli import config as config_mod

    monkeypatch.setenv("TERMINAL_ENV", "docker")

    home = config_mod.get_hermes_home()
    # TERMINAL_ENV agrees with config — no spurious "override" note.
    _seed(home, config_yaml="terminal:\n  backend: docker\n")

    config_mod.show_config()

    line = _backend_line(capsys.readouterr().out)
    assert "docker" in line
    assert "overrides" not in line
