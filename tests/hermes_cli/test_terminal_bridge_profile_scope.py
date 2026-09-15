"""Regression: the terminal config→env re-bridge is scoped to the *true*
process profile, never a per-task ``set_hermes_home_override``.

The multiplex dashboard serves every profile on the host from one process,
so ``os.environ`` is shared across all of them. A per-turn handler scoped to
a secondary profile (via ``set_hermes_home_override``) must NOT cause that
profile's ``terminal.backend`` to be bridged into the shared environment —
otherwise a ``local`` sibling profile silently clobbers the launch profile's
``TERMINAL_ENV=ssh`` (leaving its ``TERMINAL_SSH_*`` intact), and the launch
session runs every command locally instead of over SSH.

The guard in ``env_loader._reapply_terminal_config_bridge`` compares the
loaded home against the process home. It must use the override-immune
``get_process_hermes_home()`` so the comparison reflects the launch scope,
not whichever profile the current task is scoped to.
"""

import os

import pytest

import hermes_cli.env_loader as env_loader
from hermes_constants import (
    set_hermes_home_override,
    reset_hermes_home_override,
)


def _write_terminal_config(home, text: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(text)


@pytest.fixture(autouse=True)
def _clean_terminal_env(monkeypatch):
    for name in ("TERMINAL_ENV", "TERMINAL_SSH_HOST", "TERMINAL_SSH_USER"):
        monkeypatch.delenv(name, raising=False)
    yield


def test_process_hermes_home_ignores_task_override(tmp_path, monkeypatch):
    """The guard's home resolver must not follow a per-task override."""
    launch_home = tmp_path / "laptop"
    other_home = tmp_path / "tommy"
    launch_home.mkdir()
    other_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch_home))

    token = set_hermes_home_override(str(other_home))
    try:
        assert (
            env_loader._process_hermes_home().resolve() == launch_home.resolve()
        ), "process home leaked to the per-task override profile"
    finally:
        reset_hermes_home_override(token)


def test_secondary_profile_reload_does_not_bridge_into_shared_env(
    tmp_path, monkeypatch
):
    """A secondary profile's terminal.backend must not touch os.environ.

    Simulates the dashboard multiplex: process launched under ``laptop``
    (ssh), a per-turn handler scoped to ``tommy`` (local) triggers a dotenv
    reload for tommy's home. The launch session's TERMINAL_ENV must survive.
    """
    launch_home = tmp_path / "laptop"
    other_home = tmp_path / "tommy"
    _write_terminal_config(
        launch_home,
        "terminal:\n"
        "  backend: ssh\n"
        "  ssh_host: 10.10.0.103\n"
        "  ssh_user: bergmann\n",
    )
    _write_terminal_config(other_home, "terminal:\n  backend: local\n")
    monkeypatch.setenv("HERMES_HOME", str(launch_home))

    # Launch profile's backend is what the shared env carries.
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_SSH_HOST", "10.10.0.103")

    # A per-turn handler for the secondary profile is scoped via the contextvar
    # and drives a reload for tommy's home.
    token = set_hermes_home_override(str(other_home))
    try:
        env_loader._reapply_terminal_config_bridge(other_home)
    finally:
        reset_hermes_home_override(token)

    # tommy's `local` must NOT have leaked into the shared process env.
    assert os.environ["TERMINAL_ENV"] == "ssh"
    assert os.environ["TERMINAL_SSH_HOST"] == "10.10.0.103"
