"""Local prompt temp paths follow explicit overrides (regression for #147309)."""

from types import SimpleNamespace

import pytest

from hermes_constants import (
    apply_scratch_tmp_env,
    get_scratch_dir,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from tools.terminal_scope import reset_terminal_scope, set_terminal_scope


@pytest.mark.parametrize("configured", [False, True])
def test_prompt_names_the_custom_terminal_temp_directory(tmp_path, monkeypatch, configured):
    from agent.prompt_builder import _local_host_hints
    from tools.environments.local import LocalEnvironment

    custom = tmp_path / "custom temp"
    custom.mkdir()
    ambient = tmp_path / "ambient"
    ambient.mkdir()
    monkeypatch.setenv("TMPDIR", str(ambient if configured else custom))
    monkeypatch.delenv("HERMES_SCRATCH_DIR", raising=False)
    policy = {"TERMINAL_TEMP_DIR": str(custom)} if configured else {}
    token = set_terminal_scope(policy)
    try:
        child_env = dict(policy, TMPDIR=str(ambient if configured else custom))
        assert apply_scratch_tmp_env(child_env) is False
        terminal_temp = LocalEnvironment.get_temp_dir(SimpleNamespace(env=child_env))
        hints = "\n".join(_local_host_hints())
    finally:
        reset_terminal_scope(token)

    assert terminal_temp == custom.as_posix()
    assert str(custom) in hints
    assert str(get_scratch_dir()) not in hints
    assert "24h" not in hints
    assert "Hermes points at its own scratch dir" not in hints


def test_prompt_rederives_managed_scratch_for_the_served_home(tmp_path, monkeypatch):
    from agent.prompt_builder import _local_host_hints

    launch = tmp_path / "launch" / "cache" / "scratch"
    launch.mkdir(parents=True)
    monkeypatch.setenv("TMPDIR", str(launch))
    monkeypatch.setenv("HERMES_SCRATCH_DIR", str(launch))
    home_token = set_hermes_home_override(tmp_path / "served")
    scope_token = set_terminal_scope({})
    try:
        scratch = get_scratch_dir()
        hints = "\n".join(_local_host_hints())
    finally:
        reset_terminal_scope(scope_token)
        reset_hermes_home_override(home_token)

    assert str(scratch) in hints
    assert str(launch) not in hints
    assert "24h" in hints
