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


@pytest.mark.parametrize("temp_var", ("TEMP", "TMP"))
def test_prompt_names_native_temp_when_tmpdir_is_unset(tmp_path, monkeypatch, temp_var):
    """An OS-supplied TEMP/TMP wins over boot scratch, even without TMPDIR."""
    import tempfile
    from pathlib import Path

    from agent.prompt_builder import _local_host_hints

    native_temp = tmp_path / "native temp"
    native_temp.mkdir()
    for name in ("TMPDIR", "TEMP", "TMP", "HERMES_SCRATCH_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(temp_var, str(native_temp))
    monkeypatch.setattr(tempfile, "tempdir", None)

    token = set_terminal_scope({})
    try:
        assert apply_scratch_tmp_env({temp_var: str(native_temp)}) is False
        assert Path(tempfile.gettempdir()).resolve() == native_temp.resolve()
        scratch = get_scratch_dir()
        hints = "\n".join(_local_host_hints())
    finally:
        reset_terminal_scope(token)

    assert f"Temporary directory: {native_temp}" in hints
    assert f"Scratch directory: {scratch}" not in hints
    assert "24h" not in hints


def test_explicit_terminal_temp_can_select_launch_scratch(tmp_path, monkeypatch):
    """An explicit scoped override wins even when it equals the boot marker."""
    from agent.prompt_builder import _local_temp_hint
    from tools.environments.local import LocalEnvironment

    launch = tmp_path / "launch" / "cache" / "scratch"
    launch.mkdir(parents=True)
    for name in ("TMPDIR", "TEMP", "TMP", "HERMES_SCRATCH_DIR"):
        monkeypatch.setenv(name, str(launch))
    home_token = set_hermes_home_override(tmp_path / "served")
    policy = {"TERMINAL_TEMP_DIR": str(launch)}
    scope_token = set_terminal_scope(policy)
    try:
        terminal_temp = LocalEnvironment.get_temp_dir(SimpleNamespace(env=policy))
        hint = _local_temp_hint()
    finally:
        reset_terminal_scope(scope_token)
        reset_hermes_home_override(home_token)

    assert terminal_temp == launch.as_posix()
    assert f"Temporary directory: {launch}" in hint
    assert "24h" not in hint


@pytest.mark.parametrize("winner", ("TMPDIR", "TEMP", "TMP"))
def test_inherited_temp_precedence_matches_python(tmp_path, monkeypatch, winner):
    import tempfile
    from pathlib import Path

    from agent.prompt_builder import _local_temp_hint

    names = ("TMPDIR", "TEMP", "TMP")
    monkeypatch.delenv("HERMES_SCRATCH_DIR", raising=False)
    expected = None
    for index, name in enumerate(names):
        path = tmp_path / name
        if index >= names.index(winner):
            path.mkdir()
        monkeypatch.setenv(name, str(path))
        if name == winner:
            expected = path
    monkeypatch.setattr(tempfile, "tempdir", None)
    token = set_terminal_scope({})
    try:
        assert Path(tempfile.gettempdir()).resolve() == expected.resolve()
        hint = _local_temp_hint()
    finally:
        reset_terminal_scope(token)
    assert f"Temporary directory: {expected}" in hint
    assert "24h" not in hint
