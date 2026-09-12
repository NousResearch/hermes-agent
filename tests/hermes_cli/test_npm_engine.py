"""The frozen npm retry import stops old callers without changing dependencies."""

import pytest

from hermes_cli.npm_engine import maybe_repair_npm_engine


@pytest.mark.parametrize("quiet,output", [(True, "EBADENGINE"), (False, "unrelated failure")])
def test_retired_retry_stops_old_updater_without_provisioning(tmp_path, monkeypatch, capsys, quiet, output):
    import pm

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(home / "tools"))
    calls = []

    def forbidden_ensure(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("historical updater must not install")

    monkeypatch.setattr(pm, "ensure", forbidden_ensure)
    with pytest.raises(SystemExit) as stopped:
        maybe_repair_npm_engine("/caller-owned/npm", output, quiet=quiet)
    assert stopped.value.code == 0
    assert "run `hermes` again" in capsys.readouterr().err
    assert calls == []
    assert list(home.iterdir()) == []
