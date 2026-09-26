"""Update ownership resolution for generated dependency environments."""

from pathlib import Path

from hermes_cli import update_owning_install


def test_lease_managed_workspace_retargets_to_recorded_git_install(tmp_path, monkeypatch):
    """A generated workspace must update the source checkout that owns its environment."""
    owner = tmp_path / "source-install"
    (owner / ".git").mkdir(parents=True)
    (owner / "hermes_cli").mkdir()
    (owner / "hermes_cli" / "main.py").write_text("", encoding="utf-8")

    state = tmp_path / "hermes-home" / "installs" / "install-key"
    generation = state / "environments" / "generation"
    venv = generation / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = fixture\n", encoding="utf-8")
    (generation / ".lease-managed").touch()
    inputs = state / "inputs"
    inputs.mkdir()
    (inputs / ".project-root").write_text(f"{owner}\n", encoding="utf-8")

    workspace = generation / "workspace"
    (workspace / "hermes_cli").mkdir(parents=True)
    (workspace / "hermes_cli" / "main.py").write_text("", encoding="utf-8")

    monkeypatch.setattr(update_owning_install.sys, "prefix", str(venv))
    monkeypatch.setattr(update_owning_install.sys, "base_prefix", str(tmp_path / "base-python"))
    monkeypatch.setenv("PYTHONPATH", str(workspace))

    assert update_owning_install.owning_install_root(workspace) == owner
