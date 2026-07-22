"""Behavioral coverage for recursive uninstall root safety."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import uninstall


def _stub_external_cleanup(monkeypatch, tmp_path: Path) -> None:
    """Keep run_uninstall real while isolating machine-level cleanup effects."""
    monkeypatch.setattr(uninstall, "uninstall_gateway_service", lambda: False)
    monkeypatch.setattr(uninstall, "remove_path_from_shell_configs", lambda: [])
    monkeypatch.setattr(uninstall, "remove_wrapper_script", lambda: [])
    monkeypatch.setattr(uninstall, "remove_node_symlinks", lambda _home: [])
    monkeypatch.setattr(uninstall, "_discover_named_profiles", lambda: [])
    monkeypatch.setattr(uninstall, "_is_windows", lambda: False)

    from hermes_cli import gui_uninstall

    monkeypatch.setattr(gui_uninstall, "uninstall_gui", lambda _home: [])
    monkeypatch.setattr(gui_uninstall, "packaged_gui_app_paths", lambda: [])
    monkeypatch.setattr(gui_uninstall, "desktop_userdata_dir", lambda: tmp_path / "none")
    monkeypatch.setattr(
        "builtins.input",
        lambda *_args, **_kwargs: pytest.fail("--yes uninstall prompted"),
    )


@pytest.mark.parametrize("package_dir_name", ["site-packages", "dist-packages"])
def test_run_uninstall_refuses_shared_python_package_directory(
    tmp_path,
    monkeypatch,
    capsys,
    package_dir_name,
):
    """A wheel-shaped install must preserve Hermes and every sibling package."""
    package_root = tmp_path / "venv" / "lib" / "python3.13" / package_dir_name
    (package_root / "hermes_cli").mkdir(parents=True)
    (package_root / "hermes_cli" / "uninstall.py").write_text("# installed wheel\n")
    (package_root / "hermes_agent-1.0.dist-info").mkdir()

    sentinel_package = package_root / "sentinel_package"
    sentinel_package.mkdir()
    sentinel_file = sentinel_package / "__init__.py"
    sentinel_file.write_text("SENTINEL = True\n")
    (package_root / "sentinel_package-1.0.dist-info").mkdir()

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    config = hermes_home / "config.yaml"
    config.write_text("model: {}\n")

    monkeypatch.setattr(uninstall, "get_project_root", lambda: package_root)
    monkeypatch.setattr(uninstall, "get_hermes_home", lambda: hermes_home)
    _stub_external_cleanup(monkeypatch, tmp_path)

    uninstall.run_uninstall(SimpleNamespace(yes=True, full=False, dry_run=False))

    assert sentinel_file.read_text() == "SENTINEL = True\n"
    assert (package_root / "hermes_cli" / "uninstall.py").exists()
    assert config.exists()
    assert "Refusing to recursively remove" in capsys.readouterr().out


def test_run_uninstall_refuses_nonstandard_root_with_unrelated_distribution(
    tmp_path,
    monkeypatch,
    capsys,
):
    """Distribution metadata also protects shared roots with nonstandard names."""
    shared_root = tmp_path / "python-libs"
    (shared_root / "hermes_cli").mkdir(parents=True)
    (shared_root / "hermes_cli" / "uninstall.py").write_text("# installed\n")
    (shared_root / "pyproject.toml").write_text(
        '[project]\nname = "hermes-agent"\nversion = "1.0"\n'
    )
    sentinel = shared_root / "other_project-2.0.dist-info"
    sentinel.mkdir()
    (sentinel / "METADATA").write_text("Name: other-project\n")

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setattr(uninstall, "get_project_root", lambda: shared_root)
    monkeypatch.setattr(uninstall, "get_hermes_home", lambda: hermes_home)
    _stub_external_cleanup(monkeypatch, tmp_path)

    uninstall.run_uninstall(SimpleNamespace(yes=True, full=False, dry_run=False))

    assert (sentinel / "METADATA").exists()
    assert (shared_root / "hermes_cli" / "uninstall.py").exists()
    assert "unrelated Python distributions" in capsys.readouterr().out


def test_run_uninstall_still_removes_verified_source_checkout(
    tmp_path,
    monkeypatch,
):
    checkout = tmp_path / "hermes-agent"
    (checkout / "hermes_cli").mkdir(parents=True)
    (checkout / "hermes_cli" / "uninstall.py").write_text("# source checkout\n")
    (checkout / ".git").mkdir()
    (checkout / "pyproject.toml").write_text(
        '[project]\nname = "hermes-agent"\nversion = "1.0"\n'
    )
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()

    monkeypatch.setattr(uninstall, "get_project_root", lambda: checkout)
    monkeypatch.setattr(uninstall, "get_hermes_home", lambda: hermes_home)
    _stub_external_cleanup(monkeypatch, tmp_path)

    uninstall.run_uninstall(SimpleNamespace(yes=True, full=False, dry_run=False))

    assert not checkout.exists()
    assert hermes_home.exists()


def test_run_uninstall_refuses_unrelated_git_repository(
    tmp_path,
    monkeypatch,
    capsys,
):
    """A copied Hermes module must not authorize deleting an unrelated repo."""
    unrelated_repo = tmp_path / "unrelated-monorepo"
    (unrelated_repo / "hermes_cli").mkdir(parents=True)
    (unrelated_repo / "hermes_cli" / "uninstall.py").write_text("# copied file\n")
    (unrelated_repo / ".git").mkdir()
    sentinel = unrelated_repo / "important-project" / "data.txt"
    sentinel.parent.mkdir()
    sentinel.write_text("keep me\n")

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setattr(uninstall, "get_project_root", lambda: unrelated_repo)
    monkeypatch.setattr(uninstall, "get_hermes_home", lambda: hermes_home)
    _stub_external_cleanup(monkeypatch, tmp_path)

    uninstall.run_uninstall(SimpleNamespace(yes=True, full=False, dry_run=False))

    assert sentinel.read_text() == "keep me\n"
    assert "not a verifiable Hermes source checkout" in capsys.readouterr().out


def test_run_uninstall_wheel_full_still_removes_hermes_home(
    tmp_path,
    monkeypatch,
):
    package_root = tmp_path / "venv" / "lib" / "python3.13" / "site-packages"
    (package_root / "hermes_cli").mkdir(parents=True)
    (package_root / "hermes_cli" / "uninstall.py").write_text("# installed wheel\n")
    sentinel = package_root / "sentinel_package" / "__init__.py"
    sentinel.parent.mkdir()
    sentinel.write_text("SENTINEL = True\n")

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("model: {}\n")

    monkeypatch.setattr(uninstall, "get_project_root", lambda: package_root)
    monkeypatch.setattr(uninstall, "get_hermes_home", lambda: hermes_home)
    _stub_external_cleanup(monkeypatch, tmp_path)

    uninstall.run_uninstall(SimpleNamespace(yes=True, full=True, dry_run=False))

    assert sentinel.read_text() == "SENTINEL = True\n"
    assert (package_root / "hermes_cli" / "uninstall.py").exists()
    assert not hermes_home.exists()


def test_root_and_interpreter_prefixes_are_never_safe(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()

    filesystem_root = Path(tmp_path.anchor)
    safe, reason = uninstall._project_root_removal_safety(filesystem_root, hermes_home)
    assert safe is False
    assert "filesystem root" in reason

    safe, reason = uninstall._project_root_removal_safety(Path.home(), hermes_home)
    assert safe is False
    assert "home directory" in reason

    safe, reason = uninstall._project_root_removal_safety(hermes_home, hermes_home)
    assert safe is False
    assert "Hermes data directory" in reason

    interpreter_prefix = tmp_path / "python"
    interpreter_prefix.mkdir()
    monkeypatch.setattr(uninstall.sys, "prefix", str(interpreter_prefix))
    safe, reason = uninstall._project_root_removal_safety(
        interpreter_prefix,
        hermes_home,
    )
    assert safe is False
    assert "interpreter prefix" in reason
