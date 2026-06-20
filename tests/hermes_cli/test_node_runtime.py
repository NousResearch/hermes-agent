import os
from pathlib import Path


def _write_executable(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _fake_node_at(path: Path, version: str) -> Path:
    return _write_executable(path, f"#!/bin/sh\necho {version}\n")


def _fake_private_hermes_node(hermes_home: Path, version: str) -> Path:
    node_bin = hermes_home / "node" / "bin"
    _fake_node_at(node_bin / "node", version)
    _write_executable(node_bin / "npm", "#!/bin/sh\necho npm\n")
    return node_bin


def test_node_satisfies_hermes_floor_matches_installer_floor(tmp_path):
    from hermes_cli.node_runtime import node_satisfies_hermes_floor

    assert node_satisfies_hermes_floor(_fake_node_at(tmp_path / "n2018", "v20.18.9")) is False
    assert node_satisfies_hermes_floor(_fake_node_at(tmp_path / "n2019", "v20.19.0")) is True
    assert node_satisfies_hermes_floor(_fake_node_at(tmp_path / "n2211", "v22.11.0")) is False
    assert node_satisfies_hermes_floor(_fake_node_at(tmp_path / "n2212", "v22.12.0")) is True
    assert node_satisfies_hermes_floor(_fake_node_at(tmp_path / "n24", "v24.0.0")) is True


def test_node_satisfies_hermes_floor_rejects_bad_version_output(tmp_path):
    from hermes_cli.node_runtime import node_satisfies_hermes_floor

    assert node_satisfies_hermes_floor(_fake_node_at(tmp_path / "node", "not-a-version")) is False


def test_augment_path_with_hermes_node_uses_private_node_when_path_node_missing(
    tmp_path, monkeypatch
):
    from hermes_cli.node_runtime import augment_path_with_hermes_node

    private_bin = _fake_private_hermes_node(tmp_path, "v22.12.0")
    empty_path = tmp_path / "empty"
    empty_path.mkdir()
    monkeypatch.setenv("PATH", str(empty_path))
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)

    assert augment_path_with_hermes_node() is True
    assert os.environ["PATH"].split(os.pathsep)[0] == str(private_bin)


def test_augment_path_with_hermes_node_uses_private_node_when_path_node_too_old(
    tmp_path, monkeypatch
):
    from hermes_cli.node_runtime import augment_path_with_hermes_node

    user_bin = tmp_path / "user" / "bin"
    _fake_node_at(user_bin / "node", "v20.18.9")
    private_bin = _fake_private_hermes_node(tmp_path, "v22.12.0")
    monkeypatch.setenv("PATH", str(user_bin))
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)

    assert augment_path_with_hermes_node() is True
    assert os.environ["PATH"].split(os.pathsep)[0] == str(private_bin)


def test_augment_path_with_hermes_node_does_not_shadow_modern_node_when_npm_missing(
    tmp_path, monkeypatch
):
    from hermes_cli.node_runtime import augment_path_with_hermes_node

    user_bin = tmp_path / "user" / "bin"
    _fake_node_at(user_bin / "node", "v24.0.0")
    _fake_private_hermes_node(tmp_path, "v22.12.0")
    monkeypatch.setenv("PATH", str(user_bin))
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)

    assert augment_path_with_hermes_node() is False
    assert os.environ["PATH"].split(os.pathsep)[0] == str(user_bin)


def test_remove_legacy_node_symlinks_only_removes_current_hermes_home(
    tmp_path, monkeypatch
):
    from hermes_cli.node_runtime import remove_legacy_node_symlinks

    home = tmp_path / "home"
    hermes_home = home / ".hermes"
    local_bin = home / ".local" / "bin"
    local_bin.mkdir(parents=True)
    node_bin = _fake_private_hermes_node(hermes_home, "v22.12.0")

    for name in ("node", "npm", "npx"):
        (local_bin / name).symlink_to(node_bin / name)

    nvm_bin = home / ".nvm" / "versions" / "node" / "v24.0.0" / "bin"
    nvm_node = _fake_node_at(nvm_bin / "node", "v24.0.0")
    (local_bin / "user-node").symlink_to(nvm_node)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    removed = remove_legacy_node_symlinks(hermes_home)

    assert sorted(path.name for path in removed) == ["node", "npm", "npx"]
    assert (local_bin / "user-node").is_symlink()
    for name in ("node", "npm", "npx"):
        assert not (local_bin / name).is_symlink()
