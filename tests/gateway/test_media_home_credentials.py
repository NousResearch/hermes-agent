"""HOME credential-file screening for native media delivery (#99182)."""
import os
from pathlib import Path

import pytest

from gateway.platforms import base


@pytest.fixture(autouse=True)
def isolate_account_home(tmp_path, monkeypatch):
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "_account_home", lambda: tmp_path / "account-home")


@pytest.mark.parametrize("root_name", ["home", "real_home", "profile_home"])
@pytest.mark.parametrize("name", [".netrc", ".pgpass", ".npmrc", ".pypirc", ".git-credentials"])
def test_credential_identity_does_not_inherit_cache_trust(tmp_path, monkeypatch, root_name, name):
    home = tmp_path / "home"
    real_home = tmp_path / "real"
    profile = tmp_path / "profile"
    profile_home = profile / "home"
    for root in (home, real_home, profile_home):
        root.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setattr(base, "get_real_home", lambda: str(real_home))
    monkeypatch.setattr(base, "_credential_home_roots", lambda: [profile])
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(base, "MEDIA_DELIVERY_SAFE_ROOTS", (cache,))
    secret = {"home": home, "real_home": real_home, "profile_home": profile_home}[root_name] / name
    secret.write_bytes(b"placeholder")
    alias = cache / "report.txt"
    os.link(secret, alias)
    ordinary = cache / "ordinary.txt"
    ordinary.write_bytes(b"artifact")
    symlink = cache / "linked.txt"
    symlink.symlink_to(secret)
    assert base.BasePlatformAdapter.filter_local_delivery_paths(
        [str(secret), str(alias), str(symlink), str(ordinary)]) == [str(ordinary.resolve())]
    # No stale inode cache: an old hardlink becomes ordinary after the store rotates.
    secret.unlink()
    secret.write_bytes(b"replacement")
    assert base.validate_media_delivery_path(str(alias)) == str(alias.resolve())


def test_routed_profiles_preserve_generated_and_operator_trust(tmp_path, monkeypatch):
    from gateway import media_policy
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home = tmp_path / "home"
    root = home / ".hermes"
    exports = home / "exports"
    exports.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(base, "_HERMES_ROOT", root)
    monkeypatch.setattr(base, "MEDIA_DELIVERY_SAFE_ROOTS", ())
    for env in ("HERMES_MEDIA_DELIVERY_STRICT", "HERMES_MEDIA_TRUST_RECENT_FILES", "HERMES_MEDIA_ALLOW_DIRS"):
        monkeypatch.delenv(env, raising=False)
    profiles = [root / "profiles" / name for name in ("alpha", "beta")]
    secret = home / ".netrc"
    secret.write_bytes(b"placeholder")
    exported = exports / "report.txt"
    exported.write_bytes(b"artifact")
    for profile in profiles:
        generated = profile / "cache" / "generated" / "images"
        generated.mkdir(parents=True)
        (profile / "config.yaml").write_text(
            "gateway:\n  strict: true\n  trust_recent_files: false\n"
            f"  media_delivery_allow_dirs:\n    - {exports}\n")
        (generated / "report.png").write_bytes(b"image")
        os.link(secret, generated / "alias.png")
    for profile in (profiles[0], profiles[1], profiles[0]):
        token = set_hermes_home_override(profile)
        try:
            media_policy.apply_media_policy_env()
            generated = profile / "cache" / "generated" / "images"
            assert base.BasePlatformAdapter.filter_local_delivery_paths([
                str(generated / "alias.png"), str(generated / "report.png"), str(exported)
            ]) == [str((generated / "report.png").resolve()), str(exported.resolve())]
        finally:
            reset_hermes_home_override(token)


@pytest.mark.parametrize("mount_kind", ["persistent", "directory", "file"])
def test_docker_home_credential_identity_is_not_delivered(tmp_path, monkeypatch, mount_kind):
    from gateway.platforms import docker_media_paths

    home = tmp_path / "container-home"
    workspace = tmp_path / "workspace"
    home.mkdir()
    workspace.mkdir()
    secret = home / ".netrc"
    secret.write_bytes(b"placeholder")
    alias = workspace / "report.txt"
    os.link(secret, alias)
    ordinary = workspace / "ordinary.txt"
    ordinary.write_bytes(b"artifact")
    home_artifact = home / "report.txt"
    home_artifact.write_bytes(b"artifact")
    mounts = [(workspace, Path("/workspace"))]
    if mount_kind == "directory":
        mounts.append((home, Path("/root")))
    elif mount_kind == "file":
        mounts.append((secret, Path("/root/.netrc")))
    monkeypatch.setattr(base, "_parse_docker_volume_mounts", lambda: mounts)
    monkeypatch.setattr(base, "_cache_dir_container_mounts", list)
    monkeypatch.setattr(docker_media_paths, "_docker_persistent_sandbox_roots",
                        lambda session, leaf, workspace_key="": [home] if mount_kind == "persistent" and leaf == "home" else [])
    monkeypatch.setattr(base, "MEDIA_DELIVERY_SAFE_ROOTS", (home, workspace))
    assert base.validate_media_delivery_path("/root/.netrc", session_key="session-1") is None
    assert base.validate_media_delivery_path("/workspace/report.txt", session_key="session-1") is None
    assert base.validate_media_delivery_path("/workspace/ordinary.txt", session_key="session-1") == str(ordinary)
    if mount_kind != "file":
        assert base.validate_media_delivery_path("/root/report.txt", session_key="session-1") == str(home_artifact)


@pytest.mark.parametrize("raw", ["", '""', "``"])
def test_empty_media_paths_do_not_raise_during_fallback(raw):
    assert base.BasePlatformAdapter.filter_media_delivery_paths([(raw, False)]) == []


@pytest.mark.parametrize("mount_kind", ["persistent", "directory"])
def test_docker_home_credential_symlink_outside_mount_is_screened(tmp_path, monkeypatch, mount_kind):
    from gateway.platforms import docker_media_paths

    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    home.mkdir()
    workspace.mkdir()
    secret = workspace / "secret.txt"
    secret.write_bytes(b"placeholder")
    (home / ".netrc").symlink_to("../workspace/secret.txt")
    mounts = [(workspace, Path("/workspace"))]
    if mount_kind == "directory":
        mounts.append((home, Path("/root")))
    monkeypatch.setattr(base, "_parse_docker_volume_mounts", lambda: mounts)
    monkeypatch.setattr(base, "_cache_dir_container_mounts", list)
    monkeypatch.setattr(docker_media_paths, "_docker_persistent_sandbox_roots",
                        lambda session, leaf, workspace_key="": [home] if mount_kind == "persistent" and leaf == "home" else [])
    monkeypatch.setattr(base, "MEDIA_DELIVERY_SAFE_ROOTS", (workspace,))
    assert base._translate_docker_container_media_path(Path("/root/.netrc"), session_key="s", quiet=True) is None
    assert base.validate_media_delivery_path("/workspace/secret.txt", session_key="s") is None
    (home / ".netrc").unlink()
    (home / ".netrc").write_bytes(b"replacement")
    assert base.validate_media_delivery_path("/workspace/secret.txt", session_key="s") == str(secret)


def test_docker_credential_file_overlay_wins_over_home_mount(tmp_path, monkeypatch):
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    home.mkdir()
    workspace.mkdir()
    hidden = workspace / "hidden.txt"
    hidden.write_bytes(b"artifact")
    (home / ".netrc").symlink_to(hidden)
    overlay = tmp_path / "mounted-file"
    overlay.write_bytes(b"placeholder")
    alias = workspace / "alias.txt"
    os.link(overlay, alias)
    monkeypatch.setattr(base, "_parse_docker_volume_mounts", lambda: [
        (home, Path("/root")), (workspace, Path("/workspace")), (overlay, Path("/root/.netrc"))])
    monkeypatch.setattr(base, "_cache_dir_container_mounts", list)
    monkeypatch.setattr(base, "MEDIA_DELIVERY_SAFE_ROOTS", (workspace,))
    assert base.validate_media_delivery_path("/workspace/alias.txt", session_key="s") is None
    assert base.validate_media_delivery_path("/workspace/hidden.txt", session_key="s") == str(hidden)


def test_distinct_account_home_credential_identity_is_screened(tmp_path, monkeypatch):
    import hermes_constants

    account = tmp_path / "account"
    home = tmp_path / "environment"
    cache = tmp_path / "cache"
    for directory in (account, home, cache):
        directory.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
    monkeypatch.setattr(hermes_constants, "_account_home", lambda: account)
    monkeypatch.setattr(base, "get_real_home", lambda: str(home))
    monkeypatch.setattr(base, "_credential_home_roots", list)
    monkeypatch.setattr(base, "MEDIA_DELIVERY_SAFE_ROOTS", (cache,))
    secret = account / ".netrc"
    secret.write_bytes(b"placeholder")
    alias = cache / "artifact.txt"
    os.link(secret, alias)
    assert base.validate_media_delivery_path(str(alias)) is None
