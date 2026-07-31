from __future__ import annotations

import asyncio
import os
import stat
import tarfile
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import profiles, uninstall
from hermes_constants import (
    HermesAuthHomeError,
    get_hermes_auth_home,
    get_hermes_auth_home_for,
    get_hermes_auth_home_override,
    get_hermes_auth_home_override_strict,
    get_hermes_auth_home_strict,
    is_hermes_auth_home_relocated,
    is_hermes_auth_home_relocated_strict,
    reset_hermes_home_override,
    set_hermes_home_override,
)


@pytest.fixture
def auth_profile_env(tmp_path, monkeypatch):
    operator_home = tmp_path / "operator"
    runtime_home = tmp_path / "runtime"
    auth_home = tmp_path / "auth-residence"
    operator_home.mkdir()
    runtime_home.mkdir()
    auth_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: operator_home)
    monkeypatch.setenv("HOME", str(operator_home))
    monkeypatch.setenv("HERMES_HOME", str(runtime_home))
    monkeypatch.setenv("HERMES_AUTH_HOME", str(auth_home))
    monkeypatch.setenv(
        "HERMES_SHARED_AUTH_DIR",
        str(auth_home / "shared"),
    )
    return runtime_home, auth_home


def _disable_profile_services(monkeypatch):
    monkeypatch.setattr(profiles, "_cleanup_gateway_service", lambda *args: None)
    monkeypatch.setattr(
        profiles,
        "_maybe_unregister_gateway_service",
        lambda *args: None,
    )
    monkeypatch.setattr(profiles, "_stop_profile_backends", lambda *args: None)


def _leave_restore_residue(target: Path) -> Path:
    """Create the same temporary filename used by backup credential restore."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        dir=str(target.parent),
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    return Path(name)


def _leave_atomic_json_residue(target: Path) -> Path:
    """Create the same temporary filename used by utils.atomic_json_write."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        dir=str(target.parent),
        prefix=f".{target.stem}_",
        suffix=".tmp",
    )
    os.close(fd)
    return Path(name)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "relative/auth",
        "~/auth",
        "/tmp/auth\nhome",
    ],
)
def test_strict_resolver_rejects_invalid_raw_values(
    tmp_path,
    monkeypatch,
    value,
):
    runtime_home = tmp_path / "runtime"
    monkeypatch.setenv("HERMES_HOME", str(runtime_home))
    monkeypatch.setenv("HERMES_AUTH_HOME", value)

    with pytest.raises(HermesAuthHomeError):
        get_hermes_auth_home_override_strict()
    with pytest.raises(HermesAuthHomeError):
        get_hermes_auth_home_strict()

    assert get_hermes_auth_home_override() is None
    assert get_hermes_auth_home() == runtime_home
    assert is_hermes_auth_home_relocated() is False


def test_strict_resolver_rejects_nul_before_path_operations(monkeypatch):
    import hermes_constants

    monkeypatch.setattr(
        hermes_constants,
        "_raw_hermes_auth_home",
        lambda: "/tmp/auth\x00home",
    )
    with pytest.raises(HermesAuthHomeError, match="control character"):
        get_hermes_auth_home_override_strict()


def test_strict_resolver_rejects_padding_around_absolute_path(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "runtime"))
    for value in (f" {tmp_path / 'auth'}", f"{tmp_path / 'auth'} "):
        monkeypatch.setenv("HERMES_AUTH_HOME", value)
        with pytest.raises(HermesAuthHomeError, match="whitespace"):
            get_hermes_auth_home_override_strict()


def test_strict_resolver_rejects_file_and_loop_but_accepts_directory_symlink(
    tmp_path,
    monkeypatch,
):
    runtime_home = tmp_path / "runtime"
    monkeypatch.setenv("HERMES_HOME", str(runtime_home))

    file_path = tmp_path / "auth-file"
    file_path.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("HERMES_AUTH_HOME", str(file_path))
    with pytest.raises(HermesAuthHomeError, match="directory"):
        get_hermes_auth_home_override_strict()

    loop_a = tmp_path / "loop-a"
    loop_b = tmp_path / "loop-b"
    loop_a.symlink_to(loop_b, target_is_directory=True)
    loop_b.symlink_to(loop_a, target_is_directory=True)
    monkeypatch.setenv("HERMES_AUTH_HOME", str(loop_a))
    with pytest.raises(HermesAuthHomeError):
        get_hermes_auth_home_override_strict()

    residence = tmp_path / "residence"
    residence.mkdir()
    residence_link = tmp_path / "residence-link"
    residence_link.symlink_to(residence, target_is_directory=True)
    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence_link))
    assert get_hermes_auth_home_override_strict() == residence.resolve()


def test_explicit_mapping_handles_default_named_and_path_equal(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "runtime"
    named_home = root / "profiles" / "work"
    named_home.mkdir(parents=True)
    residence = tmp_path / "residence"
    residence.mkdir()

    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))
    assert get_hermes_auth_home_for(root) == residence.resolve()
    assert get_hermes_auth_home_for(named_home) == (residence / "profiles" / "work")
    assert is_hermes_auth_home_relocated(root)
    assert is_hermes_auth_home_relocated_strict(named_home)

    monkeypatch.setenv("HERMES_AUTH_HOME", str(root))
    assert get_hermes_auth_home_for(root) == root.resolve()
    assert get_hermes_auth_home_for(named_home) == named_home.resolve()
    assert not is_hermes_auth_home_relocated(root)
    assert not is_hermes_auth_home_relocated_strict(named_home)

    monkeypatch.setenv("HERMES_AUTH_HOME", str(named_home))
    assert get_hermes_auth_home_for(named_home) == named_home.resolve()
    assert not is_hermes_auth_home_relocated_strict(named_home)


def test_process_path_equal_is_noop_for_entire_layout_and_request_scope(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "runtime"
    active = root / "profiles" / "work"
    sibling = root / "profiles" / "sibling"
    clone = root / "profiles" / "copy"
    active.mkdir(parents=True)
    sibling.mkdir()
    override_link = tmp_path / "auth-home-link"
    override_link.symlink_to(active, target_is_directory=True)

    monkeypatch.setenv("HERMES_HOME", str(active))
    monkeypatch.setenv("HERMES_AUTH_HOME", str(override_link))

    assert get_hermes_auth_home_for(root) == root
    assert get_hermes_auth_home_for(active) == active
    assert get_hermes_auth_home_for(sibling) == sibling
    assert get_hermes_auth_home_for(clone) == clone

    request_home = root / "profiles" / "request"
    token = set_hermes_home_override(request_home)
    try:
        assert get_hermes_auth_home() == request_home
        assert get_hermes_auth_home_for(root) == root
    finally:
        reset_hermes_home_override(token)


def test_lexical_profile_symlink_keeps_named_residence_identity(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "runtime"
    profiles_root = root / "profiles"
    profiles_root.mkdir(parents=True)
    external_runtime = tmp_path / "external-runtime"
    external_runtime.mkdir()
    profile_link = profiles_root / "work"
    profile_link.symlink_to(external_runtime, target_is_directory=True)
    residence = tmp_path / "residence"
    residence.mkdir()

    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))

    assert get_hermes_auth_home_for(profile_link) == (
        residence.resolve() / "profiles" / "work"
    )

    sibling = profiles_root / "sibling"
    monkeypatch.setenv("HERMES_HOME", str(profile_link))
    monkeypatch.setenv("HERMES_AUTH_HOME", str(external_runtime))
    assert get_hermes_auth_home_for(root) == root
    assert get_hermes_auth_home_for(profile_link) == profile_link
    assert get_hermes_auth_home_for(sibling) == sibling


def test_path_equal_clone_sources_and_full_backup_keep_flat_layout(
    tmp_path,
    monkeypatch,
):
    from hermes_cli.backup import _write_full_zip_backup

    operator = tmp_path / "operator"
    root = operator / ".hermes"
    active = root / "profiles" / "work"
    active.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: operator)
    monkeypatch.setenv("HOME", str(operator))
    monkeypatch.setenv("HERMES_HOME", str(active))
    override_link = tmp_path / "path-equal-auth"
    override_link.symlink_to(active, target_is_directory=True)
    monkeypatch.setenv("HERMES_AUTH_HOME", str(override_link))
    monkeypatch.delenv("HERMES_SHARED_AUTH_DIR", raising=False)

    (root / "config.yaml").write_text("model: default\n", encoding="utf-8")
    (root / "auth.json").write_text('{"store":"default"}\n', encoding="utf-8")
    (active / "config.yaml").write_text("model: work\n", encoding="utf-8")
    (active / "auth.json").write_text('{"store":"work"}\n', encoding="utf-8")

    default_clone = profiles.create_profile(
        "default-copy",
        clone_from="default",
        clone_all=True,
        no_alias=True,
    )
    active_clone = profiles.create_profile(
        "work-copy",
        clone_from="work",
        clone_all=True,
        no_alias=True,
    )
    assert (default_clone / "auth.json").read_text(encoding="utf-8") == (
        '{"store":"default"}\n'
    )
    assert (active_clone / "auth.json").read_text(encoding="utf-8") == (
        '{"store":"work"}\n'
    )
    assert not (active / "profiles").exists()

    archive = tmp_path / "full.zip"
    assert _write_full_zip_backup(archive, root) == archive
    with zipfile.ZipFile(archive) as backup:
        assert backup.read("auth.json") == b'{"store":"default"}\n'
        assert backup.read("profiles/work/auth.json") == b'{"store":"work"}\n'
        assert not any(
            name.startswith("_auth-residence/") for name in backup.namelist()
        )


def test_clone_all_securely_copies_only_profile_auth_stores(
    auth_profile_env,
):
    _, auth_home = auth_profile_env
    source = profiles.create_profile("work", no_alias=True)
    (source / "config.yaml").write_text("model: test\n", encoding="utf-8")
    source_auth = auth_home / "profiles" / "work"
    source_auth.mkdir(parents=True)
    (source_auth / "auth.json").write_text('{"secret": "primary"}\n')
    (source_auth / ".anthropic_oauth.json").write_text('{"secret": "anthropic"}\n')
    (source_auth / "auth.json.corrupt").write_text("stale")
    os.chmod(source_auth / "auth.json", 0o644)
    os.chmod(source_auth / ".anthropic_oauth.json", 0o644)

    target = profiles.create_profile(
        "copy",
        clone_from="work",
        clone_all=True,
        no_alias=True,
    )
    target_auth = auth_home / "profiles" / "copy"

    assert (target / "config.yaml").read_text() == "model: test\n"
    assert (target_auth / "auth.json").read_text() == '{"secret": "primary"}\n'
    assert (target_auth / ".anthropic_oauth.json").read_text() == (
        '{"secret": "anthropic"}\n'
    )
    assert stat.S_IMODE((target_auth / "auth.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((target_auth / ".anthropic_oauth.json").stat().st_mode) == 0o600
    assert not (target_auth / "auth.json.corrupt").exists()

    profiles.create_profile(
        "selective",
        clone_from="work",
        clone_config=True,
        no_alias=True,
    )
    assert not (auth_home / "profiles" / "selective").exists()


def test_clone_all_excludes_transient_and_shared_auth_without_override(
    tmp_path,
    monkeypatch,
):
    operator_home = tmp_path / "operator"
    runtime_home = tmp_path / "runtime"
    operator_home.mkdir()
    runtime_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: operator_home)
    monkeypatch.setenv("HOME", str(operator_home))
    monkeypatch.setenv("HERMES_HOME", str(runtime_home))
    monkeypatch.delenv("HERMES_AUTH_HOME", raising=False)

    source = profiles.create_profile("work", no_alias=True)
    (source / "auth.json").write_text('{"secret": "primary"}\n')
    (source / ".anthropic_oauth.json").write_text('{"secret": "anthropic"}\n')
    (source / ".anthropic_oauth.json.tmp.crash").write_text("transient")
    (source / ".anthropic_oauth_deadbeef.tmp").write_text("transient")
    shared = source / "shared"
    shared.mkdir()
    (shared / "nous_auth.json").write_text("shared credential")

    target = profiles.create_profile(
        "copy",
        clone_from="work",
        clone_all=True,
        no_alias=True,
    )

    assert (target / "auth.json").is_file()
    assert (target / ".anthropic_oauth.json").is_file()
    assert not (target / ".anthropic_oauth.json.tmp.crash").exists()
    assert not (target / ".anthropic_oauth_deadbeef.tmp").exists()
    assert not (target / "shared").exists()


def test_delete_removes_auth_mirror_and_same_name_recreate_is_clean(
    auth_profile_env,
    monkeypatch,
):
    _, auth_home = auth_profile_env
    _disable_profile_services(monkeypatch)
    runtime_profile = profiles.create_profile("work", no_alias=True)
    auth_profile = auth_home / "profiles" / "work"
    auth_profile.mkdir(parents=True)
    (auth_profile / "auth.json").write_text('{"secret": true}\n')

    profiles.delete_profile("work", yes=True)

    assert not runtime_profile.exists()
    assert not auth_profile.exists()
    recreated = profiles.create_profile("work", no_alias=True)
    assert recreated.is_dir()
    assert not auth_profile.exists()


def test_rename_moves_auth_mirror_and_preflights_both_destinations(
    auth_profile_env,
    monkeypatch,
):
    _, auth_home = auth_profile_env
    monkeypatch.setattr(profiles, "check_alias_collision", lambda name: "skip")
    old_runtime = profiles.create_profile("work", no_alias=True)
    old_auth = auth_home / "profiles" / "work"
    old_auth.mkdir(parents=True)
    (old_auth / "auth.json").write_text('{"secret": true}\n')

    new_runtime = profiles.rename_profile("work", "ops")
    new_auth = auth_home / "profiles" / "ops"

    assert new_runtime.is_dir()
    assert not old_runtime.exists()
    assert (new_auth / "auth.json").is_file()
    assert not old_auth.exists()

    profiles.create_profile("other", no_alias=True)
    other_auth = auth_home / "profiles" / "other"
    other_auth.mkdir(parents=True)
    (other_auth / "auth.json").write_text('{"secret": "other"}\n')
    collision = auth_home / "profiles" / "taken"
    collision.mkdir(parents=True)

    with pytest.raises(FileExistsError):
        profiles.rename_profile("other", "taken")

    assert profiles.get_profile_dir("other").is_dir()
    assert (other_auth / "auth.json").is_file()
    assert not profiles.get_profile_dir("taken").exists()


def test_rename_rolls_runtime_back_when_auth_move_fails(
    auth_profile_env,
    monkeypatch,
):
    _, auth_home = auth_profile_env
    monkeypatch.setattr(profiles, "check_alias_collision", lambda name: "skip")
    old_runtime = profiles.create_profile("work", no_alias=True)
    old_auth = auth_home / "profiles" / "work"
    old_auth.mkdir(parents=True)
    (old_auth / "auth.json").write_text('{"secret": true}\n')
    original_rename = Path.rename

    def fail_auth_move(path, target):
        if path == old_auth:
            raise OSError("auth move failed")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_auth_move)

    with pytest.raises(OSError, match="auth move failed"):
        profiles.rename_profile("work", "ops")

    assert old_runtime.is_dir()
    assert old_auth.is_dir()
    assert not profiles.get_profile_dir("ops").exists()
    assert not (auth_home / "profiles" / "ops").exists()


def test_profile_symlink_mirror_is_moved_or_unlinked_without_recursive_follow(
    auth_profile_env,
    monkeypatch,
    tmp_path,
):
    _, auth_home = auth_profile_env
    _disable_profile_services(monkeypatch)
    monkeypatch.setattr(profiles, "check_alias_collision", lambda name: "skip")
    profiles.create_profile("work", no_alias=True)
    external = tmp_path / "external-profile-auth"
    external.mkdir()
    (external / "auth.json").write_text('{"secret": true}\n')
    restore_residue = _leave_restore_residue(
        external / ".anthropic_oauth.json"
    )
    atomic_residue = _leave_atomic_json_residue(
        external / ".anthropic_oauth.json"
    )
    quarantine = external / ".anthropic_oauth.json.corrupt"
    quarantine.write_text("quarantine", encoding="utf-8")
    (external / "unknown.keep").write_text("keep")
    old_mirror = auth_home / "profiles" / "work"
    old_mirror.parent.mkdir(parents=True)
    old_mirror.symlink_to(external, target_is_directory=True)

    profiles.rename_profile("work", "ops")

    new_mirror = auth_home / "profiles" / "ops"
    assert not os.path.lexists(old_mirror)
    assert new_mirror.is_symlink()
    assert (external / "auth.json").is_file()
    assert (external / "unknown.keep").is_file()

    profiles.delete_profile("ops", yes=True)

    assert not os.path.lexists(new_mirror)
    assert not (external / "auth.json").exists()
    assert not restore_residue.exists()
    assert not atomic_residue.exists()
    assert not quarantine.exists()
    assert (external / "unknown.keep").read_text() == "keep"


def test_named_export_excludes_both_provider_stores(
    auth_profile_env,
    tmp_path,
):
    source = profiles.create_profile("work", no_alias=True)
    (source / "auth.json").write_text("primary", encoding="utf-8")
    (source / ".anthropic_oauth.json").write_text(
        "anthropic",
        encoding="utf-8",
    )
    (source / "config.yaml").write_text("model: test\n", encoding="utf-8")

    archive = profiles.export_profile("work", str(tmp_path / "work.tar.gz"))
    with tarfile.open(archive, "r:gz") as exported:
        names = set(exported.getnames())

    assert "work/config.yaml" in names
    assert "work/auth.json" not in names
    assert "work/.anthropic_oauth.json" not in names


def test_import_rejects_stale_relocated_mirror_before_runtime_mutation(
    auth_profile_env,
    tmp_path,
):
    _, auth_home = auth_profile_env
    source = profiles.create_profile("source", no_alias=True)
    (source / "config.yaml").write_text("model: source\n", encoding="utf-8")
    archive = profiles.export_profile(
        "source",
        str(tmp_path / "source.tar.gz"),
    )
    stale_mirror = auth_home / "profiles" / "imported"
    stale_mirror.mkdir(parents=True)
    stale_auth = stale_mirror / "auth.json"
    stale_auth.write_text('{"secret":"stale"}\n', encoding="utf-8")

    with pytest.raises(FileExistsError, match="credential data"):
        profiles.import_profile(str(archive), name="imported")

    assert not os.path.lexists(profiles.get_profile_dir("imported"))
    assert stale_auth.read_text(encoding="utf-8") == '{"secret":"stale"}\n'


def test_dashboard_rename_and_delete_routes_use_auth_aware_lifecycle(
    auth_profile_env,
    monkeypatch,
):
    pytest.importorskip("fastapi")
    from hermes_cli.web_routers.profiles import (
        ProfileRename,
        delete_profile_endpoint,
        rename_profile_endpoint,
    )

    _, auth_home = auth_profile_env
    _disable_profile_services(monkeypatch)
    monkeypatch.setattr(profiles, "check_alias_collision", lambda name: "skip")
    profiles.create_profile("work", no_alias=True)
    old_auth = auth_home / "profiles" / "work"
    old_auth.mkdir(parents=True)
    (old_auth / "auth.json").write_text('{"secret": true}\n')

    renamed = asyncio.run(
        rename_profile_endpoint("work", ProfileRename(new_name="ops"))
    )
    assert renamed["ok"] is True
    new_auth = auth_home / "profiles" / "ops"
    assert (new_auth / "auth.json").is_file()
    assert not old_auth.exists()

    deleted = asyncio.run(delete_profile_endpoint("ops"))
    assert deleted["ok"] is True
    assert not new_auth.exists()
    assert not profiles.get_profile_dir("ops").exists()


def test_uninstall_profile_removes_runtime_and_auth_mirror(
    auth_profile_env,
    monkeypatch,
):
    _, auth_home = auth_profile_env
    runtime_profile = profiles.create_profile("work", no_alias=True)
    auth_profile = auth_home / "profiles" / "work"
    auth_profile.mkdir(parents=True)
    (auth_profile / "auth.json").write_text('{"secret": true}\n')
    monkeypatch.setattr(uninstall.subprocess, "run", lambda *args, **kwargs: None)

    uninstall._uninstall_profile(
        SimpleNamespace(
            name="work",
            path=runtime_profile,
            alias_path=None,
        )
    )

    assert not runtime_profile.exists()
    assert not auth_profile.exists()


def _disable_uninstall_side_effects(monkeypatch):
    monkeypatch.setattr(uninstall, "uninstall_gateway_service", lambda: False)
    monkeypatch.setattr(uninstall, "remove_path_from_shell_configs", lambda: [])
    monkeypatch.setattr(uninstall, "remove_wrapper_script", lambda: [])
    monkeypatch.setattr(uninstall, "remove_node_symlinks", lambda home: [])
    monkeypatch.setattr(uninstall, "_is_windows", lambda: False)
    monkeypatch.setattr(uninstall.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "hermes_cli.gui_uninstall.uninstall_gui",
        lambda home: [],
    )


def test_full_uninstall_cleans_known_residence_artifacts_but_preserves_unknown(
    auth_profile_env,
    monkeypatch,
    tmp_path,
):
    runtime_home, auth_home = auth_profile_env
    _disable_uninstall_side_effects(monkeypatch)
    project_root = tmp_path / "code"
    project_root.mkdir()
    (runtime_home / "config.yaml").write_text("model: test\n")

    for name in (
        "auth.json",
        "auth.lock",
        "auth.json.corrupt",
        ".anthropic_oauth.json",
        ".anthropic_oauth.lock",
    ):
        (auth_home / name).write_text("credential")
    root_restore = _leave_restore_residue(auth_home / "auth.json")
    root_anthropic_restore = _leave_restore_residue(
        auth_home / ".anthropic_oauth.json"
    )
    root_anthropic_atomic = _leave_atomic_json_residue(
        auth_home / ".anthropic_oauth.json"
    )
    legacy_anthropic = auth_home / ".anthropic_oauth.tmp.4242.deadbeef"
    legacy_anthropic.write_text("legacy", encoding="utf-8")
    anthropic_quarantine = auth_home / ".anthropic_oauth.json.corrupt"
    anthropic_quarantine.write_text("quarantine", encoding="utf-8")
    (auth_home / "unknown.keep").write_text("keep")

    profile_residues: list[Path] = []
    for profile_name in ("work", "orphan"):
        profile_home = auth_home / "profiles" / profile_name
        profile_home.mkdir(parents=True)
        (profile_home / "auth.json").write_text("credential")
        (profile_home / ".anthropic_oauth.json").write_text("credential")
        profile_residues.extend(
            (
                _leave_restore_residue(profile_home / "auth.json"),
                _leave_restore_residue(
                    profile_home / ".anthropic_oauth.json"
                ),
            )
        )
    (auth_home / "profiles" / "work" / "unknown.keep").write_text("keep")

    shared = auth_home / "shared"
    shared.mkdir()
    (shared / "nous_auth.json").write_text("credential")
    (shared / "nous_auth.lock").write_text("lock")
    shared_restore = _leave_restore_residue(shared / "nous_auth.json")
    (shared / "unknown.keep").write_text("keep")
    external_profile = tmp_path / "external-profile-auth"
    external_profile.mkdir()
    external_restore = _leave_restore_residue(
        external_profile / ".anthropic_oauth.json"
    )
    symlink_mirror = auth_home / "profiles" / "linked"
    symlink_mirror.symlink_to(external_profile, target_is_directory=True)
    broken_mirror = auth_home / "profiles" / "broken"
    broken_mirror.symlink_to(
        tmp_path / "missing-profile-auth",
        target_is_directory=True,
    )
    runtime_profile = runtime_home / "profiles" / "work"
    runtime_profile.mkdir(parents=True)

    uninstall._perform_uninstall(
        project_root=project_root,
        hermes_home=runtime_home,
        full_uninstall=True,
        remove_profiles=True,
        named_profiles=[
            SimpleNamespace(
                name="work",
                path=runtime_profile,
                alias_path=None,
            )
        ],
    )

    assert not runtime_home.exists()
    assert not project_root.exists()
    assert auth_home.is_dir()
    assert (auth_home / "unknown.keep").read_text() == "keep"
    assert not (auth_home / "auth.json").exists()
    assert not (auth_home / ".anthropic_oauth.json").exists()
    assert not root_restore.exists()
    assert not root_anthropic_restore.exists()
    assert not root_anthropic_atomic.exists()
    assert not legacy_anthropic.exists()
    assert not anthropic_quarantine.exists()
    assert (auth_home / "profiles" / "work" / "unknown.keep").read_text() == "keep"
    assert not (auth_home / "profiles" / "work" / "auth.json").exists()
    assert not (auth_home / "profiles" / "orphan").exists()
    assert all(not residue.exists() for residue in profile_residues)
    assert not os.path.lexists(symlink_mirror)
    assert not os.path.lexists(broken_mirror)
    assert external_profile.is_dir()
    assert not external_restore.exists()
    assert (shared / "unknown.keep").read_text() == "keep"
    assert not (shared / "nous_auth.json").exists()
    assert not shared_restore.exists()


def test_named_full_uninstall_keeps_default_sibling_and_shared_credentials(
    tmp_path,
    monkeypatch,
):
    operator = tmp_path / "operator"
    root = operator / ".hermes"
    active = root / "profiles" / "work"
    sibling = root / "profiles" / "sibling"
    active.mkdir(parents=True)
    sibling.mkdir()
    residence = tmp_path / "residence"
    (residence / "profiles" / "work").mkdir(parents=True)
    (residence / "profiles" / "sibling").mkdir()
    (residence / "shared").mkdir()
    monkeypatch.setattr(Path, "home", lambda: operator)
    monkeypatch.setenv("HOME", str(operator))
    monkeypatch.setenv("HERMES_HOME", str(active))
    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(residence / "shared"))
    _disable_uninstall_side_effects(monkeypatch)

    (root / "config.yaml").write_text("default", encoding="utf-8")
    (active / "config.yaml").write_text("active", encoding="utf-8")
    (sibling / "config.yaml").write_text("sibling", encoding="utf-8")
    default_auth = residence / "auth.json"
    active_auth = residence / "profiles" / "work" / "auth.json"
    sibling_auth = residence / "profiles" / "sibling" / "auth.json"
    shared_auth = residence / "shared" / "nous_auth.json"
    for path in (default_auth, active_auth, sibling_auth, shared_auth):
        path.write_text("credential", encoding="utf-8")

    project_root = tmp_path / "code"
    project_root.mkdir()
    uninstall._perform_uninstall(
        project_root=project_root,
        hermes_home=active,
        full_uninstall=True,
        remove_profiles=True,
        named_profiles=[
            SimpleNamespace(
                name="sibling",
                path=sibling,
                alias_path=None,
            )
        ],
    )

    assert not active.exists()
    assert not active_auth.exists()
    assert (root / "config.yaml").read_text(encoding="utf-8") == "default"
    assert (sibling / "config.yaml").read_text(encoding="utf-8") == "sibling"
    assert default_auth.read_text(encoding="utf-8") == "credential"
    assert sibling_auth.read_text(encoding="utf-8") == "credential"
    assert shared_auth.read_text(encoding="utf-8") == "credential"


def test_full_uninstall_preserves_nested_residence_root_and_unknown_contents(
    tmp_path,
    monkeypatch,
):
    runtime_home = tmp_path / "runtime"
    residence = runtime_home / "launcher-auth"
    shared = residence / "shared"
    profile_auth = residence / "profiles" / "work"
    profile_auth.mkdir(parents=True)
    shared.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(runtime_home))
    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(shared))
    _disable_uninstall_side_effects(monkeypatch)

    (runtime_home / "config.yaml").write_text("runtime", encoding="utf-8")
    runtime_state = runtime_home / "sessions"
    runtime_state.mkdir()
    (runtime_state / "session.json").write_text("runtime", encoding="utf-8")
    (residence / "auth.json").write_text("credential", encoding="utf-8")
    (profile_auth / "auth.json").write_text("credential", encoding="utf-8")
    (shared / "nous_auth.json").write_text("credential", encoding="utf-8")
    (residence / "unknown.keep").write_text("root unknown", encoding="utf-8")
    (profile_auth / "unknown.keep").write_text(
        "profile unknown",
        encoding="utf-8",
    )
    (shared / "unknown.keep").write_text("shared unknown", encoding="utf-8")

    project_root = tmp_path / "code"
    project_root.mkdir()
    uninstall._perform_uninstall(
        project_root=project_root,
        hermes_home=runtime_home,
        full_uninstall=True,
        remove_profiles=False,
        named_profiles=[],
    )

    assert not (runtime_home / "config.yaml").exists()
    assert not runtime_state.exists()
    assert residence.is_dir()
    assert (residence / "unknown.keep").read_text(encoding="utf-8") == (
        "root unknown"
    )
    assert (profile_auth / "unknown.keep").read_text(encoding="utf-8") == (
        "profile unknown"
    )
    assert (shared / "unknown.keep").read_text(encoding="utf-8") == (
        "shared unknown"
    )
    assert not (residence / "auth.json").exists()
    assert not (profile_auth / "auth.json").exists()
    assert not (shared / "nous_auth.json").exists()
    assert set(runtime_home.iterdir()) == {residence}


def test_full_uninstall_preserves_nested_residence_symlink_entry(
    tmp_path,
    monkeypatch,
):
    runtime_home = tmp_path / "runtime"
    runtime_home.mkdir()
    external_residence = tmp_path / "external-auth"
    external_residence.mkdir()
    residence_link = runtime_home / "launcher-auth"
    residence_link.symlink_to(external_residence, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(runtime_home))
    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence_link))
    monkeypatch.delenv("HERMES_SHARED_AUTH_DIR", raising=False)
    _disable_uninstall_side_effects(monkeypatch)

    (runtime_home / "config.yaml").write_text("runtime", encoding="utf-8")
    (external_residence / "auth.json").write_text(
        "credential",
        encoding="utf-8",
    )
    (external_residence / "unknown.keep").write_text(
        "unknown",
        encoding="utf-8",
    )
    project_root = tmp_path / "code"
    project_root.mkdir()

    uninstall._perform_uninstall(
        project_root=project_root,
        hermes_home=runtime_home,
        full_uninstall=True,
        remove_profiles=False,
        named_profiles=[],
    )

    assert residence_link.is_symlink()
    assert external_residence.is_dir()
    assert not (external_residence / "auth.json").exists()
    assert (external_residence / "unknown.keep").read_text(encoding="utf-8") == (
        "unknown"
    )
    assert set(runtime_home.iterdir()) == {residence_link}


def test_keep_data_uninstall_preserves_runtime_and_residence(
    auth_profile_env,
    monkeypatch,
    tmp_path,
):
    runtime_home, auth_home = auth_profile_env
    _disable_uninstall_side_effects(monkeypatch)
    project_root = tmp_path / "code"
    project_root.mkdir()
    runtime_auth = runtime_home / "config.yaml"
    residence_auth = auth_home / "auth.json"
    runtime_auth.write_text("model: test\n")
    residence_auth.write_text('{"secret": true}\n')

    uninstall._perform_uninstall(
        project_root=project_root,
        hermes_home=runtime_home,
        full_uninstall=False,
        remove_profiles=False,
        named_profiles=[],
    )

    assert not project_root.exists()
    assert runtime_auth.is_file()
    assert residence_auth.is_file()
