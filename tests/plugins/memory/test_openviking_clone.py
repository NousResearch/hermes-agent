"""Cloning preserves remote identity, never another run's recovery ownership."""

import copy
import json
import os
import shutil
import socket
import stat
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

from agent import secret_scope


@contextmanager
def profile_scope(home):
    from hermes_cli.profile_clone import clone_source_scope

    with clone_source_scope(home):
        yield


@pytest.fixture(params=[False, True], ids=["standalone", "multiplex"])
def isolated(tmp_path, monkeypatch, request):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch"))
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", request.param)

    def forbidden(*args, **kwargs):
        pytest.fail("Clone/config resolution must not perform networking")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    return tmp_path


def companion(tmp_path):
    # Relocate the actual artifact; the real discovery path must stay lazy even
    # when an external provider's runtime cannot be imported at all.
    from plugins.memory.surfaces import load_provider_companion

    artifact = Path(__file__).parents[3] / "plugins/memory/openviking/clone.py"
    assert artifact.is_file(), "OpenViking needs a lazy clone companion"
    installed = tmp_path / "installed-home/plugins/relocated_openviking"
    installed.mkdir(parents=True, exist_ok=True)
    (installed / "__init__.py").write_text('raise AssertionError("MemoryProvider activated during clone")\n')
    shutil.copyfile(artifact, installed / "clone.py")
    with profile_scope(tmp_path / "installed-home"):
        module = load_provider_companion("relocated_openviking", "clone")
    assert module is not None and module.__package__ is not None
    assert not getattr(sys.modules[module.__package__], "__file__", None)
    return module


def stage(source, staging, clone_all):
    if clone_all:
        shutil.copytree(source, staging, symlinks=True)
    else:
        staging.mkdir()
        for name in ("config.yaml", ".env"):
            shutil.copyfile(source / name, staging / name)


@pytest.mark.parametrize("clone_all", [False, True])
@pytest.mark.parametrize("kind", ["yaml", "yaml-ref", "yaml-env-ref", "env", "private-link", "external", "external-link", "global", "unlinked", "null-memory", "null-openviking"])
@pytest.mark.parametrize("peer", [None, "shared-peer"])
def test_clone_preserves_connection_with_private_local_ownership(isolated, monkeypatch, clone_all, kind, peer):
    root = isolated
    source, staging, destination = (root / name for name in ("source", "staging", "destination"))
    source.mkdir()
    relative = Path("private config") / "ovcli.conf"
    private = source / relative
    private.parent.mkdir()
    connection = {"url": "http://127.0.0.1:1933", "api_key": "test-only-key", "account": "shared-account", "user": "shared-user"}
    if peer is not None:
        connection["agent_id"] = peer
    private.write_text(json.dumps(connection))
    external = root / ".openviking/ovcli.conf"
    external.parent.mkdir()
    external.write_bytes(private.read_bytes())
    inactive = kind in {"null-memory", "null-openviking"}
    pointer = external if kind in {"external", "global"} or inactive else private
    if kind in {"private-link", "external-link"}:
        pointer = (private.parent if kind == "private-link" else external.parent) / "linked.conf"
        pointer.symlink_to(private if kind == "private-link" else external)
        if kind == "private-link":
            relative = pointer.relative_to(source)
    provider_config = {"use_ovcli_config": kind != "unlinked"}
    if kind != "global":
        provider_config["ovcli_config_path"] = str(pointer)
    if kind == "env":
        provider_config["ovcli_config_path"] = str(external)  # env must win
    if kind in {"yaml-ref", "yaml-env-ref"}:
        provider_config["ovcli_config_path"] = "${OVCLI_PATH}" if kind == "yaml-ref" else "${env:OVCLI_PATH}"
    config = {"memory": {"provider": "openviking", "openviking": provider_config}, "unrelated": "${UNRELATED}"}
    if inactive:
        config["memory"] = None if kind == "null-memory" else {"provider": "builtin", "openviking": None}
        provider_config = {}
    (source / "config.yaml").write_text(yaml.safe_dump(config))
    env = "# keep this comment\nUNRELATED=test-only-value\n"
    if kind in {"yaml-ref", "yaml-env-ref"}:
        env += f'OVCLI_PATH="{private}"\n'
    if kind == "env":
        env += f'export OPENVIKING_CLI_CONFIG_FILE="{private}"\n'
    (source / ".env").write_text(env)
    for name in ("pending_sessions", "runs", "keep"):
        directory = source / "openviking" / name
        directory.mkdir(parents=True)
        (directory / "owned.json").write_text('{"session_id":"source-session","owner_run_id":"source-run"}')
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    monkeypatch.setenv("OPENVIKING_CLI_CONFIG_FILE", str(root / "wrong-launch.conf"))
    monkeypatch.setenv("OVCLI_PATH", str(root / "wrong-launch-ref.conf"))
    environment_before = dict(os.environ)

    from hermes_cli.config import _expand_env_vars
    import plugins.memory.openviking as runtime
    with profile_scope(source):
        effective_provider = _expand_env_vars(provider_config)
        expected = runtime._resolve_ovcli_config_path(effective_provider.get("ovcli_config_path", ""))
        assert expected == pointer
        settings = runtime._resolve_connection_settings(effective_provider)

    stage(source, staging, clone_all)
    module = companion(root)
    with profile_scope(source):
        module.prepare_clone(source_home=source, source_name="source", staging_home=staging,
                             destination_home=destination, destination_name="destination", clone_all=clone_all)
    staging.rename(destination)
    cloned_raw = yaml.safe_load((destination / "config.yaml").read_text())
    cloned = (cloned_raw.get("memory") or {}).get("openviking") or {}
    is_private = kind in {"yaml", "yaml-ref", "yaml-env-ref", "env", "private-link"}
    expected_raw = copy.deepcopy(config)
    if is_private and kind != "env":
        expected_raw["memory"]["openviking"]["ovcli_config_path"] = str(destination / relative)
    assert cloned_raw == expected_raw
    for home, cfg in ((source, provider_config), (destination, cloned), (source, provider_config)):
        with profile_scope(home):
            cfg = _expand_env_vars(cfg)
            assert runtime._resolve_connection_settings(cfg) == settings
            expected_path = destination / relative if home == destination and is_private else pointer
            assert runtime._resolve_ovcli_config_path(cfg.get("ovcli_config_path", "")) == expected_path
    if is_private:
        copied = destination / relative
        assert copied.read_bytes() == private.read_bytes()
        if os.name == "posix":  # Windows stat synthesizes mode bits, not ACLs.
            assert stat.S_IMODE(copied.stat().st_mode) == 0o600
        assert not copied.is_symlink()
    for name in ("pending_sessions", "runs"):
        assert not (destination / "openviking" / name).exists()
    assert (destination / "openviking/keep/owned.json").exists() == clone_all
    assert {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()} == before
    assert (destination / ".env").read_text().startswith("# keep this comment\nUNRELATED=test-only-value\n")
    assert dict(os.environ) == environment_before


@pytest.mark.parametrize("case", ["relative", "missing", "escape", "private-parent", "external-private-link", "external-private-parent", "source-root", "external-root-link", "link-loop", "staging-parent", "recovery-parent", "config-link", "env-link", "nul", "staging-root", "memory-list", "memory-false", "openviking-list", "openviking-string"])
def test_clone_refuses_ambiguous_or_escaping_paths_without_touching_targets(isolated, case):
    root = isolated
    source, staging, destination = (root / name for name in ("source", "staging", "destination"))
    source.mkdir()
    staging.mkdir()
    outside = root / "outside"
    outside.mkdir()
    sentinel = outside / "ovcli.conf"
    sentinel.write_text('{"api_key":"do-not-disclose"}')
    private = source / "private/ovcli.conf"
    private.parent.mkdir()
    private.write_bytes(sentinel.read_bytes())
    pointer = str(private)
    if case == "relative":
        pointer = "private/ovcli.conf"
    if case == "missing":
        private.unlink()
    if case == "escape":
        private.unlink()
        private.symlink_to(sentinel)
    if case == "private-parent":
        private.unlink()
        private.parent.rmdir()
        private.parent.symlink_to(outside, target_is_directory=True)
    if case == "external-private-link":
        alias = outside / "alias.conf"
        alias.symlink_to(private)
        pointer = str(alias)
    if case == "external-private-parent":
        alias = outside / "alias"
        alias.symlink_to(private.parent, target_is_directory=True)
        pointer = str(alias / private.name)
    if case == "source-root":
        pointer = str(source)
    if case == "external-root-link":
        alias = outside / "alias"
        alias.symlink_to(source, target_is_directory=True)
        pointer = str(alias)
    if case == "link-loop":
        private.unlink()
        private.symlink_to(private)
    if case == "nul":
        pointer += "\x00do-not-disclose"
    config = {"memory": {"openviking": {"use_ovcli_config": True, "ovcli_config_path": pointer}}}
    malformed = {"memory-list": [], "memory-false": False, "openviking-list": [], "openviking-string": "do-not-disclose"}
    if case in malformed:
        if case.startswith("memory-"):
            config["memory"] = malformed[case]
        else:
            config["memory"]["openviking"] = malformed[case]
    for home in (source, staging):
        (home / "config.yaml").write_text(yaml.safe_dump(config))
        (home / ".env").write_text(f'OPENVIKING_CLI_CONFIG_FILE="{private}"\n' if case == "env-link" else "")
    if case in {"staging-parent", "recovery-parent"}:
        (staging / ("private" if case == "staging-parent" else "openviking")).symlink_to(outside, target_is_directory=True)
    if case in {"config-link", "env-link"}:
        target = staging / ("config.yaml" if case == "config-link" else ".env")
        target.unlink()
        target.symlink_to(source / target.name)
    if case == "staging-root":
        alias = root / "alias"
        alias.symlink_to(staging, target_is_directory=True)
        staging = alias
    originals = {p: p.read_bytes() for p in (sentinel, source / "config.yaml", source / ".env")}
    module = companion(root)
    with profile_scope(source), pytest.raises(ValueError, match="OpenViking") as error:
        module.prepare_clone(source_home=source, source_name="source", staging_home=staging,
                             destination_home=destination, destination_name="destination", clone_all=True)
    assert "do-not-disclose" not in str(error.value)
    if case in malformed:
        assert "mapping" in str(error.value) and "config.yaml" in str(error.value)
    assert {p: p.read_bytes() for p in originals} == originals


@pytest.mark.macos_only
@pytest.mark.parametrize("alias_parent", ["/tmp", "/var/tmp"])
@pytest.mark.parametrize("canonical_source", [False, True])
@pytest.mark.parametrize("root_spelling", ["source", "SOURCE"])
@pytest.mark.parametrize("active", [False, True], ids=["inactive", "private"])
@pytest.mark.parametrize("clone_all", [False, True])
def test_clone_accepts_host_aliases_above_profile_roots(isolated, alias_parent, canonical_source, root_spelling, active, clone_all):
    # Exercise the actual host aliases, not a mocked platform or resolved tmp_path.
    assert Path(alias_parent).resolve() != Path(alias_parent)
    module = companion(isolated)
    with tempfile.TemporaryDirectory(prefix="ovcli-clone-test-", dir=alias_parent) as temporary:
        root = Path(temporary)
        source = root / "source"
        source.mkdir()
        private = source / "private/ovcli.conf"
        private.parent.mkdir()
        private.write_text('{"api_key":"test-only-key"}')
        # Either spelling of the source root must recognize the other spelling
        # of a private pointer; otherwise the clone silently shares credentials.
        pointer = root / root_spelling / "private/ovcli.conf"
        if root_spelling != source.name and not pointer.exists():
            pytest.skip("Requires a case-insensitive test volume")
        assert pointer.samefile(private)
        pointer = pointer if canonical_source else pointer.resolve()
        source = source.resolve() if canonical_source else source
        config: dict = {"memory": {"provider": "builtin"}}
        if active:
            config["memory"] = {"provider": "openviking", "openviking": {
                "use_ovcli_config": True, "ovcli_config_path": str(pointer),
            }}
        (source / "config.yaml").write_text(yaml.safe_dump(config))
        (source / ".env").write_text("# source environment\n")
        for name in ("runs", "pending_sessions"):
            owned = source / "openviking" / name
            owned.mkdir(parents=True)
            (owned / "owned.json").write_text("source-run")
        before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
        # A host-owned hidden sibling, addressed through /tmp or /var, is the
        # boundary. Its parent is trusted; the staging leaf is not resolved away.
        staging, destination = root / ".destination-staging", root / "destination"
        stage(source, staging, clone_all)
        with profile_scope(source):
            result = module.prepare_clone(
                source_home=source, source_name="source", staging_home=staging,
                destination_home=destination, destination_name="destination", clone_all=clone_all,
            )
        assert result["private_config_materialized"] is active
        staging.rename(destination)
        cloned = yaml.safe_load((destination / "config.yaml").read_text())
        if active:
            copied = destination / "private/ovcli.conf"
            assert copied.read_bytes() == private.read_bytes()
            assert not copied.is_symlink()
            assert not copied.samefile(private)
            assert stat.S_IMODE(copied.stat().st_mode) == 0o600
            assert cloned["memory"]["openviking"]["ovcli_config_path"] == str(copied)
        else:
            assert cloned == config
        assert all(not (destination / "openviking" / name).exists() for name in ("runs", "pending_sessions"))
        assert {p: p.read_bytes() for p in before} == before


@pytest.mark.macos_only
@pytest.mark.parametrize("root_spelling", ["source", "SOURCE"])
def test_host_alias_does_not_hide_copied_private_parent_link(isolated, root_spelling):
    module = companion(isolated)
    with tempfile.TemporaryDirectory(prefix="ovcli-clone-test-", dir="/tmp") as temporary:
        root = Path(temporary)
        source, staging = root / "source", root / ".destination-staging"
        source.mkdir()
        (source / "ovcli.conf").write_text('{"api_key":"test-only-key"}')
        (source / "private").symlink_to(source, target_is_directory=True)
        pointer = root / root_spelling / "private/ovcli.conf"
        if root_spelling != source.name and not pointer.exists():
            pytest.skip("Requires a case-insensitive test volume")
        assert pointer.samefile(source / "ovcli.conf")
        config = {"memory": {"openviking": {
            "use_ovcli_config": True, "ovcli_config_path": str(pointer),
        }}}
        (source / "config.yaml").write_text(yaml.safe_dump(config))
        (source / ".env").write_text("")
        stage(source, staging, True)
        before = {p: p.read_bytes() for p in (source / "ovcli.conf", source / "config.yaml", staging / "config.yaml")}
        # Resolve the caller's source only: pointer spelling uses /tmp, and the
        # nested link also resolves to source. It must not redefine the boundary.
        with profile_scope(source.resolve()), pytest.raises(ValueError, match="staging link/reparse"):
            module.prepare_clone(
                source_home=source.resolve(), source_name="source", staging_home=staging,
                destination_home=root / "destination", destination_name="destination", clone_all=True,
            )
        assert {p: p.read_bytes() for p in before} == before


@pytest.mark.macos_only
@pytest.mark.parametrize("clone_all", [False, True])
@pytest.mark.parametrize("relative", ["openviking/RUNS/ovcli.conf", "OPENVIKING/Pending_Sessions/ovcli.conf", "CONFIG.YAML", ".ENV"])
def test_clone_refuses_case_colliding_private_paths_before_writing(isolated, clone_all, relative):
    root = isolated
    source, staging, destination = (root / name for name in ("source", "staging", "destination"))
    source.mkdir()
    private = source / relative
    private.parent.mkdir(parents=True, exist_ok=True)
    private.write_text('{"api_key":"test-only-key"}')
    config = {"memory": {"openviking": {"use_ovcli_config": True, "ovcli_config_path": str(private)}}}
    (source / "config.yaml").write_text(yaml.safe_dump(config))
    (source / ".env").write_text("# source environment\n")
    stage(source, staging, clone_all)
    before = {p: p.read_bytes() for home in (source, staging) for p in home.rglob("*") if p.is_file()}
    module = companion(root)
    with profile_scope(source), pytest.raises(ValueError, match="overlaps clone metadata"):
        module.prepare_clone(source_home=source, source_name="source", staging_home=staging,
                             destination_home=destination, destination_name="destination", clone_all=clone_all)
    assert {p: p.read_bytes() for home in (source, staging) for p in home.rglob("*") if p.is_file()} == before
    assert not destination.exists()


@pytest.mark.windows_only
@pytest.mark.parametrize("relative", ["openviking", "openviking/runs", "private", "private/ovcli.conf", "staging-root"])
def test_clone_refuses_junctions_before_writing(isolated, relative):
    import _winapi
    from hermes_cli.profiles import _copytree_keep_junctions

    root = isolated
    source, staging, destination = (root / name for name in ("source", "staging", "destination"))
    source.mkdir()
    outside = root / "outside"
    outside.mkdir()
    for name in ("runs", "pending_sessions"):
        (outside / name).mkdir()
        (outside / name / "owned.json").write_text("source-run")
    (outside / "ovcli.conf").write_text('{"api_key":"test-only-key"}')
    private = source / "private/ovcli.conf"
    private.parent.mkdir()
    private.write_bytes((outside / "ovcli.conf").read_bytes())
    config = {"memory": {"openviking": {"use_ovcli_config": True, "ovcli_config_path": str(private)}}}
    (source / "config.yaml").write_text(yaml.safe_dump(config))
    (source / ".env").write_text("# source environment\n")
    # Use the host's copy path: unlike shutil.copytree, it preserves junctions.
    if relative.startswith("openviking"):
        junction = source / relative
        junction.parent.mkdir(parents=True, exist_ok=True)
        _winapi.CreateJunction(str(outside), str(junction))
    _copytree_keep_junctions(source, staging, lambda directory, names: set())
    if relative.startswith("private"):
        junction = staging / relative
        if junction.is_dir():
            shutil.rmtree(junction)
        else:
            junction.unlink()
        _winapi.CreateJunction(str(outside), str(junction))
    if relative == "staging-root":
        alias = root / "alias"
        _winapi.CreateJunction(str(staging), str(alias))
        staging = alias
    before = {p: p.read_bytes() for home in (source, staging, outside) for p in home.rglob("*") if p.is_file()}
    module = companion(root)
    with profile_scope(source), pytest.raises(ValueError, match="OpenViking"):
        module.prepare_clone(source_home=source, source_name="source", staging_home=staging,
                             destination_home=destination, destination_name="destination", clone_all=True)
    assert {p: p.read_bytes() for home in (source, staging, outside) for p in home.rglob("*") if p.is_file()} == before
    assert not destination.exists()
