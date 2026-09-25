"""Release metadata must reach PM admission before source completion (#122054)."""

import json
import subprocess

import pytest

from hermes_cli import version_info
from hermes_cli.source_stamp import write_source_stamp
from pm.plugin_declarations import manifest_version_error, read_native_manifest


def _git(root, *args):
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True,
    ).stdout.strip()


@pytest.fixture
def release_checkout(tmp_path, monkeypatch):
    remote = tmp_path / "remote"
    remote.mkdir()
    _git(remote, "init", "-q", "-b", "main")
    _git(remote, "config", "user.name", "Hermes test")
    _git(remote, "config", "user.email", "test@example.invalid")
    old, new = "0.10.0", "0.11.0"
    for version in (old, new, "0.0.0"):
        (remote / "pyproject.toml").write_text(
            f'[project]\nname="metadata-fixture"\nversion="{version}"\n', encoding="utf-8",
        )
        _git(remote, "add", ".")
        _git(remote, "-c", "commit.gpgsign=false", "commit", "-qm", version)
        if version == old:
            _git(remote, "tag", "v2025.1.1")
        elif version == new:
            release = _git(remote, "rev-parse", "HEAD")
    root = tmp_path / "checkout"
    _git(tmp_path, "clone", "-q", "--no-tags", remote.as_uri(), str(root))
    _git(root, "fetch", "-q", "origin", "tag", "v2025.1.1")
    stamp = write_source_stamp(root)
    assert stamp["baseVersion"] == old
    # The new release ref exists upstream but neither HEAD nor the old stamp
    # changes. Even an ordinary branch fetch need not acquire this tag.
    _git(remote, "tag", "v2025.2.1", release)
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(root))
    monkeypatch.setattr(version_info, "_resolve_repo_dir", lambda: root)
    version_info._reset_version_info_cache()
    yield root, old, new
    version_info._reset_version_info_cache()


def test_source_stamp_does_not_freeze_tags_but_packaged_identity_stays_authoritative(
    release_checkout, monkeypatch,
):
    root, old, new = release_checkout
    stamp_path = root / "install-stamp.json"
    before = stamp_path.read_bytes()
    assert version_info.get_version_info().base_version == old
    _git(root, "fetch", "-q", "origin", "tag", "v2025.2.1")
    run = subprocess.run

    def local_only(command, **kwargs):
        assert "fetch" not in command and "ls-remote" not in command
        return run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", local_only)
    version_info._reset_version_info_cache()
    assert version_info.get_version_info().base_version == new
    assert stamp_path.read_bytes() == before

    packaged = json.loads(before)
    packaged.update(source="ci", distribution="docker", updateMechanism="external")
    stamp_path.write_text(json.dumps(packaged), encoding="utf-8")
    version_info._reset_version_info_cache()
    assert version_info.get_version_info().base_version == old


@pytest.mark.parametrize("entrypoint", ["current", "historical", "startup"])
def test_update_preparation_refreshes_release_metadata_before_plugin_admission(
    release_checkout, tmp_path, monkeypatch, entrypoint,
):
    import pm
    from hermes_cli import _update_takeover, update_completion, venv_sync

    root, old, new = release_checkout
    plugin_dir = tmp_path / "fixture"
    plugin_dir.mkdir()
    plugin = plugin_dir / "plugin.yaml"
    plugin.write_text(f'name: fixture\nrequires_hermes: ">={new}"\n', encoding="utf-8")
    manifest = read_native_manifest(plugin)
    assert manifest_version_error(manifest, "fixture") is not None  # also warm the cache
    before = (root / "install-stamp.json").read_bytes()
    head = _git(root, "rev-parse", "HEAD")
    fetched = _git(root, "rev-parse", "FETCH_HEAD")

    class Admitted(Exception):
        pass

    def admit(*args, **kwargs):
        assert manifest_version_error(manifest, "fixture") is None
        assert manifest_version_error({"requires_hermes": f">{new}"}, "future") is not None
        raise Admitted  # No dependency install or application completion in this test.

    monkeypatch.setattr(pm, "sync_venv", admit)
    monkeypatch.setattr("pm.client.sync_venv", admit)
    monkeypatch.setattr("pm.client.ensure_tools_for_sync", lambda: None)
    monkeypatch.setattr("pm.extras.legacy_selection", lambda root: [])
    monkeypatch.setattr("hermes_cli.update_stage.ensure_panel", lambda root: None)
    monkeypatch.setattr("hermes_cli.update_stage.publish_stage", lambda *args: None)
    request = {"root": str(root), "source": str(root), "update_id": "metadata-test",
               "receipt": {"update_id": "metadata-test"}}
    if entrypoint == "startup":
        from pm.environments import runtime_facts_path
        from pm.install import venv_is_current
        from pm.lock import Facts
        from pm.package import InstallError

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
        # A recorded PM install validates its plugin union in the currency
        # check itself, before startup can decide to finish the update.
        (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
        record = runtime_facts_path(root)
        record.parent.mkdir(parents=True)
        Facts(record).record_state("venv", "previous", [])
        monkeypatch.setattr("pm.workspace.enabled_plugin_dirs", lambda **kwargs: [plugin_dir])
        monkeypatch.setattr(pm, "venv_is_current", venv_is_current)
        with pytest.raises(InstallError, match="requires hermes"):
            pm.venv_is_current(project_root=root)
    with pytest.raises(Admitted):
        if entrypoint == "current":
            update_completion._prepare(request, tmp_path / "request.json", tmp_path / "result.json")
        elif entrypoint == "historical":
            _update_takeover.prepare(request)
        else:
            venv_sync.prepare_launch(root, [])

    assert version_info.get_version_info().base_version == new
    assert _git(root, "rev-parse", "HEAD") == head
    assert _git(root, "rev-parse", "FETCH_HEAD") == fetched
    # Identity publication still belongs to successful completion, not preparation.
    assert (root / "install-stamp.json").read_bytes() == before
