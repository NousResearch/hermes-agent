"""Portable transfers keep user files, not machine-specific PM state."""
from argparse import Namespace
from pathlib import Path
import tarfile
import zipfile

import pytest


@pytest.fixture
def transfer_home(tmp_path, monkeypatch):
    home = tmp_path / "source-home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import backup, gateway, profiles

    # Restoring data must not install host services or modify the test user's shell.
    monkeypatch.setattr(gateway, "ensure_gateway_service", lambda **kwargs: False)
    monkeypatch.setattr(gateway, "_is_service_running", lambda: False)
    monkeypatch.setattr(profiles, "check_alias_collision", lambda name: "test has no aliases")
    monkeypatch.setattr(backup, "_collect_memory_provider_external_paths", lambda: [])
    return home


def _write_files(root, files):
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


@pytest.mark.parametrize("route", ["backup", "default", "named", "distribution", "explicit"])
def test_transfers_preserve_user_files_without_porting_pm_state(
    transfer_home, tmp_path, monkeypatch, route
):
    from hermes_cli.backup import run_backup, run_import
    from hermes_cli.profile_distribution import (
        DistributionManifest, install_distribution, write_manifest,
    )
    from hermes_cli.profiles import export_profile, get_profile_dir, import_profile

    portable = {
        "config.yaml": "model:\n  model: portable\n",
        "SOUL.md": "Be kind.\n",
        "plugins/example/facts.json": '{"owner": "plugin"}\n',
        "skills/example/tools/helper.py": "print('portable tool')\n",
        "skills/example/cache/notes.txt": "user-authored data\n",
    }
    runtime = {
        "installs/install-id/facts.json": '{"environment": "C:/other-machine/generation"}\n',
        "installs/install-id/environments/generation/pyvenv.cfg": "home = other-machine\n",
        "tools/uv/bin/uv": "machine executable\n",
        "cache/partials/transfer.partial": "unfinished transfer\n",
    }
    source = get_profile_dir("porter") if route == "named" else transfer_home
    _write_files(source, portable | runtime)
    if route == "backup":
        portable |= {f"profiles/coder/{rel}": text for rel, text in list(portable.items())}
        runtime |= {f"profiles/coder/{rel}": text for rel, text in list(runtime.items())}
        _write_files(source, portable | runtime)
        archive = tmp_path / "portable.zip"
        run_backup(Namespace(output=str(archive)))
        with zipfile.ZipFile(archive) as reader:
            names = set(reader.namelist())
    elif route in {"default", "named"}:
        name = "porter" if route == "named" else "default"
        archive = export_profile(name, str(tmp_path / "portable.tar.gz"))
        with tarfile.open(archive) as reader:
            names = {member.name.removeprefix(f"{name}/") for member in reader if member.isfile()}
    else:
        owned = sorted({Path(rel).parts[0] for rel in portable | runtime}) if route == "explicit" else []
        write_manifest(source, DistributionManifest(name="portable", distribution_owned=owned))
        names = None

    if names is not None:
        assert set(portable) <= names
        assert not (set(runtime) & names)

    target_home = tmp_path / "target-home"
    target_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(target_home))
    if route == "backup":
        run_import(Namespace(zipfile=str(archive), force=True))
        restored = target_home
    elif route in {"default", "named"}:
        restored = import_profile(str(archive), name="restored")
    else:
        restored = install_distribution(str(source), name="restored").target_dir

    for rel, content in portable.items():
        assert (restored / rel).read_text(encoding="utf-8") == content, (route, rel)
    for rel, content in runtime.items():
        assert not (restored / rel).exists(), (route, rel)
        assert (source / rel).read_text(encoding="utf-8") == content


@pytest.mark.parametrize("route", ["backup", "profile", "profile-denied", "distribution", "explicit"])
def test_incoming_runtime_state_never_replaces_target_runtime(
    transfer_home, tmp_path, monkeypatch, route
):
    from hermes_cli.backup import run_import
    from hermes_cli.profile_distribution import (
        DistributionManifest, install_distribution, write_manifest,
    )
    from hermes_cli.profiles import get_profile_dir, import_profile

    incoming = {
        "config.yaml": "model:\n  model: restored\n",
        "plugins/example/facts.json": '{"owner": "plugin"}\n',
        "skills/example/tools/helper.py": "print('user tool')\n",
        "skills/example/cache/notes.txt": "keep user files\n",
    }
    machine = {
        "installs/key/facts.json": "foreign generation\n",
        "tools/uv/bin/uv": "foreign executable\n",
        "cache": "foreign cache file\n",
    }
    if route == "backup":
        incoming |= {f"profiles/coder/{rel}": text for rel, text in list(incoming.items())}
        machine |= {f"profiles/coder/{rel}": text for rel, text in list(machine.items())}
    staged = tmp_path / "archive-source"
    _write_files(staged, incoming | machine)
    target = transfer_home if route == "backup" else get_profile_dir("restored")
    if not route.startswith("profile"):
        _write_files(target, dict.fromkeys(machine, "target-owned runtime\n"))

    if route == "backup":
        archive = tmp_path / "legacy.zip"
        with zipfile.ZipFile(archive, "w") as writer:
            for rel, content in (incoming | machine).items():
                writer.writestr(f".hermes/{rel}", content)
        run_import(Namespace(zipfile=str(archive), force=True))
    elif route.startswith("profile"):
        archive = tmp_path / "legacy.tar.gz"
        with tarfile.open(archive, "w:gz") as writer:
            writer.add(staged, arcname="legacy")
        if route == "profile-denied":
            from hermes_cli import profiles

            real_rmtree = profiles.shutil.rmtree

            def deny_runtime_removal(path, *args, **kwargs):
                if Path(path).name == "installs":
                    if kwargs.get("ignore_errors"):
                        return
                    raise PermissionError("staged runtime is locked")
                return real_rmtree(path, *args, **kwargs)

            with monkeypatch.context() as scoped:
                scoped.setattr(profiles.shutil, "rmtree", deny_runtime_removal)
                with pytest.raises(PermissionError, match="staged runtime is locked"):
                    import_profile(str(archive), name="restored")
            assert not target.exists()
            return
        target = import_profile(str(archive), name="restored")
    else:
        owned = sorted({Path(rel).parts[0] for rel in incoming | machine}) if route == "explicit" else []
        write_manifest(staged, DistributionManifest(name="legacy", distribution_owned=owned))
        target = install_distribution(str(staged), name="restored", force=True).target_dir

    for rel, content in incoming.items():
        assert (target / rel).read_text(encoding="utf-8") == content, (route, rel)
    for rel, content in machine.items():
        if route == "profile":
            assert not (target / rel).exists(), rel
        else:
            assert (target / rel).read_text(encoding="utf-8") == "target-owned runtime\n", (route, rel)
        assert (staged / rel).read_text(encoding="utf-8") == content

    if route == "explicit":
        # Windows-authored manifests use separators that must not bypass ownership.
        write_manifest(staged, DistributionManifest(
            name="legacy", distribution_owned=[rel.replace("/", "\\") for rel in incoming | machine],
        ))
        install_distribution(str(staged), name="restored", force=True)
        for rel in machine:
            assert (target / rel).read_text(encoding="utf-8") == "target-owned runtime\n", rel
    elif route == "backup":
        # An in-home traversal must not bypass the runtime-root classification.
        with zipfile.ZipFile(archive, "w") as writer:
            writer.writestr(".hermes/config.yaml", incoming["config.yaml"])
            writer.writestr(".hermes/skills/../installs/key/facts.json", "foreign generation\n")
        run_import(Namespace(zipfile=str(archive), force=True))
        assert (target / "installs/key/facts.json").read_text(encoding="utf-8") == "target-owned runtime\n"
