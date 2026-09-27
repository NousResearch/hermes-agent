"""Uninstall and profile cloning respect the install/profile bucket split.

Runtime artifacts belong to an install; config, sessions and skills belong
to a profile. Uninstall removes the former (in either mode — they are not
data); profile clone/export never copies them.
"""

from pathlib import Path
from types import SimpleNamespace
import tarfile

import pytest

from hermes_cli.uninstall import remove_legacy_runtime_trees


@pytest.mark.platforms("posix")
def test_legacy_cleanup_removes_only_runtime_bytes_and_is_idempotent(tmp_path):
    # The pre-PM installer staged astral standalone BINARIES — the cleanup's
    # ownership gate is shape+size, so fixtures must match that shape, and a
    # small user file named ``uv`` must survive it. POSIX-only: automatic
    # legacy-uv cleanup fails closed elsewhere (the Windows twin below).
    runtime = ('bin/uv', 'bin/uv.exe', 'bin/uvx', 'bin/uvx.exe')
    user = ('bin/my-script', 'bin/tiny-uv', 'config.yaml', 'auth.json', 'SOUL.md',
            'sessions/session', 'skills/demo/SKILL.md', 'memories/MEMORY.md', 'profiles/other/config.yaml')
    for name in runtime:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\x7fELF" + b"\0" * (2 << 20))
    for name in user:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(name, encoding='utf-8')
    (tmp_path / 'node/bin').mkdir(parents=True, exist_ok=True)
    (tmp_path / 'node/bin/node').write_text('node', encoding='utf-8')
    removed = remove_legacy_runtime_trees(tmp_path)
    assert set(removed) == {tmp_path / 'node'} | {tmp_path / name for name in runtime}
    assert all(not (tmp_path / name).exists() for name in runtime)
    assert all((tmp_path / name).read_text(encoding='utf-8') == name for name in user)
    assert remove_legacy_runtime_trees(tmp_path) == []


def test_full_uninstall_defers_the_fail_closed_uv_cleanup_to_the_wipe(tmp_path, monkeypatch):
    """``defer_uv_to_wipe`` keeps the manual-removal warning out of a full wipe.

    Step 5 of ``_perform_uninstall`` deletes the whole home right after step 4c;
    on Windows the shared cleanup fails closed (POSIX anchor unavailable) and
    would print "remove the uv family manually" for files the wipe is about to
    delete — so a full uninstall defers and lets the home removal own them.
    Keep-data uninstalls still run the cleanup and keep that instruction.
    """
    from hermes_cli import legacy_uv

    (tmp_path / "node").mkdir(parents=True, exist_ok=True)
    calls: list = []
    monkeypatch.setattr(
        legacy_uv, "remove_legacy_managed_uv", lambda home: calls.append(home) or []
    )

    removed = remove_legacy_runtime_trees(tmp_path, defer_uv_to_wipe=True)
    assert removed == [tmp_path / "node"]
    assert calls == []  # the wipe owns the uv family: no early manual instruction

    remove_legacy_runtime_trees(tmp_path)
    assert calls == [tmp_path]  # keep-data (and POSIX) still run the shared cleanup


def test_full_uninstall_defers_uv_cleanup_via_the_call_site(monkeypatch, tmp_path):
    """The production call site wires ``windows and full_uninstall`` into ``defer_uv_to_wipe``.

    Driven through ``run_uninstall`` with ``_is_windows`` forced, so the wiring
    itself runs rather than being re-stated: a Windows full wipe must not ask
    for a manual uv removal the home removal is about to perform (step 5 deletes
    the whole home right after step 4c), and keep-data still keeps the shared
    cleanup active.
    """
    from hermes_cli import uninstall

    project_root = tmp_path / "hermes-agent"
    hermes_home = tmp_path / ".hermes"
    project_root.mkdir()
    (project_root / ".git").mkdir()  # marks a removable git checkout (kind gate)
    hermes_home.mkdir()

    calls: list = []
    for attr, value in (
        ("get_project_root", lambda: project_root),
        ("_is_windows", lambda: True),
        ("_is_default_hermes_home", lambda home: False),
        ("_discover_named_profiles", lambda: []),
        ("_refuse_if_steward_owned", lambda: None),
        ("uninstall_gateway_service", lambda: True),
        ("remove_path_from_shell_configs", lambda: []),
        ("remove_path_from_windows_registry", lambda *a, **kw: []),
        ("remove_hermes_env_vars_windows", lambda: []),
        ("remove_wrapper_script", lambda: []),
        ("remove_windows_bin_launchers", lambda **kw: []),
        ("remove_node_symlinks", lambda home: []),
        ("remove_portable_tooling_windows", lambda home: []),
        ("remove_legacy_runtime_trees", lambda home, **kw: calls.append((home, kw)) or []),
    ):
        monkeypatch.setattr(uninstall, attr, value)
    monkeypatch.setattr("hermes_cli.gui_uninstall.uninstall_gui", lambda home, **kw: True)
    monkeypatch.setattr(
        uninstall, "_rmtree_step",
        lambda path, **kw: None if path in (project_root, hermes_home)
        else (_ for _ in ()).throw(AssertionError(f"unexpected rmtree {path}")),
    )
    monkeypatch.setattr(uninstall, "get_hermes_home", lambda: hermes_home)

    def args(full: bool):
        from types import SimpleNamespace
        return SimpleNamespace(dry_run=False, yes=True, full=full, data=False, gui=False)

    uninstall.run_uninstall(args(full=True))
    assert calls == [(hermes_home, {"defer_uv_to_wipe": True})]

    calls.clear()
    uninstall.run_uninstall(args(full=False))
    assert calls == [(hermes_home, {"defer_uv_to_wipe": False})]


@pytest.mark.platforms("posix")
def test_update_self_heal_purges_legacy_uv_only_once_the_store_has_its_own(tmp_path, monkeypatch):
    """``hermes update`` drops the pre-PM ``uv``/``uvx`` from every home's ``bin``.

    The pre-PM resolver was profile-scoped, so each home can hold a copy. The
    removal must stay gated on PM's store already carrying uv — while it does
    not, that binary is the install's only uv (#101269).

    The ACTIVE home is purged too, not just what ``list_profiles()`` enumerates:
    a ``HERMES_HOME`` outside the default and ``profiles/`` is not a profile, so
    a profile-only walk left its leftover shadowing the box forever.
    """
    import pm
    from hermes_cli import update_cmd_maint

    homes = [tmp_path / "default", tmp_path / "named", tmp_path / "active-elsewhere"]
    for home in homes:
        bin_dir = home / "bin"
        bin_dir.mkdir(parents=True)
        # The pre-PM installer staged astral BINARIES — fixtures must match that
        # shape or the ownership gate skips them, which is the point of it.
        for name in ("uv.exe", "uvx.exe", "hermes.exe"):
            (bin_dir / name).write_bytes(b"MZ" + b"\0" * (2 << 20))
    monkeypatch.setattr(
        "hermes_cli.profiles.list_profiles",
        lambda **_kw: [SimpleNamespace(path=home) for home in homes[:2]],
    )
    monkeypatch.setenv("HERMES_HOME", str(homes[2]))

    monkeypatch.setattr(pm, "is_installed", lambda name: False)
    update_cmd_maint._purge_legacy_managed_uv()
    assert all((home / "bin" / "uv.exe").exists() for home in homes), \
        "removed the only uv this install has"

    monkeypatch.setattr(pm, "is_installed", lambda name: True)
    update_cmd_maint._purge_legacy_managed_uv()
    for home in homes:
        assert not (home / "bin" / "uv.exe").exists(), f"{home} was never visited"
        assert not (home / "bin" / "uvx.exe").exists()
        assert (home / "bin" / "hermes.exe").exists(), "the launchers are not runtime bytes"


@pytest.mark.platforms("windows")
def test_update_self_heal_skips_the_fail_closed_platform_entirely(tmp_path, monkeypatch, capsys):
    """Windows twin: no retained deletion anchor means the self-heal never calls the
    cleanup at all — the leftovers stay for the doctor's manual step, with none of
    the per-profile warn noise a doomed removal attempt would print."""
    from hermes_cli import update_cmd_maint

    home = tmp_path / "default"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    for name in ("uv.exe", "uvx.exe"):
        (bin_dir / name).write_bytes(b"MZ" + b"\0" * (2 << 20))
    monkeypatch.setattr(
        "hermes_cli.profiles.list_profiles",
        lambda **_kw: [SimpleNamespace(path=home)],
    )

    update_cmd_maint._purge_legacy_managed_uv()
    assert (home / "bin" / "uv.exe").is_file()
    assert (home / "bin" / "uvx.exe").is_file()
    assert "Skipping" not in capsys.readouterr().out


@pytest.mark.platforms("windows")
def test_windows_uninstall_sweeps_node_but_fails_closed_on_the_uv_family(tmp_path, capsys):
    """Windows twin of the POSIX sweep. Windows has no retained deletion anchor, so
    the shared cleanup fails closed by design: ``node/`` (wholly installer-owned)
    goes, the ``bin/uv*.exe`` family and every user file survive, and the call
    names the manual step instead of deleting through a mutable path."""
    runtime = ('bin/uv', 'bin/uv.exe', 'bin/uvx', 'bin/uvx.exe')
    user = ('bin/my-script', 'config.yaml', 'auth.json')
    for name in runtime:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"MZ" + b"\0" * (2 << 20))
    for name in user:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(name, encoding='utf-8')
    (tmp_path / 'node/bin').mkdir(parents=True, exist_ok=True)
    (tmp_path / 'node/bin/node.exe').write_text('node', encoding='utf-8')

    assert remove_legacy_runtime_trees(tmp_path) == [tmp_path / 'node']
    assert all((tmp_path / name).read_bytes().startswith(b"MZ") for name in runtime)
    assert all((tmp_path / name).read_text(encoding='utf-8') == name for name in user)
    assert "manual" in capsys.readouterr().out, "the skip must name the manual step"


class TestProfileCopyExclusions:
    @pytest.mark.parametrize("operation", ["clone", "export", "distribution"])
    def test_copies_profile_payload_without_install_artifacts(self, tmp_path, monkeypatch, operation):
        from hermes_cli import profiles, profile_distribution

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(profiles, "_maybe_register_gateway_service", lambda name: None)
        kept = {"config.yaml": "model: {}\n", "SOUL.md": "profile identity\n",
                "skills/demo/SKILL.md": "demo instructions\n"}
        if operation != "distribution":
            kept["memories/MEMORY.md"] = "profile memory\n"
        excluded = (".hermes-runtime", "node", "hermes-agent", "profiles")
        for rel, content in kept.items():
            path = home / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        for name in excluded:
            path = home / name / "must-not-copy"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("install state", encoding="utf-8")

        if operation == "clone":
            target = profiles.create_profile("clone", clone_from="default", clone_all=True, no_alias=True)
        elif operation == "distribution":
            profile_distribution.write_manifest(home, profile_distribution.DistributionManifest(name="copy", version="1.0.0"))
            profile_distribution.install_distribution(str(home), name="copy", create_alias=False)
            target = profiles.get_profile_dir("copy")
        else:
            archive = profiles.export_profile("default", str(tmp_path / "profile.tar.gz"))
            with tarfile.open(archive) as bundle:
                for rel, content in kept.items():
                    payload = bundle.extractfile(f"default/{rel}")
                    assert payload is not None, rel
                    assert payload.read().decode() == content
                roots = {name.split("/")[1] for name in bundle.getnames() if "/" in name}
                assert not roots.intersection(excluded)
            return

        for rel, content in kept.items():
            assert (target / rel).read_text(encoding="utf-8") == content
        for name in excluded:
            assert not (target / name).exists(), name
