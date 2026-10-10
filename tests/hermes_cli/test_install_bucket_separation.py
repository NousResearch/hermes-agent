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


def _pre_pm_binary(path: Path) -> None:
    """The pre-PM family's real shape: a full ``uv`` build, or the thin ``uvx``
    launcher (own message, none of ``uv``'s markers)."""
    if "uvx" in path.name.lower():
        path.write_bytes(
            b"\x7fELF" + b"\0" * (200 << 10)
            + b"Could not determine the location of the `uvx` binary"
        )
    else:
        path.write_bytes(b"\x7fELF" + b"\0" * (2 << 20) + b"UV_CACHE_DIR")
    path.chmod(0o755)


@pytest.mark.platforms("posix")
def test_legacy_cleanup_removes_only_the_node_tree_and_preserves_bin(tmp_path):
    # The sweep owns node/ outright; the bin/uv family is identity-only evidence
    # (no receipt proves Hermes installed it), so it is PRESERVED in both
    # uninstall modes — the full wipe deletes the home, the explicit migration
    # is 'hermes doctor --fix'.
    runtime = ('bin/uv', 'bin/uv.exe', 'bin/uvx', 'bin/uvx.exe')
    user = ('bin/my-script', 'bin/tiny-uv', 'config.yaml', 'auth.json', 'SOUL.md',
            'sessions/session', 'skills/demo/SKILL.md', 'memories/MEMORY.md', 'profiles/other/config.yaml')
    for name in runtime:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        _pre_pm_binary(p)
    for name in user:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(name, encoding='utf-8')
    (tmp_path / 'node/bin').mkdir(parents=True, exist_ok=True)
    (tmp_path / 'node/bin/node').write_text('node', encoding='utf-8')
    removed = remove_legacy_runtime_trees(tmp_path)
    assert removed == [tmp_path / "node"]
    assert all((tmp_path / name).is_file() for name in runtime), "identity is not ownership"
    assert all((tmp_path / name).read_text(encoding='utf-8') == name for name in user)
    assert remove_legacy_runtime_trees(tmp_path) == []


def test_runtime_tree_sweep_never_touches_the_bin_uv_family(tmp_path, monkeypatch):
    """No uninstall mode deletes the uv family from the heuristic sweep: the
    full wipe owns the home wholesale, and keep-data preserves it."""
    from hermes_cli import legacy_uv

    (tmp_path / "node").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        legacy_uv, "remove_legacy_managed_uv",
        lambda home: (_ for _ in ()).throw(AssertionError("uninstall never calls the heuristic delete")),
    )

    assert remove_legacy_runtime_trees(tmp_path) == [tmp_path / "node"]


def test_uninstall_call_site_runs_the_runtime_tree_sweep(monkeypatch, tmp_path):
    """The production call site sweeps runtime trees in BOTH modes (node goes
    regardless); the uv family is no longer a parameter — it is simply never
    touched here."""
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
        ("remove_legacy_runtime_trees", lambda home: calls.append(home) or []),
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
    uninstall.run_uninstall(args(full=False))
    assert calls == [hermes_home, hermes_home]


@pytest.mark.platforms("posix")
def test_update_self_heal_only_notices_legacy_uv_and_never_deletes(tmp_path, monkeypatch, capsys):
    """``hermes update`` NAMES the pre-PM ``uv``/``uvx`` in every home's ``bin`` but
    never deletes it unattended: shape cannot prove Hermes installed the copy, so
    the explicit migration is ``hermes doctor --fix``.

    Gating stays: no notice while PM's store lacks uv (the legacy binary is then
    the install's only uv). The ACTIVE home is walked too, not just what
    ``list_profiles()`` enumerates — a custom ``HERMES_HOME`` is not a profile.
    """
    import pm
    from hermes_cli import update_cmd_maint

    homes = [tmp_path / "default", tmp_path / "named", tmp_path / "active-elsewhere"]
    for home in homes:
        bin_dir = home / "bin"
        bin_dir.mkdir(parents=True)
        # The pre-PM installer staged astral BINARIES — fixtures must match that
        # shape (native magic + uv's own identity marker) or they are not named.
        for name in ("uv.exe", "uvx.exe", "hermes.exe"):
            (bin_dir / name).write_bytes(b"MZ" + b"\0" * (2 << 20) + b"UV_CACHE_DIR")
    monkeypatch.setattr(
        "hermes_cli.profiles.list_profiles",
        lambda **_kw: [SimpleNamespace(path=home) for home in homes[:2]],
    )
    monkeypatch.setenv("HERMES_HOME", str(homes[2]))

    monkeypatch.setattr(pm, "is_installed", lambda name: False)
    update_cmd_maint._notice_legacy_managed_uv()
    assert capsys.readouterr().out == ""
    assert all((home / "bin" / "uv.exe").exists() for home in homes), \
        "removed the only uv this install has"

    monkeypatch.setattr(pm, "is_installed", lambda name: True)
    update_cmd_maint._notice_legacy_managed_uv()
    out = capsys.readouterr().out
    for home in homes:
        assert (home / "bin" / "uv.exe").is_file(), f"{home}'s copy was deleted by maintenance"
        assert (home / "bin" / "uvx.exe").is_file()
        assert (home / "bin" / "hermes.exe").is_file(), "the launchers are not runtime bytes"
        assert str(home) in out, f"{home} was never visited"
    assert "hermes doctor --fix" in out
    assert "cannot prove" in out, "the notice must state the provenance limit"


@pytest.mark.platforms("windows")
def test_update_self_heal_skips_the_fail_closed_platform_entirely(tmp_path, monkeypatch, capsys):
    """Windows twin: no anchored fix exists there, so the self-heal stays silent —
    the leftovers are named by doctor's manual guidance instead."""
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

    update_cmd_maint._notice_legacy_managed_uv()
    assert (home / "bin" / "uv.exe").is_file()
    assert (home / "bin" / "uvx.exe").is_file()
    assert capsys.readouterr().out == ""


@pytest.mark.platforms("windows")
def test_windows_uninstall_sweeps_node_but_preserves_the_uv_family(tmp_path):
    """Windows twin of the POSIX sweep: the identity-only bin family and every
    user file survive the runtime-tree sweep in both uninstall modes."""
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
