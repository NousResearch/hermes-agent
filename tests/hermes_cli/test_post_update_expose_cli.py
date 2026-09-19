"""expose_cli — the launcher owner repairs PATH conveniences.

The installers write ~/.local/bin/hermes* once, at install time. This
step rewrites them when they drift (moved checkout, recreated venv,
deleted by hand) and — just as load-bearing — REFUSES to touch a
launcher that belongs to a different install sharing the link dir.
Real files under a temp HOME; no mocks of the things being tested.

The wrapper-writing surface is POSIX-only by design (Windows exposure is
installer-owned), so the behaviour tests skip on win32 and the win32
gate gets its own test.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from hermes_cli import _launchers, post_update

posix_only = pytest.mark.platforms("posix")


def _write_bundled_stamp(repo_root: Path) -> None:
    """The minimal install-stamp.json this branch accepts as a bundled
    payload (payload marker + a valid updateMechanism)."""
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "install-stamp.json").write_text(
        json.dumps({"payload": "bundled", "updateMechanism": "electron-updater"}),
        encoding="utf-8",
    )


@pytest.fixture
def fake_install(tmp_path, monkeypatch):
    """A venv-shaped install root plus an isolated HOME."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    monkeypatch.setenv("HERMES_HOME", str(home))

    root = tmp_path / "checkout"
    (root / "venv" / "bin").mkdir(parents=True)
    (root / "venv" / "bin" / "python").write_text("#!fake\n", encoding="utf-8")
    (root / "hermes").write_text("# entrypoint\n", encoding="utf-8")
    (root / "run_agent.py").write_text("# agent\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(root))
    store = tmp_path / "tools"
    store.mkdir()
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(store))
    interpreter = Path(sys._base_executable).resolve()
    (store / "facts.json").write_text(json.dumps({"schema": 1, "packages": {"python": {
        "entry": str(interpreter.parent if os.name == "nt" else interpreter.parents[1]),
    }}}), encoding="utf-8")
    return home, root


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("create", [True, False])
def test_windows_is_installer_owned(create):
    assert _launchers.expose_cli(create=create) == {
        "ok": True,
        "skipped": "windows-installer-owned",
    }


def test_registered_as_a_home_step():
    assert ("expose_cli", _launchers.expose_cli) in post_update.HOME_STEPS


class TestExposeCli:
    @posix_only
    @pytest.mark.parametrize("hermes", ["missing", "foreign"])
    def test_repair_only_preserves_missing_and_foreign_commands(self, fake_install, monkeypatch, hermes):
        home, root = fake_install
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: pytest.fail("repair read config"))
        wrapper_dir = home / ".local" / "bin"
        extra = home / "bin"
        assert _launchers.expose_cli(create=False) == {"ok": True, "written": []}
        assert not wrapper_dir.exists()
        assert not extra.exists()
        assert {p.name for p in (root / ".hermes/bin").iterdir()} == set(_launchers.ENTRY_POINTS)

        # An owned ACP entry must be repaired even without an owned hermes
        # sibling admitting this directory. Its old target is already gone.
        extra.mkdir()
        dangling = root / "venv/bin/hermes-acp"
        acp = extra / "hermes-acp"
        acp.symlink_to(dangling)
        assert not acp.exists() and acp.is_symlink()
        foreign = b"#!/bin/sh\n# user-owned caf\xc3\xa9 command\nexit 19\n"
        if hermes == "foreign":
            (extra / "hermes").write_bytes(foreign)
        wrapper_dir.mkdir(parents=True)
        (wrapper_dir / "hermes").write_bytes(foreign)
        foreign_link = wrapper_dir / "hermes-acp"
        foreign_target = home / "other-install/hermes-acp"
        foreign_link.symlink_to(foreign_target)
        before = {path: path.lstat().st_mtime_ns for path in (wrapper_dir, wrapper_dir / "hermes", foreign_link)}

        assert _launchers.expose_cli(create=False) == {"ok": True, "written": ["hermes-acp"]}
        assert acp.is_file() and not acp.is_symlink() and os.access(acp, os.X_OK)
        assert not dangling.exists()
        assert not (extra / "hermes-agent").exists()
        assert not (wrapper_dir / "hermes-agent").exists()
        if hermes == "foreign":
            assert (extra / "hermes").read_bytes() == foreign
        else:
            assert not (extra / "hermes").exists()
        assert (wrapper_dir / "hermes").read_bytes() == foreign
        assert foreign_link.is_symlink() and foreign_link.readlink() == foreign_target
        assert before == {path: path.lstat().st_mtime_ns for path in before}
        assert _launchers.expose_cli(create=False) == {"ok": True, "written": []}

    @posix_only
    def test_writes_all_three_wrappers_fresh(self, fake_install):
        home, root = fake_install
        result = _launchers.expose_cli()
        assert result["ok"] is True
        assert sorted(result["written"]) == ["hermes", "hermes-acp", "hermes-agent"]
        for name in ("hermes", "hermes-agent", "hermes-acp"):
            wrapper = home / ".local" / "bin" / name
            body = wrapper.read_text(encoding="utf-8-sig")
            assert str(root) in body
            assert str(root / ".hermes" / "bin") in body
            assert os.access(wrapper, os.X_OK)



    @posix_only
    def test_leaves_another_installs_wrapper_alone(self, fake_install):
        home, root = fake_install
        other = "/somewhere/else/checkout"
        wrapper_dir = home / ".local" / "bin"
        wrapper_dir.mkdir(parents=True)
        foreign = f'#!/bin/sh\nexec "{other}/venv/bin/python" "{other}/hermes" "$@"\n'
        (wrapper_dir / "hermes").write_text(foreign, encoding="utf-8")
        result = _launchers.expose_cli()
        assert (wrapper_dir / "hermes").read_text(encoding="utf-8-sig") == foreign
        assert "hermes" not in result["written"]
        # The other two had no file at all — those ARE written.
        assert sorted(result["written"]) == ["hermes-acp", "hermes-agent"]

    @posix_only
    def test_config_gate_disables(self, fake_install, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"cli": {"expose_on_path": False}},
        )
        result = _launchers.expose_cli()
        assert result == {"ok": True, "skipped": "config-disabled"}

    @posix_only
    def test_sealed_tree_without_venv_skips(self, fake_install, monkeypatch, tmp_path):
        (tmp_path / "tools" / "facts.json").unlink()
        bare = tmp_path / "sealed"
        bare.mkdir()
        monkeypatch.setenv("HERMES_INSTALL_ROOT", str(bare))
        result = _launchers.expose_cli()
        assert result == {"ok": True, "skipped": "no-store-python"}

    @pytest.mark.parametrize("create", [
        pytest.param(True, marks=pytest.mark.platforms("linux")),
        pytest.param(False, marks=posix_only),
    ])
    def test_bundled_tree_skips_bundle_owns_launchers(
        self, fake_install, monkeypatch, tmp_path, create
    ):
        """The bundle owns its shims: on Linux (AppImage: transient mount)
        the step names the shape and writes nothing."""
        payload = tmp_path / "agent-payload"
        (payload / "bin").mkdir(parents=True)
        (payload / "repo").mkdir()
        _write_bundled_stamp(payload / "repo")
        for name in ("hermes", "hermes-agent", "hermes-acp"):
            (payload / "bin" / name).write_text("\x7fELF fake shim\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_INSTALL_ROOT", str(payload / "repo"))
        if not create:
            monkeypatch.setattr("hermes_cli.config.load_config", lambda: pytest.fail("repair read config"))
        result = _launchers.expose_cli(create=create)
        assert result == {"ok": True, "skipped": "bundle-owns-launchers"}
        assert not (fake_install[0] / ".local/bin").exists()

    @posix_only
    def test_unstamped_tree_with_sibling_bin_is_not_a_bundle(
        self, fake_install, monkeypatch, tmp_path
    ):
        """The stamp is the shape authority. A venv-less checkout whose
        PARENT happens to carry a bin/hermes (the installers' launcher
        dir shares ~/.hermes with the checkout) must skip, not enter the
        sealed branch — on every platform."""
        (tmp_path / "tools" / "facts.json").unlink()
        parent = tmp_path / "hermes-home"
        (parent / "bin").mkdir(parents=True)
        (parent / "bin" / "hermes").write_text("#!/bin/sh\n# installer launcher\n", encoding="utf-8")
        checkout = parent / "hermes-agent"
        checkout.mkdir()
        monkeypatch.setenv("HERMES_INSTALL_ROOT", str(checkout))
        result = _launchers.expose_cli()
        assert result == {"ok": True, "skipped": "no-store-python"}
        assert _launchers._is_bundled_payload(checkout) is False

    @posix_only
    def test_replaces_a_dangling_symlink_from_old_installs(self, fake_install):
        """#21454: `cat >` used to follow an old symlink into the venv and
        clobber the console script. The step must unlink FIRST."""
        home, root = fake_install
        wrapper_dir = home / ".local" / "bin"
        wrapper_dir.mkdir(parents=True)
        console_script = root / "venv" / "bin" / "hermes"
        console_script.write_text("# real console script\n", encoding="utf-8")
        (wrapper_dir / "hermes").symlink_to(console_script)
        result = _launchers.expose_cli()
        assert "hermes" in result["written"]
        # The venv console script survives untouched…
        assert console_script.read_text(encoding="utf-8-sig") == "# real console script\n"
        # …and the link-dir entry is now a real file, not a symlink.
        assert not (wrapper_dir / "hermes").is_symlink()


@pytest.mark.platforms("macos")
def test_direct_packaged_cli_exposes_shims_before_electron(tmp_path):
    import subprocess
    import shlex

    home = tmp_path / "home"
    home.mkdir()
    payload = tmp_path / "Hermes.app/Contents/Resources/agent-payload"
    repo = payload / "repo"
    _write_bundled_stamp(repo)
    (repo / "install-stamp.json").write_text(json.dumps({
        "payload": "bundled", "commit": "abcdef012345", "updateMechanism": "electron-updater",
    }), encoding="utf-8")
    bin_dir = payload / "bin"
    bin_dir.mkdir()
    code = f"import sys; sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r}); from hermes_cli.main import main; main()"
    for name in ("hermes", "hermes-agent", "hermes-acp"):
        shim = bin_dir / name
        shim.write_text(f'#!/bin/sh\nexec {shlex.join([sys.executable, "-I", "-c", code])} "$@"\n', encoding="utf-8")
        shim.chmod(0o755)
    env = dict(os.environ, HOME=str(home), HERMES_HOME=str(home / ".hermes"),
               HERMES_INSTALL_ROOT=str(repo), HERMES_RUNTIME_DIR=str(tmp_path / "tools"))
    # Execute the package CLI directly. No Electron process or linking helper runs.
    result = subprocess.run([str(bin_dir / "hermes"), "--version"], env=env,
                            capture_output=True, text=True, timeout=30, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    for name in ("hermes", "hermes-agent", "hermes-acp"):
        assert (home / ".local/bin" / name).is_symlink()
        assert (home / ".local/bin" / name).resolve() == bin_dir / name


@posix_only
class TestSymlinkSealedLaunchers:
    """The macOS sealed-bundle exposure helper, tested directly — the
    symlink/ownership logic is platform-free; only its call site in
    expose_cli is darwin-gated."""

    @pytest.fixture
    def payload(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
        payload_bin = tmp_path / "Hermes.app" / "Contents" / "Resources" / "agent-payload" / "bin"
        payload_bin.mkdir(parents=True)
        for name in ("hermes", "hermes-agent", "hermes-acp"):
            (payload_bin / name).write_text("fake mach-o shim\n", encoding="utf-8")
        return home, payload_bin

    def test_links_all_three_fresh(self, payload):
        home, payload_bin = payload
        result = _launchers._symlink_sealed_launchers(payload_bin)
        assert result["ok"] is True
        assert sorted(result["written"]) == ["hermes", "hermes-acp", "hermes-agent"]
        for name in ("hermes", "hermes-agent", "hermes-acp"):
            link = home / ".local" / "bin" / name
            assert link.is_symlink()
            assert os.readlink(link) == str(payload_bin / name)

    def test_second_run_is_a_no_op(self, payload):
        _, payload_bin = payload
        _launchers._symlink_sealed_launchers(payload_bin)
        result = _launchers._symlink_sealed_launchers(payload_bin)
        assert result["written"] == []

    def test_retargets_own_link_after_app_moved(self, payload, tmp_path):
        """An app update / move leaves ~/.local/bin pointing at the old
        bundle path INSIDE this payload tree — that link is ours; retarget."""
        home, payload_bin = payload
        old = payload_bin.parent / "bin-old"
        link_dir = home / ".local" / "bin"
        link_dir.mkdir(parents=True)
        (link_dir / "hermes").symlink_to(old / "hermes")  # dangling, old payload path
        result = _launchers._symlink_sealed_launchers(payload_bin)
        assert "hermes" in result["written"]
        assert os.readlink(link_dir / "hermes") == str(payload_bin / "hermes")

    def test_never_touches_a_live_foreign_entry(self, payload, tmp_path):
        home, payload_bin = payload
        link_dir = home / ".local" / "bin"
        link_dir.mkdir(parents=True)
        # A real file (pipx-style launcher)…
        (link_dir / "hermes").write_text("#!/bin/sh\n# pipx launcher\n", encoding="utf-8")
        # …and a live symlink to a different tool.
        other = tmp_path / "other-tool"
        other.write_text("other\n", encoding="utf-8")
        (link_dir / "hermes-agent").symlink_to(other)
        result = _launchers._symlink_sealed_launchers(payload_bin)
        assert (link_dir / "hermes").read_text(encoding="utf-8-sig") == "#!/bin/sh\n# pipx launcher\n"
        assert os.readlink(link_dir / "hermes-agent") == str(other)
        assert sorted(result["written"]) == ["hermes-acp"]
