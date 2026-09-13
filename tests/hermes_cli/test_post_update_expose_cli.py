"""step_expose_cli — the post-update side owns launcher-wrapper repair.

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

from hermes_cli import post_update

posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="wrapper exposure is POSIX-only; Windows is installer-owned"
)


def _write_bundled_stamp(repo_root: Path) -> None:
    """The minimal install-stamp.json this branch accepts as a bundled
    payload (payload marker + a valid updateMechanism)."""
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "install-stamp.json").write_text(
        json.dumps({"payload": "bundled", "updateMechanism": "electron-updater"})
    )


@pytest.fixture
def fake_install(tmp_path, monkeypatch):
    """A venv-shaped install root plus an isolated HOME."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))

    root = tmp_path / "checkout"
    (root / "venv" / "bin").mkdir(parents=True)
    (root / "venv" / "bin" / "python").write_text("#!fake\n")
    (root / "hermes").write_text("# entrypoint\n")
    (root / "run_agent.py").write_text("# agent\n")
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(root))
    return home, root


def test_windows_is_installer_owned(monkeypatch):
    if sys.platform != "win32":
        monkeypatch.setattr(post_update.sys, "platform", "win32")
    assert post_update.step_expose_cli() == {
        "ok": True,
        "skipped": "windows-installer-owned",
    }


def test_registered_as_a_home_step():
    assert ("expose_cli", post_update.step_expose_cli) in post_update.HOME_STEPS


@posix_only
class TestExposeCli:
    def test_writes_all_three_wrappers_fresh(self, fake_install):
        home, root = fake_install
        result = post_update.step_expose_cli()
        assert result["ok"] is True
        assert sorted(result["written"]) == ["hermes", "hermes-acp", "hermes-agent"]
        for name in ("hermes", "hermes-agent", "hermes-acp"):
            wrapper = home / ".local" / "bin" / name
            body = wrapper.read_text()
            assert str(root) in body
            assert "PYTHONPATH" in body
            assert os.access(wrapper, os.X_OK)

    def test_second_run_is_a_no_op(self, fake_install):
        post_update.step_expose_cli()
        result = post_update.step_expose_cli()
        assert result == {"ok": True, "written": []}

    def test_repairs_a_stale_same_install_wrapper(self, fake_install):
        home, root = fake_install
        post_update.step_expose_cli()
        wrapper = home / ".local" / "bin" / "hermes"
        wrapper.write_text(f'#!/bin/sh\nexec "{root}/venv/bin/python" OLD-SHAPE\n')
        result = post_update.step_expose_cli()
        assert "hermes" in result["written"]
        assert "OLD-SHAPE" not in wrapper.read_text()

    def test_leaves_another_installs_wrapper_alone(self, fake_install):
        home, root = fake_install
        other = "/somewhere/else/checkout"
        wrapper_dir = home / ".local" / "bin"
        wrapper_dir.mkdir(parents=True)
        foreign = f'#!/bin/sh\nexec "{other}/venv/bin/python" "{other}/hermes" "$@"\n'
        (wrapper_dir / "hermes").write_text(foreign)
        result = post_update.step_expose_cli()
        assert (wrapper_dir / "hermes").read_text() == foreign
        assert "hermes" not in result["written"]
        # The other two had no file at all — those ARE written.
        assert sorted(result["written"]) == ["hermes-acp", "hermes-agent"]

    def test_config_gate_disables(self, fake_install, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"cli": {"expose_on_path": False}},
        )
        result = post_update.step_expose_cli()
        assert result == {"ok": True, "skipped": "config-disabled"}

    def test_sealed_tree_without_venv_skips(self, fake_install, monkeypatch, tmp_path):
        bare = tmp_path / "sealed"
        bare.mkdir()
        monkeypatch.setenv("HERMES_INSTALL_ROOT", str(bare))
        result = post_update.step_expose_cli()
        assert result == {"ok": True, "skipped": "no-venv-layout"}

    @pytest.mark.skipif(sys.platform == "darwin", reason="darwin takes the symlink branch")
    def test_bundled_tree_skips_bundle_owns_launchers(
        self, fake_install, monkeypatch, tmp_path
    ):
        """The bundle owns its shims: on Linux (AppImage: transient mount)
        the step names the shape and writes nothing."""
        payload = tmp_path / "agent-payload"
        (payload / "bin").mkdir(parents=True)
        (payload / "repo").mkdir()
        _write_bundled_stamp(payload / "repo")
        for name in ("hermes", "hermes-agent", "hermes-acp"):
            (payload / "bin" / name).write_text("\x7fELF fake shim\n")
        monkeypatch.setenv("HERMES_INSTALL_ROOT", str(payload / "repo"))
        result = post_update.step_expose_cli()
        assert result == {"ok": True, "skipped": "bundle-owns-launchers"}

    def test_unstamped_tree_with_sibling_bin_is_not_a_bundle(
        self, fake_install, monkeypatch, tmp_path
    ):
        """The stamp is the shape authority. A venv-less checkout whose
        PARENT happens to carry a bin/hermes (the installers' launcher
        dir shares ~/.hermes with the checkout) must skip, not enter the
        sealed branch — on every platform."""
        parent = tmp_path / "hermes-home"
        (parent / "bin").mkdir(parents=True)
        (parent / "bin" / "hermes").write_text("#!/bin/sh\n# installer launcher\n")
        checkout = parent / "hermes-agent"
        checkout.mkdir()
        monkeypatch.setenv("HERMES_INSTALL_ROOT", str(checkout))
        result = post_update.step_expose_cli()
        assert result == {"ok": True, "skipped": "no-venv-layout"}
        assert post_update._is_bundled_payload(checkout) is False

    def test_replaces_a_dangling_symlink_from_old_installs(self, fake_install):
        """#21454: `cat >` used to follow an old symlink into the venv and
        clobber the console script. The step must unlink FIRST."""
        home, root = fake_install
        wrapper_dir = home / ".local" / "bin"
        wrapper_dir.mkdir(parents=True)
        console_script = root / "venv" / "bin" / "hermes"
        console_script.write_text("# real console script\n")
        (wrapper_dir / "hermes").symlink_to(console_script)
        result = post_update.step_expose_cli()
        assert "hermes" in result["written"]
        # The venv console script survives untouched…
        assert console_script.read_text() == "# real console script\n"
        # …and the link-dir entry is now a real file, not a symlink.
        assert not (wrapper_dir / "hermes").is_symlink()


@posix_only
class TestSymlinkSealedLaunchers:
    """The macOS sealed-bundle exposure helper, tested directly — the
    symlink/ownership logic is platform-free; only its call site in
    step_expose_cli is darwin-gated."""

    @pytest.fixture
    def payload(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
        payload_bin = tmp_path / "Hermes.app" / "Contents" / "Resources" / "agent-payload" / "bin"
        payload_bin.mkdir(parents=True)
        for name in ("hermes", "hermes-agent", "hermes-acp"):
            (payload_bin / name).write_text("fake mach-o shim\n")
        return home, payload_bin

    def test_links_all_three_fresh(self, payload):
        home, payload_bin = payload
        result = post_update._symlink_sealed_launchers(payload_bin)
        assert result["ok"] is True
        assert sorted(result["written"]) == ["hermes", "hermes-acp", "hermes-agent"]
        for name in ("hermes", "hermes-agent", "hermes-acp"):
            link = home / ".local" / "bin" / name
            assert link.is_symlink()
            assert os.readlink(link) == str(payload_bin / name)

    def test_second_run_is_a_no_op(self, payload):
        _, payload_bin = payload
        post_update._symlink_sealed_launchers(payload_bin)
        result = post_update._symlink_sealed_launchers(payload_bin)
        assert result["written"] == []

    def test_retargets_own_link_after_app_moved(self, payload, tmp_path):
        """An app update / move leaves ~/.local/bin pointing at the old
        bundle path INSIDE this payload tree — that link is ours; retarget."""
        home, payload_bin = payload
        old = payload_bin.parent / "bin-old"
        link_dir = home / ".local" / "bin"
        link_dir.mkdir(parents=True)
        (link_dir / "hermes").symlink_to(old / "hermes")  # dangling, old payload path
        result = post_update._symlink_sealed_launchers(payload_bin)
        assert "hermes" in result["written"]
        assert os.readlink(link_dir / "hermes") == str(payload_bin / "hermes")

    def test_never_touches_a_live_foreign_entry(self, payload, tmp_path):
        home, payload_bin = payload
        link_dir = home / ".local" / "bin"
        link_dir.mkdir(parents=True)
        # A real file (pipx-style launcher)…
        (link_dir / "hermes").write_text("#!/bin/sh\n# pipx launcher\n")
        # …and a live symlink to a different tool.
        other = tmp_path / "other-tool"
        other.write_text("other\n")
        (link_dir / "hermes-agent").symlink_to(other)
        result = post_update._symlink_sealed_launchers(payload_bin)
        assert (link_dir / "hermes").read_text() == "#!/bin/sh\n# pipx launcher\n"
        assert os.readlink(link_dir / "hermes-agent") == str(other)
        assert sorted(result["written"]) == ["hermes-acp"]
