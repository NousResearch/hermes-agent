"""`hermes desktop` must not report up to date when the INSTALLED app is behind.

A failed stage-and-swap leaves the source-content build stamp current while the
packaged app in ``release/`` still holds an older commit. The content hash
describes the source tree, not the output, so it can't see the gap — the app
silently runs weeks behind. `_desktop_build_needed()` now also treats an
installed-stamp commit that diverges from HEAD as a rebuild trigger (#107542).
"""

import json

import hermes_cli.main_desktop as md
import hermes_cli.version_info as version_info

SHA_A = "a" * 40
SHA_B = "b" * 40
FALLBACK = "0" * 40


def _install_app(tmp_path, platform, *, stamp=None):
    """Build a fake unpacked app for *platform* and return (exe_path, resources_dir)."""
    if platform == "darwin":
        exe = tmp_path / "release" / "mac" / "Hermes.app" / "Contents" / "MacOS" / "Hermes"
        resources = exe.parent.parent / "Resources"
    elif platform == "win32":
        exe = tmp_path / "release" / "win-unpacked" / "Hermes.exe"
        resources = exe.parent / "resources"
    else:
        exe = tmp_path / "release" / "linux-unpacked" / "hermes"
        resources = exe.parent / "resources"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_text("", encoding="utf-8")
    resources.mkdir(parents=True, exist_ok=True)
    if stamp is not None:
        (resources / "install-stamp.json").write_text(json.dumps(stamp), encoding="utf-8")
    return exe, resources


class TestInstalledDesktopCommit:
    def test_reads_commit_for_each_platform(self, tmp_path, monkeypatch):
        for platform in ("darwin", "win32", "linux"):
            base = tmp_path / platform
            exe, _ = _install_app(base, platform, stamp={"schemaVersion": 1, "commit": SHA_A})
            monkeypatch.setattr(md.sys, "platform", platform)
            monkeypatch.setattr(md, "_desktop_packaged_executable", lambda _d, exe=exe: exe)
            assert md._installed_desktop_commit(base) == SHA_A

    def test_missing_stamp_returns_none(self, tmp_path, monkeypatch):
        exe, _ = _install_app(tmp_path, "linux", stamp=None)
        monkeypatch.setattr(md.sys, "platform", "linux")
        monkeypatch.setattr(md, "_desktop_packaged_executable", lambda _d: exe)
        assert md._installed_desktop_commit(tmp_path) is None

    def test_fallback_commit_is_not_a_signal(self, tmp_path, monkeypatch):
        exe, _ = _install_app(tmp_path, "linux", stamp={"commit": FALLBACK})
        monkeypatch.setattr(md.sys, "platform", "linux")
        monkeypatch.setattr(md, "_desktop_packaged_executable", lambda _d: exe)
        assert md._installed_desktop_commit(tmp_path) is None

    def test_malformed_stamp_returns_none(self, tmp_path, monkeypatch):
        exe, resources = _install_app(tmp_path, "linux", stamp=None)
        (resources / "install-stamp.json").write_text("{not json", encoding="utf-8")
        monkeypatch.setattr(md.sys, "platform", "linux")
        monkeypatch.setattr(md, "_desktop_packaged_executable", lambda _d: exe)
        assert md._installed_desktop_commit(tmp_path) is None

    def test_bom_prefixed_stamp_still_parses(self, tmp_path, monkeypatch):
        # Windows tooling (PowerShell Set-Content/Out-File) BOMs files it touches;
        # the read must tolerate that (reads utf-8-sig, per repo policy).
        exe, resources = _install_app(tmp_path, "linux", stamp=None)
        (resources / "install-stamp.json").write_bytes(json.dumps({"commit": SHA_A}).encode("utf-8-sig"))
        monkeypatch.setattr(md.sys, "platform", "linux")
        monkeypatch.setattr(md, "_desktop_packaged_executable", lambda _d: exe)
        assert md._installed_desktop_commit(tmp_path) == SHA_A

    def test_no_packaged_exe_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(md, "_desktop_packaged_executable", lambda _d: None)
        assert md._installed_desktop_commit(tmp_path) is None


class TestInstalledStampBehindHead:
    def test_true_when_installed_differs_from_head(self, tmp_path, monkeypatch):
        monkeypatch.setattr(md, "_installed_desktop_commit", lambda _d: SHA_A)
        monkeypatch.setattr(version_info, "_run_git", lambda _p, *a: SHA_B)
        assert md._installed_desktop_stamp_behind_head(tmp_path, tmp_path) is True

    def test_false_when_installed_matches_head(self, tmp_path, monkeypatch):
        monkeypatch.setattr(md, "_installed_desktop_commit", lambda _d: SHA_A)
        monkeypatch.setattr(version_info, "_run_git", lambda _p, *a: SHA_A)
        assert md._installed_desktop_stamp_behind_head(tmp_path, tmp_path) is False

    def test_false_when_no_installed_commit(self, tmp_path, monkeypatch):
        monkeypatch.setattr(md, "_installed_desktop_commit", lambda _d: None)
        monkeypatch.setattr(version_info, "_run_git", lambda _p, *a: SHA_B)
        assert md._installed_desktop_stamp_behind_head(tmp_path, tmp_path) is False

    def test_false_when_head_unresolvable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(md, "_installed_desktop_commit", lambda _d: SHA_A)
        monkeypatch.setattr(version_info, "_run_git", lambda _p, *a: None)
        assert md._installed_desktop_stamp_behind_head(tmp_path, tmp_path) is False


class TestBuildNeededHonorsInstalledStamp:
    def _isolate(self, monkeypatch, tmp_path):
        # Packaged mode present, intact bundle, source content current — so only
        # the installed-stamp path can flip the verdict.
        monkeypatch.setattr(md, "_desktop_packaged_executable", lambda _d: object())
        monkeypatch.setattr(md, "_renderer_bundle_dir", lambda _d, source_mode: tmp_path / "dist")
        monkeypatch.setattr(md, "_renderer_bundle_torn", lambda _d: False)
        monkeypatch.setattr(md, "_packaged_node_pty_missing", lambda _d: False)
        import hermes_cli.source_build as source_build
        monkeypatch.setattr(source_build, "source_product_current", lambda *a, **k: True)

    def test_rebuild_when_installed_behind_even_if_content_current(self, tmp_path, monkeypatch):
        self._isolate(monkeypatch, tmp_path)
        monkeypatch.setattr(md, "_installed_desktop_stamp_behind_head", lambda _d, _p: True)
        assert md._desktop_build_needed(tmp_path, tmp_path, source_mode=False) is True

    def test_up_to_date_when_installed_current_and_content_current(self, tmp_path, monkeypatch):
        self._isolate(monkeypatch, tmp_path)
        monkeypatch.setattr(md, "_installed_desktop_stamp_behind_head", lambda _d, _p: False)
        assert md._desktop_build_needed(tmp_path, tmp_path, source_mode=False) is False

    def test_source_mode_ignores_installed_stamp(self, tmp_path, monkeypatch):
        self._isolate(monkeypatch, tmp_path)
        # would flip the verdict if consulted — it must not be in source mode.
        called = []
        monkeypatch.setattr(md, "_installed_desktop_stamp_behind_head",
                            lambda _d, _p: called.append(1) or True)
        monkeypatch.setattr(md, "_desktop_dist_exists", lambda _d: True)
        assert md._desktop_build_needed(tmp_path, tmp_path, source_mode=True) is False
        assert called == []
