from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from hermes_cli import main as cli_main
from hermes_cli import termux_desktop


def _completed(code: int = 0):
    return SimpleNamespace(returncode=code)


def test_ensure_x11_packages_is_noop_when_runtime_is_present():
    calls: list[list[str]] = []
    bins = {
        "pkg": "/data/data/com.termux/files/usr/bin/pkg",
        "termux-x11": "/data/data/com.termux/files/usr/bin/termux-x11",
        "chromium-browser": "/data/data/com.termux/files/usr/bin/chromium-browser",
    }

    runtime = termux_desktop.ensure_termux_x11_packages(
        run=lambda argv, **kwargs: calls.append(list(argv)) or _completed(),
        which=bins.get,
        env={},
    )

    assert runtime.display == ":1"
    assert runtime.browser.endswith("chromium-browser")
    assert calls == []


def test_ensure_x11_packages_enables_repo_and_installs_missing_runtime():
    calls: list[list[str]] = []
    bins = {"pkg": "/termux/bin/pkg"}

    def run(argv, **kwargs):
        calls.append(list(argv))
        if argv[:3] == ["/termux/bin/pkg", "install", "-y"]:
            if "termux-x11-nightly" in argv:
                bins["termux-x11"] = "/termux/bin/termux-x11"
            if "chromium" in argv:
                bins["chromium-browser"] = "/termux/bin/chromium-browser"
        return _completed()

    runtime = termux_desktop.ensure_termux_x11_packages(
        run=run,
        which=bins.get,
        env={},
    )

    assert calls == [
        ["/termux/bin/pkg", "install", "-y", "x11-repo"],
        [
            "/termux/bin/pkg",
            "install",
            "-y",
            "termux-x11-nightly",
            "chromium",
        ],
    ]
    assert runtime.x11 == "/termux/bin/termux-x11"
    assert runtime.browser == "/termux/bin/chromium-browser"


def test_chromium_spec_uses_app_mode_without_disabling_sandbox():
    runtime = termux_desktop.TermuxDesktopRuntime(
        browser="/termux/bin/chromium-browser",
        display=":1",
        x11="/termux/bin/termux-x11",
    )
    spec = termux_desktop.chromium_browser_spec(runtime)
    assert "--app=%s" in spec
    assert "--no-sandbox" not in spec


def test_android_companion_probe_uses_package_manager():
    calls: list[list[str]] = []

    assert termux_desktop.termux_x11_android_app_installed(
        run=lambda argv, **kwargs: calls.append(list(argv)) or _completed(),
        which=lambda name: "/system/bin/pm" if name == "pm" else None,
    )
    assert calls == [["/system/bin/pm", "path", "com.termux.x11"]]


def test_display_ready_uses_termux_tmpdir(tmp_path: Path):
    socket_dir = tmp_path / ".X11-unix"
    socket_dir.mkdir()
    (socket_dir / "X1").touch()
    assert termux_desktop.termux_x11_display_ready(
        env={"TMPDIR": str(tmp_path)}, display=":1"
    )
    assert not termux_desktop.termux_x11_display_ready(
        env={"TMPDIR": str(tmp_path)}, display=":2"
    )


def test_existing_x11_display_is_reused_and_foregrounded(tmp_path: Path):
    socket_dir = tmp_path / ".X11-unix"
    socket_dir.mkdir()
    (socket_dir / "X1").touch()
    calls: list[list[str]] = []
    popen_calls: list[list[str]] = []
    runtime = termux_desktop.TermuxDesktopRuntime(
        browser="/termux/bin/chromium-browser",
        display=":1",
        x11="/termux/bin/termux-x11",
    )

    result = termux_desktop.launch_termux_x11(
        runtime,
        env={"TMPDIR": str(tmp_path)},
        popen=lambda argv, **kwargs: popen_calls.append(list(argv)),
        run=lambda argv, **kwargs: calls.append(list(argv)) or _completed(),
        which=lambda name: "/system/bin/am" if name == "am" else None,
    )

    assert result is None
    assert popen_calls == []
    assert calls == [
        [
            "/system/bin/am",
            "start",
            "--user",
            "0",
            "-n",
            "com.termux.x11/com.termux.x11.MainActivity",
        ]
    ]


def test_x11_apk_auto_acquisition_is_pinned_to_official_project():
    assert termux_desktop.TERMUX_X11_APK_URL.startswith(
        "https://github.com/termux/termux-x11/releases/download/"
    )
    assert termux_desktop.TERMUX_X11_APK_URL.endswith(
        "termux-x11-universal-debug.apk"
    )


def test_x11_start_fails_when_display_socket_never_becomes_ready(tmp_path: Path):
    runtime = termux_desktop.TermuxDesktopRuntime(
        browser="/termux/bin/chromium-browser",
        display=":1",
        x11="/termux/bin/termux-x11",
    )

    class Proc:
        def poll(self):
            return None

    try:
        termux_desktop.launch_termux_x11(
            runtime,
            env={"TMPDIR": str(tmp_path)},
            popen=lambda argv, **kwargs: Proc(),
            run=lambda argv, **kwargs: _completed(),
            which=lambda name: None,
            sleep=lambda _seconds: None,
            ready_timeout=0,
        )
    except RuntimeError as exc:
        assert "did not make DISPLAY=:1 ready" in str(exc)
    else:
        raise AssertionError("launch must fail instead of opening Chromium against an unready display")


def test_termux_renderer_stamp_records_hash_and_timezone(tmp_path: Path, monkeypatch):
    stamp = tmp_path / "termux-desktop-stamp.json"
    monkeypatch.setattr(cli_main, "_termux_desktop_stamp_path", lambda: stamp)
    monkeypatch.setattr(cli_main, "_compute_desktop_content_hash", lambda _root: "abc123")

    cli_main._write_termux_desktop_renderer_stamp()

    payload = json.loads(stamp.read_text(encoding="utf-8"))
    assert payload["contentHash"] == "abc123"
    assert payload["surface"] == "termux-browser-hosted-desktop"
    assert datetime.fromisoformat(payload["builtAt"]).utcoffset() is not None


def test_termux_renderer_build_runs_nested_workspace_from_root(tmp_path: Path, monkeypatch):
    root = tmp_path / "repo"
    desktop = root / "apps" / "desktop"
    desktop.mkdir(parents=True)
    (root / "package.json").write_text('{"workspaces":["apps/*"]}', encoding="utf-8")
    (root / "package-lock.json").write_text("{}", encoding="utf-8")
    (desktop / "package.json").write_text("{}", encoding="utf-8")

    install_calls = []
    build_calls = []

    monkeypatch.setattr(cli_main, "_resolve_node_runtime_npm", lambda: "/termux/bin/npm")
    monkeypatch.setattr(cli_main, "_write_termux_desktop_renderer_stamp", lambda: None)

    def install(npm, cwd, **kwargs):
        install_calls.append((npm, cwd, kwargs))
        return _completed()

    def run(argv, **kwargs):
        build_calls.append((list(argv), kwargs))
        dist = desktop / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("ok", encoding="utf-8")
        return _completed()

    monkeypatch.setattr(cli_main, "_run_npm_install_deterministic", install)
    monkeypatch.setattr(cli_main.subprocess, "run", run)

    cli_main._build_termux_desktop_renderer(
        desktop,
        skip_build=False,
        force_build=True,
    )

    assert len(install_calls) == 1
    assert install_calls[0][1] == root
    assert install_calls[0][2]["extra_args"][:3] == (
        "--workspace",
        "apps/desktop",
        "--include-workspace-root=false",
    )
    assert len(build_calls) == 1
    argv, kwargs = build_calls[0]
    assert argv == [
        "/termux/bin/npm",
        "run",
        "--workspace",
        "apps/desktop",
        "build:renderer",
    ]
    assert kwargs["cwd"] == root
    assert "npm_config_ignore_scripts" not in kwargs["env"]
    assert kwargs["env"]["ELECTRON_SKIP_BINARY_DOWNLOAD"] == "1"

def test_official_x11_release_digest_is_required():
    payload = {
        "assets": [
            {
                "name": termux_desktop.TERMUX_X11_APK_NAME,
                "browser_download_url": termux_desktop.TERMUX_X11_APK_URL,
                "digest": "sha256:" + "a" * 64,
            }
        ]
    }

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    asset = termux_desktop.resolve_official_termux_x11_apk(
        urlopen=lambda _request, timeout: Response(json.dumps(payload).encode("utf-8"))
    )

    assert asset == termux_desktop.TermuxX11ApkAsset(
        url=termux_desktop.TERMUX_X11_APK_URL,
        sha256="a" * 64,
    )


def test_official_x11_release_rejects_missing_digest():
    payload = {
        "assets": [
            {
                "name": termux_desktop.TERMUX_X11_APK_NAME,
                "browser_download_url": termux_desktop.TERMUX_X11_APK_URL,
                "digest": None,
            }
        ]
    }

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    assert (
        termux_desktop.resolve_official_termux_x11_apk(
            urlopen=lambda _request, timeout: Response(json.dumps(payload).encode("utf-8"))
        )
        is None
    )


def test_sha256_file(tmp_path: Path):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"termux-x11")
    assert termux_desktop._sha256_file(payload) == (
        "4d5e084b7e1de4ac3fb43fd3cbe5d99c6f02c4400d098f82f0e113731227679a"
    )
