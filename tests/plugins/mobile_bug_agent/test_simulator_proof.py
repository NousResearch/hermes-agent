from __future__ import annotations

import base64
import os

from pathlib import Path
from types import SimpleNamespace

import pytest

from plugins.mobile_bug_agent import simulator_proof
from plugins.mobile_bug_agent.simulator_proof import SimulatorProofHarness


@pytest.fixture(autouse=True)
def _fast_simulator_settle(monkeypatch):
    monkeypatch.setenv("MONICA_IOS_SETTLE_SECONDS", "0")
    monkeypatch.setenv("MONICA_ANDROID_SETTLE_SECONDS", "0")


def _worktree(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / ".git").write_text("gitdir: /tmp/fake-mobile-worktree-git-dir", encoding="utf-8")
    (path / "package.json").write_text('{"scripts":{"ios":"expo run:ios","android":"expo run:android"}}', encoding="utf-8")
    return path


def _png_bytes() -> bytes:
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAFElEQVR4nGNgYGD4//8/w38GEAMAIewE/ITr/YQAAAAASUVORK5CYII="
    )


def test_simulator_proof_ios_builds_launches_deep_link_and_screenshots(tmp_path):
    calls = []
    proof_dir = tmp_path / "proof"
    worktree = _worktree(tmp_path / "app")

    def run_text(args, cwd, timeout):
        calls.append((args, cwd, timeout))
        if args[:4] == ("xcrun", "simctl", "io", "SIM-123"):
            Path(args[-1]).write_bytes(_png_bytes())
        return "ok"

    harness = SimulatorProofHarness(run_text=run_text)

    result = harness.run(
        worktree=worktree,
        proof_dir=proof_dir,
        platforms=("ios",),
        ios_simulator_udid="SIM-123",
        deep_link="elixir://marketplace/offer/fitness-first",
        timeout_seconds=90,
    )

    assert result == [str(proof_dir / "ios-screenshot.png")]
    assert calls == [
        (("xcrun", "--find", "simctl"), worktree, 90),
        (("xcodebuild", "-version"), worktree, 90),
        (("npx", "expo", "prebuild", "--platform", "ios", "--no-install"), worktree, 90),
        (("pod", "install", "--ansi"), worktree / "ios", 90),
        (("npx", "expo", "run:ios", "--no-install"), worktree, 90),
        (
            ("xcrun", "simctl", "openurl", "SIM-123", "elixir://marketplace/offer/fitness-first"),
            worktree,
            90,
        ),
        (("xcrun", "simctl", "io", "SIM-123", "screenshot", str(proof_dir / "ios-screenshot.png")), worktree, 90),
    ]


def test_simulator_proof_ios_patches_fmt_before_no_install_run(tmp_path):
    calls = []
    proof_dir = tmp_path / "proof"
    worktree = _worktree(tmp_path / "app")
    app_dir = worktree / "apps" / "elixir-card"
    app_dir.mkdir(parents=True)
    (app_dir / "package.json").write_text('{"scripts":{"ios":"expo run:ios"}}', encoding="utf-8")
    fmt_dir = app_dir / "ios" / "Pods" / "Target Support Files" / "fmt"
    fmt_debug = fmt_dir / "fmt.debug.xcconfig"
    fmt_release = fmt_dir / "fmt.release.xcconfig"
    mmkvcore_dir = app_dir / "ios" / "Pods" / "Target Support Files" / "MMKVCore"
    mmkvcore_debug = mmkvcore_dir / "MMKVCore.debug.xcconfig"
    mmkvcore_release = mmkvcore_dir / "MMKVCore.release.xcconfig"
    pbxproj = app_dir / "ios" / "Pods" / "Pods.xcodeproj" / "project.pbxproj"

    def run_text(args, cwd, timeout):
        calls.append((args, cwd, timeout))
        if args == ("pod", "install", "--ansi"):
            fmt_dir.mkdir(parents=True)
            fmt_debug.write_text("CLANG_CXX_LANGUAGE_STANDARD = c++20\n", encoding="utf-8")
            fmt_release.write_text("CLANG_CXX_LANGUAGE_STANDARD = c++20\n", encoding="utf-8")
            mmkvcore_dir.mkdir(parents=True)
            mmkvcore_debug.write_text("CLANG_CXX_LANGUAGE_STANDARD = c++17\n", encoding="utf-8")
            mmkvcore_release.write_text("CLANG_CXX_LANGUAGE_STANDARD = c++17\n", encoding="utf-8")
            pbxproj.parent.mkdir(parents=True)
            pbxproj.write_text(
                "\n".join(
                    [
                        "\tobjects = {",
                        "\t\tAAA /* Debug */ = {",
                        "\t\t\tisa = XCBuildConfiguration;",
                        "\t\t\tbaseConfigurationReference = BBB /* fmt.debug.xcconfig */;",
                        "\t\t\tbuildSettings = {",
                        '\t\t\t\tCLANG_CXX_LANGUAGE_STANDARD = "c++20";',
                        "\t\t\t};",
                        "\t\t\tname = Debug;",
                        "\t\t};",
                        "\t\tCCC /* Release */ = {",
                        "\t\t\tisa = XCBuildConfiguration;",
                        "\t\t\tbaseConfigurationReference = DDD /* fmt.release.xcconfig */;",
                        "\t\t\tbuildSettings = {",
                        '\t\t\t\tCLANG_CXX_LANGUAGE_STANDARD = "c++20";',
                        "\t\t\t};",
                        "\t\t\tname = Release;",
                        "\t\t};",
                        "\t\tGGG /* Debug */ = {",
                        "\t\t\tisa = XCBuildConfiguration;",
                        "\t\t\tbaseConfigurationReference = HHH /* MMKVCore.debug.xcconfig */;",
                        "\t\t\tbuildSettings = {",
                        '\t\t\t\tCLANG_CXX_LANGUAGE_STANDARD = "c++17";',
                        "\t\t\t};",
                        "\t\t\tname = Debug;",
                        "\t\t};",
                        "\t\tIII /* Release */ = {",
                        "\t\t\tisa = XCBuildConfiguration;",
                        "\t\t\tbaseConfigurationReference = JJJ /* MMKVCore.release.xcconfig */;",
                        "\t\t\tbuildSettings = {",
                        '\t\t\t\tCLANG_CXX_LANGUAGE_STANDARD = "c++17";',
                        "\t\t\t};",
                        "\t\t\tname = Release;",
                        "\t\t};",
                        "\t\tEEE /* Debug */ = {",
                        "\t\t\tisa = XCBuildConfiguration;",
                        "\t\t\tbaseConfigurationReference = FFF /* Other.debug.xcconfig */;",
                        "\t\t\tbuildSettings = {",
                        '\t\t\t\tCLANG_CXX_LANGUAGE_STANDARD = "c++20";',
                        "\t\t\t};",
                        "\t\t\tname = Debug;",
                        "\t\t};",
                        "\t};",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
        if args == ("npx", "expo", "run:ios", "--no-install"):
            assert fmt_debug.read_text(encoding="utf-8") == "CLANG_CXX_LANGUAGE_STANDARD = c++17\n"
            assert fmt_release.read_text(encoding="utf-8") == "CLANG_CXX_LANGUAGE_STANDARD = c++17\n"
            assert mmkvcore_debug.read_text(encoding="utf-8") == "CLANG_CXX_LANGUAGE_STANDARD = gnu++20\n"
            assert mmkvcore_release.read_text(encoding="utf-8") == "CLANG_CXX_LANGUAGE_STANDARD = gnu++20\n"
            pbxproj_content = pbxproj.read_text(encoding="utf-8")
            assert 'baseConfigurationReference = BBB /* fmt.debug.xcconfig */;' in pbxproj_content
            assert 'baseConfigurationReference = DDD /* fmt.release.xcconfig */;' in pbxproj_content
            assert 'baseConfigurationReference = HHH /* MMKVCore.debug.xcconfig */;' in pbxproj_content
            assert 'baseConfigurationReference = JJJ /* MMKVCore.release.xcconfig */;' in pbxproj_content
            assert pbxproj_content.count('CLANG_CXX_LANGUAGE_STANDARD = "c++17";') == 2
            assert pbxproj_content.count('CLANG_CXX_LANGUAGE_STANDARD = "gnu++20";') == 2
            assert 'baseConfigurationReference = FFF /* Other.debug.xcconfig */;' in pbxproj_content
            assert pbxproj_content.count('CLANG_CXX_LANGUAGE_STANDARD = "c++20";') == 1
        if args[:4] == ("xcrun", "simctl", "io", "SIM-123"):
            Path(args[-1]).write_bytes(_png_bytes())
        return "ok"

    harness = SimulatorProofHarness(run_text=run_text)

    result = harness.run(
        worktree=worktree,
        proof_dir=proof_dir,
        platforms=("ios",),
        ios_simulator_udid="SIM-123",
        timeout_seconds=90,
    )

    assert result == [str(proof_dir / "ios-screenshot.png")]
    assert calls == [
        (("xcrun", "--find", "simctl"), worktree, 90),
        (("xcodebuild", "-version"), worktree, 90),
        (("npx", "expo", "prebuild", "--platform", "ios", "--no-install"), app_dir, 90),
        (("pod", "install", "--ansi"), app_dir / "ios", 90),
        (("npx", "expo", "run:ios", "--no-install"), app_dir, 90),
        (("xcrun", "simctl", "io", "SIM-123", "screenshot", str(proof_dir / "ios-screenshot.png")), worktree, 90),
    ]


def test_simulator_proof_ios_captures_while_expo_runner_is_alive(monkeypatch, tmp_path):
    monkeypatch.setenv("MONICA_PACKAGER_HOSTNAME", "localhost")
    text_calls = []
    ready_calls = []
    proof_dir = tmp_path / "proof"
    worktree = _worktree(tmp_path / "app")

    def run_text(args, cwd, timeout):
        text_calls.append((args, cwd, timeout))
        if args[:4] == ("xcrun", "simctl", "io", "SIM-123"):
            Path(args[-1]).write_bytes(_png_bytes())
        return "ok"

    def run_ios_until_ready(args, cwd, timeout, target, bundle_id, dev_client_url, log_dir, while_ready):
        ready_calls.append((args, cwd, timeout, target, bundle_id, dev_client_url, log_dir))
        while_ready()

    harness = SimulatorProofHarness(run_text=run_text, run_ios_until_ready=run_ios_until_ready)

    result = harness.run(
        worktree=worktree,
        proof_dir=proof_dir,
        platforms=("ios",),
        dev_client_scheme="elixir-card",
        ios_simulator_udid="SIM-123",
        ios_bundle_id="com.elixir.card",
        deep_link="elixir://marketplace/offer/fitness-first",
        timeout_seconds=90,
    )

    assert result == [str(proof_dir / "ios-screenshot.png")]
    assert ready_calls == [
        (
            ("npx", "expo", "start", "--dev-client", "--host", "localhost", "--port", "8081"),
            worktree,
            90,
            "SIM-123",
            "com.elixir.card",
            "elixir-card://expo-development-client/?url=http%3A%2F%2Flocalhost%3A8081&disableOnboarding=1",
            proof_dir,
        )
    ]
    assert text_calls == [
        (("xcrun", "--find", "simctl"), worktree, 90),
        (("xcodebuild", "-version"), worktree, 90),
        (("npx", "expo", "prebuild", "--platform", "ios", "--no-install"), worktree, 90),
        (("pod", "install", "--ansi"), worktree / "ios", 90),
        (("npx", "expo", "run:ios", "--no-install", "--no-bundler"), worktree, 90),
        (
            ("xcrun", "simctl", "openurl", "SIM-123", "elixir://marketplace/offer/fitness-first"),
            worktree,
            90,
        ),
        (("xcrun", "simctl", "io", "SIM-123", "screenshot", str(proof_dir / "ios-screenshot.png")), worktree, 90),
    ]


def test_simulator_proof_ios_links_sibling_node_modules(tmp_path):
    calls = []
    workspace = tmp_path / "workspace"
    proof_dir = tmp_path / "proof"
    worktree = _worktree(workspace / "worktrees" / "monica-ENG-123")
    source_repo = workspace / "repos" / "elixir-card-app"
    source = source_repo / "node_modules"
    (source / "react-native").mkdir(parents=True)
    (source / "@expo" / "cli").mkdir(parents=True)
    (source_repo / "package.json").write_text("{}", encoding="utf-8")

    def run_text(args, cwd, timeout):
        calls.append((args, cwd, timeout))
        if args[:4] == ("xcrun", "simctl", "io", "SIM-123"):
            Path(args[-1]).write_bytes(_png_bytes())
        return "ok"

    harness = SimulatorProofHarness(run_text=run_text)

    result = harness.run(
        worktree=worktree,
        proof_dir=proof_dir,
        platforms=("ios",),
        ios_simulator_udid="SIM-123",
    )

    assert result == [str(proof_dir / "ios-screenshot.png")]
    assert calls[2] == (("npx", "expo", "prebuild", "--platform", "ios", "--no-install"), worktree, 600)
    assert (worktree / "node_modules").is_dir()
    assert not (worktree / "node_modules").is_symlink()
    assert (worktree / "node_modules" / "react-native").resolve() == source / "react-native"
    assert (worktree / "node_modules" / "@expo" / "cli").resolve() == source / "@expo" / "cli"


def test_simulator_proof_android_builds_launches_deep_link_and_screenshots(tmp_path):
    text_calls = []
    bytes_calls = []
    proof_dir = tmp_path / "proof"
    worktree = _worktree(tmp_path / "app")

    def run_text(args, cwd, timeout):
        text_calls.append((args, cwd, timeout))
        return "ok"

    def run_bytes(args, cwd, timeout):
        bytes_calls.append((args, cwd, timeout))
        return _png_bytes()

    harness = SimulatorProofHarness(run_text=run_text, run_bytes=run_bytes)

    result = harness.run(
        worktree=worktree,
        proof_dir=proof_dir,
        platforms=("android",),
        android_serial="emulator-5554",
        deep_link="elixir://marketplace/offer/fitness-first",
        timeout_seconds=120,
    )

    assert result == [str(proof_dir / "android-screenshot.png")]
    assert text_calls == [
        (("emulator", "-list-avds"), worktree, 120),
        (("adb", "-s", "emulator-5554", "version"), worktree, 120),
        (("adb", "-s", "emulator-5554", "reverse", "tcp:8081", "tcp:8081"), worktree, 120),
        (("npm", "run", "android"), worktree, 120),
        (
            (
                "adb",
                "-s",
                "emulator-5554",
                "shell",
                "am",
                "start",
                "-a",
                "android.intent.action.VIEW",
                "-d",
                "elixir://marketplace/offer/fitness-first",
            ),
            worktree,
            120,
        ),
    ]
    assert bytes_calls == [
        (("adb", "-s", "emulator-5554", "exec-out", "screencap", "-p"), worktree, 120)
    ]
    assert (proof_dir / "android-screenshot.png").read_bytes() == _png_bytes()


def test_simulator_proof_android_captures_while_long_lived_run_is_foreground(monkeypatch, tmp_path):
    monkeypatch.delenv("MONICA_ANDROID_PACKAGER_HOSTNAME", raising=False)
    monkeypatch.delenv("MONICA_PACKAGER_HOSTNAME", raising=False)
    monkeypatch.delenv("REACT_NATIVE_PACKAGER_HOSTNAME", raising=False)
    text_calls = []
    bytes_calls = []
    foreground_calls = []
    proof_dir = tmp_path / "proof"
    worktree = _worktree(tmp_path / "app")

    def run_text(args, cwd, timeout):
        text_calls.append((args, cwd, timeout))
        return "ok"

    def run_bytes(args, cwd, timeout):
        bytes_calls.append((args, cwd, timeout))
        return _png_bytes()

    def run_android_until_foreground(
        args,
        cwd,
        timeout,
        adb,
        package,
        log_dir,
        open_dev_client,
        capture_foreground,
    ):
        foreground_calls.append((args, cwd, timeout, adb, package, log_dir))
        open_dev_client()
        capture_foreground()

    harness = SimulatorProofHarness(
        run_text=run_text,
        run_bytes=run_bytes,
        run_android_until_foreground=run_android_until_foreground,
    )

    result = harness.run(
        worktree=worktree,
        proof_dir=proof_dir,
        platforms=("android",),
        dev_client_scheme="elixir-card",
        android_serial="emulator-5554",
        android_package="com.elixir.card.staging",
        deep_link="elixir://marketplace/offer/fitness-first",
        timeout_seconds=120,
    )

    assert result == [str(proof_dir / "android-screenshot.png")]
    assert foreground_calls == [
        (
            ("npm", "run", "android"),
            worktree,
            120,
            ("adb", "-s", "emulator-5554"),
            "com.elixir.card.staging",
            proof_dir,
        )
    ]
    assert text_calls == [
        (("emulator", "-list-avds"), worktree, 120),
        (("adb", "-s", "emulator-5554", "version"), worktree, 120),
        (("adb", "-s", "emulator-5554", "reverse", "tcp:8081", "tcp:8081"), worktree, 120),
        (("adb", "-s", "emulator-5554", "shell", "am", "force-stop", "com.elixir.card.staging"), worktree, 120),
        (
            (
                "adb",
                "-s",
                "emulator-5554",
                "shell",
                "am",
                "start",
                "-a",
                "android.intent.action.VIEW",
                "-d",
                "elixir-card://expo-development-client/?url=http%3A%2F%2F127.0.0.1%3A8081&disableOnboarding=1",
                "-p",
                "com.elixir.card.staging",
            ),
            worktree,
            120,
        ),
        (
            (
                "adb",
                "-s",
                "emulator-5554",
                "shell",
                "am",
                "start",
                "-a",
                "android.intent.action.VIEW",
                "-d",
                "elixir://marketplace/offer/fitness-first",
                "-p",
                "com.elixir.card.staging",
            ),
            worktree,
            120,
        ),
    ]
    assert bytes_calls == [
        (("adb", "-s", "emulator-5554", "exec-out", "screencap", "-p"), worktree, 120)
    ]


def test_android_foreground_runner_waits_for_bundle_before_capture(monkeypatch, tmp_path):
    events = []
    monkeypatch.setenv("MONICA_PACKAGER_HOSTNAME", "localhost")

    class FakeProcess:
        def poll(self):
            return None

        def terminate(self):
            events.append("terminate")

        def wait(self, timeout):
            return 0

        def kill(self):
            events.append("kill")

    def fake_popen(*_args, **_kwargs):
        events.append("metro-start")
        return FakeProcess()

    monkeypatch.setattr(simulator_proof.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(simulator_proof, "_android_package_is_foreground", lambda *_args: True)
    monkeypatch.setattr(
        simulator_proof,
        "_wait_for_metro_bundle_request",
        lambda *_args, **_kwargs: events.append("bundle-requested"),
    )

    simulator_proof._run_android_until_foreground(
        ("npm", "run", "android"),
        tmp_path,
        30,
        ("adb",),
        "com.elixir.card",
        tmp_path,
        lambda: events.append("open-dev-client"),
        lambda: events.append("capture"),
    )

    assert events[:4] == ["metro-start", "open-dev-client", "bundle-requested", "capture"]


def test_simulator_proof_refuses_non_mobile_worktree(tmp_path):
    proof_dir = tmp_path / "proof"
    worktree = tmp_path / "app"
    worktree.mkdir()
    (worktree / ".git").write_text("gitdir: /tmp/fake-mobile-worktree-git-dir", encoding="utf-8")

    harness = SimulatorProofHarness()

    try:
        harness.run(worktree=worktree, proof_dir=proof_dir, platforms=("ios",))
    except RuntimeError as exc:
        assert "package.json" in str(exc)
    else:
        raise AssertionError("expected proof harness to fail closed")


def test_android_foreground_detector_requires_focused_window(monkeypatch, tmp_path):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout="\n".join(
                [
                    "mCurrentFocus=Window{abc u0 com.google.android.apps.nexuslauncher/.NexusLauncherActivity}",
                    "Package somewhere else: com.elixir.card.staging",
                ]
            ),
        )

    monkeypatch.setattr(simulator_proof.subprocess, "run", fake_run)

    assert not simulator_proof._android_package_is_foreground(("adb",), tmp_path, "com.elixir.card.staging")


def test_android_foreground_detector_accepts_focused_app(monkeypatch, tmp_path):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout="mFocusedApp=ActivityRecord{123 u0 com.elixir.card.staging/.MainActivity t9}",
        )

    monkeypatch.setattr(simulator_proof.subprocess, "run", fake_run)

    assert simulator_proof._android_package_is_foreground(("adb",), tmp_path, "com.elixir.card.staging")


def test_launch_android_package_prefers_resolved_activity(monkeypatch, tmp_path):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((tuple(args), kwargs))
        if "resolve-activity" in args:
            return SimpleNamespace(
                returncode=0,
                stdout="priority=0 preferredOrder=0 match=0x108000 specificIndex=-1 isDefault=false\n"
                "com.example.app/.MainActivity\n",
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(simulator_proof.subprocess, "run", fake_run)

    simulator_proof._launch_android_package(("adb", "-s", "emulator-5554"), tmp_path, "com.example.app")

    assert calls == [
        (
            (
                "adb",
                "-s",
                "emulator-5554",
                "shell",
                "cmd",
                "package",
                "resolve-activity",
                "--brief",
                "com.example.app",
            ),
            {
                "cwd": str(tmp_path),
                "text": True,
                "capture_output": True,
                "timeout": 15,
                "check": False,
            },
        ),
        (
            (
                "adb",
                "-s",
                "emulator-5554",
                "shell",
                "am",
                "start",
                "-n",
                "com.example.app/.MainActivity",
            ),
            {
                "cwd": str(tmp_path),
                "text": True,
                "capture_output": True,
                "timeout": 15,
                "check": False,
            },
        ),
    ]


def test_launch_android_package_falls_back_to_monkey(monkeypatch, tmp_path):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((tuple(args), kwargs))
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    monkeypatch.setattr(simulator_proof.subprocess, "run", fake_run)

    simulator_proof._launch_android_package(("adb", "-s", "emulator-5554"), tmp_path, "com.example.app")

    assert calls[-1] == (
        (
            "adb",
            "-s",
            "emulator-5554",
            "shell",
            "monkey",
            "-p",
            "com.example.app",
            "-c",
            "android.intent.category.LAUNCHER",
            "1",
        ),
        {
            "cwd": str(tmp_path),
            "text": True,
            "capture_output": True,
            "timeout": 15,
            "check": False,
        },
    )


def test_resolve_android_launch_activity_ignores_metadata(monkeypatch, tmp_path):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout="priority=0 preferredOrder=0\ncom.example.app/.MainActivity\n",
            stderr="",
        )

    monkeypatch.setattr(simulator_proof.subprocess, "run", fake_run)

    assert (
        simulator_proof._resolve_android_launch_activity(
            ("adb", "-s", "emulator-5554"), tmp_path, "com.example.app"
        )
        == "com.example.app/.MainActivity"
    )


def test_android_screenshot_validation_rejects_blank_image(tmp_path):
    screenshot = tmp_path / "blank.png"

    try:
        from PIL import Image
    except ImportError:
        return

    Image.new("RGB", (20, 20), color="white").save(screenshot)

    try:
        simulator_proof._assert_screenshot_has_visual_content(screenshot)
    except RuntimeError as exc:
        assert "appears blank" in str(exc)
    else:
        raise AssertionError("expected blank screenshot to fail validation")


def test_prepare_android_worktree_links_sibling_node_modules(tmp_path):
    workspace = tmp_path / "workspace"
    worktree = workspace / "worktrees" / "monica-ENG-123"
    source_repo = workspace / "repos" / "elixir-card-app"
    source = source_repo / "node_modules"
    worktree.mkdir(parents=True)
    (source / "expo").mkdir(parents=True)
    (source_repo / "package.json").write_text("{}", encoding="utf-8")

    simulator_proof._prepare_android_worktree(worktree)

    assert (worktree / "node_modules").is_dir()
    assert not (worktree / "node_modules").is_symlink()
    assert (worktree / "node_modules" / "expo").resolve() == source / "expo"


def test_prepare_react_native_worktree_excludes_local_symlinks_from_git(tmp_path):
    workspace = tmp_path / "workspace"
    worktree = workspace / "worktrees" / "monica-ENG-123"
    git_info = worktree / ".git" / "info"
    source_repo = workspace / "repos" / "elixir-card-app"
    source_app = source_repo / "apps" / "elixir-card"
    target_app = worktree / "apps" / "elixir-card"
    git_info.mkdir(parents=True)
    source_repo.mkdir(parents=True)
    target_app.mkdir(parents=True)
    (worktree / "package.json").write_text("{}", encoding="utf-8")
    (source_repo / "package.json").write_text("{}", encoding="utf-8")
    (source_repo / "node_modules").mkdir(parents=True)
    source_app.mkdir(parents=True)
    (source_app / ".env").write_text("SECRET=value\n", encoding="utf-8")

    simulator_proof._prepare_react_native_worktree(worktree)

    exclude = (git_info / "exclude").read_text(encoding="utf-8")
    assert "/node_modules\n" in exclude
    assert "/apps/elixir-card/.env\n" in exclude
    assert "/apps/elixir-card/.env.*\n" in exclude
    assert "fingerprint.config.js\n" in exclude


def test_git_info_exclude_path_uses_common_dir_for_linked_worktree(tmp_path):
    worktree = tmp_path / "workspace" / "worktrees" / "monica-ENG-123"
    common_git_dir = tmp_path / "workspace" / "repos" / "elixir-card-app" / ".git"
    worktree_git_dir = common_git_dir / "worktrees" / "monica-ENG-123"
    worktree.mkdir(parents=True)
    worktree_git_dir.mkdir(parents=True)
    (worktree / ".git").write_text(f"gitdir: {worktree_git_dir}\n", encoding="utf-8")
    (worktree_git_dir / "commondir").write_text("../..\n", encoding="utf-8")

    assert simulator_proof._git_info_exclude_path(worktree) == common_git_dir / "info" / "exclude"


def test_android_run_env_uses_default_sdk_dir(monkeypatch, tmp_path):
    fake_home = tmp_path / "home"
    sdk = fake_home / "Library" / "Android" / "sdk"
    sdk.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("ANDROID_HOME", raising=False)
    monkeypatch.delenv("ANDROID_SDK_ROOT", raising=False)
    monkeypatch.delenv("MONICA_ANDROID_SDK_DIR", raising=False)

    env = simulator_proof._android_run_env()

    assert env["ANDROID_HOME"] == str(sdk)
    assert env["ANDROID_SDK_ROOT"] == str(sdk)


def test_android_run_env_defaults_packager_to_loopback(monkeypatch):
    monkeypatch.delenv("MONICA_ANDROID_PACKAGER_HOSTNAME", raising=False)
    monkeypatch.delenv("MONICA_PACKAGER_HOSTNAME", raising=False)
    monkeypatch.delenv("REACT_NATIVE_PACKAGER_HOSTNAME", raising=False)

    env = simulator_proof._android_run_env()

    assert env["REACT_NATIVE_PACKAGER_HOSTNAME"] == "127.0.0.1"


def test_simulator_run_env_includes_user_gem_bin(monkeypatch, tmp_path):
    fake_home = tmp_path / "home"
    gem_bin = fake_home / ".gem" / "ruby" / "2.6.0" / "bin"
    gem_bin.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("PATH", "/usr/bin")
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = tuple(args)
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(simulator_proof.subprocess, "run", fake_run)

    simulator_proof._run(("npm", "run", "ios"), tmp_path, 10, capture_bytes=False)

    assert captured["args"] == ("npm", "run", "ios")
    assert captured["kwargs"]["env"]["PATH"].split(os.pathsep)[0] == str(gem_bin)


def test_simulator_run_env_preloads_ruby_logger(monkeypatch):
    monkeypatch.delenv("RUBYOPT", raising=False)
    monkeypatch.setenv("MONICA_PACKAGER_HOSTNAME", "localhost")

    env = simulator_proof._simulator_run_env()

    assert env["RUBYOPT"] == "-rlogger"
    assert env["REACT_NATIVE_PACKAGER_HOSTNAME"] == "localhost"


def test_default_packager_hostname_uses_non_loopback_ifconfig(monkeypatch):
    monkeypatch.delenv("MONICA_PACKAGER_HOSTNAME", raising=False)

    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout="\n".join(
                [
                    "\tinet 127.0.0.1 netmask 0xff000000",
                    "\tinet 10.20.1.176 netmask 0xfffffe00 broadcast 10.20.1.255",
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr(simulator_proof.subprocess, "run", fake_run)

    assert simulator_proof._default_packager_hostname() == "10.20.1.176"


def test_expo_dev_client_url_encodes_packager_url():
    assert (
        simulator_proof._expo_dev_client_url("elixir-card", "http://10.20.1.176:8081")
        == "elixir-card://expo-development-client/?url=http%3A%2F%2F10.20.1.176%3A8081&disableOnboarding=1"
    )


def test_ios_packager_hostname_defaults_to_localhost(monkeypatch):
    monkeypatch.delenv("MONICA_IOS_PACKAGER_HOSTNAME", raising=False)
    monkeypatch.delenv("MONICA_PACKAGER_HOSTNAME", raising=False)
    monkeypatch.delenv("REACT_NATIVE_PACKAGER_HOSTNAME", raising=False)

    assert simulator_proof._ios_packager_hostname() == "localhost"
    assert simulator_proof._ios_expo_host() == "localhost"


def test_ios_metro_ready_requires_packager_status(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, _limit):
            return b"packager-status:running"

    calls = []

    def fake_urlopen(url, timeout):
        calls.append((url, timeout))
        return Response()

    monkeypatch.delenv("MONICA_IOS_WAIT_FOR_METRO", raising=False)
    monkeypatch.delenv("MONICA_METRO_STATUS_URL", raising=False)
    monkeypatch.setattr(simulator_proof.urllib.request, "urlopen", fake_urlopen)

    assert simulator_proof._ios_metro_is_ready()
    assert calls == [("http://localhost:8081/status", 1)]


def test_ios_metro_ready_can_be_disabled(monkeypatch):
    monkeypatch.setenv("MONICA_IOS_WAIT_FOR_METRO", "0")

    assert simulator_proof._ios_metro_is_ready()


def test_metro_bundle_request_detector_accepts_platform_query(tmp_path):
    stdout = tmp_path / "metro.stdout.log"
    stderr = tmp_path / "metro.stderr.log"
    stdout.write_text("GET /index.bundle?platform=ios&dev=true 200", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")

    assert simulator_proof._metro_bundle_was_requested(stdout, stderr, "ios")
    assert not simulator_proof._metro_bundle_was_requested(stdout, stderr, "android")


def test_metro_bundle_request_detector_accepts_expo_bundled_line(tmp_path):
    stdout = tmp_path / "metro.stdout.log"
    stderr = tmp_path / "metro.stderr.log"
    stdout.write_text("", encoding="utf-8")
    stderr.write_text("Android Bundled 1240ms apps/elixir-card/index.js", encoding="utf-8")

    assert simulator_proof._metro_bundle_was_requested(stdout, stderr, "android")


def test_metro_bundle_request_wait_fails_closed_with_log_paths(monkeypatch, tmp_path):
    stdout = tmp_path / "metro.stdout.log"
    stderr = tmp_path / "metro.stderr.log"
    stdout.write_text("Metro waiting on http://localhost:8081", encoding="utf-8")
    stderr.write_text("No apps connected", encoding="utf-8")
    timeline = iter([10.0, 11.0, 12.5])
    monkeypatch.setattr(simulator_proof.time, "monotonic", lambda: next(timeline))
    monkeypatch.setattr(simulator_proof.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError) as exc:
        simulator_proof._wait_for_metro_bundle_request(stdout, stderr, "ios", 12.0)

    message = str(exc.value)
    assert "Metro did not receive a ios bundle request" in message
    assert str(stdout) in message
    assert "No apps connected" in message


def test_ios_bundle_wait_seconds_defaults_and_overrides(monkeypatch):
    monkeypatch.delenv("MONICA_IOS_BUNDLE_WAIT_SECONDS", raising=False)
    assert simulator_proof._ios_bundle_wait_seconds() == 90.0

    monkeypatch.setenv("MONICA_IOS_BUNDLE_WAIT_SECONDS", "45")
    assert simulator_proof._ios_bundle_wait_seconds() == 45.0

    monkeypatch.setenv("MONICA_IOS_BUNDLE_WAIT_SECONDS", "not-a-number")
    assert simulator_proof._ios_bundle_wait_seconds() == 90.0

    monkeypatch.setenv("MONICA_IOS_BUNDLE_WAIT_SECONDS", "-5")
    assert simulator_proof._ios_bundle_wait_seconds() == 90.0


def test_metro_bundle_requested_before_returns_false_without_marker(monkeypatch, tmp_path):
    stdout = tmp_path / "metro.stdout.log"
    stderr = tmp_path / "metro.stderr.log"
    stdout.write_text("Metro waiting on http://localhost:8081", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    timeline = iter([10.0, 11.0, 12.5])
    monkeypatch.setattr(simulator_proof.time, "monotonic", lambda: next(timeline))
    monkeypatch.setattr(simulator_proof.time, "sleep", lambda _seconds: None)

    assert not simulator_proof._metro_bundle_requested_before(stdout, stderr, "ios", 12.0)


def test_metro_bundle_requested_before_detects_marker_mid_wait(monkeypatch, tmp_path):
    stdout = tmp_path / "metro.stdout.log"
    stderr = tmp_path / "metro.stderr.log"
    stdout.write_text("Metro waiting on http://localhost:8081", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")

    def sleep(_seconds):
        stdout.write_text("iOS Bundled 1240ms apps/elixir-card/index.ts", encoding="utf-8")

    timeline = iter([10.0, 11.0])
    monkeypatch.setattr(simulator_proof.time, "monotonic", lambda: next(timeline))
    monkeypatch.setattr(simulator_proof.time, "sleep", sleep)

    assert simulator_proof._metro_bundle_requested_before(stdout, stderr, "ios", 12.0)


def test_launch_ios_bundle_delivers_dev_client_url_via_openurl(monkeypatch, tmp_path):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(tuple(args))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(simulator_proof.subprocess, "run", fake_run)
    dev_client_url = "elixir-card://expo-development-client/?url=http%3A%2F%2Flocalhost%3A8081&disableOnboarding=1"

    assert simulator_proof._launch_ios_bundle("SIM-123", tmp_path, "com.elixir.card", dev_client_url)

    assert len(calls) == 3
    assert calls[0] == (
        "xcrun",
        "simctl",
        "spawn",
        "SIM-123",
        "defaults",
        "write",
        "com.elixir.card",
        "EXDevMenuIsOnboardingFinished",
        "-bool",
        "true",
    )
    launch_args = calls[1]
    assert launch_args[:3] == ("xcrun", "simctl", "launch")
    assert launch_args[-2:] == ("SIM-123", "com.elixir.card")
    assert "--initialUrl" not in launch_args
    assert calls[2] == ("xcrun", "simctl", "openurl", "SIM-123", dev_client_url)


def test_launch_ios_bundle_skips_openurl_without_dev_client_url(monkeypatch, tmp_path):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(tuple(args))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(simulator_proof.subprocess, "run", fake_run)

    assert simulator_proof._launch_ios_bundle("SIM-123", tmp_path, "com.elixir.card", "")

    assert len(calls) == 2
    assert calls[0][:4] == ("xcrun", "simctl", "spawn", "SIM-123")
    assert calls[1][:3] == ("xcrun", "simctl", "launch")


def test_launch_ios_bundle_returns_false_when_openurl_fails(monkeypatch, tmp_path):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(tuple(args))
        returncode = 1 if "openurl" in args else 0
        return SimpleNamespace(returncode=returncode, stdout="", stderr="")

    monkeypatch.setattr(simulator_proof.subprocess, "run", fake_run)

    assert not simulator_proof._launch_ios_bundle(
        "SIM-123", tmp_path, "com.elixir.card", "elixir-card://expo-development-client/?url=x"
    )

    openurl_calls = [args for args in calls if "openurl" in args]
    assert len(openurl_calls) == 3


def test_simulator_proof_ios_tolerates_run_ios_failure_when_app_installed(monkeypatch, tmp_path):
    proof_dir = tmp_path / "proof"
    worktree = _worktree(tmp_path / "app")
    monkeypatch.setattr(simulator_proof, "_ios_app_is_installed", lambda *_args: True)

    def run_text(args, cwd, timeout):
        if args[:3] == ("npx", "expo", "run:ios"):
            raise RuntimeError("command failed (1): npx expo run:ios (osascript unavailable)")
        if args[:4] == ("xcrun", "simctl", "io", "SIM-123"):
            Path(args[-1]).write_bytes(_png_bytes())
        return "ok"

    def run_ios_until_ready(args, cwd, timeout, target, bundle_id, dev_client_url, log_dir, while_ready):
        while_ready()

    harness = SimulatorProofHarness(run_text=run_text, run_ios_until_ready=run_ios_until_ready)

    result = harness.run(
        worktree=worktree,
        proof_dir=proof_dir,
        platforms=("ios",),
        dev_client_scheme="elixir-card",
        ios_simulator_udid="SIM-123",
        ios_bundle_id="com.elixir.card",
        timeout_seconds=90,
    )

    assert result == [str(proof_dir / "ios-screenshot.png")]


def test_simulator_proof_ios_surfaces_run_ios_failure_when_app_missing(monkeypatch, tmp_path):
    proof_dir = tmp_path / "proof"
    worktree = _worktree(tmp_path / "app")
    monkeypatch.setattr(simulator_proof, "_ios_app_is_installed", lambda *_args: False)

    def run_text(args, cwd, timeout):
        if args[:3] == ("npx", "expo", "run:ios"):
            raise RuntimeError("command failed (65): npx expo run:ios (build error)")
        return "ok"

    harness = SimulatorProofHarness(run_text=run_text, run_ios_until_ready=lambda *_args: None)

    with pytest.raises(RuntimeError, match="build error"):
        harness.run(
            worktree=worktree,
            proof_dir=proof_dir,
            platforms=("ios",),
            dev_client_scheme="elixir-card",
            ios_simulator_udid="SIM-123",
            ios_bundle_id="com.elixir.card",
            timeout_seconds=90,
        )


def test_ensure_ios_fingerprint_config_writes_fast_config(tmp_path):
    simulator_proof._ensure_ios_fingerprint_config(tmp_path)

    config = (tmp_path / "fingerprint.config.js").read_text(encoding="utf-8")
    assert "ignorePaths" in config

    custom = "module.exports = {};\n"
    (tmp_path / "fingerprint.config.js").write_text(custom, encoding="utf-8")
    simulator_proof._ensure_ios_fingerprint_config(tmp_path)
    assert (tmp_path / "fingerprint.config.js").read_text(encoding="utf-8") == custom


def test_ensure_ios_fingerprint_config_respects_existing_cjs(tmp_path):
    (tmp_path / "fingerprint.config.cjs").write_text("module.exports = {};\n", encoding="utf-8")

    simulator_proof._ensure_ios_fingerprint_config(tmp_path)

    assert not (tmp_path / "fingerprint.config.js").exists()


def test_ios_metro_env_prefers_ipv4_loopback(monkeypatch):
    monkeypatch.delenv("NODE_OPTIONS", raising=False)
    assert simulator_proof._ios_metro_env()["NODE_OPTIONS"] == "--dns-result-order=ipv4first"

    monkeypatch.setenv("NODE_OPTIONS", "--max-old-space-size=4096")
    assert (
        simulator_proof._ios_metro_env()["NODE_OPTIONS"]
        == "--max-old-space-size=4096 --dns-result-order=ipv4first"
    )

    monkeypatch.setenv("NODE_OPTIONS", "--dns-result-order=verbatim")
    assert simulator_proof._ios_metro_env()["NODE_OPTIONS"] == "--dns-result-order=verbatim"


def test_ios_metro_env_pins_packager_hostname_to_loopback(monkeypatch):
    monkeypatch.delenv("MONICA_IOS_PACKAGER_HOSTNAME", raising=False)
    monkeypatch.delenv("MONICA_PACKAGER_HOSTNAME", raising=False)
    monkeypatch.delenv("REACT_NATIVE_PACKAGER_HOSTNAME", raising=False)
    assert simulator_proof._ios_metro_env()["REACT_NATIVE_PACKAGER_HOSTNAME"] == "localhost"

    monkeypatch.setenv("MONICA_IOS_PACKAGER_HOSTNAME", "127.0.0.1")
    assert simulator_proof._ios_metro_env()["REACT_NATIVE_PACKAGER_HOSTNAME"] == "127.0.0.1"


def test_required_env_keys_accept_local_env_file(monkeypatch, tmp_path):
    monkeypatch.delenv("MONICA_TEST_REQUIRED_ALPHA", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "staging")
    (tmp_path / ".env.staging").write_text(
        "MONICA_TEST_REQUIRED_ALPHA=secret-value\n",
        encoding="utf-8",
    )

    simulator_proof._validate_required_env_keys(tmp_path, ("MONICA_TEST_REQUIRED_ALPHA",))


def test_required_env_keys_accept_expo_app_env_file(monkeypatch, tmp_path):
    monkeypatch.delenv("MONICA_TEST_REQUIRED_ALPHA", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "prod")
    app_dir = tmp_path / "apps" / "elixir-card"
    app_dir.mkdir(parents=True)
    (app_dir / ".env.prod").write_text(
        "MONICA_TEST_REQUIRED_ALPHA=secret-value\n",
        encoding="utf-8",
    )

    simulator_proof._validate_required_env_keys(tmp_path, ("MONICA_TEST_REQUIRED_ALPHA",))


def test_simulator_proof_links_sibling_app_env_files_before_required_env_validation(monkeypatch, tmp_path):
    monkeypatch.delenv("MONICA_TEST_REQUIRED_ALPHA", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "staging")
    workspace = tmp_path / "workspace"
    proof_dir = tmp_path / "proof"
    worktree = _worktree(workspace / "worktrees" / "monica-ENG-123")
    target_app = worktree / "apps" / "elixir-card"
    target_app.mkdir(parents=True)
    source_repo = workspace / "repos" / "elixir-card-app"
    source_app = source_repo / "apps" / "elixir-card"
    source_app.mkdir(parents=True)
    (source_repo / "package.json").write_text("{}", encoding="utf-8")
    (source_app / ".env.staging").write_text(
        "MONICA_TEST_REQUIRED_ALPHA=secret-value\n",
        encoding="utf-8",
    )

    def run_text(args, cwd, timeout):
        if args[:4] == ("xcrun", "simctl", "io", "SIM-123"):
            Path(args[-1]).write_bytes(_png_bytes())
        return "ok"

    harness = SimulatorProofHarness(run_text=run_text)

    result = harness.run(
        worktree=worktree,
        proof_dir=proof_dir,
        platforms=("ios",),
        ios_simulator_udid="SIM-123",
        required_env_keys=("MONICA_TEST_REQUIRED_ALPHA",),
    )

    assert result == [str(proof_dir / "ios-screenshot.png")]
    assert (target_app / ".env.staging").is_symlink()
    assert (target_app / ".env.staging").resolve() == source_app / ".env.staging"


def test_required_env_keys_report_missing_names_without_values(monkeypatch, tmp_path):
    monkeypatch.delenv("MONICA_TEST_REQUIRED_ALPHA", raising=False)
    monkeypatch.delenv("MONICA_TEST_REQUIRED_BETA", raising=False)
    (tmp_path / ".env").write_text(
        "MONICA_TEST_REQUIRED_ALPHA=secret-value\n",
        encoding="utf-8",
    )

    try:
        simulator_proof._validate_required_env_keys(
            tmp_path,
            ("MONICA_TEST_REQUIRED_ALPHA", "MONICA_TEST_REQUIRED_BETA"),
        )
    except RuntimeError as exc:
        message = str(exc)
        assert "MONICA_TEST_REQUIRED_BETA" in message
        assert "secret-value" not in message
    else:
        raise AssertionError("expected missing required env key to fail validation")
