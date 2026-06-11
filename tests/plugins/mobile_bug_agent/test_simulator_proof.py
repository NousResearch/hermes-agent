from __future__ import annotations

import base64

from pathlib import Path
from types import SimpleNamespace

from plugins.mobile_bug_agent import simulator_proof
from plugins.mobile_bug_agent.simulator_proof import SimulatorProofHarness


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
            Path(args[-1]).write_text("png", encoding="utf-8")
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
        (("npm", "run", "ios"), worktree, 90),
        (
            ("xcrun", "simctl", "openurl", "SIM-123", "elixir://marketplace/offer/fitness-first"),
            worktree,
            90,
        ),
        (("xcrun", "simctl", "io", "SIM-123", "screenshot", str(proof_dir / "ios-screenshot.png")), worktree, 90),
    ]


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


def test_simulator_proof_android_captures_while_long_lived_run_is_foreground(tmp_path):
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

    def run_android_until_foreground(args, cwd, timeout, adb, package, while_foreground):
        foreground_calls.append((args, cwd, timeout, adb, package))
        while_foreground()

    harness = SimulatorProofHarness(
        run_text=run_text,
        run_bytes=run_bytes,
        run_android_until_foreground=run_android_until_foreground,
    )

    result = harness.run(
        worktree=worktree,
        proof_dir=proof_dir,
        platforms=("android",),
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
                "elixir://marketplace/offer/fitness-first",
            ),
            worktree,
            120,
        ),
    ]
    assert bytes_calls == [
        (("adb", "-s", "emulator-5554", "exec-out", "screencap", "-p"), worktree, 120)
    ]


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
    source = workspace / "repos" / "elixir-card-app" / "node_modules"
    worktree.mkdir(parents=True)
    source.mkdir(parents=True)

    simulator_proof._prepare_android_worktree(worktree)

    assert (worktree / "node_modules").is_symlink()
    assert (worktree / "node_modules").resolve() == source


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
