from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Sequence


RunText = Callable[[tuple[str, ...], Path, int], str]
RunBytes = Callable[[tuple[str, ...], Path, int], bytes]
RunAndroidUntilForeground = Callable[
    [tuple[str, ...], Path, int, tuple[str, ...], str, Callable[[], None]],
    None,
]


class SimulatorProofHarness:
    def __init__(
        self,
        *,
        run_text: RunText | None = None,
        run_bytes: RunBytes | None = None,
        run_android_until_foreground: RunAndroidUntilForeground | None = None,
    ) -> None:
        self._run_text = run_text or _run_text_command
        self._run_bytes = run_bytes or _run_bytes_command
        self._run_android_until_foreground = run_android_until_foreground or _run_android_until_foreground

    def run(
        self,
        *,
        worktree: str | Path,
        proof_dir: str | Path,
        platforms: Sequence[str],
        ios_simulator_udid: str = "",
        android_serial: str = "",
        android_avd: str = "",
        android_package: str = "",
        deep_link: str = "",
        required_env_keys: Sequence[str] = (),
        timeout_seconds: int = 600,
    ) -> list[str]:
        worktree_path = Path(worktree)
        proof_path = Path(proof_dir)
        self._validate_worktree(worktree_path)
        _validate_required_env_keys(worktree_path, required_env_keys)
        proof_path.mkdir(parents=True, exist_ok=True)

        artifacts: list[str] = []
        for platform in _clean_platforms(platforms):
            if platform == "ios":
                artifacts.append(
                    self._capture_ios(
                        worktree=worktree_path,
                        proof_dir=proof_path,
                        simulator_udid=ios_simulator_udid,
                        deep_link=deep_link,
                        timeout_seconds=timeout_seconds,
                    )
                )
            elif platform == "android":
                artifacts.append(
                    self._capture_android(
                        worktree=worktree_path,
                        proof_dir=proof_path,
                        android_serial=android_serial,
                        android_avd=android_avd,
                        android_package=android_package,
                        deep_link=deep_link,
                        timeout_seconds=timeout_seconds,
                    )
                )
            else:
                raise RuntimeError(f"unsupported proof platform: {platform}")
        return artifacts

    @staticmethod
    def _validate_worktree(worktree: Path) -> None:
        if not worktree.is_dir():
            raise RuntimeError(f"worktree does not exist: {worktree}")
        if not (worktree / ".git").exists():
            raise RuntimeError(f"worktree is not a git worktree: {worktree}")
        if not (worktree / "package.json").exists():
            raise RuntimeError(f"package.json was not found in worktree: {worktree}")

    def _capture_ios(
        self,
        *,
        worktree: Path,
        proof_dir: Path,
        simulator_udid: str,
        deep_link: str,
        timeout_seconds: int,
    ) -> str:
        target = simulator_udid.strip() or "booted"
        screenshot = proof_dir / "ios-screenshot.png"
        self._run_text(("xcrun", "--find", "simctl"), worktree, timeout_seconds)
        self._run_text(("xcodebuild", "-version"), worktree, timeout_seconds)
        self._run_text(("npm", "run", "ios"), worktree, timeout_seconds)
        if deep_link.strip():
            self._run_text(("xcrun", "simctl", "openurl", target, deep_link.strip()), worktree, timeout_seconds)
        self._run_text(("xcrun", "simctl", "io", target, "screenshot", str(screenshot)), worktree, timeout_seconds)
        if not screenshot.is_file():
            raise RuntimeError(f"iOS simulator screenshot was not created: {screenshot}")
        return str(screenshot)

    def _capture_android(
        self,
        *,
        worktree: Path,
        proof_dir: Path,
        android_serial: str,
        android_avd: str,
        android_package: str,
        deep_link: str,
        timeout_seconds: int,
    ) -> str:
        _prepare_android_worktree(worktree)
        screenshot = proof_dir / "android-screenshot.png"
        serial = android_serial.strip() or ("emulator-5554" if android_avd.strip() else "")
        adb = ("adb", "-s", serial) if serial else ("adb",)
        avds = self._run_text(("emulator", "-list-avds"), worktree, timeout_seconds)
        if android_avd.strip() and android_avd.strip() not in set(avds.splitlines()):
            raise RuntimeError(f"Android AVD was not found: {android_avd.strip()}")
        emulator_cleanup = _ensure_android_emulator(
            android_avd.strip(),
            adb,
            worktree,
            timeout_seconds,
        )

        try:
            self._run_text((*adb, "version"), worktree, timeout_seconds)
            self._run_text((*adb, "reverse", "tcp:8081", "tcp:8081"), worktree, timeout_seconds)

            def capture_ready_app() -> None:
                if deep_link.strip():
                    self._run_text(
                        (
                            *adb,
                            "shell",
                            "am",
                            "start",
                            "-a",
                            "android.intent.action.VIEW",
                            "-d",
                            deep_link.strip(),
                        ),
                        worktree,
                        timeout_seconds,
                    )
                screenshot.write_bytes(
                    self._run_bytes((*adb, "exec-out", "screencap", "-p"), worktree, timeout_seconds)
                )

            package = android_package.strip()
            if package:
                self._run_text((*adb, "shell", "am", "force-stop", package), worktree, timeout_seconds)
                self._run_android_until_foreground(
                    ("npm", "run", "android"),
                    worktree,
                    timeout_seconds,
                    adb,
                    package,
                    capture_ready_app,
                )
            else:
                self._run_text(("npm", "run", "android"), worktree, timeout_seconds)
                capture_ready_app()
        finally:
            emulator_cleanup()

        if not screenshot.is_file() or screenshot.stat().st_size == 0:
            raise RuntimeError(f"Android emulator screenshot was not created: {screenshot}")
        _assert_screenshot_has_visual_content(screenshot)
        return str(screenshot)


def _clean_platforms(platforms: Sequence[str]) -> tuple[str, ...]:
    values = tuple(str(platform).strip().lower() for platform in platforms if str(platform).strip())
    return values or ("ios", "android")


def _run_text_command(args: tuple[str, ...], cwd: Path, timeout: int) -> str:
    proc = _run(args, cwd, timeout, capture_bytes=False)
    return str(proc.stdout or "")


def _run_bytes_command(args: tuple[str, ...], cwd: Path, timeout: int) -> bytes:
    proc = _run(args, cwd, timeout, capture_bytes=True)
    return bytes(proc.stdout or b"")


def _run(
    args: tuple[str, ...],
    cwd: Path,
    timeout: int,
    *,
    capture_bytes: bool,
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    try:
        proc = subprocess.run(
            list(args),
            cwd=str(cwd),
            text=not capture_bytes,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"executable not found: {args[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"command timed out: {' '.join(args)}") from exc
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace") if isinstance(proc.stderr, bytes) else proc.stderr
        stdout = proc.stdout.decode("utf-8", errors="replace") if isinstance(proc.stdout, bytes) else proc.stdout
        raise RuntimeError(
            "\n".join(
                part
                for part in [
                    f"command failed ({proc.returncode}): {' '.join(args)}",
                    f"stdout: {str(stdout or '').strip()[-2000:]}",
                    f"stderr: {str(stderr or '').strip()[-2000:]}",
                ]
                if part
            )
        )
    return proc


def _ensure_android_emulator(
    avd: str,
    adb: tuple[str, ...],
    cwd: Path,
    timeout: int,
) -> Callable[[], None]:
    if not avd or _android_device_is_booted(adb, cwd):
        return lambda: None

    stdout_path, stderr_path = _start_log_files()
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_file:
            try:
                proc = subprocess.Popen(
                    [
                        "emulator",
                        "-avd",
                        avd,
                        "-no-snapshot",
                        "-no-audio",
                        "-no-boot-anim",
                        "-no-window",
                        "-gpu",
                        "swiftshader_indirect",
                    ],
                    cwd=str(cwd),
                    env=_android_run_env(),
                    text=True,
                    stdout=stdout_file,
                    stderr=stderr_file,
                )
            except FileNotFoundError as exc:
                raise RuntimeError("executable not found: emulator") from exc

            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if _android_device_is_booted(adb, cwd):
                    return lambda: _shutdown_started_android_emulator(adb, cwd, proc, stdout_path, stderr_path)

                returncode = proc.poll()
                if returncode is not None:
                    raise RuntimeError(
                        "\n".join(
                            [
                                f"Android emulator exited before booting AVD: {avd}",
                                f"exit code: {returncode}",
                                f"stdout: {_tail_text(stdout_path)}",
                                f"stderr: {_tail_text(stderr_path)}",
                            ]
                        )
                    )
                time.sleep(2)

            _terminate_process(proc)
            raise RuntimeError(
                "\n".join(
                    [
                        f"timed out waiting for Android emulator to boot AVD: {avd}",
                        f"stdout: {_tail_text(stdout_path)}",
                        f"stderr: {_tail_text(stderr_path)}",
                    ]
                )
            )
    except Exception:
        stdout_path.unlink(missing_ok=True)
        stderr_path.unlink(missing_ok=True)
        raise


def _android_device_is_booted(adb: tuple[str, ...], cwd: Path) -> bool:
    try:
        proc = subprocess.run(
            [*adb, "shell", "getprop", "sys.boot_completed"],
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and str(proc.stdout or "").strip() == "1"


def _shutdown_started_android_emulator(
    adb: tuple[str, ...],
    cwd: Path,
    proc: subprocess.Popen[str],
    stdout_path: Path,
    stderr_path: Path,
) -> None:
    try:
        subprocess.run(
            [*adb, "emu", "kill"],
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    _terminate_process(proc)
    stdout_path.unlink(missing_ok=True)
    stderr_path.unlink(missing_ok=True)


def _run_android_until_foreground(
    args: tuple[str, ...],
    cwd: Path,
    timeout: int,
    adb: tuple[str, ...],
    package: str,
    while_foreground: Callable[[], None],
) -> None:
    stdout_path, stderr_path = _start_log_files()
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_file:
            try:
                proc = subprocess.Popen(
                    list(args),
                    cwd=str(cwd),
                    env=_android_run_env(),
                    text=True,
                    stdout=stdout_file,
                    stderr=stderr_file,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(f"executable not found: {args[0]}") from exc

            try:
                deadline = time.monotonic() + timeout
                started = time.monotonic()
                last_direct_launch = 0.0
                while time.monotonic() < deadline:
                    if _android_package_is_foreground(adb, cwd, package):
                        time.sleep(min(_android_settle_seconds(), max(deadline - time.monotonic(), 0)))
                        if not _android_package_is_foreground(adb, cwd, package):
                            continue
                        while_foreground()
                        if not _android_package_is_foreground(adb, cwd, package):
                            raise RuntimeError(
                                "\n".join(
                                    [
                                        f"Android package left foreground before proof capture: {package}",
                                        f"command: {' '.join(args)}",
                                        f"stdout: {_tail_text(stdout_path)}",
                                        f"stderr: {_tail_text(stderr_path)}",
                                    ]
                                )
                            )
                        return

                    now = time.monotonic()
                    if now - started >= 5 and now - last_direct_launch >= 5:
                        _launch_android_package(adb, cwd, package)
                        last_direct_launch = now

                    returncode = proc.poll()
                    if returncode is not None:
                        if returncode != 0:
                            raise RuntimeError(
                                "\n".join(
                                    [
                                        f"command failed ({returncode}): {' '.join(args)}",
                                        f"stdout: {_tail_text(stdout_path)}",
                                        f"stderr: {_tail_text(stderr_path)}",
                                    ]
                                )
                            )
                        raise RuntimeError(
                            "\n".join(
                                [
                                    f"Android package did not reach foreground: {package}",
                                    f"command exited successfully before foregrounding: {' '.join(args)}",
                                    f"stdout: {_tail_text(stdout_path)}",
                                    f"stderr: {_tail_text(stderr_path)}",
                                ]
                            )
                        )
                    time.sleep(2)

                raise RuntimeError(
                    "\n".join(
                        [
                            f"timed out waiting for Android package to reach foreground: {package}",
                            f"command: {' '.join(args)}",
                            f"stdout: {_tail_text(stdout_path)}",
                            f"stderr: {_tail_text(stderr_path)}",
                        ]
                    )
                )
            finally:
                _terminate_process(proc)
    finally:
        stdout_path.unlink(missing_ok=True)
        stderr_path.unlink(missing_ok=True)


def _android_package_is_foreground(adb: tuple[str, ...], cwd: Path, package: str) -> bool:
    try:
        proc = subprocess.run(
            [*adb, "shell", "dumpsys", "window"],
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    if proc.returncode != 0:
        return False
    focus_markers = (
        "mCurrentFocus=",
        "mFocusedApp=",
        "mResumedActivity=",
        "topResumedActivity=",
    )
    return any(
        package in line and any(marker in line for marker in focus_markers)
        for line in str(proc.stdout or "").splitlines()
    )


def _launch_android_package(adb: tuple[str, ...], cwd: Path, package: str) -> None:
    activity = _resolve_android_launch_activity(adb, cwd, package)
    if activity:
        try:
            proc = subprocess.run(
                [*adb, "shell", "am", "start", "-n", activity],
                cwd=str(cwd),
                text=True,
                capture_output=True,
                timeout=15,
                check=False,
            )
            if proc.returncode == 0:
                return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    try:
        subprocess.run(
            [
                *adb,
                "shell",
                "monkey",
                "-p",
                package,
                "-c",
                "android.intent.category.LAUNCHER",
                "1",
            ],
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return


def _resolve_android_launch_activity(adb: tuple[str, ...], cwd: Path, package: str) -> str:
    try:
        proc = subprocess.run(
            [*adb, "shell", "cmd", "package", "resolve-activity", "--brief", package],
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    if proc.returncode != 0:
        return ""
    for line in reversed(str(proc.stdout or "").splitlines()):
        value = line.strip()
        if "/" in value and not value.startswith("priority="):
            return value
    return ""


def _terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def _start_log_files() -> tuple[Path, Path]:
    stdout_file = tempfile.NamedTemporaryFile(prefix="monica-android-run-", suffix=".stdout.log", delete=False)
    stderr_file = tempfile.NamedTemporaryFile(prefix="monica-android-run-", suffix=".stderr.log", delete=False)
    stdout_path = Path(stdout_file.name)
    stderr_path = Path(stderr_file.name)
    stdout_file.close()
    stderr_file.close()
    return stdout_path, stderr_path


def _android_run_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("REACT_NATIVE_PACKAGER_HOSTNAME", "127.0.0.1")
    android_sdk = (
        env.get("ANDROID_HOME", "").strip()
        or env.get("ANDROID_SDK_ROOT", "").strip()
        or env.get("MONICA_ANDROID_SDK_DIR", "").strip()
        or _default_android_sdk_dir()
    )
    if android_sdk:
        env.setdefault("ANDROID_HOME", android_sdk)
        env.setdefault("ANDROID_SDK_ROOT", android_sdk)
    return env


def _prepare_android_worktree(worktree: Path) -> None:
    node_modules = worktree / "node_modules"
    if node_modules.exists() or node_modules.is_symlink():
        return

    source = _node_modules_source(worktree)
    if not source:
        return
    try:
        node_modules.symlink_to(source, target_is_directory=True)
    except FileExistsError:
        return
    except OSError as exc:
        raise RuntimeError(f"failed to link node_modules for Android proof: {source}") from exc


def _node_modules_source(worktree: Path) -> Path | None:
    configured = os.environ.get("MONICA_NODE_MODULES_SOURCE", "").strip()
    if configured:
        source = Path(configured).expanduser()
        return source if source.is_dir() else None

    try:
        workspace = worktree.parent.parent
    except IndexError:
        return None
    repos = workspace / "repos"
    if not repos.is_dir():
        return None
    candidates = [path for path in sorted(repos.glob("*/node_modules")) if path.is_dir()]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _default_android_sdk_dir() -> str:
    candidate = Path.home() / "Library" / "Android" / "sdk"
    return str(candidate) if candidate.is_dir() else ""


def _android_settle_seconds() -> int:
    raw_value = os.environ.get("MONICA_ANDROID_SETTLE_SECONDS", "30")
    try:
        return max(int(raw_value), 0)
    except ValueError:
        return 30


def _tail_text(path: Path, limit: int = 2000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()[-limit:]
    except FileNotFoundError:
        return ""


def _split_env_keys(raw_value: str) -> tuple[str, ...]:
    return tuple(key for key in (part.strip() for part in raw_value.replace("\n", ",").split(",")) if key)


def _validate_required_env_keys(worktree: Path, required_env_keys: Sequence[str]) -> None:
    keys = tuple(dict.fromkeys(key.strip() for key in required_env_keys if key.strip()))
    if not keys:
        return

    declared = set(_nonempty_process_env_keys(keys))
    for env_file in _candidate_env_files(worktree):
        declared.update(_nonempty_env_file_keys(env_file, keys))

    missing = [key for key in keys if key not in declared]
    if missing:
        raise RuntimeError(
            "required proof env keys are missing: "
            + ", ".join(missing)
            + ". Add them to the proof environment or the app's local .env files."
        )


def _nonempty_process_env_keys(keys: Sequence[str]) -> tuple[str, ...]:
    return tuple(key for key in keys if os.environ.get(key, "").strip())


def _candidate_env_files(worktree: Path) -> tuple[Path, ...]:
    environment = os.environ.get("ENVIRONMENT", "staging").strip() or "staging"
    env_names = (
        ".env",
        ".env.local",
        f".env.{environment}",
        f".env.{environment}.local",
    )
    roots = (
        worktree,
        worktree / "apps" / "elixir-card",
    )
    candidates = tuple(root / name for root in roots for name in env_names)
    return tuple(dict.fromkeys(candidates))


def _nonempty_env_file_keys(path: Path, keys: Sequence[str]) -> tuple[str, ...]:
    if not path.is_file():
        return ()

    wanted = set(keys)
    found: list[str] = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        clean_key = key.strip().removeprefix("export ").strip()
        clean_value = value.strip().strip("'\"")
        if clean_key in wanted and clean_value:
            found.append(clean_key)
    return tuple(found)


def _assert_screenshot_has_visual_content(path: Path) -> None:
    try:
        from PIL import Image, ImageStat
    except ImportError:
        return

    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            sample_width = 64
            sample_height = max(1, round(sample_width * rgb.height / rgb.width))
            sample = rgb.resize((sample_width, sample_height))
            stats = ImageStat.Stat(sample)
    except Exception as exc:
        raise RuntimeError(f"simulator screenshot is not a readable image: {path}") from exc

    if max(stats.stddev) < 8:
        raise RuntimeError(f"simulator screenshot appears blank or near-uniform: {path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture Monica simulator proof artifacts.")
    parser.add_argument("--worktree", default="")
    parser.add_argument("--proof-dir", default="")
    parser.add_argument("--platform", dest="platforms", action="append", default=[])
    parser.add_argument("--ios-simulator-udid", default="")
    parser.add_argument("--android-serial", default="")
    parser.add_argument("--android-avd", default="")
    parser.add_argument("--android-package", default="")
    parser.add_argument("--deep-link", default="")
    parser.add_argument("--required-env-key", dest="required_env_keys", action="append", default=[])
    parser.add_argument("--timeout-seconds", type=int, default=600)
    args = parser.parse_args(argv)

    worktree = args.worktree or os.getenv("MONICA_WORKTREE", "")
    proof_dir = args.proof_dir or os.getenv("MONICA_PROOF_DIR", "")
    platforms = args.platforms or os.getenv("MONICA_PROOF_PLATFORM_ORDER", "").split(",")
    deep_link = args.deep_link or os.getenv("MONICA_DEEP_LINK", "")
    required_env_keys = tuple(args.required_env_keys) or _split_env_keys(os.getenv("MONICA_REQUIRED_ENV_KEYS", ""))
    if not worktree:
        raise SystemExit("MONICA_WORKTREE is required")
    if not proof_dir:
        raise SystemExit("MONICA_PROOF_DIR is required")

    try:
        artifacts = SimulatorProofHarness().run(
            worktree=worktree,
            proof_dir=proof_dir,
            platforms=platforms,
            ios_simulator_udid=args.ios_simulator_udid or os.getenv("MONICA_IOS_SIMULATOR_UDID", ""),
            android_serial=args.android_serial or os.getenv("MONICA_ANDROID_SERIAL", ""),
            android_avd=args.android_avd or os.getenv("MONICA_ANDROID_AVD", ""),
            android_package=args.android_package or os.getenv("MONICA_ANDROID_PACKAGE", ""),
            deep_link=deep_link,
            required_env_keys=required_env_keys,
            timeout_seconds=max(args.timeout_seconds, 1),
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({"artifacts": artifacts}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
