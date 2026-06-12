from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Sequence


RunText = Callable[[tuple[str, ...], Path, int], str]
RunBytes = Callable[[tuple[str, ...], Path, int], bytes]
RunAndroidUntilForeground = Callable[
    [tuple[str, ...], Path, int, tuple[str, ...], str, Path | None, Callable[[], None], Callable[[], None]],
    None,
]
RunIosUntilReady = Callable[
    [tuple[str, ...], Path, int, str, str, str, Path | None, Callable[[], None]],
    None,
]


class SimulatorProofHarness:
    def __init__(
        self,
        *,
        run_text: RunText | None = None,
        run_bytes: RunBytes | None = None,
        run_android_until_foreground: RunAndroidUntilForeground | None = None,
        run_ios_until_ready: RunIosUntilReady | None = None,
    ) -> None:
        self._run_text = run_text or _run_text_command
        self._run_bytes = run_bytes or _run_bytes_command
        self._run_android_until_foreground = run_android_until_foreground or _run_android_until_foreground
        self._run_ios_until_ready = run_ios_until_ready or _run_ios_until_ready

    def run(
        self,
        *,
        worktree: str | Path,
        proof_dir: str | Path,
        platforms: Sequence[str],
        dev_client_scheme: str = "",
        ios_simulator_udid: str = "",
        ios_bundle_id: str = "",
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
        _prepare_react_native_worktree(worktree_path)
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
                        dev_client_scheme=dev_client_scheme,
                        bundle_id=ios_bundle_id,
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
                        dev_client_scheme=dev_client_scheme,
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
        dev_client_scheme: str,
        bundle_id: str,
        deep_link: str,
        timeout_seconds: int,
    ) -> str:
        _prepare_react_native_worktree(worktree)
        target = simulator_udid.strip() or "booted"
        screenshot = proof_dir / "ios-screenshot.png"
        self._run_text(("xcrun", "--find", "simctl"), worktree, timeout_seconds)
        self._run_text(("xcodebuild", "-version"), worktree, timeout_seconds)
        app_dir = _expo_project_dir(worktree)
        packager_host = _ios_packager_hostname()
        packager_url = _packager_url(packager_host, 8081)
        restore_fingerprint_config = _install_temporary_ios_fast_fingerprint_config(app_dir)
        try:
            self._prepare_ios_native_project(app_dir, timeout_seconds)

            def capture_ready_app() -> None:
                if deep_link.strip():
                    self._run_text(("xcrun", "simctl", "openurl", target, deep_link.strip()), worktree, timeout_seconds)
                    time.sleep(_ios_settle_seconds())
                self._run_text(("xcrun", "simctl", "io", target, "screenshot", str(screenshot)), worktree, timeout_seconds)

            clean_bundle_id = bundle_id.strip()
            if clean_bundle_id:
                _uninstall_ios_bundle(target, app_dir, clean_bundle_id)
                try:
                    self._run_text(
                        ("npx", "expo", "run:ios", "--no-install", "--no-bundler"),
                        app_dir,
                        timeout_seconds,
                    )
                except RuntimeError:
                    # expo run:ios can fail after build+install on its cosmetic
                    # Simulator.app activation step (osascript is unavailable in
                    # headless contexts). The uninstall above guarantees an
                    # installed app reflects this build, so only re-raise when
                    # the install never happened.
                    if not _ios_app_is_installed(target, app_dir, clean_bundle_id):
                        raise
                self._run_ios_until_ready(
                    ("npx", "expo", "start", "--dev-client", "--host", _ios_expo_host(), "--port", "8081"),
                    app_dir,
                    timeout_seconds,
                    target,
                    clean_bundle_id,
                    _expo_dev_client_url(dev_client_scheme, packager_url),
                    proof_dir,
                    capture_ready_app,
                )
            else:
                self._run_text(("npx", "expo", "run:ios", "--no-install"), app_dir, timeout_seconds)
                capture_ready_app()
        finally:
            restore_fingerprint_config()
        if not screenshot.is_file():
            raise RuntimeError(f"iOS simulator screenshot was not created: {screenshot}")
        _assert_screenshot_has_visual_content(screenshot)
        return str(screenshot)

    def _prepare_ios_native_project(self, app_dir: Path, timeout_seconds: int) -> None:
        self._run_text(("npx", "expo", "prebuild", "--platform", "ios", "--no-install"), app_dir, timeout_seconds)
        ios_dir = app_dir / "ios"
        self._run_text(("pod", "install", "--ansi"), ios_dir, timeout_seconds)
        _patch_ios_fmt_cxx_standard(ios_dir)

    def _capture_android(
        self,
        *,
        worktree: Path,
        proof_dir: Path,
        android_serial: str,
        android_avd: str,
        android_package: str,
        dev_client_scheme: str,
        deep_link: str,
        timeout_seconds: int,
    ) -> str:
        _prepare_react_native_worktree(worktree)
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

            def open_dev_client() -> None:
                if dev_client_scheme.strip():
                    _open_android_url(
                        self._run_text,
                        adb,
                        worktree,
                        timeout_seconds,
                        _expo_dev_client_url(dev_client_scheme, _packager_url(_android_packager_hostname(), 8081)),
                        android_package.strip(),
                    )
                    time.sleep(_android_settle_seconds())

            def capture_ready_app() -> None:
                if deep_link.strip():
                    _open_android_url(
                        self._run_text,
                        adb,
                        worktree,
                        timeout_seconds,
                        deep_link.strip(),
                        android_package.strip(),
                    )
                    time.sleep(_android_settle_seconds())
                screenshot.write_bytes(
                    self._run_bytes((*adb, "exec-out", "screencap", "-p"), worktree, timeout_seconds)
                )
                if android_package.strip():
                    _assert_android_not_expo_dev_client_launcher(self._run_text, adb, worktree, timeout_seconds)

            package = android_package.strip()
            if package:
                self._run_text((*adb, "shell", "am", "force-stop", package), worktree, timeout_seconds)
                self._run_android_until_foreground(
                    ("npm", "run", "android"),
                    worktree,
                    timeout_seconds,
                    adb,
                    package,
                    proof_dir,
                    open_dev_client,
                    capture_ready_app,
                    open_dev_client_before_foreground=bool(dev_client_scheme.strip()),
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
            env=_simulator_run_env(),
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


def _run_ios_until_ready(
    args: tuple[str, ...],
    cwd: Path,
    timeout: int,
    target: str,
    bundle_id: str,
    dev_client_url: str,
    log_dir: Path | None,
    while_ready: Callable[[], None],
) -> None:
    stdout_path, stderr_path = _start_log_files("ios-metro-", log_dir)
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_file:
            try:
                proc = subprocess.Popen(
                    list(args),
                    cwd=str(cwd),
                    env=_ios_metro_env(),
                    text=True,
                    stdout=stdout_file,
                    stderr=stderr_file,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(f"executable not found: {args[0]}") from exc

            try:
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    if (
                        _ios_app_is_installed(target, cwd, bundle_id)
                        and _ios_metro_is_ready()
                        and _terminate_ios_bundle(target, cwd, bundle_id)
                        and _launch_ios_bundle(target, cwd, bundle_id, dev_client_url, log_dir)
                    ):
                        # The dev client gets the URL exactly once per launch
                        # (re-delivery aborts an in-flight load). If a launch
                        # attempt produces no Metro bundle request in time, do
                        # a clean terminate -> relaunch cycle instead.
                        attempt_deadline = min(
                            deadline, time.monotonic() + _ios_bundle_wait_seconds()
                        )
                        if not _metro_bundle_requested_before(
                            stdout_path, stderr_path, "ios", attempt_deadline
                        ):
                            continue
                        time.sleep(min(_ios_settle_seconds(), max(deadline - time.monotonic(), 0)))
                        while_ready()
                        return

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
                                    f"iOS bundle did not become launchable: {bundle_id}",
                                    f"command exited successfully before proof capture: {' '.join(args)}",
                                    f"stdout: {_tail_text(stdout_path)}",
                                    f"stderr: {_tail_text(stderr_path)}",
                                ]
                            )
                        )
                    time.sleep(2)

                raise RuntimeError(
                    "\n".join(
                        [
                            f"timed out waiting for the iOS app to load a Metro bundle: {bundle_id}",
                            f"command: {' '.join(args)}",
                            f"stdout: {_tail_text(stdout_path)}",
                            f"stderr: {_tail_text(stderr_path)}",
                        ]
                    )
                )
            finally:
                _terminate_process(proc)
    finally:
        if log_dir is None:
            stdout_path.unlink(missing_ok=True)
            stderr_path.unlink(missing_ok=True)


def _ios_app_is_installed(target: str, cwd: Path, bundle_id: str) -> bool:
    try:
        proc = subprocess.run(
            ["xcrun", "simctl", "get_app_container", target, bundle_id, "app"],
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and bool(str(proc.stdout or "").strip())


def _ios_metro_is_ready() -> bool:
    if not _ios_wait_for_metro():
        return True
    try:
        with urllib.request.urlopen(_metro_status_url(), timeout=1) as response:
            body = response.read(200).decode("utf-8", errors="replace")
    except Exception:
        return False
    return "packager-status:running" in body or body.strip().lower() == "running"


def _ios_wait_for_metro() -> bool:
    raw_value = os.environ.get("MONICA_IOS_WAIT_FOR_METRO", "1").strip().lower()
    return raw_value not in {"0", "false", "no", "off"}


def _metro_status_url() -> str:
    return os.environ.get("MONICA_METRO_STATUS_URL", "http://localhost:8081/status").strip() or "http://localhost:8081/status"


def _uninstall_ios_bundle(target: str, cwd: Path, bundle_id: str) -> None:
    try:
        subprocess.run(
            ["xcrun", "simctl", "uninstall", target, bundle_id],
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return


def _terminate_ios_bundle(target: str, cwd: Path, bundle_id: str) -> bool:
    try:
        subprocess.run(
            ["xcrun", "simctl", "terminate", target, bundle_id],
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return True


def _launch_ios_bundle(
    target: str,
    cwd: Path,
    bundle_id: str,
    dev_client_url: str = "",
    log_dir: Path | None = None,
) -> bool:
    _mark_dev_menu_onboarded(target, cwd, bundle_id)
    clean_url = dev_client_url.strip()
    if clean_url:
        return _open_ios_url(target, cwd, clean_url)

    stdout_path, stderr_path = _start_log_files("ios-app-launch-", log_dir)
    args = [
        "xcrun",
        "simctl",
        "launch",
        f"--stdout={stdout_path}",
        f"--stderr={stderr_path}",
        target,
        bundle_id,
    ]

    def _cleanup_logs() -> None:
        if log_dir is None:
            stdout_path.unlink(missing_ok=True)
            stderr_path.unlink(missing_ok=True)

    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        _cleanup_logs()
        return False
    if proc.returncode != 0:
        _cleanup_logs()
        return False
    _cleanup_logs()
    return True


def _mark_dev_menu_onboarded(target: str, cwd: Path, bundle_id: str) -> None:
    # Skip the expo-dev-menu first-launch intro sheet so proof screenshots
    # show the app instead of the developer-menu overlay. Best effort.
    try:
        subprocess.run(
            [
                "xcrun",
                "simctl",
                "spawn",
                target,
                "defaults",
                "write",
                bundle_id,
                "EXDevMenuIsOnboardingFinished",
                "-bool",
                "true",
            ],
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return


def _open_ios_url(target: str, cwd: Path, url: str, attempts: int = 3) -> bool:
    for _attempt in range(max(attempts, 1)):
        time.sleep(min(_ios_settle_seconds(), 2))
        try:
            proc = subprocess.run(
                ["xcrun", "simctl", "openurl", target, url],
                cwd=str(cwd),
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if proc.returncode == 0:
            return True
    return False


def _run_android_until_foreground(
    args: tuple[str, ...],
    cwd: Path,
    timeout: int,
    adb: tuple[str, ...],
    package: str,
    log_dir: Path | None,
    open_dev_client: Callable[[], None],
    capture_foreground: Callable[[], None],
    open_dev_client_before_foreground: bool = False,
) -> None:
    stdout_path, stderr_path = _start_log_files("android-metro-", log_dir)
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
                last_dev_client_launch = 0.0
                dev_client_delivered = False
                while time.monotonic() < deadline:
                    if _android_package_is_foreground(adb, cwd, package):
                        time.sleep(min(_android_settle_seconds(), max(deadline - time.monotonic(), 0)))
                        if not _android_package_is_foreground(adb, cwd, package):
                            continue
                        if not dev_client_delivered:
                            open_dev_client()
                            dev_client_delivered = True
                        _wait_for_metro_bundle_request(
                            stdout_path,
                            stderr_path,
                            "android",
                            deadline,
                            proc=proc,
                            args=args,
                        )
                        time.sleep(min(_android_settle_seconds(), max(deadline - time.monotonic(), 0)))
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
                        capture_foreground()
                        if not _android_package_is_foreground(adb, cwd, package):
                            raise RuntimeError(
                                "\n".join(
                                    [
                                        f"Android package left foreground during proof capture: {package}",
                                        f"command: {' '.join(args)}",
                                        f"stdout: {_tail_text(stdout_path)}",
                                        f"stderr: {_tail_text(stderr_path)}",
                                    ]
                                )
                            )
                        return

                    now = time.monotonic()
                    if (
                        open_dev_client_before_foreground
                        and now - started >= 5
                        and now - last_dev_client_launch >= 5
                        and _android_package_is_installed(adb, cwd, package)
                    ):
                        open_dev_client()
                        dev_client_delivered = True
                        last_dev_client_launch = now
                        continue
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
        if log_dir is None:
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


def _android_package_is_installed(adb: tuple[str, ...], cwd: Path, package: str) -> bool:
    try:
        proc = subprocess.run(
            [*adb, "shell", "pm", "path", package],
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and str(proc.stdout or "").strip().startswith("package:")


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


def _open_android_url(
    run_text: RunText,
    adb: tuple[str, ...],
    cwd: Path,
    timeout: int,
    url: str,
    package: str = "",
) -> None:
    args = [
        *adb,
        "shell",
        "am",
        "start",
        "-a",
        "android.intent.action.VIEW",
        "-d",
        shlex.quote(url),
    ]
    if package.strip():
        args.extend(("-p", shlex.quote(package.strip())))
    run_text(tuple(args), cwd, timeout)


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


def _assert_android_not_expo_dev_client_launcher(
    run_text: RunText,
    adb: tuple[str, ...],
    cwd: Path,
    timeout: int,
) -> None:
    try:
        ui_tree = run_text((*adb, "exec-out", "uiautomator", "dump", "/dev/tty"), cwd, timeout)
    except RuntimeError:
        return

    normalized = " ".join(str(ui_tree or "").lower().split())
    if "development servers" in normalized and (
        "new development server" in normalized
        or "recently opened" in normalized
        or "expo" in normalized
    ):
        raise RuntimeError("Android proof captured Expo Dev Client launcher instead of real app UI.")


def _terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def _start_log_files(prefix: str = "monica-android-run-", log_dir: Path | None = None) -> tuple[Path, Path]:
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        stem = prefix.rstrip("-") or "monica-run"
        stdout_path = log_dir / f"{stem}.stdout.log"
        stderr_path = log_dir / f"{stem}.stderr.log"
        stdout_path.unlink(missing_ok=True)
        stderr_path.unlink(missing_ok=True)
        stdout_path.touch()
        stderr_path.touch()
        return stdout_path, stderr_path

    stdout_file = tempfile.NamedTemporaryFile(prefix=prefix, suffix=".stdout.log", delete=False)
    stderr_file = tempfile.NamedTemporaryFile(prefix=prefix, suffix=".stderr.log", delete=False)
    stdout_path = Path(stdout_file.name)
    stderr_path = Path(stderr_file.name)
    stdout_file.close()
    stderr_file.close()
    return stdout_path, stderr_path


def _ios_bundle_wait_seconds() -> float:
    raw_value = os.environ.get("MONICA_IOS_BUNDLE_WAIT_SECONDS", "").strip()
    try:
        parsed = float(raw_value)
    except ValueError:
        return 90.0
    return parsed if parsed > 0 else 90.0


def _metro_bundle_requested_before(
    stdout_path: Path,
    stderr_path: Path,
    platform: str,
    deadline: float,
) -> bool:
    while time.monotonic() < deadline:
        if _metro_bundle_was_requested(stdout_path, stderr_path, platform):
            return True
        time.sleep(1)
    return _metro_bundle_was_requested(stdout_path, stderr_path, platform)


def _wait_for_metro_bundle_request(
    stdout_path: Path,
    stderr_path: Path,
    platform: str,
    deadline: float,
    *,
    proc: subprocess.Popen[str] | None = None,
    args: tuple[str, ...] = (),
) -> None:
    while time.monotonic() < deadline:
        if _metro_bundle_was_requested(stdout_path, stderr_path, platform):
            return
        if proc is not None:
            _raise_if_process_exited(
                proc,
                args,
                stdout_path,
                stderr_path,
                f"command exited before Metro received a {platform} bundle request: {' '.join(args)}",
            )
        time.sleep(1)
    if _metro_bundle_was_requested(stdout_path, stderr_path, platform):
        return
    if proc is not None:
        _raise_if_process_exited(
            proc,
            args,
            stdout_path,
            stderr_path,
            f"command exited before Metro received a {platform} bundle request: {' '.join(args)}",
        )
    raise RuntimeError(
        "\n".join(
            [
                f"Metro did not receive a {platform} bundle request before proof capture.",
                f"stdout log: {stdout_path}",
                f"stderr log: {stderr_path}",
                f"stdout tail: {_tail_text(stdout_path)}",
                f"stderr tail: {_tail_text(stderr_path)}",
            ]
        )
    )


def _raise_if_process_exited(
    proc: subprocess.Popen[str],
    args: tuple[str, ...],
    stdout_path: Path,
    stderr_path: Path,
    success_message: str,
) -> None:
    returncode = proc.poll()
    if returncode is None:
        return
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
                success_message,
                f"stdout: {_tail_text(stdout_path)}",
                f"stderr: {_tail_text(stderr_path)}",
            ]
        )
    )


def _metro_bundle_was_requested(stdout_path: Path, stderr_path: Path, platform: str) -> bool:
    platform = platform.strip().lower()
    text = "\n".join((_tail_text(stdout_path, 12000), _tail_text(stderr_path, 12000))).lower()
    return any(
        marker in text
        for marker in (
            f"platform={platform}",
            f"{platform} bundled",
            f"{platform} bundle",
            f"bundling `{platform}`",
        )
    )


def _android_run_env() -> dict[str, str]:
    env = _simulator_run_env()
    if not os.environ.get("MONICA_PACKAGER_HOSTNAME", "").strip() and not os.environ.get(
        "REACT_NATIVE_PACKAGER_HOSTNAME", ""
    ).strip():
        env["REACT_NATIVE_PACKAGER_HOSTNAME"] = _android_packager_hostname()
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


def _simulator_run_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("REACT_NATIVE_PACKAGER_HOSTNAME", _default_packager_hostname())
    _prepend_path_dirs(env, _user_gem_bin_dirs())
    _ensure_ruby_logger_preload(env)
    return env


def _default_packager_hostname() -> str:
    configured = os.environ.get("MONICA_PACKAGER_HOSTNAME", "").strip()
    if configured:
        return configured
    return _first_non_loopback_ipv4() or "localhost"


def _android_packager_hostname() -> str:
    return (
        os.environ.get("MONICA_ANDROID_PACKAGER_HOSTNAME", "").strip()
        or os.environ.get("MONICA_PACKAGER_HOSTNAME", "").strip()
        or os.environ.get("REACT_NATIVE_PACKAGER_HOSTNAME", "").strip()
        or "127.0.0.1"
    )


def _ios_packager_hostname() -> str:
    return (
        os.environ.get("MONICA_IOS_PACKAGER_HOSTNAME", "").strip()
        or os.environ.get("MONICA_PACKAGER_HOSTNAME", "").strip()
        or os.environ.get("REACT_NATIVE_PACKAGER_HOSTNAME", "").strip()
        or "localhost"
    )


def _ios_expo_host() -> str:
    return os.environ.get("MONICA_IOS_EXPO_HOST", "").strip() or "localhost"


def _ios_metro_env() -> dict[str, str]:
    # Metro must listen on 127.0.0.1: the Expo manifest hardcodes the literal IP
    # in launchAsset URLs, while Node resolves localhost to ::1 first by default.
    env = _simulator_run_env()
    # The simulator shares the host loopback; the LAN-IP default that
    # _simulator_run_env applies would make the manifest advertise an
    # address Metro is not bound to.
    env["REACT_NATIVE_PACKAGER_HOSTNAME"] = _ios_packager_hostname()
    node_options = env.get("NODE_OPTIONS", "").strip()
    if "--dns-result-order" not in node_options:
        env["NODE_OPTIONS"] = f"{node_options} --dns-result-order=ipv4first".strip()
    return env


def _packager_url(host: str, port: int) -> str:
    clean_host = host.strip() or "localhost"
    if ":" in clean_host and not clean_host.startswith("["):
        clean_host = f"[{clean_host}]"
    return f"http://{clean_host}:{port}"


def _expo_dev_client_url(scheme: str, packager_url: str) -> str:
    clean_scheme = scheme.strip()
    if not clean_scheme:
        return ""
    encoded_url = urllib.parse.quote(packager_url.strip(), safe="")
    return f"{clean_scheme}://expo-development-client/?url={encoded_url}&disableOnboarding=1"


def _first_non_loopback_ipv4() -> str:
    try:
        proc = subprocess.run(
            ["ifconfig"],
            text=True,
            capture_output=True,
            timeout=3,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    if proc.returncode != 0:
        return ""
    for line in str(proc.stdout or "").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0] == "inet" and not parts[1].startswith("127."):
            return parts[1]
    return ""


def _prepend_path_dirs(env: dict[str, str], dirs: Sequence[Path]) -> None:
    prefixes = [str(path) for path in dirs if path.is_dir()]
    if not prefixes:
        return
    current = [part for part in env.get("PATH", "").split(os.pathsep) if part]
    env["PATH"] = os.pathsep.join([*prefixes, *[part for part in current if part not in prefixes]])


def _user_gem_bin_dirs() -> tuple[Path, ...]:
    root = Path.home() / ".gem" / "ruby"
    if not root.is_dir():
        return ()
    return tuple(path for path in sorted(root.glob("*/bin"), reverse=True) if path.is_dir())


def _ensure_ruby_logger_preload(env: dict[str, str]) -> None:
    current = [part for part in env.get("RUBYOPT", "").split() if part]
    if "-rlogger" in current:
        return
    env["RUBYOPT"] = " ".join(("-rlogger", *current))


def _expo_project_dir(worktree: Path) -> Path:
    app_dir = worktree / "apps" / "elixir-card"
    if (app_dir / "package.json").is_file():
        return app_dir
    return worktree


_FINGERPRINT_CONFIG_CONTENT = """\
// Written by the Monica proof harness (not committed; see .git/info/exclude).
// The dev client aborts manifest requests after 10s, and the fingerprint
// runtime-version policy recomputes the project fingerprint per request.
/** @type {import('@expo/fingerprint').Config} */
module.exports = {
  ignorePaths: ['**/*'],
};
"""


def _ensure_ios_fingerprint_config(app_dir: Path) -> None:
    if (app_dir / "fingerprint.config.js").exists() or (app_dir / "fingerprint.config.cjs").exists():
        return
    (app_dir / "fingerprint.config.js").write_text(_FINGERPRINT_CONFIG_CONTENT, encoding="utf-8")


def _install_temporary_ios_fast_fingerprint_config(app_dir: Path) -> Callable[[], None]:
    if os.environ.get("MONICA_IOS_FAST_FINGERPRINT", "1").strip().lower() in {"0", "false", "no", "off"}:
        return lambda: None

    config_path = app_dir / "fingerprint.config.js"
    original_content = config_path.read_bytes() if config_path.exists() else None
    config_path.write_text(_FINGERPRINT_CONFIG_CONTENT, encoding="utf-8")

    def restore() -> None:
        if original_content is None:
            config_path.unlink(missing_ok=True)
            return
        config_path.write_bytes(original_content)

    return restore


def _patch_ios_fmt_cxx_standard(ios_dir: Path) -> None:
    fmt_dir = ios_dir / "Pods" / "Target Support Files" / "fmt"
    for xcconfig in (fmt_dir / "fmt.debug.xcconfig", fmt_dir / "fmt.release.xcconfig"):
        if not xcconfig.is_file():
            continue
        content = xcconfig.read_text(encoding="utf-8")
        patched = content.replace("CLANG_CXX_LANGUAGE_STANDARD = c++20", "CLANG_CXX_LANGUAGE_STANDARD = c++17")
        if patched != content:
            xcconfig.write_text(patched, encoding="utf-8")
    pbxproj_path = ios_dir / "Pods" / "Pods.xcodeproj" / "project.pbxproj"
    _patch_ios_pbxproj_cxx_standard(
        pbxproj_path,
        xcconfig_names=("fmt.debug.xcconfig", "fmt.release.xcconfig"),
        cxx_standard="c++17",
    )
    _patch_ios_mmkvcore_cxx_standard(ios_dir, pbxproj_path)


def _patch_ios_mmkvcore_cxx_standard(ios_dir: Path, pbxproj_path: Path) -> None:
    mmkvcore_dir = ios_dir / "Pods" / "Target Support Files" / "MMKVCore"
    for xcconfig in (mmkvcore_dir / "MMKVCore.debug.xcconfig", mmkvcore_dir / "MMKVCore.release.xcconfig"):
        if not xcconfig.is_file():
            continue
        content = xcconfig.read_text(encoding="utf-8")
        patched = content
        for current_standard in ("c++17", "c++20", "gnu++17"):
            patched = patched.replace(
                f"CLANG_CXX_LANGUAGE_STANDARD = {current_standard}",
                "CLANG_CXX_LANGUAGE_STANDARD = gnu++20",
            )
        if patched != content:
            xcconfig.write_text(patched, encoding="utf-8")
    _patch_ios_pbxproj_cxx_standard(
        pbxproj_path,
        xcconfig_names=("MMKVCore.debug.xcconfig", "MMKVCore.release.xcconfig"),
        cxx_standard="gnu++20",
    )


def _patch_ios_pbxproj_cxx_standard(
    pbxproj_path: Path,
    *,
    xcconfig_names: Sequence[str],
    cxx_standard: str,
) -> None:
    if not pbxproj_path.is_file():
        return

    content = pbxproj_path.read_text(encoding="utf-8")
    markers = tuple(f"/* {name} */" for name in xcconfig_names)
    lines = content.splitlines(keepends=True)
    output: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if "/*" not in line or "*/ = {" not in line:
            output.append(line)
            index += 1
            continue

        block = [line]
        depth = line.count("{") - line.count("}")
        index += 1
        while index < len(lines) and depth > 0:
            block_line = lines[index]
            block.append(block_line)
            depth += block_line.count("{") - block_line.count("}")
            index += 1

        block_text = "".join(block)
        if (
            "isa = XCBuildConfiguration;" in block_text
            and any(marker in block_text for marker in markers)
        ):
            for current_standard in ("c++17", "c++20", "gnu++17", "gnu++20"):
                block_text = block_text.replace(
                    f'CLANG_CXX_LANGUAGE_STANDARD = "{current_standard}";',
                    f'CLANG_CXX_LANGUAGE_STANDARD = "{cxx_standard}";',
                ).replace(
                    f"CLANG_CXX_LANGUAGE_STANDARD = {current_standard};",
                    f"CLANG_CXX_LANGUAGE_STANDARD = {cxx_standard};",
                )
        output.append(block_text)

    patched = "".join(output)
    if patched != content:
        pbxproj_path.write_text(patched, encoding="utf-8")


def _prepare_react_native_worktree(worktree: Path) -> None:
    _exclude_local_worktree_artifacts(worktree)
    _link_sibling_app_env_files(worktree)
    node_modules = worktree / "node_modules"
    if node_modules.exists() and not node_modules.is_dir() and not node_modules.is_symlink():
        return

    source = _node_modules_source(worktree)
    if not source:
        return
    _mirror_node_modules(source, node_modules)


def _mirror_node_modules(source: Path, target: Path) -> None:
    try:
        if target.is_symlink():
            target.unlink()
        target.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            if item.name.startswith("@") and item.is_dir():
                scope_target = target / item.name
                scope_target.mkdir(exist_ok=True)
                for scoped_item in item.iterdir():
                    _symlink_node_module(scoped_item, scope_target / scoped_item.name)
                continue
            _symlink_node_module(item, target / item.name)
    except OSError as exc:
        raise RuntimeError(f"failed to mirror node_modules for simulator proof: {source}") from exc


def _symlink_node_module(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        return
    target.symlink_to(source, target_is_directory=source.is_dir())


def _prepare_android_worktree(worktree: Path) -> None:
    _prepare_react_native_worktree(worktree)


def _exclude_local_worktree_artifacts(worktree: Path) -> None:
    exclude_path = _git_info_exclude_path(worktree)
    if not exclude_path:
        return
    exclude_path.parent.mkdir(parents=True, exist_ok=True)
    existing = exclude_path.read_text(encoding="utf-8") if exclude_path.is_file() else ""
    entries = (
        "/node_modules",
        "/apps/elixir-card/.env",
        "/apps/elixir-card/.env.*",
        "fingerprint.config.js",
    )
    missing = [entry for entry in entries if entry not in set(existing.splitlines())]
    if not missing:
        return
    prefix = "" if not existing or existing.endswith("\n") else "\n"
    exclude_path.write_text(existing + prefix + "\n".join(missing) + "\n", encoding="utf-8")


def _git_info_exclude_path(worktree: Path) -> Path | None:
    git_path = worktree / ".git"
    if git_path.is_dir():
        return _git_common_dir(git_path) / "info" / "exclude"
    if not git_path.is_file():
        return None

    raw_value = git_path.read_text(encoding="utf-8", errors="replace").strip()
    if not raw_value.startswith("gitdir:"):
        return None
    git_dir = Path(raw_value.removeprefix("gitdir:").strip())
    if not git_dir.is_absolute():
        git_dir = (worktree / git_dir).resolve()
    if not git_dir.exists():
        return None
    return _git_common_dir(git_dir) / "info" / "exclude"


def _git_common_dir(git_dir: Path) -> Path:
    common_dir_file = git_dir / "commondir"
    if not common_dir_file.is_file():
        return git_dir
    raw_value = common_dir_file.read_text(encoding="utf-8", errors="replace").strip()
    if not raw_value:
        return git_dir
    common_dir = Path(raw_value)
    if not common_dir.is_absolute():
        common_dir = (git_dir / common_dir).resolve()
    return common_dir if common_dir.exists() else git_dir


def _link_sibling_app_env_files(worktree: Path) -> None:
    source_repo = _source_repo_root(worktree)
    if not source_repo:
        return
    source_app = source_repo / "apps" / "elixir-card"
    target_app = worktree / "apps" / "elixir-card"
    if not source_app.is_dir() or not target_app.is_dir():
        return

    for source in sorted(source_app.glob(".env*")):
        if source.name == ".env.example" or not source.is_file():
            continue
        target = target_app / source.name
        if target.exists() or target.is_symlink():
            continue
        try:
            target.symlink_to(source)
        except FileExistsError:
            continue
        except OSError as exc:
            raise RuntimeError(f"failed to link app env file for simulator proof: {source}") from exc


def _node_modules_source(worktree: Path) -> Path | None:
    configured = os.environ.get("MONICA_NODE_MODULES_SOURCE", "").strip()
    if configured:
        source = Path(configured).expanduser()
        return source if source.is_dir() else None

    source_repo = _source_repo_root(worktree)
    if not source_repo:
        return None
    source = source_repo / "node_modules"
    return source if source.is_dir() else None


def _source_repo_root(worktree: Path) -> Path | None:
    try:
        workspace = worktree.parent.parent
    except IndexError:
        return None
    repos = workspace / "repos"
    if not repos.is_dir():
        return None
    candidates = [path for path in sorted(repos.iterdir()) if (path / "package.json").is_file()]
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


def _ios_settle_seconds() -> int:
    raw_value = os.environ.get("MONICA_IOS_SETTLE_SECONDS", "30")
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
    parser.add_argument("--dev-client-scheme", default="")
    parser.add_argument("--ios-simulator-udid", default="")
    parser.add_argument("--ios-bundle-id", default="")
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
            dev_client_scheme=args.dev_client_scheme or os.getenv("MONICA_DEV_CLIENT_SCHEME", ""),
            ios_simulator_udid=args.ios_simulator_udid or os.getenv("MONICA_IOS_SIMULATOR_UDID", ""),
            ios_bundle_id=args.ios_bundle_id or os.getenv("MONICA_IOS_BUNDLE_ID", ""),
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
