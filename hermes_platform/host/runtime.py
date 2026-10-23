"""Cached runtime-environment predicates for WSL, containers, and Termux."""

from __future__ import annotations

import os


def is_termux() -> bool:
    """Return whether this process runs inside Termux."""
    prefix = os.getenv("PREFIX", "")
    return bool(os.getenv("TERMUX_VERSION") or "com.termux/files/usr" in prefix)


_wsl_detected: bool | None = None


def is_wsl() -> bool:
    """Return whether this process runs inside WSL."""
    global _wsl_detected
    if _wsl_detected is None:
        try:
            with open("/proc/version", "r", encoding="utf-8") as f:
                _wsl_detected = "microsoft" in f.read().lower()
        except Exception:
            _wsl_detected = False
    return _wsl_detected


_container_detected: bool | None = None


def is_container() -> bool:
    """Return whether this process runs inside a container."""
    global _container_detected
    if _container_detected is None:
        _container_detected = _detect_container()
    return _container_detected


def _read_proc(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def _proc_file_has_marker(path: str, markers: tuple[str, ...]) -> bool:
    content = _read_proc(path)
    return any(marker in content for marker in markers)


def _detect_container() -> bool:
    if (
        os.path.exists("/.dockerenv")
        or os.path.exists("/run/.containerenv")
        or os.environ.get("KUBERNETES_SERVICE_HOST")
        or _proc_file_has_marker("/proc/1/cgroup", ("docker", "podman", "/lxc/", "kubepods", "containerd", "crio"))
    ):
        return True
    # Under cgroup v2, only the root mount identifies this process's container.
    # Scanning all mountinfo lines misclassifies hosts that run containers (#58135).
    return _root_mount_has_marker("/proc/self/mountinfo", ("kubepods", "containerd", "crio"))


def _root_mount_has_marker(path: str, markers: tuple[str, ...]) -> bool:
    """Return whether the root mount contains any marker."""
    root_lines = [line for line in _read_proc(path).splitlines() if len(f := line.split()) >= 5 and f[4] == "/"]
    return any(marker in line for line in root_lines for marker in markers)
