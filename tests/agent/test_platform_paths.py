"""Tests for agent/_platform_paths.py — platform-aware path display."""

from agent._platform_paths import (
    _display_hermes_root_for,
    _display_profile_path_for,
    _is_native_windows,
    _is_wsl,
)


class TestPlatformDetection:
    def test_native_windows_is_nt(self, monkeypatch):
        monkeypatch.setattr("os.name", "nt")
        assert _is_native_windows() is True

    def test_posix_is_not_native_windows(self, monkeypatch):
        monkeypatch.setattr("os.name", "posix")
        assert _is_native_windows() is False

    def test_is_wsl_returns_false_when_proc_version_missing(self, monkeypatch):
        """On non-WSL Linux /proc/version won't contain 'microsoft'."""
        monkeypatch.setattr("builtins.open", _fake_open_linux)
        assert _is_wsl() is False

    def test_is_wsl_returns_true_when_microsoft_in_proc_version(self, monkeypatch):
        monkeypatch.setattr("builtins.open", _fake_open_wsl)
        assert _is_wsl() is True


class TestDisplayHermesRoot:
    def test_windows_returns_localappdata(self):
        assert _display_hermes_root_for("nt") == r"%LOCALAPPDATA%\hermes"

    def test_posix_returns_tilde_hermes(self):
        assert _display_hermes_root_for("posix") == "~/.hermes"


class TestDisplayProfilePath:
    def test_windows_non_default_profile(self):
        result = _display_profile_path_for("nt", "hades")
        assert result == r"%LOCALAPPDATA%\hermes\profiles\hades"

    def test_windows_default_profile_placeholder(self):
        result = _display_profile_path_for("nt", "<name>")
        assert result == r"%LOCALAPPDATA%\hermes\profiles\<name>"

    def test_posix_non_default_profile(self):
        result = _display_profile_path_for("posix", "hades")
        assert result == "~/.hermes/profiles/hades"

    def test_posix_default_profile_placeholder(self):
        result = _display_profile_path_for("posix", "<name>")
        assert result == "~/.hermes/profiles/<name>"


# ── Fake /proc filesystem helpers ───────────────────────────────────────────

_LINUX_PROCFILE = {
    "/proc/version": (
        "Linux version 6.8.0-35-generic (buildd@lcy02-amd64-008) "
        "(gcc (Ubuntu 13.2.0-23ubuntu4) 13.2.0, GNU ld (GNU Binutils for "
        "Ubuntu) 2.42) #35-Ubuntu SMP PREEMPT_DYNAMIC Mon May 20 19:55:16 "
        "UTC 2024"
    ),
    "/proc/sys/kernel/osrelease": "6.8.0-35-generic",
}

_WSL_PROCFILE = {
    "/proc/version": (
        "Linux version 5.15.153.1-microsoft-standard-WSL2 "
        "(root@90410d14-ebbb-408b-9ea7-02f2dacdbce4) "
        "(gcc (GCC) 11.2.0, GNU ld (GNU Binutils) 2.37) "
        "#1 SMP Fri Apr 4 23:40:18 UTC 2025"
    ),
    "/proc/sys/kernel/osrelease": "5.15.153.1-microsoft-standard-WSL2",
}


def _fake_open_linux(path, *args, **kwargs):
    content = _LINUX_PROCFILE.get(str(path), "")
    from io import StringIO
    return StringIO(content)


def _fake_open_wsl(path, *args, **kwargs):
    content = _WSL_PROCFILE.get(str(path), "")
    from io import StringIO
    return StringIO(content)
