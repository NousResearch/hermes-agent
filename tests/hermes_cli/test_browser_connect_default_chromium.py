"""Default-Chromium detection and profile-dir resolution (hermes_cli.browser_connect).

These exercise the parsers with real command output shapes instead of
patching the detectors themselves, so a change in what macOS / xdg report is
caught here rather than in a user's browser session.
"""
from unittest.mock import patch

import pytest

import os
import posixpath
import socket

import hermes_cli.browser_connect as bc


def _ls_dump(*entries: str) -> str:
    return "(\n" + ",\n".join(entries) + "\n)\n"


def _handler(scheme: str, bundle: str) -> str:
    return (
        "    {\n"
        "        LSHandlerPreferredVersions =         {\n"
        '            LSHandlerRoleAll = "-";\n'
        "        };\n"
        f'        LSHandlerRoleAll = "{bundle}";\n'
        f"        LSHandlerURLScheme = {scheme};\n"
        "    }"
    )


def _content_type_handler(uti: str, bundle: str) -> str:
    return (
        "    {\n"
        f'        LSHandlerContentType = "{uti}";\n'
        f'        LSHandlerRoleViewer = "{bundle}";\n'
        "    }"
    )


class TestLaunchServicesHttpsHandler:
    def test_https_entry_wins_over_other_schemes(self):
        dump = _ls_dump(
            _handler("ftp", "com.google.chrome"),
            _handler("https", "com.apple.safari"),
        )
        assert bc._launchservices_https_handler(dump) == "com.apple.safari"

    def test_content_type_registration_is_not_an_https_handler(self):
        dump = _ls_dump(_content_type_handler("public.html", "com.google.chrome"))
        assert bc._launchservices_https_handler(dump) is None

    def test_no_entries_means_no_recorded_handler(self):
        assert bc._launchservices_https_handler("(\n)\n") is None
        assert bc._launchservices_https_handler("") is None

    def test_nested_dictionary_does_not_split_the_entry(self):
        dump = _ls_dump(_handler("https", "com.microsoft.edgemac"))
        assert bc._launchservices_https_handler(dump) == "com.microsoft.edgemac"


class _FakeKey(str):
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _FakeWinreg:
    """``winreg`` stand-in: ``values`` maps an https association subkey to its ProgId value;
    a subkey mapped to None exists but has no ProgId (``QueryValueEx`` raises WinError 2)."""
    HKEY_CURRENT_USER = 0

    def __init__(self, values):
        self.values = values

    def OpenKey(self, _root, path):
        sub = path.split("\\https\\", 1)[1]
        if sub not in self.values:
            raise FileNotFoundError(path)
        return _FakeKey(sub)

    def QueryValueEx(self, sub, _name):
        if self.values[sub] is None:
            raise FileNotFoundError(2, "The system cannot find the file specified")
        return self.values[sub], 1

    def CloseKey(self, _key):
        pass


class TestDetectDefaultWindows:
    def test_shell_association_wins_over_stale_legacy_userchoice(self):
        stale = _FakeWinreg({"UserChoice": "VivaldiHTM.ABC"})
        with patch.object(bc, "_windows_shell_progid", return_value="ChromeHTML", create=True), \
                patch.dict("sys.modules", {"winreg": stale}):
            assert bc._detect_default_windows() == "chrome"
        with patch.object(bc, "_windows_shell_progid", return_value="ChromeBHTML", create=True):
            assert bc._detect_default_windows() == bc.UNSUPPORTED_CHANNEL

    @pytest.mark.parametrize("values,expected", [
        # 25H2: Settings writes only UserChoiceLatest\ProgId; legacy key exists without a value.
        ({"UserChoiceLatest\\ProgId": "MSEdgeHTM", "UserChoice": None}, "edge"),
        ({"UserChoiceLatest\\ProgId": "ChromeHTML", "UserChoice": "MSEdgeHTM"}, "chrome"),
        ({"UserChoice": "BraveHTML"}, "brave"),  # pre-25H2
        ({"UserChoice": "FirefoxURL-308046B0AF4A39CB"}, None),
    ])
    def test_registry_fallback_prefers_userchoicelatest(self, values, expected):
        with patch.object(bc, "_windows_shell_progid", return_value=None, create=True), \
                patch.dict("sys.modules", {"winreg": _FakeWinreg(values)}):
            assert bc._detect_default_windows() == expected


class TestDetectDefaultDarwin:
    def _run_with(self, dump: str):
        class _Proc:
            stdout = dump

        return patch.object(bc.subprocess, "run", return_value=_Proc())

    def test_chrome_as_https_handler(self):
        with self._run_with(_ls_dump(_handler("https", "com.google.chrome"))):
            assert bc._detect_default_darwin() == "chrome"

    def test_safari_default_with_chrome_installed_fails_closed(self):
        """The old fallback returned the first installed Chromium app; a
        non-Chromium default must resolve to None even when Chrome exists."""
        dump = _ls_dump(
            _handler("https", "com.apple.safari"),
            _handler("ftp", "com.google.chrome"),
        )
        with self._run_with(dump), \
             patch.object(bc, "chromium_executable", return_value="/Applications/Google Chrome.app/x"):
            assert bc._detect_default_darwin() is None

    def test_no_handler_recorded_fails_closed(self):
        with self._run_with("(\n)\n"), \
             patch.object(bc, "chromium_executable", return_value="/Applications/Google Chrome.app/x"):
            assert bc._detect_default_darwin() is None

    def test_firefox_default_fails_closed(self):
        with self._run_with(_ls_dump(_handler("https", "org.mozilla.firefox"))):
            assert bc._detect_default_darwin() is None

    def test_reader_failure_fails_closed(self):
        with patch.object(bc.subprocess, "run", side_effect=OSError("no defaults")):
            assert bc._detect_default_darwin() is None

    @pytest.mark.parametrize(
        "bundle,expected",
        [
            ("com.google.Chrome", "chrome"),
            ("company.thebrowser.Dia", "dia"),
            ("company.thebrowser.Browser", None),
            ("com.brave.Browser", "brave"),
            ("com.brave.Browser.origin", "brave-origin"),
            ("com.microsoft.edgemac", "edge"),
            ("org.chromium.Chromium", "chromium"),
            ("com.brave.Browser.origin.beta", bc.UNSUPPORTED_CHANNEL),
            ("com.brave.Browser.origin.nightly", bc.UNSUPPORTED_CHANNEL),
        ],
    )
    def test_bundle_map(self, bundle, expected):
        with self._run_with(_ls_dump(_handler("https", bundle))):
            assert bc._detect_default_darwin() == expected


class TestDetectDefaultLinux:
    def _run_with(self, output: str):
        class _Proc:
            stdout = output

        return patch.object(bc.subprocess, "run", return_value=_Proc())

    @pytest.mark.parametrize(
        "desktop,expected",
        [
            ("google-chrome.desktop", "chrome"),
            ("com.google.Chrome.desktop", "chrome"),
            ("chromium_chromium.desktop", "chromium"),
            ("org.chromium.Chromium.desktop", "chromium"),
            ("brave-browser.desktop", "brave"),
            ("com.brave.Browser.desktop", "brave"),
            ("brave-origin.desktop", "brave-origin"),
            ("brave-origin-beta.desktop", bc.UNSUPPORTED_CHANNEL),
            ("brave-origin-nightly.desktop", bc.UNSUPPORTED_CHANNEL),
            ("microsoft-edge.desktop", "edge"),
            ("com.microsoft.Edge.desktop", "edge"),
            ("firefox.desktop", None),
            ("org.mozilla.firefox.desktop", None),
            ("", None),
        ],
    )
    def test_xdg_desktop_names(self, desktop, expected):
        with self._run_with(desktop + "\n"):
            assert bc._detect_default_linux() == expected

    def test_missing_xdg_settings_fails_closed(self):
        with patch.object(bc.subprocess, "run", side_effect=FileNotFoundError("xdg-settings")):
            assert bc._detect_default_linux() is None


class TestLinuxProfileDir:
    def _env(self, monkeypatch, home):
        # The code under test resolves the user's home via
        # os.path.expanduser("~"), which reads HOME on POSIX but USERPROFILE
        # on Windows — patch expanduser to a POSIX-FORM home so the
        # Linux-target path resolution is exercised identically on every
        # host (posixpath.join only inserts '/' between components; a
        # backslash-drive home would leak host separators into the result).
        monkeypatch.setattr(bc.os.path, "expanduser", lambda _p: home.as_posix())
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)

    def test_native_path_when_nothing_exists(self, tmp_path, monkeypatch):
        self._env(monkeypatch, tmp_path)
        assert bc.real_profile_data_dir("chromium", "Linux") == posixpath.join(tmp_path.as_posix(), ".config", "chromium")

    def test_snap_chromium_profile_is_found(self, tmp_path, monkeypatch):
        self._env(monkeypatch, tmp_path)
        snap = tmp_path / "snap" / "chromium" / "common" / "chromium"
        snap.mkdir(parents=True)
        assert bc.real_profile_data_dir("chromium", "Linux") == snap.as_posix()

    def test_flatpak_chrome_profile_is_found(self, tmp_path, monkeypatch):
        self._env(monkeypatch, tmp_path)
        flatpak = tmp_path / ".var" / "app" / "com.google.Chrome" / "config" / "google-chrome"
        flatpak.mkdir(parents=True)
        assert bc.real_profile_data_dir("chrome", "Linux") == flatpak.as_posix()

    def test_native_profile_wins_when_present(self, tmp_path, monkeypatch):
        self._env(monkeypatch, tmp_path)
        native = tmp_path / ".config" / "BraveSoftware" / "Brave-Browser"
        native.mkdir(parents=True)
        (tmp_path / ".var" / "app" / "com.brave.Browser" / "config" / "BraveSoftware" / "Brave-Browser").mkdir(parents=True)
        assert bc.real_profile_data_dir("brave", "Linux") == native.as_posix()

    def test_xdg_config_home_is_honoured(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("XDG_CONFIG_HOME", "/home/t/.config")
        assert bc.real_profile_data_dir("edge", "Linux") == "/home/t/.config/microsoft-edge"


class TestDiaBrowser:
    """Dia (The Browser Company) is Chromium-based but macOS-only, and differs from Chrome in
    two ways that each broke real-profile browsing: it never writes the DevToolsActivePort
    bootstrap file, and it has no Windows/Linux build."""

    def test_bundle_map_hits_dia(self):
        assert bc._classify_default("company.thebrowser.dia", bc._DARWIN_CHANNEL_BUNDLES,
                                    bc._DARWIN_BUNDLE_MAP, str.__eq__) == "dia"

    def test_arc_bundle_does_not_alias_dia(self):
        """Arc shares The Browser Company's prefix and is NOT supported: exact match, not prefix."""
        assert bc._classify_default("company.thebrowser.browser", bc._DARWIN_CHANNEL_BUNDLES,
                                    bc._DARWIN_BUNDLE_MAP, str.__eq__) is None

    def test_profile_nests_under_user_data(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        assert bc.real_profile_data_dir("dia", "Darwin") == str(
            tmp_path / "Library" / "Application Support" / "Dia" / "User Data")

    @pytest.mark.parametrize("system", ["Windows", "Linux"])
    def test_no_profile_dir_off_mac(self, system, tmp_path, monkeypatch):
        """Off-macOS must be None, never LOCALAPPDATA / ~/.config themselves — snapshotting a
        parent dir is the wrong-principal bug."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        assert bc.real_profile_data_dir("dia", system) is None

    @pytest.mark.parametrize("system", ["Windows", "Linux"])
    def test_no_executable_off_mac(self, system):
        assert bc.chromium_executable("dia", system) is None

    def test_never_writes_devtools_port(self):
        """The launch path picks an explicit port and probes HTTP for Dia; the file-driven
        bootstrap would poll for a file Dia never creates and time out."""
        assert bc.browser_writes_devtools_port("dia") is False

    @pytest.mark.parametrize("browser", ["chrome", "chromium", "brave", "brave-origin", "edge"])
    def test_other_browsers_keep_file_bootstrap(self, browser):
        assert bc.browser_writes_devtools_port(browser) is True

    def test_unknown_browser_defaults_to_file_bootstrap(self):
        assert bc.browser_writes_devtools_port(None) is True
        assert bc.browser_writes_devtools_port("nonexistent") is True

    def test_free_tcp_port_is_usable(self):
        port = bc.free_tcp_port()
        assert 1024 <= port <= 65535
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", port))

    def test_dia_nests_user_data(self):
        """Dia appends ``User Data`` to the dir it is handed, so its snapshot must nest or the
        browser resolves an empty child and boots signed out beside a correct snapshot."""
        assert bc.browser_nests_user_data_dir("dia") is True

    @pytest.mark.parametrize("browser", ["chrome", "chromium", "brave", "brave-origin", "edge"])
    def test_chromium_family_does_not_nest(self, browser):
        """Chromium treats --user-data-dir as the profile dir itself."""
        assert bc.browser_nests_user_data_dir(browser) is False

    def test_unknown_browser_does_not_nest(self):
        assert bc.browser_nests_user_data_dir(None) is False
        assert bc.browser_nests_user_data_dir("nonexistent") is False

    def test_snapshot_dir_nests_only_for_dia(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        root = str(tmp_path / "browser-profile")
        assert bc.real_profile_snapshot_dir("dia") == os.path.join(root, "dia", "User Data")
        assert bc.real_profile_snapshot_dir("chrome") == os.path.join(root, "chrome")
