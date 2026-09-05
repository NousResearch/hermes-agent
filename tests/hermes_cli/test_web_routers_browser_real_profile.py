"""E2E for ``GET /api/tools/browser/real-profile`` — the desktop picker's data source.

Exercises the real FastAPI route against a real temp ``HERMES_HOME``, because the
whole point of the endpoint is config propagation across profile scopes: the desktop
Capabilities surface can configure ANY profile, and each must report the browsing
identity ITS own config.yaml resolves to. A mocked resolver would prove nothing here.
"""

import json

import pytest


def _make_user_data_dir(root, profiles=("Default", "Profile 2"), last_used="Default",
                        names=None):
    names = names or {}
    for prof in profiles:
        (root / prof / "Network").mkdir(parents=True)
        (root / prof / "Cookies").write_text(f"cookies-{prof}")
    (root / "Local State").write_text(json.dumps({
        "profile": {"last_used": last_used,
                    "info_cache": {p: {"name": names.get(p, p)} for p in profiles}}}))
    return root


class TestBrowserRealProfileEndpoint:
    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch, tmp_path, _isolate_hermes_home):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        import hermes_cli.browser_connect as bc
        import hermes_state
        from hermes_constants import get_hermes_home
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
        self.home = get_hermes_home()
        self.client = TestClient(app)
        self.client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

        # Two synthetic browsers on disk so the route has something real to list.
        self.chrome = _make_user_data_dir(tmp_path / "chrome-data",
                                          profiles=("Default", "Profile 2"),
                                          names={"Profile 2": "Personal"})
        self.brave = _make_user_data_dir(tmp_path / "brave-data", profiles=("Default",))
        monkeypatch.setattr(bc, "real_profile_data_dir", lambda browser, system=None: {
            "chrome": str(self.chrome), "brave": str(self.brave)}.get(browser))
        monkeypatch.setattr(bc, "chromium_executable", lambda browser, system=None:
                            "/usr/bin/x" if browser in ("chrome", "brave") else None)
        monkeypatch.setattr(bc, "detect_default_chromium", lambda system=None: "brave")

    def _write_config(self, **browser_keys):
        import yaml
        path = self.home / "config.yaml"
        config = yaml.safe_load(path.read_text()) if path.exists() else {}
        config.setdefault("browser", {}).update(browser_keys)
        path.write_text(yaml.safe_dump(config))

    def test_lists_installed_browsers_and_their_profiles(self):
        body = self.client.get("/api/tools/browser/real-profile").json()

        rows = {row["key"]: row for row in body["browsers"]}
        assert rows["chrome"]["installed"] and rows["chrome"]["has_profile"]
        assert [p["directory"] for p in rows["chrome"]["profiles"]] == ["Default", "Profile 2"]
        assert rows["chrome"]["profiles"][1]["name"] == "Personal"
        # A browser with no data dir is still listed (disabled in the UI), not hidden.
        assert rows["edge"]["has_profile"] is False

    def test_no_credentials_are_exposed(self):
        """Discovery must never leak cookie contents — only names and paths."""
        raw = self.client.get("/api/tools/browser/real-profile").text

        assert "cookies-Default" not in raw and "cookies-Profile 2" not in raw

    def test_unset_config_follows_the_system_default(self):
        body = self.client.get("/api/tools/browser/real-profile").json()

        assert body["detected_default"] == "brave"
        assert body["resolved_browser"] == "brave"
        assert body["pinned_browser"] is None
        assert body["error"] is None

    def test_config_pin_changes_the_resolved_identity(self):
        self._write_config(use_real_profile=True, real_profile_browser="chrome",
                           real_profile_pin="Profile 2")

        body = self.client.get("/api/tools/browser/real-profile").json()

        assert (body["resolved_browser"], body["resolved_profile"]) == ("chrome", "Profile 2")
        assert body["error"] is None

    def test_bad_pin_reports_a_fixable_error_instead_of_falling_back(self):
        self._write_config(use_real_profile=True, real_profile_browser="chrome",
                           real_profile_pin="Profile 99")

        body = self.client.get("/api/tools/browser/real-profile").json()

        assert body["error"] and "Profile 99" in body["error"]
        assert body["resolved_profile"] != "Default", "must not silently use another profile"

    def test_unknown_browser_reports_an_error(self):
        self._write_config(use_real_profile=True, real_profile_browser="firefox")

        body = self.client.get("/api/tools/browser/real-profile").json()

        assert body["resolved_browser"] is None
        assert body["error"] and "firefox" in body["error"]

    def test_profile_scope_reads_that_profiles_config(self, monkeypatch):
        """The desktop can point Capabilities at another profile; ?profile= must read
        THAT profile's config.yaml, not the active one — this is what makes per-agent
        browsing identities work from a single window."""
        import yaml

        # Active profile: brave/Default (default resolution, no pins).
        other = self.home / "profiles" / "omar"
        other.mkdir(parents=True, exist_ok=True)
        (other / "config.yaml").write_text(yaml.safe_dump(
            {"browser": {"use_real_profile": True, "real_profile_browser": "chrome",
                         "real_profile_pin": "Profile 2"}}))

        active = self.client.get("/api/tools/browser/real-profile").json()
        scoped = self.client.get("/api/tools/browser/real-profile?profile=omar").json()

        assert active["resolved_browser"] == "brave"
        assert (scoped["resolved_browser"], scoped["resolved_profile"]) == ("chrome", "Profile 2")
        assert active["resolved_browser"] != scoped["resolved_browser"], \
            "two profiles on one gateway must be able to browse as different identities"

    def test_response_shape_is_stable_for_the_picker(self):
        """The desktop types this response; every key it reads must be present."""
        body = self.client.get("/api/tools/browser/real-profile").json()

        for key in ("supported", "platform", "detected_default", "detected_unsupported_channel",
                    "resolved_browser", "resolved_profile", "pinned_browser", "pinned_profile",
                    "error", "browsers"):
            assert key in body, f"picker reads {key}"
        for row in body["browsers"]:
            assert set(row) >= {"key", "label", "installed", "has_profile", "is_system_default",
                                "data_dir", "profiles"}
