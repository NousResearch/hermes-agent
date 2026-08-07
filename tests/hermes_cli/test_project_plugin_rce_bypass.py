"""Regression coverage for GHSA-5qr3-c538-wm9j (#29156) — Remote Code
Execution via the ``HERMES_ENABLE_PROJECT_PLUGINS`` bypass in the web
server's dashboard plugin loader.

Two primitives combined into the original advisory chain:

1. ``hermes_cli.web_server._discover_dashboard_plugins`` opted into
   the untrusted ``./.hermes/plugins/`` source via
   ``os.environ.get("HERMES_ENABLE_PROJECT_PLUGINS")`` — truthy for
   any non-empty string, so ``=0`` / ``=false`` / ``=no`` (all of
   which the agent loader treats as off, and which operators set to
   *disable* project plugins) silently *enabled* the source.
2. ``hermes_cli.web_server._mount_plugin_api_routes`` then imported
   each plugin's manifest ``api`` field as a Python module via
   ``importlib.util.spec_from_file_location``.  The field was used
   raw, with no path-traversal check, so a single manifest line
   ``{"api": "/tmp/payload.py"}`` was enough to redirect the
   importer at any Python file on disk (``Path('safe') / '/abs'``
   resolves to ``/abs`` in Python).

These tests pin each layer of the new defence:

* Truthy env semantics now match the agent loader.
* ``_safe_plugin_api_relpath`` rejects absolute paths, ``..``
  traversal, and non-string / empty values.
* ``_mount_plugin_api_routes`` re-validates at import time and
  refuses project-source plugins outright.
* End-to-end the original PoC manifest no longer triggers
  ``importlib`` for ``/tmp/payload.py``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import web_server
from hermes_cli.plugin_activation import PluginActivationState


@pytest.fixture(autouse=True)
def _reset_plugin_cache(monkeypatch):
    """The plugin scanner caches its result per-process.  Bust the
    cache before *and* after each test so leakage between tests can't
    mask a regression — and so the production cache the import-time
    ``_mount_plugin_api_routes()`` populated doesn't bleed in."""
    web_server._dashboard_plugins_cache = None
    web_server._dashboard_plugins_cache_fingerprint = None
    yield
    web_server._dashboard_plugins_cache = None
    web_server._dashboard_plugins_cache_fingerprint = None


def _write_plugin_manifest(root: Path, name: str, manifest: dict) -> Path:
    """Drop a manifest under ``root/<name>/dashboard/manifest.json`` and
    return the dashboard dir path."""
    dashboard_dir = root / name / "dashboard"
    dashboard_dir.mkdir(parents=True)
    (dashboard_dir / "manifest.json").write_text(json.dumps(manifest))
    return dashboard_dir


def _write_runtime_plugin(
    root: Path,
    directory: str,
    *,
    runtime_name: str,
    dashboard_name: str | None = None,
    with_api: bool = False,
) -> Path:
    """Create a flat runtime plugin, optionally with a dashboard extension."""
    plugin_root = root / directory
    plugin_root.mkdir(parents=True)
    (plugin_root / "plugin.yaml").write_text(
        f"name: {runtime_name}\nkind: backend\nversion: 1.0.0\n"
    )
    if dashboard_name is not None:
        dashboard_dir = plugin_root / "dashboard"
        dashboard_dir.mkdir()
        dist_dir = dashboard_dir / "dist"
        dist_dir.mkdir()
        (dist_dir / "index.js").write_text("console.log('winner');")
        manifest = {
            "name": dashboard_name,
            "entry": "dist/index.js",
        }
        if with_api:
            (dashboard_dir / "api.py").write_text(
                "from fastapi import APIRouter\nrouter = APIRouter()\n"
            )
            manifest["api"] = "api.py"
        (dashboard_dir / "manifest.json").write_text(json.dumps(manifest))
    return plugin_root


# ---------------------------------------------------------------------------
# Layer 1 — HERMES_ENABLE_PROJECT_PLUGINS env gate uses truthy semantics.
# ---------------------------------------------------------------------------


class TestProjectPluginsEnvGate:
    """Project plugins must only be discovered when the env var is set
    to a documented truthy value.  Pre-#29156 any non-empty string —
    including ``0`` / ``false`` / ``no`` — silently enabled the source."""

    @pytest.fixture
    def project_plugin(self, tmp_path, monkeypatch):
        """Plant a project-source plugin under CWD's ``.hermes/plugins``
        and isolate the user-plugins dir to an empty tmp tree."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        (tmp_path / "home").mkdir()
        cwd = tmp_path / "evil-repo"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        _write_plugin_manifest(
            cwd / ".hermes" / "plugins",
            "evil",
            {
                "name": "evil",
                "label": "Evil",
                "entry": "dist/index.js",
            },
        )
        return cwd

    @pytest.mark.parametrize("value", ["", "0", "false", "FALSE", "no", "off", "False"])
    def test_falsy_values_keep_project_plugins_disabled(
        self, project_plugin, monkeypatch, value
    ):
        if value == "":
            monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)
        else:
            monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", value)

        plugins = web_server._get_dashboard_plugins(force_rescan=True)
        names = {p["name"] for p in plugins}
        assert "evil" not in names, (
            f"HERMES_ENABLE_PROJECT_PLUGINS={value!r} must NOT enable the "
            "project source — that's the GHSA-5qr3-c538-wm9j env bypass."
        )

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "YES"])
    def test_truthy_values_enable_project_plugins(
        self, project_plugin, monkeypatch, value
    ):
        monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", value)
        plugins = web_server._get_dashboard_plugins(force_rescan=True)
        evil = next((p for p in plugins if p["name"] == "evil"), None)
        assert evil is not None
        assert evil["source"] == "project"


class TestDashboardDiscoveryScopeAndWinners:
    def test_populated_project_cache_is_invalidated_on_cwd_change(
        self,
        tmp_path,
        monkeypatch,
    ):
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "1")
        old_cwd = tmp_path / "old-project"
        new_cwd = tmp_path / "new-project"
        old_cwd.mkdir()
        new_cwd.mkdir()
        _write_plugin_manifest(
            old_cwd / ".hermes" / "plugins",
            "old-extension",
            {"name": "old-extension", "entry": "dist/index.js"},
        )

        monkeypatch.chdir(old_cwd)
        cached = web_server._get_dashboard_plugins(force_rescan=True)
        stale = next(p for p in cached if p["name"] == "old-extension")
        assert Path(stale["_dir"]).is_dir()

        monkeypatch.chdir(new_cwd)
        refreshed = web_server._get_dashboard_plugins()
        assert "old-extension" not in {p["name"] for p in refreshed}
        assert web_server._dashboard_plugin_status(
            stale,
            [],
            PluginActivationState(enabled=frozenset({"old-extension"})),
        ) == "not enabled"

    def test_user_home_and_bundled_root_are_cache_identities(
        self,
        tmp_path,
        monkeypatch,
    ):
        home_one = tmp_path / "home-one"
        home_two = tmp_path / "home-two"
        home_one.mkdir()
        home_two.mkdir()
        _write_plugin_manifest(
            home_one / "plugins",
            "old-user",
            {"name": "old-user", "entry": "dist/index.js"},
        )
        bundled_one = tmp_path / "bundled-one"
        bundled_two = tmp_path / "bundled-two"
        bundled_one.mkdir()
        bundled_two.mkdir()
        _write_plugin_manifest(
            bundled_one,
            "old-bundled",
            {"name": "old-bundled", "entry": "dist/index.js"},
        )

        monkeypatch.setenv("HERMES_HOME", str(home_one))
        with patch(
            "hermes_cli.plugins.get_bundled_plugins_dir",
            return_value=bundled_one,
        ):
            first = web_server._get_dashboard_plugins(force_rescan=True)
        assert {"old-user", "old-bundled"} <= {p["name"] for p in first}

        monkeypatch.setenv("HERMES_HOME", str(home_two))
        with patch(
            "hermes_cli.plugins.get_bundled_plugins_dir",
            return_value=bundled_two,
        ):
            second = web_server._get_dashboard_plugins()
        assert "old-user" not in {p["name"] for p in second}
        assert "old-bundled" not in {p["name"] for p in second}

    def test_project_dashboard_is_last_winner_by_runtime_key(
        self,
        tmp_path,
        monkeypatch,
    ):
        bundled = tmp_path / "bundled"
        user_home = tmp_path / "home"
        project = tmp_path / "project"
        for root, directory, dashboard in (
            (bundled, "bundled-copy", "bundled-ui"),
            (user_home / "plugins", "user-copy", "user-ui"),
            (project / ".hermes" / "plugins", "project-copy", "project-ui"),
        ):
            _write_runtime_plugin(
                root, directory, runtime_name="shared-runtime",
                dashboard_name=dashboard,
            )
        monkeypatch.setenv("HERMES_HOME", str(user_home))
        monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "1")
        monkeypatch.chdir(project)
        activation = PluginActivationState(enabled=frozenset({"shared-runtime"}))

        with patch(
            "hermes_cli.plugins.get_bundled_plugins_dir",
            return_value=bundled,
        ), patch(
            "hermes_cli.config.load_plugin_activation_state",
            return_value=activation,
        ):
            plugins = web_server._get_dashboard_plugins(force_rescan=True)
            shared = [p for p in plugins if p.get("_runtime_key") == "shared-runtime"]
            assert len(shared) == 1
            assert shared[0]["name"] == "project-ui"
            assert shared[0]["source"] == "project"
            assert web_server._dashboard_plugin_status(
                shared[0], activation=activation,
            ) == "enabled"

    @pytest.mark.parametrize(
        ("user_dashboard", "activation", "expected_status", "expected_source"),
        (
            (None, PluginActivationState(), "enabled", "bundled"),
            (
                None,
                PluginActivationState(enabled=frozenset({"shared-runtime"})),
                "not enabled",
                "bundled",
            ),
            (
                "user-ui",
                PluginActivationState(
                    enabled=frozenset({"shared-runtime"}),
                    disabled=frozenset({"shared-runtime"}),
                ),
                "disabled",
                None,
            ),
        ),
        ids=("inactive-user", "enabled-headless-user", "explicit-disable"),
    )
    def test_runtime_candidate_activation_states(
        self, tmp_path, monkeypatch, user_dashboard, activation,
        expected_status, expected_source,
    ):
        bundled = tmp_path / "bundled"
        user_home = tmp_path / "home"
        _write_runtime_plugin(
            bundled,
            "bundled-copy",
            runtime_name="shared-runtime",
            dashboard_name="bundled-ui",
            with_api=True,
        )
        _write_runtime_plugin(
            user_home / "plugins",
            "user-copy",
            runtime_name="shared-runtime",
            dashboard_name=user_dashboard,
        )
        monkeypatch.setenv("HERMES_HOME", str(user_home))
        monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)

        with patch(
            "hermes_cli.plugins.get_bundled_plugins_dir",
            return_value=bundled,
        ), patch(
            "hermes_cli.config.load_plugin_activation_state",
            return_value=activation,
        ):
            plugins = web_server._get_dashboard_plugins(force_rescan=True)
            (shared,) = [p for p in plugins if p.get("_runtime_key") == "shared-runtime"]
            if expected_source is not None:
                assert shared["source"] == expected_source
            assert web_server._dashboard_plugin_status(
                shared, activation=activation
            ) == expected_status

    def test_activation_change_reselects_cached_dashboard_candidate(
        self,
        tmp_path,
        monkeypatch,
    ):
        bundled = tmp_path / "bundled"
        user_home = tmp_path / "home"
        _write_runtime_plugin(
            bundled,
            "bundled-copy",
            runtime_name="shared-runtime",
            dashboard_name="bundled-ui",
        )
        _write_runtime_plugin(
            user_home / "plugins",
            "user-copy",
            runtime_name="shared-runtime",
            dashboard_name="user-ui",
        )
        monkeypatch.setenv("HERMES_HOME", str(user_home))
        state = {"activation": PluginActivationState()}

        with patch(
            "hermes_cli.plugins.get_bundled_plugins_dir",
            return_value=bundled,
        ), patch(
            "hermes_cli.config.load_plugin_activation_state",
            side_effect=lambda: state["activation"],
        ):
            for index, (activation, expected) in enumerate((
                (PluginActivationState(), "bundled-ui"),
                (PluginActivationState(enabled=frozenset({"shared-runtime"})), "user-ui"),
                (PluginActivationState(), "bundled-ui"),
            )):
                state["activation"] = activation
                plugins = web_server._get_dashboard_plugins(force_rescan=index == 0)
                assert expected in {p["name"] for p in plugins}



# ---------------------------------------------------------------------------
# Layer 2 — _safe_plugin_api_relpath rejects path-traversal payloads.
# ---------------------------------------------------------------------------


class TestApiPathSanitizer:
    """Unit-level coverage for the new ``_safe_plugin_api_relpath``
    helper.  Anything that escapes the plugin's dashboard directory
    must come back as ``None``."""

    def _dashboard_dir(self, tmp_path):
        d = tmp_path / "plug" / "dashboard"
        d.mkdir(parents=True)
        return d

    def test_simple_relative_path_accepted(self, tmp_path):
        d = self._dashboard_dir(tmp_path)
        (d / "api.py").write_text("router = None\n")
        assert web_server._safe_plugin_api_relpath("api.py", dashboard_dir=d) == "api.py"


    @pytest.mark.parametrize("payload", [
        "../../../etc/passwd",
        "../neighbour/api.py",
        "../../../../tmp/evil.py",
        "subdir/../../../../etc/passwd",
    ])
    def test_traversal_rejected(self, tmp_path, payload):
        d = self._dashboard_dir(tmp_path)
        assert web_server._safe_plugin_api_relpath(payload, dashboard_dir=d) is None



# ---------------------------------------------------------------------------
# Layer 3 — _discover_dashboard_plugins scrubs ``_api_file`` early.
# ---------------------------------------------------------------------------


class TestDiscoveryScrubsApiField:
    """The cached plugin entry must NEVER carry an unsanitised api path.
    A regression here would re-arm the RCE for any caller that uses
    ``plugin['_api_file']`` directly."""

    @pytest.fixture
    def user_plugin_factory(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)

        def _make(name: str, manifest: dict) -> None:
            _write_plugin_manifest(tmp_path / "plugins", name, manifest)

        return _make

    def test_absolute_api_path_in_manifest_is_scrubbed(self, user_plugin_factory):
        user_plugin_factory("evil", {
            "name": "evil",
            "label": "Evil",
            "api": "/tmp/payload.py",
            "entry": "dist/index.js",
        })
        plugins = web_server._get_dashboard_plugins(force_rescan=True)
        evil = next(p for p in plugins if p["name"] == "evil")
        assert evil["_api_file"] is None
        assert evil["has_api"] is False

    def test_traversal_api_path_in_manifest_is_scrubbed(self, user_plugin_factory):
        user_plugin_factory("traverse", {
            "name": "traverse",
            "label": "Traverse",
            "api": "../../../../tmp/evil.py",
            "entry": "dist/index.js",
        })
        plugins = web_server._get_dashboard_plugins(force_rescan=True)
        entry = next(p for p in plugins if p["name"] == "traverse")
        assert entry["_api_file"] is None
        assert entry["has_api"] is False


# ---------------------------------------------------------------------------
# Layer 4 — _mount_plugin_api_routes refuses project-source + traversal.
# ---------------------------------------------------------------------------


class TestMountApiRoutesRefusesUntrusted:
    """The mount routine is the actual ``importlib`` call site — these
    tests poke synthetic plugin entries directly into the cache and
    assert the importer is *not* invoked."""

    def _payload_plugin(self, tmp_path, *, source: str, api_file: str = "api.py"):
        dash = tmp_path / "plug" / "dashboard"
        dash.mkdir(parents=True)
        # Write a benign router file; the test asserts it's NOT imported
        # regardless of whether it exists, since the source/path checks
        # short-circuit before the importer runs.
        (dash / "api.py").write_text(
            "from fastapi import APIRouter\nrouter = APIRouter()\n"
        )
        return {
            "name": "synthetic",
            "label": "Synthetic",
            "tab": {"path": "/synthetic", "position": "end"},
            "slots": [],
            "entry": "dist/index.js",
            "css": None,
            "has_api": True,
            "source": source,
            "_dir": str(dash),
            "_api_file": api_file,
        }

    def test_project_source_api_is_not_imported(self, tmp_path):
        plugin = self._payload_plugin(tmp_path, source="project")
        with patch.object(
            web_server,
            "_get_dashboard_plugins",
            return_value=[plugin],
        ), patch.object(
            web_server,
            "_dashboard_plugin_is_active",
            return_value=True,
        ), patch("importlib.util.spec_from_file_location") as spec:
            web_server._mount_plugin_api_routes()
        assert spec.call_count == 0, (
            "project-source plugin's api file was imported — "
            "GHSA-5qr3-c538-wm9j defence-in-depth regression"
        )


    def test_traversal_api_caught_at_mount_time(self, tmp_path):
        """Defence-in-depth: if discovery is bypassed (e.g. cache
        tampering), mount-time validation still refuses to import a
        file outside the dashboard dir."""
        plugin = self._payload_plugin(tmp_path, source="user",
                                       api_file="../../../tmp/evil.py")
        with patch.object(
            web_server,
            "_get_dashboard_plugins",
            return_value=[plugin],
        ), patch.object(
            web_server,
            "_dashboard_plugin_is_active",
            return_value=True,
        ), patch("importlib.util.spec_from_file_location") as spec:
            web_server._mount_plugin_api_routes()
        assert spec.call_count == 0


# ---------------------------------------------------------------------------
# Layer 5 — End-to-end: the original PoC manifest no longer triggers RCE.
# ---------------------------------------------------------------------------


class TestEndToEndPocBlocked:
    """Reproduces the original advisory PoC shape: untrusted CWD with a
    manifest pointing ``api`` at an attacker-chosen Python file, with
    ``HERMES_ENABLE_PROJECT_PLUGINS=0`` (so the operator believed the
    project source was disabled).  Post-fix, the importer must never
    be invoked for the payload path, regardless of how the bypass is
    framed (``=0`` truthy-string bypass, absolute path bypass,
    project-source bypass)."""

    def test_full_chain_blocked(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        (tmp_path / "home").mkdir()
        cwd = tmp_path / "evil-repo"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        # The original bypass: operator sets the var to a "disabled"
        # string the web server pre-fix treated as enabled.
        monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "0")
        # Payload: absolute path inside a manifest dropped in CWD.
        payload_py = tmp_path / "payload.py"
        payload_py.write_text("OWNED = True\n")
        _write_plugin_manifest(
            cwd / ".hermes" / "plugins",
            "evil",
            {
                "name": "evil",
                "label": "Evil",
                "api": str(payload_py),
                "entry": "dist/index.js",
            },
        )

        with patch("importlib.util.spec_from_file_location") as spec:
            plugins = web_server._get_dashboard_plugins(force_rescan=True)
            web_server._mount_plugin_api_routes()

        # The project source must stay disabled because ``0`` is no
        # longer truthy.  Even if the operator *had* opted in, the
        # absolute-path api would be scrubbed at discovery, and even
        # if discovery missed it the project-source guard in mount
        # would refuse the import.
        assert "evil" not in {p["name"] for p in plugins}
        # Bundled plugins shipped with the repo may legitimately have
        # ``api`` files and so ``spec_from_file_location`` can fire for
        # those — the regression is specifically that the *payload*
        # path / *evil* module are never targeted.
        for call in spec.call_args_list:
            module_name = call.args[0]
            target = Path(call.args[1])
            assert module_name != "hermes_dashboard_plugin_evil"
            assert target != payload_py
            assert "evil-repo" not in target.parts
        assert "hermes_dashboard_plugin_evil" not in sys.modules
