"""Declaration parser, availability, and skill-gate contracts.

Fixtures are dicts passed to ``parse_declaration`` (never YAML files); fixture paths satisfy the
OS rule they test under (`C:/x/y.exe` for win32, `/x` elsewhere).
"""

from __future__ import annotations

import plistlib
import sys
from pathlib import Path

import pytest

from hermes_platform.declaration import DeclarationError, parse_declaration
from hermes_platform.resolver.app import AppDef, AppResolver
from hermes_platform.resolver.availability import Availability, availability, version_at_least

WHERE = "test-plugin/plugin.yaml"


def _bundle(tmp_path: Path, version: str) -> Path:
    app = tmp_path / "Applications" / "Thing.app"
    (app / "Contents").mkdir(parents=True, exist_ok=True)
    with open(app / "Contents" / "Info.plist", "wb") as fh:
        plistlib.dump({"CFBundleShortVersionString": version}, fh)
    return app


def test_app_block_parses_into_one_appdef_per_os():
    decl = parse_declaration("thing-mcp", {
        "darwin": {"presence": "bundle", "location": "/Applications/Thing.app", "version": {"kind": "plist"}},
        "win32": {
            "presence": "executable",
            "location": "%ProgramFiles%/Thing/thing.exe",
            "version": {"kind": "uninstall_registry", "display_name_prefix": "Thing"},
            "liveness": {"kind": "server_json", "path": "%LOCALAPPDATA%/Thing/server.json", "endpoint_path": "/rpc"},
        },
    }, {"app": True, "min_version": "2.3.0"}, where=WHERE)
    assert decl.app is not None and set(decl.app.per_os) == {"darwin", "win32"}
    mac, win = decl.app_for("darwin"), decl.app_for("win32")
    assert mac is not None and win is not None
    assert mac.presence == "bundle" and mac.version_kind == "plist" and mac.liveness_kind == "none"
    assert win.version_arg == "Thing" and win.endpoint_path == "/rpc" and win.liveness_pid_key == "pid"
    assert decl.requires_app is True and decl.min_version == "2.3.0"


@pytest.mark.parametrize("raw_app, raw_requires, message", [
    (None, {"app": True}, "no 'app' block"),
    (None, {"min_version": "1.0\n"}, "dotted numeric"),
    ({"linux": {"presence": "executable", "location": "/x/y", "version": {"kind": "plist"}}}, None,
     "only valid under app.darwin"),
    ({"win32": {"presence": "executable", "location": "C:/x/y.exe", "version": {"kind": "uninstall_registry"}}}, None,
     "display_name_prefix"),
    ({"freebsd": {"presence": "executable", "location": "/x"}}, None, "unknown OS keys"),
    ({"win32": {"presence": "executable", "location": "C:/x/y.exe"}}, {"min_version": "1.0"},
     "needs requires.app"),
    ({"win32": {"presence": "executable", "location": "C:/x/y.exe"}}, {"app": True, "min_version": "latest"},
     "dotted numeric"),
    ({"win32": {"presence": "executable", "location": "C:/x/y.exe"}}, {"app": True, "min_version": "1.0"},
     "needs a version source"),
    ({"darwin": {"presence": "bundle", "location": "../Up/Thing.app"}}, None, "must be absolute"),
    ({"darwin": {"presence": "bundle", "location": "Relative/Thing.app"}}, None, "must be absolute"),
    ({"darwin": {"presence": "bundle", "location": "https://example.test/Thing.app"}}, None, "must be absolute"),
    ({"win32": {"presence": "executable", "location": "Thing/thing.exe"}}, None, "must be absolute"),
])
def test_invalid_blocks_name_the_rule(raw_app, raw_requires, message):
    with pytest.raises(DeclarationError, match=message):
        parse_declaration("thing-mcp", raw_app, raw_requires, where=WHERE)


def test_availability_no_requirements_does_no_io():
    decl = parse_declaration("thing-mcp", None, None, where=WHERE)
    result = availability(decl, os_family="freebsd")
    assert result.state == "no_requirements" and result.offerable


def test_availability_unsupported_os_when_no_block_for_host():
    decl = parse_declaration(
        "thing-mcp",
        {"win32": {"presence": "executable", "location": "C:/x/y.exe"}},
        {"app": True},
        where=WHERE,
    )
    assert availability(decl, os_family="darwin").state == "unsupported_os"


@pytest.mark.macos_only
def test_availability_missing_then_present_then_version_gate(tmp_path):
    location = tmp_path / "Applications" / "Thing.app"
    decl = parse_declaration(
        "thing-mcp",
        {"darwin": {"presence": "bundle", "location": str(location), "version": {"kind": "plist"}}},
        {"app": True, "min_version": "2.3.0"},
        where=WHERE,
    )
    missing = availability(decl, os_family="darwin")
    assert missing.state == "missing_app" and str(tmp_path) in (missing.path or "")
    _bundle(tmp_path, "2.2.9")
    old = availability(decl, os_family="darwin")
    assert old.state == "version_too_old" and old.version == "2.2.9" and old.min_version == "2.3.0"
    _bundle(tmp_path, "2.3.0")
    ok = availability(decl, os_family="darwin")
    assert ok.state == "available" and ok.version == "2.3.0" and ok.offerable
    (location / "Contents" / "Info.plist").write_bytes(b"invalid plist")
    assert availability(decl, os_family="darwin").state == "version_too_old"


def test_availability_is_not_a_boolean():
    with pytest.raises(TypeError):
        bool(Availability("available"))


@pytest.mark.parametrize("found, minimum, expected", [
    ("2.3.0", "2.3.0", True),
    ("2.3.0-beta", "2.3.0", False),
    ("11.0.9.509", "11.0.9", True),
    ("11.0.9", "11.0.9.509", False),
    ("2.\u0663.0", "2.3.0", False),
    ("1234567890", "2.3.0", False),
])
def test_version_at_least_accepts_only_bounded_ascii_components(found, minimum, expected):
    assert version_at_least(found, minimum) is expected


def test_unexpanded_locations_are_missing_even_when_cwd_contains_them(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for location in ("%HERMES_TEST_UNSET_VAR%/x.exe", "$HERMES_TEST_UNSET_VAR/x"):
        path = tmp_path / location
        path.parent.mkdir(exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
        definition = AppDef("thing", sys.platform, "executable", location)
        assert AppResolver(definition).locate().kind == "missing"


def test_registry_mutations_are_visible_and_notify_once(monkeypatch):
    from hermes_platform import declaration

    decl = parse_declaration("thing-mcp", None, None, where=WHERE)
    changes = []
    monkeypatch.setattr(declaration, "on_change", lambda: changes.append(None))
    declaration.register("thing-mcp", decl)
    assert declaration.lookup("thing-mcp") is decl
    declaration.unregister("thing-mcp")
    assert declaration.lookup("thing-mcp") is None
    declaration.register("other", decl)
    declaration.clear()
    assert declaration.lookup("other") is None
    assert len(changes) == 4


def test_skill_requires_apps_gate(tmp_path, monkeypatch):
    from agent import skill_utils
    from hermes_platform import declaration

    monkeypatch.setattr(declaration, "_REGISTRY", {})
    decl = parse_declaration(
        "thing-mcp",
        {sys.platform: {"presence": "executable", "location": str(tmp_path / "thing")}},
        {"app": True},
        where=WHERE,
    )
    declaration.clear()
    try:
        assert skill_utils.skill_matches_apps({}) is True
        assert skill_utils.skill_matches_apps({"requires_apps": ["unknown"]}) is False
        declaration.register("thing-mcp", decl)
        assert skill_utils.skill_matches_apps({"requires_apps": ["thing-mcp"]}) is False
        (tmp_path / "thing").write_text("fixture", encoding="utf-8")
        assert skill_utils.skill_matches_apps({"requires_apps": ["thing-mcp"]}) is True
    finally:
        declaration.clear()
