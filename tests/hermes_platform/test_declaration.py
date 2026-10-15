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
from hermes_platform.resolver.app import AppDef, AppLocation, AppResolver
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
    (None, {"gpu": "amd"}, "requires.gpu must be one of"),
    ({"darwin": {"presence": "bundle", "location": "/Applications/Thing */**/Thing.app"}}, None, "must be absolute"),
    ({"linux": {"presence": "executable", "location": []}}, None, "location is required"),
    ({"linux": {"presence": "executable", "location": [{"kind": "registry", "name": "thing"}]}}, None,
     "must be a path or a mapping with kind one of"),
    ({"linux": {"presence": "executable", "location": [{"kind": "app_bundle", "name": "Thing.app"}]}}, None,
     "only valid under app.darwin"),
    ({"darwin": {"presence": "bundle", "location": [{"kind": "command", "name": "thing"}]}}, None,
     "needs app.darwin.presence executable"),
    ({"linux": {"presence": "executable", "location": [{"kind": "snap", "name": "../thing"}]}}, None,
     "location\\[0\\].name must be a bare name"),
    ({"win32": {"presence": "executable", "location": [{"kind": "uninstall_registry", "display_name_prefix": "Thing"}]}},
     None, "location\\[0\\].file must be a relative path"),
    ({"linux": {"presence": "executable", "location": "/usr/bin/thing"}}, {"app": True, "min_version": "1.0"},
     "needs a version source under app.linux"),
])
def test_invalid_blocks_name_the_rule(raw_app, raw_requires, message):
    with pytest.raises(DeclarationError, match=message):
        parse_declaration("thing-mcp", raw_app, raw_requires, where=WHERE)


def test_availability_no_requirements_does_no_io():
    decl = parse_declaration("thing-mcp", None, None, where=WHERE)
    result = availability(decl, os_family="freebsd")
    assert result.state == "no_requirements" and result.offerable


@pytest.mark.parametrize("gpu_class, state", [
    ("nvidia", "no_requirements"),
    ("unknown", "no_requirements"),
    ("intel", "unsupported_gpu"),
    ("apple_silicon", "unsupported_gpu"),
    ("none", "unsupported_gpu"),
])
def test_requires_gpu_gates_on_the_host_gpu_class(monkeypatch, gpu_class, state):
    """No `app:` block is needed; an unreadable GPU passes; the OS gate runs first and reads no GPU."""
    from hermes_platform.host import facts

    monkeypatch.setattr(facts, "gpu_class", lambda: gpu_class)
    decl = parse_declaration("thing-mcp", None, {"gpu": "nvidia"}, where=WHERE)
    result = availability(decl, os_family="win32")
    assert result.state == state and result.offerable == (state == "no_requirements")

    monkeypatch.setattr(facts, "gpu_class", lambda: pytest.fail("the OS gate must not read the GPU"))
    both = parse_declaration("thing-mcp", {"win32": {"presence": "executable", "location": "C:/x/y.exe"}},
                             {"app": True, "gpu": "nvidia"}, where=WHERE)
    assert availability(both, os_family="darwin").state == "unsupported_os"


def test_availability_unsupported_os_when_no_block_for_host():
    decl = parse_declaration(
        "thing-mcp",
        {"win32": {"presence": "executable", "location": "C:/x/y.exe"}},
        {"app": True},
        where=WHERE,
    )
    assert availability(decl, os_family="darwin").state == "unsupported_os"


@pytest.mark.platforms("macos")
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


def test_location_list_parses_each_kind_in_order():
    decl = parse_declaration("thing-mcp", {
        "win32": {"presence": "executable", "location": [
            {"kind": "uninstall_registry", "display_name_prefix": "Thing", "file": "bin/thing.exe"},
            "%ProgramFiles%/Thing */thing.exe",
        ]},
        "linux": {"presence": "executable", "location": [
            {"kind": "command", "name": "thing"}, {"kind": "flatpak", "app_id": "org.thing.Thing"},
        ]},
    }, None, where=WHERE)
    win, linux = decl.app_for("win32"), decl.app_for("linux")
    assert win is not None and linux is not None
    assert win.locations == (AppLocation("uninstall_registry", "Thing", "bin/thing.exe"),
                             AppLocation("path", "%ProgramFiles%/Thing */thing.exe"))
    assert linux.locations == (AppLocation("command", "thing"), AppLocation("flatpak", "org.thing.Thing"))


def test_min_version_lets_a_linux_block_go_without_a_version_source():
    decl = parse_declaration("thing-mcp", {
        "darwin": {"presence": "bundle", "location": "/Applications/Thing.app", "version": {"kind": "plist"}},
        "linux": {"presence": "executable", "location": [{"kind": "command", "name": "thing"}]},
    }, {"app": True, "min_version": "2.0"}, where=WHERE)
    linux = decl.app_for("linux")
    assert decl.min_version == "2.0" and linux is not None and linux.version_kind == "none"


@pytest.mark.platforms("macos")
def test_min_version_picks_a_newer_copy_found_after_an_old_one(tmp_path, monkeypatch):
    monkeypatch.setenv("THINGROOT", str(tmp_path))
    _bundle(tmp_path / "old", "1.0")
    _bundle(tmp_path / "new", "2.4")
    decl = parse_declaration("thing-mcp", {"darwin": {
        "presence": "bundle",
        "location": ["$THINGROOT/old/Applications/Thing.app", "$THINGROOT/new/Applications/Thing.app"],
        "version": {"kind": "plist"},
    }}, {"app": True, "min_version": "2.0"}, where=WHERE)
    ok = availability(decl, os_family="darwin")
    assert ok.state == "available" and ok.version == "2.4" and str(tmp_path / "new") in (ok.path or "")
    too_old = parse_declaration("thing-mcp", {"darwin": {
        "presence": "bundle", "location": "$THINGROOT/old/Applications/Thing.app", "version": {"kind": "plist"},
    }}, {"app": True, "min_version": "2.0"}, where=WHERE)
    assert availability(too_old, os_family="darwin").version == "1.0"


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
        definition = AppDef("thing", sys.platform, "executable", (AppLocation("path", location),))
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
