"""Plugin packs (#64166): parse/validate, SHA enforcement, install fan-out,
export round-trip, consent-not-bypassed, partial-failure exit codes.

No live network — index resolution and installs are mocked.
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from unittest import mock

import pytest
import yaml

from hermes_cli.plugin_packs import (
    PackError,
    PackPluginEntry,
    PluginPack,
    ResolvedPackPlugin,
    _sanitized_entry_config as real_sanitized_entry_config,
    cmd_pack_install,
    export_pack,
    install_pack_plugins,
    load_pack,
    parse_pack,
    resolve_pack_plugins,
    validate_config_seed,
)
from hermes_cli.subcommands.plugins import build_plugins_parser

SHA_A = "a" * 40
SHA_B = "b" * 40


class FakeConsole:
    """Minimal Rich-console stand-in recording printed lines."""

    def __init__(self, answers=None):
        self.lines = []
        self._answers = list(answers or [])

    def print(self, *args, **kwargs):
        self.lines.append(" ".join(str(a) for a in args))

    def input(self, prompt=""):
        if self._answers:
            return self._answers.pop(0)
        return ""

    @property
    def text(self):
        return "\n".join(self.lines)


def _pack_yaml(**overrides) -> str:
    doc = {
        "name": "voice-pack",
        "description": "test pack",
        "author": "hyper",
        "version": "1.0.0",
        "plugins": [
            {"repo": "owner/tts-plugin", "ref": SHA_A},
            {"name": "index-plugin", "ref": SHA_B},
        ],
    }
    doc.update(overrides)
    return yaml.safe_dump(doc)


# ---------------------------------------------------------------------------
# Parse / validate
# ---------------------------------------------------------------------------

def test_parse_pack_happy_path():
    pack = parse_pack(_pack_yaml())
    assert pack.name == "voice-pack"
    assert pack.author == "hyper"
    assert len(pack.plugins) == 2
    assert pack.plugins[0].repo == "owner/tts-plugin"
    assert pack.plugins[0].ref == SHA_A
    assert pack.plugins[1].name == "index-plugin"


def test_parse_pack_accepts_nested_pack_meta_and_source_alias():
    text = yaml.safe_dump(
        {
            "pack": {"name": "nested", "version": "2.0.0"},
            "plugins": [
                {"source": "github:owner/repo", "version": SHA_A, "subdir": "sub/dir"}
            ],
        }
    )
    pack = parse_pack(text)
    assert pack.name == "nested"
    assert pack.version == "2.0.0"
    assert pack.plugins[0].repo == "owner/repo"
    assert pack.plugins[0].subdir == "sub/dir"
    assert pack.plugins[0].install_identifier == "owner/repo/sub/dir"


@pytest.mark.parametrize("bad_ref", ["main", "v1.2.0", "a" * 39, "a" * 41, "", None])
def test_parse_pack_rejects_non_sha_refs_naming_the_entry(bad_ref):
    text = yaml.safe_dump(
        {"name": "p", "plugins": [{"repo": "owner/repo", "ref": bad_ref}]}
    )
    with pytest.raises(PackError) as exc:
        parse_pack(text)
    assert "40-character" in str(exc.value)
    assert "owner/repo" in str(exc.value)


def test_parse_pack_normalizes_ref_to_lowercase():
    text = yaml.safe_dump(
        {"name": "p", "plugins": [{"repo": "o/r", "ref": "A" * 40}]}
    )
    assert parse_pack(text).plugins[0].ref == "a" * 40


@pytest.mark.parametrize(
    "doc,fragment",
    [
        ("[]", "mapping"),
        ("name: p", "plugins"),
        ("name: p\nplugins: []", "plugins"),
        ("plugins:\n  - repo: o/r\n    ref: " + SHA_A, "name"),
        ("name: p\nplugins:\n  - ref: " + SHA_A, "either 'name'"),
        ("name: p\nplugins:\n  - not-a-mapping", "mapping"),
        ("{", "YAML"),
    ],
)
def test_parse_pack_rejects_malformed_documents(doc, fragment):
    with pytest.raises(PackError) as exc:
        parse_pack(doc)
    assert fragment in str(exc.value)


def test_config_seed_rejects_secret_shaped_keys():
    for key in ("api_key", "MY_TOKEN", "password", "auth_header", "private_key"):
        with pytest.raises(PackError) as exc:
            validate_config_seed("p", {key: "x"})
        assert "secret" in str(exc.value).lower()


def test_config_seed_rejects_capability_and_trust_gate_keys():
    for key in ("granted_capabilities", "capabilities_consent", "allow_tool_override"):
        with pytest.raises(PackError) as exc:
            validate_config_seed("p", {key: True})
        assert "reserved" in str(exc.value)


def test_parse_pack_validates_config_section():
    text = _pack_yaml(config={"tts-plugin": {"granted_capabilities": ["tools"]}})
    with pytest.raises(PackError):
        parse_pack(text)
    ok = parse_pack(_pack_yaml(config={"tts-plugin": {"voice": "nova"}}))
    assert ok.config["tts-plugin"] == {"voice": "nova"}


def test_parse_pack_collects_skills_as_declared_seam():
    pack = parse_pack(_pack_yaml(skills=["hub/skill-a", "hub/skill-b"]))
    assert pack.skills == ["hub/skill-a", "hub/skill-b"]


def test_load_pack_rejects_insecure_url_schemes(tmp_path):
    with pytest.raises(PackError) as exc:
        load_pack("http://example.com/pack.yaml")
    assert "https" in str(exc.value)


def test_load_pack_reads_local_file(tmp_path):
    f = tmp_path / "hermes-pack.yaml"
    f.write_text(_pack_yaml(), encoding="utf-8")
    assert load_pack(str(f)).name == "voice-pack"


def test_load_pack_missing_file_errors():
    with pytest.raises(PackError) as exc:
        load_pack("/nonexistent/pack.yaml")
    assert "not found" in str(exc.value)


# ---------------------------------------------------------------------------
# Resolution (bare index names) — index mocked, no network
# ---------------------------------------------------------------------------

def test_resolve_pack_plugins_uses_community_index_for_bare_names():
    pack = parse_pack(_pack_yaml())
    fake_entry = SimpleNamespace(
        install_identifier="idx-owner/idx-repo", capabilities=["tools"]
    )
    with mock.patch(
        "hermes_cli.plugin_index.load_index", return_value=([fake_entry], "seed")
    ), mock.patch(
        "hermes_cli.plugin_index.resolve_name",
        return_value=(fake_entry, [fake_entry]),
    ):
        resolved = resolve_pack_plugins(pack)

    assert resolved[0].identifier == "owner/tts-plugin"  # repo entries skip the index
    assert resolved[1].identifier == "idx-owner/idx-repo"
    assert resolved[1].index_capabilities == ["tools"]


def test_resolve_pack_plugins_carries_index_miss_as_error():
    pack = parse_pack(
        yaml.safe_dump(
            {"name": "p", "plugins": [{"name": "ghost", "ref": SHA_A}]}
        )
    )
    with mock.patch(
        "hermes_cli.plugin_index.load_index", return_value=([], "seed")
    ), mock.patch(
        "hermes_cli.plugin_index.resolve_name", return_value=(None, [])
    ):
        resolved = resolve_pack_plugins(pack)
    assert resolved[0].identifier is None
    assert "not found" in resolved[0].resolve_error


# ---------------------------------------------------------------------------
# Install fan-out — installer mocked
# ---------------------------------------------------------------------------

def _resolved(pack):
    return [
        ResolvedPackPlugin(entry=e, identifier=e.install_identifier)
        for e in pack.plugins
    ]


def _fanout_patches(install_side_effect, consent_mock=None):
    """Patch the plugins_cmd seams install_pack_plugins pulls in."""
    patches = {
        "_install_plugin_core": mock.MagicMock(side_effect=install_side_effect),
        "_prompt_plugin_env_vars": mock.MagicMock(),
        "_get_enabled_set": mock.MagicMock(return_value=set()),
        "_get_disabled_set": mock.MagicMock(return_value=set()),
        "_save_enabled_set": mock.MagicMock(),
        "_save_disabled_set": mock.MagicMock(),
        "_run_capability_consent": consent_mock or mock.MagicMock(return_value=True),
        "_declared_capabilities_from_manifest": mock.MagicMock(
            side_effect=lambda manifest, name: manifest.get("capabilities", [])
        ),
    }
    return patches


def test_install_fan_out_passes_pinned_refs_to_installer(tmp_path):
    pack = parse_pack(
        yaml.safe_dump(
            {
                "name": "p",
                "plugins": [
                    {"repo": "o/a", "ref": SHA_A},
                    {"repo": "o/b", "ref": SHA_B},
                ],
            }
        )
    )
    installer = mock.MagicMock(
        side_effect=[
            (tmp_path / "a", {"name": "a"}, "a"),
            (tmp_path / "b", {"name": "b"}, "b"),
        ]
    )
    patches = _fanout_patches(None)
    patches["_install_plugin_core"] = installer
    with mock.patch.multiple("hermes_cli.plugins_cmd", **patches):
        results = install_pack_plugins(pack, _resolved(pack), FakeConsole())

    assert [r.ok for r in results] == [True, True]
    assert installer.call_args_list == [
        mock.call("o/a", force=False, ref=SHA_A),
        mock.call("o/b", force=False, ref=SHA_B),
    ]


def test_install_fan_out_invokes_capability_consent_per_plugin(tmp_path):
    """Consent is NOT bypassed: the standard per-plugin consent function
    runs once for every installed plugin that declares capabilities."""
    pack = parse_pack(
        yaml.safe_dump(
            {
                "name": "p",
                "plugins": [
                    {"repo": "o/a", "ref": SHA_A},
                    {"repo": "o/b", "ref": SHA_B},
                ],
            }
        )
    )
    consent = mock.MagicMock(return_value=True)
    patches = _fanout_patches(
        [
            (tmp_path / "a", {"name": "a", "capabilities": ["tools"]}, "a"),
            (tmp_path / "b", {"name": "b", "capabilities": ["platform"]}, "b"),
        ],
        consent_mock=consent,
    )
    with mock.patch.multiple("hermes_cli.plugins_cmd", **patches):
        install_pack_plugins(pack, _resolved(pack), FakeConsole())

    assert consent.call_count == 2
    called_ids = [c.args[1] for c in consent.call_args_list]
    assert called_ids == ["a", "b"]
    called_caps = [c.args[2] for c in consent.call_args_list]
    assert called_caps == [["tools"], ["platform"]]


def test_install_fan_out_continues_past_failures_and_reports():
    from hermes_cli.plugins_cmd import PluginOperationError

    pack = parse_pack(
        yaml.safe_dump(
            {
                "name": "p",
                "plugins": [
                    {"repo": "o/bad", "ref": SHA_A},
                    {"repo": "o/good", "ref": SHA_B},
                ],
            }
        )
    )

    def installer(identifier, *, force, ref):
        if "bad" in identifier:
            raise PluginOperationError("clone exploded")
        return (mock.MagicMock(), {"name": "good"}, "good")

    patches = _fanout_patches(installer)
    console = FakeConsole()
    with mock.patch.multiple("hermes_cli.plugins_cmd", **patches):
        results = install_pack_plugins(pack, _resolved(pack), console)

    assert [r.ok for r in results] == [False, True]
    assert "clone exploded" in results[0].error
    assert results[1].installed_name == "good"


def test_pack_install_exits_nonzero_on_partial_failure(tmp_path, monkeypatch):
    pack_file = tmp_path / "pack.yaml"
    pack_file.write_text(
        yaml.safe_dump(
            {
                "name": "p",
                "plugins": [
                    {"repo": "o/bad", "ref": SHA_A},
                    {"repo": "o/good", "ref": SHA_B},
                ],
            }
        ),
        encoding="utf-8",
    )
    from hermes_cli.plugins_cmd import PluginOperationError

    def installer(identifier, *, force, ref):
        if "bad" in identifier:
            raise PluginOperationError("boom")
        return (mock.MagicMock(), {"name": "good"}, "good")

    fake_console = FakeConsole(answers=["y"])
    patches = _fanout_patches(installer)
    monkeypatch.setattr("sys.stdin", mock.MagicMock(isatty=lambda: True))
    monkeypatch.setattr("sys.stdout", mock.MagicMock(isatty=lambda: True))
    with mock.patch.multiple("hermes_cli.plugins_cmd", **patches), mock.patch(
        "rich.console.Console", return_value=fake_console
    ):
        with pytest.raises(SystemExit) as exc:
            cmd_pack_install(str(pack_file))
    assert exc.value.code == 1
    assert "1 installed, 1 failed" in fake_console.text


def test_pack_install_refuses_noninteractive_sessions(tmp_path, monkeypatch):
    pack_file = tmp_path / "pack.yaml"
    pack_file.write_text(_pack_yaml(), encoding="utf-8")
    fake_console = FakeConsole()
    monkeypatch.setattr("sys.stdin", mock.MagicMock(isatty=lambda: False))
    monkeypatch.setattr("sys.stdout", mock.MagicMock(isatty=lambda: False))
    installer = mock.MagicMock()
    with mock.patch(
        "rich.console.Console", return_value=fake_console
    ), mock.patch(
        "hermes_cli.plugin_packs.resolve_pack_plugins",
        side_effect=lambda pack: _resolved(pack),
    ), mock.patch("hermes_cli.plugins_cmd._install_plugin_core", installer):
        with pytest.raises(SystemExit) as exc:
            cmd_pack_install(str(pack_file))
    assert exc.value.code == 1
    installer.assert_not_called()
    assert "interactive" in fake_console.text


def test_pack_install_aborts_cleanly_on_decline(tmp_path, monkeypatch):
    pack_file = tmp_path / "pack.yaml"
    pack_file.write_text(_pack_yaml(), encoding="utf-8")
    fake_console = FakeConsole(answers=["n"])
    monkeypatch.setattr("sys.stdin", mock.MagicMock(isatty=lambda: True))
    monkeypatch.setattr("sys.stdout", mock.MagicMock(isatty=lambda: True))
    installer = mock.MagicMock()
    with mock.patch(
        "rich.console.Console", return_value=fake_console
    ), mock.patch(
        "hermes_cli.plugin_packs.resolve_pack_plugins",
        side_effect=lambda pack: _resolved(pack),
    ), mock.patch("hermes_cli.plugins_cmd._install_plugin_core", installer):
        with pytest.raises(SystemExit):
            cmd_pack_install(str(pack_file))
    installer.assert_not_called()
    assert "Aborted" in fake_console.text


# ---------------------------------------------------------------------------
# Export round-trip
# ---------------------------------------------------------------------------

def _seed_install_state(home, monkeypatch, *, metadata, config=None, plugins=()):
    plugins_dir = home / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    for name in plugins:
        (plugins_dir / name).mkdir(exist_ok=True)
    (plugins_dir / ".install-metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    monkeypatch.setattr(
        "hermes_cli.plugins_cmd._read_install_metadata", lambda: metadata
    )
    monkeypatch.setattr("hermes_cli.plugins_cmd._plugins_dir", lambda: plugins_dir)
    cfg = config or {}
    monkeypatch.setattr(
        "hermes_cli.plugins_cmd._get_enabled_set",
        lambda: set((cfg.get("plugins") or {}).get("enabled") or []),
    )
    monkeypatch.setattr("hermes_cli.plugin_packs._sanitized_entry_config",
                        lambda pid: ((cfg.get("plugins") or {}).get("entries") or {}).get(pid, {}))


def test_export_round_trips_through_parse(tmp_path, monkeypatch):
    metadata = {
        "tts": {
            "pinned": True,
            "revision": SHA_A,
            "source": "https://github.com/owner/tts-plugin.git",
        },
        "relay": {
            "pinned": True,
            "revision": SHA_B,
            "source": "https://github.com/owner/mono.git#plugins/relay",
        },
    }
    config = {
        "plugins": {
            "enabled": ["tts", "relay"],
            "entries": {"tts": {"voice": "nova"}},
        }
    }
    _seed_install_state(
        tmp_path, monkeypatch, metadata=metadata, config=config,
        plugins=("tts", "relay"),
    )

    text, warnings = export_pack(pack_name="exported")
    assert warnings == []
    pack = parse_pack(text)
    assert pack.name == "exported"
    by_repo = {p.repo: p for p in pack.plugins}
    assert by_repo["owner/tts-plugin"].ref == SHA_A
    assert by_repo["owner/mono"].subdir == "plugins/relay"
    assert by_repo["owner/mono"].ref == SHA_B
    assert pack.config["tts"] == {"voice": "nova"}


def test_export_warns_on_local_only_plugins(tmp_path, monkeypatch):
    _seed_install_state(
        tmp_path, monkeypatch, metadata={}, plugins=("local-hack",)
    )
    text, warnings = export_pack()
    assert any("local-hack" in w for w in warnings)
    assert "# WARNING" in text and "local-hack" in text
    # A pack with zero installable plugins won't parse back as installable.
    with pytest.raises(PackError):
        parse_pack(text)


def test_export_strips_secret_and_capability_config_keys(tmp_path, monkeypatch):
    metadata = {
        "tts": {
            "pinned": True,
            "revision": SHA_A,
            "source": "https://github.com/owner/tts.git",
        }
    }
    _seed_install_state(
        tmp_path, monkeypatch, metadata=metadata, plugins=("tts",)
    )
    # Use the real sanitizer against a fake loaded config.
    monkeypatch.setattr(
        "hermes_cli.plugin_packs._sanitized_entry_config",
        real_sanitized_entry_config,
    )
    fake_cfg = {
        "plugins": {
            "entries": {
                "tts": {
                    "voice": "nova",
                    "api_key": "sk-super-secret",
                    "granted_capabilities": ["tools"],
                    "allow_tool_override": True,
                }
            }
        }
    }
    with mock.patch("hermes_cli.config.load_config", return_value=fake_cfg):
        text, _warnings = export_pack()
    assert "sk-super-secret" not in text
    assert "api_key" not in text
    assert "granted_capabilities" not in text
    assert "allow_tool_override" not in text
    assert "voice: nova" in text


def test_export_enabled_only_filters(tmp_path, monkeypatch):
    metadata = {
        "on": {"pinned": True, "revision": SHA_A,
               "source": "https://github.com/o/on.git"},
        "off": {"pinned": True, "revision": SHA_B,
                "source": "https://github.com/o/off.git"},
    }
    config = {"plugins": {"enabled": ["on"]}}
    _seed_install_state(
        tmp_path, monkeypatch, metadata=metadata, config=config,
        plugins=("on", "off"),
    )
    text, _ = export_pack(enabled_only=True)
    pack = parse_pack(text)
    assert [p.repo for p in pack.plugins] == ["o/on"]


# ---------------------------------------------------------------------------
# Parser wiring
# ---------------------------------------------------------------------------

def test_parser_wires_pack_subcommands():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_plugins_parser(subparsers, cmd_plugins=lambda _args: None)

    args = parser.parse_args(["plugins", "pack", "install", "pack.yaml", "--force"])
    assert args.plugins_action == "pack"
    assert args.pack_action == "install"
    assert args.source == "pack.yaml"
    assert args.force is True

    args = parser.parse_args(["plugins", "pack", "export", "--enabled-only"])
    assert args.pack_action == "export"
    assert args.enabled_only is True

    args = parser.parse_args(["plugins", "pack", "show", "https://x/p.yaml"])
    assert args.pack_action == "show"
