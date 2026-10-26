"""Tests for banner toolset name normalization and skin color usage."""

from unittest.mock import patch

from rich.console import Console

from hermes_cli import banner
import model_tools
import tools.mcp_tool_discovery

def test_banner_snapshot_accepts_bom_without_weakening_freshness(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(banner, "get_git_banner_state", lambda: None)
    monkeypatch.setattr(banner, "get_available_skills", lambda: {"notes": ["café"]})
    tools = [{"function": {"name": "read_file"}}]
    banner.save_banner_snapshot(tools, ["file"], {}, {"read_file": "file"})
    path = banner._banner_snapshot_path()
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    expected = json.loads(raw)
    path.write_text(json.dumps(expected, ensure_ascii=False), encoding="utf-8-sig")
    assert banner.load_banner_snapshot(["file"]) == expected
    assert banner.load_banner_snapshot(["web"]) is None
    (tmp_path / "config.yaml").write_text("display: {skin: mono}", encoding="utf-8")
    assert banner.load_banner_snapshot(["file"]) is None

def test_cprint_falls_back_to_plain_print_when_prompt_toolkit_has_no_console(capsys):
    with patch(
        "prompt_toolkit.print_formatted_text",
        side_effect=RuntimeError("no console screen buffer"),
    ):
        banner.cprint("fallback text")

    assert capsys.readouterr().out == "fallback text\n"

def test_empty_model_shows_no_model_configured(tmp_path, monkeypatch):
    """The banner prints before credentials resolve; with no model it says so rather than guessing."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir()
    with (
        patch.object(model_tools, "check_tool_availability", return_value=([], [])),
        patch.object(banner, "get_available_skills", return_value={}),
        patch.object(banner, "get_update_result", return_value=None),
        patch.object(tools.mcp_tool_discovery, "get_mcp_status", return_value=[]),
    ):
        console = Console(record=True, force_terminal=False, color_system=None, width=160)
        banner.build_welcome_banner(console=console, model="", cwd="/tmp/project", tools=[],
                                    enabled_toolsets=[], provider="auto")
    assert "no model configured" in console.export_text()


def test_build_welcome_banner_does_not_center_pad_hero_art():
    """A braille hero relies on its own U+2800 padding for symmetry; Rich centering inserts
    ASCII spaces around the left column and distorts the silhouette (#9879). The hero line
    must start flush at the column start."""
    import io
    from types import SimpleNamespace

    skin = SimpleNamespace(banner_hero="[green]\u2800X[/]", banner_logo="")
    buf = io.StringIO()
    with (
        patch.object(model_tools, "check_tool_availability", return_value=([], [])),
        patch.object(banner, "get_available_skills", return_value={}),
        patch.object(banner, "get_update_result", return_value=None),
        patch.object(banner, "get_latest_release_tag", return_value=None),
        patch.object(tools.mcp_tool_discovery, "get_mcp_status", return_value=[]),
        patch.object(banner, "_active_skin", return_value=skin),
    ):
        console = Console(file=buf, force_terminal=False, color_system=None, width=80)
        banner.build_welcome_banner(console=console, model="m", cwd="/tmp", tools=[],
                                    get_toolset_for_tool=lambda _: None)

    hero_line = next(line for line in buf.getvalue().splitlines() if "\u2800X" in line)
    assert hero_line.startswith("\u2502  \u2800X"), repr(hero_line)


def test_baked_banner_uses_live_identity(monkeypatch):
    from hermes_cli import banner, version_info

    monkeypatch.setattr(version_info, "get_code_identity", lambda: {"short_sha": "a1b2c3d4"})
    assert banner._baked_banner_state() == {"upstream": "a1b2c3d4", "local": "a1b2c3d4", "ahead": 0}


def test_banner_version_label_marks_a_calver_fallback_base(monkeypatch):
    """A base resolved by the CalVer describe (no SemVer release tag reachable at
    all) is a guess that flips once release tags are fetched — one checkout
    truthfully printed 0.21.0+32768 before and 0.22.5+1710 after a single
    `hermes update` (#140329). The label must say so instead of presenting the
    fallback in the same shape as a real release-based identity."""
    from hermes_cli.version_info import VersionInfo

    def _live(source_flag):
        return VersionInfo(
            base_version="0.21.0", derived_version="0.21.0+32768.g0591b66", distance=32768,
            commit="0" * 40, branch=None, source="git", dirty=False, commit_date=None,
            distribution=None, base_version_source=source_flag,
        )

    monkeypatch.setattr("hermes_cli.steward.read_install_stamp", lambda root: {})
    monkeypatch.setattr("hermes_cli.update_channel.resolve_update_channel", lambda config, root: "main")
    monkeypatch.setattr(banner, "get_git_banner_state", lambda: None)

    monkeypatch.setattr(banner, "get_version_info", lambda: _live("calver-fallback"))
    label = banner.format_banner_version_label()
    assert "0.21.0+32768.g0591b66" in label
    assert "(calver-fallback; release tags not fetched)" in label

    monkeypatch.setattr(banner, "get_version_info", lambda: _live("release-tag"))
    assert "calver-fallback" not in banner.format_banner_version_label()
