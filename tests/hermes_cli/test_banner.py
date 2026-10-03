"""Tests for banner toolset name normalization and skin color usage."""

from unittest.mock import patch

from rich.console import Console

import hermes_cli.banner as banner
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

def test_empty_model_shows_the_free_tier_route_when_it_carries_inference(tmp_path, monkeypatch):
    """The banner prints before credentials resolve, so ``model`` is empty on a fresh install. On the
    free tier the route is known locally (identity on disk + tier on): the banner shows its model.
    When nothing resolves the red "no model configured" line stays."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir()
    import hermes_cli.anon_auth as anon_auth

    def render(carries: bool) -> str:
        with (
            patch.object(model_tools, "check_tool_availability", return_value=([], [])),
            patch.object(banner, "get_available_skills", return_value={}),
            patch.object(banner, "get_update_result", return_value=None),
            patch.object(tools.mcp_tool_discovery, "get_mcp_status", return_value=[]),
            patch.object(anon_auth, "free_tier_route", return_value=carries),
        ):
            console = Console(record=True, force_terminal=False, color_system=None, width=160)
            banner.build_welcome_banner(console=console, model="", cwd="/tmp/project", tools=[],
                                        enabled_toolsets=[], provider="auto")
        return console.export_text()

    assert "welcome" in render(True) and "no model configured" not in render(True)
    assert "no model configured" in render(False)


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


def test_docker_banner_prefers_installed_image_identity_over_mounted_git(tmp_path, monkeypatch):
    """A mutable source checkout must not impersonate immutable Docker code."""
    import subprocess
    from hermes_cli import version_info

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-q")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
        "commit", "-q", "--allow-empty", "-m", "source")
    git("update-ref", "refs/remotes/origin/main", "HEAD")
    mounted_sha = git("rev-parse", "HEAD")[:8]
    assert mounted_sha != "12345678"

    image = version_info.VersionInfo("1.0", "1.0", 0, "12345678abcdef01", None,
                                     "docker", distribution="docker")
    monkeypatch.setattr(version_info, "get_version_info", lambda: image)
    monkeypatch.setattr(banner, "_resolve_repo_dir", lambda: tmp_path)
    assert banner._compute_git_banner_state() == {
        "upstream": "12345678", "local": "12345678", "ahead": 0,
    }
    # An explicit checkout request still reports that checkout.
    explicit = banner._compute_git_banner_state(tmp_path)
    assert explicit is not None
    assert explicit["local"] == mounted_sha
