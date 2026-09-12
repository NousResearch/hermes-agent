from pathlib import Path


def test_command_catalog_falls_back_to_english_for_unknown_language():
    from hermes_cli.commands import localized_command_catalog

    catalog = localized_command_catalog("xx")

    assert dict(catalog["pairs"])["/new"].startswith("Start a new session")
    assert catalog["categories"][0]["name"] == "Session"


def test_command_catalog_localizes_korean_descriptions_and_categories():
    from hermes_cli.commands import localized_command_catalog

    catalog = localized_command_catalog("ko")
    pairs = dict(catalog["pairs"])

    assert "/new" in pairs
    assert pairs["/new"] != catalog["english_pairs"]["/new"]
    assert any(section["name"] == "세션" for section in catalog["categories"])


def test_gm_skill_frontmatter_descriptions_are_korean_without_body_changes():
    from agent.skill_utils import parse_frontmatter

    root = Path("/root/.hermes/profiles/gm/skills")
    files = sorted(root.glob("**/SKILL.md"))
    assert files
    targets = {
        root / "yuanbao/SKILL.md",
        root / "hermes-themes/SKILL.md",
        root / "hermes-desktop-plugins/SKILL.md",
        root / "codex-usage-widget-design/SKILL.md",
    }
    for path in targets:
        frontmatter, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        assert frontmatter["description"]
        assert any("가" <= char <= "힣" for char in frontmatter["description"]), path
        assert body.lstrip().startswith("#")
    assert files


def test_gateway_command_catalog_uses_requested_language(monkeypatch):
    from tui_gateway import server

    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    response = server.handle_request(
        {"id": "ko", "method": "commands.catalog", "params": {"language": "ko"}}
    )

    pairs = dict(response["result"]["pairs"])
    assert pairs["/new"] != "Start a new session (fresh session ID + history)"
    assert any(section["name"] == "세션" for section in response["result"]["categories"])
