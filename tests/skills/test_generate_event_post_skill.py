"""Tests for the generate-event-post optional skill."""

import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO / "optional-skills" / "social-media" / "generate-event-post"
SKILL_MD = SKILL_DIR / "SKILL.md"
PROFILE = SKILL_DIR / "references" / "marimira-profile.md"


def _frontmatter_and_body():
    content = SKILL_MD.read_text(encoding="utf-8")
    assert content.startswith("---")
    match = re.search(r"\n---\s*\n", content[3:])
    assert match, "frontmatter must close with ---"
    frontmatter = yaml.safe_load(content[3 : match.start() + 3])
    body = content[match.end() + 3 :]
    return frontmatter, body


def test_skill_files_exist():
    assert SKILL_MD.is_file()
    assert PROFILE.is_file()


def test_frontmatter_meets_authoring_standard():
    frontmatter, _ = _frontmatter_and_body()
    for field in ("name", "description", "version", "author", "license", "platforms"):
        assert field in frontmatter, f"missing frontmatter field: {field}"
    assert frontmatter["name"] == "generate-event-post"
    assert len(frontmatter["description"]) <= 60
    assert frontmatter["description"].endswith(".")
    assert frontmatter["platforms"] == ["linux", "macos", "windows"]
    assert frontmatter["author"].startswith("Hiroyuki Taniichi (HiroSnow0413)")
    assert frontmatter["metadata"]["hermes"]["category"] == "social-media"


def test_related_skills_resolve_in_repo():
    frontmatter, _ = _frontmatter_and_body()
    for name in frontmatter["metadata"]["hermes"]["related_skills"]:
        hits = list((REPO / "skills").glob(f"**/{name}/SKILL.md")) + list(
            (REPO / "optional-skills").glob(f"**/{name}/SKILL.md")
        )
        assert hits, f"related skill does not resolve in-repo: {name}"


def test_body_has_modern_sections_and_completion_criteria():
    _, body = _frontmatter_and_body()
    for section in (
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ):
        assert section in body, f"missing section: {section}"
    steps = re.findall(r"^### \d+\..*?(?=^### \d+\.|^## )", body, re.MULTILINE | re.DOTALL)
    assert len(steps) == 9
    assert all("完了条件:" in step for step in steps)


def test_field_specific_precedence_and_conflict_policy_are_explicit():
    _, body = _frontmatter_and_body()
    assert "チケットサイト → 主催本文 → 画像" in body
    assert "チケットサイト → 画像 → 本文" in body
    assert "表にない項目には優先順位を作らない" in body
    assert "最新版の公開タイムテーブル" in body
    assert "⚠️要確認" in body


def test_safety_and_output_contract_are_explicit():
    _, body = _frontmatter_and_body()
    assert "公開厳禁" in body
    assert "出力は必ず1つのコードブロック" in body
    assert "警告にはコードフェンスを付けない" in body
    assert "告知本文だけが唯一のコードブロック" in body
    assert "URLをMarkdownリンクへ変えない" in body
    assert "ユーザー確認なしのX投稿" in body
    assert "実際のX投稿は行っていない" in body


def test_marimira_profile_contains_brand_contract():
    profile = PROFILE.read_text(encoding="utf-8")
    for target in (
        "Marionette in the Mirror",
        "心春るり",
        "日向れいら",
        "甘海老えびな",
        "星屑あめ",
        "姫鈴らぶか",
    ):
        assert target in profile
    assert "可愛あん" not in profile
    assert "2,000円以上" in profile
    assert "マリミラルーレット" in profile
    assert "2,000円未満" in profile
    assert "写メ" in profile
    assert "#マリミライベント情報" in profile
    assert "https://x.com/mari_mira_idol" in profile


def test_no_machine_local_paths_and_size_limit():
    for path in (SKILL_MD, PROFILE):
        content = path.read_text(encoding="utf-8")
        assert "/Users/" not in content
        assert "/home/" not in content
        assert len(content) <= 100_000
