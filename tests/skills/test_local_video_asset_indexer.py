from pathlib import Path


def test_asset_indexer_skill_has_required_contract():
    root = Path("skills/local-video-asset-indexer")
    text = (root / "SKILL.md").read_text(encoding="utf-8")

    assert "name: local-video-asset-indexer" in text
    assert "python -m capabilities.video_library.cli scan" in text
    assert "不得移动、改名或删除原始视频" in text
    assert "director-lapian" in text
    assert (root / "references" / "shot-schema.md").is_file()
    assert (root / "references" / "beef-noodle-taxonomy.md").is_file()
    assert (root / "assets" / "SOUL.md").is_file()


def test_profile_template_routes_two_video_workflows():
    text = Path("skills/local-video-asset-indexer/assets/SOUL.md").read_text(encoding="utf-8")

    assert "拉片专家" in text
    assert "local-video-asset-indexer" in text
    assert "director-lapian" in text
    assert "配置过的素材目录" in text
