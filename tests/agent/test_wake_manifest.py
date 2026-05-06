from pathlib import Path

import agent.wake_manifest as wake_manifest_module
from agent.wake_manifest import build_wake_manifest
from hermes_cli.config import save_config


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _base_route_catalog():
    return {
        "research": {
            "command": "research",
            "display_command": "/research",
            "append_toolsets": ("web",),
        },
        "multi_agent": {
            "command": "background",
            "display_command": "/bg",
            "append_toolsets": ("delegation",),
        },
        "automation": {
            "command": "background",
            "display_command": "/bg",
            "append_toolsets": ("cronjob",),
        },
        "doc_feishu": {
            "command": "doc",
            "display_command": "/doc",
        },
        "ppt": {
            "command": "ppt",
            "display_command": "/ppt",
            "conditional_skills": {
                "powerpoint": (".pptx", "editable", "可编辑", "powerpoint", "pptx"),
            },
        },
    }


def test_wake_manifest_module_comes_from_current_worktree():
    assert Path(wake_manifest_module.__file__).resolve().is_relative_to(_PROJECT_ROOT)


def _write_skill(skill_dir, name: str, frontmatter_tail: str = ""):
    target = skill_dir / name
    target.mkdir(parents=True)
    (target / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: demo\n"
        f"{frontmatter_tail}"
        "---\n"
        "# Demo\n",
        encoding="utf-8",
    )


def test_build_wake_manifest_binds_skills_to_routes(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    skills_dir = tmp_path / "skills" / "productivity"

    _write_skill(
        skills_dir,
        "feishu-cloud-doc-delivery",
        frontmatter_tail=(
            "metadata:\n"
            "  hermes:\n"
            "    wake:\n"
            "      route: doc_feishu\n"
            "      aliases: [feishu-doc]\n"
            "      keywords: [飞书文档]\n"
            "      risk: internal_write\n"
            "      delivery: feishu_cloud_doc\n"
        ),
    )
    _write_skill(
        skills_dir,
        "plain-skill",
    )

    manifest = build_wake_manifest(_base_route_catalog(), platform="feishu")

    assert manifest["errors"] == ()
    assert manifest["routes"]["doc_feishu"]["skills"] == ("feishu-cloud-doc-delivery",)
    assert manifest["routes"]["doc_feishu"]["aliases"] == ("feishu-doc",)
    assert manifest["routes"]["doc_feishu"]["keywords"] == ("飞书文档",)

    skills = {row["name"]: row for row in manifest["skills"]}
    assert skills["feishu-cloud-doc-delivery"]["wake"]["route"] == "doc_feishu"
    assert skills["plain-skill"]["wake"] == {}


def test_build_wake_manifest_reports_unknown_route_without_binding(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    skills_dir = tmp_path / "skills" / "productivity"

    _write_skill(
        skills_dir,
        "bad-route-skill",
        frontmatter_tail=(
            "metadata:\n"
            "  hermes:\n"
            "    wake:\n"
            "      route: nope\n"
            "      aliases: [bad-route]\n"
        ),
    )

    manifest = build_wake_manifest(_base_route_catalog(), platform="feishu")

    assert manifest["routes"]["doc_feishu"]["skills"] == ()
    assert any("bad-route-skill" in item and "unknown wake route 'nope'" in item for item in manifest["errors"])

    skills = {row["name"]: row for row in manifest["skills"]}
    assert skills["bad-route-skill"]["wake"] == {"aliases": ("bad-route",)}


def test_build_wake_manifest_rejects_invalid_risk_and_delivery(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    skills_dir = tmp_path / "skills" / "productivity"

    _write_skill(
        skills_dir,
        "bad-risk-delivery",
        frontmatter_tail=(
            "metadata:\n"
            "  hermes:\n"
            "    wake:\n"
            "      route: doc_feishu\n"
            "      aliases: [bad-risk]\n"
            "      risk: dangerous\n"
            "      delivery: BadDelivery\n"
        ),
    )

    manifest = build_wake_manifest(_base_route_catalog(), platform="feishu")

    assert any(
        "bad-risk-delivery" in item and "invalid wake metadata" in item
        for item in manifest["errors"]
    )
    skills = {row["name"]: row for row in manifest["skills"]}
    assert skills["bad-risk-delivery"]["wake"] == {}


def test_build_wake_manifest_keeps_route_behavior_fields(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    manifest = build_wake_manifest(_base_route_catalog(), platform="feishu")

    assert manifest["routes"]["research"]["append_toolsets"] == ("web",)
    assert manifest["routes"]["multi_agent"]["append_toolsets"] == (
        "delegation",
    )
    assert manifest["routes"]["automation"]["append_toolsets"] == (
        "cronjob",
    )
    assert manifest["routes"]["multi_agent"]["drop_toolsets"] == ()
    assert manifest["routes"]["ppt"]["conditional_skills"]["powerpoint"] == (
        ".pptx",
        "editable",
        "可编辑",
        "powerpoint",
        "pptx",
    )


def test_build_wake_manifest_applies_config_wake_override(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    skills_dir = tmp_path / "skills" / "presentations"

    _write_skill(
        skills_dir,
        "frontend-slides",
        frontmatter_tail=(
            "metadata:\n"
            "  hermes:\n"
            "    wake:\n"
            "      route: doc_feishu\n"
            "      aliases: [old-alias]\n"
            "      keywords: [old keyword]\n"
            "      risk: internal_write\n"
            "      delivery: feishu_cloud_doc\n"
        ),
    )
    save_config(
        {
            "skills": {
                "wake_overrides": {
                    "frontend-slides": {
                        "route": "ppt",
                        "aliases": ["slides wake"],
                        "keywords": ["presentation mode"],
                    }
                }
            }
        }
    )

    manifest = build_wake_manifest(_base_route_catalog(), platform="feishu")

    assert manifest["errors"] == ()
    skills = {row["name"]: row for row in manifest["skills"]}
    assert skills["frontend-slides"]["wake"] == {
        "route": "ppt",
        "aliases": ("slides wake",),
        "keywords": ("presentation mode",),
        "risk": "internal_write",
        "delivery": "feishu_cloud_doc",
    }
    assert manifest["routes"]["ppt"]["skills"] == ("frontend-slides",)
    assert manifest["routes"]["ppt"]["aliases"] == ("slides wake",)
    assert manifest["routes"]["ppt"]["keywords"] == ("presentation mode",)
    assert manifest["routes"]["doc_feishu"]["skills"] == ()


def test_build_wake_manifest_reports_invalid_config_wake_override(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    skills_dir = tmp_path / "skills" / "productivity"
    _write_skill(skills_dir, "feishu-cloud-doc-delivery")
    save_config(
        {
            "skills": {
                "wake_overrides": {
                    "feishu-cloud-doc-delivery": {
                        "route": "nope",
                        "risk": "dangerous",
                    }
                }
            }
        }
    )

    manifest = build_wake_manifest(_base_route_catalog(), platform="feishu")

    assert any(
        "wake_override:feishu-cloud-doc-delivery: invalid wake metadata" in item
        or "wake_override:feishu-cloud-doc-delivery: unknown wake route 'nope'" in item
        for item in manifest["errors"]
    )
    skills = {row["name"]: row for row in manifest["skills"]}
    assert skills["feishu-cloud-doc-delivery"]["wake"] == {}
