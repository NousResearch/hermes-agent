from __future__ import annotations

import json
from pathlib import Path

from gateway.image2_prompt import build_visual_brief_from_payload, compile_image2_prompt_payload
from gateway.image2_store import Image2JobStore


def test_source_image_edit_prompt_uses_reference_subject_not_firepalace_default(tmp_path):
    source_image = tmp_path / "source.jpg"
    source_image.write_bytes(b"source image bytes")
    payload = {
        "source_platform": "feishu",
        "feishu_message_id": "om_redesign",
        "chat_id": "oc_chat",
        "root_id": "om_root",
        "thread_id": "om_root",
        "text": "你再重新设计下这个海报，我觉得不好看",
        "source_files": [
            {"path": str(source_image), "mime_type": "image/jpeg", "source": "feishu_thread_root_image"}
        ],
    }

    compiled = compile_image2_prompt_payload(payload)
    summary = compiled["visual_brief_summary"]
    prompt = compiled["one_shot_design_prompt"]

    assert summary["source_edit_mode"] is True
    assert summary["asset_type"] == "参考图精修海报"
    assert summary["main_visual_object"] == "引用图片中的原主体"
    assert "基于用户引用图片" in prompt
    assert "必须忠实保留引用图片中的原主体" in prompt
    assert "主视觉对象：辣椒小炒肉" not in prompt
    assert "T3 到店即点" not in prompt
    assert "用于华卓/火宫殿 T3 营销物料" not in prompt


def test_continuation_subject_correction_emits_hard_subject_anchor():
    payload = {
        "source_platform": "feishu",
        "feishu_message_id": "om_followup",
        "chat_id": "oc_chat",
        "root_id": "om_choudoufu",
        "thread_id": "om_choudoufu",
        "text": "我是说臭豆腐，你生成的是什么菜",
        "source_files": [],
    }

    brief = build_visual_brief_from_payload(payload)
    compiled = compile_image2_prompt_payload(payload)
    prompt = compiled["one_shot_design_prompt"]

    assert brief["main_visual_object"] == "臭豆腐"
    assert brief["chatgpt_continuation_mode"] is True
    assert "主视觉对象：臭豆腐" in prompt
    assert "本次必须仍是「臭豆腐」海报/设计" in prompt
    assert "不得改成任何其他菜品、饮品或泛海报主题" in prompt
    assert "主视觉对象：饮品" not in prompt


def test_job_store_writes_prompt_artifacts_inside_hermes_runtime(tmp_path):
    runtime = tmp_path / "runtime" / "image2"
    source_image = tmp_path / "source.jpg"
    source_image.write_bytes(b"source image bytes")
    payload = {
        "source_platform": "feishu",
        "feishu_message_id": "om_source_edit",
        "chat_id": "oc_chat",
        "root_id": "om_root",
        "thread_id": "om_root",
        "text": "补总标题：夏日鲜果冰柠系列 中英文\n补卖点：鲜果入饮 冰爽解腻",
        "source_files": [
            {"path": str(source_image), "mime_type": "image/jpeg", "source": "feishu_quoted_parent"}
        ],
    }

    result = Image2JobStore(db_path=runtime / "image2_jobs.sqlite", runtime_root=runtime).enqueue_feishu(payload)
    job_dir = Path(result["job_dir"])
    brief = json.loads((job_dir / "brief.json").read_text(encoding="utf-8"))
    compiled = json.loads((job_dir / "compiled_prompt.json").read_text(encoding="utf-8"))
    prompt = (job_dir / "one_shot_design_prompt.txt").read_text(encoding="utf-8")

    assert brief["source_edit_mode"] is True
    assert brief["source_files"][0]["source"] == "feishu_quoted_parent"
    assert compiled["one_shot_design_prompt"] == prompt
    assert (job_dir / "prompt.txt").read_text(encoding="utf-8") == prompt
    assert "夏日鲜果冰柠系列" in prompt
    assert "marketing-hub/scripts" not in prompt
    assert "image2_job_pipeline.py" not in prompt


def test_title_request_for_choudoufu_uses_short_human_copy():
    payload = {
        "source_platform": "feishu",
        "feishu_message_id": "om_title",
        "chat_id": "oc_chat",
        "root_id": "",
        "thread_id": "",
        "text": "/image2 帮我把这个臭豆腐的海报设计的好看一点，更有食欲一点。然后标题你帮我想一个好一点的文案，控制在 8 个字以内",
        "source_files": [],
    }

    brief = build_visual_brief_from_payload(payload)
    prompt = compile_image2_prompt_payload(payload)["one_shot_design_prompt"]

    assert brief["copy"]["headline"] == "外酥里嫩臭豆腐"
    assert len(brief["copy"]["headline"]) <= 8
    assert "主标题「外酥里嫩臭豆腐」" in prompt
    assert "主标题「臭豆腐」" not in prompt


def test_explicit_plain_title_label_is_preserved_for_source_edit(tmp_path):
    source_image = tmp_path / "source.jpg"
    source_image.write_bytes(b"source")
    payload = {
        "source_platform": "feishu",
        "feishu_message_id": "om_title_explicit",
        "chat_id": "oc_chat",
        "root_id": "",
        "thread_id": "",
        "text": "/image2 重新美化这张图，增加水果元素。标题：夏日鲜果 冰柠系列",
        "source_files": [{"path": str(source_image), "mime_type": "image/jpeg", "source": "feishu_direct_media"}],
    }

    brief = build_visual_brief_from_payload(payload)
    prompt = compile_image2_prompt_payload(payload)["one_shot_design_prompt"]

    assert brief["copy"]["headline"] == "夏日鲜果 冰柠系列"
    assert "主标题「夏日鲜果 冰柠系列」" in prompt
