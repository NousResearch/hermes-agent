#!/usr/bin/env python3
"""Regression checks for director-lapian delivery workflow helpers."""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

import detect_shot_cuts as shot_detect
import lapian_delivery_status as status
import lapian_finalize_manifest as finalize
import qa_lapian_delivery as qa
import select_lapian_report_frames as frame_selector


def write_fake_docx(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "word/document.xml",
            '<w:document><w:drawing><a:blip r:embed="rId1"/></w:drawing></w:document>',
        )
        archive.writestr("word/media/image1.jpg", b"fake")


def make_project(root: Path) -> dict[str, Path]:
    project = root / "测试片_导演级拉片"
    frames = project / "01_逐秒抽帧" / "frames"
    audio = project / "02_音频分析"
    md_dir = project / "03_MD报告"
    docx_dir = project / "04_飞书Word交付"
    pdf_dir = project / "05_PDF预览"
    qa_dir = project / "06_QA审计"
    for item in [frames, audio, md_dir, docx_dir, pdf_dir, qa_dir]:
        item.mkdir(parents=True, exist_ok=True)

    for idx in range(1, 4):
        (frames / f"frame_{idx:06d}.jpg").write_bytes(b"jpg")

    report = md_dir / "测试片_导演级逐秒拉片报告.md"
    report.write_text(
        "\n".join(
            [
                "# 测试片",
                "## 素材诊断",
                "## 全片结构",
                "## 叙事",
                "## 视听语言",
                "## 主角表情与情感",
                "## 主题",
                "## 可学习方法",
                "## 交付审计",
                "![00:00](<../01_逐秒抽帧/frames/frame_000001.jpg>)",
                "![00:01](<../01_逐秒抽帧/frames/frame_000002.jpg>)",
                "![00:02](<../01_逐秒抽帧/frames/frame_000003.jpg>)",
            ]
        ),
        encoding="utf-8",
    )

    (audio / "测试片_audio_16k_mono.wav").write_bytes(b"wav")
    (audio / "测试片_audio_levels.csv").write_text("time,db\n0,-20\n", encoding="utf-8")
    (audio / "测试片_asr_base.txt").write_text("测试对白", encoding="utf-8")

    docx = docx_dir / "测试片_嵌图版.docx"
    write_fake_docx(docx)

    pdf = pdf_dir / "测试片_预览版.pdf"
    pdf.write_bytes(b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n%%EOF\n")

    manifest = project / "manifest.json"
    manifest.write_text(json.dumps({"source": {"title": "测试片"}}, ensure_ascii=False), encoding="utf-8")

    return {
        "project": project,
        "frames_dir": project / "01_逐秒抽帧",
        "audio_dir": audio,
        "report": report,
        "docx": docx,
        "pdf": pdf,
        "qa_dir": qa_dir,
        "manifest": manifest,
    }


def add_three_deliverables(paths: dict[str, Path]) -> dict[str, Path]:
    md_dir = paths["report"].parent
    docx_dir = paths["docx"].parent
    professional = md_dir / "测试片_专业导演分析_v01.md"
    professional.write_text(
        "\n".join(
            [
                "# 测试片_专业导演分析",
                "## 【第一幕】开场（00:00-00:02）",
                "| 时间 | 帧图 | 为什么选这一帧 |",
                "|---:|---|---|",
                "| 00:00 | ![](../01_逐秒抽帧/frames/frame_000001.jpg) | 情节：角色推门进入。构图：门框夹住人物。导演意图：用空间压迫代替解释。含义：自由被门槛限制。 |",
                "| 00:01 | ![](../01_逐秒抽帧/frames/frame_000002.jpg) | 情节：角色回头。构图：背影占前景。导演意图：停在犹豫。含义：离开的是旧关系。 |",
            ]
        ),
        encoding="utf-8",
    )
    bible = md_dir / "测试片_源片风格圣经_v01.md"
    bible.write_text(
        "\n".join(
            [
                "# 测试片_源片风格圣经",
                "## 1. 风格定位总述",
                "## 2. 风格 DNA 总表",
                "## 3. 影片段落式图片资产索引",
                "![00:00](../01_逐秒抽帧/frames/frame_000001.jpg)",
                "## 4. 场景级风格资产卡",
                "### 场景资产卡 S01：测试场景",
                "## 5. 元素级风格资产卡",
                "### 元素资产卡 E01：测试元素",
                "## 6. 图像生成变量模板",
                "## 7. 视频生成变量模板",
                "## 8. 风格保持与跑偏警报",
            ]
        ),
        encoding="utf-8",
    )
    professional_docx = docx_dir / "测试片_专业导演分析_嵌图版_v01.docx"
    write_fake_docx(professional_docx)
    bible_docx = docx_dir / "测试片_源片风格圣经_嵌图版_v01.docx"
    write_fake_docx(bible_docx)
    return {
        "professional_md": professional,
        "style_bible_md": bible,
        "professional_docx": professional_docx,
        "style_bible_docx": bible_docx,
    }


def assert_true(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def test_defaults_find_numbered_dirs() -> None:
    with tempfile.TemporaryDirectory() as temp:
        paths = make_project(Path(temp))
        defaults = qa.default_from_project(paths["project"])
        assert_true(defaults["frames_dir"] == paths["frames_dir"], "frames dir should be found by 01_ prefix")
        assert_true(defaults["audio_dir"] == paths["audio_dir"], "audio dir should be found by 02_ prefix")
        assert_true(defaults["report"] == paths["report"], "latest markdown report should be detected")
        assert_true(defaults["docx"] == paths["docx"], "latest docx should be detected")
        assert_true(defaults["pdf"] == paths["pdf"], "latest pdf should be detected")


def test_stable_output_and_pdf_audit() -> None:
    with tempfile.TemporaryDirectory() as temp:
        paths = make_project(Path(temp))
        out = paths["qa_dir"] / "lapian_delivery_qa.json"
        out.write_text("old", encoding="utf-8")
        assert_true(qa.unique_file(out, overwrite=True) == out, "overwrite mode should keep stable output path")
        assert_true(qa.unique_file(out, overwrite=False) != out, "versioned mode should avoid existing output")
        pdf_audit = qa.audit_pdf(paths["pdf"])
        assert_true(pdf_audit["exists"], "pdf should exist")
        assert_true(pdf_audit["has_pdf_header"], "pdf header should be detected")
        assert_true(pdf_audit["has_eof_marker"], "pdf EOF marker should be detected")


def test_status_and_manifest_update() -> None:
    with tempfile.TemporaryDirectory() as temp:
        paths = make_project(Path(temp))
        qa_json = paths["qa_dir"] / "lapian_delivery_qa.json"
        qa_json.write_text("{}", encoding="utf-8")
        (paths["qa_dir"] / "测试片_交付清单.json").write_text("{}", encoding="utf-8")
        scan = status.scan_project(paths["project"])
        assert_true(scan["done"]["frames"], "status should detect frames")
        assert_true(scan["done"]["markdown_report"], "status should detect markdown")
        assert_true(scan["done"]["docx_embedded"], "status should detect embedded docx")
        assert_true(scan["done"]["pdf_preview"], "status should detect valid pdf")
        assert_true(scan["done"]["asr_files"], "status should detect asr files")
        assert_true(scan["done"]["qa_json"], "status should detect final qa json")
        assert_true(
            Path(scan["paths"]["qa_json"]).resolve() == qa_json.resolve(),
            "status should prefer QA audit JSON over delivery manifest",
        )

        manifest_update = finalize.build_manifest_update(paths["project"])
        assert_true(
            manifest_update["reports"]["markdown_main"].startswith("03_"),
            "manifest should store project-relative markdown path",
        )
        assert_true(manifest_update["evidence"]["frame_count"] == 3, "manifest should include frame count")
        assert_true(manifest_update["completion"]["pdf_valid"], "manifest should include pdf validity")


def test_report_frame_selection_keeps_sparse_view_and_key_seconds() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        frames = root / "frames"
        frames.mkdir()
        manifest = root / "frames_manifest.json"
        items = []
        for idx in range(1, 13):
            second = idx - 1
            frame = frames / f"frame_{idx:06d}.jpg"
            frame.write_bytes(b"jpg")
            items.append(
                {
                    "index": idx,
                    "timestamp_seconds": float(second),
                    "timestamp_timecode": f"00:{second:02d}.000",
                    "file": str(frame),
                }
            )
        manifest.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
        subtitle_json = root / "subtitles.json"
        subtitle_json.write_text(
            json.dumps({"segments": [{"start": 3.1, "end": 3.8, "text": "key line"}]}, ensure_ascii=False),
            encoding="utf-8",
        )
        levels_csv = root / "levels.csv"
        levels_csv.write_text("second,start,end,rms,peak\n5,5.000,6.000,0.9,0.95\n", encoding="utf-8")

        plan = frame_selector.build_selection_plan(
            manifest_path=manifest,
            report_interval=2,
            subtitle_path=subtitle_json,
            levels_path=levels_csv,
            top_audio_peaks=1,
        )
        selected_seconds = {item["second"] for item in plan["selected_frames"]}
        assert_true(plan["selected_count"] < plan["total_frames"], "report plan should be sparser than evidence frames")
        assert_true(0 in selected_seconds and 11 in selected_seconds, "selection should keep first and last evidence seconds")
        assert_true(3 in selected_seconds, "selection should keep subtitle key second")
        assert_true(5 in selected_seconds, "selection should keep audio peak second")
        assert_true({2, 4, 6, 8, 10}.issubset(selected_seconds), "selection should keep regular 2-second cadence")


def test_qa_expected_image_count_can_use_selection_plan() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        plan_path = root / "selection.json"
        plan_path.write_text(
            json.dumps({"total_frames": 12, "selected_count": 8, "selected_frames": []}, ensure_ascii=False),
            encoding="utf-8",
        )
        assert_true(
            qa.expected_markdown_image_count(plan_path, fallback_frame_count=12) == 8,
            "QA should use report-frame selection count when a plan exists",
        )
        assert_true(
            qa.expected_markdown_image_count(None, fallback_frame_count=12) == 12,
            "QA should fall back to full frame count when no selection plan exists",
        )


def test_professional_frame_reason_audit_requires_deep_reading() -> None:
    bad_text = "\n".join(
        [
            "# 测试片_专业导演分析",
            "| 时间 | 帧图 | 为什么选这一帧 |",
            "|---:|---|---|",
            "| 00:00 | ![](frame_000001.jpg) | 作为本幕导演策略的视觉证据：它能显示空间、道具、表情、色彩或类型转换中的关键一步。 |",
            "| 00:01 | ![](frame_000002.jpg) | 作为本幕导演策略的视觉证据：它能显示空间、道具、表情、色彩或类型转换中的关键一步。 |",
        ]
    )
    bad_audit = qa.audit_professional_frame_reasons(bad_text, Path("测试片_专业导演分析.md"))
    assert_true(bad_audit["enabled"], "professional frame reason audit should enable for professional reports")
    assert_true(bad_audit["representative_frame_reason_rows"] == 2, "should detect frame reason rows")
    assert_true(bad_audit["four_part_rows"] == 0, "generic reasons should not count as deep readings")
    assert_true(bad_audit["duplicate_exact_count"] == 1, "duplicate exact reasons should be detected")
    assert_true(bool(bad_audit["generic_phrase_hits"]), "generic template phrase should be detected")

    good_text = "\n".join(
        [
            "# 测试片_专业导演分析",
            "| 时间 | 帧图 | 为什么选这一帧 |",
            "|---:|---|---|",
            "| 00:00 | ![](frame_000001.jpg) | 情节：角色推门进入。构图：门框把人物夹在窄缝里。导演意图：用空间压迫代替解释。含义：自由从一开始就被门槛限制。 |",
            "| 00:02 | ![](frame_000003.jpg) | 情节：角色回头看向屋内。构图：背影占前景，室内暗部吞掉后景。导演意图：让观众停在犹豫而非动作结果。含义：人物真正离开的不是房间，而是旧关系。 |",
        ]
    )
    good_audit = qa.audit_professional_frame_reasons(good_text, Path("测试片_专业导演分析.md"))
    assert_true(good_audit["four_part_rows"] == 2, "deep-reading reasons should contain all required terms")
    assert_true(good_audit["duplicate_exact_count"] == 0, "good reasons should be unique")
    assert_true(not good_audit["generic_phrase_hits"], "good reasons should not contain generic phrases")


def test_three_deliverable_enumeration_and_main_report_selection() -> None:
    with tempfile.TemporaryDirectory() as temp:
        paths = make_project(Path(temp))
        extra = add_three_deliverables(paths)
        reports = qa.list_markdown_reports(paths["project"])
        assert_true(len(reports) == 3, "all three markdown deliverables should be enumerated")
        assert_true(
            qa.latest_by_role(reports, qa.ROLE_PROFESSIONAL) == extra["professional_md"],
            "professional analysis should be classified by filename",
        )
        assert_true(
            qa.latest_by_role(reports, qa.ROLE_STYLE_BIBLE) == extra["style_bible_md"],
            "style bible should be classified by filename",
        )
        defaults = qa.default_from_project(paths["project"])
        assert_true(defaults["report"] == paths["report"], "main report selection should ignore the other two deliverables")
        assert_true(defaults["docx"] == paths["docx"], "main docx selection should ignore the other two deliverables")

        scan = status.scan_project(paths["project"])
        assert_true(scan["done"]["professional_analysis_md"], "status should detect professional analysis markdown")
        assert_true(scan["done"]["professional_analysis_docx"], "status should detect professional analysis docx")
        assert_true(scan["done"]["style_bible_md"], "status should detect style bible markdown")
        assert_true(scan["done"]["style_bible_docx"], "status should detect style bible docx")

        manifest_update = finalize.build_manifest_update(paths["project"])
        assert_true(
            manifest_update["reports"]["style_bible_md"].startswith("03_"),
            "manifest should store project-relative style bible path",
        )
        assert_true(
            manifest_update["deliverables"]["professional_analysis"]["representative_frame_rows"] == 2,
            "manifest should record representative frame row count",
        )
        assert_true(
            manifest_update["deliverables"]["professional_analysis"]["scene_block_count"] == 1,
            "manifest should record scene block count",
        )
        assert_true(
            manifest_update["deliverables"]["style_bible"]["asset_card_count"] == 2,
            "manifest should record style bible asset card count",
        )
        assert_true(
            manifest_update["deliverables"]["style_bible"]["missing_sections"] == [],
            "complete style bible should report no missing sections",
        )


def test_zero_image_report_is_flagged() -> None:
    audit = {
        "path": "测试片_图文逐秒报告.md",
        "role": qa.ROLE_MAIN,
        "exists": True,
        "markdown_images": 0,
        "missing_images": [],
    }
    failures, warnings = qa.evaluate_markdown_audit(audit, expected_report_images=8)
    assert_true(not failures, "zero-image main report should not hard-fail by itself")
    assert_true(
        any("fewer than expected" in w for w in warnings),
        "zero-image main report should warn against expected report image count",
    )
    assert_true(
        any("no image references" in w for w in warnings),
        "zero-image report should warn about missing image references",
    )


def test_style_bible_structure_audit() -> None:
    incomplete = qa.audit_style_bible_structure("# 风格圣经\n## 风格定位总述\n")
    assert_true("DNA" in incomplete["missing_sections"], "missing DNA section should be reported")
    assert_true(incomplete["asset_card_count"] == 0, "no asset cards should be counted in incomplete bible")
    audit = {
        "path": "测试片_源片风格圣经_v01.md",
        "role": qa.ROLE_STYLE_BIBLE,
        "exists": True,
        "markdown_images": 0,
        "missing_images": [],
        "style_bible_audit": incomplete,
    }
    failures, _ = qa.evaluate_markdown_audit(audit)
    assert_true(
        any("missing required sections" in f for f in failures),
        "style bible with missing sections should fail QA",
    )
    assert_true(
        any("no image asset references" in f for f in failures),
        "style bible without images should fail QA",
    )


def test_half_width_colon_reasons_accepted() -> None:
    text = "\n".join(
        [
            "# 测试片_专业导演分析",
            "| 时间 | 帧图 | 为什么选这一帧 |",
            "|---:|---|---|",
            "| 00:00 | ![](frame_000001.jpg) | 情节: 角色进入。构图: 门框压迫。导演意图: 空间代替台词。含义: 自由被限制。 |",
        ]
    )
    audit = qa.audit_professional_frame_reasons(text, Path("测试片_专业导演分析.md"))
    assert_true(audit["four_part_rows"] == 1, "half-width colons should also count as four-part deep readings")


def test_selection_plan_tolerates_missing_optional_inputs() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        frames = root / "frames"
        frames.mkdir()
        manifest = root / "frames_manifest.json"
        items = []
        for idx in range(1, 7):
            second = idx - 1
            frame = frames / f"frame_{idx:06d}.jpg"
            frame.write_bytes(b"jpg")
            items.append(
                {
                    "index": idx,
                    "timestamp_seconds": float(second),
                    "timestamp_timecode": f"00:{second:02d}.000",
                    "file": str(frame),
                }
            )
        manifest.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
        plan = frame_selector.build_selection_plan(
            manifest_path=manifest,
            report_interval=2,
            subtitle_path=root / "does_not_exist.json",
            levels_path=None,
        )
        assert_true(plan["selected_count"] > 0, "plan should be produced without subtitle/levels inputs")
        assert_true(plan["selected_count"] < plan["total_frames"], "plan should stay sparse without optional inputs")


def test_shot_cut_metadata_parsing_and_shot_building() -> None:
    metadata = "\n".join(
        [
            "frame:0    pts:123  pts_time:4.800",
            "lavfi.scene_score=0.456013",
            "frame:1    pts:456  pts_time:5.100",
            "lavfi.scene_score=0.310000",
            "frame:2    pts:789  pts_time:12.000",
            "lavfi.scene_score=0.900000",
            "frame:3    pts:999  pts_time:99.000",
            "lavfi.scene_score=0.500000",
        ]
    )
    raw = shot_detect.parse_scene_metadata(metadata)
    assert_true(len(raw) == 4, "all metadata cut candidates should be parsed")
    assert_true(raw[0]["time"] == 4.8 and raw[0]["score"] == 0.456013, "time and score should pair correctly")

    cuts = shot_detect.select_cuts(raw, duration=20.0, min_shot_seconds=0.5)
    times = [c["time"] for c in cuts]
    assert_true(99.0 not in times, "cuts past the video duration should be dropped")
    assert_true(5.1 not in times and 4.8 in times, "cuts closer than min shot length should merge into one")
    assert_true(12.0 in times, "distinct cuts should be kept")

    shots = shot_detect.build_shots(cuts, duration=20.0)
    assert_true(len(shots) == 3, "two cuts should produce three shots")
    assert_true(shots[0]["start_seconds"] == 0.0 and shots[-1]["end_seconds"] == 20.0, "shots should span the full video")
    stats = shot_detect.shot_stats(shots)
    assert_true(stats["shot_count"] == 3, "stats should count shots")
    assert_true(stats["max_shot_seconds"] == 8.0, "longest shot should be 12.0-20.0")
    assert_true(shot_detect.seconds_to_timecode(4.8) == "00:04.800", "timecode formatting should be MM:SS.mmm")


def test_shot_cut_merge_keeps_stronger_cut() -> None:
    raw = [
        {"time": 10.0, "score": 0.31},
        {"time": 10.3, "score": 0.95},
    ]
    cuts = shot_detect.select_cuts(raw, duration=30.0, min_shot_seconds=0.5)
    assert_true(len(cuts) == 1, "cuts within the merge window should collapse to one")
    assert_true(cuts[0]["time"] == 10.3 and cuts[0]["score"] == 0.95, "the stronger cut should win the merge")


def main() -> int:
    test_defaults_find_numbered_dirs()
    test_stable_output_and_pdf_audit()
    test_status_and_manifest_update()
    test_report_frame_selection_keeps_sparse_view_and_key_seconds()
    test_qa_expected_image_count_can_use_selection_plan()
    test_professional_frame_reason_audit_requires_deep_reading()
    test_three_deliverable_enumeration_and_main_report_selection()
    test_zero_image_report_is_flagged()
    test_style_bible_structure_audit()
    test_half_width_colon_reasons_accepted()
    test_selection_plan_tolerates_missing_optional_inputs()
    test_shot_cut_metadata_parsing_and_shot_building()
    test_shot_cut_merge_keeps_stronger_cut()
    print("lapian workflow helper tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
