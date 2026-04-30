from agent.response_style import (
    apply_response_style_guard,
    build_response_style_prompt,
    is_response_style_enabled,
)


def _cfg(**overrides):
    base = {
        "enabled": True,
        "profile": "secretary",
        "platforms": ["telegram"],
        "require_labels": True,
        "max_chars": 700,
    }
    base.update(overrides)
    return {"response_style": base}


def test_secretary_prompt_injected_for_enabled_platform():
    prompt = build_response_style_prompt(_cfg(), "telegram")

    assert "secretary mode" in prompt
    assert "Result:" in prompt
    assert "Blocker:" in prompt
    assert "Next step:" in prompt


def test_style_disabled_for_other_platform():
    assert not is_response_style_enabled(_cfg(), "discord")
    assert build_response_style_prompt(_cfg(), "discord") == ""


def test_guard_reformats_long_unlabeled_gateway_response():
    response = "已完成配置和代码修改。\n- 文件 A 已修改\n- 文件 B 已修改\n下一步：等待静默窗口热加载。"

    guarded = apply_response_style_guard(response, _cfg(), "telegram", "继续")

    assert guarded.startswith("Result:\n")
    assert "\n\nBlocker:\n" in guarded
    assert "\n\nNext step:\n" in guarded
    assert len(guarded) <= 700


def test_guard_preserves_explicit_detail_requests():
    response = "第一行\n第二行\n第三行"

    guarded = apply_response_style_guard(response, _cfg(), "telegram", "详细展开")

    assert guarded == response


def test_guard_preserves_media_responses():
    response = "MEDIA:/tmp/example.png\n这是图片"

    guarded = apply_response_style_guard(response, _cfg(), "telegram", "发图")

    assert guarded == response


def test_guard_preserves_exact_silent_marker():
    response = "[SILENT]"

    guarded = apply_response_style_guard(response, _cfg(), "telegram", "继续")

    assert guarded == "[SILENT]"


def test_guard_preserves_silent_marker_with_boundary_whitespace():
    response = "\n  [SILENT]\t\n"

    guarded = apply_response_style_guard(response, _cfg(), "telegram", "继续")

    assert guarded == "[SILENT]"


def test_guard_does_not_treat_mixed_silent_content_as_control_marker():
    response = "[SILENT]\nResult: PR state changed"

    guarded = apply_response_style_guard(response, _cfg(), "telegram", "继续")

    assert guarded != "[SILENT]"
    assert "PR state changed" in guarded
