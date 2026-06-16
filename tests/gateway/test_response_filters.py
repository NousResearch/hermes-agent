from gateway.response_filters import (
    collapse_user_visible_runtime_error,
    is_intentional_silence_agent_result,
    is_intentional_silence_response,
    is_model_identity_query,
    model_identity_private_reply,
    redact_user_visible_local_paths,
    sanitize_user_visible_gateway_text,
)


def test_exact_silence_tokens_are_intentional_silence():
    for token in ("[SILENT]", " SILENT ", "NO_REPLY", "no reply"):
        assert is_intentional_silence_response(token)


def test_blank_and_prose_mentions_are_not_silence():
    assert not is_intentional_silence_response("")
    assert not is_intentional_silence_response("Use NO_REPLY when no answer is needed.")
    assert not is_intentional_silence_response("The reply was [SILENT], intentionally.")


def test_failed_agent_result_never_counts_as_intentional_silence():
    assert is_intentional_silence_agent_result({"failed": False}, "NO_REPLY")
    assert not is_intentional_silence_agent_result({"failed": True}, "NO_REPLY")


def test_redacts_user_visible_codex_and_hermes_paths():
    text = (
        "Saved image at /home/dgx/.codex/generated_images/abc/image.png "
        "and cache /home/dgx/.hermes/cache/images/out.png"
    )

    result = redact_user_visible_local_paths(text)

    assert "/home/dgx/.codex" not in result
    assert "/home/dgx/.hermes" not in result
    assert result.count("[local file]") == 2


def test_redacts_general_user_home_paths():
    text = '日志文件：`"/home/dgx/.hermes/logs/gateway.log"` 和 /home/dgx/github/file.txt'

    result = redact_user_visible_local_paths(text)

    assert "/home/dgx/" not in result
    assert result.count("[local file]") == 2


def test_can_skip_path_redaction_before_media_extraction():
    text = "图片在 /home/dgx/.codex/generated_images/a/out.png，Hermes 会发送。"

    result = sanitize_user_visible_gateway_text(text, redact_paths=False)

    assert "/home/dgx/.codex/generated_images/a/out.png" in result
    assert "Hermes" not in result
    assert "当前模型服务" in result


def test_preserves_media_directive_path_for_attachment_extraction():
    text = (
        "Done.\n"
        "MEDIA:/home/dgx/.codex/generated_images/abc/image.png\n"
        "Visible path: /home/dgx/.codex/generated_images/abc/image.png"
    )

    result = redact_user_visible_local_paths(text)

    assert "MEDIA:/home/dgx/.codex/generated_images/abc/image.png" in result
    assert "Visible path: [local file]" in result


def test_preserves_quoted_media_directive_path():
    text = 'MEDIA:"/home/dgx/.hermes/cache/images/out.png"'

    result = redact_user_visible_local_paths(text)

    assert result == text


def test_sanitizes_model_identity_from_user_visible_text():
    text = (
        "我是 GPT-5.5，通过 openai-codex / codex_app_server 运行，"
        "后端是 OpenAI。"
    )

    result = sanitize_user_visible_gateway_text(text)

    assert "GPT-5.5" not in result
    assert "openai-codex" not in result
    assert "codex_app_server" not in result
    assert "OpenAI" not in result
    assert "当前大模型" in result
    assert "当前模型服务" in result


def test_sanitizes_codex_context_status_notice():
    text = (
        "Codex gpt-5.5 caps context at 272K, so auto-compaction was raised."
    )

    result = sanitize_user_visible_gateway_text(text)

    assert "Codex" not in result
    assert "gpt-5.5" not in result
    assert "当前模型服务" in result
    assert "当前大模型" in result


def test_sanitizes_hermes_identity_from_user_visible_text():
    text = "这是 Hermes / hermes-agent 提供的回复。"

    result = sanitize_user_visible_gateway_text(text)

    assert "Hermes" not in result
    assert "hermes-agent" not in result
    assert result.count("当前模型服务") == 2


def test_sanitizes_recent_gateway_log_leak_example():
    text = (
        '主人，月见喵找到了当前系统的日志文件：`"/home/dgx/.hermes/logs/gateway.log"`，'
        "共 433 行。"
    )

    result = sanitize_user_visible_gateway_text(text)

    assert "/home/dgx" not in result
    assert ".hermes" not in result
    assert "主人，月见喵找到了" in result
    assert "[local file]" in result


def test_collapses_raw_runtime_failure_dump_before_gateway_delivery():
    text = (
        '⚠️ turn ended status=failed: {"type":"error","status":400,'
        '"error":{"type":"invalid_request_error"}}\n'
        "当前模型服务 stderr (last 12 lines):\n"
        "<!doctype html><html><title>Cloudflare</title><script>cf_chl</script>"
    )

    result = sanitize_user_visible_gateway_text(text)

    assert "调用失败" in result
    assert "turn ended" not in result
    assert "stderr" not in result
    assert "Cloudflare" not in result
    assert "cf_chl" not in result
    assert "<html" not in result


def test_preserves_normal_text_when_collapsing_runtime_errors():
    text = "这是一条正常回复，提到了 Cloudflare 的公开概念。"

    assert collapse_user_visible_runtime_error(text) == text


def test_detects_current_assistant_model_identity_queries():
    assert is_model_identity_query("你现在用的是什么大模型")
    assert is_model_identity_query("你是什么大模型和什么程序？")
    assert is_model_identity_query("what model are you running?")


def test_does_not_block_general_model_recommendation_question():
    assert not is_model_identity_query("帮我推荐一个适合写代码的大模型")


def test_model_identity_private_reply_has_no_specific_backend_names():
    reply = model_identity_private_reply()

    assert "GPT" not in reply
    assert "Codex" not in reply
    assert "OpenAI" not in reply
