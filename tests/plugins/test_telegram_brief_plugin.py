import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.platforms.base import BasePlatformAdapter


PLUGIN_PATH = Path(__file__).parents[2] / "plugins" / "telegram-brief" / "__init__.py"


def _load_plugin():
    spec = importlib.util.spec_from_file_location("telegram_brief_test_plugin", PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module._reset_state_for_tests()
    return module


def _telegram_event(text="hello", user_id="u1", chat_id="c1", thread_id=None, profile=None):
    source = SimpleNamespace(
        platform=SimpleNamespace(value="telegram"),
        user_id=user_id,
        user_id_alt=None,
        chat_id=chat_id,
        thread_id=thread_id,
        profile=profile,
    )
    return SimpleNamespace(
        text=text,
        source=source,
        get_command=lambda: text.lstrip("/").split()[0] if text.startswith("/") else None,
    )


def _bind_event(plugin, event):
    plugin._turn_identity.set(plugin._identity_from_source(event.source))


def test_default_brief_prompt_has_required_final_schema():
    plugin = _load_plugin()
    _bind_event(plugin, _telegram_event())
    result = plugin._on_pre_llm_call(
        platform="telegram", sender_id="u1", session_id="s1", user_message="finish the task"
    )
    context = result["context"]
    for label in ("結果：", "變更：", "驗證：", "交付：", "阻礙：", "下一步："):
        assert label in context
    assert "10" in context
    assert "code fence" in context


def test_brief_and_detail_commands_switch_mode_for_current_identity():
    plugin = _load_plugin()
    detail_event = _telegram_event("/detail")
    _bind_event(plugin, detail_event)
    assert "detail" in plugin._handle_detail("", event=detail_event).lower()
    detail = plugin._on_pre_llm_call(
        platform="telegram", sender_id="u1", session_id="s1", user_message="normal request"
    )
    assert "DETAIL" in detail["context"]
    assert plugin._transform_llm_output(
        response_text="```python\nprint(1)\n```", platform="telegram", session_id="s1"
    ) is None

    brief_event = _telegram_event("/brief")
    _bind_event(plugin, brief_event)
    assert "brief" in plugin._handle_brief("", event=brief_event).lower()
    brief = plugin._on_pre_llm_call(
        platform="telegram", sender_id="u1", session_id="s2", user_message="normal request"
    )
    assert "BRIEF" in brief["context"]


def test_identityless_senders_are_isolated_by_chat_and_thread():
    plugin = _load_plugin()
    detail_event = _telegram_event("/detail", user_id=None, chat_id="admin-chat", thread_id="1")
    _bind_event(plugin, detail_event)
    plugin._handle_detail("", event=detail_event)
    assert plugin._mode_for_identity("default:telegram:chat:admin-chat:thread:1") == "detail"

    _bind_event(plugin, _telegram_event("hello", user_id=None, chat_id="other-chat", thread_id="2"))
    result = plugin._on_pre_llm_call(
        platform="telegram", sender_id=None, session_id="s2", user_message="normal request"
    )
    assert "BRIEF" in result["context"]


def test_explicit_detail_request_is_one_turn_exception_without_mode_switch():
    plugin = _load_plugin()
    _bind_event(plugin, _telegram_event())
    result = plugin._on_pre_llm_call(
        platform="telegram", sender_id="u1", session_id="s1", user_message="請顯示 diff"
    )
    assert "DETAIL" in result["context"]
    original = "```python\nprint('ok')\n```\ndiff --git a/x b/x"
    assert plugin._transform_llm_output(
        response_text=original, platform="telegram", session_id="s1"
    ) is None
    assert plugin._turn_allows_detail.get() is False

    next_turn = plugin._on_pre_llm_call(
        platform="telegram", sender_id="u1", session_id="s2", user_message="normal request"
    )
    assert "BRIEF" in next_turn["context"]


def test_negated_or_quoted_detail_phrases_do_not_enable_exception():
    plugin = _load_plugin()
    _bind_event(plugin, _telegram_event())
    for message in (
        "不要貼程式碼，也不用詳細說明",
        "do not show the code or show the diff",
        "請分析這段文字：『show the diff』，但維持精簡",
        "show the diff is a phrase to classify, not a request",
    ):
        result = plugin._on_pre_llm_call(
            platform="telegram", sender_id="u1", session_id="s1", user_message=message
        )
        assert "BRIEF" in result["context"]


def test_brief_transform_is_conservative_allowlist_for_code_diff_commands_and_logs():
    plugin = _load_plugin()
    _bind_event(plugin, _telegram_event())
    plugin._on_pre_llm_call(
        platform="telegram", sender_id="u1", session_id="s1", user_message="do it"
    )
    response = "\n".join([
        "結果：部分完成",
        "變更：updated several files",
        "def leaked_function():",
        "    return 'code'",
        "~~~python",
        "print('tilde fence')",
        "~~~",
        "diff --git a/a b/a",
        "+added_line()",
        "pytest -q",
        "Traceback (most recent call last):",
        "  File \"x.py\", line 1, in <module>",
        "RuntimeError: boom",
        "tool result contained private output",
        "阻礙：RuntimeError；測試未完成",
        "下一步：檢查本機完整 log 路徑",
    ])
    transformed = plugin._transform_llm_output(
        response_text=response, platform="telegram", session_id="s1"
    )
    assert transformed is not None
    for forbidden in (
        "def leaked", "return 'code'", "~~~", "diff --git", "+added_line",
        "pytest -q", "Traceback", 'File "x.py"', "tool result", "RuntimeError: boom",
    ):
        assert forbidden not in transformed
    assert "阻礙：RuntimeError；測試未完成" in transformed
    assert "print(secret_token)" not in plugin._sanitize_brief_response(
        "結果：完成\n變更：print(secret_token)"
    )
    assert "private_payload" not in plugin._sanitize_brief_response(
        "結果：完成\n變更：MEDIA:/tmp/private_payload.unknown"
    )
    assert len(transformed.splitlines()) <= 10
    assert len(transformed) <= 3500


def test_safe_english_punctuation_remains_usable():
    plugin = _load_plugin()
    for value in (
        "Updated parser (tests passed)",
        "Changed Bob's report safely; validation passed.",
        'Changed the “final report” — tests passed.',
    ):
        assert plugin._safe_field_value(value) == value


def test_requested_hardware_values_survive_brief_projection():
    plugin = _load_plugin()
    response = "\n".join(
        (
            "結果：完成",
            "變更：已即時讀取家中電腦硬體。",
            "驗證：以下為實際數值：",
            "- CPU：AMD Ryzen 5 5600X 6-Core Processor",
            "- 記憶體：總量 16.0 GB；目前可用 6.5 GB",
            "- GPU：NVIDIA GeForce RTX 3060 Ti",
            "- 專用 GPU 記憶體：8.0 GB",
            "- C 槽：總容量 952.9 GB；已用 895.4 GB；剩餘 57.5 GB",
            "- D 槽：總容量 953.9 GB；已用 681.6 GB；剩餘 272.3 GB",
        )
    )

    delivered = plugin._sanitize_brief_response(response)

    for expected in (
        "AMD Ryzen 5 5600X 6-Core Processor",
        "總量 16.0 GB",
        "目前可用 6.5 GB",
        "NVIDIA GeForce RTX 3060 Ti",
        "專用 GPU 記憶體：8.0 GB",
        "C 槽：總容量 952.9 GB",
        "D 槽：總容量 953.9 GB",
    ):
        assert expected in delivered
    assert "技術細節已省略" not in delivered


def test_plain_hardware_answer_without_brief_schema_is_preserved_verbatim():
    plugin = _load_plugin()
    response = (
        "CPU：AMD Ryzen 5 5600X 6-Core Processor\n"
        "RAM 總量：15.91 GiB；可用：2.87 GiB\n"
        "GPU：NVIDIA GeForce RTX 3060 Ti\n"
        "專用 VRAM：8192 MiB；shared memory 不計入\n"
        "C: 總量 952.90 GiB；剩餘 57.08 GiB"
    )

    assert plugin._sanitize_brief_response(response) == response


def test_non_schema_final_answer_keeps_safe_prose_instead_of_generic_fallback():
    plugin = _load_plugin()
    response = "答案是 42。\n這是使用者明確要求的完整正文。"

    delivered = plugin._sanitize_brief_response(response)

    assert delivered == response
    assert "最終摘要未符合 brief 格式" not in delivered
    assert "明確要求詳細說明" not in delivered


def test_fragmented_secret_across_answer_lines_is_redacted_as_a_whole():
    plugin = _load_plugin()
    left = "Ab9xY7zQ2mN8pL4r"
    right = "T6uV0wK3sJ5hG1fD7cB"
    response = f"結果：完成\n- {left}\n- {right}"

    delivered = plugin._sanitize_brief_response(response)

    assert left not in delivered
    assert right not in delivered
    assert "敏感內容已遮蔽" in delivered


def test_telegram_text_delivery_keeps_required_body_alongside_voice_media():
    plugin = _load_plugin()
    response = (
        "CPU：AMD Ryzen 5 5600X\n"
        "專用 VRAM：8 GiB\n"
        "[[audio_as_voice]]\nMEDIA:C:\\Users\\User\\memo.ogg"
    )

    delivered = plugin._sanitize_brief_response(response)
    files, text = BasePlatformAdapter.extract_media(delivered)

    assert "CPU：AMD Ryzen 5 5600X" in text
    assert "專用 VRAM：8 GiB" in text
    assert files == [("C:\\Users\\User\\memo.ogg", True)]
    assert "最終摘要未符合 brief 格式" not in delivered


def test_explicit_detailed_data_request_keeps_body_instead_of_omission_placeholder():
    plugin = _load_plugin()
    event = _telegram_event("請詳細列出 CPU、記憶體、GPU 專用記憶體與磁碟空間")
    _bind_event(plugin, event)
    plugin._on_pre_llm_call(
        platform="telegram", sender_id="u1", user_message=event.text
    )
    response = (
        "結果：完成\n"
        "驗證：硬體資料如下：\n"
        "- CPU：AMD Ryzen 5 5600X\n"
        "- 記憶體：16 GB\n"
        "- 專用 GPU 記憶體：8 GB\n"
        "- C 槽剩餘：57.5 GB"
    )
    delivered = plugin._transform_llm_output(
        platform="telegram", response_text=response
    ) or response

    assert "CPU：AMD Ryzen 5 5600X" in delivered
    assert "專用 GPU 記憶體：8 GB" in delivered
    assert "C 槽剩餘：57.5 GB" in delivered
    assert "技術細節已省略" not in delivered
    assert "明確要求詳細說明" not in delivered


def test_simple_success_operation_remains_compact():
    plugin = _load_plugin()
    response = "結果：完成\n變更：設定已儲存。\n驗證：回讀值一致。"
    assert plugin._sanitize_brief_response(response) == response


def test_markdown_heading_and_requested_list_survive_projection():
    plugin = _load_plugin()
    response = (
        "結果：完成\n"
        "驗證：實際硬體如下：\n"
        "**硬體數值**\n"
        "- CPU：AMD Ryzen 5 5600X\n"
        "- 專用 GPU 記憶體：8 GB"
    )
    delivered = plugin._sanitize_brief_response(response)

    assert "**硬體數值**" in delivered
    assert "CPU：AMD Ryzen 5 5600X" in delivered
    assert "專用 GPU 記憶體：8 GB" in delivered


def test_long_requested_list_is_preserved_for_telegram_adapter_chunking():
    plugin = _load_plugin()
    rows = [f"- 磁碟 {index:03d}：總容量 1000 GB；已用 400 GB；剩餘 600 GB；檔案系統 NTFS；狀態正常；已完成容量與可用空間實際回讀驗證，數值可供使用者直接採用" for index in range(55)]
    response = "結果：完成\n驗證：完整清單如下：\n" + "\n".join(rows)

    delivered = plugin._sanitize_brief_response(response)
    chunks = BasePlatformAdapter.truncate_message(delivered, 4096)

    assert len(delivered) > 4096
    assert rows[0] in delivered
    assert rows[-1] in delivered
    assert len(chunks) > 1
    assert all(len(chunk) <= 4096 for chunk in chunks)


def test_actionable_error_reason_survives_without_stack_trace():
    plugin = _load_plugin()
    response = "\n".join(
        (
            "結果：失敗",
            "阻礙：連線逾時，無法連上本機服務 127.0.0.1:9332。",
            "Traceback (most recent call last):",
            "RuntimeError: connection timed out",
            "下一步：確認服務已啟動並檢查 9332 連接埠後重試。",
        )
    )
    delivered = plugin._sanitize_brief_response(response)

    assert "連線逾時" in delivered
    assert "127.0.0.1:9332" in delivered
    assert "確認服務已啟動" in delivered
    assert "Traceback" not in delivered
    assert "RuntimeError" not in delivered
    assert "技術細節已省略" not in delivered


@pytest.mark.parametrize(
    "secret",
    (
        "sk-proj-" + "A" * 36,
        "ghp_" + "B" * 36,
        "xoxb-1234567890-" + "C" * 26,
        "hf_" + "D" * 36,
        "eyJhbGciOiJIUzI1NiJ9." + "eyJzdWIiOiIxMjM0NTY3ODkwIn0.signaturevalue",
        "Bearer " + "E" * 36,
        "api_key=" + "F" * 36,
    ),
)
def test_credential_shaped_values_are_forcibly_redacted(secret):
    plugin = _load_plugin()
    result = plugin._sanitize_brief_response(f"結果：完成\n變更：{secret}")
    assert secret not in result
    assert "敏感內容已遮蔽" in result


def test_unlabelled_high_entropy_values_are_redacted():
    plugin = _load_plugin()
    opaque_values = (
        "9f4a8c2e7b1d6f3a0c5e9b2d7a4f8c1e9d6b3a7f2c5e8d1a0b4f7c9e2d6a3b8f",
        "Ab9_xY7-" + "zQ2mN8pL4rT6uV0w" + "K3sJ5hG1fD7cB9eX" + "2aP8nM6qR4tZ",
        "a4ayc/80/OGda4BO/1o/" + "V0etpOqiLx1JwB5S3beH",
        "Ab9xY7zQ2mN8pL4r:" + "T6uV0wK3sJ5hG1fD7cB",
        "Ab9xY7zQ2mN8pL4r|" + "T6uV0wK3sJ5hG1fD7cB",
        "9f4a8c2e7b1d6f3a:" + "0c5e9b2d7a4f8c1e:" + "9d6b3a7f2c5e8d1a",
    )
    for value in opaque_values:
        result = plugin._sanitize_brief_response(f"結果：完成\n變更：{value}")
        assert value not in result
        assert "敏感內容已遮蔽" in result


def test_control_character_fragments_cannot_reconstruct_secret_after_screening():
    plugin = _load_plugin()
    left = "Ab9xY7zQ2mN8pL4r"
    right = "T6uV0wK3sJ5hG1fD7cB"
    reconstructed = left + right

    for separator in ("\t", "\x00", "\x1f", "\x7f"):
        fragmented = f"{left}{separator}{right}"
        assert plugin._contains_sensitive_material(fragmented) is True
        result = plugin._sanitize_brief_response(
            f"結果：完成\n變更：{fragmented}"
        )
        assert reconstructed not in result
        assert "敏感內容已遮蔽" in result


def test_high_entropy_media_basename_is_rejected():
    plugin = _load_plugin()
    opaque_names = (
        "Ab9_xY7-" + "zQ2mN8pL4rT6uV0w" + "K3sJ5hG1fD7cB9eX" + "2aP8nM6qR4tZ",
        "a4ayc/80/OGda4BO/1o/" + "V0etpOqiLx1JwB5S3beH",
        "Ab9xY7zQ2mN8pL4r:" + "T6uV0wK3sJ5hG1fD7cB",
        "Ab9xY7zQ2mN8pL4r|" + "T6uV0wK3sJ5hG1fD7cB",
        "9f4a8c2e7b1d6f3a:" + "0c5e9b2d7a4f8c1e:" + "9d6b3a7f2c5e8d1a",
    )
    for opaque_name in opaque_names:
        _bind_event(plugin, _telegram_event())
        plugin._on_pre_llm_call(
            platform="telegram", sender_id="u1", session_id="s1", user_message="do it"
        )
        delivered = plugin._transform_llm_output(
            response_text=f"結果：完成\nMEDIA:/tmp/{opaque_name}.txt",
            platform="telegram",
            session_id="s1",
        )
        assert delivered is not None
        assert opaque_name not in delivered
        files, _cleaned = BasePlatformAdapter.extract_media(delivered)
        assert files == []


def test_brief_transform_preserves_bounded_media_separately_from_text_limit():
    plugin = _load_plugin()
    _bind_event(plugin, _telegram_event())
    plugin._on_pre_llm_call(
        platform="telegram", sender_id="u1", session_id="s1", user_message="make attachments"
    )
    media = [f"MEDIA:C:\\Users\\User\\report-{i}.txt" for i in range(12)]
    response = "結果：完成\n變更：已建立附件\n驗證：實際檔案檢查完成\n" + "\n".join(media)
    transformed = plugin._transform_llm_output(
        response_text=response, platform="telegram", session_id="s1"
    )
    delivered = transformed or response
    files, cleaned = BasePlatformAdapter.extract_media(delivered)
    assert len(files) == 10
    assert all(item in delivered for item in media[:10])
    assert media[10] not in delivered and media[11] not in delivered
    assert len(delivered) <= 3900
    assert len(cleaned) <= 3500
    assert len(cleaned.splitlines()) <= 10


def test_media_directives_use_canonical_parser_and_preserve_delivery_semantics():
    plugin = _load_plugin()
    voice = plugin._sanitize_brief_response(
        "結果：完成\n[[audio_as_voice]]\nMEDIA:C:\\Users\\User\\memo.ogg\n"
        "MEDIA:<sensitive tool output>"
    )
    assert "[[audio_as_voice]]" in voice
    assert "MEDIA:C:\\Users\\User\\memo.ogg" in voice
    assert "sensitive tool output" not in voice

    document = plugin._sanitize_brief_response(
        "結果：完成\n[[as_document]]\nMEDIA:C:\\Users\\User\\chart.png"
    )
    assert "[[as_document]]" in document
    assert "MEDIA:C:\\Users\\User\\chart.png" in document


def test_unstructured_long_output_fails_closed_without_leaking_details():
    plugin = _load_plugin()
    _bind_event(plugin, _telegram_event())
    plugin._on_pre_llm_call(
        platform="telegram", sender_id="u1", session_id="s1", user_message="do it"
    )
    raw = "\n".join(["def secret():", "  return token"] + ["log output" for _ in range(50)])
    transformed = plugin._transform_llm_output(
        response_text=raw, platform="telegram", session_id="s1"
    )
    assert transformed.startswith("結果：無法安全顯示")
    assert "def secret" not in transformed
    assert "token" not in transformed
    assert "log output" not in transformed


def test_non_telegram_commands_are_rejected_without_mutating_telegram_mode():
    plugin = _load_plugin()
    event = _telegram_event("/detail")
    event.source.platform = SimpleNamespace(value="discord")
    assert "僅適用於 Telegram" in plugin._handle_detail("", event=event)
    assert plugin._modes_by_identity == {}
    plugin._turn_identity.set("default:telegram:user:stale")
    plugin._on_pre_gateway_dispatch(event=event, gateway=object(), session_store=object())
    assert plugin._turn_identity.get() is None


def test_non_telegram_output_is_untouched():
    plugin = _load_plugin()
    assert plugin._transform_llm_output(
        response_text="```python\nprint(1)\n```", platform="cli", session_id="s1"
    ) is None


def test_status_extension_reports_mode_without_rewriting_command():
    plugin = _load_plugin()
    event = _telegram_event("/status")
    _bind_event(plugin, event)
    status = plugin._extend_gateway_status(status="Hermes status", event=event, gateway=object())
    assert status == "Hermes status\n回報模式：brief"
    assert event.text == "/status"


def test_normal_messages_attachments_and_project_commands_are_not_rewritten(monkeypatch):
    plugin = _load_plugin()
    monkeypatch.setattr(plugin, "_transport_is_safe", lambda source, gateway: True)
    gateway = object()
    for text in ("normal task", "/status", "/focus C:\\Users\\User\\Dev\\folio", "/new"):
        event = _telegram_event(text)
        event.media_urls = ["C:\\Users\\User\\report.txt"]
        before = (event.text, list(event.media_urls))
        assert plugin._on_pre_gateway_dispatch(
            event=event, gateway=gateway, session_store=object()
        ) is None
        assert (event.text, event.media_urls) == before


def test_transport_safety_is_read_only_profile_scoped_and_revalidated(monkeypatch):
    from contextlib import contextmanager

    plugin = _load_plugin()
    safe = dict(plugin._REQUIRED_TELEGRAM_DISPLAY)
    configs = {
        "default": {"display": {"platforms": {"telegram": dict(safe)}}},
        "work": {"display": {"platforms": {"telegram": dict(safe)}}},
    }
    active = []

    @contextmanager
    def profile_scope(profile_home):
        active.append(str(profile_home))
        try:
            yield
        finally:
            active.pop()

    monkeypatch.setattr("gateway.run._profile_runtime_scope", profile_scope)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: configs[active[-1]])

    class Gateway:
        @staticmethod
        def _resolve_profile_home_for_source(source):
            return source.profile

    gateway = Gateway()
    default_source = _telegram_event(profile="default").source
    work_source = _telegram_event(profile="work").source
    assert plugin._transport_is_safe(default_source, gateway) is True
    assert plugin._transport_is_safe(work_source, gateway) is True

    configs["work"]["display"]["platforms"]["telegram"]["streaming"] = True
    snapshot = repr(configs["work"])
    assert plugin._transport_is_safe(work_source, gateway) is False
    assert repr(configs["work"]) == snapshot
    configs["work"]["display"]["platforms"]["telegram"]["streaming"] = False
    assert plugin._transport_is_safe(work_source, gateway) is True

    configs["work"]["display"]["runtime_footer"] = {"enabled": True}
    assert plugin._transport_is_safe(work_source, gateway) is False
    plugin._modes_by_identity[plugin._identity_from_source(work_source)] = "detail"
    with pytest.raises(RuntimeError, match="transport"):
        plugin._transform_gateway_output(
            platform="telegram",
            source=work_source,
            gateway=gateway,
            user_message="show detail",
            response_text="REASONING: hidden deliberation about tool calls",
        )
    configs["work"]["display"]["platforms"]["telegram"]["runtime_footer"] = {
        "enabled": False
    }
    assert plugin._transport_is_safe(work_source, gateway) is True


def test_transport_safety_failure_skips_message_before_dispatch(monkeypatch):
    plugin = _load_plugin()
    monkeypatch.setattr(plugin, "_transport_is_safe", lambda source, gateway: False)
    result = plugin._on_pre_gateway_dispatch(
        event=_telegram_event(), gateway=object(), session_store=object()
    )
    assert result["action"] == "skip"


def test_gateway_terminal_boundary_preserves_authorized_detail():
    plugin = _load_plugin()
    event = _telegram_event("run it")
    plugin._handle_detail("", event=event)
    detail = "Detailed explanation\n```python\nprint('ok')\n```"

    assert plugin._transform_gateway_output(
        platform="telegram",
        source=event.source,
        user_message=event.text,
        response_text=detail,
    ) is None
    assert plugin._validate_gateway_output(
        platform="telegram",
        source=event.source,
        user_message=event.text,
        response_text=detail,
    ) is True

    forced = plugin._transform_gateway_output(
        platform="telegram",
        source=event.source,
        user_message=event.text,
        response_text="REASONING: gateway-added hidden deliberation",
        force_brief=True,
    )
    assert forced is not None
    assert "hidden deliberation" not in forced


def test_gateway_terminal_boundary_rechecks_post_agent_mutations():
    plugin = _load_plugin()
    event = _telegram_event("run it")
    raw = (
        "The request failed: internal stack detail at C:/private/project\n"
        "MEDIA:C:/private/project/Ab9xY7zQ2mN8pL4rT6uV0wK3sJ5hG1fD7cB.txt\n"
        "model-name · C:/private/project"
    )

    transformed = plugin._transform_gateway_output(
        platform="telegram",
        source=event.source,
        user_message=event.text,
        response_text=raw,
    )

    assert transformed.startswith("結果：")
    assert "internal stack" not in transformed
    assert "MEDIA:" not in transformed
    assert "private/project" not in transformed
    assert plugin._validate_gateway_output(
        platform="telegram",
        source=event.source,
        user_message=event.text,
        response_text=transformed,
    ) is True


def test_terminal_validator_rejects_post_sanitizer_append():
    plugin = _load_plugin()
    event = _telegram_event("hello")
    _bind_event(plugin, event)
    plugin._on_pre_llm_call(
        platform="telegram", user_message="hello", sender_id="u1"
    )
    safe = plugin._transform_llm_output(
        platform="telegram",
        response_text="結果：完成\n變更：無\n下一步：無",
    )
    assert safe is None

    with pytest.raises(RuntimeError, match="terminal validation"):
        plugin._validate_llm_output(
            platform="telegram",
            response_text=(
                "結果：完成\n變更：無\n下一步：無\n"
                + "sk-"
                + "proj-unsafe-appended-value"
            ),
        )


def test_registration_adds_only_mode_commands_and_presentation_hooks_not_tools():
    plugin = _load_plugin()

    class Context:
        def __init__(self):
            self.commands = []
            self.hooks = []

        def register_command(self, name, handler, description="", args_hint=""):
            self.commands.append(name)

        def register_hook(self, name, handler):
            self.hooks.append(name)

    ctx = Context()
    plugin.register(ctx)
    assert set(ctx.commands) == {"brief", "detail"}
    assert set(ctx.hooks) == {
        "pre_gateway_transport",
        "pre_llm_call",
        "finalize_llm_output",
        "validate_llm_output",
        "finalize_gateway_output",
        "validate_gateway_output",
        "extend_gateway_status",
    }
