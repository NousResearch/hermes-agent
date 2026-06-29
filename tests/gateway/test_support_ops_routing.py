from gateway.config import Platform
from gateway.run import _prepare_gateway_status_message
from gateway.support_ops_routing import (
    BACKEND_MENTION,
    FATIH_MENTION,
    KOZHUHAROV_MENTION,
    ALEX_MENTION,
    PLAMENA_MENTION,
    SKYVISION_BACKEND_CHANNEL_ID,
    SKYVISION_CONTROL_TOWER_CHANNEL_ID,
    lint_discord_target_for_content,
    lint_and_resolve_discord_content,
)


def test_kozhuharov_pbx_words_do_not_route_or_block():
    text = "PBX/SIP outage SIP1/SIP2, ново IP 37.63.76.203 — да пишем на Емо Кожухаров"

    result = lint_and_resolve_discord_content(text)

    assert result.ok is True
    assert result.blocked_reason is None
    assert result.content == text
    assert not hasattr(result, "route")


def test_backend_words_and_plain_names_do_not_route_or_block():
    text = "Алекс/Ивчо voucher backend казусът е за проверка"

    result = lint_and_resolve_discord_content(text)

    assert result.ok is True
    assert result.blocked_reason is None
    assert result.content == text
    assert BACKEND_MENTION not in result.content
    assert not hasattr(result, "route")


def test_backend_plain_ivo_name_does_not_route_or_rewrite():
    text = "Иво да види reservation backend грешката"

    result = lint_and_resolve_discord_content(text)

    assert result.ok is True
    assert result.blocked_reason is None
    assert result.content == text


def test_frontend_words_and_plain_name_do_not_route_or_block():
    text = "Фатих frontend FAB бутонът не се показва"

    result = lint_and_resolve_discord_content(text)

    assert result.ok is True
    assert result.blocked_reason is None
    assert result.content == text
    assert FATIH_MENTION not in result.content
    assert not hasattr(result, "route")


def test_exact_known_mentions_pass_without_route_inference():
    for text in (
        f"PBX/SIP outage SIP1/SIP2, ново IP 37.63.76.203 — {KOZHUHAROV_MENTION}",
        f"Voucher VS941215 / автоматична резервация не е сработила — {BACKEND_MENTION}",
        f"frontend FAB бутонът не се показва — {FATIH_MENTION}",
    ):
        result = lint_and_resolve_discord_content(text)

        assert result.ok is True
        assert result.blocked_reason is None
        assert result.content == text
        assert not hasattr(result, "route")


def test_backend_resolver_mention_requires_backend_lane_without_keyword_inference():
    text = f"{ALEX_MENTION} моля за действие по клиентския бонус."

    result = lint_discord_target_for_content(
        text,
        chat_id=SKYVISION_CONTROL_TOWER_CHANNEL_ID,
        thread_id="1521047924069371954",
    )

    assert result.ok is False
    assert result.blocked_reason == "blocked_backend_resolver_mention_wrong_discord_lane"
    assert result.expected_channel_id == SKYVISION_BACKEND_CHANNEL_ID


def test_backend_resolver_mention_passes_in_backend_lane():
    text = f"{ALEX_MENTION} моля за действие по клиентския бонус."

    result = lint_discord_target_for_content(
        text,
        chat_id=SKYVISION_BACKEND_CHANNEL_ID,
        thread_id="1521049963428053125",
    )

    assert result.ok is True


def test_mixed_backend_resolver_and_requester_mentions_are_blocked_even_in_backend_lane():
    text = f"{ALEX_MENTION} {PLAMENA_MENTION} — нов клиентски кейс за действие."

    result = lint_discord_target_for_content(
        text,
        chat_id=SKYVISION_BACKEND_CHANNEL_ID,
        thread_id="1521049963428053125",
    )

    assert result.ok is False
    assert result.blocked_reason == "blocked_mixed_backend_resolver_and_requester_mentions"


def test_raw_text_backend_mentions_fail_closed_without_rewrite():
    for text in (
        "Voucher VS941215 / автоматична резервация не е сработила — @Алекс @Ивчо",
        "Voucher VS941215 / автоматична резервация не е сработила — @Алекс / @Ивчо",
        "VD5Y4664 reservation backend — @Алекс @Иво",
        "FYI @Алекс @Ивчо",
    ):
        result = lint_and_resolve_discord_content(text)

        assert result.ok is False
        assert result.blocked_reason == "blocked_unresolved_text_teammate_mention"
        assert result.content == text


def test_raw_text_kozhuharov_mention_fails_closed_without_route_inference():
    text = "PBX/SIP outage — @Кожухаров"

    result = lint_and_resolve_discord_content(text)

    assert result.ok is False
    assert result.blocked_reason == "blocked_unresolved_text_teammate_mention"
    assert result.content == text


def test_raw_text_fatih_mention_fails_closed_without_route_inference():
    text = "@Фатих frontend FAB бутонът не се показва"

    result = lint_and_resolve_discord_content(text)

    assert result.ok is False
    assert result.blocked_reason == "blocked_unresolved_text_teammate_mention"
    assert result.content == text


def test_unknown_user_without_exact_route_fails_closed():
    result = lint_and_resolve_discord_content("моля @unknown-user да погледне")

    assert result.ok is False
    assert result.blocked_reason == "blocked_unresolved_unknown_user_placeholder"


def test_plamena_display_handle_normalized_in_authored_bulgarian_output():
    result = lint_and_resolve_discord_content("Пламена, ще проверя казуса и ще върна статус.")

    assert result.ok is True
    assert "Пламенка" in result.content
    assert "Пламена" not in result.content


def test_plamena_raw_quote_ambiguity_blocks_instead_of_rewriting_quote():
    result = lint_and_resolve_discord_content('Клиентът написа "Пламена ми каза" — проверявам.')

    assert result.ok is False
    assert result.blocked_reason == "blocked_plamena_raw_quote_ambiguity"


def test_plamena_request_to_write_emil_is_not_keyword_authority_in_lint_layer():
    text = "[Plamena] моля пиши на Емо Ломлиев - клиент е летял с него на 18 юни, на Приморско. Иска видеото си във формат МР4"

    result = lint_and_resolve_discord_content(text)

    assert result.ok is True
    assert "Пламенка" in result.content
    assert not hasattr(result, "route")


def test_case_closure_words_are_not_classified_by_lint_layer():
    for text in ("Централата вече работи", "случаят е готов"):
        result = lint_and_resolve_discord_content(text)

        assert result.ok is True
        assert result.blocked_reason is None
        assert result.content == text


def test_discord_internal_codex_runtime_notice_is_suppressed():
    assert (
        _prepare_gateway_status_message(
            Platform.DISCORD,
            "compression",
            "Runtime Codex compression notice: compacting context — summarizing earlier conversation",
        )
        is None
    )


def test_discord_exact_codex_gpt55_autoraise_notice_is_suppressed():
    assert (
        _prepare_gateway_status_message(
            Platform.DISCORD,
            "status",
            "ℹ Codex gpt-5.5 caps context at 272K, so auto-compaction was raised to 85% (from 50%) to use more of the window before summarizing.\n  Opt back out: hermes config set compression.codex_gpt55_autoraise false",
        )
        is None
    )


def test_telegram_exact_codex_gpt55_autoraise_notice_is_suppressed_safely():
    assert (
        _prepare_gateway_status_message(
            Platform.TELEGRAM,
            "status",
            "ℹ Codex gpt-5.5 caps context at 272K, so auto-compaction was raised to 85% (from 50%) to use more of the window before summarizing.\n  Opt back out: hermes config set compression.codex_gpt55_autoraise false",
        )
        is None
    )


def test_normal_discord_status_update_still_passes():
    assert _prepare_gateway_status_message(Platform.DISCORD, "status", "Working — 2 min — running tests") == "Working — 2 min — running tests"
