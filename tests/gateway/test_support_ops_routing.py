from gateway.config import Platform
from gateway.run import _prepare_gateway_status_message
from gateway.support_ops_routing import (
    BACKEND_MENTION,
    FATIH_MENTION,
    KOZHUHAROV_MENTION,
    classify_support_ops_case_signal,
    lint_and_resolve_discord_content,
    resolve_teammate_route,
)


def test_kozhuharov_pbx_wrong_backend_mention_fails_closed():
    text = f"PBX/SIP outage, ново IP — Кожухаров {BACKEND_MENTION}"

    result = lint_and_resolve_discord_content(text)

    assert result.ok is False
    assert result.route is not None
    assert result.route.lane == "devops_kozhuharov"
    assert result.blocked_reason == "blocked_kozhuharov_route_requires_exact_mention"


def test_kozhuharov_pbx_no_exact_mention_fails_closed_by_design():
    text = "PBX/SIP outage SIP1/SIP2, ново IP 37.63.76.203 — да пишем на Емо Кожухаров"

    result = lint_and_resolve_discord_content(text)

    assert result.ok is False
    assert result.route is not None
    assert result.route.lane == "devops_kozhuharov"
    assert result.blocked_reason == "blocked_kozhuharov_route_missing_exact_mention"


def test_kozhuharov_pbx_exact_mention_passes():
    text = f"PBX/SIP outage SIP1/SIP2, ново IP 37.63.76.203 — {KOZHUHAROV_MENTION}"

    result = lint_and_resolve_discord_content(text)

    assert result.ok is True
    assert result.route is not None
    assert result.route.lane == "devops_kozhuharov"
    assert KOZHUHAROV_MENTION in result.content


def test_dm_kozhuharov_without_owner_dm_approval_fails_closed():
    text = f"DM Кожухаров за SIP/PBX firewall казуса {KOZHUHAROV_MENTION}"

    result = lint_and_resolve_discord_content(text)

    assert result.ok is False
    assert result.blocked_reason == "blocked_dm_requires_exact_owner_approval"


def test_alex_ivcho_voucher_unknown_user_fails_closed():
    text = "Voucher VS941215 / автоматична резервация не е сработила — Алекс/Ивчо @unknown-user"

    result = lint_and_resolve_discord_content(text)

    assert result.ok is False
    assert result.route is not None
    assert result.route.lane == "backend_alex_ivcho"
    assert result.blocked_reason == "blocked_backend_route_wrong_or_unknown_mention"


def test_alex_ivcho_text_at_mentions_resolve_to_single_backend_mention():
    for text in (
        "Voucher VS941215 / автоматична резервация не е сработила — @Алекс @Ивчо",
        "Voucher VS941215 / автоматична резервация не е сработила — @Алекс / @Ивчо",
        "VD5Y4664 reservation backend — @Алекс @Иво",
    ):
        result = lint_and_resolve_discord_content(text)

        assert result.ok is True
        assert result.route is not None
        assert result.route.lane == "backend_alex_ivcho"
        assert result.content.count(BACKEND_MENTION) == 1
        assert "@Алекс" not in result.content
        assert "@Ивчо" not in result.content
        assert "@Иво" not in result.content
        assert "1504852408227069993" not in result.content


def test_backend_plain_alex_ivcho_names_fail_closed_without_text_at_mentions():
    result = lint_and_resolve_discord_content("Алекс/Ивчо voucher backend казусът е за проверка")

    assert result.ok is False
    assert result.route is not None
    assert result.route.lane == "backend_alex_ivcho"
    assert result.blocked_reason == "blocked_plain_name_route_requires_explicit_text_at_mention"
    assert BACKEND_MENTION not in result.content


def test_backend_plain_ivo_name_fails_closed_without_text_at_mention():
    result = lint_and_resolve_discord_content("Иво да види reservation backend грешката")

    assert result.ok is False
    assert result.route is not None
    assert result.route.lane == "backend_alex_ivcho"
    assert result.blocked_reason == "blocked_plain_name_route_requires_explicit_text_at_mention"
    assert BACKEND_MENTION not in result.content


def test_backend_text_at_ivo_resolves_to_exact_backend_mentions_and_normalizes_display():
    result = lint_and_resolve_discord_content("@Иво да види reservation backend грешката")

    assert result.ok is True
    assert result.route is not None
    assert result.route.lane == "backend_alex_ivcho"
    assert result.content.count(BACKEND_MENTION) == 1
    assert "@Иво" not in result.content


def test_backend_wrong_known_mention_fails_closed():
    result = lint_and_resolve_discord_content(f"Алекс/Ивчо voucher backend — {KOZHUHAROV_MENTION}")

    assert result.ok is False
    assert result.blocked_reason == "blocked_backend_route_wrong_or_unknown_mention"


def test_fatih_frontend_route_resolves_exact_mention():
    result = lint_and_resolve_discord_content("@Фатих frontend FAB бутонът не се показва")

    assert result.ok is True
    assert result.route is not None
    assert result.route.lane == "frontend_fatih"
    assert FATIH_MENTION in result.content
    assert "@Фатих" not in result.content


def test_fatih_plain_name_frontend_route_fails_closed_without_text_at_mention():
    result = lint_and_resolve_discord_content("Фатих frontend FAB бутонът не се показва")

    assert result.ok is False
    assert result.route is not None
    assert result.route.lane == "frontend_fatih"
    assert result.blocked_reason == "blocked_plain_name_route_requires_explicit_text_at_mention"
    assert FATIH_MENTION not in result.content


def test_plamena_display_handle_normalized_in_authored_bulgarian_output():
    result = lint_and_resolve_discord_content("Пламена, ще проверя казуса и ще върна статус.")

    assert result.ok is True
    assert "Пламенка" in result.content
    assert "Пламена" not in result.content


def test_plamena_raw_quote_ambiguity_blocks_instead_of_rewriting_quote():
    result = lint_and_resolve_discord_content('Клиентът написа "Пламена ми каза" — проверявам.')

    assert result.ok is False
    assert result.blocked_reason == "blocked_plamena_raw_quote_ambiguity"


def test_alex_ivcho_text_at_mentions_without_exact_route_fail_closed():
    result = lint_and_resolve_discord_content("FYI @Алекс @Ивчо")

    assert result.ok is False
    assert result.blocked_reason == "blocked_unresolved_text_teammate_mention"


def test_unknown_user_without_exact_route_fails_closed():
    result = lint_and_resolve_discord_content("моля @unknown-user да погледне")

    assert result.ok is False
    assert result.blocked_reason == "blocked_unresolved_unknown_user_placeholder"


def test_route_requires_name_and_domain_context():
    assert resolve_teammate_route("Кожухаров FYI") is None
    assert resolve_teammate_route("има SIP проблем, но няма зададен teammate") is None


def test_plamena_request_to_write_emil_is_not_keyword_authority_in_lint_layer():
    text = "[Plamena] моля пиши на Емо Ломлиев - клиент е летял с него на 18 юни, на Приморско. Иска видеото си във формат МР4"

    result = lint_and_resolve_discord_content(text)

    assert result.ok is True
    assert result.route is None
    assert "Пламенка" in result.content


def test_discord_internal_codex_runtime_notice_is_suppressed():
    assert _prepare_gateway_status_message(
        Platform.DISCORD,
        "compression",
        "Runtime Codex compression notice: compacting context — summarizing earlier conversation",
    ) is None


def test_discord_exact_codex_gpt55_autoraise_notice_is_suppressed():
    assert _prepare_gateway_status_message(
        Platform.DISCORD,
        "status",
        "ℹ Codex gpt-5.5 caps context at 272K, so auto-compaction was raised to 85% (from 50%) to use more of the window before summarizing.\n  Opt back out: hermes config set compression.codex_gpt55_autoraise false",
    ) is None


def test_telegram_exact_codex_gpt55_autoraise_notice_is_suppressed_safely():
    assert _prepare_gateway_status_message(
        Platform.TELEGRAM,
        "status",
        "ℹ Codex gpt-5.5 caps context at 272K, so auto-compaction was raised to 85% (from 50%) to use more of the window before summarizing.\n  Opt back out: hermes config set compression.codex_gpt55_autoraise false",
    ) is None


def test_normal_discord_status_update_still_passes():
    assert _prepare_gateway_status_message(Platform.DISCORD, "status", "Working — 2 min — running tests") == "Working — 2 min — running tests"


def test_case_closure_phrases_classified():
    assert classify_support_ops_case_signal("Централата вече работи") == "case_closure"
    assert classify_support_ops_case_signal("случаят е готов") == "case_closure"
