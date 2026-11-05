"""Regression tests for the i18n model-fallback notice.

The notice emitted by ``try_activate_fallback`` used to be a hardcoded
English f-string.  It must now go through ``agent.i18n`` so non-English
users see a localized message while English output stays byte-identical.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent import i18n
from agent import chat_completion_helpers
from agent.error_classifier import FailoverReason
from run_agent import AIAgent


OLD_MODEL = "gpt-5.6-sol"
OLD_PROVIDER = "openai-codex"
NEW_MODEL = "glm-5.2"
NEW_PROVIDER = "zai"

EN_REASON_LABELS = {
    FailoverReason.rate_limit: "rate limit",
    FailoverReason.auth: "authentication failed",
    FailoverReason.billing: "billing or quota exhausted",
    FailoverReason.timeout: "request timeout",
    FailoverReason.unknown: "provider failure",
}

ZH_REASON_LABELS = {
    FailoverReason.rate_limit: "限流",
    FailoverReason.auth: "认证失败",
    FailoverReason.billing: "额度耗尽",
    FailoverReason.timeout: "超时",
    FailoverReason.unknown: "供应商故障",
}

ZH_HANT_REASON_LABELS = {
    FailoverReason.rate_limit: "限流",
    FailoverReason.auth: "認證失敗",
    FailoverReason.billing: "額度耗盡",
    FailoverReason.timeout: "逾時",
    FailoverReason.unknown: "供應商故障",
}


@pytest.fixture(autouse=True)
def _fresh_i18n_catalogs(monkeypatch):
    """Pin English and keep language/catalog state from leaking between tests."""
    monkeypatch.setenv("HERMES_LANGUAGE", "en")
    i18n.reset_language_cache()
    yield
    i18n.reset_language_cache()


def _make_agent() -> AIAgent:
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model={"provider": NEW_PROVIDER, "model": NEW_MODEL},
        )
    agent.client = MagicMock()
    agent.model = OLD_MODEL
    agent.provider = OLD_PROVIDER
    return agent


def _activate(agent: AIAgent, reason) -> str:
    client = MagicMock()
    client.base_url = "https://api.z.ai/v1"
    client.api_key = "fallback-key"
    with patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(client, NEW_MODEL),
    ), patch("agent.chat_completion_helpers.time.monotonic", return_value=1000):
        assert agent._try_activate_fallback(reason) is True
    return agent._pending_fallback_notice[-1]


def _expected_cooldown_suffix(lang: str) -> str:
    return i18n.t("agent.fallback.cooldown_suffix", lang=lang, seconds=60)


def _expected_zh_notice(reason_text: str, *, cooldown: bool = False) -> str:
    suffix = _expected_cooldown_suffix("zh") if cooldown else ""
    return (
        f"⚠️ 模型降级：{OLD_MODEL}（{OLD_PROVIDER}）不可用（{reason_text}）；"
        f"已切换为 {NEW_MODEL}（{NEW_PROVIDER}）。{suffix}"
    )


def _expected_zh_hant_notice(reason_text: str, *, cooldown: bool = False) -> str:
    suffix = _expected_cooldown_suffix("zh-hant") if cooldown else ""
    return (
        f"⚠️ 模型降級：{OLD_MODEL}（{OLD_PROVIDER}）無法使用（{reason_text}）；"
        f"已切換為 {NEW_MODEL}（{NEW_PROVIDER}）。{suffix}"
    )


def _expected_en_notice(reason_text: str) -> str:
    return (
        f"⚠️ Model fallback: {OLD_MODEL} via {OLD_PROVIDER} unavailable "
        f"({reason_text}); using {NEW_MODEL} via {NEW_PROVIDER}."
    )


# ---------------------------------------------------------------------------
# Simplified Chinese
# ---------------------------------------------------------------------------


def test_simplified_chinese_notice_is_translated(monkeypatch):
    monkeypatch.setenv("HERMES_LANGUAGE", "zh")
    notice = _activate(_make_agent(), FailoverReason.rate_limit)
    assert notice == _expected_zh_notice("限流", cooldown=True)


def test_translated_notice_preserves_dynamic_model_and_provider_values(monkeypatch):
    monkeypatch.setenv("HERMES_LANGUAGE", "zh")
    notice = _activate(_make_agent(), FailoverReason.billing)
    assert OLD_MODEL in notice
    assert OLD_PROVIDER in notice
    assert NEW_MODEL in notice
    assert NEW_PROVIDER in notice
    assert notice == _expected_zh_notice("额度耗尽", cooldown=True)
    assert "{" not in notice and "}" not in notice


@pytest.mark.parametrize(
    ("reason", "reason_label"),
    list(ZH_REASON_LABELS.items()),
    ids=[reason.value for reason in ZH_REASON_LABELS],
)
def test_all_five_supported_reasons_are_translated(monkeypatch, reason, reason_label):
    monkeypatch.setenv("HERMES_LANGUAGE", "zh")
    notice = _activate(_make_agent(), reason)
    assert f"（{reason_label}）" in notice
    assert notice == _expected_zh_notice(
        reason_label, cooldown=reason in {FailoverReason.rate_limit, FailoverReason.billing}
    )


def test_missing_reason_uses_provider_failure_translation(monkeypatch):
    monkeypatch.setenv("HERMES_LANGUAGE", "zh")
    notice = _activate(_make_agent(), None)
    assert notice == _expected_zh_notice("供应商故障")


@pytest.mark.parametrize(
    ("reason", "reason_label"),
    list(ZH_HANT_REASON_LABELS.items()),
    ids=[reason.value for reason in ZH_HANT_REASON_LABELS],
)
def test_traditional_chinese_notice_is_translated(monkeypatch, reason, reason_label):
    monkeypatch.setenv("HERMES_LANGUAGE", "zh-hant")
    notice = _activate(_make_agent(), reason)
    assert f"（{reason_label}）" in notice
    assert notice == _expected_zh_hant_notice(
        reason_label, cooldown=reason in {FailoverReason.rate_limit, FailoverReason.billing}
    )


# ---------------------------------------------------------------------------
# English -- must stay byte-identical to the pre-i18n implementation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reason", "reason_label"),
    list(EN_REASON_LABELS.items()),
    ids=[reason.value for reason in EN_REASON_LABELS],
)
def test_english_reason_output_is_unchanged(reason, reason_label):
    assert chat_completion_helpers._fallback_reason_text(reason) == reason_label


def test_english_notice_output_is_unchanged(monkeypatch):
    monkeypatch.setenv("HERMES_LANGUAGE", "en")
    notice = _activate(_make_agent(), FailoverReason.rate_limit)
    assert notice == _expected_en_notice("rate limit") + _expected_cooldown_suffix("en")


@pytest.mark.parametrize(
    ("lang", "expected"),
    [
        ("zh", " 主模型预计可在约 60 秒后重试；不保证恢复。"),
        ("zh-hant", " 主模型預計可於約 60 秒後重試；不保證恢復。"),
    ],
)
def test_cooldown_suffix_is_translated(monkeypatch, lang, expected):
    monkeypatch.setenv("HERMES_LANGUAGE", lang)
    notice = _activate(_make_agent(), FailoverReason.rate_limit)
    assert notice.endswith(expected)


@pytest.mark.parametrize(
    "lang", [lang for lang in i18n.SUPPORTED_LANGUAGES if lang not in {"zh", "zh-hant"}]
)
def test_other_locales_use_english_cooldown_suffix(lang):
    assert i18n.t("agent.fallback.cooldown_suffix", lang=lang, seconds=60) == (
        " Primary retry eligible in ~60 s; recovery is not guaranteed."
    )


def test_english_missing_reason_is_unchanged(monkeypatch):
    monkeypatch.setenv("HERMES_LANGUAGE", "en")
    assert chat_completion_helpers._fallback_reason_text(None) == "provider failure"
    notice = _activate(_make_agent(), None)
    assert notice == _expected_en_notice("provider failure")


# ---------------------------------------------------------------------------
# Unsupported reasons -- preserve the existing operator-facing labels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reason", "reason_label"),
    [
        (FailoverReason.overloaded, "provider overloaded"),
        (FailoverReason.server_error, "provider server error"),
        (FailoverReason.upstream_rate_limit, "upstream model rate limit"),
        (FailoverReason.auth_permanent, "authentication permanently failed"),
    ],
)
def test_unsupported_reason_keeps_existing_english_label(monkeypatch, reason, reason_label):
    monkeypatch.setenv("HERMES_LANGUAGE", "zh")
    assert chat_completion_helpers._fallback_reason_text(reason) == reason_label


def test_unknown_reason_object_keeps_existing_value_based_behavior(monkeypatch):
    monkeypatch.setenv("HERMES_LANGUAGE", "zh")

    class FutureReason:
        value = "future_failure"

    assert (
        chat_completion_helpers._fallback_reason_text(FutureReason())
        == "future failure"
    )


# ---------------------------------------------------------------------------
# Locale selection and i18n fallback behavior
# ---------------------------------------------------------------------------


def test_locale_selection_uses_hermes_language(monkeypatch):
    kwargs = dict(
        old_model=OLD_MODEL,
        old_provider=OLD_PROVIDER,
        reason="限流",
        new_model=NEW_MODEL,
        new_provider=NEW_PROVIDER,
    )
    monkeypatch.setenv("HERMES_LANGUAGE", "zh")
    assert i18n.t("agent.fallback.notice", **kwargs) == _expected_zh_notice("限流")

    monkeypatch.setenv("HERMES_LANGUAGE", "en")
    assert i18n.t("agent.fallback.notice", **kwargs) == _expected_en_notice("限流")


def test_explicit_lang_override_still_wins(monkeypatch):
    monkeypatch.setenv("HERMES_LANGUAGE", "en")
    translated = i18n.t(
        "agent.fallback.reason.rate_limit",
        lang="zh",
    )
    assert translated == "限流"


def test_missing_translation_key_falls_back_to_english(tmp_path, monkeypatch):
    """The i18n fallback contract must survive the new keys."""
    fake_locales = tmp_path / "locales"
    fake_locales.mkdir()
    (fake_locales / "en.yaml").write_text(
        "agent:\n  fallback:\n    notice: \"EN {reason}\"\n",
        encoding="utf-8",
    )
    (fake_locales / "zh.yaml").write_text("agent: {}\n", encoding="utf-8")
    monkeypatch.setattr(i18n, "_locales_dir", lambda: fake_locales)
    i18n.reset_language_cache()
    try:
        assert (
            i18n.t("agent.fallback.notice", lang="zh", reason="fallback")
            == "EN fallback"
        )
    finally:
        i18n.reset_language_cache()


def test_fallback_catalog_keys_exist_in_every_locale():
    required = {
        "agent.fallback.notice",
        "agent.fallback.reason.rate_limit",
        "agent.fallback.reason.auth",
        "agent.fallback.reason.billing",
        "agent.fallback.reason.timeout",
        "agent.fallback.reason.provider_failure",
        "agent.fallback.cooldown_suffix",
    }
    for lang in i18n.SUPPORTED_LANGUAGES:
        catalog = i18n._load_catalog(lang)
        missing = required - catalog.keys()
        assert not missing, f"{lang}.yaml missing {sorted(missing)}"
