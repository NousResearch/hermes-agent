"""Gateway notification strings resolve ``display.language`` at send time.

Contract (not snapshots): flipping the configured language must flip the
user-visible label while keeping every interpolated value (:: prefix, ids,
summaries) intact, and the Discord history recognizer must treat a localized
self-improvement review boundary exactly like the English one.
"""

from __future__ import annotations

import pytest

from agent import i18n


def _use_language(monkeypatch, tmp_path, lang):
    """Point HERMES_HOME at a fresh dir with (optionally) display.language set."""
    home = tmp_path / f"home_{lang or 'default'}"
    home.mkdir(exist_ok=True)
    if lang is not None:
        (home / "config.yaml").write_text(f"display:\n  language: {lang}\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Hermetic catalog resolution: never inherit the developer's override dir.
    monkeypatch.delenv("HERMES_BUNDLED_LOCALES", raising=False)
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    i18n.reset_language_cache()
    try:
        yield home
    finally:
        i18n.reset_language_cache()


@pytest.fixture()
def lang_ja(monkeypatch, tmp_path):
    yield from _use_language(monkeypatch, tmp_path, "ja")


@pytest.fixture()
def lang_en(monkeypatch, tmp_path):
    yield from _use_language(monkeypatch, tmp_path, None)


def test_default_language_renders_english_labels(lang_en):
    """With no language configured, the English source strings render as before."""
    from gateway.run import _format_exec_approval_fallback

    approval = _format_exec_approval_fallback("rm", "d", "/")
    assert "Reason: d" in approval
    assert "理由" not in approval


def test_ja_flips_user_visible_labels(lang_ja):
    """Japanese config renders localized labels, not the English source strings."""
    from gateway.delivery_ledger import recovered_marker
    from gateway.run import _format_exec_approval_fallback

    approval = _format_exec_approval_fallback("rm -rf /tmp/x", "dangerous", "/")
    assert "理由: dangerous" in approval
    assert "Reason: " not in approval
    assert "/approve" in approval  # command tokens are identifiers, never translated

    marker = recovered_marker()
    assert "ゲートウェイ" in marker
    assert "restarted during delivery" not in marker

    assert "自己改善レビュー" in i18n.t("gateway.review.summary", summary="x")
    assert "Self-improvement review" not in i18n.t("gateway.review.summary", summary="x")


def test_localized_renders_keep_interpolated_values(lang_ja):
    """Localization must never drop the dynamic values the user must act on."""
    from gateway.delivery_ledger import flood_marker
    from gateway.run import _format_concise_process_notification, _format_exec_approval_fallback

    approval = _format_exec_approval_fallback("sudo systemctl restart x", "desc", "!")
    assert "!approve" in approval and "!deny" in approval
    assert "sudo systemctl restart x" in approval

    failed = _format_concise_process_notification("s1", "make build", 3, "boom", 5)
    assert "3" in failed and "make build" in failed

    assert "flood_marker" not in flood_marker()  # catalog hit, not a key echo
    busy = i18n.t("gateway.busy.drain_queued", action="restarting")
    assert "restarting" in busy
    lease = i18n.t("gateway.lease.timeout")
    assert "Hermes" in lease


def test_discord_recognizer_covers_localized_review_label():
    """Localized review boundary is skipped in history exactly like English."""
    from plugins.platforms.discord import adapter as discord_platform

    recognize = discord_platform._looks_like_nonconversational_history_message
    assert recognize("💾 Self-improvement review: Memory updated")
    assert recognize("💾 自己改善レビュー: メモリを更新")
    assert not recognize("💾 明日の買い物リスト: 牛乳と卵を買う")
