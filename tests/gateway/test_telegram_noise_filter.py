"""Telegram-specific gateway filtering for noisy status/error output."""

from pathlib import Path

from gateway.config import Platform
from gateway.run import (
    _discord_kanban_terminal_message,
    _prepare_gateway_status_message,
    _sanitize_gateway_final_response,
)


def test_telegram_status_suppresses_auxiliary_and_retry_noise():
    """Auxiliary failures and retry backoff chatter should not hit Telegram."""
    noisy_messages = [
        "⚠ Auxiliary title generation failed: HTTP 400: Operation contains cybersecurity risk",
        "⚠ Compression summary failed: upstream error. Inserted a fallback context marker.",
        "📦 Preflight compression: ~138,265 tokens >= 136,000 threshold. This may take a moment.",
        "ℹ Configured compression model 'small-model' failed (timeout). Recovered using main model — check auxiliary.compression.model in config.yaml.",
        "⏳ Retrying in 4.2s (attempt 1/3)...",
        "⏱️ Rate limited. Waiting 30.0s (attempt 2/3)...",
        "⚠️ Max retries (3) exhausted — trying fallback...",
    ]

    for message in noisy_messages:
        assert _prepare_gateway_status_message(Platform.TELEGRAM, "warn", message) is None


def test_discord_status_suppresses_internal_noise_like_telegram():
    """Discord should keep chat clean from gateway/internal lifecycle chatter."""
    message = "⏳ Retrying in 4.2s (attempt 1/3)..."
    preflight = "📦 Preflight compression: ~138,265 tokens >= 136,000 threshold. This may take a moment."
    working = "確認します。"

    assert _prepare_gateway_status_message(Platform.DISCORD, "lifecycle", message) is None
    assert _prepare_gateway_status_message(Platform.DISCORD, "lifecycle", preflight) is None
    assert _prepare_gateway_status_message(Platform.DISCORD, "interim", working) is None
    assert _sanitize_gateway_final_response(Platform.DISCORD, preflight) == ""
    assert _prepare_gateway_status_message("local", "lifecycle", message) == message


def test_telegram_status_sanitizes_raw_provider_security_errors():
    """Provider policy/security bodies should be replaced before chat delivery."""
    raw = (
        "❌ API failed after 3 retries — HTTP 400: request blocked because "
        "Operation contains cybersecurity risk. request_id=req_123"
    )

    sanitized = _prepare_gateway_status_message(Platform.TELEGRAM, "lifecycle", raw)

    assert sanitized is not None
    assert "provider rejected" in sanitized.lower()
    assert "cybersecurity risk" not in sanitized.lower()
    assert "HTTP 400" not in sanitized
    assert "req_123" not in sanitized


def test_telegram_final_response_sanitizes_raw_provider_errors():
    """Final Telegram replies should not expose raw provider/security details."""
    raw = (
        "API call failed after 3 retries: HTTP 400: This request was blocked "
        "under the provider cybersecurity risk policy. request_id=req_abc"
    )

    sanitized = _sanitize_gateway_final_response(Platform.TELEGRAM, raw)

    assert "provider rejected" in sanitized.lower()
    assert "cybersecurity risk" not in sanitized.lower()
    assert "HTTP 400" not in sanitized
    assert "req_abc" not in sanitized


def test_telegram_final_response_redacts_auth_secrets():
    """Authentication errors should be useful without leaking key material."""
    raw = (
        "⚠️ Provider authentication failed: Incorrect API key provided: "
        "sk-live_abcdefghijklmnopqrstuvwxyz1234567890"
    )

    sanitized = _sanitize_gateway_final_response(Platform.TELEGRAM, raw)

    assert "authentication failed" in sanitized.lower()
    assert "check the configured credentials" in sanitized.lower()
    assert "sk-live" not in sanitized


def test_telegram_final_response_keeps_normal_answers():
    """Normal assistant content should not be rewritten."""
    answer = "Here is the clean summary you asked for."

    assert _sanitize_gateway_final_response(Platform.TELEGRAM, answer) == answer


def test_discord_status_strips_only_known_gateway_chrome_prefixes():
    gateway = "⏳ Gateway再起動中です。復旧後の次の返答に回しました。"
    restart = "♻ Gateway restarted successfully. Your session continues."
    normal = "ユーザー本文の絵文字は残す 😄"
    code = "```python\nprint('⚠️ hello')\n```"

    assert _prepare_gateway_status_message(Platform.DISCORD, "lifecycle", gateway) is None
    assert _prepare_gateway_status_message(Platform.DISCORD, "lifecycle", restart) is None
    assert _prepare_gateway_status_message(Platform.DISCORD, "lifecycle", normal) is None
    assert _sanitize_gateway_final_response(Platform.DISCORD, code) == code


def test_discord_final_response_strips_gateway_fallback_prefix_but_keeps_model_emoji():
    fallback = "⚠️ 処理が途中で止まりました: timeout。もう一度送ってください。"
    model_answer = "⚠️ 注意: これはモデルが本文として書いた注意です。"

    assert _sanitize_gateway_final_response(Platform.DISCORD, fallback) == (
        "処理が途中で止まりました: timeout。もう一度送ってください。"
    )
    assert _sanitize_gateway_final_response(Platform.DISCORD, model_answer) == model_answer


def test_discord_final_response_strips_leading_acknowledgements_only():
    answer = "内田さん、確認しました\n\n有効スキルは108件です。"
    inline = "調べました。たぶんHermesのスキルのことです。"
    code = "```python\nprint('確認しました')\n```"

    assert _sanitize_gateway_final_response(Platform.DISCORD, answer) == (
        "有効スキルは108件です。"
    )
    assert _sanitize_gateway_final_response(Platform.DISCORD, inline) == (
        "たぶんHermesのスキルのことです。"
    )
    assert _sanitize_gateway_final_response(Platform.DISCORD, code) == code


def test_discord_final_response_drops_checking_only_replies():
    assert _sanitize_gateway_final_response(Platform.DISCORD, "確認します。") == ""
    assert _sanitize_gateway_final_response(Platform.DISCORD, "では確認します。") == ""
    assert _sanitize_gateway_final_response(Platform.DISCORD, "こちらで調べて回答します。") == ""
    assert (
        _sanitize_gateway_final_response(
            Platform.DISCORD,
            "確認して回答します。\n\n有効なスキルは102件です。",
        )
        == "有効なスキルは102件です。"
    )


def test_discord_final_response_returns_result_without_work_report_prefixes():
    skill_count = (
        "Hermesスキルを再確認しました。現在この環境では102個のスキルが有効で、"
        "`hermes skills list` と `hermes skills search test` の動作も確認済みです。"
        "カテゴリはAI Company、GitHub、カンバン、Google Workspace、デザイン、調査です。"
    )
    skill_audit = (
        "Hermesスキルの利用状況を調査し、未使用候補79件・低使用候補1件を抽出しました。"
        "調査レポートとJSONをワークスペースに保存し、"
        "今回は削除・archive・設定変更など外部影響のある操作は行っていません。"
    )
    orchestration = (
        "依頼を3タスクに分解しました。Hermesスキル整理、"
        "GootHands Web流入・検索状況調査を並行で進め、"
        "その結果をbusiness-advisorが内田さん向けの短い回答に統合する流れにしました。"
    )
    live_bad_skill_count = "Hermesのスキル一覧を確認し、件数は102件でした。"

    assert _sanitize_gateway_final_response(Platform.DISCORD, skill_count).startswith(
        "有効なスキルは102個です。"
    )
    assert "再確認しました" not in _sanitize_gateway_final_response(
        Platform.DISCORD, skill_count
    )
    assert "確認できています" not in _sanitize_gateway_final_response(
        Platform.DISCORD, skill_count
    )
    assert _sanitize_gateway_final_response(Platform.DISCORD, skill_audit) == (
        "未使用候補は79件、低使用候補は1件でした。"
        "削除やアーカイブ、設定変更はしていません。"
    )
    assert _sanitize_gateway_final_response(Platform.DISCORD, orchestration) == ""
    assert _sanitize_gateway_final_response(
        Platform.DISCORD, live_bad_skill_count
    ) == "有効なスキルは102件です。"


def test_discord_kanban_error_notifications_do_not_say_will_check():
    event = type("Event", (), {"payload": {}})()

    assert _discord_kanban_terminal_message(
        "crashed", None, event, "テストタスク"
    ) == "「テストタスク」の処理が途中で止まりました。再実行対象です。"
    assert _discord_kanban_terminal_message(
        "timed_out", None, event, "テストタスク"
    ) == "「テストタスク」の処理に時間がかかりすぎました。再実行対象です。"


def test_telegram_existing_status_display_is_preserved():
    message = "⏳ Gateway再起動中です。復旧後の次の返答に回しました。"

    assert _prepare_gateway_status_message(Platform.TELEGRAM, "lifecycle", message) == message


def test_discord_gateway_chrome_static_paths_are_covered():
    source = Path(__import__("gateway.run").run.__file__).read_text()

    assert "source.platform not in {Platform.WEBHOOK, Platform.DISCORD}" in source
    assert "warning_message = _strip_discord_gateway_chrome_emoji(warning_message)" in source
    assert "if source.platform == Platform.DISCORD:" in source
