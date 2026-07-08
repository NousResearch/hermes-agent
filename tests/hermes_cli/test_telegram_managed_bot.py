"""Tests for hermes_cli.telegram_managed_bot — QR codes, deep links, pairing."""

from __future__ import annotations

from pathlib import PureWindowsPath
from unittest.mock import MagicMock, patch

from hermes_cli.telegram_managed_bot import (
    DEFAULT_MANAGER_BOT,
    TELEGRAM_ONBOARDING_URL_ENV,
    TelegramBotSetupResult,
    TelegramPairing,
    create_pairing,
    generate_bot_username,
    generate_deep_link,
    generate_pairing_nonce,
    poll_for_setup_result,
    poll_for_token,
    print_qr_code,
    render_qr_terminal,
)


VALID_TOKEN = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef"
SECOND_VALID_TOKEN = "987654321:abcdefghijklmnopqrstuvwxyzABCDEF"


class TestGenerateBotUsername:
    def test_secure_default_format(self):
        name = generate_bot_username()
        assert name.startswith("ht_")
        assert name.endswith("_bot")
        assert len(name) == len("ht_") + 16 + len("_bot")
        assert len(name) <= 32

    def test_profile_name_not_embedded(self):
        name = generate_bot_username("work")
        assert "work" not in name
        assert name.startswith("ht_")
        assert name.endswith("_bot")

    def test_slug_uses_telegram_safe_base32_chars(self):
        name = generate_bot_username()
        slug = name.removeprefix("ht_").removesuffix("_bot")
        assert len(slug) == 16
        assert set(slug) <= set("abcdefghijklmnopqrstuvwxyz234567")

    def test_uniqueness(self):
        names = {generate_bot_username() for _ in range(20)}
        assert len(names) == 20


class TestGenerateDeepLink:
    def test_basic_format(self):
        link = generate_deep_link(
            manager_bot="TestBot",
            suggested_username="my_bot",
        )
        assert link == "https://t.me/newbot/TestBot/my_bot"

    def test_with_name(self):
        link = generate_deep_link(
            manager_bot="@TestBot",
            suggested_username="my_bot",
            suggested_name="My Agent",
        )
        assert "https://t.me/newbot/TestBot/my_bot?" in link
        assert "name=My+Agent" in link

    def test_defaults(self):
        link = generate_deep_link()
        assert f"https://t.me/newbot/{DEFAULT_MANAGER_BOT}/" in link
        assert "ht_" in link

    def test_name_url_encoded(self):
        link = generate_deep_link(
            manager_bot="Bot",
            suggested_username="test_bot",
            suggested_name="Hermes & Friends",
        )
        assert "Hermes+%26+Friends" in link


class TestPairingNonce:
    def test_length(self):
        nonce = generate_pairing_nonce()
        assert len(nonce) == 32

    def test_hex_chars(self):
        nonce = generate_pairing_nonce()
        assert all(c in "0123456789abcdef" for c in nonce)

    def test_uniqueness(self):
        nonces = {generate_pairing_nonce() for _ in range(100)}
        assert len(nonces) == 100


class TestQRCode:
    def test_render_returns_string(self):
        result = render_qr_terminal("https://example.com")
        if result:
            assert isinstance(result, str)
            assert len(result) > 10

    def test_render_graceful_without_qrcode(self):
        with patch.dict("sys.modules", {"qrcode": None}):
            render_qr_terminal("https://example.com")

    def test_print_qr_code_with_url(self, capsys):
        print_qr_code("https://t.me/newbot/Bot/test_bot")
        captured = capsys.readouterr()
        assert "https://t.me/newbot/Bot/test_bot" in captured.out


class TestCreatePairing:
    def test_success(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {
            "pairing_id": "abcdefghijklmnop",
            "poll_token": "secret-token",
            "suggested_username": "hermes_abcdefghijklmnop_bot",
            "deep_link": "https://t.me/newbot/HermesSetupBot/hermes_abcdefghijklmnop_bot?name=Hermes+Agent",
            "qr_payload": "https://t.me/newbot/HermesSetupBot/hermes_abcdefghijklmnop_bot?name=Hermes+Agent",
            "expires_at": "2026-05-18T00:00:00.000Z",
        }

        with patch(
            "hermes_cli.telegram_managed_bot.httpx.post", return_value=mock_resp
        ) as post:
            pairing = create_pairing("https://api.example.com", bot_name="Hermes Agent")

        assert pairing == TelegramPairing(
            pairing_id="abcdefghijklmnop",
            poll_token="secret-token",
            suggested_username="hermes_abcdefghijklmnop_bot",
            deep_link="https://t.me/newbot/HermesSetupBot/hermes_abcdefghijklmnop_bot?name=Hermes+Agent",
            qr_payload="https://t.me/newbot/HermesSetupBot/hermes_abcdefghijklmnop_bot?name=Hermes+Agent",
            expires_at="2026-05-18T00:00:00.000Z",
        )
        post.assert_called_once_with(
            "https://api.example.com/v1/telegram/pairings",
            json={"bot_name": "Hermes Agent"},
            timeout=10.0,
        )

    def test_failure_status(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        with patch(
            "hermes_cli.telegram_managed_bot.httpx.post", return_value=mock_resp
        ):
            assert create_pairing("https://api.example.com") is None

    def test_invalid_payload(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {"pairing_id": "missing-poll-token"}
        with patch(
            "hermes_cli.telegram_managed_bot.httpx.post", return_value=mock_resp
        ):
            assert create_pairing("https://api.example.com") is None

    def test_uses_env_override(self, monkeypatch):
        monkeypatch.setenv(TELEGRAM_ONBOARDING_URL_ENV, "https://worker.example")
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        with patch(
            "hermes_cli.telegram_managed_bot.httpx.post", return_value=mock_resp
        ) as post:
            create_pairing()
        assert post.call_args.args[0] == "https://worker.example/v1/telegram/pairings"


class TestPollForToken:
    def pairing(self):
        return TelegramPairing(
            pairing_id="abcdefghijklmnop",
            poll_token="secret-token",
            suggested_username="hermes_abcdefghijklmnop_bot",
            deep_link="https://t.me/newbot/HermesSetupBot/hermes_abcdefghijklmnop_bot",
            qr_payload="https://t.me/newbot/HermesSetupBot/hermes_abcdefghijklmnop_bot",
        )

    def test_immediate_success(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "bot_username": "hermes_abcdefghijklmnop_bot",
            "owner_user_id": 42,
            "status": "ready",
            "token": VALID_TOKEN,
        }

        with patch(
            "hermes_cli.telegram_managed_bot.httpx.get", return_value=mock_resp
        ) as get:
            with patch("hermes_cli.telegram_managed_bot.time.sleep"):
                token = poll_for_token(
                    "https://api.example.com", self.pairing(), timeout=5
                )

        assert token == VALID_TOKEN
        assert (
            get.call_args.args[0]
            == "https://api.example.com/v1/telegram/pairings/abcdefghijklmnop"
        )
        assert get.call_args.kwargs["headers"] == {
            "Authorization": "Bearer secret-token"
        }

    def test_setup_result_includes_owner_user_id(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "bot_username": "hermes_abcdefghijklmnop_bot",
            "owner_user_id": 42,
            "status": "ready",
            "token": VALID_TOKEN,
        }

        with patch("hermes_cli.telegram_managed_bot.httpx.get", return_value=mock_resp):
            with patch("hermes_cli.telegram_managed_bot.time.sleep"):
                result = poll_for_setup_result(
                    "https://api.example.com", self.pairing(), timeout=5
                )

        assert result == TelegramBotSetupResult(
            token=VALID_TOKEN,
            bot_username="hermes_abcdefghijklmnop_bot",
            owner_user_id=42,
        )

    def test_setup_result_accepts_string_owner_user_id(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "bot_username": "hermes_abcdefghijklmnop_bot",
            "owner_user_id": "42",
            "status": "ready",
            "token": VALID_TOKEN,
        }

        with patch("hermes_cli.telegram_managed_bot.httpx.get", return_value=mock_resp):
            result = poll_for_setup_result(
                "https://api.example.com", self.pairing(), timeout=5
            )

        assert result == TelegramBotSetupResult(
            token=VALID_TOKEN,
            bot_username="hermes_abcdefghijklmnop_bot",
            owner_user_id=42,
        )

    def test_invalid_ready_token_returns_none(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "bot_username": "hermes_abcdefghijklmnop_bot",
            "owner_user_id": 42,
            "status": "ready",
            "token": "not-a-real-token",
        }

        with patch("hermes_cli.telegram_managed_bot.httpx.get", return_value=mock_resp):
            with patch("hermes_cli.telegram_managed_bot.time.sleep"):
                with patch(
                    "hermes_cli.telegram_managed_bot.time.monotonic"
                ) as mock_time:
                    mock_time.side_effect = [0, 0, 999]
                    assert (
                        poll_for_token(
                            "https://api.example.com", self.pairing(), timeout=1
                        )
                        is None
                    )

    def test_timeout_returns_none(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "waiting"}

        with patch("hermes_cli.telegram_managed_bot.httpx.get", return_value=mock_resp):
            with patch("hermes_cli.telegram_managed_bot.time.sleep"):
                with patch(
                    "hermes_cli.telegram_managed_bot.time.monotonic"
                ) as mock_time:
                    mock_time.side_effect = [0, 0, 999]
                    token = poll_for_token(
                        "https://api.example.com", self.pairing(), timeout=1
                    )
                    assert token is None

    def test_eventual_success(self):
        not_ready = MagicMock()
        not_ready.status_code = 200
        not_ready.json.return_value = {"status": "waiting"}

        ready = MagicMock()
        ready.status_code = 200
        ready.json.return_value = {"status": "ready", "token": SECOND_VALID_TOKEN}

        call_count = 0

        def fake_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return not_ready
            return ready

        with patch("hermes_cli.telegram_managed_bot.httpx.get", side_effect=fake_get):
            with patch("hermes_cli.telegram_managed_bot.time.sleep"):
                token = poll_for_token(
                    "https://api.example.com", self.pairing(), timeout=30
                )
                assert token == SECOND_VALID_TOKEN


class TestSetupTelegramAuto:
    def test_setup_helper_exists(self):
        from hermes_cli.setup import _setup_telegram_auto

        assert callable(_setup_telegram_auto)

    def test_setup_result_passes_profile_name_for_profile_home(self, monkeypatch, tmp_path):
        from hermes_cli import setup

        seen = {}
        profile_home = tmp_path / ".hermes" / "profiles" / "oracle"
        profile_home.mkdir(parents=True)

        monkeypatch.setattr(setup, "get_hermes_home", lambda: profile_home)

        def fake_auto_setup_telegram_bot_result(*, profile_name=None):
            seen["profile_name"] = profile_name
            return None

        monkeypatch.setattr(
            "hermes_cli.telegram_managed_bot.auto_setup_telegram_bot_result",
            fake_auto_setup_telegram_bot_result,
        )

        assert setup._setup_telegram_auto_result() is None
        assert seen["profile_name"] == "oracle"

    def test_profile_name_from_home_path_handles_windows_separators(self):
        from hermes_cli.setup import _profile_name_from_hermes_home

        assert (
            _profile_name_from_hermes_home(
                PureWindowsPath(r"C:\Users\test\AppData\Local\hermes\profiles\oracle")
            )
            == "oracle"
        )
