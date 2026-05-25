"""Tests for gateway.platforms.qqbot.constants — package-level constants."""

from __future__ import annotations

from gateway.platforms.qqbot.constants import (
    API_BASE,
    CONNECT_TIMEOUT_SECONDS,
    DEDUP_MAX_SIZE,
    DEDUP_WINDOW_SECONDS,
    DEFAULT_API_TIMEOUT,
    FILE_UPLOAD_TIMEOUT,
    GATEWAY_URL_PATH,
    MAX_MESSAGE_LENGTH,
    MAX_QUICK_DISCONNECT_COUNT,
    MAX_RECONNECT_ATTEMPTS,
    MEDIA_TYPE_FILE,
    MEDIA_TYPE_IMAGE,
    MEDIA_TYPE_VIDEO,
    MEDIA_TYPE_VOICE,
    MSG_TYPE_INPUT_NOTIFY,
    MSG_TYPE_MARKDOWN,
    MSG_TYPE_MEDIA,
    MSG_TYPE_TEXT,
    ONBOARD_API_TIMEOUT,
    ONBOARD_CREATE_PATH,
    ONBOARD_POLL_INTERVAL,
    ONBOARD_POLL_PATH,
    PORTAL_HOST,
    QR_URL_TEMPLATE,
    QQBOT_VERSION,
    QUICK_DISCONNECT_THRESHOLD,
    RATE_LIMIT_DELAY,
    RECONNECT_BACKOFF,
    TOKEN_URL,
)


class TestVersion:
    def test_version_is_string(self):
        assert isinstance(QQBOT_VERSION, str)

    def test_version_is_semver(self):
        parts = QQBOT_VERSION.split(".")
        assert len(parts) == 3


class TestEndpoints:
    def test_portal_host_is_string(self):
        assert isinstance(PORTAL_HOST, str)
        assert len(PORTAL_HOST) > 0

    def test_api_base_is_https(self):
        assert API_BASE.startswith("https://")

    def test_token_url_is_https(self):
        assert TOKEN_URL.startswith("https://")

    def test_gateway_path_starts_with_slash(self):
        assert GATEWAY_URL_PATH.startswith("/")

    def test_onboard_paths(self):
        assert ONBOARD_CREATE_PATH.startswith("/")
        assert ONBOARD_POLL_PATH.startswith("/")

    def test_qr_template_has_placeholders(self):
        assert "{task_id}" in QR_URL_TEMPLATE


class TestTimeouts:
    def test_timeouts_are_positive(self):
        assert DEFAULT_API_TIMEOUT > 0
        assert FILE_UPLOAD_TIMEOUT > 0
        assert CONNECT_TIMEOUT_SECONDS > 0

    def test_timeouts_are_numeric(self):
        assert isinstance(DEFAULT_API_TIMEOUT, (int, float))
        assert isinstance(FILE_UPLOAD_TIMEOUT, (int, float))

    def test_file_upload_longer_than_default(self):
        assert FILE_UPLOAD_TIMEOUT > DEFAULT_API_TIMEOUT

    def test_onboard_timeouts(self):
        assert ONBOARD_POLL_INTERVAL > 0
        assert ONBOARD_API_TIMEOUT > 0


class TestReconnect:
    def test_backoff_is_list_of_ints(self):
        assert isinstance(RECONNECT_BACKOFF, list)
        assert all(isinstance(v, int) for v in RECONNECT_BACKOFF)

    def test_backoff_is_ascending(self):
        for i in range(1, len(RECONNECT_BACKOFF)):
            assert RECONNECT_BACKOFF[i] >= RECONNECT_BACKOFF[i - 1]

    def test_max_reconnect_is_positive(self):
        assert MAX_RECONNECT_ATTEMPTS > 0

    def test_rate_limit_delay_is_positive(self):
        assert RATE_LIMIT_DELAY > 0

    def test_quick_disconnect_values(self):
        assert QUICK_DISCONNECT_THRESHOLD > 0
        assert MAX_QUICK_DISCONNECT_COUNT > 0


class TestMessageLimits:
    def test_max_message_length_is_positive(self):
        assert MAX_MESSAGE_LENGTH > 0

    def test_dedup_window_positive(self):
        assert DEDUP_WINDOW_SECONDS > 0

    def test_dedup_max_size_positive(self):
        assert DEDUP_MAX_SIZE > 0


class TestMessageTypes:
    def test_types_are_ints(self):
        assert isinstance(MSG_TYPE_TEXT, int)
        assert isinstance(MSG_TYPE_MARKDOWN, int)
        assert isinstance(MSG_TYPE_MEDIA, int)
        assert isinstance(MSG_TYPE_INPUT_NOTIFY, int)

    def test_types_are_distinct(self):
        types = [MSG_TYPE_TEXT, MSG_TYPE_MARKDOWN, MSG_TYPE_MEDIA, MSG_TYPE_INPUT_NOTIFY]
        assert len(types) == len(set(types))


class TestMediaTypes:
    def test_media_types_are_ints(self):
        assert isinstance(MEDIA_TYPE_IMAGE, int)
        assert isinstance(MEDIA_TYPE_VIDEO, int)
        assert isinstance(MEDIA_TYPE_VOICE, int)
        assert isinstance(MEDIA_TYPE_FILE, int)

    def test_media_types_are_distinct(self):
        types = [MEDIA_TYPE_IMAGE, MEDIA_TYPE_VIDEO, MEDIA_TYPE_VOICE, MEDIA_TYPE_FILE]
        assert len(types) == len(set(types))
