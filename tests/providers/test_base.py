"""Tests for providers/base.py — covering the base ProviderProfile class and helpers."""

from unittest import mock

from providers.base import (
    OMIT_TEMPERATURE,
    ProviderProfile,
    _profile_user_agent,
)


class TestProfileUserAgent:
    """Coverage for _profile_user_agent() — lines 31-35."""

    def test_success_path_returns_versioned_ua(self):
        """When hermes_cli.__version__ is importable, returns hermes-cli/<version>."""
        ua = _profile_user_agent()
        assert ua.startswith("hermes-cli/")

    def test_exception_path_returns_bare_cli(self):
        """When import fails, returns bare 'hermes-cli'."""
        import builtins
        import sys

        original_import = builtins.__import__
        removed = {}
        for key in list(sys.modules):
            if "hermes_cli" in key:
                removed[key] = sys.modules.pop(key)
        try:
            with mock.patch("builtins.__import__") as mock_import:

                def side_effect(name, *args, **kwargs):
                    if name == "hermes_cli":
                        raise ImportError("mocked")
                    return original_import(name, *args, **kwargs)

                mock_import.side_effect = side_effect
                result = _profile_user_agent()
            assert result == "hermes-cli"
        finally:
            sys.modules.update(removed)


class TestGetHostname:
    """Coverage for ProviderProfile.get_hostname() — lines 98-109."""

    def test_returns_explicit_hostname(self):
        p = ProviderProfile(name="test", hostname="api.example.com")
        assert p.get_hostname() == "api.example.com"

    def test_derives_from_base_url(self):
        p = ProviderProfile(name="test", base_url="https://api.example.com/v1")
        assert p.get_hostname() == "api.example.com"

    def test_derives_from_base_url_without_scheme(self):
        p = ProviderProfile(name="test", base_url="http://192.168.1.1:8080/api")
        assert p.get_hostname() == "192.168.1.1"

    def test_fallback_empty(self):
        p = ProviderProfile(name="test")
        assert p.get_hostname() == ""


class TestFixedTemperature:
    """Coverage for fixed_temperature field — line 89."""

    def test_default_is_none(self):
        p = ProviderProfile(name="test")
        assert p.fixed_temperature is None

    def test_omit_temperature_sentinel(self):
        p = ProviderProfile(name="test", fixed_temperature=OMIT_TEMPERATURE)
        assert p.fixed_temperature is OMIT_TEMPERATURE


class TestGetMaxTokens:
    """Coverage for ProviderProfile.get_max_tokens() — lines 148-160."""

    def test_returns_default(self):
        p = ProviderProfile(name="test", default_max_tokens=4096)
        assert p.get_max_tokens(model=None) == 4096
        assert p.get_max_tokens(model="some-model") == 4096

    def test_default_is_none(self):
        p = ProviderProfile(name="test")
        assert p.get_max_tokens(model=None) is None

    def test_ignores_model_arg(self):
        """Base implementation ignores the model argument."""
        p = ProviderProfile(name="test", default_max_tokens=8192)
        assert p.get_max_tokens(model="gpt-4") == 8192
        assert p.get_max_tokens(model="claude-3") == 8192


class TestFetchModels:
    """Coverage for ProviderProfile.fetch_models() — lines 162-214."""

    def test_returns_none_when_no_urls(self):
        """Neither models_url nor base_url -> None."""
        p = ProviderProfile(name="test")
        assert p.fetch_models() is None

    def test_returns_none_with_empty_strings(self):
        """Empty models_url and base_url -> None."""
        p = ProviderProfile(name="test", models_url="", base_url="")
        assert p.fetch_models() is None

    @mock.patch("urllib.request.urlopen")
    def test_uses_models_url_directly(self, mock_urlopen):
        """models_url is used directly when set, ignoring base_url."""
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = b'{"data": [{"id": "m1"}]}'
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        p = ProviderProfile(
            name="test",
            models_url="https://custom.example.com/api/v1/models",
            base_url="https://api.example.com/v1",
        )
        result = p.fetch_models()
        assert result == ["m1"]

        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "https://custom.example.com/api/v1/models"

    @mock.patch("urllib.request.urlopen")
    def test_falls_back_to_base_url_slash_models(self, mock_urlopen):
        """Without models_url, constructs URL from base_url + '/models'."""
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = b'{"data": [{"id": "m1"}]}'
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        p = ProviderProfile(name="test", base_url="https://api.example.com/v1")
        result = p.fetch_models()
        assert result == ["m1"]

        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "https://api.example.com/v1/models"

    @mock.patch("urllib.request.urlopen")
    def test_base_url_trailing_slash_stripped(self, mock_urlopen):
        """Trailing slash on base_url is stripped before appending '/models'."""
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = b'{"data": [{"id": "m1"}]}'
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        p = ProviderProfile(name="test", base_url="https://api.example.com/v1/")
        p.fetch_models()

        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "https://api.example.com/v1/models"

    @mock.patch("urllib.request.urlopen")
    def test_sets_bearer_auth_when_api_key_given(self, mock_urlopen):
        """api_key is forwarded as Authorization Bearer header."""
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = b'{"data": [{"id": "m1"}]}'
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        p = ProviderProfile(name="test", base_url="https://api.example.com/v1")
        p.fetch_models(api_key="sk-abc")

        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Authorization") == "Bearer sk-abc"

    @mock.patch("urllib.request.urlopen")
    def test_no_auth_header_when_no_api_key(self, mock_urlopen):
        """Without api_key, no Authorization header is set."""
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = b'{"data": [{"id": "m1"}]}'
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        p = ProviderProfile(name="test", base_url="https://api.example.com/v1")
        p.fetch_models()

        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Authorization") is None

    @mock.patch("urllib.request.urlopen")
    def test_sets_user_agent_header(self, mock_urlopen):
        """Request includes the hermes-cli User-Agent header."""
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = b'{"data": [{"id": "m1"}]}'
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        p = ProviderProfile(name="test", base_url="https://api.example.com/v1")
        p.fetch_models()

        req = mock_urlopen.call_args[0][0]
        ua = req.get_header("User-agent")
        assert ua is not None
        assert ua.startswith("hermes-cli")

    @mock.patch("urllib.request.urlopen")
    def test_sets_accept_header(self, mock_urlopen):
        """Request includes Accept application/json header."""
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = b'{"data": [{"id": "m1"}]}'
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        p = ProviderProfile(name="test", base_url="https://api.example.com/v1")
        p.fetch_models()

        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Accept") == "application/json"

    @mock.patch("urllib.request.urlopen")
    def test_forwards_default_headers(self, mock_urlopen):
        """default_headers dict items are added to the request."""
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = b'{"data": [{"id": "m1"}]}'
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        p = ProviderProfile(
            name="test",
            base_url="https://api.example.com/v1",
            default_headers={"X-Custom": "value", "X-API-Version": "2"},
        )
        p.fetch_models()

        req = mock_urlopen.call_args[0][0]
        # urllib.request.Request normalises header names on add_header()
        # so X-Custom becomes X-custom. get_header() is case-sensitive.
        assert req.get_header("X-custom") == "value"
        assert req.get_header("X-api-version") == "2"

    @mock.patch("urllib.request.urlopen")
    def test_parses_list_response(self, mock_urlopen):
        """When response body is a JSON list, extracts model IDs directly."""
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = b'[{"id": "m1"}, {"id": "m2"}]'
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        p = ProviderProfile(name="test", base_url="https://api.example.com/v1")
        result = p.fetch_models()
        assert result == ["m1", "m2"]

    @mock.patch("urllib.request.urlopen")
    def test_parses_dict_response_with_data_key(self, mock_urlopen):
        """When response body is a dict, reads model IDs from the 'data' key."""
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = b'{"data": [{"id": "m1"}, {"id": "m2"}]}'
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        p = ProviderProfile(name="test", base_url="https://api.example.com/v1")
        result = p.fetch_models()
        assert result == ["m1", "m2"]

    @mock.patch("urllib.request.urlopen")
    def test_skips_items_without_id(self, mock_urlopen):
        """Dict items missing the 'id' key are filtered out of the result."""
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = (
            b'{"data": [{"id": "m1"}, {"name": "no-id"}, {"id": "m2"}]}'
        )
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        p = ProviderProfile(name="test", base_url="https://api.example.com/v1")
        result = p.fetch_models()
        assert result == ["m1", "m2"]

    @mock.patch("urllib.request.urlopen")
    def test_returns_none_on_network_error(self, mock_urlopen):
        """When urlopen raises, fetch_models returns None."""
        mock_urlopen.side_effect = OSError("connection failed")

        p = ProviderProfile(name="test", base_url="https://api.example.com/v1")
        result = p.fetch_models()
        assert result is None

    @mock.patch("urllib.request.urlopen")
    def test_returns_none_on_json_decode_error(self, mock_urlopen):
        """When response is not valid JSON, fetch_models returns None."""
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = b"not-json"
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        p = ProviderProfile(name="test", base_url="https://api.example.com/v1")
        result = p.fetch_models()
        assert result is None

    @mock.patch("urllib.request.urlopen")
    def test_returns_none_on_error(self, mock_urlopen):
        """HTTP errors cause fetch_models to return None."""
        mock_urlopen.side_effect = OSError("403 Forbidden")

        p = ProviderProfile(name="test", base_url="https://api.example.com/v1")
        result = p.fetch_models()
        assert result is None

    @mock.patch("urllib.request.urlopen")
    def test_timeout_passed_through(self, mock_urlopen):
        """Custom timeout is forwarded to urlopen."""
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = b'{"data": [{"id": "m1"}]}'
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        p = ProviderProfile(name="test", base_url="https://api.example.com/v1")
        p.fetch_models(api_key="key", timeout=15.0)

        mock_urlopen.assert_called_once()
        assert mock_urlopen.call_args[1].get("timeout") == 15.0

    @mock.patch("urllib.request.urlopen")
    def test_empty_default_headers_no_error(self, mock_urlopen):
        """Empty default_headers dict does not cause errors."""
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = b'{"data": [{"id": "m1"}]}'
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        p = ProviderProfile(
            name="test",
            base_url="https://api.example.com/v1",
            default_headers={},
        )
        result = p.fetch_models()
        assert result == ["m1"]
