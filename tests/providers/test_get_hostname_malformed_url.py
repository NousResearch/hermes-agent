"""ProviderProfile.get_hostname() must fail closed on a malformed base_url.

Sibling of issue #87219 / PR #87220 (which fixed the shared
``utils.base_url_hostname`` helper): ``get_hostname`` derives a hostname from
``base_url`` for URL-based provider detection and its contract returns ``""``
when no hostname is available, but a malformed bracketed IPv6 URL makes
``urllib.parse`` raise ``ValueError: Invalid IPv6 URL`` instead. Because this
runs during provider/URL classification, the exception could abort setup on a
bad custom endpoint before normal validation. It must return ``""`` instead.
"""

from providers.base import ProviderProfile


class TestGetHostnameMalformedUrl:
    def test_malformed_base_url_returns_empty(self):
        # Unmatched IPv6 bracket: urlparse raises "ValueError: Invalid IPv6 URL".
        profile = ProviderProfile(name="custom", base_url="http://[::1")
        assert profile.get_hostname() == ""

    def test_valid_base_url_still_resolves(self):
        profile = ProviderProfile(
            name="custom", base_url="https://api.gmi-serving.com/v1"
        )
        assert profile.get_hostname() == "api.gmi-serving.com"

    def test_explicit_hostname_short_circuits_malformed_base_url(self):
        profile = ProviderProfile(
            name="custom", hostname="api.example.com", base_url="http://[::1"
        )
        assert profile.get_hostname() == "api.example.com"

    def test_non_string_base_url_returns_empty(self):
        # A mis-typed config value is truthy but not a string (here an int,
        # on which urlparse would raise AttributeError). The isinstance guard
        # fails closed for any non-string rather than relying on the exact
        # exception urlparse happens to raise for that type.
        profile = ProviderProfile(name="custom", base_url=12345)  # type: ignore[arg-type]
        assert profile.get_hostname() == ""

    def test_malformed_base_url_does_not_abort_detection(self):
        # The stated motivation: an empty hostname must let URL-based provider
        # detection proceed, not raise. Callers compare the return against a
        # host suffix; an empty string simply fails to match, never aborts.
        profile = ProviderProfile(name="custom", base_url="http://[::1")
        hostname = profile.get_hostname()
        assert hostname == ""
        assert not hostname.endswith("gmi-serving.com")
