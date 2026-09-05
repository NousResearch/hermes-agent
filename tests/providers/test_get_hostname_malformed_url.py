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
