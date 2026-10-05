"""The Google Chat OAuth redirect has to fail in the browser, but only in one specific way.

This flow deliberately points Google at a loopback address nothing is listening on, so the
browser's failed navigation leaves ?code= in the address bar for the user to copy. A port on the
browsers' blocked list fails differently and silently: Chrome and Firefox refuse to navigate at
all, the consent screen hangs after "Allow", and the user never sees a code to paste. Port 1
(tcpmux) shipped here for a while and broke the flow that way.

The assertion guards the whole blocked list rather than the one value, because any future edit
picking another "surely unused" low port reintroduces the same hang.
"""

from __future__ import annotations

from urllib.parse import urlparse

from plugins.platforms.google_chat import oauth

# Chromium's list, net/base/port_util.cc (kRestrictedPorts); Firefox's is a near-identical set.
BLOCKED_PORTS = frozenset({
    1, 7, 9, 11, 13, 15, 17, 19, 20, 21, 22, 23, 37, 42, 43, 53, 69, 77, 79, 87, 95, 101, 102,
    103, 104, 109, 110, 111, 113, 115, 117, 119, 123, 135, 137, 139, 143, 161, 179, 389, 427, 465,
    512, 513, 514, 515, 526, 530, 531, 532, 540, 548, 554, 556, 563, 587, 601, 636, 989, 990, 993,
    995, 1719, 1720, 1723, 2049, 3659, 4045, 4190, 5060, 5061, 6000, 6566, 6665, 6666, 6667, 6668,
    6669, 6679, 6697, 10080,
})


def test_redirect_uri_is_loopback_on_a_port_browsers_will_navigate_to():
    parsed = urlparse(oauth._REDIRECT_URI)

    assert parsed.scheme == "http"
    assert parsed.hostname in {"localhost", "127.0.0.1"}
    assert parsed.port is not None, "an explicit port keeps the redirect off port 80"
    assert parsed.port not in BLOCKED_PORTS, (
        f"port {parsed.port} is on the browsers' blocked-port list; the consent screen will hang "
        "instead of redirecting, and the user will never see the authorization code"
    )
