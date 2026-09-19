"""The sign-in pages declare the dashboard favicon.

Behind an auth gate the browser reaches ``/login`` BEFORE it ever sees the
SPA's ``index.html``, so the tab's icon impression is formed on a page the
SPA never rendered. When these templates carried no ``<link rel="icon">``
the tab fell back to a blank/generic icon even though ``/favicon.ico`` is
served fine (and is a gate-public path).

Contract, not a snapshot: every HTML document these renderers emit points at
an icon path the auth gate lets through unauthenticated.
"""

from __future__ import annotations

import re

import pytest

from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth.login_page import (
    render_login_html,
    render_native_provider_choice_html,
)
from hermes_cli.dashboard_auth.middleware import _path_is_public
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

_ICON_LINK = re.compile(r"""<link[^>]*rel=["']icon["'][^>]*>""", re.IGNORECASE)
_HREF = re.compile(r"""href=["']([^"']+)["']""", re.IGNORECASE)


@pytest.fixture
def stub_provider():
    clear_providers()
    register_provider(StubAuthProvider())
    yield
    clear_providers()


def _icon_href(html: str) -> str:
    link = _ICON_LINK.search(html)
    assert link, f"no <link rel=icon> in rendered page: {html[:200]!r}"
    href = _HREF.search(link.group(0))
    assert href, f"icon link has no href: {link.group(0)!r}"
    return href.group(1)


def test_login_page_icon_is_reachable_through_the_gate(stub_provider):
    """The sign-in page's icon must be a path the gate serves unauthenticated.

    An icon the gate redirects to ``/login`` is worse than no icon: the browser
    caches an HTML 302 as the tab's image source.
    """
    href = _icon_href(render_login_html())
    assert _path_is_public(href), f"{href} is not gate-public — the icon would 302 to /login"


def test_every_login_document_declares_the_same_icon():
    """All three documents these renderers emit agree on one icon.

    ``render_login_html`` falls back to the 'sign-in unavailable' page when no
    provider is registered, and the native flow renders a provider picker — a
    user can land on any of them, so none may be the one that drops the icon.
    """
    clear_providers()
    unavailable = render_login_html()

    register_provider(StubAuthProvider())
    try:
        sign_in = render_login_html()
        picker = render_native_provider_choice_html(
            providers=[StubAuthProvider()],
            authorize_path="/auth/native/authorize",
            code_challenge="c",
            code_challenge_method="S256",
            redirect_uri="http://127.0.0.1:1/cb",
            state="s",
        )
    finally:
        clear_providers()

    assert _icon_href(unavailable) == _icon_href(sign_in) == _icon_href(picker)
