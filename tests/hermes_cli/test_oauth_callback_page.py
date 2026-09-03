"""Behavior tests for the shared OAuth callback page renderer.

The page is the last thing a user sees after authorizing, served by loopback
listeners that shut down the moment the redirect lands. That constrains it in
ways worth pinning: it must be self-contained (no network fetch for the logo),
it must escape provider-supplied text, and it must degrade to a logo-less page
rather than raising when the branding directory is missing or unusable.
"""

import base64

import pytest

# The stylesheet always defines the .has-dark-logo rules; only the body tag
# tells you whether a dark variant was actually found. Assert on the tag.
_DARK_BODY = '<body class="has-dark-logo">'
_PLAIN_BODY = "<body>"

from hermes_cli.oauth_callback_page import (
    _MAX_LOGO_BYTES,
    render_callback_page,
    render_callback_page_bytes,
)

# Smallest thing a browser will accept as a PNG; contents are irrelevant here.
_PNG_BYTES = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH"
    b"/q842iQAAAABJRU5ErkJggg=="
)


@pytest.fixture
def branding_dir(tmp_path, monkeypatch):
    """Point HERMES_HOME at a temp dir and return its (uncreated) branding dir."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home / "branding"


def test_page_is_a_self_contained_document():
    page = render_callback_page("Authorization received", "You can close this tab.")
    assert page.startswith("<!doctype html>")
    assert "Authorization received" in page
    assert "You can close this tab." in page
    # Self-contained: no external stylesheet, script, or image to fetch. The
    # listener is gone by the time the browser would ask for one.
    assert "<style>" in page
    assert 'src="http' not in page
    assert "<link" not in page


def test_heading_doubles_as_the_tab_title():
    page = render_callback_page("Authorization received", "Done.")
    assert "<title>Authorization received</title>" in page


@pytest.mark.parametrize("status", ["success", "error", "pending"])
def test_status_selects_a_distinct_glyph_class(status):
    page = render_callback_page("Heading", "Message", status=status)
    assert f'class="glyph glyph-{status}"' in page


def test_provider_supplied_text_is_escaped():
    """A hostile ?error= value must not become markup.

    tools/mcp_oauth.py used to interpolate the raw parameter into the page.
    """
    page = render_callback_page(
        "<script>alert(1)</script>",
        '<img src=x onerror="alert(2)">',
        status="error",
    )
    assert "<script>alert(1)</script>" not in page
    assert "<img src=x" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_auto_close_is_opt_in():
    assert "window.close()" not in render_callback_page("H", "M")
    assert "window.close()" in render_callback_page("H", "M", auto_close=True)


def test_bytes_helper_matches_the_string_helper():
    assert render_callback_page_bytes("H", "M") == render_callback_page("H", "M").encode(
        "utf-8"
    )


def test_no_logo_markup_when_branding_dir_is_absent(branding_dir):
    assert not branding_dir.exists()
    page = render_callback_page("Authorization received", "Done.")
    assert "<img" not in page
    assert _PLAIN_BODY in page


def test_light_logo_is_inlined_as_a_data_uri(branding_dir):
    branding_dir.mkdir()
    (branding_dir / "oauth-logo.png").write_bytes(_PNG_BYTES)

    page = render_callback_page("Authorization received", "Done.")
    expected = base64.b64encode(_PNG_BYTES).decode("ascii")
    assert f'src="data:image/png;base64,{expected}"' in page
    # With no dark variant supplied, the single logo serves both schemes.
    assert _PLAIN_BODY in page
    assert page.count("<img") == 1


def test_dark_variant_adds_a_second_scheme_specific_logo(branding_dir):
    branding_dir.mkdir()
    (branding_dir / "oauth-logo.png").write_bytes(_PNG_BYTES)
    (branding_dir / "oauth-logo-dark.png").write_bytes(_PNG_BYTES)

    page = render_callback_page("Authorization received", "Done.")
    assert _DARK_BODY in page
    assert 'class="logo logo-light"' in page
    assert 'class="logo logo-dark"' in page


def test_svg_branding_gets_the_right_media_type(branding_dir):
    branding_dir.mkdir()
    (branding_dir / "oauth-logo.svg").write_bytes(b"<svg xmlns='http://www.w3.org/2000/svg'/>")

    assert "data:image/svg+xml;base64," in render_callback_page("H", "M")


def test_unsupported_extension_is_ignored(branding_dir):
    branding_dir.mkdir()
    (branding_dir / "oauth-logo.bmp").write_bytes(b"BM not inlined")

    assert "<img" not in render_callback_page("H", "M")


def test_oversized_logo_is_skipped_rather_than_inlined(branding_dir):
    """A stray large asset would be base64'd into every callback response."""
    branding_dir.mkdir()
    (branding_dir / "oauth-logo.png").write_bytes(b"\x89PNG" + b"\0" * _MAX_LOGO_BYTES)

    assert "<img" not in render_callback_page("H", "M")


def test_empty_logo_file_is_skipped(branding_dir):
    branding_dir.mkdir()
    (branding_dir / "oauth-logo.png").write_bytes(b"")

    assert "<img" not in render_callback_page("H", "M")


def test_unreadable_logo_does_not_break_the_callback(branding_dir, monkeypatch):
    """The user is mid-OAuth; a bad branding file must not fail the redirect."""
    branding_dir.mkdir()
    logo = branding_dir / "oauth-logo.png"
    logo.write_bytes(_PNG_BYTES)

    real_read_bytes = type(logo).read_bytes

    def exploding_read_bytes(self):
        if self.name.startswith("oauth-logo"):
            raise OSError("permission denied")
        return real_read_bytes(self)

    monkeypatch.setattr(type(logo), "read_bytes", exploding_read_bytes)

    page = render_callback_page("Authorization received", "Done.")
    assert "Authorization received" in page
    assert "<img" not in page


def test_branding_is_resolved_per_call_not_cached_process_wide(branding_dir):
    """One gateway process serves many profiles; caching would pin the first."""
    assert "<img" not in render_callback_page("H", "M")

    branding_dir.mkdir()
    (branding_dir / "oauth-logo.png").write_bytes(_PNG_BYTES)

    assert "<img" in render_callback_page("H", "M")
