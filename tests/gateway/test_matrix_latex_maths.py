"""Outbound ``$...$`` / ``$$...$$`` become Element ``data-mx-maths`` markup (#106458)."""

from gateway.config import PlatformConfig
from plugins.platforms.matrix.adapter import MatrixAdapter


def _adapter() -> MatrixAdapter:
    return MatrixAdapter(PlatformConfig(
        enabled=True, token="syt_test_token",
        extra={"homeserver": "https://matrix.example.org", "user_id": "@bot:example.org"}))


def test_inline_and_display_math_survive_sanitizer_with_escaped_tex():
    content = _adapter()._build_text_message_content("Wave: $a<b$ and $$\\hat{H}\\Psi = E\\Psi$$")
    html = content["formatted_body"]
    assert '<span data-mx-maths="a&lt;b">a&lt;b</span>' in html
    assert '<div data-mx-maths="\\hat{H}\\Psi = E\\Psi">\\hat{H}\\Psi = E\\Psi</div>' in html
    assert "$" not in html
    # Plain-text fallback keeps the raw TeX for clients without the Labs feature.
    assert content["body"] == "Wave: $a<b$ and $$\\hat{H}\\Psi = E\\Psi$$"


def test_unpaired_dollars_and_sentinel_collisions_pass_through_unchanged():
    text = "Costs $5 or $10 today; HERMESTEXINLINE7HERMESTEXEND is not ours"
    assert _adapter()._markdown_to_html(text) == text
