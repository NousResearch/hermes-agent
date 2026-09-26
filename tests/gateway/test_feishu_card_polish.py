"""Tests for Feishu card body polish, the footer meta line, and the auto-card threshold.

These cover the helpers that render long/structured replies as OpenClaw-style
interactive cards:

* ``_optimize_card_markdown`` — body polish (heading downgrade, code-fence
  protection, ``img_xxx``-only image refs, blank-line collapse, removal of a
  duplicated trailing footer line the model may have written itself).
* ``_build_markdown_card_payload`` — card payload construction.
* ``_card_footer`` / ``_load_card_model`` — the ``⏱ <elapsed> · <model>`` meta line.
* ``_card_min_chars`` — ``FEISHU_CARD_MIN_CHARS`` override (0 disables auto cards).
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from unittest.mock import patch


def _adapter_module():
    """Import the adapter with a writable app-data root (Windows-CI compatible)."""
    with patch.dict(os.environ, {"LOCALAPPDATA": tempfile.gettempdir()}):
        from plugins.platforms.feishu import adapter

    return adapter


A = _adapter_module()


class _OutboundStub:
    """Minimal stand-in exposing only what the outbound helpers touch."""

    _chat_inbound_ts: dict = {}
    _card_model: str = "deepseek-v4-flash"
    _card_min_chars = A.FeishuAdapter._card_min_chars
    _card_footer = A.FeishuAdapter._card_footer


def _stub(**overrides) -> _OutboundStub:
    stub = _OutboundStub()
    stub._chat_inbound_ts = dict(overrides.pop("chat_inbound_ts", {}))
    stub._card_model = overrides.pop("card_model", "deepseek-v4-flash")
    return stub


# ---------------------------------------------------------------------------
# _optimize_card_markdown
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty():
    assert A._optimize_card_markdown("") == ""


def test_headings_are_downgraded_when_h1_to_h3_present():
    out = A._optimize_card_markdown("# Title\n\n## Section\n\n### Sub")
    assert "#### Title" in out
    assert "##### Section" in out
    assert "##### Sub" in out


def test_plain_text_without_headings_is_left_alone():
    text = "Just prose about #hashtags and C# code."
    assert A._optimize_card_markdown(text) == text


def test_fenced_code_content_is_never_rewritten():
    text = "# Heading\n\n```bash\n# a comment inside the fence\n## another\n```\n"
    out = A._optimize_card_markdown(text)
    assert "# a comment inside the fence" in out
    assert "## another" in out
    assert "#### Heading" in out


def test_only_img_keys_survive_image_refs():
    out = A._optimize_card_markdown("![a](https://example.com/x.png) ![b](img_v3_abc)")
    assert "https://example.com/x.png" not in out
    assert "img_v3_abc" in out


def test_blank_line_runs_are_collapsed():
    assert "\n\n\n" not in A._optimize_card_markdown("a\n\n\n\n\nb")


def test_duplicated_trailing_footer_line_is_stripped():
    for line in ("⏱ 12.3s", "deepseek-v4-flash · 8s", "deepseek/v4-flash:1 · 1m05s"):
        out = A._optimize_card_markdown(f"Body text here\n\n{line}")
        assert out.strip() == "Body text here", f"footer line survived: {line!r}"


def test_ordinary_trailing_line_is_kept():
    out = A._optimize_card_markdown("Body text\n\nSome concluding remark.")
    assert "Some concluding remark." in out


# ---------------------------------------------------------------------------
# _build_markdown_card_payload
# ---------------------------------------------------------------------------


def test_payload_shape_without_footer():
    card = json.loads(A._build_markdown_card_payload("# Title\n\nbody text"))
    assert card["config"]["wide_screen_mode"] is True
    assert len(card["elements"]) == 1
    assert card["elements"][0]["tag"] == "markdown"


def test_first_heading_becomes_header_and_leaves_the_body():
    card = json.loads(A._build_markdown_card_payload("# My Title\n\nbody text"))
    assert card["header"]["title"]["content"] == "My Title"
    body = card["elements"][0]["content"]
    assert "My Title" not in body
    assert "body text" in body


def test_footer_appends_hr_and_markdown_element():
    card = json.loads(
        A._build_markdown_card_payload("# T\n\nbody", footer="⏱ 3.0s · deepseek-v4-flash")
    )
    assert [e["tag"] for e in card["elements"]] == ["markdown", "hr", "markdown"]
    assert "deepseek-v4-flash" in card["elements"][-1]["content"]


# ---------------------------------------------------------------------------
# _card_footer
# ---------------------------------------------------------------------------


def test_footer_uses_seconds_below_one_minute():
    stub = _stub(chat_inbound_ts={"oc_1": time.time() - 12.4})
    footer = A.FeishuAdapter._card_footer(stub, "oc_1")
    assert footer.startswith("⏱ 12"), footer
    assert footer.endswith("deepseek-v4-flash"), footer


def test_footer_uses_minutes_above_one_minute():
    stub = _stub(chat_inbound_ts={"oc_1": time.time() - 125})
    footer = A.FeishuAdapter._card_footer(stub, "oc_1")
    assert "⏱ 2m05s" in footer, footer


def test_footer_without_inbound_timestamp_is_model_only():
    stub = _stub(chat_inbound_ts={})
    assert A.FeishuAdapter._card_footer(stub, "oc_unknown") == "deepseek-v4-flash"


def test_footer_without_model_is_elapsed_only():
    stub = _stub(chat_inbound_ts={"oc_1": time.time() - 5}, card_model="")
    footer = A.FeishuAdapter._card_footer(stub, "oc_1")
    assert footer.startswith("⏱ 5")
    assert "·" not in footer


# ---------------------------------------------------------------------------
# _card_min_chars (FEISHU_CARD_MIN_CHARS)
# ---------------------------------------------------------------------------


def test_threshold_defaults_to_disabled_when_unset():
    """Unset threshold = auto cards off, so existing behaviour is untouched."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("FEISHU_CARD_MIN_CHARS", None)
        assert A.FeishuAdapter._card_min_chars(_stub()) == 0


def test_threshold_can_be_raised():
    with patch.dict(os.environ, {"FEISHU_CARD_MIN_CHARS": "400"}):
        assert A.FeishuAdapter._card_min_chars(_stub()) == 400


def test_threshold_zero_disables_auto_cards():
    with patch.dict(os.environ, {"FEISHU_CARD_MIN_CHARS": "0"}):
        assert A.FeishuAdapter._card_min_chars(_stub()) == 0


def test_invalid_threshold_falls_back_to_disabled():
    with patch.dict(os.environ, {"FEISHU_CARD_MIN_CHARS": "not-a-number"}):
        assert A.FeishuAdapter._card_min_chars(_stub()) == 0


def test_negative_threshold_is_clamped_to_zero():
    with patch.dict(os.environ, {"FEISHU_CARD_MIN_CHARS": "-5"}):
        assert A.FeishuAdapter._card_min_chars(_stub()) == 0


# ---------------------------------------------------------------------------
# _build_outbound_payload — auto-card branch
# ---------------------------------------------------------------------------

_LONG_MARKDOWN = "## Heading\n\n" + ("some reasonably long body text " * 5)


def test_default_keeps_long_markdown_as_post():
    """No threshold configured → long replies keep the existing post path."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("FEISHU_CARD_MIN_CHARS", None)
        msg_type, _ = A.FeishuAdapter._build_outbound_payload(
            _stub(), _LONG_MARKDOWN, chat_id="oc_1",
        )
    assert msg_type == "post"


def test_long_markdown_becomes_an_interactive_card_when_enabled():
    with patch.dict(os.environ, {"FEISHU_CARD_MIN_CHARS": "80"}):
        msg_type, payload = A.FeishuAdapter._build_outbound_payload(
            _stub(), _LONG_MARKDOWN, chat_id="oc_1",
        )
    assert msg_type == "interactive"
    card = json.loads(payload)
    assert card["elements"][-1]["tag"] == "markdown"  # footer element present


def test_short_reply_stays_a_post_even_when_enabled():
    with patch.dict(os.environ, {"FEISHU_CARD_MIN_CHARS": "80"}):
        msg_type, _ = A.FeishuAdapter._build_outbound_payload(
            _stub(), "## Hi\n\nshort", chat_id="oc_1",
        )
    assert msg_type == "post"


def test_threshold_zero_keeps_long_replies_as_post():
    with patch.dict(os.environ, {"FEISHU_CARD_MIN_CHARS": "0"}):
        msg_type, _ = A.FeishuAdapter._build_outbound_payload(
            _stub(), _LONG_MARKDOWN, chat_id="oc_1",
        )
    assert msg_type == "post"


def test_force_post_never_produces_a_card():
    with patch.dict(os.environ, {"FEISHU_CARD_MIN_CHARS": "80"}):
        msg_type, _ = A.FeishuAdapter._build_outbound_payload(
            _stub(), _LONG_MARKDOWN, force_post=True, chat_id="oc_1",
        )
    assert msg_type == "post"
