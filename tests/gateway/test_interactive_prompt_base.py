"""Tests for the shared interactive-prompt formatting cores in BasePlatformAdapter.

Covers ``_format_exec_approval`` (template-attr driven exec-approval text),
``_format_choice_page`` (picker pagination core), ``_truncate_preview``, and
byte-parity of the rewired adapters (telegram/feishu/matrix) against their
historical inline formatting.
"""

import html as _html

from gateway.platforms.base import BasePlatformAdapter


def _bare(cls):
    """Bare instance without running __init__ (documented test pattern)."""
    return object.__new__(cls)


class _DefaultAdapter(BasePlatformAdapter):
    """Concrete subclass using only base-class template attrs."""

    async def connect(self):  # pragma: no cover - not used
        pass

    async def disconnect(self):  # pragma: no cover - not used
        pass

    async def get_chat_info(self, chat_id):  # pragma: no cover - not used
        return {}

    async def send(self, *a, **k):  # pragma: no cover - not used
        raise NotImplementedError


class TestTruncatePreview:
    def test_short_text_unchanged(self):
        assert BasePlatformAdapter._truncate_preview("abc", 10) == "abc"

    def test_exact_budget_unchanged(self):
        assert BasePlatformAdapter._truncate_preview("x" * 10, 10) == "x" * 10

    def test_ea_fit_measures_the_escaped_rendering(self):
        """An escaping platform's budget applies to the wire payload, never to raw chars."""
        class _Html(_DefaultAdapter):
            def _ea_escape(self, text):
                return _html.escape(text)

        adapter = _bare(_Html)
        assert adapter._ea_fit("&" * 100, 20) == "&&&&..."  # 4 x "&amp;" == 20 escaped chars
        assert adapter._ea_fit("&" * 4, 20) == "&" * 4  # fits once escaped → untouched


class TestFormatExecApproval:

    def test_deadline_line_tracks_configured_timeout(self, monkeypatch):
        """The card must say how long the user has and that silence means NO — for any timeout."""
        monkeypatch.setattr("gateway.platforms.base.approval_timeout_seconds", lambda: 90)
        text = _bare(_DefaultAdapter)._format_exec_approval("rm -rf /", "scary")
        assert "within 90 seconds it will NOT run" in text
        monkeypatch.setattr("gateway.platforms.base.approval_timeout_seconds", lambda: 7200)
        text = _bare(_DefaultAdapter)._format_exec_approval("rm -rf /", "scary")
        assert "within 2 hours it will NOT run" in text


    def test_escape_hook_applied_to_command_and_reason(self):
        class Escaping(_DefaultAdapter):
            def _ea_escape(self, text: str) -> str:
                return _html.escape(text)

        ad = _bare(Escaping)
        text = ad._format_exec_approval("echo <hi>", "a & b")
        assert "echo &lt;hi&gt;" in text
        assert "a &amp; b" in text


class TestFormatChoicePage:
    def test_single_page_no_page_info(self):
        opts, meta = BasePlatformAdapter._format_choice_page([1, 2, 3], 0, 10)
        assert opts == [1, 2, 3]
        assert meta["page_info"] == ""
        assert meta["total_pages"] == 1
        assert meta["page"] == 0


    def test_page_clamped_high(self):
        opts, meta = BasePlatformAdapter._format_choice_page(list(range(25)), 99, 10)
        assert meta["page"] == 2
        assert opts == list(range(20, 25))
        assert meta["page_info"] == " (21–25 of 25)"


