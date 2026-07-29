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

    def test_over_budget_truncates_with_suffix(self):
        out = BasePlatformAdapter._truncate_preview("x" * 11, 10)
        assert out == "x" * 10 + "..."

    def test_none_coerced_to_empty(self):
        assert BasePlatformAdapter._truncate_preview(None, 10) == ""

    def test_custom_suffix(self):
        out = BasePlatformAdapter._truncate_preview("abcdef", 3, suffix="!")
        assert out == "abc!"


class TestFormatExecApproval:
    def test_default_template(self):
        ad = _bare(_DefaultAdapter)
        text = ad._format_exec_approval("rm -rf /", "scary")
        assert text == (
            "⚠️ Command Approval Required\n\n"
            "```\nrm -rf /\n```\n"
            "Reason: scary"
        )

    def test_smart_denied_appends_line(self):
        ad = _bare(_DefaultAdapter)
        text = ad._format_exec_approval("ls", "d", smart_denied=True)
        assert text.endswith(
            "\n\nSmart DENY: owner override applies to this one operation only."
        )

    def test_command_truncated_to_budget(self):
        ad = _bare(_DefaultAdapter)
        text = ad._format_exec_approval("x" * 5000, "d")
        assert "x" * 3000 + "..." in text
        assert "x" * 3001 not in text

    def test_escape_hook_applied_to_command_and_reason(self):
        class Escaping(_DefaultAdapter):
            def _ea_escape(self, text: str) -> str:
                return _html.escape(text)

        ad = _bare(Escaping)
        text = ad._format_exec_approval("echo <hi>", "a & b")
        assert "echo &lt;hi&gt;" in text
        assert "a &amp; b" in text

    def test_empty_command(self):
        ad = _bare(_DefaultAdapter)
        text = ad._format_exec_approval("", "d")
        assert "```\n\n```" in text


class TestFormatChoicePage:
    def test_single_page_no_page_info(self):
        opts, meta = BasePlatformAdapter._format_choice_page([1, 2, 3], 0, 10)
        assert opts == [1, 2, 3]
        assert meta["page_info"] == ""
        assert meta["total_pages"] == 1
        assert meta["page"] == 0

    def test_multi_page_slicing_and_info(self):
        options = list(range(25))
        opts, meta = BasePlatformAdapter._format_choice_page(options, 1, 10)
        assert opts == list(range(10, 20))
        assert meta == {
            "page": 1,
            "total_pages": 3,
            "start": 10,
            "end": 20,
            "total": 25,
            "page_info": " (11–20 of 25)",
        }

    def test_page_clamped_high(self):
        opts, meta = BasePlatformAdapter._format_choice_page(list(range(25)), 99, 10)
        assert meta["page"] == 2
        assert opts == list(range(20, 25))
        assert meta["page_info"] == " (21–25 of 25)"

    def test_page_clamped_negative(self):
        opts, meta = BasePlatformAdapter._format_choice_page(list(range(25)), -5, 10)
        assert meta["page"] == 0
        assert opts == list(range(10))

    def test_empty_options(self):
        opts, meta = BasePlatformAdapter._format_choice_page([], 0, 10)
        assert opts == []
        assert meta["total_pages"] == 1
        assert meta["page_info"] == ""

    def test_last_partial_page(self):
        opts, meta = BasePlatformAdapter._format_choice_page(list(range(11)), 1, 10)
        assert opts == [10]
        assert meta["page_info"] == " (11–11 of 11)"


class TestAdapterParity:
    """Rewired adapters produce byte-identical text vs their historical inline code."""

    def test_telegram_parity(self):
        from plugins.platforms.telegram.adapter import TelegramAdapter

        def old(command, description, smart_denied):
            cmd_preview = command[:3800] + "..." if len(command) > 3800 else command
            text = (
                f"⚠️ <b>Command Approval Required</b>\n\n"
                f"<pre>{_html.escape(cmd_preview)}</pre>\n\n"
                f"Reason: {_html.escape(description)}"
            )
            if smart_denied:
                text += "\n\n<b>Smart DENY:</b> owner override applies to this one operation only."
            return text

        ad = _bare(TelegramAdapter)
        for cmd in ["rm -rf /", "x" * 5000, "echo <hi> & 'stuff'", ""]:
            for sd in (False, True):
                assert ad._format_exec_approval(cmd, "why <b>&</b>", sd) == old(
                    cmd, "why <b>&</b>", sd
                )

    def test_feishu_parity(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter

        def old(command, description, smart_denied):
            cmd_preview = command[:3000] + "..." if len(command) > 3000 else command
            scope_note = (
                "\n\n**Smart DENY:** owner override applies to this one operation only."
                if smart_denied
                else ""
            )
            return f"```\n{cmd_preview}\n```\n**Reason:** {description}{scope_note}"

        ad = _bare(FeishuAdapter)
        for cmd in ["rm -rf /", "x" * 5000, ""]:
            for sd in (False, True):
                assert ad._format_exec_approval(cmd, "reason", sd) == old(cmd, "reason", sd)

    def test_matrix_parity(self):
        from plugins.platforms.matrix.adapter import MatrixAdapter

        def old_head(command, description):
            cmd_preview = command[:2000] + "..." if len(command) > 2000 else command
            return (
                "⚠️ **Dangerous command requires approval**\n"
                f"```\n{cmd_preview}\n```\n"
                f"Reason: {description}"
            )

        ad = _bare(MatrixAdapter)
        for cmd in ["rm -rf /", "x" * 5000, ""]:
            assert ad._format_exec_approval(cmd, "reason") == old_head(cmd, "reason")

    def test_telegram_pagination_parity(self):
        """_format_choice_page matches the old _build_*_keyboard arithmetic."""

        def old(options, page, page_size):
            total = len(options)
            total_pages = max(1, (total + page_size - 1) // page_size)
            page = max(0, min(page, total_pages - 1))
            start = page * page_size
            end = min(start + page_size, total)
            page_info = f" ({start + 1}–{end} of {total})" if total_pages > 1 else ""
            return options[start:end], page, total_pages, page_info

        for n in (0, 1, 8, 9, 10, 25):
            options = list(range(n))
            for page in (-3, 0, 1, 2, 99):
                for per in (8, 10):
                    o_opts, o_page, o_tp, o_info = old(options, page, per)
                    n_opts, meta = BasePlatformAdapter._format_choice_page(
                        options, page, per
                    )
                    assert n_opts == o_opts
                    assert meta["page"] == o_page
                    assert meta["total_pages"] == o_tp
                    assert meta["page_info"] == o_info
