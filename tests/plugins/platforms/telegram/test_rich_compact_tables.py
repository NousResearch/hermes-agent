"""Bot API 10.3 compact-table rewrite: GFM tables -> ``<table compact>`` rich HTML.

Fork-specific contract tests (written against current upstream behavior): the rich
send path must hand ``sendRichMessage`` a markdown payload whose pipe tables are
expressed as inline rich HTML with the ``compact`` flag, while everything that is
not a real table (fences, prose pipes, over-wide tables) survives untouched.
"""

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter
from plugins.platforms.telegram.telegram_markdown import compact_rich_tables


@pytest.fixture()
def adapter() -> TelegramAdapter:
    return TelegramAdapter(PlatformConfig(
        enabled=True, token="123:abc",
        extra={"rich_messages": True, "rich_compact_tables": True},
    ))


def test_compact_tables_is_opt_in_by_default():
    """Without the extra the payload keeps the plain-markdown table path (upstream contract)."""
    plain = TelegramAdapter(PlatformConfig(enabled=True, token="123:abc", extra={"rich_messages": True}))
    md = "| A | B |\n|---|---|\n| 1 | 2 |"
    out = plain._rich_message_payload(md)["markdown"]
    assert "<table" not in out
    assert "| A | B |" in out and "|---|---|" in out


def test_simple_table_becomes_compact_html():
    md = "| A | B |\n|---|---|\n| 1 | 2 |\n"
    out = compact_rich_tables(md)
    assert out == (
        "<table compact>"
        "<tr><th>A</th><th>B</th></tr>"
        "<tr><td>1</td><td>2</td></tr>"
        "</table>\n"
    )


def test_alignment_maps_to_align_attrs():
    md = "| L | C | R |\n|:--|:-:|--:|\n| a | b | c |\n"
    out = compact_rich_tables(md)
    assert '<th align="left">L</th>' in out
    assert '<th align="center">C</th>' in out
    assert '<th align="right">R</th>' in out
    assert '<td align="center">b</td>' in out


def test_inline_formatting_in_cells_becomes_html():
    md = "| X | Y |\n|---|---|\n| **bold** | `code` |\n| ~~gone~~ | [t](https://x.it) |\n"
    out = compact_rich_tables(md)
    assert "<b>bold</b>" in out
    assert "<code>code</code>" in out
    assert "<s>gone</s>" in out
    assert '<a href="https://x.it">t</a>' in out


def test_escaped_pipe_stays_inside_its_cell():
    md = "| A | B |\n|---|---|\n| x \\| y | z |\n"
    out = compact_rich_tables(md)
    assert "<td>x | y</td>" in out


def test_html_specials_are_escaped():
    md = "| A | B |\n|---|---|\n| a < b & c | d > e |\n"
    out = compact_rich_tables(md)
    assert "<td>a &lt; b &amp; c</td>" in out
    assert "<td>d &gt; e</td>" in out


def test_currency_escape_is_reverted_inside_cells():
    # escape_rich_currency_dollars turns "$5" into "\$5" upstream in the pipeline;
    # math never pairs inside HTML cells, so the cell must show a plain $.
    md = "| Prezzo | Q |\n|---|---|\n| $5 | 2 |\n"
    from plugins.platforms.telegram.telegram_markdown import escape_rich_currency_dollars

    out = compact_rich_tables(escape_rich_currency_dollars(md))
    assert "<td>$5</td>" in out
    assert "\\$" not in out


def test_table_shape_inside_code_fence_is_untouched():
    md = "```md\n| A | B |\n|---|---|\n| 1 | 2 |\n```\n"
    assert compact_rich_tables(md) == md


def test_over_20_columns_left_as_markdown():
    cols = [f"C{i}" for i in range(21)]
    md = "| " + " | ".join(cols) + " |\n" + "|---|" * 21 + "\n| " + " | ".join("x" for _ in cols) + " |\n"
    assert compact_rich_tables(md) == md


def test_prose_pipes_and_no_table_pass_through():
    md = "a | b | c\n\nnext paragraph\n"
    assert compact_rich_tables(md) == md


def test_surrounding_text_is_preserved():
    md = "Ecco la tabella:\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\nFine.\n"
    out = compact_rich_tables(md)
    assert out.startswith("Ecco la tabella:\n\n<table compact>")
    assert out.endswith("</table>\n\nFine.\n")


def test_two_tables_both_compact():
    md = "| A | B |\n|---|---|\n| 1 | 2 |\n\ntext\n\n| C | D |\n|---|---|\n| 3 | 4 |\n"
    out = compact_rich_tables(md)
    assert out.count("<table compact>") == 2
    assert out.count("</table>") == 2


def test_rich_message_payload_carries_compact_table(adapter):
    payload = adapter._rich_message_payload("| A | B |\n|---|---|\n| 1 | 2 |\n")
    assert "<table compact>" in payload["markdown"]
    assert "|---|" not in payload["markdown"]
