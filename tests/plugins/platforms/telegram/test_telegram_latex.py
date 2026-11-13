"""LaTeX macros in ordinary prose reach a Telegram chat as Unicode symbols, not raw backslash text."""

import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hermes_agent.gateway.config import PlatformConfig
from hermes_agent.plugins.platforms.telegram.adapter import TelegramAdapter
from tests.fakes.platforms.telegram_standin import mdv2_error


def _shown(formatted: str) -> str:
    """What the Telegram client displays for a MarkdownV2 string (escapes dropped)."""
    return re.sub(r"\\(.)", r"\1", formatted)


@pytest.mark.parametrize(
    ("markdown", "visible"),
    [
        ("Flow: A \\rightarrow B \\rightarrow C", "Flow: A → B → C"),
        ("If $x \\leq y$ and $y \\neq 0$ then \\alpha \\Rightarrow \\beta", "If x ≤ y and y ≠ 0 then α ⇒ β"),
        ("Cost: $5 \\times 3$ items, total \\approx $15", "Cost: 5 × 3 items, total ≈ $15"),
        ("compact $x\\leq y$ math", "compact x≤ y math"),
        # Controls: code, paths, an escaped backslash, unknown macros and currency stay as written.
        ("Run `a \\rightarrow b` then\n```tex\n\\to\n```", "Run `a \\rightarrow b` then\n```tex\n\\to\n```"),
        ("C:\\in\\data, \\\\rightarrow, \\textbf{x}", "C:\\in\\data, \\\\rightarrow, \\textbf{x}"),
        ("Prices $395k vs $483k", "Prices $395k vs $483k"),
        # Relative Windows paths, env-var paths, hyphenated words, link targets and bare URLs stay as written.
        ("Run .\\to\\file.ps1 then ..\\top\\x", "Run .\\to\\file.ps1 then ..\\top\\x"),
        ("%APPDATA%\\top\\cfg and ~\\to\\x", "%APPDATA%\\top\\cfg and ~\\to\\x"),
        ("edit \\in-place", "edit \\in-place"),
        ("[docs](https://e.com/?q=\\to) A \\to B", "[docs](https://e.com/?q=\\to) A → B"),
        ("see https://e.com/a?x=\\in&y", "see https://e.com/a?x=\\in&y"),
        # A currency pair keeps both dollar signs even with a space before the amount.
        ("$5 \\to $ 10", "$5 → $ 10"),
        # A digit on the next line or after a space is prose, never an amount: the math still maps.
        ("1. Show $x \\to y$\n2. Then $a \\le b$\n3. Done", "1. Show x → y\n2. Then a ≤ b\n3. Done"),
        ("$\\alpha$ 3 times", "α 3 times"),
        # Adjacent macros all map, not only the first of the run.
        ("\\alpha\\beta and \\alpha\\rightarrow\\beta", "αβ and α→β"),
        # A link target with parentheses and a query keeps its macro; the prose after it still maps.
        ("[plot](https://example.com/f(x)?q=\\alpha) A \\to B", "[plot](https://example.com/f(x)?q=\\alpha) A → B"),
    ],
)
def test_format_message_renders_latex_macros_as_unicode(markdown, visible):
    formatted = TelegramAdapter(PlatformConfig(enabled=True, token="123:abc")).format_message(markdown)
    assert mdv2_error(formatted) is None
    assert _shown(formatted) == visible


@pytest.mark.asyncio
async def test_rich_send_maps_bare_macros_and_keeps_math_for_the_client():
    """A rich reply (a table makes it rich-eligible) maps bare macros; math spans the rich client
    typesets itself, and code, reach sendRichMessage unchanged."""
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="123:abc", extra={"rich_messages": True}))
    bot = MagicMock()
    bot.do_api_request = AsyncMock(return_value=SimpleNamespace(message_id=7))
    bot.send_chat_action = AsyncMock()
    adapter._bot = bot
    long_formula = "$" + " + ".join(f"x_{i} \\leq y_{i}" for i in range(20)) + "$"  # > 120 chars
    content = ("Pipeline: A \\rightarrow B\n\n| a | b |\n|---|---|\n| $x \\leq y$ | `\\to` |\n\n$$\\alpha + \\beta$$\n\n"
               + long_formula)

    assert (await adapter.send("4242", content)).success
    method, = bot.do_api_request.call_args.args
    markdown = bot.do_api_request.call_args.kwargs["api_kwargs"]["rich_message"]["markdown"]
    assert method == "sendRichMessage"
    assert "A → B" in markdown and "\\rightarrow" not in markdown
    assert "$x \\leq y$" in markdown and "`\\to`" in markdown and "$$\\alpha + \\beta$$" in markdown
    assert long_formula in markdown


@pytest.mark.parametrize(
    "markdown",
    ["Run .\\to\\file.ps1", "%APPDATA%\\top\\cfg", "[docs](https://e.com/?q=\\to)", "see https://e.com/a?x=\\in&y",
     "[plot](https://example.com/f(x)?q=\\alpha)", "\\alpha\\beta\\data", "\\alpha\\beta-file"],
)
def test_rich_latex_keeps_paths_and_link_targets(markdown):
    from hermes_agent.plugins.platforms.telegram.telegram_markdown import normalize_rich_latex

    assert normalize_rich_latex(markdown) == markdown
