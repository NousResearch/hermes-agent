"""Plain transports remove code delimiters, never the payload inside them."""
import pytest

from gateway.platforms.helpers import strip_markdown
from plugins.platforms.feishu.adapter import _strip_markdown_to_plain_text
from plugins.platforms.irc.adapter import IRCAdapter
from plugins.platforms.line.adapter import strip_markdown_preserving_urls

_PLAIN_STRIPPERS = {
    "shared": strip_markdown,
    "imessage": lambda t: strip_markdown(t, keep_link_targets=True),
    "feishu": _strip_markdown_to_plain_text,
    "irc": IRCAdapter._strip_markdown,
    "line": strip_markdown_preserving_urls,
}


@pytest.mark.parametrize("surface", sorted(_PLAIN_STRIPPERS))
@pytest.mark.parametrize("wrapped,payload", [
    ("`a * b * c`", "a * b * c"),
    ("``value = `literal` * 2``", "value = `literal` * 2"),
    ("`f(**kwargs)`", "f(**kwargs)"),
    ("```text\n> not a quote\n~~x~~ [a](b)\n```", "> not a quote\n~~x~~ [a](b)"),
    ("```python\n# comment\nvalue = a ** b * c\nx = my_var_name\n```", "# comment\nvalue = a ** b * c\nx = my_var_name"),
    ("~~~~python\n# comment\na * b * c\n~~~~~", "# comment\na * b * c"),
    # A shorter fence inside a longer one is code, never the close (CommonMark; buzz#9349).
    ('````md\nfence = "```"\nx = "**literal**"\n````', 'fence = "```"\nx = "**literal**"'),
])
def test_code_payload_survives_every_plain_surface(surface, wrapped, payload):
    out = _PLAIN_STRIPPERS[surface](f"**Run this:**\n{wrapped}\n**Done.**")
    assert out == f"Run this:\n{payload}\nDone."


def test_unclosed_fence_keeps_the_rest_literal_and_prose_rules_still_apply():
    assert strip_markdown("**Run:**\n```sh\nrm *.tmp\n**Done.**") == "Run:\nrm *.tmp\n**Done.**"
    text = "**Run** `a * b * c` then [docs](https://example.com) and _finish_."
    assert strip_markdown(text) == "Run a * b * c then docs and finish."
    assert strip_markdown(text, keep_link_targets=True) == "Run a * b * c then docs\nhttps://example.com and finish."
