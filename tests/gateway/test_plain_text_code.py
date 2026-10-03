"""Plain transports remove code delimiters, never the payload inside them."""
import pytest

from gateway.config import PlatformConfig
from gateway.platforms.bluebubbles import BlueBubblesAdapter
from gateway.platforms.helpers import strip_markdown


@pytest.mark.parametrize("wrapped,payload", [
    ("`a * b * c`", "a * b * c"),
    ("``value = `literal` * 2``", "value = `literal` * 2"),
    ("```python\n# comment\nvalue = a ** b * c\n\n\n[docs](https://example.com)\n```", "# comment\nvalue = a ** b * c\n\n\n[docs](https://example.com)\n"),
    ("~~~~python\n# comment\na * b * c\n~~~~~", "# comment\na * b * c\n"),
    ("```python\n# comment\na * b * c", "# comment\na * b * c"),
])
def test_code_payload_survives_real_plain_adapter(wrapped, payload):
    adapter = BlueBubblesAdapter(PlatformConfig(enabled=True, extra={"server_url": "http://localhost:1234", "password": "unused"}))
    text = f"**Run this:**\n{wrapped}\n**Done.**"
    expected = f"Run this:\n{payload}\nDone."
    # An unclosed fence consumes the rest of the message as literal code.
    if wrapped.startswith("```python") and not wrapped.endswith("```"):
        expected = f"Run this:\n{payload}\n**Done.**"
    assert adapter.format_message(text) == expected
    assert strip_markdown(text) == expected


def test_prose_link_and_emphasis_rules_still_apply_around_inline_code():
    text = "**Run** `a * b * c` then [docs](https://example.com) and _finish_."
    assert strip_markdown(text) == "Run a * b * c then docs and finish."
    assert strip_markdown(text, keep_link_targets=True) == "Run a * b * c then docs\nhttps://example.com and finish."
