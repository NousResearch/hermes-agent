"""Top-level ``hermes --help`` epilog keeps onboarding examples visible.

The epilog in ``hermes_cli._parser`` is the first place new users look for
concrete commands. Assert the help text still surfaces the stable onboarding
set (model / tools / doctor / skills / gateway start) rather than freezing the
entire epilog string.
"""

from __future__ import annotations

from hermes_cli._parser import build_top_level_parser

# Commands called out for new-user discovery in the top-level help epilog.
_ONBOARDING_EXAMPLES = (
    "hermes model",
    "hermes tools",
    "hermes doctor",
    "hermes skills",
    "hermes gateway start",
)


def test_top_level_help_includes_onboarding_examples():
    help_text = build_top_level_parser()[0].format_help()
    assert "Examples:" in help_text
    missing = [example for example in _ONBOARDING_EXAMPLES if example not in help_text]
    assert not missing, f"top-level --help epilog missing examples: {missing}"
