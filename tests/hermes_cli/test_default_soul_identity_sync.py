"""DEFAULT_SOUL_MD and DEFAULT_AGENT_IDENTITY must stay the same text.

default_soul.py seeds SOUL.md on first run while DEFAULT_AGENT_IDENTITY is the
no-SOUL.md fallback, so any drift splits the default persona in two. Behavior
contract via imports — never by reading source files.
"""

from agent.prompt_builder import DEFAULT_AGENT_IDENTITY
from hermes_cli.default_soul import DEFAULT_SOUL_MD


def test_default_soul_matches_default_agent_identity():
    assert DEFAULT_SOUL_MD == DEFAULT_AGENT_IDENTITY
