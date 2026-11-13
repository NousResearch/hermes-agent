"""Cached facts about the machine running this Python process.

Facts use hardware sources without environment-variable input or subprocesses.
"""

from hermes_agent.hermes_platform.host.facts import interactive_session

__all__ = ["interactive_session"]
