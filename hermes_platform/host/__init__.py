"""Cached facts about the machine running this Python process.

Facts use hardware sources without environment-variable input or subprocesses.
"""

from hermes_platform.host.facts import interactive_session, windows_session_id

__all__ = ["interactive_session", "windows_session_id"]
