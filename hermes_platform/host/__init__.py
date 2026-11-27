"""Facts about the machine running this Python process.

``facts``: cached host facts (session, Windows session id); ``interfaces``: live interface
addresses, deliberately uncached. Both use hardware sources without environment-variable input or
subprocesses.
"""

from hermes_platform.host.facts import interactive_session, windows_session_id

__all__ = ["interactive_session", "windows_session_id"]
