"""Agent internals -- extracted modules from run_agent.py.

These modules contain pure utility functions and self-contained classes
that were previously embedded in the 3,600-line run_agent.py. Extracting
them makes run_agent.py focused on the AIAgent orchestrator class.
"""

try:
    from . import jiter_preload as _jiter_preload  # noqa: F401
except ImportError:
    pass

# SSL compatibility: ensure certifi-based CA bundle on old macOS / Windows
# corporate proxies before any network calls are made.
try:
    from .ssl_compat import setup_ssl_compat  # noqa: F401
    setup_ssl_compat()
except ImportError:
    pass
