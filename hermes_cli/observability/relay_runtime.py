"""Compatibility alias for the core Hermes Relay runtime.

New code should import :mod:`agent.relay_runtime`. This module remains an
alias, rather than a copy, so existing plugins and tests share the same
profile registry and test-reset state during the migration.
"""

from __future__ import annotations

import sys

from agent import relay_runtime as _core_relay_runtime

sys.modules[__name__] = _core_relay_runtime
