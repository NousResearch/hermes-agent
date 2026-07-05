"""Provider package init.

Imports all adapter modules so they register themselves on package
import. The Engine depends on this side effect.

Each adapter module is responsible for calling
`agent.providers.registry.register_provider(name, adapter_class)` at
module load time.

NOTE: this file deliberately imports the adapter modules in a
specific order:
1. FakeProviderAdapter (test-only)
2. MiniMaxAdapter (default conversational)
3. CodexAuthAdapter (critical supervision)

The Engine never explicitly imports MiniMax or Codex; it uses the
registry. Tests can swap in FakeProviderAdapter by calling
register_provider("minimax", FakeProviderAdapter) before engine
invocation.
"""

from __future__ import annotations

from agent.providers import fake
from agent.providers import minimax
from agent.providers import codex_auth

__all__ = ["fake", "minimax", "codex_auth"]