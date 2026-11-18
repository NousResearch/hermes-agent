"""The core imports catalog plugins make through a module the storage refactor removed or emptied still resolve."""
from __future__ import annotations

import importlib
import sys

import pytest

from hermes_agent.compat.finder import Finder
from hermes_agent.compat.moves import MOVES

# (module, names) a catalog plugin imports at module top level, so without them it never registers:
# OpenViking 2.1.2 (plugin-catalog/openviking.yaml), hermes-council and Honcho 1.2.0 (plugin-catalog/honcho.yaml).
# Every MOVES key is checked as well.
_PLUGIN_IMPORTS = {
    "agent.message_content": ("flatten_message_text", "_text_from_part"),
    "agent.context_compressor_retry": ("retryable_user_text",),
    "agent.skill_commands": ("extract_user_instruction_from_skill_message", "register_auto_load_scaffold"),
    "hermes_state_common": ("TITLE_SOURCE_DERIVED", "TITLE_SOURCE_LLM"),
}


@pytest.mark.parametrize("old", sorted(MOVES.keys() | _PLUGIN_IMPORTS.keys()))
def test_old_import_paths_resolve_through_the_compat_finder(old, monkeypatch):
    for name in [n for n in sys.modules if n == old or n.startswith(f"{old}.")]:
        monkeypatch.delitem(sys.modules, name)
    # The finder hermes_agent.compat.install() puts in a running Hermes, not the strict one tests use.
    monkeypatch.setattr(sys, "meta_path", [Finder(MOVES), *(f for f in sys.meta_path if not isinstance(f, Finder))])

    module = importlib.import_module(old)

    if old in MOVES:
        assert module is importlib.import_module(MOVES[old])
    for name in _PLUGIN_IMPORTS.get(old, ()):
        assert hasattr(module, name), f"{old} has no {name!r}"


def test_the_plugin_title_constants_are_the_values_the_store_writes():
    """Honcho compares a session's stored ``title_source`` with them, so they must stay the store's values."""
    import hermes_state_common
    from persistence.sqlite._state import common as store

    assert (hermes_state_common.TITLE_SOURCE_DERIVED, hermes_state_common.TITLE_SOURCE_LLM) == (
        store.TITLE_SOURCE_DERIVED, store.TITLE_SOURCE_LLM)
