"""Session-id minting contract: every surface that creates a session mints through
``hermes_state_ids.new_session_id`` and the id it produces is what lost-and-found salvage classifies
as a session id (the shape is the recovery sentinel for schema-less rows).
"""

from __future__ import annotations

import importlib
import re
from datetime import datetime

import pytest

import hermes_state_ids
from hermes_state_ids import SESSION_ID_PATTERN, new_session_id

# (module, callable(mint) -> str) for each minting site; ``mint`` is the patched helper.
SITES = [
    ("hermes_cli.foreign_sessions", None),
    ("agent.conversation_compression", None),
    ("hermes_cli.cli_commands_mixin", None),
    ("hermes_cli.cli_session_mixin", None),
    ("agent.agent_init", None),
    ("tui_gateway.server", lambda mod: mod._new_session_key()),
    ("gateway.session_lifecycle", lambda mod: mod._new_session_id(datetime(2026, 1, 2, 3, 4, 5))),
    ("hermes_state_portability", None),
    ("cli", None),
]


@pytest.mark.parametrize("hex_len,expected_re", [(6, r"^\d{8}_\d{6}_[0-9a-f]{6}$"), (8, r"^\d{8}_\d{6}_[0-9a-f]{8}$"),
                                                 (12, r"^\d{8}_\d{6}_[0-9a-f]{12}$")])
def test_minted_ids_are_what_salvage_classifies_as_session_ids(hex_len, expected_re):
    from hermes_cli.session_lost_and_found import _is_session_id
    sid = new_session_id(datetime(2026, 1, 2, 3, 4, 5), hex_len=hex_len)
    assert re.fullmatch(expected_re, sid) and sid.startswith("20260102_030405_")
    assert _is_session_id(sid)
    assert SESSION_ID_PATTERN is importlib.import_module("hermes_cli.session_lost_and_found").SESSION_ID_PATTERN


@pytest.mark.parametrize("module_name,call", SITES)
def test_every_minting_site_imports_the_one_helper(module_name, call):
    mod = importlib.import_module(module_name)
    minted = [name for name in ("new_session_id", "mint_session_id")
              if getattr(mod, name, None) is hermes_state_ids.new_session_id]
    assert minted, f"{module_name} does not mint through hermes_state_ids.new_session_id"
    if call is not None:
        assert SESSION_ID_PATTERN.match(call(mod))
