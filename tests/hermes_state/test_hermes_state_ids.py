"""Session-id minting contract: every surface that creates a session mints through
``hermes_state_ids.new_session_id`` and the id it produces is what lost-and-found salvage classifies
as a session id (the shape is the recovery sentinel for schema-less rows).
"""

from __future__ import annotations

import importlib
import re
from datetime import datetime

import pytest

from hermes_state_ids import SESSION_ID_PATTERN, new_session_id

@pytest.mark.parametrize("hex_len,expected_re", [(6, r"^\d{8}_\d{6}_[0-9a-f]{6}$"), (8, r"^\d{8}_\d{6}_[0-9a-f]{8}$"),
                                                 (12, r"^\d{8}_\d{6}_[0-9a-f]{12}$")])
def test_minted_ids_are_what_salvage_classifies_as_session_ids(hex_len, expected_re):
    from hermes_cli.session_lost_and_found import _is_session_id
    sid = new_session_id(datetime(2026, 1, 2, 3, 4, 5), hex_len=hex_len)
    assert re.fullmatch(expected_re, sid) and sid.startswith("20260102_030405_")
    assert _is_session_id(sid)
    assert SESSION_ID_PATTERN is importlib.import_module("hermes_cli.session_lost_and_found").SESSION_ID_PATTERN
