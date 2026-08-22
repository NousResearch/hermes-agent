"""Tests for gateway.whatsapp_identity alias resolution path."""

import json

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from gateway.whatsapp_identity import (
    canonical_whatsapp_identifier,
    expand_whatsapp_aliases,
)


@pytest.fixture
def paired_home(tmp_path, monkeypatch):
    """A gateway home holding the bridge's LID->phone mapping."""
    tmp_home = tmp_path / "hermes-home"
    mapping_dir = tmp_home / "platforms" / "whatsapp" / "session"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    (mapping_dir / "lid-mapping-999999999999999.json").write_text(
        json.dumps("15551234567@s.whatsapp.net"),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_home))
    return tmp_home


def test_aliases_resolve_on_modern_platforms_layout(paired_home):
    assert expand_whatsapp_aliases("999999999999999@lid") == {
        "999999999999999",
        "15551234567",
    }


def test_aliases_survive_a_profile_scope_override(paired_home):
    """A routed turn must still resolve a group sender's LID to their phone.

    The gateway multiplexer wraps each routed turn in ``_profile_runtime_scope``,
    which points ``get_hermes_home()`` at ``profiles/<name>``. The WhatsApp
    session store is a PROCESS asset and only ever exists under the launch
    home, so resolving it through that override made every mapping lookup
    miss: a group sender arrives as a LID, no longer expanded to their phone
    number, and every phone-keyed gate (``allow_admin_from``,
    ``group_allow_admin_from``, ``allow_from``, the pairing store) silently
    stopped matching them.
    """
    profile_home = paired_home / "profiles" / "village"
    profile_home.mkdir(parents=True, exist_ok=True)

    expected = {"999999999999999", "15551234567"}
    assert expand_whatsapp_aliases("999999999999999@lid") == expected

    token = set_hermes_home_override(str(profile_home))
    try:
        assert expand_whatsapp_aliases("999999999999999@lid") == expected
        # The canonical identity must not change either: session keys and
        # authz would otherwise disagree about who a sender is depending on
        # whether the code happened to run inside the scope.
        assert canonical_whatsapp_identifier("999999999999999@lid") == "15551234567"
    finally:
        reset_hermes_home_override(token)


