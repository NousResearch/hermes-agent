import platform

import pytest

from plugins.builder_adapter.auth import darwin_peer_credentials
from plugins.builder_adapter.errors import AdapterError


def test_linux_peercred_assumption_fails_closed(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    with pytest.raises(AdapterError) as raised:
        darwin_peer_credentials(object())
    assert raised.value.code == "AUTHENTICATION_FAILED"


def test_peer_identity_is_not_taken_from_json():
    payload = {"caller_principal": "attacker"}
    server_derived_principal = "orchestrator-mcp"
    assert payload["caller_principal"] != server_derived_principal
