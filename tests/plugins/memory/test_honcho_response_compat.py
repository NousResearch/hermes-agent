"""Compatibility tests for Honcho peer responses from older servers."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest
from pydantic import ValidationError

from plugins.memory.honcho.client import _install_peer_response_compat


LEGACY_PEER_RESPONSE = {
    "id": "runi",
    "workspace_id": "abel-cell-v1",
    "created_at": "2026-08-01T00:00:00Z",
    "metadata": {},
    "configuration": {"observe_others": False},
}


def test_legacy_observe_others_is_accepted_only_on_peer_responses():
    import honcho.api_types as api_types
    import honcho.peer as peer_module

    starting_refs = {
        module_name: getattr(module, "PeerResponse", None)
        for module_name, module in tuple(sys.modules.items())
        if module_name == "honcho" or module_name.startswith("honcho.")
    }
    current_response = api_types.PeerResponse
    original_response = (
        current_response.__mro__[1]
        if getattr(current_response, "__hermes_legacy_observe_others_compat__", False)
        else current_response
    )

    try:
        with pytest.raises(ValidationError):
            original_response.model_validate(LEGACY_PEER_RESPONSE)
        with pytest.raises(ValidationError):
            api_types.PeerConfig.model_validate({"observe_others": False})

        compat_response = _install_peer_response_compat()
        assert compat_response is not None

        parsed = compat_response.model_validate(LEGACY_PEER_RESPONSE)
        assert parsed.id == "runi"
        assert "observe_others" not in parsed.configuration.model_dump()
        assert api_types.PeerResponse is compat_response
        assert peer_module.PeerResponse is compat_response
        assert _install_peer_response_compat() is compat_response

        # Request models remain strict: the response shim must not make the
        # deprecated peer-level setting writable again.
        with pytest.raises(ValidationError):
            api_types.PeerConfig.model_validate({"observe_others": False})
    finally:
        for module_name, response_class in starting_refs.items():
            module = sys.modules.get(module_name)
            if module is not None and response_class is not None:
                setattr(module, "PeerResponse", response_class)


def test_peer_response_compat_noops_for_minimal_fake_honcho_module():
    saved_modules = {
        name: module
        for name, module in tuple(sys.modules.items())
        if name == "honcho" or name.startswith("honcho.")
    }
    for name in saved_modules:
        sys.modules.pop(name, None)
    sys.modules["honcho"] = ModuleType("honcho")

    try:
        assert _install_peer_response_compat() is None
    finally:
        for name in tuple(sys.modules):
            if name == "honcho" or name.startswith("honcho."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
