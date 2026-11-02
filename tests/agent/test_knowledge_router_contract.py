import pytest

from agent.knowledge_router import INFORMATION_OWNER_MAP, resolve_information_owner


def test_router_public_mapping_is_closed_and_deterministic():
    assert INFORMATION_OWNER_MAP == {
        "HISTORICAL_TRANSCRIPT": "SESSION_SEARCH",
        "PROCEDURE": "SKILLS",
        "DOCUMENTARY_KNOWLEDGE": "CANONICAL_KB",
    }
    assert resolve_information_owner("HISTORICAL_TRANSCRIPT") == "SESSION_SEARCH"
    assert resolve_information_owner("PROCEDURE") == "SKILLS"
    assert resolve_information_owner("DOCUMENTARY_KNOWLEDGE", kb_id="kb-alpha") == "CANONICAL_KB"


def test_documentary_requires_explicit_kb_id():
    with pytest.raises(ValueError):
        resolve_information_owner("DOCUMENTARY_KNOWLEDGE")
    with pytest.raises(ValueError):
        resolve_information_owner("DOCUMENTARY_KNOWLEDGE", kb_id="")


def test_deferred_and_unknown_types_fail_closed():
    for label in ("CURRENT_STATE", "HOT_CONTEXT", "UNKNOWN_TYPE"):
        with pytest.raises(ValueError):
            resolve_information_owner(label)
