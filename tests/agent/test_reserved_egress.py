from types import SimpleNamespace

import pytest

from agent.reserved_egress import assert_no_unauthorized_reserved_egress


def test_unclassified_payload_is_allowed():
    assert_no_unauthorized_reserved_egress(SimpleNamespace(), {"messages": [{"content": "ordinary"}]})


def test_reserved_payload_requires_explicit_authorization():
    with pytest.raises(PermissionError, match="Blocked outbound LLM request"):
        assert_no_unauthorized_reserved_egress(
            SimpleNamespace(reserved_content_authorized=False),
            {"extra_body": {"classification": "restricted", "content": "internal"}},
        )


def test_explicit_authorization_allows_reserved_payload():
    assert_no_unauthorized_reserved_egress(
        SimpleNamespace(reserved_content_authorized=True),
        {"extra_body": {"classification": "restricted", "content": "internal"}},
    )
