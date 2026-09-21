"""TDD tests for tests.content_trust_helpers.loads_fenced_json.

These tests pin the contract of the helper that decodes the
UNTRUSTED_DOCUMENT JSON fence used by 8 content-trust consumer tests.
The fence format is defined in hermes_cli/content_trust.py:

    <<<UNTRUSTED_DOCUMENT>>> source="web_search" tier="external-untrusted">>
>
    {json payload}
    <<<END_UNTRUSTED_DOCUMENT>>>

The helper must extract the JSON between the fences and parse it, or
raise a clear error if the fence is absent/malformed.
"""

import json

import pytest


def test_loads_fenced_json_decodes_simple_fence():
    """Standard fence with JSON payload is decoded correctly."""
    from hermes_cli.content_trust import UNTRUSTED_BEGIN, UNTRUSTED_END
    from tests.content_trust_helpers import loads_fenced_json

    payload = json.dumps({"results": [{"title": "x", "url": "y"}]})
    fenced = f"{UNTRUSTED_BEGIN}\n{payload}\n{UNTRUSTED_END}"
    parsed = loads_fenced_json(fenced)
    assert parsed == {"results": [{"title": "x", "url": "y"}]}


def test_loads_fenced_json_decodes_fence_with_header_attrs():
    """Fence with source= and tier= attributes in the header is decoded."""
    from tests.content_trust_helpers import loads_fenced_json

    payload = json.dumps({"error": "something failed"})
    fenced = f'<<<UNTRUSTED_DOCUMENT>>> source="web_search" tier="external-untrusted">>>\n{payload}\n<<<END_UNTRUSTED_DOCUMENT>>>'
    parsed = loads_fenced_json(fenced)
    assert parsed == {"error": "something failed"}


def test_loads_fenced_json_strips_whitespace_around_payload():
    """Leading/trailing whitespace between fence and JSON is tolerated."""
    from tests.content_trust_helpers import loads_fenced_json

    payload = json.dumps({"ok": True})
    fenced = f'<<<UNTRUSTED_DOCUMENT>>>\n\n  {payload}  \n\n<<<END_UNTRUSTED_DOCUMENT>>>'
    parsed = loads_fenced_json(fenced)
    assert parsed == {"ok": True}


def test_loads_fenced_json_raises_on_missing_fence():
    """Plain JSON without a fence should raise a ValueError, not silently parse."""
    from tests.content_trust_helpers import loads_fenced_json

    with pytest.raises(ValueError, match="fence|UNTRUSTED|missing"):
        loads_fenced_json(json.dumps({"no": "fence"}))


def test_loads_fenced_json_raises_on_malformed_json_in_fence():
    """Fenced but non-JSON content should raise a ValueError."""
    from tests.content_trust_helpers import loads_fenced_json

    fenced = '<<<UNTRUSTED_DOCUMENT>>>\nnot valid json at all\n<<<END_UNTRUSTED_DOCUMENT>>>'
    with pytest.raises(ValueError, match="json|parse|JSON"):
        loads_fenced_json(fenced)
