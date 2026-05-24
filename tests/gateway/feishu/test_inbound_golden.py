"""Golden file regression for inbound event normalization.

Each fixture pair (`<name>.event.json` + `<name>.expected.json`) under
`fixtures/inbound/` is dispatched through the adapter and the resulting
MessageEvent's observable fields are compared against the expected JSON.

Add new fixtures by creating both files; no code change needed.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict

import pytest

pytest.importorskip("lark_oapi.channel")

from .conftest import FIXTURES_DIR, dispatch_inbound_event, load_fixture


def _discover_inbound_fixtures():
    inbound_dir = FIXTURES_DIR / "inbound"
    return sorted(p.stem.removesuffix(".event") for p in inbound_dir.glob("*.event.json"))


def _project_event(event: Any) -> Dict[str, Any]:
    """Extract observable, stable fields from a MessageEvent into a dict.

    Intentionally drops raw_message + timestamp + any other unstable fields.
    """
    src = event.source
    return {
        "text": event.text,
        "message_type": event.message_type.value if hasattr(event.message_type, "value") else str(event.message_type),
        "message_id": event.message_id,
        "media_urls_count": len(event.media_urls or []),
        "reply_to_message_id": event.reply_to_message_id,
        "source": {
            "platform": src.platform.value if hasattr(src.platform, "value") else str(src.platform),
            "chat_id": src.chat_id,
            "chat_type": src.chat_type,
            "user_id": src.user_id,
        },
    }


@pytest.mark.parametrize("fixture_name", _discover_inbound_fixtures())
def test_inbound_golden(fixture_name: str, adapter_harness):
    raw_event = load_fixture("inbound", f"{fixture_name}.event.json")
    expected = load_fixture("inbound", f"{fixture_name}.expected.json")

    asyncio.run(dispatch_inbound_event(adapter_harness, raw_event))

    assert len(adapter_harness.captured_inbound) == 1, (
        f"Expected exactly 1 captured inbound event for fixture {fixture_name!r}, "
        f"got {len(adapter_harness.captured_inbound)}"
    )
    actual = _project_event(adapter_harness.captured_inbound[0])
    assert actual == expected, (
        f"Inbound projection mismatch for {fixture_name!r}.\n"
        f"  expected: {json.dumps(expected, indent=2, ensure_ascii=False)}\n"
        f"  actual:   {json.dumps(actual,   indent=2, ensure_ascii=False)}"
    )
