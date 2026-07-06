import re
from pathlib import Path

CHAT_SIDEBAR = Path(__file__).resolve().parent.parent / "web/src/components/ChatSidebar.tsx"


def test_sidecar_session_create_requests_close_on_disconnect():
    """The sidecar must opt its session into close_on_disconnect so the gateway
    reaps the slash_worker on WS disconnect (the #21370/#21467 leak)."""
    source = CHAT_SIDEBAR.read_text(encoding="utf-8")
    call = re.search(r'"session\.create",\s*\{(.*?)\}', source, re.DOTALL)
    assert call, "sidecar session.create call not found"
    assert re.search(r"close_on_disconnect:\s*true", call.group(1))
