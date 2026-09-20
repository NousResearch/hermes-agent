"""RoomLink byte-transport readiness is independent of optional PDF rendering.

Retained from Files donor 691bb08a5cd310fffc5f3d01653dc93f394fc080.
The direct capability assertion was not duplicated by the canonical custody suites.
"""

from gateway.platforms import api_server_room_attachments as room_attachments


def test_capability_does_not_require_the_optional_pdf_renderer(monkeypatch):
    monkeypatch.setattr(room_attachments, "web", object())
    assert room_attachments.roomlink_attachments_available() is True

    monkeypatch.setattr(room_attachments, "web", None)
    assert room_attachments.roomlink_attachments_available() is False
