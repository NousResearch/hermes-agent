"""The module docstring documents env vars near MATRIX_ALLOWED_ROOMS/MATRIX_REQUIRE_MENTION/
MATRIX_AUTO_THREAD as if they applied uniformly to rooms, but _resolve_room_identity classifies
any room with <=2 joined members as a DM regardless of m.direct or an explicit room name, which
silently bypasses those settings (#114733). Assert the docstring actually says so."""

from plugins.platforms.matrix import adapter


def test_module_docstring_covers_member_count_dm_classification():
    doc = adapter.__doc__

    assert "<=2 joined members" in doc
    assert "MATRIX_ALLOWED_ROOMS" in doc.split("Note:", 1)[1]
    assert "MATRIX_FREE_RESPONSE_ROOMS" in doc.split("Note:", 1)[1]
    assert "MATRIX_REQUIRE_MENTION" in doc.split("Note:", 1)[1]
    assert "MATRIX_DM_AUTO_THREAD" in doc.split("Note:", 1)[1]
