"""The module docstring documents env vars near MATRIX_ALLOWED_ROOMS/MATRIX_REQUIRE_MENTION/
MATRIX_AUTO_THREAD as if they applied uniformly to rooms, but _resolve_room_identity classifies
any room with <=2 joined members as a DM regardless of m.direct or an explicit room name, which
silently bypasses those settings (#114733). Assert the docstring actually says so.

Issue #114733 says this behavior "is not surfaced anywhere a user would reasonably
encounter it" and specifically asks for a callout wherever MATRIX_ALLOWED_ROOMS,
MATRIX_FREE_RESPONSE_ROOMS, MATRIX_REQUIRE_MENTION, and MATRIX_AUTO_THREAD are
documented for operators — i.e. the env-var reference table and the Matrix user
guide, not just the adapter's source docstring. Assert those two operator-facing
docs carry the same callout.
"""

from pathlib import Path

from plugins.platforms.matrix import adapter

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_module_docstring_covers_member_count_dm_classification():
    doc = adapter.__doc__

    assert "<=2 joined members" in doc
    assert "MATRIX_ALLOWED_ROOMS" in doc.split("Note:", 1)[1]
    assert "MATRIX_FREE_RESPONSE_ROOMS" in doc.split("Note:", 1)[1]
    assert "MATRIX_REQUIRE_MENTION" in doc.split("Note:", 1)[1]
    assert "MATRIX_DM_AUTO_THREAD" in doc.split("Note:", 1)[1]


def test_env_var_reference_covers_dm_classification_bypass():
    text = (REPO_ROOT / "website/docs/reference/environment-variables.md").read_text()

    for var in (
        "MATRIX_ALLOWED_ROOMS",
        "MATRIX_REQUIRE_MENTION",
        "MATRIX_FREE_RESPONSE_ROOMS",
        "MATRIX_AUTO_THREAD",
    ):
        row = next(line for line in text.splitlines() if line.startswith(f"| `{var}` |"))
        assert "2 or fewer joined members" in row, f"{var} row missing the DM-classification callout"


def test_matrix_user_guide_covers_dm_classification_bypass():
    text = (REPO_ROOT / "website/docs/user-guide/messaging/matrix.md").read_text()

    dms_row = next(line for line in text.splitlines() if line.startswith("| **DMs**"))
    rooms_row = next(line for line in text.splitlines() if line.startswith("| **Rooms**"))

    assert "2 or fewer joined members" in dms_row
    assert "MATRIX_ALLOWED_ROOMS" in rooms_row
    assert "MATRIX_REQUIRE_MENTION" in rooms_row
    assert "MATRIX_FREE_RESPONSE_ROOMS" in rooms_row
