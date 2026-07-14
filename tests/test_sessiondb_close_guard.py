"""Regression test for SessionDB use-after-close guard."""
import tempfile
from pathlib import Path
import sys

HERMES_ROOT = Path(__file__).resolve().parent.parent
if str(HERMES_ROOT) not in sys.path:
    sys.path.insert(0, str(HERMES_ROOT))

from hermes_state import SessionDB


def test_use_after_close_raises():
    d = tempfile.mkdtemp()
    db = SessionDB(Path(d) / "state.db")
    db.close()
    try:
        db.get_session("foo")
    except RuntimeError as exc:
        assert "closed" in str(exc).lower()
    else:
        raise AssertionError("expected RuntimeError after close")


def test_close_is_idempotent():
    d = tempfile.mkdtemp()
    db = SessionDB(Path(d) / "state.db")
    db.close()
    db.close()  # should not raise


if __name__ == "__main__":
    test_use_after_close_raises()
    test_close_is_idempotent()
    print("ALL TESTS PASSED")
