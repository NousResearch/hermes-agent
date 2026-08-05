"""RED phase: failing tests for hashline_core.verify_anchor().
hashline_core.py does not exist yet — these tests MUST fail until Task 3 implements it.
"""
import importlib.util
import os

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_PATH = os.path.join(PLUGIN_DIR, "hashline_core.py")


def _load_core():
    """Load hashline_core via importlib; returns None if module absent."""
    if not os.path.exists(CORE_PATH):
        return None
    spec = importlib.util.spec_from_file_location("hashline_core", CORE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_exact_once_ok(tmp_path):
    """Anchor present exactly once should yield ('ok', None)."""
    core = _load_core()
    if core is None:
        raise AssertionError("hashline_core.py not found — RED phase, expected to fail here")
    f = tmp_path / "f.txt"
    f.write_text("alpha\nbeta\ngamma\n")
    assert core.verify_anchor(f.read_text(), "beta") == ("ok", None)


def test_absent_blocks(tmp_path):
    """Missing anchor should block with drift reason."""
    core = _load_core()
    if core is None:
        raise AssertionError("hashline_core.py not found — RED phase, expected to fail here")
    f = tmp_path / "f.txt"
    f.write_text("alpha\ngamma\n")
    status, reason = core.verify_anchor(f.read_text(), "beta")
    assert status == "block"
    assert "drifted" in (reason or "").lower()


def test_ambiguous_blocks(tmp_path):
    """Duplicate anchor should block with ambiguous reason."""
    core = _load_core()
    if core is None:
        raise AssertionError("hashline_core.py not found — RED phase, expected to fail here")
    f = tmp_path / "f.txt"
    f.write_text("alpha\nbeta\ngamma\nbeta\ndelta\n")
    status, reason = core.verify_anchor(f.read_text(), "beta")
    assert status == "block"
    assert "ambiguous" in (reason or "").lower()


def test_empty_old_string_blocks(tmp_path):
    """Empty old_string should block immediately."""
    core = _load_core()
    if core is None:
        raise AssertionError("hashline_core.py not found — RED phase, expected to fail here")
    f = tmp_path / "f.txt"
    f.write_text("alpha\n")
    status, reason = core.verify_anchor(f.read_text(), "")
    assert status == "block"
    assert "non-empty" in (reason or "").lower()
