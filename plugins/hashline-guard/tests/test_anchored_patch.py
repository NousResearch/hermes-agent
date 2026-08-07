"""RED phase: failing tests for anchored_patch handler.

These tests require the `anchored_patch` tool handler to exist in the plugin.
They MUST fail until the handler is implemented.
"""
import importlib.util
import os
from pathlib import Path

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
INIT_PATH = os.path.join(PLUGIN_DIR, "..", "__init__.py")
CORE_PATH = os.path.join(PLUGIN_DIR, "..", "src", "hashline_core.py")


def _load_plugin():
    spec = importlib.util.spec_from_file_location("hashline_guard_plugin", INIT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_anchored_patch_applies_at_pinned_occurrence(tmp_path):
    """2 identical anchors, pin the second occurrence via hashline."""
    plugin = _load_plugin()
    if not hasattr(plugin, "handle_anchored_patch"):
        raise AssertionError("handle_anchored_patch not implemented — RED phase expected")

    target = tmp_path / "file.txt"
    target.write_text("alpha\nbeta\ngamma\nbeta\ndelta\n")
    core = importlib.util.spec_from_file_location("hashline_core", CORE_PATH)
    core_mod = importlib.util.module_from_spec(core)
    core.loader.exec_module(core_mod)

    h1 = core_mod.compute_hashline(target.read_text(encoding="utf-8"), "beta", 1, window=2)

    before = target.read_text(encoding="utf-8")
    result = plugin.handle_anchored_patch({
        "path": str(target),
        "old_string": "beta\n",
        "new_string": "BETA\n",
        "expected_hashline": h1,
        "window": 2,
    })
    after = target.read_text(encoding="utf-8")

    assert result.get("applied") is True
    assert result.get("occurrence") == 1
    assert result.get("hashline") == h1
    assert after == "alpha\nbeta\ngamma\nBETA\ndelta\n"
    # first occurrence remains unchanged
    assert "beta\n" in after


def test_anchored_patch_blocks_on_hash_mismatch(tmp_path):
    """When hashline does not match any occurrence, file must remain unchanged."""
    plugin = _load_plugin()
    if not hasattr(plugin, "handle_anchored_patch"):
        raise AssertionError("handle_anchored_patch not implemented — RED phase expected")

    target = tmp_path / "file.txt"
    original = "alpha\nbeta\ngamma\nbeta\ndelta\n"
    target.write_text(original)

    result = plugin.handle_anchored_patch({
        "path": str(target),
        "old_string": "beta\n",
        "new_string": "BETA\n",
        "expected_hashline": "0" * 64,
        "window": 2,
    })

    assert result.get("applied") is False
    assert target.read_text(encoding="utf-8") == original


def test_anchored_patch_blocks_on_absent_anchor(tmp_path):
    """If old_string is absent from the file, it must block and not write."""
    plugin = _load_plugin()
    if not hasattr(plugin, "handle_anchored_patch"):
        raise AssertionError("handle_anchored_patch not implemented — RED phase expected")

    target = tmp_path / "file.txt"
    original = "alpha\ngamma\ndelta\n"
    target.write_text(original)

    result = plugin.handle_anchored_patch({
        "path": str(target),
        "old_string": "beta\n",
        "new_string": "BETA\n",
        "expected_hashline": "any",
        "window": 2,
    })

    assert result.get("applied") is False
    assert target.read_text(encoding="utf-8") == original


def test_anchored_patch_atomicity(tmp_path):
    """After a block condition, the file must remain byte-identical."""
    plugin = _load_plugin()
    if not hasattr(plugin, "handle_anchored_patch"):
        raise AssertionError("handle_anchored_patch not implemented — RED phase expected")

    target = tmp_path / "file.txt"
    original = "alpha\nbeta\ngamma\nbeta\ndelta\n"
    target.write_text(original)

    plugin.handle_anchored_patch({
        "path": str(target),
        "old_string": "beta\n",
        "new_string": "BETA\n",
        "expected_hashline": "0" * 64,
        "window": 2,
    })

    assert target.read_text(encoding="utf-8") == original
