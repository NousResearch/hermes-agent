"""Tests for the anchored_patch tool handler.

The handler returns a JSON string (the tool-handler contract — see
tools/registry._normalize_handler_result); tests decode before asserting.
"""
import importlib.util
import json
import os
from pathlib import Path

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
INIT_PATH = os.path.join(PLUGIN_DIR, "..", "__init__.py")
CORE_PATH = os.path.join(PLUGIN_DIR, "..", "src", "hashline_core.py")


def _load(out: str) -> dict:
    return json.loads(out)


def _load_plugin():
    spec = importlib.util.spec_from_file_location("hashline_guard_plugin", INIT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_core():
    spec = importlib.util.spec_from_file_location("hashline_core", CORE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_anchored_patch_applies_at_pinned_occurrence(tmp_path):
    """2 identical anchors, pin the second occurrence via hashline."""
    plugin = _load_plugin()
    core_mod = _load_core()

    target = tmp_path / "file.txt"
    target.write_text("alpha\nbeta\ngamma\nbeta\ndelta\n")

    h1 = core_mod.compute_hashline(target.read_text(encoding="utf-8"), "beta", 1, window=2)

    before = target.read_text(encoding="utf-8")
    result = _load(plugin.handle_anchored_patch({
        "path": str(target),
        "old_string": "beta\n",
        "new_string": "BETA\n",
        "expected_hashline": h1,
        "window": 2,
    }))
    after = target.read_text(encoding="utf-8")

    assert result.get("applied") is True
    assert result.get("occurrence") == 1
    assert result.get("hashline") == h1
    assert after == "alpha\nbeta\ngamma\nBETA\ndelta\n"
    assert before != after
    # first occurrence remains unchanged
    assert "beta\n" in after


def test_anchored_patch_returns_json_string_contract(tmp_path):
    """Handlers must return a JSON STRING (tools/registry dispatch contract),
    never a bare dict — dict returns become tool_result_contract errors."""
    plugin = _load_plugin()
    core_mod = _load_core()
    target = tmp_path / "file.txt"
    target.write_text("alpha\nbeta\ngamma\n")
    h0 = core_mod.compute_hashline(target.read_text(encoding="utf-8"), "beta", 0, window=2)

    out = plugin.handle_anchored_patch({
        "path": str(target),
        "old_string": "beta\n",
        "new_string": "BETA\n",
        "expected_hashline": h0,
        "window": 2,
    })
    assert isinstance(out, str), f"handler must return a str, got {type(out).__name__}"
    decoded = _load(out)
    assert decoded.get("applied") is True


def test_anchored_patch_blocks_on_hash_mismatch(tmp_path):
    """When hashline does not match any occurrence, file must remain unchanged."""
    plugin = _load_plugin()

    target = tmp_path / "file.txt"
    original = "alpha\nbeta\ngamma\nbeta\ndelta\n"
    target.write_text(original)

    result = _load(plugin.handle_anchored_patch({
        "path": str(target),
        "old_string": "beta\n",
        "new_string": "BETA\n",
        "expected_hashline": "0" * 64,
        "window": 2,
    }))

    assert result.get("applied") is False
    assert target.read_text(encoding="utf-8") == original


def test_anchored_patch_blocks_on_absent_anchor(tmp_path):
    """If old_string is absent from the file, it must block and not write."""
    plugin = _load_plugin()

    target = tmp_path / "file.txt"
    original = "alpha\ngamma\ndelta\n"
    target.write_text(original)

    result = _load(plugin.handle_anchored_patch({
        "path": str(target),
        "old_string": "beta\n",
        "new_string": "BETA\n",
        "expected_hashline": "any",
        "window": 2,
    }))

    assert result.get("applied") is False
    assert target.read_text(encoding="utf-8") == original


def test_anchored_patch_atomicity(tmp_path):
    """After a block condition, the file must remain byte-identical."""
    plugin = _load_plugin()

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
