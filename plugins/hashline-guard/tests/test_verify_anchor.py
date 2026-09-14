"""Regression tests for hashline_core.

Covers the count-based guard (verify_anchor) and the full hashline anchoring
primitives (find_all, compute_hashline, verify_anchor_by_hash, canonicalization).
"""
import importlib.util
import os
import hashlib
import json

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_PATH = os.path.join(PLUGIN_DIR, "..", "src", "hashline_core.py")


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
        raise AssertionError("hashline_core.py not found")
    f = tmp_path / "f.txt"
    f.write_text("alpha\nbeta\ngamma\n")
    assert core.verify_anchor(f.read_text(), "beta") == ("ok", None)


def test_absent_blocks(tmp_path):
    """Missing anchor should block with drift reason."""
    core = _load_core()
    if core is None:
        raise AssertionError("hashline_core.py not found")
    f = tmp_path / "f.txt"
    f.write_text("alpha\ngamma\n")
    status, reason = core.verify_anchor(f.read_text(), "beta")
    assert status == "block"
    assert "drifted" in (reason or "").lower()


def test_ambiguous_blocks(tmp_path):
    """Duplicate anchor should block with ambiguous reason."""
    core = _load_core()
    if core is None:
        raise AssertionError("hashline_core.py not found")
    f = tmp_path / "f.txt"
    f.write_text("alpha\nbeta\ngamma\nbeta\ndelta\n")
    status, reason = core.verify_anchor(f.read_text(), "beta")
    assert status == "block"
    assert "ambiguous" in (reason or "").lower()


def test_empty_old_string_blocks(tmp_path):
    """Empty old_string should block immediately."""
    core = _load_core()
    if core is None:
        raise AssertionError("hashline_core.py not found")
    f = tmp_path / "f.txt"
    f.write_text("alpha\n")
    status, reason = core.verify_anchor(f.read_text(), "")
    assert status == "block"
    assert "non-empty" in (reason or "").lower()


# ---- Task 8: per-occurrence hashline + canonicalization ----

def test_find_all_occurrences():
    """find_all should return (start, end, line_number) for every non-overlapping match."""
    core = _load_core()
    if core is None:
        raise AssertionError("hashline_core.py not found")
    occ = core.find_all("alpha\nbeta\ngamma\nbeta\ndelta\n", "beta")
    assert len(occ) == 2
    assert occ[0][2] == 2   # first beta on line 2 (1-based)
    assert occ[1][2] == 4   # second beta on line 4
    # byte offsets should point at the anchor text
    text = "alpha\nbeta\ngamma\nbeta\ndelta\n"
    assert text[occ[0][0]:occ[0][1]] == "beta"
    assert text[occ[1][0]:occ[1][1]] == "beta"


def test_per_occurrence_hashes(tmp_path):
    """Two identical anchors in different context must produce different hashlines."""
    core = _load_core()
    if core is None:
        raise AssertionError("hashline_core.py not found")
    f = tmp_path / "f.txt"
    f.write_text("alpha\nbeta\ngamma\nbeta\ndelta\n")
    text = f.read_text()
    h0 = core.compute_hashline(text, "beta", 0)
    h1 = core.compute_hashline(text, "beta", 1)
    assert h0 != h1
    assert len(h0) == 64  # SHA-256 hex


def test_verify_by_hash_ok(tmp_path):
    """Pinning the correct hashline should return ('ok', occurrence_index)."""
    core = _load_core()
    if core is None:
        raise AssertionError("hashline_core.py not found")
    f = tmp_path / "f.txt"
    f.write_text("alpha\nbeta\ngamma\nbeta\ndelta\n")
    text = f.read_text()
    h0 = core.compute_hashline(text, "beta", 0)
    status, idx = core.verify_anchor_by_hash(text, "beta", h0)
    assert status == "ok"
    assert idx == 0  # first beta
    h1 = core.compute_hashline(text, "beta", 1)
    status, idx = core.verify_anchor_by_hash(text, "beta", h1)
    assert status == "ok"
    assert idx == 1  # second beta


def test_verify_by_hash_block_returns_found(tmp_path):
    """Pinning a bogus hashline should block and return every found hashline + line."""
    core = _load_core()
    if core is None:
        raise AssertionError("hashline_core.py not found")
    f = tmp_path / "f.txt"
    f.write_text("alpha\nbeta\ngamma\nbeta\ndelta\n")
    text = f.read_text()
    bogus = hashlib.sha256(b"not_this").hexdigest()
    status, payload = core.verify_anchor_by_hash(text, "beta", bogus)
    assert status == "block"
    assert isinstance(payload, dict)
    assert "reason" in payload
    assert "found" in payload
    assert "lines" in payload
    assert len(payload["found"]) == 2
    assert payload["lines"] == [2, 4]


def test_verify_by_hash_single_occurrence_mismatch(tmp_path):
    """Single occurrence with wrong hashline should block (drift by context)."""
    core = _load_core()
    if core is None:
        raise AssertionError("hashline_core.py not found")
    f = tmp_path / "f.txt"
    f.write_text("alpha\nbeta\ngamma\n")
    text = f.read_text()
    bogus = hashlib.sha256(b"not_this").hexdigest()
    status, payload = core.verify_anchor_by_hash(text, "beta", bogus)
    assert status == "block"
    assert payload["lines"] == [2]
    assert len(payload["found"]) == 1


def test_crlf_canonicalization(tmp_path):
    """CRLF and LF variants of the same file must hash identically."""
    core = _load_core()
    if core is None:
        raise AssertionError("hashline_core.py not found")
    f = tmp_path / "f.txt"
    f.write_bytes(b"alpha\r\nbeta\r\ngamma\r\n")
    h_crlf = core.compute_hashline(f.read_text(), "beta", 0)
    h_lf = core.compute_hashline("alpha\nbeta\ngamma\n", "beta", 0)
    assert h_crlf == h_lf


# ---- Task 10: hashline_compute tool + pre_tool_call pin-drift support ----

PLUGIN_PATH = os.path.join(PLUGIN_DIR, "..", "__init__.py")


def _load_plugin():
    """Load the plugin module (__init__.py) via importlib."""
    if not os.path.exists(PLUGIN_PATH):
        return None
    spec = importlib.util.spec_from_file_location("hashline_guard_plugin", PLUGIN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_hashline_compute_returns_all_occurrences(tmp_path):
    """hashline_compute should return one entry per occurrence."""
    plugin = _load_plugin()
    if plugin is None:
        raise AssertionError("__init__.py not found — RED phase, expected to fail here")
    if not hasattr(plugin, "hashline_compute"):
        raise AssertionError("hashline_compute not yet implemented — RED phase expected")
    f = tmp_path / "f.txt"
    f.write_text("alpha\nbeta\ngamma\nbeta\ndelta\n")
    result = json.loads(plugin.hashline_compute({"path": str(f), "old_string": "beta", "window": 2}))
    assert result["count"] == 2
    assert len(result["hashlines"]) == 2
    assert result["hashlines"][0]["line"] == 2
    assert result["hashlines"][1]["line"] == 4
    assert len(result["hashlines"][0]["hashline"]) == 64
    assert "context" in result["hashlines"][0]


def test_hashline_compute_context_snippet(tmp_path):
    """Context should contain surrounding lines including the anchor line."""
    plugin = _load_plugin()
    if plugin is None:
        raise AssertionError("__init__.py not found — RED phase, expected to fail here")
    if not hasattr(plugin, "hashline_compute"):
        raise AssertionError("hashline_compute not yet implemented — RED phase expected")
    f = tmp_path / "f.txt"
    f.write_text("alpha\nbeta\ngamma\nbeta\ndelta\n")
    result = json.loads(plugin.hashline_compute({"path": str(f), "old_string": "beta", "window": 2}))
    first = result["hashlines"][0]
    ctx = first["context"]
    assert "alpha" in ctx
    assert "gamma" in ctx


def test_pre_tool_call_blocks_with_actual_hashline_when_pinned_but_drifted(tmp_path):
    """pre_tool_call should block with actual hashline when count==1 but expected_hashline drifted."""
    plugin = _load_plugin()
    if plugin is None:
        raise AssertionError("__init__.py not found — RED phase, expected to fail here")
    if not hasattr(plugin, "on_pre_tool_call"):
        raise AssertionError("on_pre_tool_call not yet implemented — RED phase expected")
    f = tmp_path / "f.txt"
    f.write_text("alpha\nbeta\ngamma\n")
    text = f.read_text()
    actual_hashline = plugin.compute_hashline(text, "beta", 0, window=2)
    wrong_hashline = "0" * 64
    args = {
        "mode": "replace",
        "path": str(f),
        "old_string": "beta",
        "expected_hashline": wrong_hashline,
    }
    result = plugin.on_pre_tool_call(tool_name="patch", args=args, cwd=str(tmp_path))
    assert result is not None
    assert result["action"] == "block"
    assert actual_hashline in result["message"]
    assert "Re-pin" in result["message"] or "expected_hashline" in result["message"]
    assert "context_hash()" not in result["message"]



def test_anchored_patch_schema_is_full_openai_shape():
    """SCHEMA must expose parameters.properties so coerce_tool_args can coerce."""
    plugin = _load_plugin()
    if plugin is None:
        raise AssertionError("__init__.py not found")
    s = plugin.SCHEMA
    assert s.get("name") == "anchored_patch"
    assert "description" in s
    params = s.get("parameters") or {}
    props = params.get("properties") or {}
    # coerce_tool_args reads properties; window must be typed integer
    assert props["window"]["type"] == "integer"
    assert set(params.get("required", [])) == {"path", "old_string", "new_string", "expected_hashline"}
    # every property has a description (schema-sanitizer / registry contract)
    for k, v in props.items():
        assert v.get("description"), f"property {k} missing description"
