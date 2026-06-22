"""Tests for agent/tool_output_compressor.py.

Covers:
- Compression disabled by default (config)
- headroom-ai absent (pass-through)
- SmartCrusher exception → pass-through
- JSON compression via SmartCrusher
- Heuristic text compression (repetition, truncation)
- Error-line preservation in truncation
- Small output pass-through
- Cache operations
- Config cache TTL
"""

from __future__ import annotations

import json
import threading
from unittest.mock import patch, MagicMock

import pytest

import agent.tool_output_compressor as comp


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset module state before and after each test."""
    comp._reset_state()
    yield
    comp._reset_state()


# ---------------------------------------------------------------------------
# Config gating
# ---------------------------------------------------------------------------

def test_disabled_by_default():
    """Compression is off unless context.headroom.enabled is True."""
    result = comp.compress_tool_result("dummy", "x" * 5000)
    assert result is None


def test_enabled_compresses_when_configured():
    """With config, compress_tool_result attempts compression."""
    cfg = {"enabled": True, "min_chars": 100}
    # Build output that will trigger heuristic (>200 lines)
    text = "\n".join([f"line {i}: some content here" for i in range(250)])
    result = comp.compress_tool_result("test_tool", text, config=cfg)
    # Should be compressed (heuristic truncation)
    assert result is not None
    assert "COMPRESSED" in result
    assert len(result) < len(text)


def test_env_var_disable():
    """HERMES_COMPRESS_DISABLE=1 bypasses compression."""
    import os
    old = os.environ.get("HERMES_COMPRESS_DISABLE")
    os.environ["HERMES_COMPRESS_DISABLE"] = "1"
    try:
        comp._config_cache = None  # force re-read
        result = comp.compress_tool_result(
            "test_tool",
            "\n".join([f"  {i}|def h_{i}(): pass" for i in range(500)]),
        )
        assert result is None  # disabled by env var
    finally:
        if old is None:
            os.environ.pop("HERMES_COMPRESS_DISABLE", None)
        else:
            os.environ["HERMES_COMPRESS_DISABLE"] = old


# ---------------------------------------------------------------------------
# headroom-ai absent
# ---------------------------------------------------------------------------

def test_headroom_absent_pass_through():
    """When SmartCrusher import fails, SmartCrusher path is skipped."""
    with patch.object(comp, "_get_crusher", return_value=None):
        result = comp.compress_tool_result(
            "search_files",
            json.dumps([{"a": i} for i in range(100)]),
            config={"enabled": True, "min_chars": 0},
        )
        assert result is None


def test_crusher_init_failed_sentinel():
    """Failed SmartCrusher init sets sentinel so we don't retry import."""
    import builtins
    real_import = builtins.__import__
    headroom_imports = 0

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        nonlocal headroom_imports
        if name == "headroom":
            headroom_imports += 1
            raise ImportError("no headroom")
        return real_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=fake_import):
        for _ in range(5):
            comp.compress_tool_result(
                "x",
                "x" * 3000,
                config={"enabled": True, "min_chars": 0},
            )
    assert headroom_imports == 1
    assert comp._crusher is False


# ---------------------------------------------------------------------------
# SmartCrusher compression
# ---------------------------------------------------------------------------

def test_json_compression():
    """SmartCrusher compresses JSON data."""
    data = json.dumps([{"file": f"mod_{i}.py", "line": i * 10, "match": f"def f_{i}()"} for i in range(200)], indent=2)

    # Mock SmartCrusher
    crushed = data.replace("    ", "")  # simulate whitespace removal
    mock_cr = MagicMock()
    mock_cr.compressed = crushed
    mock_cr.was_modified = True

    mock_crusher = MagicMock()
    mock_crusher.crush.return_value = mock_cr

    with patch.object(comp, "_get_crusher", return_value=mock_crusher):
        result = comp.compress_tool_result(
            "search_files",
            data,
            config={"enabled": True, "min_chars": 0, "min_reduction_pct": 5},
        )

    assert result is not None
    assert "COMPRESSED" in result
    assert "smartcrusher" in result
    assert "smartcrusher" in result.lower()


def test_smartcrusher_exception_pass_through():
    """Exception in SmartCrusher.crush() → fall through to heuristic."""
    mock_crusher = MagicMock()
    mock_crusher.crush.side_effect = RuntimeError("crush error")

    # Build output that will trigger heuristic
    text = "\n".join([f"  {i}|def h_{i}(): pass" for i in range(500)])

    with patch.object(comp, "_get_crusher", return_value=mock_crusher):
        result = comp.compress_tool_result(
            "file_read",
            text,
            config={"enabled": True, "min_chars": 0, "min_reduction_pct": 10},
        )

    # Should fall through to heuristic and succeed
    assert result is not None
    assert "COMPRESSED" in result


# ---------------------------------------------------------------------------
# Heuristic text compression
# ---------------------------------------------------------------------------

def test_heuristic_numbered_lines():
    """Numbered lines (N|pattern) are detected and collapsed."""
    lines = [f"  {i}|def handler_{i}(): pass" for i in range(500)]
    text = "\n".join(lines)

    # Ensure SmartCrusher returns None (was_modified=False)
    mock_cr = MagicMock()
    mock_cr.was_modified = False
    mock_crusher = MagicMock()
    mock_crusher.crush.return_value = mock_cr

    with patch.object(comp, "_get_crusher", return_value=mock_crusher):
        result = comp.compress_tool_result(
            "file_read",
            text,
            config={"enabled": True, "min_chars": 0, "min_reduction_pct": 10},
        )

    assert result is not None
    assert "COMPRESSED" in result
    assert "heuristic" in result
    # Should have collapsed most lines
    assert len(result) < len(text) * 0.1  # at most 10% of original


def test_heuristic_uniform_prefix():
    """Uniform prefix lines are detected and collapsed."""
    # Ensure the first 30 chars are identical across all lines
    lines = [f"INFO  Processing item: status OK  padding_{i}" for i in range(200)]
    text = "\n".join(lines)

    mock_cr = MagicMock()
    mock_cr.was_modified = False
    mock_crusher = MagicMock()
    mock_crusher.crush.return_value = mock_cr

    with patch.object(comp, "_get_crusher", return_value=mock_crusher):
        result = comp.compress_tool_result(
            "execute_code",
            text,
            config={"enabled": True, "min_chars": 0, "min_reduction_pct": 10},
        )

    assert result is not None
    assert "similar lines omitted" in result


def test_heuristic_truncation_diverse_lines():
    """Diverse lines with >200 lines get head+tail truncation."""
    lines = []
    for i in range(300):
        lines.append(f"def unique_function_{i}(arg_a, arg_b):")
        lines.append(f"    # Implementation for function {i}")
        lines.append(f"    return compute({i}, arg_a) + process(arg_b)")
        lines.append("")
    text = "\n".join(lines)

    mock_cr = MagicMock()
    mock_cr.was_modified = False
    mock_crusher = MagicMock()
    mock_crusher.crush.return_value = mock_cr

    with patch.object(comp, "_get_crusher", return_value=mock_crusher):
        result = comp.compress_tool_result(
            "web_extract",
            text,
            config={"enabled": True, "min_chars": 0, "min_reduction_pct": 10},
        )

    assert result is not None
    assert "unique patterns" in result


# ---------------------------------------------------------------------------
# Error-line preservation
# ---------------------------------------------------------------------------

def test_error_lines_preserved_in_truncation():
    """Error lines in the middle are preserved even during truncation."""
    lines = []
    for i in range(250):
        if i == 125:
            lines.append("Traceback (most recent call last):")
            lines.append('  File "app.py", line 42, in main')
            lines.append("    ValueError: something went wrong")
        else:
            lines.append(f"Normal log line {i}")
    text = "\n".join(lines)

    mock_cr = MagicMock()
    mock_cr.was_modified = False
    mock_crusher = MagicMock()
    mock_crusher.crush.return_value = mock_cr

    with patch.object(comp, "_get_crusher", return_value=mock_crusher):
        result = comp.compress_tool_result(
            "execute_code",
            text,
            config={"enabled": True, "min_chars": 0, "min_reduction_pct": 10},
        )

    assert result is not None
    # Traceback should be preserved
    assert "Traceback" in result
    # Error line preservation marker
    assert "error lines preserved" in result


# ---------------------------------------------------------------------------
# Pass-through cases
# ---------------------------------------------------------------------------

def test_small_output_passed_through():
    """Small outputs skip compression entirely."""
    result = comp.compress_tool_result(
        "file_read",
        "Small content",
        config={"enabled": True, "min_chars": 2000},
    )
    assert result is None


def test_below_reduction_threshold_passed_through():
    """If compression saves less than min_reduction_pct, pass through."""
    # SmartCrusher that barely compresses
    mock_cr = MagicMock()
    mock_cr.compressed = "x" * 990  # 1% reduction on 1000 chars
    mock_cr.was_modified = True

    mock_crusher = MagicMock()
    mock_crusher.crush.return_value = mock_cr

    text = "x" * 1000
    with patch.object(comp, "_get_crusher", return_value=mock_crusher):
        result = comp.compress_tool_result(
            "test",
            text,
            config={"enabled": True, "min_chars": 0, "min_reduction_pct": 20},
        )

    assert result is None


def test_tool_exclusion():
    """Excluded tools skip compression."""
    result = comp.compress_tool_result(
        "memory_search",
        "x" * 5000,
        config={"enabled": True, "exclude_tools": ["memory_search"]},
    )
    assert result is None


# ---------------------------------------------------------------------------
# Cache operations
# ---------------------------------------------------------------------------

def test_cache_populated_after_compression():
    """Compressed results store originals in cache."""
    comp.clear_cache()
    text = "\n".join([f"  {i}|def h_{i}(): pass" for i in range(500)])

    mock_cr = MagicMock()
    mock_cr.was_modified = False
    mock_crusher = MagicMock()
    mock_crusher.crush.return_value = mock_cr

    with patch.object(comp, "_get_crusher", return_value=mock_crusher):
        result = comp.compress_tool_result(
            "file_read",
            text,
            config={"enabled": True, "min_chars": 0, "min_reduction_pct": 10},
        )

    assert result is not None
    # Extract hash from result
    assert "hash=" in result
    stats = comp.cache_stats()
    assert stats["cached_items"] >= 1


def test_cache_fifo_eviction():
    """_build_output evicts oldest entry (FIFO) when cache is full."""
    comp.clear_cache()
    # Pre-fill cache to max
    with comp._cache_lock:
        for i in range(50):
            comp._cache[f"h{i:03d}"] = f"original {i}"

    # Now _build_output should evict h000 (FIFO)
    out = comp._build_output("new original", "compressed", 50.0, "test")
    assert "h000" not in comp._cache
    # New hash should be present in the output
    assert "hash=" in out


def test_retrieve_cached():
    """retrieve_cached returns the stored original."""
    comp.clear_cache()
    with comp._cache_lock:
        comp._cache["abc123"] = "original content"

    result = comp.retrieve_cached("abc123")
    assert result == "original content"

    assert comp.retrieve_cached("nonexistent") is None


def test_clear_cache():
    """clear_cache empties the cache."""
    with comp._cache_lock:
        comp._cache["test"] = "data"
    comp.clear_cache()
    assert len(comp._cache) == 0


# ---------------------------------------------------------------------------
# Config caching
# ---------------------------------------------------------------------------

def test_config_cache_ttl():
    """Config is cached for _CONFIG_TTL seconds."""
    comp._config_cache = None

    call_count = 0

    def mock_load():
        nonlocal call_count
        call_count += 1
        return {"context": {"headroom": {"enabled": True}}}

    with patch("hermes_cli.config.load_config_readonly", mock_load):
        # First call: loads config
        comp._config()
        assert call_count == 1

        # Second call within TTL: uses cache
        comp._config()
        assert call_count == 1  # no additional load

        # Expire TTL
        comp._config_cache = (-999.0, {"enabled": True})
        comp._config()
        assert call_count == 2  # reload after TTL expired


def test_config_read_error_returns_empty():
    """If config read fails, empty dict is returned (fail-open)."""
    comp._config_cache = None
    with patch("hermes_cli.config.load_config_readonly", side_effect=RuntimeError("config error")):
        cfg = comp._config()
        assert cfg == {}


# ---------------------------------------------------------------------------
# _reset_state
# ---------------------------------------------------------------------------

def test_reset_state_clears_everything():
    """_reset_state clears cache, config cache, and crusher."""
    comp._config_cache = (0, {"enabled": True})
    comp._crusher = "fake_instance"
    with comp._cache_lock:
        comp._cache["test"] = "data"

    comp._reset_state()

    assert comp._config_cache is None
    assert comp._crusher is None
    assert len(comp._cache) == 0


# ---------------------------------------------------------------------------
# Build output format
# ---------------------------------------------------------------------------

def test_output_format():
    """Compressed output contains expected markers."""
    out = comp._build_output("original" * 100, "compressed", 50.0, "smartcrusher")
    assert "COMPRESSED by headroom" in out
    assert "smartcrusher" in out
    assert "50% saved" in out
    assert "hash=" in out
    assert "compressed" in out


def test_md5_usedforsecurity_false():
    """_build_output() uses usedforsecurity=False for FIPS compatibility."""
    import hashlib
    real_md5 = hashlib.md5
    seen_kwargs = {}

    def wrapped(*args, **kwargs):
        nonlocal seen_kwargs
        seen_kwargs = dict(kwargs)
        return real_md5(*args, **kwargs)

    with patch.object(hashlib, "md5", side_effect=wrapped):
        comp._build_output("original" * 10, "compressed", 50.0, "smartcrusher")
    assert seen_kwargs.get("usedforsecurity") is False
