"""Seam identity for context_compressor_text_utils leaf extract (LB-textutil).

Part of #78645 + #78647.
"""

from agent import context_compressor as cc
from agent import context_compressor_text_utils as tu


def test_redact_and_content_text_resolve_is_identical_through_godfile():
    assert cc._redact_compaction_text is tu._redact_compaction_text
    assert cc._content_text_for_contains is tu._content_text_for_contains


def test_no_duplicate_defs_in_godfile():
    import inspect
    from pathlib import Path

    src = Path(cc.__file__).read_text(encoding="utf-8")
    assert src.count("def _redact_compaction_text") == 0
    assert src.count("def _content_text_for_contains") == 0
    assert "context_compressor_text_utils" in src


def test_redact_behavior_smoke():
    # None-safety preserved
    assert cc._redact_compaction_text(None) == ""
    # content text view
    assert cc._content_text_for_contains(None) == ""
    assert cc._content_text_for_contains("hi") == "hi"
    assert cc._content_text_for_contains([{"type": "text", "text": "a"}, "b"]) == "a\nb"


def test_import_orders_no_cycle():
    import importlib
    import agent.context_compressor_text_utils as a
    import agent.context_compressor as b

    importlib.reload(a)
    importlib.reload(b)
    assert b._redact_compaction_text is a._redact_compaction_text
