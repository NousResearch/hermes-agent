"""Tests for repeated tool-call loop detection."""

from __future__ import annotations


def test_loop_detector_flags_repeated_identical_tool_calls():
    from agent.loop_detector import ToolLoopDetector

    detector = ToolLoopDetector(max_window=5, repeat_threshold=3)
    assert detector.record("web_search", {"query": "self improving agents"}) is None
    assert detector.record("web_search", {"query": "self improving agents"}) is None
    warning = detector.record("web_search", {"query": "self improving agents"})

    assert warning is not None
    assert warning["kind"] == "repeated_tool_call"
    assert warning["tool_name"] == "web_search"
    assert "switch strategy" in warning["recommendation"].lower()


def test_loop_detector_flags_similar_search_queries_after_low_information_results():
    from agent.loop_detector import ToolLoopDetector

    detector = ToolLoopDetector(max_window=6, repeat_threshold=3, similarity_threshold=0.6)
    detector.record("web_search", {"query": "agent memory benchmark"}, result_summary="")
    detector.record("web_search", {"query": "agent memory benchmarks"}, result_summary="no results")
    warning = detector.record("web_search", {"query": "llm agent memory benchmark"}, result_summary="no results")

    assert warning is not None
    assert warning["kind"] == "low_information_loop"


def test_loop_detector_does_not_flag_different_tools_or_queries():
    from agent.loop_detector import ToolLoopDetector

    detector = ToolLoopDetector(max_window=5, repeat_threshold=3)
    detector.record("web_search", {"query": "agent memory"})
    detector.record("web_extract", {"urls": ["https://example.com"]})
    warning = detector.record("search_files", {"pattern": "memory"})

    assert warning is None
