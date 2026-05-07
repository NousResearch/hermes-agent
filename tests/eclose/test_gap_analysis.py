import pytest
from eclose.evolution.gap_analysis import GapAnalysisEngine
from eclose.events.events import GapType, Severity, GapEvent, EventType

def test_gap_analysis_initialization():
    engine = GapAnalysisEngine()
    assert engine is not None

def test_identify_capability_gap():
    engine = GapAnalysisEngine()
    gaps = engine.identify_gaps(
        needs=["video_processing"],
        capabilities=["code_generation", "file_operations"],
    )
    assert len(gaps) > 0
    assert any(g.gap_type == GapType.SKILL for g in gaps)