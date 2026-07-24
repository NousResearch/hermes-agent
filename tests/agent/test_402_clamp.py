
import pytest
from agent.conversation_loop import _affordable_clamp

def test_affordable_clamp_first_attempt_decreases_cap():
    # cur_cap=None, affordable=1000, attempts=0, max=1
    retry, safe_out = _affordable_clamp(None, 1000, 0, 1)
    assert retry is True
    assert safe_out == 936

def test_affordable_clamp_strictly_lowers_cap():
    # cur_cap=900, affordable=1000 (safe_out=936), attempts=0, max=1
    # 936 is NOT < 900, so helps=False
    retry, safe_out = _affordable_clamp(900, 1000, 0, 1)
    assert retry is False

def test_affordable_clamp_exhausts_attempts():
    # cur_cap=None, affordable=1000, attempts=1, max=1
    retry, safe_out = _affordable_clamp(None, 1000, 1, 1)
    assert retry is False

def test_affordable_clamp_floor():
    # affordable=10 -> safe_out=1
    retry, safe_out = _affordable_clamp(None, 10, 0, 1)
    assert safe_out == 1
