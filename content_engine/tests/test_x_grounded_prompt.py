"""Regression checks for the prompt actually sent by the scout."""
import importlib.util
from pathlib import Path


def scout():
    path = Path(__file__).resolve().parents[2] / 'scripts/content_engine/x_quote_scout.py'
    spec = importlib.util.spec_from_file_location('grounded_scout_test', path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_conversation_prompt_does_not_demand_essay_fields():
    mod = scout()
    assert 'all free-form replies/quotes' not in mod.LLM_SYSTEM
    assert 'Pack-free replies/quotes' not in mod.LLM_SYSTEM


def test_runtime_reads_approved_calibration(monkeypatch, tmp_path):
    mod = scout()
    skill = tmp_path / 'SKILL.md'
    skill.write_text('CURRENT VOICE RULES')
    refs = tmp_path / 'references'
    refs.mkdir()
    (refs / 'runtime-voice.md').write_text('CURRENT VOICE RULES')
    (refs / 'approved-conversational-calibration.md').write_text('APPROVED STYLE EXAMPLES')
    monkeypatch.setattr(mod, 'VOICE_SKILL', skill)
    monkeypatch.setattr(mod, '_approved_voice', lambda runtime: 'OWN HISTORICAL POSTS')
    result = mod._runtime_voice()
    assert 'CURRENT VOICE RULES' in result
    assert 'APPROVED STYLE EXAMPLES' in result
    assert 'OWN HISTORICAL POSTS' in result
