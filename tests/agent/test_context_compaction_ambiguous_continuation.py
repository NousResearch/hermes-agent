from agent.context_compressor import SUMMARY_PREFIX


def test_summary_prefix_blocks_ambiguous_continuation_from_historical_sections():
    assert "ambiguous continuation cue" in SUMMARY_PREFIX
    assert "давай" in SUMMARY_PREFIX
    assert "продолжай" in SUMMARY_PREFIX
    assert "do NOT infer the target from Historical sections" in SUMMARY_PREFIX
    assert "session_search hits" in SUMMARY_PREFIX
