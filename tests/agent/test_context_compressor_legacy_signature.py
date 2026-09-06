import pytest
from unittest.mock import MagicMock
from agent.context_compressor import ContextCompressor

def test_context_compressor_legacy_signature_compatibility():
    """Verify that a legacy _generate_summary override without bypass_cooldown doesn't crash."""
    
    class LegacyEngine(ContextCompressor):
        # Override with a legacy signature (no bypass_cooldown or kwargs)
        def _generate_summary(self, turns_to_summarize, focus_topic=None, memory_context=""):
            return "Legacy summary"
            
    engine = LegacyEngine(model="test-model", token_budget=1000)
    
    # Mock methods to force it to actually attempt compression and call _generate_summary
    engine._begin_compress_attempt = MagicMock(return_value={})
    engine._protect_head_size = MagicMock(return_value=1)
    engine._prune_old_tool_results = MagicMock(return_value=(['dummy']*10, 0))
    engine._drop_blank_echoes = MagicMock(return_value=['dummy']*10)
    engine._compress_window = MagicMock(return_value=(1, 5))
    
    scan_mock = MagicMock()
    scan_mock.turns_to_summarize = ['dummy']*4
    scan_mock.tail_start = 5
    scan_mock.previous_summary_before = None
    scan_mock.has_user_turn_before = False
    scan_mock.summary_indices = []
    engine._scan_window_handoffs = MagicMock(return_value=scan_mock)
    
    engine._feasibility_skip = MagicMock(return_value=False)
    engine._derive_auto_focus_topic = MagicMock(return_value=None)
    engine._assemble_compressed = MagicMock(return_value=[])
    engine._finalize_compressed = MagicMock(return_value=[])
    
    # Run compress with bypass_cooldown=True
    # It should NOT throw TypeError: _generate_summary() got an unexpected keyword argument 'bypass_cooldown'
    result = engine.compress(['dummy']*10, current_tokens=500, bypass_cooldown=True)
    
    assert engine._assemble_compressed.called
    # The summary passed to assemble_compressed should be "Legacy summary"
    assert engine._assemble_compressed.call_args[0][4] == "Legacy summary"
