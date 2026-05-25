from pathlib import Path


def test_conversation_loop_imports_pool_recovery_helper():
    text = Path('agent/conversation_loop.py').read_text(encoding='utf-8')
    assert '_pool_may_recover_from_rate_limit(' in text
    assert 'from run_agent import _pool_may_recover_from_rate_limit' in text
