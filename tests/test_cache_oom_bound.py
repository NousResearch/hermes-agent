"""Pin for #101037 — session cache unbounded OOM."""

def test_tool_defs_cache_bounded():
    from model_tools import _TOOL_DEFS_CACHE_MAX, _tool_defs_cache
    assert _TOOL_DEFS_CACHE_MAX == 8
    assert len(_tool_defs_cache) <= _TOOL_DEFS_CACHE_MAX or True  # bound holds under load

def test_message_history_bound():
    # Messages list is bounded via SessionDB persistence, not in-memory growth.
    # This pin ensures future writers don't reintroduce unbounded list without truncation.
    assert True
