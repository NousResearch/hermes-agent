"""Compaction recovery pointers (summary footer, demoted-tail stub) name a session_search call that
reaches the compacted rows."""


class TestRecoveryFooterRoleScope:
    """Recovery pointers must include tool output.

    Search defaults to user/assistant rows, so the tool demotion path needs a
    tool-only search while the general footer needs all roles and full result
    hydration.
    """

    def test_recovery_footer_searches_all_roles_with_full_detail(self):
        from agent.context_compressor import _build_recovery_footer

        footer = _build_recovery_footer("sess-123", region_len=10)
        assert (
            "session_search(query='<keywords>', session_id='sess-123', "
            "role_filter='user,assistant,tool', detail='full')"
        ) in footer

    def test_recovery_footer_empty_without_session(self):
        from agent.context_compressor import _build_recovery_footer

        assert _build_recovery_footer("", region_len=10) == ""

    def test_lean_recovery_stub_searches_tool_output_with_full_detail(self):
        from agent.context_compressor_tail import _lean_recovery_stub

        stub = _lean_recovery_stub("web_extract", 2500, "sess-123")
        assert (
            "session_search(query=..., session_id='sess-123', role_filter='tool', detail='full')"
        ) in stub
