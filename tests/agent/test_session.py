"""Tests for SessionMixin -- session management methods extracted from AIAgent."""

import pytest


class TestSessionMixinImport:
    def test_mixin_importable(self):
        from agent.session import SessionMixin
        assert SessionMixin is not None

    def test_has_get_session_db(self):
        from agent.session import SessionMixin
        assert hasattr(SessionMixin, '_get_session_db_for_recall')

    def test_has_ensure_db_session(self):
        from agent.session import SessionMixin
        assert hasattr(SessionMixin, '_ensure_db_session')

    def test_has_reset_session_state(self):
        from agent.session import SessionMixin
        assert hasattr(SessionMixin, 'reset_session_state')

    def test_has_cleanup_task_resources(self):
        from agent.session import SessionMixin
        assert hasattr(SessionMixin, '_cleanup_task_resources')

    def test_has_flush_messages(self):
        from agent.session import SessionMixin
        assert hasattr(SessionMixin, '_flush_messages_to_session_db')

    def test_has_save_session_log(self):
        from agent.session import SessionMixin
        assert hasattr(SessionMixin, '_save_session_log')

    def test_has_commit_memory_session(self):
        from agent.session import SessionMixin
        assert hasattr(SessionMixin, 'commit_memory_session')

    def test_has_cleanup_dead_connections(self):
        from agent.session import SessionMixin
        assert hasattr(SessionMixin, '_cleanup_dead_connections')

    def test_has_persist_user_message_override(self):
        from agent.session import SessionMixin
        assert hasattr(SessionMixin, '_apply_persist_user_message_override')
