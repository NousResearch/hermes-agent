"""Tests for ModelSwitchMixin -- model switching/client methods extracted from AIAgent."""

import pytest


class TestModelSwitchMixinImport:
    def test_mixin_importable(self):
        from agent.model_switch import ModelSwitchMixin
        assert ModelSwitchMixin is not None

    def test_has_switch_model(self):
        from agent.model_switch import ModelSwitchMixin
        assert hasattr(ModelSwitchMixin, 'switch_model')

    def test_has_convert_to_trajectory_format(self):
        from agent.model_switch import ModelSwitchMixin
        assert hasattr(ModelSwitchMixin, '_convert_to_trajectory_format')

    def test_has_create_openai_client(self):
        # _create_openai_client remains in run_agent.py for monkeypatch compat
        import run_agent
        assert hasattr(run_agent.AIAgent, '_create_openai_client')

    def test_has_save_trajectory(self):
        from agent.model_switch import ModelSwitchMixin
        assert hasattr(ModelSwitchMixin, '_save_trajectory')

    def test_has_close_openai_client(self):
        from agent.model_switch import ModelSwitchMixin
        assert hasattr(ModelSwitchMixin, '_close_openai_client')

    def test_has_openai_client_lock(self):
        from agent.model_switch import ModelSwitchMixin
        assert hasattr(ModelSwitchMixin, '_openai_client_lock')

    def test_has_ensure_primary_openai_client(self):
        from agent.model_switch import ModelSwitchMixin
        assert hasattr(ModelSwitchMixin, '_ensure_primary_openai_client')

    def test_has_replace_primary_openai_client(self):
        from agent.model_switch import ModelSwitchMixin
        assert hasattr(ModelSwitchMixin, '_replace_primary_openai_client')

    def test_has_create_request_openai_client(self):
        from agent.model_switch import ModelSwitchMixin
        assert hasattr(ModelSwitchMixin, '_create_request_openai_client')

    def test_has_is_openai_client_closed(self):
        from agent.model_switch import ModelSwitchMixin
        assert hasattr(ModelSwitchMixin, '_is_openai_client_closed')
