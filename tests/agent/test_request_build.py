"""Tests for RequestBuildMixin -- request build/lifecycle methods."""

import pytest


class TestRequestBuildMixinImport:
    def test_mixin_importable(self):
        from agent.request_build import RequestBuildMixin
        assert RequestBuildMixin is not None

    def test_has_build_api_kwargs(self):
        from agent.request_build import RequestBuildMixin
        assert hasattr(RequestBuildMixin, '_build_api_kwargs')

    def test_has_build_system_prompt(self):
        from agent.request_build import RequestBuildMixin
        assert hasattr(RequestBuildMixin, '_build_system_prompt')

    def test_has_build_assistant_message(self):
        from agent.request_build import RequestBuildMixin
        assert hasattr(RequestBuildMixin, '_build_assistant_message')

    def test_has_format_tools_for_system_message(self):
        from agent.request_build import RequestBuildMixin
        assert hasattr(RequestBuildMixin, '_format_tools_for_system_message')

    def test_has_dump_api_request_debug(self):
        from agent.request_build import RequestBuildMixin
        assert hasattr(RequestBuildMixin, '_dump_api_request_debug')

    def test_has_spawn_background_review(self):
        from agent.request_build import RequestBuildMixin
        assert hasattr(RequestBuildMixin, '_spawn_background_review')

    def test_has_summarize_background_review_actions(self):
        from agent.request_build import RequestBuildMixin
        assert hasattr(RequestBuildMixin, '_summarize_background_review_actions')

    def test_has_close(self):
        from agent.request_build import RequestBuildMixin
        assert hasattr(RequestBuildMixin, 'close')

    def test_has_release_clients(self):
        from agent.request_build import RequestBuildMixin
        assert hasattr(RequestBuildMixin, 'release_clients')

    def test_has_looks_like_codex_intermediate_ack(self):
        from agent.request_build import RequestBuildMixin
        assert hasattr(RequestBuildMixin, '_looks_like_codex_intermediate_ack')

    def test_has_summarize_api_error(self):
        from agent.request_build import RequestBuildMixin
        assert hasattr(RequestBuildMixin, '_summarize_api_error')

    def test_has_get_messages_up_to_last_assistant(self):
        from agent.request_build import RequestBuildMixin
        assert hasattr(RequestBuildMixin, '_get_messages_up_to_last_assistant')

    def test_has_api_kwargs_have_image_parts(self):
        from agent.request_build import RequestBuildMixin
        assert hasattr(RequestBuildMixin, '_api_kwargs_have_image_parts')

    def test_has_invalidate_system_prompt(self):
        from agent.request_build import RequestBuildMixin
        assert hasattr(RequestBuildMixin, '_invalidate_system_prompt')

    def test_has_base_url(self):
        from agent.request_build import RequestBuildMixin
        assert hasattr(RequestBuildMixin, 'base_url')
