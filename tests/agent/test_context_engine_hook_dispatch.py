"""Per-turn context-engine hooks reach every engine that may implement them.

``select_context`` and ``on_turn_complete`` are skipped only for an in-process engine that inherits the
ABC no-ops. A hosted engine (``plugins.isolation: host``) is always dispatched: the plugin host looks the
method up on the plugin's object at call time, so a hook the engine installs after load (for example
in ``on_session_start()``) must still run, and the description the host sent at load cannot say so.
"""
from tests.agent.test_context_engine_compaction_authority import _hosted


def _compress_only():
    from agent.context_engine import ContextEngine

    class _CompressOnly(ContextEngine):
        @property
        def name(self) -> str:
            return "compress-only"

        def update_from_response(self, usage):
            return None

        def should_compress(self, prompt_tokens=None):
            return False

        def compress(self, messages, current_tokens=None, focus_topic=None, force=False, memory_context=""):
            return messages

    return _CompressOnly


def test_a_hook_a_hosted_engine_installs_after_load_is_still_dispatched(tmp_path):
    from agent.conversation_loop import _engine_overrides_hook

    plugin_object = _compress_only()()
    hosted = _hosted(plugin_object, tmp_path)
    plugin_object.select_context = lambda request_messages, **kwargs: request_messages
    plugin_object.on_turn_complete = lambda messages, usage=None, **kwargs: None

    assert _engine_overrides_hook(hosted, "select_context") is True
    assert _engine_overrides_hook(hosted, "on_turn_complete") is True


def test_an_in_process_engine_that_only_inherits_the_hooks_is_skipped():
    from agent.conversation_loop import _engine_overrides_hook

    engine = _compress_only()()

    assert _engine_overrides_hook(engine, "select_context") is False
    assert _engine_overrides_hook(engine, "on_turn_complete") is False
    engine.on_turn_complete = lambda messages, usage=None, **kwargs: None
    assert _engine_overrides_hook(engine, "on_turn_complete") is True
