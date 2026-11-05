"""Per-turn context-engine hooks are dispatched only to engines that implement them.

``select_context`` runs on every provider request and ``on_turn_complete`` after every turn, so an
engine that only inherits the ABC no-ops must not be called. Under ``plugins.isolation: host`` each
call is a round trip to the plugin host, and every method of the proxy looks overridden to an
identity check.
"""
from tests.agent.test_context_engine_compaction_authority import _hosted


def _engines():
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

    class _Observer(_CompressOnly):
        compacts_outside_compress = False

        def select_context(self, request_messages, **kwargs):
            return request_messages

        def on_turn_complete(self, messages, usage=None, **kwargs):
            return None

    return _CompressOnly, _Observer


def test_per_turn_hooks_follow_what_the_engine_implements(tmp_path):
    from agent.conversation_loop import _engine_overrides_hook

    compress_only, observer = _engines()
    for hook in ("select_context", "on_turn_complete"):
        assert _engine_overrides_hook(compress_only(), hook) is False
        assert _engine_overrides_hook(observer(), hook) is True
        assert _engine_overrides_hook(_hosted(compress_only(), tmp_path), hook) is False
        assert _engine_overrides_hook(_hosted(observer(), tmp_path), hook) is True
