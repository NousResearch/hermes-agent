"""MemoryManager strips slash-skill scaffolding for every provider.

When a user invokes a /skill or /bundle, Hermes expands the turn into a
model-facing message that embeds the full skill body. Feeding that verbatim to
memory providers pollutes their stores/embeddings with prompt scaffolding
instead of what the user actually asked. The strip lives once in MemoryManager
so it covers the whole provider fan-out — not per backend.

See: agent.skill_commands.extract_user_instruction_from_skill_message and
MemoryManager._strip_skill_scaffolding.
"""

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider
from agent.skill_commands import extract_user_instruction_from_skill_message


_SINGLE_SKILL_TURN = (
    '[IMPORTANT: The user has invoked the "skill-creator" skill, indicating they want '
    "you to follow its instructions. The full skill content is loaded below.]\n\n"
    "# Skill Creator\n\n"
    "Large skill body that must not be searched or embedded.\n\n"
    "The user has provided the following instruction alongside the skill invocation: "
    "make a skill for release triage"
)

_BUNDLE_TURN = (
    '[IMPORTANT: The user has invoked the "backend-dev" skill bundle, '
    "loading 2 skills together. Treat every skill below as active guidance for this turn.]\n\n"
    "Bundle: backend-dev\n"
    "Skills loaded: test-driven-development, code-review\n\n"
    "User instruction: fix the failing retrieval test\n\n"
    '[Loaded as part of the "backend-dev" skill bundle.]\n\n'
    "Large bundled skill body that must not be searched or embedded."
)

_BARE_SKILL_TURN = (
    '[IMPORTANT: The user has invoked the "skill-creator" skill, indicating they want '
    "you to follow its instructions. The full skill content is loaded below.]\n\n"
    "# Skill Creator\n\n"
    "Large skill body, no user instruction."
)


class _RecordingProvider(MemoryProvider):
    """Captures exactly what user text each fan-out method received."""

    _name = "recording"

    def __init__(self):
        self.prefetched = []
        self.queued = []
        self.synced = []

    @property
    def name(self) -> str:
        return self._name

    def initialize(self, session_id: str = "", **kwargs) -> None:
        pass

    def is_available(self) -> bool:
        return True

    def system_prompt_block(self) -> str:
        return ""

    def prefetch(self, query, *, session_id: str = "") -> str:
        self.prefetched.append(query)
        return ""

    def queue_prefetch(self, query, *, session_id: str = "") -> None:
        self.queued.append(query)

    def sync_turn(self, user_content, assistant_content, *, session_id: str = "", messages=None) -> None:
        self.synced.append(user_content)

    def get_tool_schemas(self):
        return []


def _manager_with_recorder():
    mgr = MemoryManager()
    provider = _RecordingProvider()
    mgr.add_provider(provider)
    return mgr, provider


class TestExtractUserInstruction:
    def test_non_string_returns_none(self):
        assert extract_user_instruction_from_skill_message(None) is None
        assert extract_user_instruction_from_skill_message(123) is None
        assert extract_user_instruction_from_skill_message([{"text": "hi"}]) is None



    def test_bundle_with_instruction(self):
        assert (
            extract_user_instruction_from_skill_message(_BUNDLE_TURN)
            == "fix the failing retrieval test"
        )




class TestMemoryManagerStripsScaffolding:

    def test_prefetch_all_skips_bare_skill(self):
        mgr, provider = _manager_with_recorder()
        result = mgr.prefetch_all(_BARE_SKILL_TURN)
        assert result == ""
        assert provider.prefetched == []

    def test_queue_prefetch_all_strips_bundle(self):
        mgr, provider = _manager_with_recorder()
        mgr.queue_prefetch_all(_BUNDLE_TURN)
        mgr.flush_pending(timeout=5.0)
        assert provider.queued == ["fix the failing retrieval test"]



    def test_sync_all_skips_bare_skill(self):
        mgr, provider = _manager_with_recorder()
        mgr.sync_all(_BARE_SKILL_TURN, "Done.")
        mgr.flush_pending(timeout=5.0)
        assert provider.synced == []


# ---------------------------------------------------------------------------
# Gateway channel/topic auto-loaded turns (#92036)
# ---------------------------------------------------------------------------

_AUTO_LOAD_NOTE = (
    '[IMPORTANT: The "example-skill" skill is auto-loaded. '
    "Follow its instructions for this session.]"
)

_INSTRUCTION_MARKER = (
    "The user has provided the following instruction alongside the skill invocation: "
)

_AUTO_LOAD_SINGLE = (
    f"{_AUTO_LOAD_NOTE}\n\n"
    "# Example Skill\n\n"
    "Framework-injected procedure text that must never reach memory.\n\n"
    f"{_INSTRUCTION_MARKER}remember only this sentence"
)

_AUTO_LOAD_MULTI = (
    f"{_AUTO_LOAD_NOTE}\n\n"
    "# Example Skill\n\n"
    "First injected body.\n\n"
    f'{_AUTO_LOAD_NOTE.replace("example-skill", "second-skill")}\n\n'
    "# Second Skill\n\n"
    "Second injected body.\n\n"
    f"{_INSTRUCTION_MARKER}remember only this sentence"
)

_BARE_AUTO_LOAD = (
    f"{_AUTO_LOAD_NOTE}\n\n# Example Skill\n\nInjected body, no user text."
)


class TestGatewayAutoLoadScaffolding:
    def test_single_auto_load_extraction(self):
        assert (
            extract_user_instruction_from_skill_message(_AUTO_LOAD_SINGLE)
            == "remember only this sentence"
        )

    def test_multi_auto_load_extraction(self):
        assert (
            extract_user_instruction_from_skill_message(_AUTO_LOAD_MULTI)
            == "remember only this sentence"
        )

    def test_bare_auto_load_returns_none(self):
        assert extract_user_instruction_from_skill_message(_BARE_AUTO_LOAD) is None

    def test_ordinary_message_unchanged(self):
        assert (
            extract_user_instruction_from_skill_message("remember only this sentence")
            == "remember only this sentence"
        )

    def test_prefetch_all_strips_auto_load(self):
        mgr, provider = _manager_with_recorder()
        mgr.prefetch_all(_AUTO_LOAD_SINGLE)
        assert provider.prefetched == ["remember only this sentence"]

    def test_sync_all_strips_auto_load(self):
        mgr, provider = _manager_with_recorder()
        mgr.sync_all(_AUTO_LOAD_SINGLE, "Done.")
        mgr.flush_pending(timeout=5.0)
        assert provider.synced == ["remember only this sentence"]

    def test_sync_all_skips_bare_auto_load(self):
        mgr, provider = _manager_with_recorder()
        mgr.sync_all(_BARE_AUTO_LOAD, "Done.")
        mgr.flush_pending(timeout=5.0)
        assert provider.synced == []

