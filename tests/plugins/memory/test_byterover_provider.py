"""ByteRover pre-compression flush strips slash-skill scaffolding.

``on_pre_compress`` reads the raw conversation about to be compressed and curates
it into the ByteRover knowledge tree. A ``/skill`` turn expands into a model-facing
message that embeds the full skill body, so the hook must strip the scaffolding
(like ``MemoryManager`` does for the prefetch/sync fan-out, #47311) before
curating — otherwise the skill body is persisted as "memory".
"""

import plugins.memory.byterover as byterover
from plugins.memory.byterover import ByteRoverMemoryProvider

_SKILL_SCAFFOLDING_TURN = (
    '[IMPORTANT: The user has invoked the "skill-creator" skill, indicating they want '
    "you to follow its instructions. The full skill content is loaded below.]\n\n"
    "# Skill Creator\n\n"
    "Large skill body that must not be curated or persisted.\n\n"
    "The user has provided the following instruction alongside the skill invocation: "
    "make a skill for release triage"
)
_BARE_SKILL_TURN = (
    '[IMPORTANT: The user has invoked the "skill-creator" skill, indicating they want '
    "you to follow its instructions. The full skill content is loaded below.]\n\n"
    "# Skill Creator\n\n"
    "Large skill body, no user instruction."
)


class _SyncThread:
    """Run the daemon flush inline so the curate call is observable in-test."""

    def __init__(self, target=None, daemon=None, name=None, **kwargs):
        self._target = target

    def start(self):
        if self._target:
            self._target()


def test_on_pre_compress_strips_skill_scaffolding(monkeypatch):
    captured = []

    def fake_run_brv(args, timeout=None, cwd=None):
        captured.append(args)
        return {"success": True, "output": ""}

    monkeypatch.setattr(byterover, "_run_brv", fake_run_brv)
    monkeypatch.setattr(byterover.threading, "Thread", _SyncThread)

    provider = ByteRoverMemoryProvider()
    provider.on_pre_compress(
        [
            {"role": "user", "content": _SKILL_SCAFFOLDING_TURN},
            {"role": "assistant", "content": "done"},
        ]
    )

    assert len(captured) == 1
    curated = captured[0][-1]  # ["curate", "--", "<combined text>"]
    assert "make a skill for release triage" in curated
    assert "Large skill body" not in curated


def test_on_pre_compress_drops_bare_skill_turn(monkeypatch):
    captured = []

    def fake_run_brv(args, timeout=None, cwd=None):
        captured.append(args)
        return {"success": True, "output": ""}

    monkeypatch.setattr(byterover, "_run_brv", fake_run_brv)
    monkeypatch.setattr(byterover.threading, "Thread", _SyncThread)

    provider = ByteRoverMemoryProvider()
    provider.on_pre_compress(
        [
            {"role": "user", "content": "what is the weather"},
            {"role": "assistant", "content": "it is sunny"},
            {"role": "user", "content": _BARE_SKILL_TURN},
            {"role": "assistant", "content": "loaded the skill"},
        ]
    )

    assert len(captured) == 1
    curated = captured[0][-1]
    assert "what is the weather" in curated
    # A bare /skill turn is dropped whole: neither its skill body nor the
    # assistant reply that followed it ("loaded the skill") gets curated.
    assert "Large skill body" not in curated
    assert "loaded the skill" not in curated
