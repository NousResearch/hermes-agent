"""Gateway ``/rollback`` must undo the conversation turn it just reverted.

``website/docs/user-guide/checkpoints-and-rollback.md`` documents ``/rollback
<N>`` as a four-step flow, ending with:

    4. **Undoes the last conversation turn** so the agent's context matches
       the restored filesystem state.

The CLI handler honours step 4 (``hermes_cli/cli_commands_mixin.py`` calls
``undo_last(prefill=False)`` after a successful restore).  The gateway handler
restored the files and stopped there, so on every messaging platform the
cached agent kept a transcript describing edits that no longer existed on
disk.  These tests drive the real handler against a real checkpoint store.
"""

import shutil

import pytest

import gateway.run as gateway_run
import tools.checkpoint_manager as cpm
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git required for /rollback"
)

BASELINE = "VERSION = 1\n"
EDITED = "VERSION = 2  # written by the agent\n"


class _SessionEntry:
    def __init__(self):
        self.session_id = "sess-1"
        self.last_prompt_tokens = 4321


class _SessionStore:
    """Records what the handler asks of the session store."""

    def __init__(self, rewind_result=None, rewind_error=None):
        self.entry = _SessionEntry()
        self.rewind_calls = []
        self._rewind_result = rewind_result
        self._rewind_error = rewind_error

    async def get_or_create_session(self, source):
        return self.entry

    async def rewind_session(self, session_id, n):
        self.rewind_calls.append((session_id, n))
        if self._rewind_error is not None:
            raise self._rewind_error
        return self._rewind_result


def _rewind_payload(turns=1):
    return {
        "target_text": "bump the version",
        "turns_undone": turns,
        "rewound_count": turns * 2,
    }


class _Runner(gateway_run.GatewayRunner):
    """A real runner with only the session-store facade swapped out.

    ``GatewayRunner.async_session_store`` is a read-only property, so the
    recorder is injected by overriding it rather than by assignment.
    """

    @property
    def async_session_store(self):
        return self._store_stub


def _runner(store):
    runner = object.__new__(_Runner)
    runner.config = None
    runner.session_store = None
    runner._store_stub = store
    runner.evicted = []
    runner._evict_cached_agent = runner.evicted.append
    return runner


def _event(text: str) -> MessageEvent:
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="user-1",
        chat_id="chat-1",
        user_name="tester",
        chat_type="dm",
    )
    return MessageEvent(text=text, source=source)


@pytest.fixture()
def project(tmp_path, monkeypatch):
    """A checkpointed working directory with one restorable baseline."""
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "checkpoints:\n  enabled: true\n", encoding="utf-8"
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", home, raising=False)
    monkeypatch.setattr(cpm, "CHECKPOINT_BASE", tmp_path / "checkpoints")

    d = tmp_path / "project"
    d.mkdir()
    app = d / "app.py"
    app.write_text(BASELINE, encoding="utf-8")
    monkeypatch.setenv("TERMINAL_CWD", str(d))

    mgr = cpm.CheckpointManager(enabled=True, max_snapshots=50)
    assert mgr.ensure_checkpoint(str(d), "before the agent edited app.py") is True

    # The agent's turn: it edits the file and the transcript now says so.
    app.write_text(EDITED, encoding="utf-8")
    return app


@pytest.mark.asyncio
async def test_rollback_restores_files_and_rewinds_transcript(project):
    store = _SessionStore(rewind_result=_rewind_payload())
    runner = _runner(store)

    reply = await runner._handle_rollback_command(_event("/rollback 1"))

    assert "Restored to checkpoint" in reply
    assert project.read_text(encoding="utf-8") == BASELINE
    # Step 4: exactly one turn dropped, for this session.
    assert store.rewind_calls == [("sess-1", 1)]
    # …and the session re-armed so the next message rebuilds from the
    # truncated transcript instead of the stale cached agent.
    assert store.entry.last_prompt_tokens == 0
    assert len(runner.evicted) == 1


@pytest.mark.asyncio
async def test_listing_checkpoints_leaves_the_session_untouched(project):
    store = _SessionStore(rewind_result=_rewind_payload())
    runner = _runner(store)

    await runner._handle_rollback_command(_event("/rollback"))

    assert store.rewind_calls == []
    assert store.entry.last_prompt_tokens == 4321
    assert runner.evicted == []
    assert project.read_text(encoding="utf-8") == EDITED


@pytest.mark.asyncio
async def test_failed_restore_does_not_drop_a_turn(project):
    store = _SessionStore(rewind_result=_rewind_payload())
    runner = _runner(store)

    reply = await runner._handle_rollback_command(
        _event("/rollback 0123456789abcdef0123456789abcdef01234567")
    )

    assert "Restored to checkpoint" not in reply
    # Nothing was reverted, so the transcript still matches the filesystem.
    assert store.rewind_calls == []
    assert project.read_text(encoding="utf-8") == EDITED


@pytest.mark.asyncio
async def test_restore_still_reported_when_the_rewind_blows_up(project):
    store = _SessionStore(rewind_error=RuntimeError("state.db is locked"))
    runner = _runner(store)

    reply = await runner._handle_rollback_command(_event("/rollback 1"))

    # The files really were restored; a bookkeeping failure afterwards must
    # not tell the user the rollback failed.
    assert "Restored to checkpoint" in reply
    assert project.read_text(encoding="utf-8") == BASELINE


@pytest.mark.asyncio
async def test_rollback_succeeds_when_there_is_no_turn_to_rewind(project):
    store = _SessionStore(rewind_result=None)
    runner = _runner(store)

    reply = await runner._handle_rollback_command(_event("/rollback 1"))

    assert "Restored to checkpoint" in reply
    assert project.read_text(encoding="utf-8") == BASELINE
    assert store.rewind_calls == [("sess-1", 1)]
    # Nothing was truncated, so the cached token count stays as it was.
    assert store.entry.last_prompt_tokens == 4321
    assert runner.evicted == []


@pytest.mark.asyncio
async def test_undo_still_rewinds_the_requested_number_of_turns():
    """/undo shares the rewind primitive with /rollback — keep it working."""
    store = _SessionStore(rewind_result=_rewind_payload(turns=3))
    runner = _runner(store)

    reply = await runner._handle_undo_command(_event("/undo 3"))

    assert store.rewind_calls == [("sess-1", 3)]
    assert store.entry.last_prompt_tokens == 0
    assert len(runner.evicted) == 1
    assert "bump the version" in reply


@pytest.mark.asyncio
async def test_undo_reports_nothing_to_rewind():
    store = _SessionStore(rewind_result=None)
    runner = _runner(store)

    reply = await runner._handle_undo_command(_event("/undo"))

    assert store.rewind_calls == [("sess-1", 1)]
    assert store.entry.last_prompt_tokens == 4321
    assert runner.evicted == []
    assert reply
