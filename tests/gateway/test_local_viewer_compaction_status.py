"""Local viewers (Ink, Desktop, ``hermes tern``) get ``status.update {kind: "compacting"}`` only for a
real compaction. A warning that mentions compression drew a "Compacting context" divider for a
compaction that never ran."""
from __future__ import annotations

from types import SimpleNamespace

from gateway.config import Platform


def _local_turn(published):
    from gateway.run_turn_runner import TurnRunner
    turn = object.__new__(TurnRunner)
    turn._ctx = SimpleNamespace(
        source=SimpleNamespace(platform=Platform.LOCAL, chat_id='chat', chat_type='dm'),
        mute_notification_reply=False, _status_adapter=object(), _run_still_current=lambda: True,
        session_key='agent:main:local:dm:chat', run_generation=1, user_config={},
    )
    turn._runner = SimpleNamespace()
    turn._approval_owner = (SimpleNamespace(sessions={'s': SimpleNamespace()}), 's', 1)
    turn._publish_execution = lambda kind, payload: published.append((kind, payload)) or True
    return turn


def test_only_a_real_compaction_reaches_local_viewers_as_compacting():
    from agent.conversation_compression import COMPACTION_STATUS
    from gateway.warning_notifications import DiagnosticText

    published = []
    turn = _local_turn(published)
    notice = DiagnosticText("⚠ Configured auxiliary compression provider 'nous' is unavailable, so older "
                            "messages in long chats will be cut without a summary.")
    turn._status_callback_sync('lifecycle', notice)
    turn._status_callback_sync('lifecycle', COMPACTION_STATUS)

    assert [(kind, payload['kind']) for kind, payload in published] == [
        ('status.update', 'lifecycle'), ('status.update', 'compacting')]
    assert published[1][1]['text'] == COMPACTION_STATUS.strip()
