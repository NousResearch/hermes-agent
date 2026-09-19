"""Privacy boundary for a real live-row copy surviving its Files scope.

A snapshot consumer may retain a deep copy during the provider call. Later
trajectory export must project or refuse it, not emit private content merely
because the originating turn's identity-bound override has been reset.
"""
import copy
import json

import pytest

from hermes_state import SessionDB
from agent.session_persistence import files_user_message_persistence
from tests.agent.files_persistence_fixtures import inert_agent


@pytest.mark.parametrize('native', [False, True])
def test_files_live_snapshot_trajectory_after_scope_is_private(tmp_path, monkeypatch, native, record_property):
    monkeypatch.chdir(tmp_path)
    db = SessionDB(tmp_path / 'snapshot.db')
    try:
        agent, sent, _ = inert_agent(monkeypatch, db, 'snapshot')
        prior = agent.run_conversation('ordinary start')
        snapshots = []
        original_provider = agent.client.chat.completions.create.side_effect
        def provider(**kwargs):
            snapshots.append(copy.deepcopy(agent._session_messages))
            return original_provider(**kwargs)
        agent.client.chat.completions.create.side_effect = provider
        live = '/private/copied-files-document.txt'
        if native:
            live = [{'type': 'text', 'text': live},
                    {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}}]
        safe = 'accepted prompt\n\n[Attached file: "copy.txt"]'
        with files_user_message_persistence(agent, safe) as transcript:
            result = agent.run_conversation(live, conversation_history=prior['messages'],
                                            persist_user_message=transcript)
        assert sent and snapshots
        assert agent._persist_user_message_override is None
        assert result['messages'][2]['content'] == safe
        assert db.get_messages('snapshot')[2]['content'] == safe
        agent.save_trajectories = True
        agent._save_trajectory(snapshots[0], 'ordinary start', completed=True)
        body = (tmp_path / 'trajectory_samples.jsonl').read_text()
        record_property('copied_trajectory', json.dumps(json.loads(body)))
        assert '/private/copied-files-document.txt' not in body
        assert 'data:image/' not in body
    finally:
        db.close()
