import pytest
from plugins.platforms.telegram import sent_messages as sm

@pytest.mark.parametrize("prefix", ["", "The new version is running.\n\nReact 👍 to this message:\n\n"])
def test_local_proposal_with_preface(tmp_path, monkeypatch, prefix):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    text = prefix + 'Proposal to Sam: “Reply Reaction received here. Send nothing to anyone else.”'
    assert sm.is_draft_message(text)
    result = sm.record_durable_consent("1", "2", "1", thread_id="3", session_key="test", text=text)
    assert result["ok"]
    assert result["bound_draft"] is None

@pytest.mark.parametrize("text", ["I read a proposal today", "There are 3 drafts ready", "Status: proposal processing", "Someone wrote Proposal to Sam yesterday"])
def test_chatter_is_not_consent(tmp_path, monkeypatch, text):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert not sm.is_draft_message(text)
