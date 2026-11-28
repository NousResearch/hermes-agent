"""Recovery must search original text that did not survive compaction verbatim."""
import json

import pytest

from evals.compaction import runner
from evals.compaction.jev_arm import fake_asker


def _run_recovery(tmp_path, monkeypatch, *, truncate, shared_prefix):
    prefix = "Build status: preliminary checks complete. " * 10
    fact = "The release approval code is LARCH-7429."
    original = prefix + fact + " Additional output." * 50
    retained = (prefix if shared_prefix else "Unrelated status. " * 25) + "No approval code."
    messages = [
        {"role": "user", "content": "Run the checks and remember the approval code."},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "first", "function": {"name": "terminal", "arguments": '{}'}}
        ]},
        {"role": "tool", "tool_call_id": "first", "content": original},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "second", "function": {"name": "terminal", "arguments": '{}'}}
        ]},
        {"role": "tool", "tool_call_id": "second", "content": retained},
        {"role": "assistant", "content": "Checks complete."},
    ]
    searched = []
    search = runner.keyword_search

    def record_search(archive, query):
        searched.extend(archive)
        return search(archive, query)

    def call(prompt, max_tokens=2000):
        if "Write the best search query" in prompt:
            return "release approval code"
        if prompt.startswith("Score this answer"):
            return json.dumps({"score": 2 if "ANSWER: LARCH-7429" in prompt else 0})
        return "LARCH-7429" if fact in prompt else "NOT IN CONTEXT"

    monkeypatch.setattr("evals.compaction.jev_arm.openrouter_asker", lambda: fake_asker(
        keep_call=0.9 if truncate else 0.1, keep_result=0.1,
    ))
    monkeypatch.setattr(runner, "_call", call)
    monkeypatch.setattr(runner, "keyword_search", record_search)
    runner.run_policy(
        "jev", {"engine": "jev", "jev": {"preserve_recent_messages": 3}},
        messages, [{"q": "What is the approval code?", "gold": "LARCH-7429"}],
        tmp_path, with_recovery=True,
    )
    result = json.loads((tmp_path / "jev_recovery.json").read_text(encoding="utf-8"))
    return messages, searched, result["results"][0]


@pytest.mark.parametrize("truncate,shared_prefix", [(True, False), (True, True), (False, True)])
def test_recovery_can_find_removed_text(tmp_path, monkeypatch, truncate, shared_prefix):
    messages, searched, result = _run_recovery(
        tmp_path, monkeypatch, truncate=truncate, shared_prefix=shared_prefix,
    )
    assert messages[2] in searched
    assert messages[4] not in searched
    assert result["answer"] == "LARCH-7429"


def test_recovery_excludes_unchanged_messages(tmp_path, monkeypatch):
    messages, searched, result = _run_recovery(
        tmp_path, monkeypatch, truncate=False, shared_prefix=False,
    )
    assert messages[2] in searched
    assert messages[4] not in searched
    assert all(m["role"] == "tool" or m.get("tool_calls") for m in searched)
    assert result["answer"] == "LARCH-7429"
