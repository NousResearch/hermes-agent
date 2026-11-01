from __future__ import annotations

import json

from claude_selfimprove import scanner


def test_discovers_nested_subagent_transcripts(sandbox):
    # A session transcript plus its subagent transcript, matching the real
    # Claude Code layout: projects/<project>/<session>.jsonl alongside
    # projects/<project>/<session>/subagents/agent-<id>.jsonl.
    sandbox.write_transcript(
        "claude", "proj-a", "sess-1",
        [{"type": "user", "sessionId": "sess-1", "message": {"role": "user", "content": "top level turn"}}],
    )
    subagent_dir = sandbox.claude_home / "projects" / "proj-a" / "sess-1" / "subagents"
    subagent_dir.mkdir(parents=True)
    (subagent_dir / "agent-abc123.jsonl").write_text(
        json.dumps(
            {"type": "user", "sessionId": "sess-1-sub", "message": {"role": "user", "content": "subagent turn"}}
        )
        + "\n"
    )

    files = scanner.discover_files()
    file_names = {f[2].name for f in files}
    assert "sess-1.jsonl" in file_names
    assert "agent-abc123.jsonl" in file_names

    cp = scanner.CheckpointStore()
    turns = list(scanner.scan_new_turns(cp))
    texts = {t.text for t in turns}
    assert "top level turn" in texts
    assert "subagent turn" in texts


def test_seed_all_covers_nested_subagent_transcripts(sandbox):
    sandbox.write_transcript(
        "claude", "proj-a", "sess-1",
        [{"type": "user", "sessionId": "sess-1", "message": {"role": "user", "content": "top level turn"}}],
    )
    subagent_dir = sandbox.claude_home / "projects" / "proj-a" / "sess-1" / "subagents"
    subagent_dir.mkdir(parents=True)
    (subagent_dir / "agent-abc123.jsonl").write_text(
        json.dumps({"type": "user", "sessionId": "sess-1-sub", "message": {"role": "user", "content": "x"}}) + "\n"
    )

    cp = scanner.CheckpointStore()
    seeded = scanner.seed_all(cp)
    assert seeded == 2
