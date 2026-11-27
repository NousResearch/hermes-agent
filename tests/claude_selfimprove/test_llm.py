from __future__ import annotations

import json

from claude_selfimprove import llm
from claude_selfimprove.heuristics import RawCandidate


def _raw(category="explicit_instruction", text="Never use --no-verify.", context=""):
    return RawCandidate(
        source="claude",
        project="proj",
        session_id="sess-1",
        file_path="/tmp/f.jsonl",
        category=category,
        text=text,
        context_text=context,
        matched_pattern="never",
        timestamp="2026-08-01T00:00:00Z",
    )


def _ok_runner(response_items):
    def runner(prompt, *, model, provider, timeout):
        return True, json.dumps(response_items)

    return runner


def test_classifies_a_real_lesson(sandbox):
    batch = [_raw()]
    runner = _ok_runner(
        [
            {
                "index": 1,
                "is_real_lesson": True,
                "canonical_key": "never-use-no-verify",
                "scope": "global",
                "target_kind": "rule",
                "title": "Never skip pre-commit hooks",
                "body": "Never use --no-verify to bypass pre-commit hooks.",
                "confidence": 0.9,
            }
        ]
    )
    results = llm.classify_batch(batch, runner=runner)
    assert len(results) == 1
    assert results[0].canonical_key == "never-use-no-verify"
    assert results[0].scope == "global"
    assert results[0].confidence == 0.9


def test_drops_items_marked_not_a_real_lesson(sandbox):
    batch = [_raw()]
    runner = _ok_runner([{"index": 1, "is_real_lesson": False}])
    assert llm.classify_batch(batch, runner=runner) == []


def test_handles_markdown_fenced_response(sandbox):
    batch = [_raw()]
    payload = json.dumps(
        [
            {
                "index": 1, "is_real_lesson": True, "canonical_key": "a-b-c",
                "scope": "global", "target_kind": "rule", "title": "t",
                "body": "Body text.", "confidence": 0.5,
            }
        ]
    )

    def runner(prompt, *, model, provider, timeout):
        return True, f"```json\n{payload}\n```"

    results = llm.classify_batch(batch, runner=runner)
    assert len(results) == 1


def test_drops_items_with_invalid_canonical_key(sandbox):
    batch = [_raw()]
    runner = _ok_runner(
        [
            {
                "index": 1, "is_real_lesson": True, "canonical_key": "Not A Slug!",
                "scope": "global", "target_kind": "rule", "title": "t",
                "body": "b", "confidence": 0.5,
            }
        ]
    )
    assert llm.classify_batch(batch, runner=runner) == []


def test_drops_items_with_invalid_scope_or_target_kind(sandbox):
    batch = [_raw()]
    runner = _ok_runner(
        [
            {
                "index": 1, "is_real_lesson": True, "canonical_key": "some-key",
                "scope": "planet-wide", "target_kind": "rule", "title": "t",
                "body": "b", "confidence": 0.5,
            }
        ]
    )
    assert llm.classify_batch(batch, runner=runner) == []


def test_confidence_is_clamped_to_0_1():
    batch = [_raw()]
    runner = _ok_runner(
        [
            {
                "index": 1, "is_real_lesson": True, "canonical_key": "some-key",
                "scope": "global", "target_kind": "rule", "title": "t",
                "body": "b", "confidence": 5.0,
            }
        ]
    )
    results = llm.classify_batch(batch, runner=runner)
    assert results[0].confidence == 1.0


def test_out_of_range_index_is_dropped():
    batch = [_raw()]
    runner = _ok_runner(
        [{"index": 7, "is_real_lesson": True, "canonical_key": "x-y", "scope": "global",
          "target_kind": "rule", "title": "t", "body": "b", "confidence": 0.5}]
    )
    assert llm.classify_batch(batch, runner=runner) == []


def test_unparseable_output_returns_empty_and_logs(sandbox):
    def runner(prompt, *, model, provider, timeout):
        return True, "this is not json at all"

    assert llm.classify_batch([_raw()], runner=runner) == []
    from claude_selfimprove import audit
    events = [e["event"] for e in audit.read_all()]
    assert "llm_classification_unparseable" in events


def test_runner_failure_returns_empty_and_logs(sandbox):
    def runner(prompt, *, model, provider, timeout):
        return False, "hermes binary not found"

    assert llm.classify_batch([_raw()], runner=runner) == []
    from claude_selfimprove import audit
    events = [e["event"] for e in audit.read_all()]
    assert "llm_classification_failed" in events


def test_empty_batch_short_circuits_without_calling_runner():
    calls = []

    def runner(prompt, *, model, provider, timeout):
        calls.append(prompt)
        return True, "[]"

    assert llm.classify_batch([], runner=runner) == []
    assert calls == []


def test_body_is_redacted_defensively():
    batch = [_raw()]
    runner = _ok_runner(
        [
            {
                "index": 1, "is_real_lesson": True, "canonical_key": "some-key",
                "scope": "global", "target_kind": "rule", "title": "t",
                "body": "contact george@flexslot.gg about this", "confidence": 0.5,
            }
        ]
    )
    results = llm.classify_batch(batch, runner=runner)
    assert "george@flexslot.gg" not in results[0].body


def test_classify_all_chunks_into_batches():
    calls = []

    def runner(prompt, *, model, provider, timeout):
        calls.append(prompt)
        n = prompt.count("\n1. ") + prompt.count("\n2. ")  # rough count, unused
        # Return one classified item per snippet number found in this batch.
        items = []
        for i in range(1, prompt.count("[category:") + 1):
            items.append(
                {
                    "index": i, "is_real_lesson": True, "canonical_key": f"key-{len(calls)}-{i}",
                    "scope": "global", "target_kind": "rule", "title": "t",
                    "body": "b", "confidence": 0.5,
                }
            )
        return True, json.dumps(items)

    batch = [_raw(text=f"Never do thing {i}.") for i in range(25)]
    results = llm.classify_all(batch, batch_size=10, runner=runner)
    assert len(calls) == 3  # 10 + 10 + 5
    assert len(results) == 25


def test_real_runner_missing_binary_fails_safe(sandbox, monkeypatch):
    # Force a PATH with no `hermes` binary; the real runner must degrade to
    # (False, ...) rather than raising.
    monkeypatch.setenv("PATH", "/nonexistent-bin-dir")
    ok, msg = llm._run_hermes_chat("prompt", model=None, provider=None, timeout=5)
    assert ok is False
    assert "not found" in msg.lower() or "no such file" in msg.lower()
