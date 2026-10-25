"""Regression tests for the weakest-sufficient-scope guard in llm.py.

Grounded in Bennett, "The Optimal Choice of Hypothesis Is the Weakest, Not
the Shortest" (arXiv:2301.12987) and its corrigendum "Optimal Policy Is
Weakest Policy": a claim scoped to every future session ("global") is a
stronger, more restrictive generalization than one scoped to the one repo
or script it was observed in ("repo"). Evidence that only ever names one
place does not support the stronger claim, so the classifier's "global"
answer is held to "repo" whenever the raw evidence says where it happened.
"""

from __future__ import annotations

import json

from claude_selfimprove import llm
from claude_selfimprove.heuristics import RawCandidate


def _raw(text, context="", category="explicit_instruction"):
    return RawCandidate(
        source="claude", project="proj", session_id="sess-1", file_path="/tmp/f.jsonl",
        category=category, text=text, context_text=context,
        matched_pattern="never", timestamp="2026-08-01T00:00:00Z",
    )


def _classified_as_global(text="Never use --no-verify.", context="", canonical_key="never-use-no-verify"):
    def runner(prompt, *, model, provider, timeout):
        return True, json.dumps([{
            "index": 1, "is_real_lesson": True, "canonical_key": canonical_key,
            "scope": "global", "target_kind": "rule", "title": "Never use --no-verify",
            "body": "Never use --no-verify.", "confidence": 0.9,
        }])
    return llm.classify_batch([_raw(text, context)], runner=runner)


def test_looks_narrowly_scoped_detects_named_repo_script_and_task():
    assert llm._looks_narrowly_scoped("Never do X in this repo.")
    assert llm._looks_narrowly_scoped("Only in this script should you skip Y.")
    assert llm._looks_narrowly_scoped("", "Fixed it in this hotfix branch.")
    assert llm._looks_narrowly_scoped("For this project, always run Z first.")


def test_looks_narrowly_scoped_false_for_genuinely_general_text():
    assert not llm._looks_narrowly_scoped("Never force push to main.")
    assert not llm._looks_narrowly_scoped("Always run tests before committing.")
    assert not llm._looks_narrowly_scoped("", "")


def test_global_scope_is_downgraded_to_repo_when_evidence_names_a_repo():
    results = _classified_as_global(text="Never use --no-verify in this repo's pre-commit hook.")
    assert len(results) == 1
    assert results[0].scope == "repo"


def test_global_scope_is_downgraded_when_only_the_context_names_a_script():
    # The user's confirmation is scope-agnostic; the preceding assistant
    # turn is where the concrete script name lives - both must be checked.
    results = _classified_as_global(
        text="That fixed it, works now.",
        context="Fixed the retry bug in this script's deploy step.",
    )
    assert len(results) == 1
    assert results[0].scope == "repo"


def test_genuinely_global_evidence_keeps_global_scope():
    results = _classified_as_global(text="Never force push to the main branch, that's a hard rule.")
    assert len(results) == 1
    assert results[0].scope == "global"


def test_model_answering_repo_is_unaffected_by_the_guard():
    def runner(prompt, *, model, provider, timeout):
        return True, json.dumps([{
            "index": 1, "is_real_lesson": True, "canonical_key": "repo-specific-thing",
            "scope": "repo", "target_kind": "rule", "title": "t",
            "body": "Only this repo needs this.", "confidence": 0.9,
        }])
    results = llm.classify_batch(
        [_raw("Never do the weird thing in this repo.")], runner=runner
    )
    assert results[0].scope == "repo"


def test_prompt_instructs_the_model_to_prefer_the_narrower_scope():
    prompt = llm._build_prompt([_raw("Never use --no-verify.")])
    assert "narrower" in prompt.lower()
    assert "one observation in one place" in prompt.lower()
