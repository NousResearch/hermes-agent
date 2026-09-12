# Codex Responses Source Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let protected Codex Responses workers replay an exact, trusted `read_file` presentation while every missing, forged, ambiguous, or non-text result remains elided or blocked.

**Architecture:** Preserve the existing content-free provenance sidecar across Chat-to-Responses conversion, reattach it to one unique internal `function_call_output` selected by exact call ID and presentation hash, normalize only unambiguous single-text Responses output, and route the presentation through the existing `_segment_read_file_presentation()` authority. No provider-visible metadata or new replay policy is introduced.

**Tech Stack:** Python 3.11, SHA-256, existing Codex Responses adapter and LLM egress firewall, pytest.

**Spec:** `docs/superpowers/specs/2026-08-30-codex-responses-source-replay-design.md`

## Global Constraints

- `_segment_read_file_presentation()` remains the sole source authorization policy.
- Match Responses results by exact provider call ID and exact normalized presentation hash; never authorize by array position.
- One sidecar entry may reattach to at most one result, and one result may consume at most one entry.
- Only a scalar string or a structured list containing exactly one text item and no media/file/malformed item is eligible.
- Missing, stale, forged, mismatched, duplicate, malformed, or already-consumed provenance remains elided or fails closed.
- Terminal, search, web, mutation, scratch-file, and arbitrary tool replay behavior remains unchanged.
- Internal provenance keys must be removed before provider serialization and raw source must never enter receipts.
- Do not alter prompt history, tool schemas, path/secret/size/identity/policy gates, or publication/merge authority.

---

### Task 1: Reattach source provenance to one exact Responses result

**Files:**
- Modify: `agent/llm_egress_runtime.py`
- Modify: `tests/agent/test_llm_egress_runtime.py`

**Interfaces:**
- Produces: `_single_text_tool_output(value: Any) -> str | None`, a strict normalizer for scalar and one-`input_text` Responses outputs.
- Extends: `_restore_source_provenance_sidecar(body, sidecar) -> dict[str, Any]` to support `input[].function_call_output` without changing the existing Chat path.
- Consumes: `tool_call_id`, `content_sha256`, `request_id`, `source_grant_digests`, and `presentation_kind` from `build_source_provenance_sidecar()`.

- [ ] **Step 1: Add failing exact-match and ambiguity tests**

Add real-result tests proving exact call-ID/hash restoration, wrong-call and wrong-hash rejection, duplicate-result rejection, no mutation of caller-owned mappings/lists, and unchanged Chat restoration.

```python
def test_codex_responses_restores_source_metadata_by_call_id_and_hash(tmp_path, monkeypatch):
    body = {"input": [{"type": "function_call_output", "call_id": call_id, "output": result}]}
    restored = _restore_source_provenance_sidecar(body, sidecar)
    assert restored["input"][0]["_source_provenance"]["content_sha256"] == sha256(result.encode()).hexdigest()
```

- [ ] **Step 2: Run the named tests and verify RED**

Run: `/Users/mikedemott/.hermes/hermes-agent/venv/bin/python -m pytest -q tests/agent/test_llm_egress_runtime.py -k 'responses_restores_source_metadata or responses_does_not_restore_ambiguous or real_read_file_wire_result'`

Expected: Responses restoration assertions fail while the Chat regression remains green.

- [ ] **Step 3: Implement strict normalization and unique matching**

Accept only an exact scalar string or `[{'type': 'input_text', 'text': exact_text}]`. Reject empty/multiple lists, non-mappings, missing or non-string text, unknown types, and media/file items. Copy `input`, collect candidates by exact type/call/hash, and reattach only when exactly one unconsumed candidate matches. Keep Chat behavior compatible and attach only the four provider-private envelope fields.

- [ ] **Step 4: Run focused restoration tests**

Run the Step 2 command plus: `/Users/mikedemott/.hermes/hermes-agent/venv/bin/python -m pytest -q tests/agent/test_llm_egress_runtime.py -k 'source_provenance_sidecar or responses_restores_source_metadata or responses_does_not_restore_ambiguous'`

- [ ] **Step 5: Commit transport restoration**

```bash
git add agent/llm_egress_runtime.py tests/agent/test_llm_egress_runtime.py
git commit -m "fix(security): bind provenance to Responses results"
```

### Task 2: Authorize exact scalar and one-text Responses read replay

**Files:**
- Modify: `agent/llm_egress_runtime.py`
- Modify: `tests/agent/test_llm_egress_runtime.py`

**Interfaces:**
- Consumes: `_single_text_tool_output()`, restored `_source_provenance`, `_segment_read_file_presentation()`, and current source registry/request identity.
- Produces: exact trusted source replay for eligible `function_call_output.output`; every other structured read remains `_READ_FILE_REPLAY_ELISION` or `untrusted_provenance`.

- [ ] **Step 1: Add failing end-to-end authorization tests**

Use the real flow: activate provenance, call `read_file_tool()`, attach metadata, build the content-free sidecar, convert to Responses shape, advance the request ID, and authorize SDK kwargs. Cover scalar and one-`input_text` positive cases.

Add a fail-closed matrix for missing sidecar, stale request, forged metadata/hash, wrong call ID, duplicate result, consumed sidecar, changed bytes/line numbering, policy mismatch, multiple text items, image-plus-text, malformed item, truncated JSON, scratch-file call, and absent metadata. Assert terminal replay remains outcome-only, arbitrary structured output gains no authority, internal keys are absent, and receipts contain no source bytes.

- [ ] **Step 2: Run the new authorization tests and verify RED**

Run: `/Users/mikedemott/.hermes/hermes-agent/venv/bin/python -m pytest -q tests/agent/test_llm_egress_runtime.py -k 'real_read_file_responses or responses_read_file_fails_closed or responses_source_metadata_is_internal or structured_terminal_replay'`

Expected: exact Responses read replay fails because structured read output is still unconditionally elided.

- [ ] **Step 3: Route only eligible structured reads through the existing validator**

Before unconditional structured-read elision, normalize eligible output and send only valid reattached metadata through `_segment_read_file_presentation()`. Preserve the provider's scalar/list shape. Never classify normalized source as generated or sanitized context, and keep projection/scratch branches fail closed.

- [ ] **Step 4: Run focused egress and provenance suites**

Run: `/Users/mikedemott/.hermes/hermes-agent/venv/bin/python -m pytest -q tests/agent/test_llm_egress_runtime.py tests/agent/test_source_provenance.py tests/tools/test_file_tools.py`

Run: `/Users/mikedemott/.hermes/hermes-agent/venv/bin/python -m pytest -q tests -k 'protected_kanban and (codex or read_file)'`

Expected: focused suites pass; broad-suite status is reported separately.

- [ ] **Step 5: Verify compile, hygiene, and no authority expansion**

Run: `/Users/mikedemott/.hermes/hermes-agent/venv/bin/python -m compileall -q agent/llm_egress_runtime.py agent/source_provenance_tools.py tests/agent/test_llm_egress_runtime.py`

Run: `git diff --check` and `git status --short`. Inspect the diff for no prompt/tool-schema changes, terminal replay expansion, provider-visible internal key, or gate weakening.

- [ ] **Step 6: Commit authorization and regression coverage**

```bash
git add agent/llm_egress_runtime.py tests/agent/test_llm_egress_runtime.py
git commit -m "fix(security): replay trusted reads in Codex Responses"
```

### Task 3: Independent whole-branch verification and acceptance boundary

**Files:**
- Review: `agent/llm_egress_runtime.py`
- Review: `agent/source_provenance_tools.py`
- Review: `tests/agent/test_llm_egress_runtime.py`
- Review: `docs/superpowers/specs/2026-08-30-codex-responses-source-replay-design.md`
- Review: `docs/superpowers/plans/2026-08-30-codex-responses-source-replay.md`

**Interfaces:**
- Consumes: committed Task 1-2 implementation and fresh exact-HEAD evidence.
- Produces: independent review disposition and a stop before live board mutation, push, or merge.

- [ ] **Step 1: Run exact-HEAD focused verification**

Run Task 2 suites, compileall, `git diff --check`, and `git status --short` from the committed branch head. Record pass counts, skipped tests, interpreter path, branch, and SHA.

- [ ] **Step 2: Conduct independent security review**

Review exact base-to-head diff for duplicate matching, sidecar consumption, structured ambiguity, metadata stripping, grant rebinding, request/policy identity, receipt privacy, and unchanged non-read behavior. Any finding returns to a fresh implementer and repeats verification.

- [ ] **Step 3: Stop before acceptance canary or publication**

Report the reviewed branch and evidence. Do not dispatch a live Hermes card, update the deployed runtime, push, merge, or touch a LunaBot PR until the user explicitly authorizes that external step against the exact reviewed SHA.
