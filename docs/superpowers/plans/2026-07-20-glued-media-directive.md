# Glued Media Directive Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract known-extension `MEDIA:` paths when `[[as_document]]` is appended without whitespace.

**Architecture:** Extend the existing shared media-tag regex delimiter set by one character. Exercise the public `BasePlatformAdapter.extract_media()` path so extraction and cleanup stay covered together.

**Tech Stack:** Python, `re`, pytest via `scripts/run_tests.sh`.

## Global Constraints

- Change only the shared media-tag delimiter and its focused regression coverage.
- Preserve protected code-block, inline-code, blockquote, and JSON behavior.
- Add no dependencies, config, directive syntax, or per-file document mode.
- Run tests only through `scripts/run_tests.sh`.

---

### Task 1: Accept a glued document directive

**Files:**
- Modify: `tests/gateway/test_platform_base.py`
- Modify: `gateway/platforms/base.py:1499`

**Interfaces:**
- Consumes: `BasePlatformAdapter.extract_media(content: str) -> tuple[list[tuple[str, bool]], str]`
- Produces: `MEDIA_TAG_CLEANUP_RE` accepting `[` immediately after a recognized extension.

- [ ] **Step 1: Write the failing regression test**

Add beside existing `[[as_document]]` extraction tests:

```python
def test_as_document_directive_can_touch_media_path(self):
    content = "Before\nMEDIA:/tmp/report.xlsx[[as_document]]\nAfter"

    media, cleaned = BasePlatformAdapter.extract_media(content)

    assert media == [("/tmp/report.xlsx", False)]
    assert "MEDIA:" not in cleaned
    assert "[[as_document]]" not in cleaned
    assert cleaned == "Before\n\nAfter"
```

- [ ] **Step 2: Run test and verify RED**

Run:

```bash
scripts/run_tests.sh tests/gateway/test_platform_base.py::TestExtractMedia::test_as_document_directive_can_touch_media_path -q
```

Expected: FAIL because `media == []`.

- [ ] **Step 3: Implement minimal regex fix**

In `MEDIA_TAG_CLEANUP_RE`, add `[` to the post-extension delimiter class:

```python
r'''(?=[\s`"',;:)\]}\[]|$)[`"']?''',
```

Do not alter any other regex branch.

- [ ] **Step 4: Run focused verification**

Run:

```bash
scripts/run_tests.sh tests/gateway/test_platform_base.py::TestExtractMedia::test_as_document_directive_can_touch_media_path -q
scripts/run_tests.sh tests/gateway/test_platform_base.py -q
```

Expected: new test PASS; full file PASS.

- [ ] **Step 5: Review and checkpoint**

Run:

```bash
git diff --check
git diff -- gateway/platforms/base.py tests/gateway/test_platform_base.py
git add gateway/platforms/base.py tests/gateway/test_platform_base.py
git commit -m "wip: fix glued media document directive extraction" -m "Co-Authored-By: OpenAI Codex <noreply@openai.com>"
```
