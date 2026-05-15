# ACP Zed Embedded Context Capability Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Advertise Hermes ACP support for embedded prompt context so Zed can send richer `@file`, `@selection`, and resource context blocks.

**Architecture:** Hermes already converts `EmbeddedResourceContentBlock` and `ResourceContentBlock` into model input parts in `acp_adapter/server.py`. This PR should only advertise `PromptCapabilities(embedded_context=True)` and add tests proving the wire key is `embeddedContext`.

**Tech Stack:** Python, ACP Python SDK, Zed ACP prompt capabilities, pytest via `scripts/run_tests.sh`.

---

### Task 1: Add capability tests

**Objective:** Lock the advertised capability and JSON alias.

**Files:**
- Modify: `tests/acp/test_server.py`

**Step 1: Extend existing initialize capability test**

In `TestInitialize.test_initialize_returns_capabilities`, add:

```python
assert caps.prompt_capabilities is not None
assert caps.prompt_capabilities.image is True
assert caps.prompt_capabilities.embedded_context is True
```

In `test_initialize_capabilities_wire_format`, add:

```python
prompt_caps = payload["promptCapabilities"]
assert prompt_caps["image"] is True
assert prompt_caps["embeddedContext"] is True
```

**Step 2: Run failing test**

```bash
scripts/run_tests.sh tests/acp/test_server.py::TestInitialize -q
```

Expected: FAIL on `embedded_context`.

### Task 2: Advertise embedded context

**Objective:** Tell ACP clients Hermes can receive embedded resources.

**Files:**
- Modify: `acp_adapter/server.py:initialize`

Change:

```python
prompt_capabilities=PromptCapabilities(image=True),
```

to:

```python
prompt_capabilities=PromptCapabilities(image=True, embedded_context=True),
```

Do not change prompt parsing in this PR unless tests prove it is broken; keep scope tight.

### Task 3: Add one conversion regression if missing

**Objective:** Make sure advertised support matches actual behavior.

**Files:**
- Modify: `tests/acp_adapter/test_acp_images.py` or `tests/acp/test_server.py` depending on existing prompt-conversion tests

Add a focused test that passes an embedded text resource into `_content_blocks_to_openai_user_content(...)` and asserts the resulting text includes the embedded resource text/name.

### Task 4: Verify

Run:

```bash
scripts/run_tests.sh tests/acp/test_server.py tests/acp_adapter/test_acp_images.py -q
```

Expected: PASS.

Manual Zed check: mention a file/selection in the ACP prompt and confirm Hermes receives the content rather than only the visible mention text.
