# SMS Long-Message Preservation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let SMS receive full long output so its existing 1,600-character chunker can deliver every segment.

**Architecture:** Declare the capability already implemented by `SmsAdapter.send()`. Add one adapter contract test and one real-adapter delivery-router regression; leave Twilio transport logic unchanged.

**Tech Stack:** Python, pytest, asyncio tests via `scripts/run_tests.sh`.

## Global Constraints

- Reuse `BasePlatformAdapter.splits_long_messages` and existing `truncate_message()`.
- Keep `MAX_SMS_LENGTH = 1600`, Twilio transport, audit saving, and error handling unchanged.
- Add no dependencies or config.
- Run tests only through `scripts/run_tests.sh`.

---

### Task 1: Advertise SMS native chunking

**Files:**
- Modify: `tests/gateway/test_sms.py`
- Modify: `tests/gateway/test_delivery.py`
- Modify: `plugins/platforms/sms/adapter.py:57`

**Interfaces:**
- Consumes: `DeliveryRouter._deliver_to_platform()`, `SmsAdapter.send()`, and `BasePlatformAdapter.truncate_message()`.
- Produces: `SmsAdapter.splits_long_messages = True`.

- [ ] **Step 1: Write failing adapter contract test**

Add under `TestSmsFormatAndTruncate`:

```python
def test_declares_native_long_message_splitting(self):
    from plugins.platforms.sms.adapter import SmsAdapter

    content = "x" * 5000
    chunks = SmsAdapter.truncate_message(content, SmsAdapter.MAX_MESSAGE_LENGTH)

    assert SmsAdapter.splits_long_messages is True
    assert len(chunks) > 1
    assert all(len(chunk) <= SmsAdapter.MAX_MESSAGE_LENGTH for chunk in chunks)
```

- [ ] **Step 2: Write failing router integration test**

Add beside existing chunking-adapter delivery tests:

```python
@pytest.mark.asyncio
async def test_long_output_preserved_for_sms_adapter(tmp_path, monkeypatch):
    from plugins.platforms.sms.adapter import SmsAdapter

    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    adapter = object.__new__(SmsAdapter)
    delivered = []

    async def capture_send(chat_id, content, metadata=None):
        delivered.append(content)
        return SendResult(success=True)

    adapter.send = capture_send
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.SMS: adapter})
    target = DeliveryTarget.parse("sms:+15551234567")
    long_content = "x" * 5000

    await router._deliver_to_platform(target, long_content, metadata={"job_id": "sms-long"})

    assert delivered == [long_content]
```

- [ ] **Step 3: Run tests and verify RED**

Run:

```bash
scripts/run_tests.sh tests/gateway/test_sms.py::TestSmsFormatAndTruncate::test_declares_native_long_message_splitting tests/gateway/test_delivery.py::test_long_output_preserved_for_sms_adapter -q
```

Expected: both FAIL because SMS inherits `splits_long_messages = False` and the router truncates.

- [ ] **Step 4: Implement minimal capability declaration**

Add beside `MAX_MESSAGE_LENGTH`:

```python
splits_long_messages = True  # send() chunks via truncate_message(MAX_MESSAGE_LENGTH)
```

- [ ] **Step 5: Run focused verification**

Run:

```bash
scripts/run_tests.sh tests/gateway/test_sms.py::TestSmsFormatAndTruncate::test_declares_native_long_message_splitting tests/gateway/test_delivery.py::test_long_output_preserved_for_sms_adapter -q
scripts/run_tests.sh tests/gateway/test_sms.py tests/gateway/test_delivery.py -q
```

Expected: focused tests PASS; both full files PASS.

- [ ] **Step 6: Review and checkpoint**

Run:

```bash
git diff --check
git diff -- plugins/platforms/sms/adapter.py tests/gateway/test_sms.py tests/gateway/test_delivery.py
git add plugins/platforms/sms/adapter.py tests/gateway/test_sms.py tests/gateway/test_delivery.py
git commit -m "wip: preserve long SMS delivery output" -m "Co-Authored-By: OpenAI Codex <noreply@openai.com>"
```
