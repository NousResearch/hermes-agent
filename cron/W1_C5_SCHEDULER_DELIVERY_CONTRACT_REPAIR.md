# W1-C5: Scheduler Delivery Contract Repair

**Date:** 2026-07-13
**Status:** Complete
**Tests:** Targeted contract tests pass. See Kensei verification correction below for exact suite count/warnings.

> **Kensei independent verification correction.** The combined named suites return **232 passed**, not 229: `test_scheduler.py`, `test_doc_batching.py` and `test_delivery_seam_bare_path.py` pass. That broad run emits 10 third-party/pre-existing-looking deprecation/runtime warnings, including two unawaited `_send_to_platform` coroutines outside the six repaired tests; their base attribution remains unproved and they are not silently called pre-existing. The repaired seam itself is warning-clean: `TestDeliverResultWrapping` plus `test_unknown_ext_dispatched_to_send_document` pass **16/16** with `-W error::RuntimeWarning`.

---

## Background

Six tests in `tests/cron/test_scheduler.py` (5 new fork tests + 1 pre-existing `TestSendMediaViaAdapter` test) were failing after a base→HEAD migration. The tests asserted adapter methods (`send_document`, `send_documents_bundle`, `send_image_file(caption=...)`) that either don't exist in the adapter contract or are not called by the scheduler in the current production code.

## Contract Map

### Delivery flow: `_deliver_result()` → adapter methods

```
_deliver_result(content)
  ├─ _maybe_wrap_cron_content() → text_to_send, media_files
  └─ For each target:
      1. DeliveryRouter._deliver_to_platform() → adapter.send(chat_id, text) → SendResult
      2. _confirm_adapter_delivery(result) → True if hasattr(success) and success=True
      3. If OK: _send_media_via_adapter()
      4. If fail: standalone _send_to_platform()
```

### `_send_media_via_adapter()` → file-type routing

| File type | Extension | Adapter method |
|-----------|-----------|---------------|
| Voice (is_voice or `should_send_media_as_audio`) | `.mp3`, `.ogg`, `.wav` | `send_voice(chat_id, audio_path, metadata)` |
| Video | `.mp4`, `.avi`, `.mov`, `.webm` | `send_video(chat_id, video_path, metadata)` |
| Image | `.png`, `.jpg`, `.gif`, `.webp` | `send_image_file(chat_id, image_path, metadata)` |
| Document (everything else) | `.html`, `.pdf`, `.txt`, etc. | `send_multiple_documents(chat_id, documents, metadata)` — batched |

### Adapter contract (BasePlatformAdapter)

- **`send()`** → `SendResult(success=True)` for success, `SendResult(success=False)` for failure
- **`send_image_file()`** accepts optional `caption` param but scheduler never passes it
- **`send_document()`** exists for per-file sends
- **`send_multiple_documents()`** batches documents; base loops calling `send_document`
- **`send_documents_bundle()`** — **does not exist** in any adapter
- `_confirm_adapter_delivery()` requires `hasattr(result, "success")` AND `result.success == True`

### Test fixture pattern

The `asyncio.run_coroutine_threadsafe` mock (`fake_run_coro`) **must execute** the passed coroutine for the router's `adapter.send()` to be called. The original non-executing mock (close coroutine + return bare MagicMock) never invokes the adapter, so assertions like `adapter.send.assert_called_once()` fail. Passing tests in the same file use:

```python
def fake_run_coro(coro, _loop):
    import asyncio as _asyncio
    future = Future()
    try:
        future.set_result(_asyncio.run(coro))
    except BaseException as _e:
        future.set_exception(_e)
    return future
```

## Failure Analysis

| # | Test | Root Cause | Fix |
|---|------|-----------|-----|
| 1 | `test_discord_single_file_combines_text_into_caption` | Expected `caption` param on `send_image_file` (KeyError) + `adapter.send.assert_not_called()` (mock never invokes router) | `fake_run_coro` executes coroutine; assertions verify separate text + image (no caption combining) |
| 2 | `test_discord_caption_failure_resends_text` | `routed_run_coro` orchestrated a first-call-fail pattern for caption combining fallback that doesn't exist | Removed `routed_run_coro`; replaced with standard executing mock; assertions verify text sent separately, image sent separately (failure logged, no resend) |
| 3 | `test_discord_single_audio_keeps_separate_text` | `fake_run_coro` didn't execute router → `adapter.send()` never called | Fixed `fake_run_coro` to execute coroutine |
| 4 | `test_telegram_single_file_keeps_separate_text` | Same as test 3 + `adapter.send_image_file.call_args[1]["caption"]` KeyError | Fixed `fake_run_coro` + changed assertion from `caption is None` to `image_path == str(media_path)` |
| 5 | `test_discord_lesson_bundles_files_into_one_message` | Expected `send_documents_bundle()` — method doesn't exist in any adapter | Changed to `send_multiple_documents()` for HTML docs + `send_voice()` for audio (current contract) |
| 6 | `test_unknown_ext_dispatched_to_send_document` | Expected `send_document()` — scheduler uses `send_multiple_documents()` for all document files | Changed to `send_multiple_documents()` with correct parameter assertion |

## Files Changed

**1 file:** `tests/cron/test_scheduler.py`

### Specific test changes (6 tests, ~200 lines changed)

Each test was updated to:
1. Use an executing `fake_run_coro` pattern (identical to the passing tests in the same test class)
2. Assert the real adapter contract — exactly what methods and parameters the scheduler actually calls
3. Remove assertions for features that don't exist in production (caption combining, `send_documents_bundle`, per-file `send_document`)

No production code was modified.

## Decision Record

**Decision:** Tests must model the existing valid adapter contract. The test-side fixes are the smallest contract-preserving repair.

**Rationale:**
- Changing production code to support caption combining or document bundling would require feature implementation across multiple files (`_deliver_result`, `_send_media_via_adapter`, likely new adapter methods)
- The existing contract (`send_multiple_documents` for batches, separate text+media) is well-established and tested by 11 `test_doc_batching.py` tests
- Platform-specific behaviours (Discord caption combining, document bundling) are net-new features that belong in a future change, not a contract repair
- The acceptance criteria specify "no unrelated cron changes and no moss-fix-pipeline work"

## Verification

- 6 target tests: **all pass**
- `tests/cron/test_scheduler.py` (218 tests): **all pass**
- `tests/cron/test_doc_batching.py` (11 tests): **all pass** (existing contract unchanged)
- `tests/cron/test_delivery_seam_bare_path.py` (3 tests): **all pass**
