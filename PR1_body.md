## What does this PR do?
Redelivered final replies (gateway restart recovery, reconnect replay, flood redelivery) are now sent as a **platform reply to the inbound message they answer**. Today a recovered reply arrives as a bare message with only the "recovered reply" marker, so on Telegram/Slack the user cannot tell which of their questions a late answer belongs to (e.g. a reply generated while the laptop lid was closed and delivered minutes later after wake).

## Related Issue
Related to the reliability wave in #111364 (ledger redelivery after `send_path_degraded`); complements #105810 / #85948, which make redelivery happen — this PR makes the redelivered message *identifiable*.

## Type of Change
- [x] 🐛 Bug fix (non-breaking change that fixes an issue)
- [x] ✅ Tests (adding or improving test coverage)

## Changes Made
- `gateway/delivery_ledger.py`: new nullable `reply_to` column (added via `add_column_if_missing`, so existing ledgers migrate in place); `record_obligation(..., reply_to=)`; `get_reply_to(obligation_id)`.
- `gateway/platforms/base.py`: the producer records the ledger message id (`ledger_message_id` or the inbound `message_id`) as the anchor.
- `gateway/run_startup.py`: `_redeliver_claimed_obligations` passes `reply_to=` to `adapter.send`. Rows written before the column existed redeliver unanchored, exactly as before. A platform whose anchor is gone (deleted message) already falls back inside its own adapter (Telegram: "message to be replied not found" → unanchored retry).

## How to Test
```
pytest -q tests/gateway/test_delivery_ledger.py tests/gateway/test_delivery_ledger_producer.py tests/gateway/test_platform_reconnect.py
```
New tests: `test_redelivery_anchors_reply_to_the_inbound_message`, `test_redelivery_without_anchor_sends_unanchored`, `test_record_obligation_stores_reply_anchor`.
Manual: with a Telegram adapter, insert a `failed`/`attempting` obligation for a real inbound message id, restart the gateway → the recovered reply arrives quoting that message.

## Checklist
- [x] Searched open and merged PRs/issues for duplicates
- [x] Tests added/updated and passing locally
- [x] No user-facing string changes; no config changes

🤖 Generated with [Claude Code](https://claude.com/claude-code)
