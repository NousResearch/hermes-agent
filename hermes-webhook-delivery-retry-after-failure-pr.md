# PR Draft: Allow webhook `deliver_only` retries after downstream delivery failure

## Title

Allow webhook `deliver_only` retries after downstream delivery failure

## Summary

This changes the webhook adapter so `deliver_only` requests only enter the idempotency cache after the downstream delivery succeeds. Previously, a failed direct delivery still poisoned the cache entry, so a provider retry with the same delivery ID was treated as a duplicate and skipped.

## Why

`deliver_only` routes are used for push-style notifications where the webhook POST itself is the delivery. If the downstream target rejects the message or returns a transient failure, the provider should be able to retry the same delivery ID instead of getting a false duplicate response.

## Changes

- Delay idempotency caching for `deliver_only` until after a successful direct delivery.
- Keep duplicate suppression for successful deliveries.
- Add a regression test that fails the first direct delivery, retries with the same delivery ID, and confirms the second attempt is delivered.

## Tests

```text
python -m pytest tests/gateway/test_webhook_adapter.py tests/gateway/test_webhook_deliver_only.py -q
83 passed in 13.42s
```

## Branch

Local branch:

```text
fix/webhook-delivery-retry-after-failure
```
