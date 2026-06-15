# PR Draft: Allow webhook retries after downstream delivery failure

## Title

Allow webhook retries after downstream delivery failure

## Summary

This changes the webhook adapter so failed deliveries do not poison the idempotency cache. `deliver_only` routes now cache only after a successful downstream delivery, and normal webhook runs release the delivery ID if the background agent task fails.

## Why

`deliver_only` routes are used for push-style notifications where the webhook POST itself is the delivery. Normal webhook routes also need this behavior: if the agent task fails after accepting the event, the provider should be able to retry the same delivery ID instead of getting a false duplicate response.

## Changes

- Delay idempotency caching for `deliver_only` until after a successful direct delivery.
- Release idempotency and delivery-info state if a normal webhook task fails after acceptance.
- Keep duplicate suppression for successful deliveries.
- Add regression tests for both failed direct delivery retries and failed agent-run retries.

## Tests

```text
python -m pytest tests/gateway/test_webhook_adapter.py tests/gateway/test_webhook_deliver_only.py -q
85 passed in ...
```

## Branch

Local branch:

```text
fix/webhook-delivery-retry-after-failure
```
