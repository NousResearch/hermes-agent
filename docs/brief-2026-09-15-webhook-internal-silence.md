# Treat agent-running webhook events as internal machinery

## Observed problem

A configured GitHub webhook route runs the agent and delivers its result to Slack. When the agent correctly returns `[SILENT]` because an event needs no visible update, the gateway replaces it with:

> ⚠️ The model returned only a silence marker for a message that needed a reply. Try again or rephrase.

Live logs identify these turns as `platform=webhook chat=webhook:pump-pr-events:<delivery-id>`. Starting a new Slack session does not help because each webhook delivery creates its own session.

`WebhookAdapter._spawn_agent_run()` creates a `MessageEvent` without `internal=True`. `display_kind_for_event()` therefore classifies the event like a human-authored message, while the comparable Microsoft Graph webhook adapter correctly marks its generated event internal.

## Required behavior

Webhook payloads that start an agent run are machine-generated events. Create their `MessageEvent` with `internal=True` so an intentional silence marker suppresses delivery instead of producing the human-turn warning.

## Acceptance cases

1. Add a focused test that exercises the real webhook event creation path and proves the emitted `MessageEvent` is internal.
2. Prove the test fails against the production behavior before changing `gateway/platforms/webhook.py`.
3. After the change, the focused webhook test passes.
4. Existing webhook adapter and gateway silence tests pass.
5. Human-authored Slack, DM, and channel messages remain non-internal and still receive the warning if the model returns only a silence marker.
6. Webhook delivery, reply targeting, HMAC validation, idempotency, and per-delivery session behavior remain unchanged.

## Expected files

- `gateway/platforms/webhook.py`
- One focused existing test file under `tests/gateway/`, preferably `tests/gateway/test_webhook_adapter.py`

## Constraints

- Use strict test-first RED/GREEN development.
- Use `scripts/run_tests.sh`, not bare pytest.
- Make the smallest change. No refactor or unrelated cleanup.
- Production-code budget: 3 changed lines. Test budget: 35 changed lines.
- Do not merge, deploy, restart services, or call production.
