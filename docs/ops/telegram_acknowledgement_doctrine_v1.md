# Telegram Acknowledgement Doctrine v1

Date: 2026-04-08
Owner: Architect
Scope: direct Telegram operating behavior for YQ

## Purpose

Make message acknowledgement consistent and persistent.

The goal is simple:
- when YQ sends a message, the system should visibly acknowledge receipt quickly
- acknowledgement should not depend on memory alone or ad hoc judgment
- failure to react should fall back to a brief textual acknowledgement, not silence

## Hard rule

For direct Telegram messages from YQ:

1. **Attempt `👀` reaction immediately** when a valid target `message_id` is available.
2. Treat the reaction as **receipt acknowledgement**, not task completion.
3. If reaction cannot be applied because the target id/context is missing or the provider rejects it, **say so briefly in the next reply** and still acknowledge in text.
4. Do not spam duplicate reactions to the same message.
5. Prefer reactions for:
   - new requests
   - confirmations
   - corrections
   - nudges like "hey?", "why didn’t you react?", "go ahead"
6. Do not rely on reaction alone when a real reply is needed.

## Operational semantics

### `👀` means
- seen
- acknowledged
- being handled or queued for handling

### `👀` does **not** mean
- completed
- approved
- safe
- successful

## Failure fallback

If the reaction attempt fails:
- keep the response short and direct
- acknowledge that the reaction failed due to message targeting/context
- continue with the work normally

## Priority

This doctrine is subordinate only to:
- tool/provider limitations
- safety constraints
- explicit user override

Otherwise it should be treated as a standing operating rule.

## Practical implementation rule

When a Telegram direct message from YQ arrives:
- first try `message.react` with `👀` on that exact message id
- then continue with the normal task flow
- if no usable message id exists, acknowledge in text and explain briefly only if the miss is noticeable or asked about
