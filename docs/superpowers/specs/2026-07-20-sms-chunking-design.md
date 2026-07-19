# SMS Long-Message Preservation Design

## Problem

`SmsAdapter.send()` already splits output into `MAX_SMS_LENGTH` chunks, but
it inherits `splits_long_messages = False`. `DeliveryRouter` therefore
truncates content above its 4,000-character platform cap before SMS receives
it, making the adapter's chunker unable to preserve the full message.

## Design

Declare `SmsAdapter.splits_long_messages = True`, matching other adapters
whose `send()` methods call `truncate_message()`. Keep existing chunk size,
Twilio request flow, audit saving, and failure handling unchanged.

## Verification

Add focused coverage proving:

- SMS advertises native long-message splitting;
- delivery routing passes content longer than 4,000 characters to SMS intact;
- the existing SMS sender divides that content into chunks no longer than
  `MAX_SMS_LENGTH`.

Run SMS and delivery-router tests through `scripts/run_tests.sh`.

## Non-goals

No configurable SMS length, Twilio API changes, retry changes, or new
dependencies.
