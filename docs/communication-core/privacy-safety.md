# Privacy, retention, and approval policy

## PII

Search returns IDs and match explanations, never matched message bodies.
People with `pii_policy=restricted` have names, timezones, external IDs,
conversation titles/IDs, event external IDs/data/provenance, and message bodies
redacted from search/timeline results. Sync issue text contains only exception
type plus a fixed operation-failed message. Credentials, browser profiles,
cookies, session files, approval tokens, and raw account rate-limit state are
never emitted by the CLI.

Retention is configurable in `config.yaml`. Purge automation is not enabled by
this goal; operators must export/verify before applying a future retention
job. Raw platform records remain account-owned even when identities merge.

## Approval invariant

```text
draft -> review/edit -> exact approval -> atomic claim -> fake sink
      -> observed postcondition -> sent | failed | uncertain
```

An approval binds actor, person, source/target accounts, source/target
endpoints, route version, immutable recipient-preview hash, payload hash, and
TTL. Any relevant mutation invalidates it. Claiming and consuming approval is
one transaction. Expired claims become `uncertain` and cannot be blindly
retried. `sent` requires an observed postcondition.

Only `FakeCommunicationAdapter` may execute an outbox item. Facebook,
Telegram, VK, and dating adapters cannot send. Both Core production worker and
test sink are disabled by default; a test explicitly injects/enables the fake
sink. Real social/network writes and Telegram publication were not authorized
or performed.

Groups and smart segments produce explainable hashed previews. Drafts store a
copy of the preview, so later group/segment changes cannot silently expand the
recipient set. Mass outreach and hidden automation are forbidden.
