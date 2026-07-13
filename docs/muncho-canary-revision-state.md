# Muncho canary revision-state gate

This runbook defines the bounded gate between sealed Canonical Writer canary
revisions. It does not grant cleanup authority. Its purpose is to distinguish
a genuinely fresh host from an observed, digest-classified transition without
ever guessing what an unfamiliar file means.

## Current verified state

The read-only Cloud inventory on 2026-07-13 found no activation plans,
staged inputs, installed canary units or configs, or activation receipts under
the fixed activation paths. The earlier release remains only in its own exact
revision directory below `/opt/muncho-canary-releases` and in source history.
This observation is not permanent deployment truth; repeat the read-only gate
immediately before every release or activation.

No transition or cleanup mutation is required for this state.

## Release namespace contract

Every sealed release lives at:

```text
/opt/muncho-canary-releases/<exact-40-character-git-revision>
```

The release builder creates that directory with no-replace semantics. An
existing exact revision is never resumed, repaired, overwritten, or reused,
including an incomplete or previously invalidated release. A different exact
revision gets a different directory, so retained historical releases may
coexist read-only with the new release.

Retention is evidence, not authority: an old directory does not authorize its
interpreter, manifest, plan, approval, or receipt for a new revision.

## Fixed-path fresh-state gate

Before the new revision's config collector or planner writes anything, collect
only existence, ownership, mode, size, and SHA-256 evidence for these bounded
coordination and install paths. Do not print file contents.

Run this gate only after the dedicated canary host identity is freshly
attested. A missing or different host binding is `blocked`; the same path names
on the production host are never transition inputs.

```text
/etc/muncho/writer-activation/staged/writer.json
/etc/muncho/writer-activation/staged/gateway.yaml
/etc/muncho/writer-activation/staged/native-observation-plan.json
/etc/muncho/writer-activation/staged/activation-plan.json
/etc/muncho/writer-activation/staged/owner-approval.json
/etc/muncho/writer-activation/staged/external-iam-receipt.json
/etc/muncho/writer-activation/staged/muncho-canonical-writer.service
/etc/muncho/writer-activation/staged/hermes-cloud-gateway.service
/etc/muncho/writer-activation/native-observation-plan.json
/etc/muncho/writer-activation/activation-plan.json
/etc/muncho/writer-activation/deployment-manifest.json
/etc/systemd/system/muncho-canonical-writer.service
/etc/systemd/system/hermes-cloud-gateway.service
/etc/systemd/system/muncho-canonical-writer-export.service
/etc/tmpfiles.d/muncho-canonical-writer.conf
/etc/muncho-canonical-writer/writer.json
/etc/hermes/config.yaml
```

The gate has only these outcomes:

1. **Fresh.** Every fixed path is absent. Retained revision-namespaced release
   and append-only evidence directories are allowed. Continue with the new
   exact sealed release.
2. **Exact idempotent state.** A higher-level approved runtime gate may accept
   a file only when its bytes and complete plan binding equal the current exact
   approved revision. Mere filename, revision prefix, or matching schema is
   insufficient.
3. **Blocked.** Any other existing path produces a `blocked` report containing
   metadata and digests only. Stop before config collection, planning, unit
   installation, or service start.

The planner already creates fixed staged plans and units exclusively: a
collision raises an error and leaves the existing bytes untouched. Operators
must not work around that error with `rm`, `mv`, broad directory cleanup, or a
new plan that silently replaces the old file.

## If a future collision is real

First collect an exact read-only inventory and trace every byte digest to its
sealed release, plan, approval, and receipt. Only then design an owner-approved
transition whose allowlist contains those observed paths and digests. Such a
transition must archive exact bytes into a revision- and digest-addressed
namespace, append a transition receipt, re-attest service inactivity, and be
idempotent across interruption.

Do not add generic cleanup authority in advance. Unknown, orphaned, drifted,
or cross-revision files remain blocked until their provenance is understood.
Append-only approval, collector, native-observation, root-preflight,
activation, route-back, and Canonical Brain event evidence is never cleanup
input and is never deleted or rewritten by this gate.

Credential files, database contents, Discord tokens, private keys, and
production runtime paths are outside this gate. No receipt may contain secret
content or a secret-content digest.
