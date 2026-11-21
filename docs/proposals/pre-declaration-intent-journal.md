# Pre-Declaration Intent Journal for Autonomous Subsystem State Changes

> Status: **Proposal (draft for discussion)**. This note argues for a design,
> not a specific implementation.

## Summary

Hermes includes an asynchronous self-improvement subsystem that autonomously
modifies skill and procedural state: `SKILL.md` files, procedural libraries,
and related persistent artifacts. This note proposes a **pre-declaration
intent journal**: a machine-readable record that any autonomous subsystem
must write *before* it modifies persistent state.

## Problem

Autonomous self-improvement writes are sanctioned by the platform, but on disk
they look like any other write. Hosts that operate local integrity
verification regimes, such as hash manifests over their `skills/` and
`scripts/` directories, trip on every autonomous change. At detection time
such a host cannot distinguish a sanctioned autonomous change from tampering.
Attribution currently requires costly forensic reconstruction after the fact:
correlating file timestamps, session records, and subsystem activity to
determine which actor produced which change, and whether that actor was
authorized to produce it.

Both sides lose:

- **Hosts** face integrity alarms on routine platform behavior, and a triage
  path that is slow, manual, and low-confidence.
- **The platform** has its benign autonomous changes treated as suspected
  intrusions, which erodes trust in self-improvement as a capability.

## Proposal

Any autonomous subsystem that modifies persistent state **MUST** write a
machine-readable intent-journal entry **before** performing the operation. At
minimum, the entry contains:

| Field | Meaning |
| --- | --- |
| `actor` | Identity of the acting subsystem |
| `targets` | Exact paths of every file the operation will modify |
| `hash_before` | Expected content hash of each target before the operation |
| `hash_after` | Expected content hash of each target after the operation |
| `timestamp` | UTC timestamp of the declared intent |
| `context` | Identifier of the authorizing session or context |

Two properties are definitional:

- **Append-only.** Entries are never edited or deleted. Corrections and
  reversions are new entries.
- **Local to the host.** The journal lives on the host whose state it
  describes and is not transmitted anywhere by default.

Verification against an entry is byte-for-byte: the observed before and after
state of each target must match the hashes the entry pre-declared.

Out of scope: changes a human operator makes through normal, reviewed
mechanisms such as version control. The journal covers autonomous subsystem
writes specifically.

## Security rationale

The journal converts integrity-trip forensics from after-the-fact attribution
into **entry-vs-evidence verification**. When an integrity regime fires, the
operator checks the affected paths against the journal:

- A matching entry, one whose `hash_before` and `hash_after` correspond to
  the observed evidence, attributes the change to a declared actor and
  authorizing context. Triage becomes a lookup, not a reconstruction.
- A change with no matching entry, or an entry whose hashes do not match the
  evidence, is treated as presumptively unauthorized.

This protects both parties:

- **Hosts** get faster, higher-confidence triage. Declared changes resolve by
  verification rather than investigation, and undeclared changes stand out
  immediately instead of hiding in a crowd of ambiguous alarms.
- **The platform** gains a way to demonstrate good behavior. Benign
  autonomous changes stop looking like intrusions, because they arrive with
  pre-declared, verifiable intent.

## Limitations

This proposal is deliberately explicit about what the journal cannot deliver:

- **The journal is evidence of intent, not proof of safety.** A declared
  change can still be harmful. Pre-declaration provides attribution, not
  approval; it does not make any change safe.
- **A compromised subsystem could forge entries.** The journal is written by
  the same subsystems it describes, so an attacker in control of a subsystem
  can write a plausible-looking entry for a malicious change. The journal is
  therefore not a source of trust; it is a source of attribution and
  evidence. This is exactly why verification against each entry's claimed
  hashes must remain byte-for-byte, and why a matching entry reads as
  "declared by actor X in context Y", never as "safe".
- **The journal is not an access-control mechanism.** It does not replace
  sandboxing, least-privilege file permissions, or any other measure in a
  defense-in-depth posture. It is an auditability layer.
