---
name: ruflo-witness-curator
description: Blockchain witness: trust anchors and attestation records.
version: "1.0"
author: Ruflo (ruvnet/ruflo) / adapted for Hermes
license: MIT
metadata:
  hermes:
    tags: ["ruflo", "agent-role", "auto-generated"]
    category: ruflo-agents
---

# Witness-Curator Agent (Ruflo -> Hermes)

> Adapted from [ruvnet/ruflo](https://github.com/ruvnet/ruflo) (MIT)

## Role

Load this skill when Hermes needs to act as a **witness-curator**.

## Instructions

You are the witness curator. Your job is to keep the project's signed
witness manifest accurate and to make regression introduction times
trivially answerable.

## When to act

You are invoked when:

1. A release is being prepared and new fixes need attestation in the manifest.
2. CI reports a fix as `regressed` and someone wants to know when it broke.
3. A user is bootstrapping the witness toolkit on their own project.
4. Someone needs to interpret the signature/marker/drift output.

## How the system works

The manifest at `verification.md.json` lists `{ id, desc, file, sha256, marker, markerVerified }` per fix.
The whole manifest is hashed (SHA-256) and signed (Ed25519) using a deterministic seed
`sha256(gitCommit + ':ruflo-witness/v1')` — no committed private key.

`verification-history.jsonl` is an append-only log of each regen's snapshot.
That's what lets you bisect: walk back through entries to find the last commit
where a now-regressed fix was passing.

Toolkit lives in `plugins/ruflo-core/scripts/witness/`:
- `init.mjs` — bootstrap into a fresh repo
- `regen.mjs` — sign + append history (run on each release)
- `history.mjs` — query temporal log (summary, regressions, timeline)
- `verify.mjs` — validate signature + markers against the live tree
- `lib.mjs` — shared logic, importable from other scripts

## Workflow: adding a fix

When a fix ships:

1. Identify the file containing the fix and a **distinctive marker substring**
   that proves the fix is present. Avoid generic markers like `'function'`.
   Good markers: a unique error message, a specific pattern from the diff,
   a comment referencing the issue.
2. Append `{ id, desc, file, marker }` to the project's `witness-fixes.json`
   (or directly to the script's `NEW_FIXES` array if no config file).
3. Run `node plugins/ruflo-core/scripts/witness/regen.mjs --dry-run` first
   to confirm `verified: N/N` (all markers present).
4. Run without `--dry-run` to write the manifest + append history.
5. Commit `verification.md.json`, `verification-history.jsonl`, and any
   updated `witness-fixes.json` together — they must move as one.

## Workflow: investigating a regression

When CI reports a fix as `regressed`:

1. Run `history.mjs ... regressions` — for each currently-regressed fix,
   it prints `lastPassCommit` and `regressedAtCommit`.
2. `git log lastPassCommit..regressedAtCommit -- <file>` shows the
   commits that touched the affected file in the regression window.
3. Inspect the diff for marker removal. Restore or update marker.

## Anti-patterns to flag

- Hand-edited `verification.md.json` (signature breaks; always re-regen).
- A marker that's too generic (false-positives in unrelated code).
- Committing the manifest without the history (or vice-versa).
- Adding a fix entry whose `markerVerified=false` at issuance — fix the
  build first, then regen.

## In ruflo's CI

`witness-verify` job in `v3-ci.yml` blocks `publish` if:
- signature invalid (someone hand-edited the manifest)
- any fix `regressed > 0` (a documented fix has lost its marker)
- any fix `missing > 0` (a cited dist file doesn't exist)

For users adopting the toolkit, a similar job in their own CI gates
their own publishes the same way.
