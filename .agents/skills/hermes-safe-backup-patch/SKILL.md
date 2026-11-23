---
name: hermes-safe-backup-patch
description: Use when creating pre-edit backups preserving relative paths to ~/.hermes/local-patches/one-dark/<timestamp>-<slug>/, generating and verifying SHA256SUMS, and making focused local commits on feature branches.
---

# Hermes Safe Backup and Patch Workflow

## Overview
This skill defines the mandatory pre-edit backup, integrity verification, and atomic commit workflow for Hermes Desktop/Webapp patches and theme adjustments.

Based on: project-specific (Hermes Desktop/Webapp).

## Actionable Steps

### 1. Pre-Edit Backup
Before modifying any source, style, or test file:
- Copy the original file preserving its relative path into a durable patch directory:
  `$HOME/.hermes/local-patches/one-dark/<timestamp>-<slug>/`
- Generate a `SHA256SUMS` manifest containing the checksums of all backed-up files.

### 2. Verify Integrity
Verify the integrity of the backup manifest before making any code edits:
```bash
sha256sum -c SHA256SUMS
```
Confirm all files report `OK` before proceeding.

### 3. Feature Branch & Atomic Commits
- **Never commit directly to `main`**. Always work on a dedicated local feature or fix branch.
- Make **one focused commit per change** using conventional commit prefixes (`feat(desktop):`, `fix(desktop):`, `docs:`).
- **Pre-commit hooks (`lefthook`)**: Docs (`.md`) and stylesheets (`.css`) must pass without `--no-verify`. `--no-verify` is permitted **only** on `.ts`/`.tsx` commits where bootstrap limitations apply, **only** with explicit maintainer approval, and must be documented in the commit message body.

## What this skill does NOT do
- Never commits or pushes directly to `main`.
- Never pushes or opens Pull Requests automatically without explicit operator approval.
- Never overwrites existing backup directories or checksum manifests.
