---
name: localization-worker
description: Run verified text-file localization workflows.
version: 0.1.0
author: Hyukjin, Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [Localization, Translation, QA, Documents]
    related_skills: [docx, xlsx, pdf]
---

# Localization Worker Skill

Translate UTF-8 text files through the `localization_worker` native-plugin toolset. The model translates only leased segment payloads; the plugin owns state, persistence, output paths, validation, and authoritative completion.

## When to Use

- Use for resumable `.txt` or `.md` file translation that requires a verified artifact.
- Do not use for casual sentence translation or formats not listed as supported in `references/formats.md`.

## Prerequisites

The `localization-worker` native plugin must be installed and enabled in a new Hermes session. Confirm that the eleven `localization_*` tools are available before creating a job. Source files must be located under a configured `plugins.entries.localization-worker.settings.source_roots` directory; the default is `<HERMES_HOME>/localization-input`.

The plugin supports Linux and macOS only. It requires POSIX directory-fd operations, `O_DIRECTORY`, `O_NOFOLLOW`, and `O_NONBLOCK`; registration fails closed when these primitives are unavailable. `<HERMES_HOME>` and its SQLite plugin-data directory are an administrator-owned trust boundary. Do not place them in an attacker-writable parent directory or replace them while Hermes is running.

## How to Run

Start a new Hermes session with the installed native plugin enabled, then invoke the registered `localization_*` tools through normal tool calls. Do not use `terminal` or direct filesystem tools to emulate the workflow. A run is eligible for success reporting only after `localization_verify_output` returns `COMPLETED` with a receipt.

## Quick Reference

```text
localization_create_job → localization_inspect_job → localization_extract_segments
→ localization_create_chunks → (localization_claim_chunk → localization_submit_chunk
→ localization_validate_chunk)×N → localization_assemble_output
→ localization_verify_output → localization_get_job_status
```

Terminal states: `COMPLETED`, `FAILED`, `BLOCKED`, `NEEDS_REVIEW`, `ABORTED`.

## Runtime Rules

- Use only the registered `localization_*` tools during a localization job.
- Never use shell commands, arbitrary code execution, or direct file writes to replace a failed localization tool.
- Never invent a replacement workflow when a tool fails.
- Stop when a job reaches `FAILED`, `BLOCKED`, `NEEDS_REVIEW`, or `ABORTED`.
- Preserve every returned segment ID and placeholder exactly.
- Never select an output path; the plugin owns its profile-scoped job directory.
- Never claim completion from reasoning, an external API response, or file existence alone.
- Report success only when `localization_verify_output` returns `COMPLETED` with a verification receipt.
- Treat missing, malformed, oversized, unsupported, or unverifiable results as failures.
- Network research may inform terminology but must not change job state or bypass validation.

## Procedure

1. Call `localization_create_job` with `source_path` and `target_locale`. Continue only from `CREATED`; record the returned `job_id` and authoritative `output_path`.
2. Call `localization_inspect_job`, then `localization_extract_segments`. Extraction returns bounded metadata, not source text. Stop on `UNSUPPORTED_FORMAT`, `SOURCE_CHANGED`, `EMPTY_DOCUMENT`, or any terminal state.
3. Call `localization_create_chunks`. Confirm returned `max_estimated_tokens` is at most 2,048. The response is bounded summary metadata; chunk IDs and source payloads are disclosed only by claims.
4. Call `localization_claim_chunk` with a stable `worker_id`. Translate only its returned segments before `lease_expires_at`.
5. Call `localization_submit_chunk` with the exact `chunk_id`, `fencing_token`, segment-ID set, and translated text.
6. Call `localization_validate_chunk`. Repeat claim, submit, and validation while the state remains `PROCESSING`.
7. Continue only when validation returns `VALIDATING`, which means every chunk is validated.
8. Call `localization_assemble_output`, then `localization_verify_output`.
9. Report completion only from the returned receipt. Later status checks can invalidate completion if the artifact changes.

## Pitfalls

- Only UTF-8 `.txt` and `.md` are currently supported. Markdown is treated as line-preserving text, not as a parsed Markdown AST.
- UTF-8 BOM, uniform LF or uniform CRLF framing, and final-newline presence are preserved for supported fixtures. Bare-CR, mixed newline documents, U+0085, U+2028, U+2029, vertical tab, and form feed fail closed with `UNSUPPORTED_NEWLINE_STYLE`.
- A single source line that exceeds the local 2,048-token estimate fails with `OVERSIZED_SEGMENT`; it is not split through placeholders or syntax.
- Leases expire after five minutes. A replacement worker receives a new fencing token; submissions using an old or expired token fail.
- Caller-provided `output_path` is rejected. Output artifacts remain under the active profile's plugin data directory.
- Source and output artifact paths are opened component by component from directory fds with no-follow and nonblocking final-open semantics. Traversal, ancestor replacement, symlink escapes, FIFOs, sockets, devices, and other non-regular files fail closed without blocking.
- Source text is disclosed only by a valid `localization_claim_chunk` lease. `localization_extract_segments` returns bounded counts and state metadata.
- Existing completed jobs are keyed by source content hash, source path, and target locale.

## Verification

A successful job has all of the following:

- every extracted segment translated and validated;
- exact segment-ID and placeholder preservation;
- an artifact in the profile-scoped job directory;
- successful UTF-8 reopen with BOM and newline framing preserved;
- content equal to the validated translations;
- an output SHA-256 stored with a verification receipt;
- authoritative state `COMPLETED`.

`localization_get_job_status` rechecks a completed artifact. Modification or deletion invalidates the receipt and returns `OUTPUT_CHANGED_AFTER_VERIFICATION`. The receipt is a profile-local workflow completion token stored with the SQLite job state; it is not a cryptographic signature for verification by an external party.
