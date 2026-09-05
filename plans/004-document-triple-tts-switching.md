# Plan 004: Document triple TTS switching and runtime behavior

> **RE-SCOPED by Plan 005**: Docs should describe the single
> `agent-meow-voice` provider contract, not three Hermes-owned
> providers. This plan is kept for reference but should not be executed
> as written.

> **Executor instructions**: Follow this plan step by step. Run every verification command and confirm the expected result before moving to the next step. If anything in the "STOP conditions" section occurs, stop and report — do not improvise. When done, update the status row for this plan in `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat 7b162e2da..HEAD -- README.md INSTALL.md data/config.yaml docker-compose.yml docker-compose.upstream.yml docker-compose.windows.yml`
> If any in-scope file changed since this plan was written, compare the "Current state" excerpts against the live code before proceeding; on a mismatch, treat it as a STOP condition.

## Status

- **Priority**: P2
- **Effort**: S
- **Risk**: LOW
- **Depends on**: plans/001-make-voice-runtime-source-of-truth-explicit.md, plans/002-implement-real-tts-fallback-chain.md, plans/003-stabilize-qwen-0.6b-model-management.md
- **Category**: docs
- **Planned at**: commit `7b162e2da`, 2026-08-07

## Why this matters

Right now there is no concise, repo-local explanation of how to switch between Edge, Piper Chinese, and Qwen3-TTS, or what “default provider” versus “fallback provider” actually means. That gap is causing repeated operator confusion: users reasonably infer an automatic chain from config entries that are currently just three selectable providers. A short, accurate note in the root docs prevents repeated misconfiguration and lowers the support burden.

## Current state

- Relevant files:
  - `README.md` — root quick-start and top-level operational guidance.
  - `INSTALL.md` — installation/deployment notes.
  - `data/config.yaml` — the concrete triple-provider example.
- Current excerpts:
  - `README.md` has general TTS mentions but no guidance for `tts.provider`, `piper-zh`, `qwen3-tts`, or `zh-CN-XiaoxiaoNeural`.
  - `data/config.yaml` already contains the triple-provider config example.
- Repo conventions to follow:
  - Root docs are concise and task-oriented. Add a short “voice/TTS switching” note, not a long tutorial.
  - Deployment docs must match the actual compose/runtime lane selected in plan 001.

## Commands you will need

| Purpose                | Command                                                                                                         | Expected on success                          |
| ---------------------- | --------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| README grep            | `findstr /i "tts.provider piper-zh qwen3-tts zh-CN-XiaoxiaoNeural" c:\Users\1\github-pr\hermes-agent\README.md` | finds the new note                           |
| INSTALL grep           | `findstr /i "tts.provider piper-zh qwen3-tts" c:\Users\1\github-pr\hermes-agent\INSTALL.md`                     | finds any added install/runtime note         |
| Compose/runtime sanity | `docker compose -f c:\Users\1\github-pr\hermes-agent\docker-compose.upstream.yml ps`                            | services listed without doc/runtime mismatch |

## Scope

**In scope**:

- `README.md`
- `INSTALL.md` (only if the chosen runtime lane from plan 001 requires install-specific clarification)

**Out of scope**:

- Python/runtime source files
- Dockerfile
- TTS provider implementation logic

## Git workflow

- Branch: `advisor/004-doc-triple-tts`
- Keep this as one docs-only commit if possible
- Do not push or open a PR unless explicitly instructed

## Steps

### Step 1: Add a short root README note on provider switching

Add a compact section that explains:

- `tts.provider` selects the active provider
- `edge` is the fast online default
- `piper-zh` is the offline Chinese fallback/provider
- `qwen3-tts` is the offline neural provider and is slower on CPU
- `zh-CN-XiaoxiaoNeural` is the current default Edge Chinese voice

Keep it operational and brief.

**Verify**: `findstr /i "tts.provider piper-zh qwen3-tts zh-CN-XiaoxiaoNeural" ...\README.md` → all key terms found

### Step 2: Add install/deployment clarification only if needed

If plan 001 kept a pulled-image lane with startup-installed packages, add one short note in `INSTALL.md` clarifying whether Edge/Piper are baked into the image or installed at startup. If plan 001 moved to local-build lane, document that instead. Skip this step if the README note is sufficient and INSTALL.md would just duplicate it.

**Verify**: `findstr /i "piper-zh qwen3-tts" ...\INSTALL.md` → new note present only if you added it

## Test plan

- No code tests; validate via grep and one repo-specific compose status check.
- Confirm the wording matches the final implementation after plans 001–003, not the pre-plan state.

## Done criteria

- [ ] README has a short, accurate provider-switching note
- [ ] INSTALL only changed if runtime/install behavior needs clarification
- [ ] No inaccurate claim of automatic fallback if the chain has not landed yet
- [ ] No out-of-scope files modified

## STOP conditions

- Plans 001–003 are not done yet and the final runtime behavior is still unsettled
- The repo’s docs style has shifted elsewhere and the new note would need a different home

## Maintenance notes

- Reviewer should check the docs against live behavior, especially whether the chain is automatic or manual.
- Keep the root note short; detailed provider docs belong in the main docs site, not in the top-level README.
