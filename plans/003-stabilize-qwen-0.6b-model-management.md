# Plan 003: Stabilize Qwen3-TTS 0.6B model management

> **RE-SCOPED by Plan 005**: Qwen model management now belongs in
> `agent-meow` (the `QwenBackend` in `hermes_voice_backends.py` owns
> model selection and calls the existing host-side bridge). This plan is
> kept for reference but should not be executed as written.

> **Executor instructions**: Follow this plan step by step. Run every verification command and confirm the expected result before moving to the next step. If anything in the "STOP conditions" section occurs, stop and report — do not improvise. When done, update the status row for this plan in `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat 7b162e2da..HEAD -- scripts/qwen3-tts-server.py scripts/_dl_0.6b_mirror.py scripts/_dl_resume.py data/config.yaml`
> If any in-scope file changed since this plan was written, compare the "Current state" excerpts against the live code before proceeding; on a mismatch, treat it as a STOP condition.

## Status

- **Priority**: P2
- **Effort**: L
- **Risk**: MED
- **Depends on**: plans/001-make-voice-runtime-source-of-truth-explicit.md
- **Category**: dx
- **Planned at**: commit `7b162e2da`, 2026-08-07

## Why this matters

The repo currently has working Qwen3-TTS only through the already-cached 1.7B model. The 0.6B path is half-productized: there are ad-hoc downloader scripts, but no reliable, supported workflow that can survive flaky mirrors, validate completion, and switch the bridge over safely. As long as 0.6B is experimental, the fallback chain can include Qwen functionally, but latency and operability remain much worse than they should be.

## Current state

- Relevant files:
  - `scripts/qwen3-tts-server.py` — current bridge sidecar; defaults to the 0.6B model name but is routinely run against the cached 1.7B model.
  - `scripts/_dl_0.6b_mirror.py` — full-file mirror downloader.
  - `scripts/_dl_resume.py` — partial resumable downloader for selected files.
  - `data/config.yaml` — current command-provider config points to the bridge at `http://host.docker.internal:17494/tts`.
- Current excerpts:
  - `scripts/qwen3-tts-server.py:34-35` — default `_model_name = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"`
  - `scripts/qwen3-tts-server.py:153-156` — CLI flag also defaults to the 0.6B model name
  - `scripts/_dl_0.6b_mirror.py` and `scripts/_dl_resume.py` both target `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` through `https://hf-mirror.com`
  - Audit/runtime evidence: the live bridge was healthy on `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`, while the 0.6B download scripts remained ad-hoc and network-sensitive
- Repo conventions to follow:
  - Keep helper scripts small and operationally explicit; this repo already uses `scripts/` for operational utilities.
  - Do not couple the downloader to secrets or interactive prompts.

## Commands you will need

| Purpose                  | Command                                                                                                                                                               | Expected on success          |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| Syntax check bridge      | `python -m py_compile c:\Users\1\github-pr\hermes-agent\scripts\qwen3-tts-server.py`                                                                                  | exit 0                       |
| Syntax check downloaders | `python -m py_compile c:\Users\1\github-pr\hermes-agent\scripts\_dl_0.6b_mirror.py c:\Users\1\github-pr\hermes-agent\scripts\_dl_resume.py`                           | exit 0                       |
| Bridge health            | `Invoke-RestMethod -Uri "http://127.0.0.1:17494/health" -TimeoutSec 3`                                                                                                | returns JSON with model name |
| End-to-end bridge smoke  | `Invoke-WebRequest -Uri "http://127.0.0.1:17494/tts" -Method POST -Body '{"text":"你好世界"}' -ContentType "application/json" -OutFile "$env:TEMP\qwen_tts_test.wav"` | writes non-empty WAV         |

## Scope

**In scope**:

- `scripts/qwen3-tts-server.py`
- `scripts/_dl_0.6b_mirror.py`
- `scripts/_dl_resume.py`
- Optional new helper under `scripts/` for model integrity/state inspection
- `data/config.yaml` only if an explicit `qwen3-tts` model selector is needed

**Out of scope**:

- `tools/tts_tool.py` fallback logic
- Docker compose files
- README/INSTALL docs (handled in plan 004)

## Git workflow

- Branch: `advisor/003-qwen-0-6b-management`
- Commit one logical unit at a time
- Do not push or open a PR unless explicitly instructed

## Steps

### Step 1: Define what “0.6B ready” means

Make the downloader workflow explicit: which files are required, where they must land in the HuggingFace cache, and how the executor verifies the model is complete before switching the bridge away from 1.7B.

At minimum, define checks for:

- main model weights present and non-zero
- speech tokenizer weights present and non-zero
- bridge `/health` reports the intended model name after restart

**Verify**: `python -m py_compile ...` on all in-scope scripts → exit 0

### Step 2: Unify the two downloader scripts into one supported operational path

The executor should either delete one script and keep one authoritative downloader, or introduce a thin wrapper that defines when to use each behavior. Avoid leaving two partially overlapping operational scripts without a supported entrypoint.

Make partial-download state and resume behavior explicit, including where partial files are stored and how to distinguish “resumeable” from “corrupt.”

**Verify**: downloader script(s) compile and print a clear status message in a dry-run / inspect mode if one is added

### Step 3: Add model-integrity verification before bridge switching

Do not rely on “the download script exited” as proof of a good model. Add an explicit integrity/readiness check the operator can run before switching the bridge default from 1.7B to 0.6B.

If full hashing is too expensive, use file-existence + file-size thresholds + a load/health smoke test.

**Verify**: bridge health + non-empty output WAV after a 0.6B switch attempt

### Step 4: Make the 1.7B fallback explicit instead of implicit

If 0.6B isn’t ready, the operator should have a clear supported way to keep 1.7B live. Avoid implicit “just use whatever is cached.” The bridge startup path should make the selected model obvious in logs and health.

**Verify**: `/health` reports the actual running model; logs show the same model name

## Test plan

- Syntax-check all Python scripts touched.
- Run one bridge health check and one Chinese synthesis smoke test on the chosen model.
- If the network is still too flaky for a full 0.6B download, validate the fallback/inspect behavior instead of forcing a broken large download.

## Done criteria

- [ ] There is one supported 0.6B acquisition workflow, not two ambiguous scripts
- [ ] There is a documented machine-checkable readiness test for switching to 0.6B
- [ ] Bridge health clearly exposes the active model
- [ ] The operator can intentionally keep 1.7B if 0.6B is not ready
- [ ] No out-of-scope files modified

## STOP conditions

- The network path to the mirror/provider is too unreliable to complete a realistic test even after retry/resume work
- The 0.6B tokenizer/model requirements differ from the already-working 1.7B path in a way not visible from this repo alone
- Switching to 0.6B introduces quality or latency regressions that contradict the user’s actual goals

## Maintenance notes

- Reviewer should insist on a deterministic readiness check, not just “downloaded a bunch of files.”
- If the environment remains network-flaky, the right outcome may be “supported 1.7B fallback + external acquisition instructions,” not a perfect downloader.
