---
failed_layers: '' # set at runtime: comma-separated list of layers that failed or returned empty
---

# Step 2: Review

## RULES

- YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`
- All review subagents must run at the same model capability as the current session.
- Run subagents synchronously: launch them together as blocking calls awaited in this turn — never backgrounded or detached, never ending the turn to await results.

## INSTRUCTIONS

1. The review layers are `{workflow.review_layers}`, resolved during activation.

2. For each layer in `{workflow.review_layers}`:
   - `instruction` empty or missing → drop the layer silently (an override disabled it).
   - `when` condition present and not satisfied by the current context (`{review_mode}`, `{spec_file}`) → drop the layer and tell the user, e.g. "Acceptance Auditor skipped — no spec file provided."
   - otherwise → the layer is active.

   If no layer is active, HALT with status `blocked` and blocking condition `no active review layers`.

3. Announce skipped layers first, then launch every active layer before handling any layer's result. Try running all active layers simultaneously: expand `{skill-root}` in each layer's `instruction` to this skill's absolute installed directory, then substitute the runtime placeholders (`{diff_file}`, `{claims_file}`, `{spec_file}`). `{diff_file}` is a path: substitute the path itself and let the layer read the file — a launch prompt never carries diff text. For an instruction that launches a reviewer subagent, launch that child with the prompt text after placeholder substitution; do not load the reviewer instruction file yourself. For any other customized instruction, execute it as written. Do not leave `{skill-root}` unresolved in a child prompt, and resolve `{diff_file}` to an absolute path — the child's working directory is not yours. If a layer's instruction requires subagents and subagents are not available, for each such layer write under `{implementation_artifacts}` that layer's child prompt with everything after its content label replaced by the contents of `{diff_file}` (not a path-only pointer) — that session may not share this filesystem, so its prompt must be self-contained. Then HALT. Ask the user to run each in a separate session (ideally a different LLM) and paste back the findings. When findings are pasted, treat them as those layers' findings and resume from this point. This is the only allowed parent-side read of a reviewer instruction file.

4. **Layer failure handling**: If any layer fails, times out, or returns empty results, append the layer's `name` to `failed_layers` (comma-separated) and proceed with findings from the remaining layers.

5. Collect all findings from the completed layers, keeping track of each finding's originating layer `id`.

## NEXT

Read fully and follow `./step-03-triage.md`
