# Nous Branding — Pitfalls & Gotchas

## 1. Style-Only Reference Matching (CRITICAL)

**The #1 mistake:** Copying the subject matter from reference images instead of just the aesthetic.

The reference guides ONLY:
- Color palette and grading
- Lighting quality and atmosphere
- Composition approach and framing
- Level of detail and finish

The reference does NOT guide:
- What subjects to depict
- What figures to include
- What scenes to compose
- Any specific objects or poses

**Every prompt MUST include** explicit "original scene" + "do NOT copy" language. See the Prompt Templates section in SKILL.md for the required phrasing.

**Signal it happened:** User says "this looks just like the reference" or rejects images for being derivative.

## 2. `keep` CLI SQLite Database Failure

Codex sessions may hang after image generation due to the `keep` reflective-memory CLI failing with `unable to open database file` (SQLite error). This is a **known issue** — the image generation itself succeeds, but the post-generation cleanup/reflect step gets stuck.

**Impact:** The `codex exec` process may not exit cleanly. Background bash scripts using `wait` may hang for minutes after images are actually done.

**Mitigation:** Don't rely on process exit as the signal that images are ready. Instead, poll the output directory directly:
```bash
ls -lt ~/.codex/generated_images/ | head -5
find ~/.codex/generated_images/ -name 'ig_*.png' -m10
```

Or use a post-hoc collection approach: run generations, then after a reasonable delay, copy the latest `ig_*.png` from the most recent session directories.

**Do NOT** try to fix the `keep` database — this is a codex-internal issue, not something the branding skill should handle.

## 3. Parallel Generation Output Conflicts

When running multiple `codex exec` sessions in parallel (background `&` + `wait`), each writes to its own session directory under `~/.codex/generated_images/`. However, the "latest session" heuristic (`ls -lt | head -1`) can race — two sessions may write at the same time and you may grab the wrong output.

**Mitigation:** Give each parallel task its own unique work directory and wait for a specific session ID rather than using the global "latest" heuristic. Or run sequentially if fidelity matters more than speed.

## 4. iCloud Placeholder Files

Reference images on `~/Desktop/` or `~/Documents/` may be iCloud File Provider placeholders — appearing to exist but returning `EDEADLK` or "Resource deadlock avoided" when read.

**Always verify** with `file /path/to/reference.png` before use. Copy to `/tmp/` first.

## 5. Codex Model Selection

Codex may choose different models (gpt-5.5, gpt-image-2, etc.) across sessions. Results vary in quality and style consistency. If a generation doesn't match expectations, retry with an explicit model hint or adjust the prompt's level of specificity.
