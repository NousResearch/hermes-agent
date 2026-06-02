# DOC_POLICY

## Purpose
- Keep project docs maintainable as project count grows.

## Rules
1. Keep root `.md` files minimal; prefer `docs/` for new long-form docs.
2. Any new operational doc must be registered in `docs/DOC_INDEX.md`.
3. Run rollover check before ending major phases:
   - `bash /Users/kevin/codex/harness/scripts/doc-rollover.sh --check`
4. If rollover is required, apply it and record archive paths in handoff.
5. Do not store secrets in markdown files.
