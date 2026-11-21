# Validate

When the user asks whether `sprint-status.yaml` is well-formed, run:

```
uv run {skill-root}/scripts/sprint_plan.py validate \
  --status-file {implementation_artifacts}/sprint-status.yaml
```

Never writes; exits 0 whether valid or not. Report `valid` in one line. If `problems` is non-empty, list them plainly (each names the key or field at fault) and offer the fix flow (`fix-sprint-status.md`). If `legacy_mapped` is non-empty, note the file still uses v6 status names and that any regenerate will rewrite them to the modern vocabulary — progress is preserved either way.
