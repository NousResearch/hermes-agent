# Harness — Agent Instructions

## Before any upstream merge

1. Read `scripts/merge_tools/hermes-merge-conflict-strategies.json` for the target paths.
2. Run `py -3 scripts/sync_all.py --dry-run` and read the inventory output.
3. Ensure working tree excludes `_docs/` and local logs from merge scope (listed in `dirty_tree_ignore`).

## During conflicts

- **Never** resolve `toolsets.py` by keeping the whole fork file — use overlay sanitizer replay.
- **Never** delete `scripts/merge_tools/` or shrink `preserve_custom` rules without operator approval.
- For `official_with_overlay` files (`pyproject.toml`, `web_tools.py`, `run_agent.py` overlaps):
  take upstream first, then apply minimal fork diff.
- Lockfiles (`uv.lock`, `package-lock.json`) default to **upstream** unless release engineering says otherwise.

## Vendor evolution pins

- AI-Scientist: `py -3 scripts/sync_ai_scientist_vendor.py --execute` (preserves `nc_kan` / `hermes_self_evolve` templates under `overlays/ai-scientist/`).
- OpenClaw layers: `scripts/merge_tools/openclaw_layered_sync.py` and `openclaw_vendor_layers.json`.

## Verification

After overlay apply:

```powershell
py -3 -c "import model_tools; model_tools.discover_builtin_tools()"
scripts\run_tests.sh tests/hermes_cli/test_config.py -q
```

## Fork docs

This folder (`fork/harness/`) is documentation only. Tooling remains under `scripts/merge_tools/`.
