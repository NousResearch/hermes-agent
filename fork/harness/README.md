# Upstream Merge Harness

The **harness** keeps this fork aligned with `NousResearch/hermes-agent` without
losing local plugins, VRChat tooling, evolution vendors, or Windows shell fixes.

## Canonical paths (do not relocate)

| Path | Role |
|------|------|
| `scripts/merge_tools/` | Policy JSON, overlay appliers, conflict resolver |
| `scripts/sync_all.py` | Top-level sync orchestrator |
| `scripts/sync_ai_scientist_vendor.py` | AI-Scientist vendor pin refresh |
| `scripts/merge_tools/overlays/` | Three-way overlay payloads (e.g. ai-scientist templates) |
| `vendor/openclaw-mirror/` | Vendored OpenClaw + ShinkaEvolve + AI-Scientist pins |

Moving these directories breaks merge replay and scheduled vendor sync jobs.

## Policy file

`scripts/merge_tools/hermes-merge-conflict-strategies.json` classifies paths:

| Action | Meaning |
|--------|---------|
| `upstream` | Take official version (lockfiles, new upstream skills) |
| `preserve_custom` | Keep fork copy entirely |
| `official_with_overlay` | Merge upstream, then re-apply fork delta |
| `manual_api_followup` | Human/agent review required |

`overlay_sanitizers` on `toolsets.py` replays only fork tool names (VRChat, VOICEVOX, harness, …) after upstream reorders core bundles.

## Typical workflow

```powershell
py -3 scripts\sync_all.py --dry-run --allow-preflight-blockers
py -3 scripts\sync_all.py --merge --target main --allow-preflight-blockers
py -3 scripts\merge_tools\apply_post_merge_overlay.py --upstream-ref upstream/main
```

After merge, run targeted tests:

```powershell
scripts\run_tests.sh tests/tools/test_vrchat_osc_tool.py -q
```

## Generated artifacts (never commit)

- `vendor/openclaw-mirror/**/scripts/generated/*`
- `_docs/merge-reports/*`, `upstream-main-diff-inventory.*`
- `.worktrees/` merge scratch

See [`AGENTS.md`](AGENTS.md) for agent rules during conflict resolution.
