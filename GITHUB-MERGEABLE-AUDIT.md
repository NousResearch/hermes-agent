# GitHub-side mergeable audit (all 41 PRs) + origin/main rebases

GitHub-computed `mergeable` for every open arminanton PR against current
origin/main. **40 of 41 MERGEABLE; the 1 conflicting is #50111** (the manifest
PR — docs + residual patches, intentionally not a clean-merge code PR).

## 6 PRs were genuinely conflicting on current origin/main — now rebased

Prior rounds verified PRs clean on v0.17.0 (the goal target), but origin/main has
drifted PAST v0.17.0, so 6 PRs needed a rebase. Each was a 1-file complementary
conflict, resolved by keeping both sides (or the correct value):

| PR | conflict file | resolution | new head | mergeable |
|---|---|---|---|---|
| #50296 | agent/agent_init.py | kept `_end_session_on_close` + `_persist_disabled` | 084b79bed | ✅ |
| #49644 | hermes_cli/commands.py | union subcommands (max + full + clamp) | bda59d73b | ✅ |
| #50041 | hermes_cli/doctor.py | kept both MiniMax + Gemini OAuth checks | 9103989ba | ✅ |
| #50073 | hermes_cli/config.py | kept PR's new keys + main's hygiene limit (5000) | c6103ae4e | ✅ |
| #50064 | tests/.../test_provider_attribution_headers.py | kept both test functions | 1ad961f76 | ✅ |
| #50033 | agent/gemini_cloudcode_adapter.py | DU conflict; kept PR's new adapter | 44dc4d733 | ✅ |

All 6 verified: 0 conflict markers, 0 leaks, net diff scoped to the PR's own files,
compiles. After rebase: **40/41 MERGEABLE on current origin/main.**
