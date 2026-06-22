# Orphan resolution — final, evidence-backed (2026-06-22)

Re-derived from CURRENT src HEAD `9633e2362` vs v0.16.0 `3c231eb3`, against the
**current open-PR set** (39 code PRs + #50111 manifest). Supersedes the stale
round-10 "0 unmapped" table, which predated four maintainer-driven PR closures.

## Why the hunk-harness showed "159 unmapped / FAIL"
Two harness artifacts, NOT a coverage regression:
1. **It counts only OPEN PRs.** Four PRs closed during the run (#50039/#50555/#50657
   agy-cli, #50033 gemini-UA), so every hunk they covered flipped to "unmapped".
2. **Two privacy-scrub commits** (`37cccdeee`, `9633e2362`) edited *comment* lines in
   src that the PR heads still carry un-scrubbed → line-mismatch on cosmetic text.

At **file granularity** the real orphan set is **15 files**, every one legitimately
out of scope:

| Group | Files | Disposition | Evidence |
|---|---|---|---|
| agy-cli | agy_cli_client.py, gemini_native_adapter.py, plugins/model-providers/agy-cli/{__init__.py,plugin.yaml}, tests/agent/test_agy_cli_client_v2.py, _v3.py, tests/plugins/test_agy_cli_plugin_v2.py | **WITHDRAWN** — superseded by merged upstream #50454 (native google-antigravity OAuth) | @teknium1 on #50039/#50555/#50657 |
| gemini-UA | agent/google_user_agent.py | **WITHDRAWN (safety)** — follows #50492 | close comment on #50033 |
| subdirectory_hints | agent/subdirectory_hints.py, tests/agent/test_subdirectory_hints.py, tests/agent/conftest.py | **SUPERSEDED** — duplicate of #29433 (superset, already on origin/main) | @alt-glitch on #50049; guard present on main |
| transcripts | transcripts/C_{opus,sonnet}_{baseline,contradiction}.txt | **DISCARD** — eval-capture artifacts, same class as .bak/.project-intel | n/a |

## Net
- Real src delta vs v0.16.0: **144 files**.
- Covered by an open PR: **129**.
- Orphaned: **15** — all WITHDRAWN (8) / SUPERSEDED (3) / DISCARD (4).
- **Genuinely-contributable-and-missing: 0.**

## auth.py / runtime_provider.py specifically (the recurring question)
- The non-agy functions in our `hermes_cli/auth.py` that aren't on main are the
  gemini-cli OAuth trio (`get_gemini_oauth_auth_status`, `_mark_google_gemini_cli_active`,
  `resolve_gemini_oauth_runtime_credentials`) = WITHDRAWN, plus `_auth_file_mtime`
  (a 3-line `return _auth_file_path().stat().st_mtime` helper).
- `hermes_cli/runtime_provider.py` has **0** functions ours-and-not-on-main.
- The agy-registration hunks they carried belonged to the now-withdrawn #50039/#50657.
- Conclusion: nothing genuinely-contributable orphaned here.

## Maintainer-supersede caveat (recorded honestly)
`origin/main` (5ff11a689) has antigravity **skill docs** but no antigravity provider
`.py` and no antigravity registration in auth.py. #50454 is MERGED per GitHub but its
provider code is not visibly on main HEAD (possibly merged-then-reverted, or merged a
different surface). This does not change the disposition: the maintainer explicitly
rejected the agy-cli direction, so it stays withdrawn regardless of main's exact state.
