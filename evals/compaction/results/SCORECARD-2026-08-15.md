# Compaction v2 — 4-transcript scorecard (2026-08-15, anchor-index build)

Four real 500K-token lineage transcripts from state.db (sweep campaign, GUI
desktop work, PR-merge campaign, ACP/PR review), 15-question recall exam
each. "recovery" = one session_search round-trip (FTS5+BM25 sim) against the
archived region. Lean build includes: 25K clamped tail, tail tool demotion,
chunked digests (noise-filtered, pristine tool contents), mechanical anchor
index, verbatim user messages, recovery footer, upgraded summarizer prompt.

## Results (recall % @ retained tokens)

policy            sweep          gui            prmerge        acp            AVG
uncompacted       93.3 @ 500K    96.7 @ 500K    96.7 @ 500K   100.0 @ 500K   96.7
current           93.3*@ 176K    26.7*@ 156K    33.3 @ 155K    30.0 @ 160K   45.8 @ 162K
lean              40.0 @  62K    60.0 @  41K    23.3 @  44K    36.7 @  50K   40.0 @  49K
lean+recovery     70.0 @  62K    80.0 @  41K    43.3 @  45K    80.0 @  50K   68.3 @  49K

* sweep/gui current scores are from the previous question banks (same
  transcripts; banks regenerated in the 4-way run). prmerge/acp are clean
  same-bank comparisons across all arms.

## Findings

1. LEAN+RECOVERY BEATS CURRENT BY +22.5pts ON AVERAGE (68.3 vs 45.8) AT 3.3x
   FEWER TOKENS (49K vs 162K). It wins on 3 of 4 transcripts and loses only
   sweep — the one transcript where current's fat tail got lucky with
   restated facts (93.3 is bank-inflated luck; see finding 3 of the previous
   scorecard).

2. THE ANCHOR INDEX FIXED THE NEEDLE-FACT CLASS. GUI closed-book went
   23.3 -> 60.0 and GUI+recovery 46.7 -> 80.0 after mechanically indexing
   exact identifiers (SHAs, ids, paths, error strings) instead of trusting
   the summarizer with them. ACP+recovery hit 80.0.

3. TWO FRESH TRANSCRIPTS CONFIRM CURRENT IS WEAK, NOT STRONG: 33.3 and 30.0
   at ~157K retained. The original sweep 93.3 was restatement luck, not
   policy quality. Current's average is 45.8% for 162K tokens — lean+recovery
   is 22 points better for less than a third of the spend.

4. prmerge IS THE HARD CASE for everyone (96.7 ceiling, best policy 43.3):
   1.1M-token lineage truncated at 500K, dense multi-PR state. Recovery
   misses there are mostly query formulation. Headroom, not a blocker.

5. Goal check (Teknium): tail = max(10K, 2.5%) ✓; summaries scoped to the
   compacted region only ✓ (sentinel tripwire test); session_search pointer ✓
   (+20-43pts measured); better accuracy AND more savings than current ✓
   (+22.5pts at 0.30x tokens).


## Codex CLI head-to-head (same transcripts, same exams, same judge)

Real OpenAI Codex CLI (v0.147.0, gpt-5.6-sol, 258K window) run end-to-end on
the identical four transcripts: chunk files read via `codex exec` until its
REAL auto-compaction fired (verified `compacted` event in the rollout jsonl;
peak context 455-483K), then quizzed post-compaction from memory with the
same 15-question banks and scored by the same judge.

policy                  sweep   gui    prmerge  acp    AVG     retained state
codex (real, post-cmp)  26.7%   40.0%  43.3%    36.7%  36.7%   ~4.5K (opaque blob + user msgs)
hermes current          93.3%*  26.7%* 33.3%    30.0%  45.8%   ~162K
hermes lean closed-book 40.0%   60.0%  23.3%    36.7%  40.0%   ~49K
hermes lean+recovery    70.0%   80.0%  43.3%    80.0%  68.3%   ~49K

Notes:
- codex answers from its own post-compaction session — the honest analog of
  our closed-book arms. It has NO session_search equivalent (its rollout is
  on disk but the agent cannot search it at runtime), so recovery has no
  codex counterpart; that gap is exactly the differentiator lean leans on.
- Apples-to-apples closed-book: lean 40.0% vs codex 36.7% — parity-plus at
  10x codex's retained state but 0.30x current's. With recovery: +31.6pts
  over codex.
- codex ties lean+recovery on prmerge (43.3%) — the dense multi-PR campaign
  is the hardest transcript for every policy and the clearest iteration
  target.
- Methodology caveats: codex ingested transcripts as FILE READS (tool
  outputs), not native conversation — this matches how its compaction
  treats tool output (drops it all into the server-side summary) but is not
  byte-identical to a native session. Its model (gpt-5.6-sol) also differs
  from the answering model in our arms; scores compare COMPACTION PIPELINES
  end-to-end, not models in isolation. One codex quiz reply was also
  capped short (~1K chars for 15 answers), which its terse post-compaction
  style invites.

## Recommendation

Ship lean as opt-in (compression.tail_mode: lean, legacy default), harness as
the permanent gate. Iterate prmerge-class recall behind the flag (query
mining, per-epoch anchor windows) before default flip.


## Appendix: full per-transcript detail

### Transcript: sweep

| policy | recall | tokens before | tokens after | compress s |
|---|---|---|---|---|
| lean | 40.0% | 499,625 | 61,567 | 114.9 |
| lean+recovery | 70.0% | 499,625 | 61,792 | 114.8 |

<details><summary>15 exam questions (questions-30b95351c7.json)</summary>

1. **What is the reason given for never using 'git checkout pr-branch -- <file>' on stale branches?**  
   gold: `the stale file version silently deletes newer main code`
2. **According to the transcript, how much RSS memory does the gateway balloon to every ~2h in the regression reported in issue #81625?**  
   gold: `~60GB`
3. **Which specific Electron setting is suspected of causing the Windows occlusion freeze in issue #83420?**  
   gold: `backgroundThrottling`
4. **What exact error message is returned when 'gh pr merge --auto' is attempted on the NousResearch/hermes-agent repository?**  
   gold: `Auto merge is not allowed for this repository (enablePullRequestAutoMerge)`
5. **What is the specified 'Rule 0' that must be included in a subagent brief?**  
   gold: `load the skill first`
6. **In the July 2026 title-cluster sweep, what was the title of the missed first submitter PR #35416?**  
   gold: `add config gate for title generation`
7. **Which file path is noted as containing the #34034/#28149 manifest guard 'test_bundled_plugin_manifests_ship_in_both_wheel_and_sdist'?**  
   gold: `tests/test_packaging_metadata.py`
8. **What was the result of the 'npm ci' command run in /home/teknium/salv-desktop according to the background process notification?**  
   gold: `completed normally (exit code 0)`
9. **What was the 'Root Cause A' identified for why 'uv sync --extra all --locked' failed daily in issue #79434?**  
   gold: `relative exclude-newer makes the committed lock stale every day`
10. **How many tasks are reported as done in the 'fangliquanflq' desktop retry truncation PR #86605?**  
   gold: `13`
11. **In the 'salv-cron' worktree, what was the exit code when the agent tried to execute a 'BLOCKED (hardline)' command?**  
   gold: `-1`
12. **What is the full title block text for the technical schematic infographic generated for the Gateway Drain?**  
   gold: `GATEWAY DRAIN × CRON — SHUTDOWN CONTRACT`
13. **Which PR number's watcher reported '=== ALL GREEN (streak=1, checks=46) ===' at [03:56:19]?**  
   gold: `82980`
14. **What is the specific Gist ID created for the PR infographic host in the cron cluster?**  
   gold: `ee33edd5804689243f974536ef7aecb9`
15. **What was the final merge SHA for Cluster D's Trigger-now PR #70638?**  
   gold: `f9d64b9a9d8b306f64851c1a13869d96ad5d7869`

</details>

<details><summary>15 exam questions (questions-5be475cde0.json)</summary>

1. **What exact command did the agent use to search for open issues related to a specific topic during Phase 1 of the cluster-sweep salvage?**  
   gold: `gh issue list --search "<topic>" --state open --limit 100 --json number,title`
2. **According to Teknium's design intent, what is the status of 'platform toolsets' in the codebase?**  
   gold: `platform toolsets are vestigial, never exposed`
3. **During the July sweep, which specific issue's config bridge was found to already exist at the exact line it was claimed to be missing?**  
   gold: `#32263`
4. **In the Aug 2026 cron-summarizer cluster sweep, which two PR numbers were discovered post-merge as the true first submitters?**  
   gold: `#60593, #61969`
5. **What is the recommended Git command to find when a specific symbol fix landed on the main branch?**  
   gold: `git log -S "<symbol>"`
6. **Why did the #39719 salvage silently delete 236 lines of code from cli-config.yaml.example?**  
   gold: `the stale file version silently deletes newer main code`
7. **What is the rule for salvaging commits with placeholder identities like 'pwn@example.com'?**  
   gold: `do NOT cherry-pick. Surgical reapply as maintainer-authored commit, Co-authored-by the GitHub PR author`
8. **How should an agent handle a 'gh pr merge' 502 error?**  
   gold: `retry the same command once after the "Merge already in progress" settles (~45s); check PR state between attempts`
9. **Which two properties shape almost every design decision in Hermes according to the Development Guide?**  
   gold: `Per-conversation prompt caching is sacred and The core is a narrow waist; capability lives at the edges.`
10. **What error message does the live-checkout git guard display when blocking a history-rewriting command?**  
   gold: `Blocked: `git <op>` would rewrite Hermes's live source checkout (/home/teknium/.hermes/hermes-agent) and can mix module `
11. **What happened to the Desktop cluster's 'npm ci' command that resulted in an error writing to /tmp/ccH06T4r.s?**  
   gold: `No space left on device`
12. **What was the GraphQL API rate limit remaining for the user when the 'API rate limit already exceeded' error first occurred?**  
   gold: `0`
13. **Which PR was identified as the salvage of HexLab98's #85283 to fix hung inline API calls?**  
   gold: `#86645`
14. **Why did PR #79268 fix invisible overlays in the TUI?**  
   gold: `renderNodeToOutput skips boxes Yoga squeezes to height 0`
15. **What was the specific ModuleNotFoundError message caused by the wheel subpackage discovery trap in #34701?**  
   gold: `ModuleNotFoundError: No module named 'hermes_cli.dashboard_auth'`

</details>

### Transcript: gui

| policy | recall | tokens before | tokens after | compress s |
|---|---|---|---|---|
| lean | 60.0% | 499,818 | 41,232 | 118.1 |
| lean+recovery | 80.0% | 499,818 | 41,306 | 115.2 |

<details><summary>15 exam questions (questions-36d3d87e0b.json)</summary>

1. **What is the PR number for the authored fix addressing mid-turn message ordering bugs in Hermes Desktop?**  
   gold: `#86617`
2. **According to the contribution rubric in AGENTS.md, which type of config belongs in '.env' and which belongs in 'config.yaml'?**  
   gold: `.env is for secrets only (API keys, tokens, passwords). All behavioral settings... go in config.yaml.`
3. **What specific file and line number were identified as the cause of an AssertionError (assert 56 == 55) in the Python tests?**  
   gold: `tests/hermes_cli/test_session_recovery_lost_and_found.py:327`
4. **What was the root cause of issue #73793 regarding mid-turn message rendering?**  
   gold: `redirect/steer paths spliced the mid-turn user bubble BEFORE the active assistant stream row`
5. **Which PR was verified to already be on 'main', resulting in nothing needing to be salvaged for it?**  
   gold: `#84287`
6. **In the Desktop virtualized-scrolling cluster, what was the fix for issue #79157 (scrollbar unclickable)?**  
   gold: `pane sash grab band made asymmetric 1px/7px`
7. **Which contributor's email was mapped to 'baihemax' during the attribution audit of PR #86588?**  
   gold: `602028@ky-tech.com.cn`
8. **What error message does the Hermes terminal tool return when a git command is blocked to prevent rewriting the live source checkout?**  
   gold: `Blocked: `git <op>` would rewrite Hermes's live source checkout`
9. **What is the core design principle regarding 'Narrow Waist' in Hermes development?**  
   gold: `The core is a narrow waist; capability lives at the edges.`
10. **What was the result of the rebase-merge attempt for PR #86589?**  
   gold: `GraphQL: Pull Request has merge conflicts (mergePullRequest)`
11. **In the infographic style picker, what vibe is associated with the 'designers-republic' style?**  
   gold: `The Designers Republic: flat orange+violet vector schematic on pewter grey`
12. **Why was PR #76286 excluded from the compaction/compression transcript-visibility cluster?**  
   gold: `conflicts with main in 4 files and introduces a second competing display-dedupe scheme`
13. **What is the 'Provenance note' date for the pr-infographic-workflow.md reference file?**  
   gold: `May 23 2026`
14. **What specific TypeScript error caused PR #86772 to fail CI linting after a rebase?**  
   gold: `Property 'onToggleUnread' is missing in type`
15. **According to the Desktop Engineering Guide, who is the authority for process lifecycle and the native filesystem?**  
   gold: `Electron`

</details>

<details><summary>15 exam questions (questions-9c55c707b6.json)</summary>

1. **What two PR numbers are associated with the 'sidebar-nav-rows-and-overlay-panels.md' and 'hud-mode-internals.md' references in the initial tool content?**  
   gold: `#85162 and #82285`
2. **According to AGENTS.md, what is the 'one exception' to the rule that nothing should rebuild the system prompt mid-conversation?**  
   gold: `context compression`
3. **In the Contribution Rubric, what are the three allowed reasons for an automated triage sweeper to close a PR?**  
   gold: `implemented_on_main, cannot_reproduce, incoherent`
4. **Which contributor is credited with adding the 'Brazilian Portuguese localization' in PR #86292?**  
   gold: `@gui8515`
5. **What specific error message is reported in issue #83562 regarding the Windows Desktop update?**  
   gold: `Hermes backend exited (0)`
6. **What is the 'core problem' identified in the parallel-subagent-salvage-orchestration.md reference?**  
   gold: `subagents share the parent's worktree + main checkout`
7. **Why was the 'nix (macos-latest)' build failing in the salvage batches according to the orchestration reference?**  
   gold: `Nix build failed due to stale npm lockfile hash`
8. **Which subagent ID was assigned the goal of salvaging the 'inflight-journal duplicate-answer cluster'?**  
   gold: `sa-2-7318d0ba`
9. **In PR #86595, why was PR #80707 by upperagent excluded from the salvage?**  
   gold: `violating this PR's UI-read-only invariant`
10. **What was the root cause of the failure in Python tests slice 4/12 for PR #86597?**  
   gold: `AssertionError: assert 't2' == 't1'`
11. **What did the fix for issue #79157 in PR #86589 involve?**  
   gold: `pane sash grab band made asymmetric 1px/7px`
12. **According to the root cause analysis for #73793, which two files spliced the mid-turn user message at streamIndex?**  
   gold: `use-prompt-actions/index.ts and session-tile-actions.ts`
13. **What was the head SHA for the 'salvage/desktop-busy-state' branch in PR #86604?**  
   gold: `bddadfe9e21e24b3d52e2b15f138c42474dede42`
14. **Why was the merge of PR #86589 aborted during the 'Merge all' command?**  
   gold: `GraphQL: Pull Request has merge conflicts (mergePullRequest)`
15. **What specific file was modified to fix the 'artifacts page timestamps render 1970' issue via PR #86749?**  
   gold: `apps/desktop/src/app/session/hooks/use-session-actions/utils.ts`

</details>

### Transcript: prmerge

| policy | recall | tokens before | tokens after | compress s |
|---|---|---|---|---|
| uncompacted_control | 96.7% | 499,663 | 499,663 | — |
| current | 33.3% | 499,663 | 155,399 | 14.9 |
| lean | 23.3% | 499,663 | 44,419 | 105.4 |
| lean+recovery | 43.3% | 499,663 | 44,977 | 95.8 |

<details><summary>15 exam questions (questions-703ae2774a.json)</summary>

1. **Which PR number added the public subagent lifecycle API?**  
   gold: `#63359`
2. **What is the name of the typed service added to PluginContext for launching and monitoring child sessions?**  
   gold: `subagent_lifecycle`
3. **How many contract and security tests were included with the subagent lifecycle API PR?**  
   gold: `42`
4. **What specific gap was identified regarding the `ctx.inject_message()` function in gateway sessions?**  
   gold: `cannot currently trigger a turn in an existing gateway session`
5. **Which PR implements gateway-safe plugin injection by extending `ctx.inject_message()` with a keyword-only `session_key`?**  
   gold: `#64436`
6. **What are the two specific constraints placed on redaction patterns in the pattern registry to prevent exposing data?**  
   gold: `must compile, must start with ≥2 literal characters`
7. **Which contributor authorized sustained help for the Phase 0–1 expansion track?**  
   gold: `Daniel`
8. **What is the issue number for the disposition gap concerning `pre_command` middleware and MCP tool access?**  
   gold: `#64204`
9. **What configuration setting is required to opt-in to reasoning deltas in streaming output?**  
   gold: `plugins.stream_reasoning_deltas: true`
10. **How many additions and across how many files were made in PR #63359?**  
   gold: `650 additions across 4 files`
11. **What is the name of the reference plugin shipped with the redaction pattern registry?**  
   gold: `nvapi-redaction`
12. **List the four observer-only streaming output plugin hooks added in PR #64317.**  
   gold: `on_stream_start, on_stream_delta, on_stream_end, on_interim_message`
13. **What was addressed in the update to PR #58541 regarding lifecycle hooks?**  
   gold: `created-hook timing and added kanban_task_promoted`
14. **Which sub-issue number is associated with the 'developer tooling' (scaffold + Plugin Doctor + test harness)?**  
   gold: `#64230`
15. **What was the Round 3 review's outcome for PR #63359 and @asimons81?**  
   gold: `sub-issue #65447`

</details>

### Transcript: acp

| policy | recall | tokens before | tokens after | compress s |
|---|---|---|---|---|
| uncompacted_control | 100.0% | 498,906 | 498,906 | — |
| current | 30.0% | 498,906 | 160,223 | 15.8 |
| lean | 36.7% | 498,906 | 49,523 | 143.3 |
| lean+recovery | 80.0% | 498,906 | 49,721 | 135.6 |

<details><summary>15 exam questions (questions-f45358df19.json)</summary>

1. **What was the specific reason Teknium gave for reverting PR #30179 in July 2026?**  
   gold: `WTF??? REVERT! DAMMIT`
2. **On which specific PR did Teknium say, 'tf are you saying to me. Stop giving me such random verbose details'?**  
   gold: `PR #6391`
3. **Which file path should be checked for the canonical list of provider models?**  
   gold: `hermes_cli/models.py`
4. **What was the identified bug in PR #2314 regarding provider names?**  
   gold: `checking for "alibaba-coding-plan"`
5. **What is the mandatory line limit for PR reviews requested by Teknium?**  
   gold: `<= 15 lines`
6. **What exact error message did the agent receive when attempting to checkout a worktree while in the live source directory?**  
   gold: `Blocked: `git checkout` would rewrite Hermes's live source checkout (/home/teknium/.hermes/hermes-agent) and can mix mod`
7. **Why was PR #74658 necessary to fix Slack 'broken on main'?**  
   gold: `SlackResponse isn't a dict subclass, so every gate is always False.`
8. **What was the final merge commit SHA for the Slack SDK response fix on main?**  
   gold: `24ba86627515ad5fda69a39ef338c365713448bc`
9. **In the 'Pop-laboratory' style infographic for the Auxiliary Client fix, what were the two specific outcomes shown in cell 2?**  
   gold: `Messages wrapper keeps /anthropic and OpenAI fallback keeps /v1`
10. **What specific SQL update was added to the migration path in hermes_cli/kanban_db.py to prevent losing active wake on upgrade?**  
   gold: `UPDATE kanban_notify_subs SET delivery_mode = 'notify+wake' WHERE platform != 'tui'`
11. **Which test failed in CI slice 5/12 for the kanban delivery modes PR?**  
   gold: `tests/gateway/test_kanban_notifier_apiserver_wake.py::test_apiserver_sub_wakes_real_session_via_self_post`
12. **According to the transcript, why is squash merging banned as of July 2026?**  
   gold: `DevOps policy`
13. **Which contributor authored the first fix for issue #73030 in July?**  
   gold: `@Tranquil-Flow`
14. **What was the 'Superman-style' shield error in the first generation of the Kanban infographic?**  
   gold: `red "S" inside the diamond shield`
15. **What specific file was modified to add the 'scope_id_for_chat' method for Slack?**  
   gold: `plugins/platforms/slack/adapter.py`

</details>

## Methodology notes

- Transcripts: 4 real session lineages reconstructed from a state.db copy
  (sweep campaign 42 rotations / GUI desktop 34 / PR-merge 17 / ACP review
  17), chronological 500K-token prefix, tool-group aligned.
- Question generation: main model, from the region the CURRENT policy would
  summarize (most conservative boundary), cached per transcript so every
  policy answers the identical exam.
- Answering: fresh LLM sees ONLY the post-compaction context (closed-book) or
  context + one FTS5+BM25 search round-trip over the archived region
  (+recovery). Judge sees gold; answerer never does. Scoring 2/1/0.
- Known caveats: 15 questions/transcript => +-1 question ~ 3.3pts noise;
  sweep/gui current-policy rows predate a question-bank regeneration
  (prmerge/acp are same-bank across all arms); the recovery sim conservatively
  approximates production session_search (same engine, no windowing).
- Cost shape: lean compaction = ~25 aux-model digest calls (~2min, one-time
  per compaction) vs 1 call today; every post-compaction turn is ~110K input
  tokens cheaper. Break-even ~1 turn.
