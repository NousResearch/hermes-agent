# Closure justifications — why closing #50484/#50487/#50049/#50457-agy lost no coverage

## #50487 (residual drift, 14 files) — CLOSED redundant
Every file's overlay behavior is delivered by a topical feature PR that applies
clean on v0.17.0. Per-file owner mapping:
- agent/conversation_loop.py, cli.py, hermes_cli/main.py, agent/system_prompt.py,
  tui_gateway/server.py → #49917 (autopilot) +#48101/#49916/#50073/#50155/#49184
- agent/anthropic_adapter.py, agent/auxiliary_client.py, gateway/platforms/api_server.py
  → #48024 (reasoning API), #50064, #49184
- tools/mcp_tool.py → #48069 ; tools/skills_tool.py → #50045
- hermes_state.py → #50296/#50056 ; gateway/run.py → #50146/#49644
- tests/* → #50064
Textual proof: `comm -23` of #50487-resolved conversation_loop autopilot lines vs
#49917's = empty (0 unique). Coverage reconcile after closure: 0 real-source orphans.

## #50484 (residual clean, 20 files) — CLOSED redundant
All 20 files owned by feature PRs (verified 0 uncovered): agent_init→#49917/#48065/
#50073/#49184/#50296, models_dev→#49449, models.py→#49644, model_metadata/inventory/
test_copilot_*→#50064, gemini_*→#50033, system_prompt_prelude→#48101,
chat_completion_helpers→#50055, run_agent→#50296/#50073/#49644, test_*→#50078/#50080.

## #50049 (subdir-hints RuntimeError guard) — CLOSED as duplicate
Superseded by the EARLIER open #29433 (by udatny), which is a strict SUPERSET:
#29433 guards all 3 expanduser() sites (subdirectory_hints.py:138/198/202);
#50049 guarded only :138. Collaborator @alt-glitch flagged the duplication.
No coverage lost — the file's fix lives in #29433 (more complete).

## Coverage invariant held across all closures
Final reconcile (git diff v0.16.0..HEAD vs union of open-PR diffs, using
`git diff --name-only origin/main...<head>` per open PR — NOT `gh pr view --files`,
which caps at 100 entries and undercounts):
**165 delta files = 140 real src (all in open PRs) + 25 DISCARD + 0 real-source orphans.**
DISCARD = 9 `.bak` snapshots + 12 `.project-intel/` generated artifacts + 4 `transcripts/`.

(An earlier "160 / 139 / 21" figure here was the gh-`--files`-based undercount; the
git-diff ground-truth numbers above supersede it.)

## #50049 content fully re-homed (2026-06-21)
The subdirectory_hints `RuntimeError` guard + its complementary regression test
(`tests/agent/test_subdirectory_hints.py`, 30 tests) and the unrelated `hermes_cli/providers.py`
`xai`->`xAI` label were re-homed into NEW open draft PR **#50626**
(`feat/subdir-hints-and-xai-label`) so no real-source file is orphaned by the #50049
closure. The source-file fix also lives in upstream #29433 (superset); #50626 carries
the regression test + the xAI label.
