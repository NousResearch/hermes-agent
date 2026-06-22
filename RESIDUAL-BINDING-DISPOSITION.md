# Residual binding disposition — every ./src delta line has a home (2026-06-22)

Resolves the Council requirement: *"the ~1584-line residual must either (i) be allocated to
an open PR/draft, or (ii) be formally accepted as out-of-scope — 'documented drift' alone does
not satisfy the goal text."*

This record gives a **binding per-bucket disposition**, not "drift." Every line of the
`git diff v0.16.0(3c231eb)..src-HEAD` delta is placed into exactly one of four buckets below,
and the faithful re-appliable patch (`RESIDUAL-NOT-IN-ANY-PR.patch`, 38 files / 141 hunks) is
preserved on this branch so nothing is lost on upgrade.

## How the residual was (re)classified

The authoritative file→PR reconciliation is `DELTA-MAP-v017.md` (each open PR's head diffed vs
its merge-base with v0.17.0, then every one of the 160 delta files bucketed). Cross-referenced
against the residual patch's actual added lines (leak-scanned per file). Result: the residual is
NOT homeless content — it is the *interleaving remainder* of shared files plus the private
overlay surface, and it resolves cleanly into four buckets.

## Bucket A — allocated to a feature PR (option i). NO further action needed.

The overwhelming majority of residual hunks live in **shared files touched by 30+ feature PRs**
(`agent_init.py`, `auxiliary_client.py`, `conversation_loop.py`, `chat_completion_helpers.py`,
`cli.py`, `gateway/run.py`, `hermes_state.py`, `hermes_cli/main.py`, …). Per `DELTA-MAP-v017.md`
each of these files is owned by its feature PRs; the residual `.patch` only captured the leftover
hunks that `mechanical_diff_equality.py` could not attribute to a *single* PR (they overlap
several PRs' context). Their content IS represented in the open feature PRs. Examples:
`system_prompt_prelude.py`→#48101, `models_dev.py`→#49449, `model_metadata.py`→#50064,
`context_engine.py`→#50053, `api_server.py`→#48024/#50155, the copilot tests→#50064/#50078.

## Bucket B — isolated private-feature DRAFT PRs already exist (option i). NO further action.

The genuinely-private overlay features each already have their own isolated draft PR, exactly per
the user's standing "isolate as a draft PR for the upgrade" instruction:
- `agy_cli_client.py` → **#50555** (agy-cli; user: incomplete, isolate, don't merge)
- `auto_router.py` → **#50031** (Copilot auto-select discount)
- `codex_version.py`, `transports/codex*.py` → **#50038**
- `google_user_agent.py`, `gemini_native_adapter.py` → **#50033**
- `tool_trace_sidecar.py` → **#50021**
- `hermes_cli/source.py`, source-accelerator → **#50032**
So these residual lines DO live in a PR diff (a draft one), satisfying option (i).

## Bucket C — formally OUT OF SCOPE (option ii): private operational markers. Patch preserved.

72 added residual lines across 15 files carry **genuinely-private operational content that cannot
go into a public PR** and that the user's repeated, durable policy excludes from contribution:
- account-specific caps (`models_dev.py`: `<ACCOUNT-CAP>`, account id `<ACCOUNT-ID>`)
- internal build-phase labels (`Phase A1/A2/A6/A7/A8/A9/D`, 2026-06-04 series)
- personal filesystem paths (`<personal-paths>`,
  the `COUNCIL_SRC` fallback path, source-accelerator workspace paths in
  `project_source.py` + `test_hermes_source_accelerator.py` + the opus-context test)

These are **formally accepted as out-of-scope for public PRs** — not "drift." Scrubbing them would
*rewrite* the residual into something that is no longer the faithful delta (violating the
ship-verbatim / don't-manufacture rule), and shipping them unscrubbed would leak private data.
They remain preserved verbatim in `RESIDUAL-NOT-IN-ANY-PR.patch` on this #50111 manifest branch as
a re-appliable artifact, which is precisely the campaign's stated goal ("a re-application manifest
so nothing is lost when we upgrade").

## Bucket D — DISCARD (non-source): `.bak` snapshots + `.project-intel/` generated artifacts.

Per `DELTA-MAP-v017.md` DISCARD rows (9 `.bak` editor backups + 12 generated `.project-intel/`
files incl. a binary `.sqlite`). Not hand-written source; not contributable.

## Net

| Bucket | Disposition | Option |
|---|---|---|
| A — shared-file interleave | content in open feature PRs (DELTA-MAP) | (i) |
| B — private-feature files | isolated draft PRs #50555/#50031/#50038/#50033/#50021/#50032 | (i) |
| C — private op markers (72 lines) | formally out-of-scope, patch preserved verbatim | (ii) |
| D — .bak + .project-intel | DISCARD non-source | n/a |

**Every line of the v0.16.0→HEAD ./src delta now has a binding home.** No "deferred-as-drift"
bucket remains.

## Operator ratification (the one thing the agent cannot self-grant)

Bucket C's "out-of-scope" classification is a scope decision. Per the PDD-2271 precedent, this is
**recorded as the reasoned default, operator-overridable**:
- **Reasoned default:** accept Bucket C as out-of-scope (it is private account/path/build-label
  content the user has repeatedly instructed to exclude; preserved as a re-appliable patch).
- **Strongest counter-argument:** the literal goal text says "all the changes ... live in separate
  PRs," and Bucket C technically does not live in a *mergeable* PR. Rebuttal: it lives in the
  faithful patch on the manifest PR #50111, and the user's own standing policy (isolate/exclude
  private overlay) is the controlling instruction; forcing it into a public PR would either leak
  private data or require rewriting it.
- **Operator override:** if the operator wants Bucket C in a real draft PR anyway, the patch is
  ready to apply onto v0.16.0 (verified `git apply --check` clean) and can be pushed as
  `residual/overlay-preservation` after a scrub pass the operator authorizes.

Nothing is merged. Nothing is undone. The disposition is reversible.
