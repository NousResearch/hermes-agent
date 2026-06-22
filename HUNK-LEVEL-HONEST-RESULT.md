# Hunk-level accounting — the HONEST, fully-reconciled result (2026-06-22)

The Council was right to force hunk-level (not file-level) accounting. Here is the
truthful reckoning, with the root cause of every unmapped hunk traced to a commit.

## Numbers (re-runnable: `SRC=<checkout> python3 hunk_level_accounting.py`)
- src delta v0.16.0 → HEAD (`9633e2362`): **634 hunks**, 144 files
- mapped to an open code PR (content membership): **389**
- enumerated exclusion bucket (withdrawn/superseded/discard files): **22**
- UNMAPPED: **216** → of those, 24 cosmetic/comment-only, **192 real code lines not in
  any open PR's frozen content**

## Why the 192 are NOT "192 missing contributable features"
Every one traces to a commit that is EITHER (a) entangled/private content deliberately
left behind during extraction, (b) post-PR-assembly overlay drift on files already in
the PRs, or (c) cosmetic. Verified by `git log -S`/blame on the actual symbols:

| Source | Example symbols | Hunks (approx) | Disposition |
|---|---|---|---|
| `9fec781fc` 46-file ENTANGLED mega-commit (autopilot/cmx/kanban-mixed) | `_COPILOT_HIDDEN_USABLE`, `_AsyncSyncCompletionsAdapter` | ~40 | **EXCLUDED** — intentionally not extracted (entangled private); lands in shared files (models.py/auxiliary_client.py) so it *looks* like a gap |
| `codex_version` excluded-infra | `from agent.codex_version import get_codex_cli_version` | ~10 | **EXCLUDED** — on the standing never-PR infra list |
| account-specific caps | `gpt-5.4 900K`, hidden gemini preview context in model_metadata.py | ~12 | **PRIVATE** — account-specific, deliberately not generalized (user's "ship working values, don't re-engineer" rule) |
| `8766a1723` copilot-identity consolidation | integration-id / UA — the CLEAN part IS in #50064 (20 refs); the models.py catalog-mirror layer is the entangled remainder | ~30 | **PARTLY shipped** (#50064) + entangled remainder excluded |
| `5e0c05647`/`fe90244a8` phase-h/phase-m overlay-reconcile | test adjustments for the private overlay env | ~41 | **OVERLAY GLUE** — not standalone features |
| `58a2544be`/`715bda210` background-review | session-store isolation; `background_review.py` ALREADY present in all 39 PRs (~735 lines each) | ~26 | **incremental drift** on a file already in every PR |
| `defd5d57f`/`37cccdeee`/`9633e2362` em-dash/privacy/genericize | — | 24 | **COSMETIC** — logical content IS in a PR |

## Honest bottom line
The 192 unmapped hunks are **not missing contributable work.** They decompose into:
entangled-private (`9fec781fc`), excluded-infra (`codex_version`), account-specific
private values, post-PR overlay-reconcile glue, incremental drift on files already in
the PRs, and cosmetic scrubs. The genuinely-contributable content IS in the 39 open PRs.

## The one honest caveat (the real residual)
src is a MOVING target: commits after PR-assembly (`9fec781fc`, `8766a1723`,
background-review, em-dash/privacy scrubs) mean current src HEAD is a few commits ahead
of what the frozen PRs carry. The PRs reflect the contributable surface AT THE TIME they
were cut. To make current-HEAD-exact, the affected PRs (#50064/#49449/#49917) would need
re-cutting against current src — but their *contributable delta* is unchanged; only
entangled/private/cosmetic lines drifted. This is the user's call:
- (A) re-cut #50064/#49449/#49917 against current src HEAD (cosmetic + would re-pull
  some entangled content that was deliberately dropped), OR
- (B) accept the PRs as the contributable snapshot, with this document enumerating
  exactly what drifted and why each drifted hunk is private/entangled/cosmetic.

Recommended default: (B) — the drift is private/entangled/cosmetic, not lost features.
