# ⚠️ DEFERRED-WORK TRACKER — NOT FOR MERGE (draft, intentionally)

This draft PR holds the residual `./src` lines that are **not** part of the 37
mergeable feature PRs, organized into **feature-scoped, individually-cherry-pickable**
patch sets under `deferred/`. Each subdirectory is one coherent concern, so a downstream
operator can pull exactly the category they want onto a later release (built on v0.17.0 =
`2bd1977d`).

## Why these are deferred rather than standalone merge PRs

Each category below either (a) carries private/account-specific content that must not be
upstreamed, (b) is the private v2026.6.5 update-merge machinery (not contributable), or
(c) depends on infrastructure that is itself deferred. They are kept here, feature-scoped
and pullable, rather than forced into public PRs that would leak private data or ship
account-specific values.

## Feature-scoped categories (each independently cherry-pickable)

| Category (dir) | Lines | Concern | Why deferred (not a merge PR) |
|---|---|---|---|
| `private-overlay/` (11 files) | ~2,000 | v2026.6.5 update-merge machinery: agy-cli client, impersonation (gemini/anthropic adapters), source-accelerators, state/run_agent overlay | **Not contributable** — `[Hermes]`-authored `phase-h` update-overlay, not feature work |
| `private-overlay-phaseh/` (6 files) | ~333 | copilot-context test modifications + inventory/skills_tool edits from the same merge | **Not contributable** — same `phase-h` provenance |
| `private-feature-mixed/` (7 files) | ~1,563 | agy-cli rows, auto-router (`_copilot_auto`), cmx (`_cmx_owns_memory`), review-paths woven into shared files | **Would leak** private tokens; the agy rows also **depend on** the deferred `_PROBE_VERIFIED_OVERRIDES` limits table |
| `copilot-limits/` (2 files) | ~899 | account-specific context/output caps (`_CODEX_OAUTH_*`, per-account limits) | **Account-specific** — user ruling: keep deferred, generalize before contributing |
| `cmx/` (2 files) | ~450 | CMX context-engine wiring (conversation_loop prefetch + schema test) | Belongs in **one CMX-implementation PR** (user rule: CMX never isolated piecemeal) — not yet opened |
| `post-branch-drift/` (4 files remaining) | ~370 | William-authored lines in overlay HEAD postdating the owner PR branch cut, where the owner branch has since diverged heavily from upstream (overlay is *behind* on those files, so a blanket refresh would revert upstream work) | Full-file patches superseding the owner PR's file; pull if you want the overlay's exact version |

## ✅ What was FOLDED OUT of deferred into real feature PRs (this session)

Two genuinely-contributable items were moved from "deferred" into their owning PRs, where
they belong:

- **cli.py autopilot re-apply block** → committed into **#49917** (the autopilot PR).
  Restores autopilot state after an agent rebuild so `/autopilot` survives route changes.
  Verified via GitHub API; 31 autopilot tests pass.
- **chat_completions `max` thinking-level** → committed into **#49644** (the max-effort PR).
  Real bug fix: `max` effort fell through to medium/low for Gemini; now maps to `high`.
  124 tests pass.

## How to pull a category

```
git fetch <fork> deferred/residual-lines-on-v0.17.0 && git checkout FETCH_HEAD
git apply --3way deferred/<category>/<file>.patch
```

## Line accounting (honest, reproducible)

`partition.py` (committed) partitions every `git diff v0.16.0..HEAD` added line:
```
A.covered-owning-PR : 11216   (incl. the 2 folded items, in their real PRs)
B.covered-some-PR   :    51
D.non-substantive   :    12
E.deferred (this PR):  2117   (each line proven in a deferred/*.patch here)
F.UNACCOUNTED       :     0
```
Every line is in either an open feature PR or this draft tracker. **0 unaccounted.**

## The one open structural decision for the maintainer (user)

Whether the genuinely-private/account-specific categories above stay here as a single
feature-scoped deferred tracker, OR whether any contributable subset (e.g. the CMX category,
once the single CMX PR is opened; or the `gateway/run.py` generic media-detection block)
should graduate to its own standalone draft PR. The private-overlay / copilot-limits /
agy-mixed categories cannot become public PRs without leaking private data or shipping
account-specific values, per prior rulings.
