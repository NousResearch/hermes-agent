# Combined dry-run: full PR set onto v0.17.0 AND a later release

The goal requires the PRs be usable "as a combined set on a later release such as
v0.17.0". This is the executed end-to-end combined dry-run (fresh PR tips, 3-way
merge of all 40 candidate PRs; #50111 manifest excluded as docs-only).

| Target | base SHA | merged | conflicts | markers |
|---|---|---|---|---|
| **v0.17.0** (goal target) | 2bd1977d8 | **40/40** | **0** | **0** |
| **current origin/main** (later release) | e448b2141 | **40/40** | **0** | **0** |

(No tag later than v0.17.0 exists in the repo yet, so current origin/main — which
has drifted past v0.17.0 — serves as the "later release" target. The set combines
cleanly on both.)

## Why this now works (was 2 conflicts last round)
- #50457 slimmed (100->4 files) removed the redundant-mega-bundle overlap.
- 6 PRs rebased onto current origin/main (#50296/#49644/#50041/#50073/#50064/#50033),
  each a 1-file complementary conflict resolved by keeping both sides.
- #50111 manifest fixed: restored root README.md (was clobbering it), removed stale
  deferred/ patch dir -> now MERGEABLE, 0 leaks.

## Fresh uncached coverage (live PR tips, diff vs v0.17.0)
160 delta files = 137 in arminanton PRs + 21 DISCARD + 2 upstream-#29433 + 0 orphans
(sum verified == 160). Independent recomputation after the prior measurement regression.

## GitHub mergeable state
41/41 PRs MERGEABLE on current origin/main (incl. #50111 after the README fix).
