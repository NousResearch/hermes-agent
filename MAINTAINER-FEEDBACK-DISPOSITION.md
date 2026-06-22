# Maintainer-feedback disposition (2026-06-22)

Acting on the user's directive to review bot/maintainer comments on the PRs and adhere
to the maintainers' newer/better implementations — especially agy.

## Genuine external-maintainer signals found

`alt-glitch` = **Siddharth Balyan (COLLABORATOR)** and `teknium1` (Nous founder) are real
maintainers — NOT the author (arminanton). No automated CI/triage bot comments exist.

### 1. agy-cli is SUPERSEDED then the replacement was REMOVED for account-ban safety — WITHDRAW our agy PRs

- **teknium1 on (closed) #50039:** *"Superseded by #50454 (merged) — We landed the native
  `google-antigravity` OAuth provider… Hermes owns the OAuth PKCE flow and talks to the
  Antigravity Code Assist backend directly, rather than shelling out to the external `agy`
  binary — so it works without `agy` installed."*
- **THEN #50492 (merged 2026-06-22) REMOVED** both `google-gemini-cli` and
  `google-antigravity` OAuth providers **entirely**, because *"Google now actively bans
  accounts for third-party tools that piggyback on Gemini CLI / Antigravity / Code Assist
  OAuth… a ban can extend to the entire Google account (Gmail/Drive)… a second violation is
  permanent."*

**Net maintainer position: no OAuth-based Google/Antigravity inference provider, by design.**
Our agy direction is doubly dead (superseded, then the whole category removed for safety).
→ **#50555** (isolated agy-cli provider) and **#50657** (agy-cli auth/runtime registration):
**CLOSE** with a pointer to #50454/#50492.

### 2. Our gemini-cli-UA PR is the SAME banned pattern — WITHDRAW

- **#50033** "present authentic `@google/gemini-cli` identity to Google backends" modifies
  `agent/gemini_cloudcode_adapter.py` — a file **#50492 deleted** — and its whole purpose
  (spoof gemini-cli identity so Google Code Assist OAuth backends accept the request) is
  exactly the account-ban risk the maintainers just removed from Hermes.
  → **CLOSE** #50033 (do NOT push the v0.17.0 conflict resolution — it would re-introduce a
  deliberately-removed ban risk). This OVERRIDES the earlier "push #50033 resolution" plan.

### 3. Maintainer triage already actioned (no work needed)

- **#50049** subdir-hints — maintainer marked `duplicate` of #29433 and **CLOSED** it. Correct;
  the fix is already covered by the earlier open PR. (Our #50626 also carries this fix — see below.)
- **#50484** private-overlay residual — maintainer marked `invalid` + `P3`, acknowledged as the
  author's explicit NOT-FOR-MERGE re-application manifest. Correct.
- **#50111** manifest — maintainer marked `invalid` + `P3`, acknowledged NOT-FOR-MERGE. Correct
  (this is the deferred-residual tracker by design).

### 4. Maintainer CONFIRMED valid / non-duplicate — keep as-is

- **#50296** (background-review session-store isolation): maintainer related it to merged #27190
  + bug #32858, confirmed it adds the missing isolation layer. Keep.
- **#50155** (enforce_response host call site): maintainer confirmed companion to #50053, not a dup. Keep.
- **#50086** (profile-sessions inode dedup): maintainer *"Confirmed against main… the fix is live
  and non-redundant."* Keep.
- **#49449** (multi-provider limits override): maintainer confirmed broader than the #42632/#29146/#29147
  cluster, "related rather than duplicate." Keep.

## Consequence for the v0.17.0 conflict-resolution work

Of the 7 conflict PRs, **#50033 is withdrawn** (ban-risk, maintainer-removed surface), so its
resolution is dropped. The remaining 6 (#49644, #49916, #50056, #50064, #50073, #50296) keep
their proven resolutions and get pushed to their own branch heads.

## #50049 vs our #50626 note
#50626 ("subdir-hints + xAI label") carries the SAME subdir-hint RuntimeError guard the maintainer
called a duplicate of #29433 on #50049. Flag for the user: #50626's subdir-hint half may also be a
duplicate of #29433 — worth checking whether to keep only its xAI-label half or withdraw it too.
