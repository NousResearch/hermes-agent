---
name: feature-parity-alignment-campaigns
description: Use when building out a Feature Parity & Alignment Campaign.
version: 1.0.0
author: Ares
license: MIT
metadata:
  hermes:
    tags: [Campaign, Parity, Platforms, Telegram, Discord, Slack, WhatsApp, 5x2x3, EPIC]
    related_skills: [campaign-primitives, api-docs-gap-analysis, swarm-dedup-campaign, worktree-hive, github-issues]
---

# Feature Parity & Alignment Campaigns

**STANDING RULE (Axl, 2026-08-05): whenever a skill is created that has ANYTHING to do with these campaigns, push it to NousResearch/hermes-agent immediately as an interlocked PR — no nagging required.** The campaign skill family is locked together: godfile-kill-campaigns (#79609), campaign-operations-kill-locks (#79779), feature-parity-alignment-campaigns (#79898). New campaign skills join at `skills/<category>/<skill>/`, PR body carries `Part of #78647` + `Related #<sibling skill PRs>` + the parity metas, DCO-signed, LF-only, and the literal PR token gets posted on the sibling PR threads (both-ways interlock). Pattern reference: PR #79898.

The reusable playbook for the platform parity campaigns Axl ordered with the
same surgical precision as the Telegram campaign (#78791). Proven live on four
platforms: **Telegram #78791, Discord #79564, Slack #79772, WhatsApp #79890**
(all NousResearch/hermes-agent). Mirror the anatomy exactly; never improvise a
different campaign shape.

## Trigger

"Build out the <Platform> Feature Parity & Alignment Campaign" / "Craft a
<Platform> campaign with the same surgical precision as the Telegram campaign".
The 20 campaign primitives (skill `campaign-primitives`) are binding: 5×2×3
double-blind (5 lanes × double blind × 3 waves), Interlock/Meta-Lock, dedup
first, EPIC with a current table.

## Anatomy (the shape every campaign must have)

1. **Recon (Wave 0)** — measure, never guess:
   - Verify no existing campaign meta: `gh search issues "<Platform> Feature Parity"`.
   - Live counts: label count (`label:<p>`), title/body union count
     (`<platform> in:title,body`), open PR count — all via search API
     `total_count`, recorded with date.
   - Adapter line count at **origin/main HEAD** (`git show HEAD:...adapter.py | wc -l`).
   - Docs surface: official platform API docs (e.g. WhatsApp Business Platform
     Cloud API + the bridge backends the adapter actually runs).
   - Dedup anchors: known high-signal open issues on the surface (search
     platform label + title keywords). Never duplicate them — interlock.
2. **Craft** — local draft at `D:/brain/AI-Systems-Intelligence/YYYY-MM-DD_<platform>-feature-parity-campaign.md` with a META header + GitHub-ready EPIC body. Body sections: Why (measured numbers) → Method (three-wave 5×2×3) → Lanes table → God-file/Decomposition lane → Deliverables → Standards (hard) → Ledger placeholder. Labels: `type/feature,comp/plugins,platform/<p>,P3` (+`needs-decision` where warranted; Discord meta used `comp/gateway` — check the platform's precedent).
3. **Lane layout** — S1..S5 domain shards (S1 core messaging, S2 business/API surface, S3 groups/communities, S4 bridge/lifecycle, S5 rich features — adapt per platform). **G lane** when the adapter is a god-file (>2,000 lines: Telegram 10,147 / Discord 10,138 / Slack 9,088). **S6 Decomposition & headroom lane when under the ceiling but multi-responsibility** (WhatsApp 1,918 = 96% of ceiling — under-ceiling rule, Axl 2026-08-05: ceiling is not a target; decompose so it never has a chance of crossing).
4. **File the EPIC** — `gh issue create --body-file` with the Windows path (gh.exe is native; MSYS `/c/...` paths fail). Title: `<Platform> Feature Parity & Alignment Campaign — meta-issue`. Write the body via write_file first (a `&` inside a shell heredoc trips the backgrounding guard).
5. **Hive** — one worktree per lane at **origin/main** pin: `git worktree add C:/tmp/<p>-campaign/wt-s<N> <pin>` (detached HEAD; never local HEAD which may be a PR branch).
6. **Ledger** — paginated search pull of ALL open issues on the surface (title+body union), classify into lanes by keyword rules, extract dependency edges (`#N` in body), write `ledger.json`, then post the full table `| Issue | Lane | Dependencies | Title |` as a comment on the EPIC (chunk at ~30KB — 400+ rows exceed GitHub's 65,536-char comment limit). Zero orphans: TRIAGE lane rows stay in the table.
7. **Wave 1** — blind analysis lanes, one per shard, each in its own worktree; deliverable per lane: JSON gap catalog (GAP_UNSUPPORTED / GAP_PARTIAL / GAP_CONFLICTED / GAP_DOCS / GAP_BUG_TRACKED) with docs-anchor + `file:line` evidence + cluster maps.
8. **Wave 2** — fresh blind cross-check workers, forbidden from wave-1 output. Agreement between independent witnesses is the only bar.
9. **Wave 3** — validation + filing: every agreed gap re-verified at current main + live tracker; issues filed with evidence, `Campaign: <Platform>` interlock tag, and links to the meta; existing coverage → interlock, never duplicate.
10. **Decomposition lane execution** — graph-before-slice: plan issue after agreement, one PR per cluster, Seam Identity + regression suite, suite green.

## Deliverables checklist

- EPIC filed with ledger table posted (never an empty EPIC).
- One issue per confirmed gap, all interlocked.
- Pre-existing issues/PRs interlocked into lanes (no orphans).
- `whatsapp-dev`-style platform skill shipped with Hermes (created during the campaign).
- Public reconciliation doc per cluster/gap (members, evidence, both waves' bases, confidence, verdict).

## Standards (hard, verbatim from the EPICs)

- Evidence everywhere: docs anchor + repo `file:line` verified at main.
- Exact numbers measured, never "about".
- No duplicate issues: existing coverage → interlock with the exact number.
- One PR per bug class; suite green; attribution to the author who earned it.
- Security/data-loss findings P1; everything else P2/P3.
- No hype: capability claims need witness receipts before public copy.

## Pitfalls (each cost real time)

- `gh issue create --body-file /c/tmp/...` fails — use `C:/tmp/...` (native gh.exe).
- Shell heredocs containing `&` are rejected as backgrounding — write bodies with write_file.
- Stale draft numbers: re-measure counts at filing time (they drift within hours).
- Search API `total_count` can exceed fetched pages — page until empty, dedup by number.
- Keyword classification needs a TRIAGE lane + lane-keyword priority (S4 bridge terms over-match; S6 refactor terms first).
- Posting a 457-row table needs chunking; verify each chunk posts before moving on.
- Worktree pins: `origin/main` after `git fetch origin main`, never local HEAD (may sit on a PR branch).
- After filing, update the Meta anchors memory entry with the new EPIC number.

## Precedent metas (canonical anatomy references)

| Platform | EPIC | Adapter @ craft | Lane shape |
|---|---|---|---|
| Telegram | #78791 | 10,147 lines | S1–S5 + G (10 mixins via #79010) |
| Discord | #79564 | 10,138 lines | S1–S5 + G |
| Slack | #79772 | 9,088 lines | S1–S4 + S5 @Hermes Tag flagship + G |
| WhatsApp | #79890 | 1,918 lines | S1–S5 + S6 decomposition/headroom (no G) |

Full EPIC body template: `references/template-epic-body.md`.
