# Verification

The trust layer — the same rules whatever produced the material (a native run's digests or a processed import). Verification happens **as material lands**, per dimension, in fresh-context verifier subagents reading digest files — never as an end-of-run rewrite pass over an hour of accumulated context. Late-pass rewrites degrade reports; landing-time checks improve them.

## The claims ledger

The memlog `claim` entries are the ledger: every claim a decision could rest on, with its class (each pack names its classes — quantitative sizes, pricing, versions/compatibility, regulatory assertions, …), source, publisher, publication date, and status. New claims enter `unverified`; on a Refresh or Deepen run, claims outside the run's scope keep their prior status from the memlog — only new and in-scope claims are (re)checked.

## Levels

Per the resolved `validation` level (request > knob > default `normal`):

- **normal** — spot-check the **load-bearing claims only**: the handful per dimension the recommendation actually rests on. One independent-source check each, at landing. Everything else ships with its single source cited and confidence marked honestly. Fast by design.
- **high** — cross-check every claim in the pack's *two-source classes*, and run the red-team pass on major conclusions regardless of `{workflow.red_team}`.
- **max** — cross-check every ledger claim, run the red-team pass below at full breadth (every major conclusion), and primary-source-priority ranking: where a primary source (filing, regulator text, official docs, original paper) exists, secondary reporting alone does not verify.

Verifier assistants run behind the research firewall on `{workflow.subagent_models}` when set; judgment work never drops to the smallest tier.

**Independent** means a different publisher with different underlying data or reporting — not a syndication, quote, or republication of the first source, and not the same vendor's marketing in two places. An imported report counts as one publisher regardless of how many sources it cites internally; two imports from different tools agreeing is genuine confirmation, and their disagreement is a finding.

Outcomes per claim: **verified** (independent source agrees within tolerance — for quantitative claims, same order of magnitude and direction), **disputed** (independent sources materially disagree — report both figures, both cited; never average), **unverified** (no independent check within budget — the claim stays, flagged, and joins the staleness map), or **overturned** (the weight of evidence contradicts it — corrected in the text, original noted). Every status change lands in the memlog as a fresh `claim` line with the same `ref=` and the new status — last status wins, which is how `scripts/recon_kit.py tally` reads the ledger. A verification outcome adjusts status and flags — it never licenses rewriting a finding's substance beyond what the new evidence says.

Confidence rendered in the report: **high** (verified, fresh, credible publishers), **medium** (single credible source, fresh), **low** (stale, weak publisher, or disputed) — plus the explicit `unverified` flag. Confidence is per-claim, never per-section.

## Red-team pass

The single adversarial mechanism — no other verifier duplicates it. Off by default (`{workflow.red_team}` = `"off"`; `"offer"` proposes it at the plan gate, `"on"` always runs; `high` validation includes it for major conclusions, `max` runs it at full breadth). When it runs: for each major conclusion, a **fresh-context** skeptic subagent — the conclusion and a search budget, no supporting evidence, no run context — hunts for disconfirming evidence: the bear case, failed attempts, contrary data, the strongest good-faith argument the conclusion is wrong.

What comes back is weighed, not appended: a conclusion that survives gets its strongest counter-argument acknowledged in the synthesis; one that doesn't is revised before the report states it. Material findings land in a **Contrary Evidence** section with full citation discipline. Zero findings after a real search is itself reportable — say what was searched for and not found.
