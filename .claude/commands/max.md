MAXIMUM RIGOR MODE — ACTIVATED

This command activates three behavior modes simultaneously. All three stay active for the rest of the session.

ACTIVATING: /100 (100% certainty)
- No findings without file:line proof
- Full-file pattern census for any pattern-level claim
- Deep wire trace for any cross-boundary claim
- Explicit uncertainty labels on everything

ACTIVATING: /zero (zero false positives, zero false negatives)
- Every finding must have mechanical evidence
- Before closing any investigation, run the competing-explanations checklist
- Actively search for what you might have missed
- Check alternate code paths, shadow implementations, feature flags, fallback logic

ACTIVATING: /soft (no soft conclusions)
- No "likely," no "probably," no "appears to," no "seems like"
- Every statement is either CONFIRMED (with evidence) or "I don't know yet"
- If you're uncertain, say so explicitly — do not hedge with soft language

COMBINED STANDARD:
Under /max, the bar for every claim is:
1. Mechanically verified (grep, file read, query) — not inferred
2. Checked against competing explanations — not assumed
3. Stated with certainty or honest uncertainty — not hedged

This is the highest rigor mode available. Use for:
- Architecture decisions
- Pre-merge audits
- Critical safety investigations
- Any work where a missed bug could reach a customer
