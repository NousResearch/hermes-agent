# Brand-doc link recovery + public-proof handoff

Use this when the operator asks for “the link” to a brand design doc or brand book that was produced in prior work.

## Recovery order
1. Search recent session/durable recall for the canonical artifact name and whether a verified Drive/Google Doc link exists.
2. Search local approved artifact surfaces for the actual PDF/HTML source, especially profile output directories and `agent-lab/state/public-proofs`.
3. If a verified Google Doc/Drive link cannot be recovered, say that plainly; do not invent or imply a Drive link.
4. If the local PDF/HTML artifact is available and appropriate to show the operator, copy only that approved artifact into `agent-lab/state/public-proofs/<artifact-slug>/`.
5. Serve only the scoped `public-proofs` docroot, never `state/local-pages`.

## Verification before giving the link
- Compute SHA256 for the copied artifact(s).
- Confirm the proof server is serving from `state/public-proofs`.
- Positive-check the exact PDF/HTML URL returns `200` and non-empty bytes.
- Negative-check broad/internal paths such as `/QUEUE.md` and `/golf-darin-memory-inventory/` return `404` from the proof port.

## Reporting pattern
Keep the answer direct:
- first line: the PDF link
- optional second line: HTML/source link
- one sentence if no verified Google Doc/Drive link was recovered
- compact verification bullets with SHA256 and scoped proof status

Do not bury the link under a long explanation when the user asked for the link.