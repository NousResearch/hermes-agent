# Process

For a report the user names or drops ("there's a research report at <path>, process it"):

1. **File it.** Find or create the run folder: if a drafted brief for this topic exists, that folder is the target; otherwise infer type and topic from the report (confirm in one line), bind `{doc_workspace}` (expand the folder name with `uv run scripts/recon_kit.py slug` as in Draft), and init the memlog. Move or copy the original into `{doc_workspace}/imports/` untouched — full fidelity is preserved there, and nowhere else.
2. **Record provenance** in the memlog: what produced it (which tool or firm), when (ask if not evident — production date drives staleness), and what the user wants decided from it.
3. **Extract.** A subagent (fresh context, firewall rules) reads the import and pulls every claim bearing on the decision into digest files under `{doc_workspace}/digests/` — standard shape `{claim, source, publisher, pub_date, accessed, confidence, class}`, keeping the original's citations (the cited source is the publisher; the import is the via). Multiple imports each get their own digest; contradictions between them are findings, not noise.
4. **Check against the pack**: which of the type's dimensions the material covers, which are open, where its claims fall inside two-source classes but rest on one publisher. Verification per the resolved `validation` level (`references/verification.md`) — at `normal` this is a spot-check of the load-bearing claims only, minutes not hours.
5. **Distill** into `research.md` per `references/synthesis.md` — the succinct, cited, decision-first summary with full metadata frontmatter (topic, type, decision, `source:` provenance, dates, status). This is the artifact downstream skills read; nobody ever reprocesses the import. Open dimensions are listed honestly with a one-line route: draft a follow-up prompt, or a targeted Run on the gap.
6. Finalize per `references/finalize.md`.
