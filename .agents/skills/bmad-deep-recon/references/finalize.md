# Finalize

Every mode ends here once `research.md` is assembled.

1. `research.md` is complete per `references/synthesis.md`: decision-first summary, findings, contrary evidence where found, recommendations with downstream bindings, source appendix, staleness map. Frontmatter metadata (`type`, `topic`, `decision`, `source`, `status`, dates) is what lets every downstream consumer trust it without reprocessing.
2. **Citation check — mechanical, then semantic.** Run `uv run scripts/recon_kit.py citations {doc_workspace}/research.md` — it diffs inline `[n]` markers against the appendix and lists dangling markers and orphaned rows exactly; fix what it reports. Then a fresh-context subagent does only the judgment half: does each cited source actually say what the text claims? It never rewrites findings — a claim whose source doesn't back it gets its confidence downgraded and the mismatch logged as an `event`.
3. Render per `{workflow.output_format}` (see `references/html-briefing.md`): `auto` renders the briefing page on interactive runs, skips on headless/skill-invoked; `html`/`both` always; `md` never. `research.md` always exists — the briefing is its regenerable face.
4. Polish: apply each `{workflow.doc_standards}` entry (a `skill:`, `file:`, or plain-text directive) to `research.md`.
5. Execute each `{workflow.external_handoffs}` entry (NotebookLM, Confluence, …) — invoke the named tool, surface returned URLs; skip and flag unavailable tools.
6. Tell the user what exists and where — report, briefing, imports, memlog — plus what the staleness map says to re-check and when, and that Refresh/Deepen handle it. Invoke `bmad-help` to suggest the next step.
7. Run `{workflow.on_complete}` if non-empty — a string is one instruction, an array is a sequence.
